"""Serving probe for MLX radix-cache prefix-hit performance.

This is intentionally small and process-isolated: it launches one SGLang
server, waits for health, runs a fixed prefix-sharing request sequence, writes
JSONL, and always terminates the server process group.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def _wait_health(base_url: str, deadline_s: float) -> None:
    deadline = time.perf_counter() + deadline_s
    last_error = None
    while time.perf_counter() < deadline:
        try:
            _request_json("GET", f"{base_url}/health")
            return
        except Exception as exc:  # noqa: BLE001 - report final health error
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


def _extract_usage(response: dict) -> dict:
    meta = response.get("meta_info") or response.get("meta") or {}
    usage = response.get("usage") or {}
    return {
        "completion_tokens": (
            usage.get("completion_tokens")
            or meta.get("completion_tokens")
            or meta.get("completion_token_count")
        ),
        "cached_tokens": (
            usage.get("cached_tokens")
            or meta.get("cached_tokens")
            or meta.get("cached_token_num")
            or meta.get("cached_tokens_count")
        ),
    }


def _run_sequence(base_url: str, output_path: Path, delay_s: float) -> None:
    short_ids = [1000 + (idx % 251) for idx in range(434)]
    extra_ids = [2000 + (idx % 251) for idx in range(1728)]
    long_ids = short_ids + extra_ids
    assert len(short_ids) == 434
    assert len(long_ids) == 2162

    try:
        _request_json("POST", f"{base_url}/flush_cache", {})
    except urllib.error.HTTPError:
        pass

    requests = [
        ("short_first", short_ids),
        ("short_hit1", short_ids),
        ("short_hit2", short_ids),
        ("long_partial", long_ids),
        ("long_full", long_ids),
    ]
    sampling_params = {
        "max_new_tokens": 20,
        "temperature": 0,
        "ignore_eos": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for name, input_ids in requests:
            row = _generate(base_url, name, input_ids, sampling_params)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

        time.sleep(delay_s)
        row = _generate(base_url, "delayed_short", short_ids, sampling_params)
        fout.write(json.dumps(row) + "\n")


def _generate(
    base_url: str,
    name: str,
    input_ids: list[int],
    sampling_params: dict,
) -> dict:
    payload = {"input_ids": input_ids, "sampling_params": sampling_params}
    start = time.perf_counter()
    response = _request_json("POST", f"{base_url}/generate", payload)
    latency_s = time.perf_counter() - start
    usage = _extract_usage(response)
    row = {
        "name": name,
        "latency_s": latency_s,
        "prompt_len": len(input_ids),
    }
    row.update({key: value for key, value in usage.items() if value is not None})
    return row


def _pythonpath(root: Path) -> str:
    return os.pathsep.join([str(root / "python"), str(root / "sgl-kernel" / "python")])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--delay-s", type=float, default=10.0)
    parser.add_argument("--profile-timing", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(root)
    env["SGLANG_USE_MLX"] = "1"
    if args.profile_timing:
        env["SGLANG_MLX_PROFILE_TIMING"] = "1"

    cmd = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        "mps",
        "--mem-fraction-static",
        "0.88",
        "--page-size",
        "1",
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            _wait_health(base_url, 180)
            _run_sequence(base_url, args.output, args.delay_s)
        finally:
            _terminate(proc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
