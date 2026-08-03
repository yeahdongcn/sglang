"""Matched Qwen3-0.6B generation-step benchmark for SGLang MPS and native MLX.

This benchmark deliberately does not use ``mlx_lm.benchmark``.  Its public
generation metric starts after the first token has been produced while the
next token is already queued, and divides by all generated tokens rather than
the remaining decode forwards.  That is useful for mlx-lm itself, but it is
not the same timing boundary as SGLang's low-level ``one_batch`` benchmark.

The parent process creates one literal token-id manifest, launches every
provider profile in a fresh process, and monitors memory while it runs.  Both
workers report the same request setup plus two timed generation phases:

* prefill / TTFT: the full prompt through the first greedy output token;
* decode: exactly ``output_len - 1`` subsequent model forwards.

At the benchmark-harness boundary, aggregate mode adds one completion fence at
the end of each phase while ``each`` mode adds one per decode step.  Provider
internals may require additional framework fences (for example, the whole-MLX
SGLang provider crosses a real Torch/MLX ownership boundary); those costs stay
inside the measured region and are not claimed to have matching fence counts.
When ``--collect-mlx-phase-timing`` is enabled, whole-MLX SGLang trials carry
an ``mlx_phase_timing`` object in each trial artifact.  It reports host
durations for the producer fence, input import, lazy graph build, shared MLX
prepare/evaluation, DLPack export, and deferred K/V commit submission.  The
phase map is not an additive end-to-end decomposition: ``kv_commit_submit`` is
enqueue-only, the harness completion fence remains in ``prefill_s``/``decode_s``,
and a later producer fence may absorb preceding asynchronous work.  Each
artifact carries this contract explicitly.
The artifact is marked as instrumented because those timed trials, including its
throughput summary, contain the diagnostic overhead and must not be used as a
normal performance baseline.
SGLang batch/forward metadata preparation and the equivalent MLX graph
submission also remain inside those boundaries.  Their CPU submit time is
reported separately but is not subtracted from the parity metric.  The
model-core MLX profile projects only the final prompt hidden state, matching
SGLang's last-token logits contract rather than materializing prompt-wide
vocabulary logits.

In ``each`` mode, the harness additionally reports a decode-tail metric that
excludes the first decode forward from every comparable engine.  The complete
decode metric remains allocator-inclusive.  This matters at the default
512-token prompt because mlx-lm's fresh 256-token-step KV cache grows on that
first decode forward, while SGLang's Torch-owned KV pool is preallocated.

``mlx-lm-public`` is deliberately separate: it drives public ``generate_step``
with its log-softmax and one-token look-ahead pipeline, and reports both the
remaining-forward numerator and mlx-lm's public generation numerator.  It is
an observed-throughput anchor, not a matched phase boundary.

Tokenizer work, output extraction, provider tracing, and correctness checks
stay outside the timed regions.  This measures the framework generation loop,
not HTTP scheduling and not isolated kernel latency; use the neighboring
kernel benchmarks to explain any gap.

Example::

    python -m sglang.benchmark.mps_qwen3 \
      --model-path /path/to/Qwen3-0.6B \
      --mlx-lm-path ./mlx-lm \
      --profiles sglang-torch sglang-whole-mlx-metal-commit \
        sglang-whole-mlx-greedy-metal-commit \
        mlx-lm-model-core mlx-lm-public \
      --output /tmp/qwen3-mps-matrix.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

_PROVIDER_ENV_VARS = (
    "SGLANG_MPS_QWEN3_MODEL_FORWARD",
    "SGLANG_MPS_QWEN3_GREEDY_TAIL",
    "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
    "SGLANG_MPS_QWEN3_RADIX_DECODE",
    "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
    "SGLANG_MPS_RMSNORM",
    "SGLANG_MPS_FUSED_ADD_RMSNORM",
    "SGLANG_MPS_SILU_AND_MUL",
    "SGLANG_FORCE_FUSED_OP_BACKEND",
    # Benchmark-only MLX graph variants.  This is deliberately kept out of
    # the serving configuration surface; a candidate can replace the baseline
    # only after matched output and performance qualification.
    "SGLANG_MPS_QWEN3_BENCH_VARIANT",
)

_COUNTER_KEYS = (
    "attention_qkv_call_count",
    "attention_qkv_fallback_count",
    "attention_decode_call_count",
    "attention_decode_fallback_count",
    "whole_model_call_count",
    "whole_model_decode_call_count",
    "whole_model_prefill_call_count",
    "whole_model_selector_call_count",
    "whole_model_selector_fallback_count",
    "whole_model_compile_call_count",
    "whole_model_compile_total_call_count",
    "whole_model_compile_fallback_count",
    "whole_model_greedy_tail_call_count",
    "whole_model_greedy_tail_torch_call_count",
    "whole_model_greedy_tail_fallback_count",
    "whole_model_greedy_compile_call_count",
)

_FALLBACK_COUNTER_KEYS = (
    "attention_qkv_fallback_count",
    "attention_decode_fallback_count",
    "whole_model_selector_fallback_count",
    "whole_model_compile_fallback_count",
)

_SUMMARY_METRICS = (
    "request_setup_s",
    "prefill_s",
    "decode_s",
    "total_s",
    "request_total_s",
    "prefill_tps",
    "decode_tps",
)

_OPTIONAL_SUMMARY_METRICS = (
    "decode_tail_s",
    "decode_tail_tps",
    "public_generation_tps",
)

_MLX_PHASE_REQUIRED_KEYS = frozenset(
    {
        "producer_fence",
        "input_import",
        "graph_build",
        "prepare_eval",
        "dlpack_export",
        "kv_commit_submit",
    }
)
_MLX_PHASE_TIMING_CONTRACT = (
    "forward phases only; prefill_s/decode_s include outer completion fences; "
    "kv_commit_submit is enqueue-only; a later producer_fence may absorb prior "
    "asynchronous work"
)


@dataclasses.dataclass(frozen=True)
class Profile:
    engine: str
    environment: dict[str, str]


@dataclasses.dataclass(frozen=True)
class _StableTrialCollection:
    """Accepted trials plus complete, one-based attempt provenance."""

    trials: list[dict[str, Any]]
    attempted: int
    accepted_attempts: list[int]
    rejected_trials: list[dict[str, Any]]


class _StableTrialExhausted(RuntimeError):
    """A worker could not collect the requested number of stable trials."""

    def __init__(self, message: str, artifact: dict[str, Any]):
        super().__init__(message)
        self.artifact = artifact


class _WorkerProfileError(RuntimeError):
    """A worker process failed, optionally after publishing an artifact."""

    def __init__(self, message: str, worker_artifact: Optional[dict[str, Any]]):
        super().__init__(message)
        self.worker_artifact = worker_artifact


class _ParentMemoryGuardViolation(RuntimeError):
    """The parent stopped a worker before memory pressure became unsafe."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


@dataclasses.dataclass
class _AvailableMemoryGuard:
    """Phase-aware hard floor plus sustained soft available-memory guard."""

    hard_min_bytes: int
    soft_min_bytes: int
    setup_grace_s: float
    low_memory_sustain_s: float
    recovery_margin_bytes: int
    started_at: float
    pressure_since: Optional[float] = None
    minimum_available_bytes: Optional[int] = None
    timing_seen: bool = False

    def observe(
        self, *, now: float, available_bytes: int, phase: str
    ) -> Optional[dict[str, Any]]:
        """Return a JSON-safe violation after observing one parent sample."""
        if phase not in {"setup", "timing"}:
            raise ValueError(f"unknown worker phase {phase!r}")
        available_bytes = int(available_bytes)
        elapsed_s = max(0.0, float(now) - self.started_at)
        self.minimum_available_bytes = (
            available_bytes
            if self.minimum_available_bytes is None
            else min(self.minimum_available_bytes, available_bytes)
        )
        self.timing_seen = self.timing_seen or phase == "timing"

        common = {
            "phase": "timing" if self.timing_seen else phase,
            "available_bytes": available_bytes,
            "minimum_available_bytes": self.minimum_available_bytes,
            "hard_min_available_bytes": self.hard_min_bytes,
            "soft_min_available_bytes": self.soft_min_bytes,
            "recovery_available_bytes": (
                self.soft_min_bytes + self.recovery_margin_bytes
            ),
            "elapsed_s": elapsed_s,
        }
        if available_bytes < self.hard_min_bytes:
            if self.pressure_since is None:
                self.pressure_since = float(now)
            return {
                **common,
                "reason": "hard_min_available",
                "pressure_duration_s": max(0.0, float(now) - self.pressure_since),
            }

        enforce_soft = self.timing_seen or elapsed_s >= self.setup_grace_s
        if not enforce_soft:
            self.pressure_since = None
            return None

        recovery_bytes = self.soft_min_bytes + self.recovery_margin_bytes
        if available_bytes >= recovery_bytes:
            self.pressure_since = None
            return None
        if available_bytes >= self.soft_min_bytes:
            # Hysteresis: the pressure episode remains active until the
            # recovery margin is reached, but it can trigger only while the
            # current sample is below the soft floor.
            return None
        if self.pressure_since is None:
            self.pressure_since = float(now)
        pressure_duration_s = max(0.0, float(now) - self.pressure_since)
        if pressure_duration_s < self.low_memory_sustain_s:
            return None
        return {
            **common,
            "reason": "sustained_unrecovered_memory_pressure",
            "pressure_duration_s": pressure_duration_s,
        }


PROFILES: dict[str, Profile] = {
    "sglang-torch": Profile("sglang", {}),
    "sglang-qkv-aot": Profile(
        "sglang",
        {"SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_aot,torch"},
    ),
    "sglang-qkv-jit": Profile(
        "sglang",
        {"SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_jit,torch"},
    ),
    "sglang-decode-aot": Profile(
        "sglang",
        {"SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_aot,torch"},
    ),
    "sglang-decode-jit": Profile(
        "sglang",
        {"SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_jit,torch"},
    ),
    "sglang-aot-aot": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_aot,torch",
            "SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_aot,torch",
        },
    ),
    "sglang-aot-jit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_aot,torch",
            "SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_jit,torch",
        },
    ),
    "sglang-jit-aot": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_aot,torch",
        },
    ),
    "sglang-jit-jit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_jit,torch",
        },
    ),
    "sglang-rmsnorm-jit": Profile("sglang", {"SGLANG_MPS_RMSNORM": "metal_jit,torch"}),
    "sglang-add-rmsnorm-jit": Profile(
        "sglang", {"SGLANG_MPS_FUSED_ADD_RMSNORM": "metal_jit,torch"}
    ),
    "sglang-silu-jit": Profile(
        "sglang", {"SGLANG_MPS_SILU_AND_MUL": "metal_jit,torch"}
    ),
    "sglang-generic-jit": Profile(
        "sglang",
        {
            "SGLANG_MPS_RMSNORM": "metal_jit,torch",
            "SGLANG_MPS_FUSED_ADD_RMSNORM": "metal_jit,torch",
            "SGLANG_MPS_SILU_AND_MUL": "metal_jit,torch",
        },
    ),
    "sglang-best-metal": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "metal_aot,torch",
            "SGLANG_MPS_QWEN3_RADIX_DECODE": "metal_jit,torch",
            "SGLANG_MPS_RMSNORM": "metal_jit,torch",
            "SGLANG_MPS_FUSED_ADD_RMSNORM": "metal_jit,torch",
            "SGLANG_MPS_SILU_AND_MUL": "metal_jit,torch",
        },
    ),
    "sglang-whole-mlx": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "torch",
        },
    ),
    "sglang-whole-mlx-greedy": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
        },
    ),
    "sglang-whole-mlx-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
        },
    ),
    "sglang-whole-mlx-greedy-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
        },
    ),
    "sglang-whole-mlx-fast-rope-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fast_rope",
        },
    ),
    "sglang-whole-mlx-shared-qk-rope-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "shared_qk_rope",
        },
    ),
    "sglang-whole-mlx-fused-qkv-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_qkv",
        },
    ),
    "sglang-whole-mlx-native-norm-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "native_norm",
        },
    ),
    "sglang-whole-mlx-fused-norm-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_norm",
        },
    ),
    "sglang-whole-mlx-fused-rms-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_rms",
        },
    ),
    "sglang-whole-mlx-fused-add-rms-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_add_rms",
        },
    ),
    "sglang-whole-mlx-fused-swiglu-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_swiglu",
        },
    ),
    "sglang-whole-mlx-fused-qkv-norm-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_qkv,fused_norm",
        },
    ),
    "sglang-whole-mlx-fused-qkv-norm-swiglu-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": ("fused_qkv,fused_norm,fused_swiglu"),
        },
    ),
    "sglang-whole-mlx-fast-rope-native-norm-metal-commit": Profile(
        "sglang",
        {
            "SGLANG_MPS_QWEN3_MODEL_FORWARD": "mlx,torch",
            "SGLANG_MPS_QWEN3_GREEDY_TAIL": "mlx,torch",
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT": "metal_jit,torch",
            "SGLANG_MPS_QWEN3_BENCH_VARIANT": "fast_rope,native_norm",
        },
    ),
    "mlx-lm-model-core": Profile("mlx_lm_core", {}),
    "mlx-lm-public": Profile("mlx_lm_public", {}),
}

_DEFAULT_PROFILES = (
    "sglang-torch",
    "sglang-best-metal",
    "sglang-whole-mlx-metal-commit",
    "sglang-whole-mlx-greedy-metal-commit",
    "mlx-lm-model-core",
    "mlx-lm-public",
)


def _read_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def _package_version(name: str, module: Any = None) -> str:
    module_version = getattr(module, "__version__", None)
    if module_version:
        return str(module_version)
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _performance_environment() -> dict[str, str]:
    prefixes = ("SGLANG_MPS_", "PYTORCH_MPS_", "MLX_")
    exact = {"SGLANG_FORCE_FUSED_OP_BACKEND", "TOKENIZERS_PARALLELISM"}
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name in exact or name.startswith(prefixes)
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_identity(path: Path) -> str:
    resolved_name = path.resolve().name.lower()
    if len(resolved_name) == 64 and all(
        character in "0123456789abcdef" for character in resolved_name
    ):
        return f"sha256:{resolved_name}"
    return f"sha256:{_sha256_file(path)}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _worker_phase_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.phase.json")


def _publish_worker_phase(args: argparse.Namespace, phase: str) -> None:
    """Atomically publish the worker phase outside every measured region."""
    if phase not in {"setup", "timing"}:
        raise ValueError(f"unknown worker phase {phase!r}")
    _write_json(
        Path(args.worker_phase),
        {
            "schema_version": 1,
            "phase": phase,
            "profile": args.worker_profile,
            "engine": args.worker_engine,
            "pid": os.getpid(),
            "updated_unix_s": time.time(),
        },
    )


def _read_worker_phase(path: Path) -> str:
    """Read an atomic phase sidecar; absence means Python is still starting."""
    if not path.is_file():
        return "setup"
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("phase") not in {
        "setup",
        "timing",
    }:
        raise RuntimeError(f"invalid worker phase sidecar {path}: {payload!r}")
    return str(payload["phase"])


def _advance_worker_phase(previous: str, observed: str) -> str:
    """Keep the setup-to-timing lifecycle monotonic for every parent guard."""
    if previous not in {"setup", "timing"} or observed not in {"setup", "timing"}:
        raise ValueError(
            f"invalid worker phase transition previous={previous!r}, "
            f"observed={observed!r}"
        )
    return "timing" if "timing" in {previous, observed} else "setup"


def _model_metadata(model_path: Path) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing model config: {config_path}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    expected = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "the matched harness is restricted to dense Qwen3-0.6B; "
            f"config mismatches: {mismatches}"
        )
    vocab_size = int(config["vocab_size"])
    weight_files = []
    for path in sorted(model_path.glob("model*.safetensors")):
        stat = path.stat()
        weight_files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "content_identity": _content_identity(path),
            }
        )
    if not weight_files:
        raise ValueError(f"no model*.safetensors files under {model_path}")
    if len(weight_files) != 1:
        raise ValueError(
            "the matched Qwen3-0.6B harness requires exactly one "
            f"model*.safetensors shard; found {len(weight_files)}"
        )
    return {
        "model_path": str(model_path),
        "snapshot_revision": (
            model_path.name if model_path.parent.name == "snapshots" else None
        ),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "vocab_size": vocab_size,
        "weight_files": weight_files,
    }


def _make_manifest(
    model_path: Path, *, input_len: int, output_len: int, seed: int
) -> dict[str, Any]:
    if input_len < 1:
        raise ValueError("input_len must be positive")
    if output_len < 1:
        raise ValueError("output_len must be positive")
    metadata = _model_metadata(model_path)
    rng = random.Random(seed)
    prompt_token_ids = [rng.randrange(metadata["vocab_size"]) for _ in range(input_len)]
    return {
        "schema_version": 1,
        **metadata,
        "seed": seed,
        "input_len": input_len,
        "output_len": output_len,
        "prompt_token_ids": prompt_token_ids,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("cannot summarize zero benchmark trials")
    for index, trial in enumerate(trials):
        missing = [metric for metric in _SUMMARY_METRICS if metric not in trial]
        if missing:
            raise ValueError(f"trial {index} is missing summary metrics: {missing}")
    metrics = list(_SUMMARY_METRICS)
    for metric in _OPTIONAL_SUMMARY_METRICS:
        present = [metric in trial for trial in trials]
        if any(present) and not all(present):
            raise ValueError(
                f"optional summary metric {metric!r} is missing from some trials"
            )
        if all(present):
            metrics.append(metric)
    summary = {}
    for metric in metrics:
        values = [float(trial[metric]) for trial in trials]
        summary[metric] = {
            "median": statistics.median(values),
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
            "values": values,
        }
    return summary


def _decode_tail_metrics(decode_step_s: list[float]) -> dict[str, float | int]:
    """Summarize decode after excluding the first synchronized forward."""
    if len(decode_step_s) < 2:
        return {}
    tail_s = sum(float(value) for value in decode_step_s[1:])
    tail_steps = len(decode_step_s) - 1
    return {
        "decode_tail_s": tail_s,
        "decode_tail_tps": tail_steps / tail_s,
        "decode_tail_steps": tail_steps,
        "decode_tail_excluded_steps": 1,
    }


def _sum_mlx_phase_samples(
    samples: list[dict[str, float]],
) -> dict[str, float]:
    """Add per-forward MLX phase timings for a diagnostic artifact."""
    totals: dict[str, float] = {}
    for sample in samples:
        for name, duration in sample.items():
            totals[name] = totals.get(name, 0.0) + float(duration)
    return dict(sorted(totals.items()))


def _profile_uses_whole_mlx(profile_name: str) -> bool:
    """Return whether a profile selects the whole-model MLX provider."""
    profile = PROFILES.get(profile_name)
    if profile is None:
        return False
    return (
        profile.environment.get("SGLANG_MPS_QWEN3_MODEL_FORWARD", "").split(",", 1)[0]
        == "mlx"
    )


def _expected_mlx_benchmark_variants(profile_name: str) -> tuple[str, ...]:
    """Return the exact benchmark-only variant tuple selected by a profile."""
    profile = PROFILES.get(profile_name)
    if profile is None:
        return ()
    raw = profile.environment.get("SGLANG_MPS_QWEN3_BENCH_VARIANT", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _validate_phase_duration_map(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a phase-duration dictionary")
    for name, duration in value.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{label} contains an invalid phase name")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise RuntimeError(f"{label}.{name} must be numeric")
        if not math.isfinite(float(duration)) or float(duration) < 0.0:
            raise RuntimeError(f"{label}.{name} must be finite and nonnegative")


def _validate_worker_diagnostics(
    result: dict[str, Any],
    args: argparse.Namespace,
    *,
    profile_name: Optional[str] = None,
) -> None:
    """Fail closed when a diagnostic/benchmark gate silently did not apply."""
    profile_name = profile_name or result.get("profile")
    if not profile_name:
        # Small unit fixtures predating diagnostic artifacts intentionally omit
        # worker identity.  Real parent-launched workers always pass it below.
        return
    if result.get("profile") not in (None, profile_name):
        raise RuntimeError("worker profile differs from the launched profile")
    if result.get("engine") != "sglang":
        return

    expected_variants = _expected_mlx_benchmark_variants(profile_name)
    actual_variants = result.get("mlx_benchmark_variants")
    if (
        not isinstance(actual_variants, list)
        or tuple(actual_variants) != expected_variants
    ):
        raise RuntimeError(
            "worker benchmark variants do not match the launched profile: "
            f"expected={list(expected_variants)}, found={actual_variants!r}"
        )

    expected_instrumented = bool(
        getattr(args, "collect_mlx_phase_timing", False)
        and _profile_uses_whole_mlx(profile_name)
    )
    instrumentation = result.get("timing_instrumentation")
    if not isinstance(instrumentation, dict):
        raise RuntimeError("worker timing_instrumentation metadata is missing")
    if instrumentation.get("mlx_phase_timing") is not expected_instrumented:
        raise RuntimeError("worker timing instrumentation flag is inconsistent")
    if (
        instrumentation.get("summary_includes_instrumentation")
        is not expected_instrumented
    ):
        raise RuntimeError("worker summary instrumentation flag is inconsistent")

    trials = result.get("trials")
    if not isinstance(trials, list):
        raise RuntimeError("worker trials are required for diagnostic validation")
    for index, trial in enumerate(trials):
        timing = trial.get("mlx_phase_timing") if isinstance(trial, dict) else None
        if not isinstance(timing, dict):
            raise RuntimeError(f"trial {index} is missing mlx_phase_timing")
        if timing.get("enabled") is not expected_instrumented:
            raise RuntimeError(f"trial {index} phase-timing flag is inconsistent")
        if timing.get("contract") != _MLX_PHASE_TIMING_CONTRACT:
            raise RuntimeError(f"trial {index} phase-timing contract is inconsistent")
        prefill = timing.get("prefill")
        decode = timing.get("decode")
        decode_steps = timing.get("decode_steps")
        _validate_phase_duration_map(prefill, label=f"trial {index} prefill")
        _validate_phase_duration_map(decode, label=f"trial {index} decode")
        if not isinstance(decode_steps, list):
            raise RuntimeError(f"trial {index} decode_steps must be a list")
        if isinstance(trial, dict) and trial.get("decode_steps") is not None:
            if len(decode_steps) != int(trial["decode_steps"]):
                raise RuntimeError(
                    f"trial {index} phase sample count does not match decode_steps"
                )
        for step_index, sample in enumerate(decode_steps):
            _validate_phase_duration_map(
                sample, label=f"trial {index} decode_steps[{step_index}]"
            )
        if decode != _sum_mlx_phase_samples(decode_steps):
            raise RuntimeError(
                f"trial {index} decode phase totals do not match step samples"
            )
        if not expected_instrumented:
            continue
        missing = _MLX_PHASE_REQUIRED_KEYS - set(prefill)
        if missing:
            raise RuntimeError(
                f"trial {index} prefill phase timing is missing {sorted(missing)}"
            )
        for step_index, sample in enumerate(decode_steps):
            missing = _MLX_PHASE_REQUIRED_KEYS - set(sample)
            if missing:
                raise RuntimeError(
                    f"trial {index} decode step {step_index} phase timing is "
                    f"missing {sorted(missing)}"
                )


def _run_sglang_forward_with_phase_timing(
    runner: Any,
    forward_batch: Any,
    *,
    enabled: bool,
) -> tuple[tuple[Any, Any], dict[str, float]]:
    """Run one forward and optionally capture MLX bridge/provider phases."""
    if not enabled:
        return runner.forward(forward_batch), {}

    from sglang.srt.utils._phase_timing import phase_recorder

    events: list[tuple[str, float]] = []
    with phase_recorder(
        lambda name, duration: events.append((str(name), float(duration)))
    ):
        result = runner.forward(forward_batch)
    return result, _sum_mlx_phase_samples(
        [{name: duration} for name, duration in events]
    )


def _select_reference_profile(
    results: list[dict[str, Any]],
) -> tuple[str, bool]:
    """Choose an output reference and state whether it is the Torch baseline."""
    if not results:
        raise ValueError("cannot select a reference from zero benchmark results")
    by_name = {result["profile"]: result for result in results}
    if "sglang-torch" in by_name:
        return "sglang-torch", True
    for result in results:
        if _token_ids_are_strictly_comparable(result):
            return str(result["profile"]), False
    return str(results[0]["profile"]), False


def _validate_sglang_pool_capacities(
    results: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Require comparable SGLang profiles to use the same adequate KV pool."""
    required = int(manifest["input_len"]) + int(manifest["output_len"]) - 1
    capacities: dict[str, int] = {}
    for result in results:
        if result.get("engine") != "sglang":
            continue
        profile = str(result["profile"])
        if "max_total_num_tokens" not in result:
            raise RuntimeError(
                f"{profile} did not report its actual max_total_num_tokens"
            )
        capacity = int(result["max_total_num_tokens"])
        if capacity < required:
            raise RuntimeError(
                f"{profile} KV pool has {capacity} tokens, fewer than the "
                f"{required} required by the measured request"
            )
        capacities[profile] = capacity
    unique = set(capacities.values())
    if len(unique) > 1:
        raise RuntimeError(
            f"SGLang profiles used different actual KV-pool capacities: {capacities}"
        )
    return {
        "required_tokens": required,
        "capacity_by_profile": capacities,
        "common_capacity": next(iter(unique)) if unique else None,
        "consistent": True,
    }


def _state_delta(before: Optional[dict], after: Optional[dict]) -> dict[str, int]:
    before = before or {}
    after = after or {}
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in _COUNTER_KEYS
    }


def _first_provider(environment: dict[str, str], name: str) -> str:
    return environment.get(name, "torch").split(",", 1)[0]


def _validate_sglang_provider_state(
    profile_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    output_len: int,
) -> dict[str, int]:
    profile = PROFILES[profile_name]
    delta = _state_delta(before, after)
    nonzero_fallbacks = {
        key: delta[key] for key in _FALLBACK_COUNTER_KEYS if delta[key] != 0
    }
    if nonzero_fallbacks:
        raise RuntimeError(
            f"{profile_name} used an unplanned runtime fallback: {nonzero_fallbacks}"
        )

    expected_qkv = _first_provider(
        profile.environment, "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE"
    )
    expected_decode = _first_provider(
        profile.environment, "SGLANG_MPS_QWEN3_RADIX_DECODE"
    )
    expected_whole = _first_provider(
        profile.environment, "SGLANG_MPS_QWEN3_MODEL_FORWARD"
    )
    expected_greedy_tail = _first_provider(
        profile.environment, "SGLANG_MPS_QWEN3_GREEDY_TAIL"
    )
    expected_deferred = (
        _first_provider(
            profile.environment,
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
        )
        if expected_whole == "mlx"
        else "off"
    )
    expected_generic = {
        "rmsnorm": _first_provider(profile.environment, "SGLANG_MPS_RMSNORM"),
        "fused_add_rmsnorm": _first_provider(
            profile.environment, "SGLANG_MPS_FUSED_ADD_RMSNORM"
        ),
        "silu_and_mul": _first_provider(profile.environment, "SGLANG_MPS_SILU_AND_MUL"),
    }
    actual = {
        "qkv": after.get("qkv_kernel_backend"),
        "decode": after.get("decode_kernel_backend"),
        "whole": after.get("whole_model_backend"),
        "greedy_tail": after.get("whole_model_greedy_tail_backend"),
        "deferred": after.get("deferred_kv_commit_backend"),
        "generic": after.get("generic_kernel_backends"),
    }
    expected = {
        "qkv": expected_qkv,
        "decode": expected_decode,
        "whole": expected_whole,
        "greedy_tail": expected_greedy_tail if expected_whole == "mlx" else "off",
        "deferred": expected_deferred,
        "generic": expected_generic,
    }
    if actual != expected:
        raise RuntimeError(
            f"{profile_name} selected unexpected providers: "
            f"expected={expected}, actual={actual}"
        )

    whole_model = expected_whole == "mlx"
    if whole_model:
        expected_decode_calls = output_len - 1
        expected_deltas = {
            "whole_model_prefill_call_count": 1,
            "whole_model_decode_call_count": expected_decode_calls,
            "whole_model_call_count": output_len,
            "whole_model_selector_call_count": output_len,
            "attention_qkv_call_count": 0,
            "attention_decode_call_count": 0,
        }
        wrong = {
            key: (delta[key], value)
            for key, value in expected_deltas.items()
            if delta[key] != value
        }
        if wrong:
            raise RuntimeError(
                f"{profile_name} did not execute the whole-model MLX island "
                f"for every forward: {wrong}"
            )
        if delta["whole_model_compile_total_call_count"] != expected_decode_calls:
            raise RuntimeError(
                f"{profile_name} primary compiled decode calls were "
                f"{delta['whole_model_compile_total_call_count']}, expected "
                f"{expected_decode_calls}"
            )
        if expected_greedy_tail == "mlx":
            if delta["whole_model_greedy_tail_call_count"] != output_len:
                raise RuntimeError(
                    f"{profile_name} MLX greedy tail calls were "
                    f"{delta['whole_model_greedy_tail_call_count']}, expected "
                    f"{output_len}"
                )
            if delta["whole_model_greedy_tail_fallback_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} unexpectedly fell back from the MLX greedy tail"
                )
            if delta["whole_model_greedy_compile_call_count"] != expected_decode_calls:
                raise RuntimeError(
                    f"{profile_name} compiled MLX greedy calls were "
                    f"{delta['whole_model_greedy_compile_call_count']}, expected "
                    f"{expected_decode_calls}"
                )
            if delta["whole_model_compile_call_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} retained or called the non-primary hidden "
                    "compiled graph"
                )
            if delta["whole_model_greedy_tail_torch_call_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} unexpectedly returned to the Torch tail"
                )
        else:
            if delta["whole_model_greedy_tail_call_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} unexpectedly executed the MLX greedy tail"
                )
            if delta["whole_model_greedy_compile_call_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} unexpectedly executed the compiled greedy graph"
                )
            if delta["whole_model_compile_call_count"] != expected_decode_calls:
                raise RuntimeError(
                    f"{profile_name} hidden compiled decode calls were "
                    f"{delta['whole_model_compile_call_count']}, expected "
                    f"{expected_decode_calls}"
                )
            if delta["whole_model_greedy_tail_torch_call_count"] != output_len:
                raise RuntimeError(
                    f"{profile_name} Torch-tail calls were not reported for every "
                    "whole-model forward"
                )
            if delta["whole_model_greedy_tail_fallback_count"] != 0:
                raise RuntimeError(
                    f"{profile_name} reported the deliberately selected Torch "
                    "tail as an MLX fallback"
                )
    else:
        if delta["whole_model_call_count"] != 0:
            raise RuntimeError(f"{profile_name} unexpectedly called whole-model MLX")
        if expected_qkv == "torch" and delta["attention_qkv_call_count"] != 0:
            raise RuntimeError(f"{profile_name} unexpectedly called custom QKV")
        if expected_qkv != "torch":
            patched_qkv = int(after.get("patched_qkv_modules", 0))
            expected_qkv_calls = patched_qkv * output_len
            if (
                patched_qkv <= 0
                or delta["attention_qkv_call_count"] != expected_qkv_calls
            ):
                raise RuntimeError(
                    f"{profile_name} QKV calls were "
                    f"{delta['attention_qkv_call_count']}, expected "
                    f"{expected_qkv_calls} from {patched_qkv} patched modules"
                )
        if expected_decode == "torch" and delta["attention_decode_call_count"] != 0:
            raise RuntimeError(f"{profile_name} unexpectedly called custom decode")
        if expected_decode != "torch":
            patched_decode = int(after.get("patched_decode_modules", 0))
            expected_decode_calls = patched_decode * (output_len - 1)
            if (
                patched_decode <= 0
                or delta["attention_decode_call_count"] != expected_decode_calls
            ):
                raise RuntimeError(
                    f"{profile_name} decode calls were "
                    f"{delta['attention_decode_call_count']}, expected "
                    f"{expected_decode_calls} from {patched_decode} patched modules"
                )
    return delta


def _torch_memory() -> dict[str, int]:
    import torch

    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
        "peak_allocated": int(torch.mps.max_memory_allocated()),
        "peak_reserved": int(torch.mps.max_memory_reserved()),
        "recommended_max": int(torch.mps.recommended_max_memory()),
    }


def _mlx_memory() -> dict[str, int]:
    import mlx.core as mx

    return {
        "active": int(mx.get_active_memory()),
        "cache": int(mx.get_cache_memory()),
        "peak": int(mx.get_peak_memory()),
    }


def _trace_summary(records: list[Any]) -> list[dict[str, Any]]:
    counts = Counter((record.op, record.backend) for record in records)
    return [
        {"op": op, "backend": backend, "calls": calls}
        for (op, backend), calls in sorted(counts.items())
    ]


def _validate_sglang_trace(
    profile_name: str,
    trace: list[dict[str, Any]],
    *,
    input_len: int,
    output_len: int,
) -> None:
    """Validate the exact Qwen3-0.6B generic-op provider distribution."""
    profile = PROFILES[profile_name]
    calls = {
        (str(item["op"]), str(item["backend"])): int(item["calls"]) for item in trace
    }
    expected: dict[tuple[str, str], int] = {}
    whole_model = (
        _first_provider(
            profile.environment,
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
        )
        == "mlx"
    )
    if whole_model:
        commit_backend = _first_provider(
            profile.environment,
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
        )
        commit_op = "kvcache.qwen3_deferred_kv_commit"
        expected[(commit_op, commit_backend)] = output_len
        alternate = "torch" if commit_backend == "metal_jit" else "metal_jit"
        expected[(commit_op, alternate)] = 0
    else:
        qkv_backend = _first_provider(
            profile.environment,
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
        )
        if qkv_backend != "torch":
            expected[("attention.qwen3_qknorm_rope_store", qkv_backend)] = (
                28 * output_len
            )
        decode_backend = _first_provider(
            profile.environment,
            "SGLANG_MPS_QWEN3_RADIX_DECODE",
        )
        if decode_backend != "torch":
            expected[("attention.qwen3_radix_decode", decode_backend)] = 28 * (
                output_len - 1
            )
        rms_backend = _first_provider(profile.environment, "SGLANG_MPS_RMSNORM")
        fused_backend = _first_provider(
            profile.environment,
            "SGLANG_MPS_FUSED_ADD_RMSNORM",
        )
        silu_backend = _first_provider(
            profile.environment,
            "SGLANG_MPS_SILU_AND_MUL",
        )
        expected[("layernorm.rmsnorm", rms_backend)] = output_len
        alternate = "metal_jit" if rms_backend == "torch" else "torch"
        expected[("layernorm.rmsnorm", alternate)] = 0
        expected[("layernorm.fused_add_rmsnorm", fused_backend)] = 56 * output_len
        alternate = "metal_jit" if fused_backend == "torch" else "torch"
        expected[("layernorm.fused_add_rmsnorm", alternate)] = 0
        if silu_backend == "metal_jit":
            metal_forwards = 1 if input_len >= 8 else 0
            expected[("activation.silu_and_mul", "metal_jit")] = 28 * metal_forwards
            expected[("activation.silu_and_mul", "torch")] = 28 * (
                output_len - metal_forwards
            )
        else:
            expected[("activation.silu_and_mul", "torch")] = 28 * output_len
            expected[("activation.silu_and_mul", "metal_jit")] = 0

    wrong = {
        f"{op}:{backend}": (calls.get((op, backend), 0), expected_calls)
        for (op, backend), expected_calls in expected.items()
        if calls.get((op, backend), 0) != expected_calls
    }
    unexpected = {
        f"{op}:{backend}": actual_calls
        for (op, backend), actual_calls in calls.items()
        if actual_calls and (op, backend) not in expected
    }
    if wrong or unexpected:
        raise RuntimeError(
            f"{profile_name} generic-op trace did not match the fixed Qwen3-0.6B "
            f"contract: wrong={wrong}, unexpected={unexpected}; trace={trace}"
        )


def _run_sglang_trial(
    runner: Any,
    manifest: dict[str, Any],
    profile_name: str,
    sync_mode: str,
    *,
    trace_ops: bool,
    collect_phase_timing: bool = False,
) -> dict[str, Any]:
    import torch

    from sglang.benchmark import one_batch

    trace_records = []
    if trace_ops:
        from sglang.kernels import clear_fused_op_trace, enable_fused_op_trace

        clear_fused_op_trace()
        enable_fused_op_trace()

    trial_swap_before = _swap_snapshot()
    input_len = int(manifest["input_len"])
    output_len = int(manifest["output_len"])
    runner.synchronize()
    reset_start = time.perf_counter()
    runner.clear()
    runner.synchronize()
    reset_s = time.perf_counter() - reset_start
    request_setup_start = time.perf_counter()
    reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
        1,
        input_len,
        [manifest["prompt_token_ids"]],
        output_len=output_len,
    )
    request_setup_s = time.perf_counter() - request_setup_start
    torch.mps.reset_peak_memory_stats()
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except ImportError:
        mx = None

    state_before = runner.torch_runner.get_platform_operator_state() or {}
    collect_mlx_phases = collect_phase_timing and _profile_uses_whole_mlx(profile_name)
    try:
        prefill_start = time.perf_counter()
        prepare_start = time.perf_counter()
        forward_batch, batch = runner.prepare_extend(reqs)
        prefill_prepare_submit_s = time.perf_counter() - prepare_start

        forward_start = time.perf_counter()
        (next_token_ids, _), prefill_mlx_phase_s = (
            _run_sglang_forward_with_phase_timing(
                runner,
                forward_batch,
                enabled=collect_mlx_phases,
            )
        )
        prefill_forward_submit_s = time.perf_counter() - forward_start
        runner.synchronize()
        prefill_s = time.perf_counter() - prefill_start
        output_tensors = [next_token_ids]

        decode_step_s = []
        decode_prepare_submit_step_s = []
        decode_forward_submit_step_s = []
        decode_mlx_phase_step_s: list[dict[str, float]] = []
        if sync_mode == "each":
            decode_start = time.perf_counter()
            for _ in range(output_len - 1):
                step_start = time.perf_counter()
                prepare_start = time.perf_counter()
                forward_batch = runner.prepare_decode(next_token_ids, batch)
                prepare_submit_s = time.perf_counter() - prepare_start

                forward_start = time.perf_counter()
                (next_token_ids, _), phase_s = _run_sglang_forward_with_phase_timing(
                    runner,
                    forward_batch,
                    enabled=collect_mlx_phases,
                )
                forward_submit_s = time.perf_counter() - forward_start
                runner.synchronize()
                decode_prepare_submit_step_s.append(prepare_submit_s)
                decode_forward_submit_step_s.append(forward_submit_s)
                decode_step_s.append(time.perf_counter() - step_start)
                decode_mlx_phase_step_s.append(phase_s)
                output_tensors.append(next_token_ids)
            decode_s = time.perf_counter() - decode_start
        else:
            decode_start = time.perf_counter()
            for _ in range(output_len - 1):
                submit_start = time.perf_counter()
                forward_batch = runner.prepare_decode(next_token_ids, batch)
                decode_prepare_submit_step_s.append(time.perf_counter() - submit_start)
                submit_start = time.perf_counter()
                (next_token_ids, _), phase_s = _run_sglang_forward_with_phase_timing(
                    runner,
                    forward_batch,
                    enabled=collect_mlx_phases,
                )
                decode_forward_submit_step_s.append(time.perf_counter() - submit_start)
                decode_mlx_phase_step_s.append(phase_s)
                output_tensors.append(next_token_ids)
            runner.synchronize()
            decode_s = time.perf_counter() - decode_start

        trial_swap_after = _swap_snapshot()
        output_ids = [int(value[0].item()) for value in output_tensors]
        state_after = runner.torch_runner.get_platform_operator_state() or {}
        counter_delta = _validate_sglang_provider_state(
            profile_name,
            state_before,
            state_after,
            output_len=output_len,
        )
        torch_memory = _torch_memory()
        mlx_memory = _mlx_memory() if mx is not None else None
    finally:
        if trace_ops:
            from sglang.kernels import (
                disable_fused_op_trace,
                get_fused_op_trace,
            )

            disable_fused_op_trace()
            trace_records = get_fused_op_trace()

    decode_steps = output_len - 1
    return {
        "metric_contract": "matched_framework_generation_step",
        "reset_s": reset_s,
        "request_setup_s": request_setup_s,
        "prefill_s": prefill_s,
        "prefill_prepare_submit_s": prefill_prepare_submit_s,
        "prefill_forward_submit_s": prefill_forward_submit_s,
        "decode_s": decode_s,
        "total_s": prefill_s + decode_s,
        "request_total_s": request_setup_s + prefill_s + decode_s,
        "lifecycle_total_s": reset_s + request_setup_s + prefill_s + decode_s,
        "prefill_tps": input_len / prefill_s,
        "decode_tps": decode_steps / decode_s if decode_steps else 0.0,
        "decode_steps": decode_steps,
        "decode_step_s": decode_step_s,
        "decode_prepare_submit_step_s": decode_prepare_submit_step_s,
        "decode_forward_submit_step_s": decode_forward_submit_step_s,
        "mlx_phase_timing": {
            "enabled": bool(collect_mlx_phases),
            "contract": _MLX_PHASE_TIMING_CONTRACT,
            "prefill": prefill_mlx_phase_s,
            "decode": _sum_mlx_phase_samples(decode_mlx_phase_step_s),
            "decode_steps": decode_mlx_phase_step_s,
        },
        "output_ids": output_ids,
        "provider_state_before": state_before,
        "provider_state_after": state_after,
        "provider_counter_delta": counter_delta,
        "fused_op_trace": _trace_summary(trace_records),
        "torch_mps_memory_bytes": torch_memory,
        "mlx_memory_bytes": mlx_memory,
        "system_swap_before": trial_swap_before,
        "system_swap_after": trial_swap_after,
        "system_swap_delta": _swap_delta(trial_swap_before, trial_swap_after),
        **_decode_tail_metrics(decode_step_s),
    }


def _install_sglang_mlx_benchmark_variant() -> tuple[str, ...]:
    """Install isolated MLX graph experiments before the provider is built.

    These variants exist only in the benchmark worker.  They let the matched
    harness measure a candidate against identical Torch ownership, scheduler,
    KV-pool, and bridge boundaries without adding an experimental serving env.
    """
    raw = os.environ.get("SGLANG_MPS_QWEN3_BENCH_VARIANT", "")
    variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not variants:
        return ()
    unknown = sorted(
        set(variants)
        - {
            "fast_rope",
            "shared_qk_rope",
            "fused_qkv",
            "native_norm",
            "fused_norm",
            "fused_rms",
            "fused_add_rms",
            "fused_swiglu",
        }
    )
    if unknown:
        raise ValueError(f"unknown Qwen3 MLX benchmark variants: {unknown}")
    if "native_norm" in variants and (
        "fused_norm" in variants
        or "fused_rms" in variants
        or "fused_add_rms" in variants
    ):
        raise ValueError("native_norm and fused RMS variants are mutually exclusive")

    import mlx.core as mx

    from sglang.srt.hardware_backend.mps.model_ops import qwen3_mlx

    if "fast_rope" in variants:

        def fast_rope(value, _cos_sin, positions):
            # Treat each flattened token as one batch item with a one-token
            # sequence.  The per-item offset preserves arbitrary Radix decode
            # positions while using MLX's fused NeoX RoPE implementation.
            return mx.fast.rope(
                value[:, :, None, :],
                qwen3_mlx.QWEN3_06B_METAL_SPEC.head_dim,
                traditional=False,
                base=qwen3_mlx.QWEN3_06B_ROPE_BASE,
                scale=1.0,
                offset=positions,
            )[:, :, 0, :]

        qwen3_mlx._rope_neox = fast_rope

    if "shared_qk_rope" in variants:
        original_cold_prefill_graph = qwen3_mlx._mlx_cold_prefill_graph

        def shared_qk_cold_prefill_graph(views, input_ids, positions):
            return original_cold_prefill_graph(
                views,
                input_ids,
                positions,
                share_qk_rope=True,
            )

        qwen3_mlx._mlx_cold_prefill_graph = shared_qk_cold_prefill_graph

    if "fused_qkv" in variants:
        from sglang.kernels.ops.attention.qwen3_mlx import (
            qwen3_qkv_prepare_deferred,
            warmup_qwen3_qkv_prepare_deferred,
        )

        warmed_epsilons = set()

        def fused_qkv(qkv, layer, positions, *, share_qk_rope=False):
            # The fused primitive already emits separate dense Q/K outputs;
            # the shared-RoPE graph hint is therefore fully subsumed.
            del share_qk_rope
            epsilon = float(layer.qk_epsilon)
            if epsilon not in warmed_epsilons:
                # The worker installs this monkeypatch before model loading, so
                # warm the actual model epsilon on the first graph build (which
                # happens during provider startup), not on a live request.
                warmup_qwen3_qkv_prepare_deferred(epsilon)
                warmed_epsilons.add(epsilon)
            return qwen3_qkv_prepare_deferred(
                qkv,
                layer.q_norm.array,
                layer.k_norm.array,
                layer.rope_cache.array,
                positions,
                epsilon=epsilon,
            )

        qwen3_mlx._prepare_qkv = fused_qkv

    if "fused_swiglu" in variants:
        # This is the same tiny shapeless helper used by mlx-lm's Qwen3 MLP.
        # Compiled elementwise fusion can change a few intermediate bf16
        # roundings for unusually large activations, so keep it benchmark-only
        # until full logits/KV parity is checked across supported prompt lengths.
        import mlx.nn as nn

        def swiglu_impl(gate, up):
            return nn.silu(gate) * up

        compiled_swiglu = mx.compile(swiglu_impl, shapeless=True)

        # Compile the shapeless executable before the first timed worker
        # trial.  The provider normally warms its own graph during startup,
        # but this benchmark-only replacement is installed after that warmup
        # hook and would otherwise charge its first Metal compile to the
        # prefill measurement.  A one-row representative input is enough for
        # the shapeless elementwise program and keeps the warmup allocation
        # negligible on 16-GB Apple Silicon hosts.
        warmup_value = mx.zeros(
            (1, qwen3_mlx.QWEN3_06B_INTERMEDIATE_SIZE), dtype=mx.bfloat16
        )
        mx.eval(compiled_swiglu(warmup_value, warmup_value))

        qwen3_mlx._swiglu = compiled_swiglu

    if "native_norm" in variants:

        def native_rms_norm(value, weight, epsilon):
            return mx.fast.rms_norm(value, weight, epsilon)

        def native_add_rms_norm(value, residual, weight, epsilon):
            summed = value + residual
            return mx.fast.rms_norm(summed, weight, epsilon), summed

        qwen3_mlx._rms_norm = native_rms_norm
        qwen3_mlx._add_rms_norm = native_add_rms_norm

    if {"fused_norm", "fused_rms", "fused_add_rms"}.intersection(variants):
        from sglang.kernels.ops.layernorm._qwen3_rmsnorm_mlx import (
            add_rms_norm as fused_add_rms_norm,
        )
        from sglang.kernels.ops.layernorm._qwen3_rmsnorm_mlx import (
            rms_norm as fused_rms_norm,
        )
        from sglang.kernels.ops.layernorm._qwen3_rmsnorm_mlx import (
            warmup_add_rms_norm,
            warmup_rms_norm,
        )

        # Qwen3 has two 128-wide Q/K RMSNorms inside the attention
        # preparation path and 1024-wide hidden/residual RMSNorms in the
        # transformer path.  The candidate kernel is deliberately specialized
        # to the latter; retain the staged MLX reference for the former.
        staged_rms_norm = qwen3_mlx._rms_norm
        staged_add_rms_norm = qwen3_mlx._add_rms_norm

        # Keep plain RMS and residual-add RMS independently selectable.  The
        # latter is called 56 times per 28-layer forward while plain RMS is
        # called only for the first layer, so a combined benchmark can hide
        # which candidate actually moves end-to-end prefill.
        if "fused_norm" in variants or "fused_rms" in variants:
            warmed_rms_epsilons = set()

            def fused_rms(value, weight, epsilon):
                if int(value.shape[-1]) != 1024:
                    return staged_rms_norm(value, weight, epsilon)
                epsilon = float(epsilon)
                if epsilon not in warmed_rms_epsilons:
                    warmup_rms_norm(epsilon)
                    warmed_rms_epsilons.add(epsilon)
                return fused_rms_norm(value, weight, epsilon)

            qwen3_mlx._rms_norm = fused_rms

        if "fused_norm" in variants or "fused_add_rms" in variants:
            warmed_add_epsilons = set()

            def fused_add_rms(value, residual, weight, epsilon):
                if int(value.shape[-1]) != 1024:
                    return staged_add_rms_norm(value, residual, weight, epsilon)
                epsilon = float(epsilon)
                if epsilon not in warmed_add_epsilons:
                    warmup_add_rms_norm(epsilon)
                    warmed_add_epsilons.add(epsilon)
                return fused_add_rms_norm(value, residual, weight, epsilon)

            qwen3_mlx._add_rms_norm = fused_add_rms

    return variants


def _run_sglang_worker(args: argparse.Namespace, manifest: dict[str, Any]) -> dict:
    import numpy as np
    import torch

    from sglang.benchmark import one_batch
    from sglang.srt.model_executor.cuda_graph_config import Phase
    from sglang.srt.server_args import PortArgs, ServerArgs

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required")
    torch.manual_seed(int(manifest["seed"]))
    np.random.seed(int(manifest["seed"]))
    mlx_benchmark_variants = _install_sglang_mlx_benchmark_variant()
    context_length = max(1024, int(manifest["input_len"]) + int(manifest["output_len"]))
    # This is a batch-one model-core benchmark.  Avoid reserving the old 2048
    # token floor when the measured request needs less than half of it; the
    # smaller Torch-owned KV pool materially lowers pressure on a 16 GiB Mac.
    max_total_tokens = max(
        1024,
        int(manifest["input_len"]) + int(manifest["output_len"]) + 64,
    )
    server_args = ServerArgs(
        model_path=manifest["model_path"],
        device="mps",
        dtype="bfloat16",
        attention_backend="mps",
        sampling_backend="pytorch",
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        context_length=context_length,
        max_total_tokens=max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
        # Qwen3-0.6B is a single-shard checkpoint. The buffered multithread
        # loader retains every CPU mmap tensor view without parallel I/O gain,
        # creating avoidable setup pressure on a 16 GiB unified-memory host.
        model_loader_extra_config={"enable_multithread_load": False},
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        log_level="warning",
    )
    if server_args.cuda_graph_config is not None:
        server_args.cuda_graph_config[Phase.DECODE].max_bs = 1
    one_batch._set_envs_and_config(server_args)
    one_batch.initialize_moe_config(server_args)
    one_batch.initialize_fp8_gemm_config(server_args)
    one_batch.initialize_fp4_gemm_config(server_args)
    port_args = PortArgs.init_new(server_args)
    runner = None
    try:
        runner, _ = one_batch.load_model(
            server_args,
            port_args,
            0,
            0,
            load_tokenizer=False,
        )
        max_total_num_tokens = int(runner.torch_runner.max_total_num_tokens)
        required_tokens = int(manifest["input_len"]) + int(manifest["output_len"]) - 1
        if max_total_num_tokens < required_tokens:
            raise RuntimeError(
                "actual SGLang KV pool is too small for the measured request: "
                f"capacity={max_total_num_tokens}, required={required_tokens}"
            )
        warmups = []
        reference_output_ids = None
        for index in range(args.warmup_trials):
            trial = _run_sglang_trial(
                runner,
                manifest,
                args.worker_profile,
                args.sync_mode,
                trace_ops=index == 0,
                collect_phase_timing=False,
            )
            warmups.append(trial)
            if index == 0:
                _validate_sglang_trace(
                    args.worker_profile,
                    trial["fused_op_trace"],
                    input_len=int(manifest["input_len"]),
                    output_len=int(manifest["output_len"]),
                )
            if reference_output_ids is None:
                reference_output_ids = trial["output_ids"]
            elif trial["output_ids"] != reference_output_ids:
                raise RuntimeError("SGLang warmup output IDs are not deterministic")

        def run_timed_trial() -> dict[str, Any]:
            return _run_sglang_trial(
                runner,
                manifest,
                args.worker_profile,
                args.sync_mode,
                trace_ops=False,
                collect_phase_timing=bool(
                    getattr(args, "collect_mlx_phase_timing", False)
                ),
            )

        def validate_timed_trial(trial: dict[str, Any]) -> None:
            if trial["output_ids"] != reference_output_ids:
                raise RuntimeError("SGLang timed output IDs differ from warmup")

        _publish_worker_phase(args, "timing")
        stable = _collect_stable_trials(
            run_timed_trial,
            requested_trials=args.trials,
            max_attempts=_resolved_max_trial_attempts(args),
            trial_delay=args.trial_delay,
            swap_limits=args,
            validate_trial=validate_timed_trial,
        )
        _require_complete_stable_trials(
            stable,
            args=args,
            engine="sglang",
            profile=args.worker_profile,
        )

        phase_timing_enabled = bool(
            getattr(args, "collect_mlx_phase_timing", False)
            and _profile_uses_whole_mlx(args.worker_profile)
        )
        return {
            "status": "completed",
            "engine": "sglang",
            "profile": args.worker_profile,
            "sync_mode": args.sync_mode,
            "warmup_trials": args.warmup_trials,
            "timed_trials": args.trials,
            "stable_trial_policy": _stable_trial_policy(args),
            "max_total_num_tokens": max_total_num_tokens,
            **_stable_collection_fields(stable),
            "reference_output_ids": reference_output_ids,
            "diagnostic_trace": warmups[0]["fused_op_trace"] if warmups else [],
            "initial_provider_state": (
                warmups[0]["provider_state_before"] if warmups else None
            ),
            "trials": stable.trials,
            "summary": _summarize_trials(stable.trials),
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_git": getattr(torch.version, "git_version", None),
                "mlx": _package_version("mlx"),
                "transformers": _package_version("transformers"),
            },
            "performance_environment": _performance_environment(),
            "timing_instrumentation": {
                "mlx_phase_timing": phase_timing_enabled,
                "summary_includes_instrumentation": phase_timing_enabled,
            },
            "mlx_benchmark_variants": list(mlx_benchmark_variants),
        }
    finally:
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.synchronize()
            with contextlib.suppress(Exception):
                runner.torch_runner.close_platform_operators()
            del runner
        with contextlib.suppress(Exception):
            torch.mps.synchronize()
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


def _mlx_cache_offsets(cache: list[Any]) -> list[int]:
    return [int(item.offset) for item in cache]


def _mlx_cache_snapshot(cache: list[Any]) -> list[dict[str, Any]]:
    """Validate and snapshot the dense Qwen3 MLX KV-cache allocation contract."""
    snapshot = []
    for layer, item in enumerate(cache):
        keys = getattr(item, "keys", None)
        values = getattr(item, "values", None)
        offset = int(item.offset)
        step = int(getattr(item, "step", 0))
        if (keys is None) != (values is None):
            raise RuntimeError(f"MLX KV cache layer {layer} has only one of K/V")
        if keys is None:
            if offset != 0:
                raise RuntimeError(
                    f"empty MLX KV cache layer {layer} has nonzero offset {offset}"
                )
            key_shape = value_shape = None
            key_dtype = value_dtype = None
            capacity = 0
        else:
            key_shape = tuple(int(dimension) for dimension in keys.shape)
            value_shape = tuple(int(dimension) for dimension in values.shape)
            if len(key_shape) != 4 or len(value_shape) != 4:
                raise RuntimeError(
                    f"MLX KV cache layer {layer} is not rank-4 dense K/V: "
                    f"K={key_shape}, V={value_shape}"
                )
            if key_shape != value_shape:
                raise RuntimeError(
                    f"MLX KV cache layer {layer} has mismatched K/V allocation: "
                    f"K={key_shape}, V={value_shape}"
                )
            capacity = key_shape[2]
            if offset > capacity:
                raise RuntimeError(
                    f"MLX KV cache layer {layer} offset {offset} exceeds "
                    f"capacity {capacity}"
                )
            key_dtype = str(keys.dtype)
            value_dtype = str(values.dtype)
        snapshot.append(
            {
                "layer": layer,
                "cache_type": type(item).__name__,
                "offset": offset,
                "step": step,
                "key_shape": key_shape,
                "value_shape": value_shape,
                "capacity_tokens": capacity,
                "key_dtype": key_dtype,
                "value_dtype": value_dtype,
            }
        )
    return snapshot


def _mlx_cache_growth_events(
    prefill: list[dict[str, Any]],
    final: list[dict[str, Any]],
    decode_steps: int,
) -> list[dict[str, Any]]:
    """Infer deterministic KVCache growth outside the measured decode region."""
    if not prefill or len(prefill) != len(final):
        raise RuntimeError("MLX KV cache snapshots have inconsistent layer counts")
    prefill_contracts = {
        (
            item["cache_type"],
            item["offset"],
            item["step"],
            item["capacity_tokens"],
        )
        for item in prefill
    }
    final_contracts = {
        (
            item["cache_type"],
            item["offset"],
            item["step"],
            item["capacity_tokens"],
        )
        for item in final
    }
    if len(prefill_contracts) != 1 or len(final_contracts) != 1:
        raise RuntimeError(
            "MLX Qwen3 KV-cache layers do not share one allocation contract"
        )
    prefill_shapes = {
        (
            item["key_shape"],
            item["value_shape"],
            item["key_dtype"],
            item["value_dtype"],
        )
        for item in prefill
    }
    final_shapes = {
        (
            item["key_shape"],
            item["value_shape"],
            item["key_dtype"],
            item["value_dtype"],
        )
        for item in final
    }
    if len(prefill_shapes) != 1 or len(final_shapes) != 1:
        raise RuntimeError("MLX Qwen3 KV-cache layer shapes or dtypes are inconsistent")
    cache_type, offset, step, capacity = next(iter(prefill_contracts))
    final_type, final_offset, final_step, final_capacity = next(iter(final_contracts))
    if cache_type != "KVCache" or final_type != cache_type:
        raise RuntimeError(f"expected dense mlx-lm KVCache, found {cache_type!r}")
    if step <= 0 or final_step != step:
        raise RuntimeError(f"invalid MLX KV-cache growth step: {step}, {final_step}")

    events = []
    for decode_index in range(decode_steps):
        next_offset = offset + 1
        if next_offset > capacity:
            before = capacity
            while next_offset > capacity:
                capacity += step
            events.append(
                {
                    "decode_index": decode_index,
                    "decode_step": decode_index + 1,
                    "offset_before": offset,
                    "offset_after": next_offset,
                    "capacity_before": before,
                    "capacity_after": capacity,
                    "grew_layers": list(range(len(prefill))),
                    "source": "inferred_from_prefill_final_and_step",
                }
            )
        offset = next_offset
    if offset != final_offset or capacity != final_capacity:
        raise RuntimeError(
            "MLX KV-cache final allocation does not match its step contract: "
            f"inferred offset/capacity={offset}/{capacity}, "
            f"observed={final_offset}/{final_capacity}"
        )
    return events


def _mlx_decode_fence(
    mx: Any, generation_stream: Any, sync_mode: str, *, final: bool
) -> None:
    """Apply exactly the harness-level fence selected for this decode point."""
    if (sync_mode == "aggregate") == final:
        mx.synchronize(generation_stream)


def _mlx_last_token_logits(model: Any, inputs: Any, cache: Any) -> Any:
    """Run one transformer call and project only the final hidden state."""
    hidden = model.model(inputs, cache=cache)
    last_hidden = hidden[:, -1:, :]
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(last_hidden)
    return model.lm_head(last_hidden)


def _run_mlx_model_core_trial(
    model: Any,
    manifest: dict[str, Any],
    sync_mode: str,
    *,
    generation_stream: Any,
    make_prompt_cache: Any,
) -> dict[str, Any]:
    import mlx.core as mx

    prompt_ids = manifest["prompt_token_ids"]
    trial_swap_before = _swap_snapshot()
    input_len = int(manifest["input_len"])
    output_len = int(manifest["output_len"])
    mx.synchronize(generation_stream)
    request_setup_start = time.perf_counter()
    cache = make_prompt_cache(model)
    request_setup_s = time.perf_counter() - request_setup_start
    mx.reset_peak_memory()

    with mx.stream(generation_stream):
        prefill_start = time.perf_counter()
        prepare_start = time.perf_counter()
        prompt = mx.array(prompt_ids, dtype=mx.int32)
        prompt_batch = prompt[None]
        prefill_prepare_submit_s = time.perf_counter() - prepare_start
        forward_start = time.perf_counter()
        logits = _mlx_last_token_logits(model, prompt_batch, cache)
        current = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(current, [item.state for item in cache])
        prefill_forward_submit_s = time.perf_counter() - forward_start
        mx.synchronize(generation_stream)
        prefill_s = time.perf_counter() - prefill_start
    prefill_snapshot = _mlx_cache_snapshot(cache)
    prefill_offsets = [item["offset"] for item in prefill_snapshot]
    if prefill_offsets != [input_len] * len(cache):
        raise RuntimeError(
            "native MLX prefill cache offsets do not match the prompt: "
            f"{prefill_offsets}"
        )

    output_arrays = [current]
    decode_step_s = []
    decode_prepare_submit_step_s = []
    decode_forward_submit_step_s = []
    with mx.stream(generation_stream):
        decode_start = time.perf_counter()
        for _ in range(output_len - 1):
            step_start = time.perf_counter()
            prepare_start = time.perf_counter()
            decode_input = current.reshape(1, 1)
            prepare_submit_s = time.perf_counter() - prepare_start
            forward_start = time.perf_counter()
            logits = _mlx_last_token_logits(model, decode_input, cache)
            current = mx.argmax(logits[:, -1, :], axis=-1)
            mx.async_eval(current, [item.state for item in cache])
            forward_submit_s = time.perf_counter() - forward_start
            _mlx_decode_fence(
                mx,
                generation_stream,
                sync_mode,
                final=False,
            )
            decode_prepare_submit_step_s.append(prepare_submit_s)
            decode_forward_submit_step_s.append(forward_submit_s)
            if sync_mode == "each":
                decode_step_s.append(time.perf_counter() - step_start)
            output_arrays.append(current)
        _mlx_decode_fence(mx, generation_stream, sync_mode, final=True)
        decode_s = time.perf_counter() - decode_start

    trial_swap_after = _swap_snapshot()
    expected_offset = input_len + output_len - 1
    final_snapshot = _mlx_cache_snapshot(cache)
    decode_offsets = [item["offset"] for item in final_snapshot]
    if decode_offsets != [expected_offset] * len(cache):
        raise RuntimeError(
            "native MLX decode cache offsets do not match evaluated forwards: "
            f"{decode_offsets}"
        )
    mx.eval(output_arrays)
    output_ids = [int(array.item()) for array in output_arrays]
    memory = _mlx_memory()
    decode_steps = output_len - 1
    result = {
        "metric_contract": "matched_model_core_last_token_logits",
        "reset_s": 0.0,
        "request_setup_s": request_setup_s,
        "prefill_s": prefill_s,
        "prefill_prepare_submit_s": prefill_prepare_submit_s,
        "prefill_forward_submit_s": prefill_forward_submit_s,
        "decode_s": decode_s,
        "total_s": prefill_s + decode_s,
        "request_total_s": request_setup_s + prefill_s + decode_s,
        "lifecycle_total_s": request_setup_s + prefill_s + decode_s,
        "prefill_tps": input_len / prefill_s,
        "decode_tps": decode_steps / decode_s if decode_steps else 0.0,
        "decode_steps": decode_steps,
        "decode_step_s": decode_step_s,
        "decode_prepare_submit_step_s": decode_prepare_submit_step_s,
        "decode_forward_submit_step_s": decode_forward_submit_step_s,
        "output_ids": output_ids,
        "cache_layers": len(cache),
        "prefill_cache_offsets": prefill_offsets,
        "prefill_cache_snapshot": prefill_snapshot,
        "decode_cache_offsets": decode_offsets,
        "final_cache_snapshot": final_snapshot,
        "cache_grow_decode_steps": _mlx_cache_growth_events(
            prefill_snapshot, final_snapshot, decode_steps
        ),
        "mlx_memory_bytes": memory,
        "system_swap_before": trial_swap_before,
        "system_swap_after": trial_swap_after,
        "system_swap_delta": _swap_delta(trial_swap_before, trial_swap_after),
        **_decode_tail_metrics(decode_step_s),
    }
    del output_arrays, cache, logits, current, prompt, prompt_batch
    mx.synchronize(generation_stream)
    return result


def _run_mlx_public_trial(
    model: Any,
    manifest: dict[str, Any],
    sync_mode: str,
    *,
    generation_stream: Any,
    make_prompt_cache: Any,
) -> dict[str, Any]:
    """Measure mlx-lm's public generate_step pipeline as a separate anchor."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt_ids = manifest["prompt_token_ids"]
    trial_swap_before = _swap_snapshot()
    input_len = int(manifest["input_len"])
    output_len = int(manifest["output_len"])
    mx.synchronize(generation_stream)
    request_setup_start = time.perf_counter()
    cache = make_prompt_cache(model)
    request_setup_s = time.perf_counter() - request_setup_start
    mx.reset_peak_memory()

    prompt = mx.array(prompt_ids, dtype=mx.int32)
    generator = generate_step(
        prompt,
        model,
        max_tokens=output_len,
        prompt_cache=cache,
        prefill_step_size=max(2048, input_len),
    )
    output_ids = []
    inter_yield_s = []
    started = time.perf_counter()
    first_yield_at = None
    previous_yield_at = None
    last_logprobs = None
    try:
        for _ in range(output_len):
            token, last_logprobs = next(generator)
            yielded_at = time.perf_counter()
            output_ids.append(int(token))
            if first_yield_at is None:
                first_yield_at = yielded_at
            else:
                inter_yield_s.append(yielded_at - previous_yield_at)
            previous_yield_at = yielded_at
    finally:
        generator.close()
    assert first_yield_at is not None and previous_yield_at is not None
    prefill_s = first_yield_at - started
    decode_s = previous_yield_at - first_yield_at
    total_s = previous_yield_at - started
    # generate_step submits one look-ahead model call before every yield.  Fence
    # it outside the public wall-clock metric before inspecting cache/memory.
    # Take the swap snapshot after that fence so the safety sample includes the
    # queued look-ahead even though its completion is not charged to throughput.
    mx.synchronize(generation_stream)
    trial_swap_after = _swap_snapshot()
    expected_offset = input_len + output_len
    cache_offsets = _mlx_cache_offsets(cache)
    if cache_offsets != [expected_offset] * len(cache):
        raise RuntimeError(
            "mlx-lm public generation cache offsets do not include its one-token "
            f"look-ahead: expected {expected_offset}, found {cache_offsets}"
        )
    memory = _mlx_memory()
    decode_steps = output_len - 1
    result = {
        "metric_contract": "mlx_lm_generate_step_public_pipeline",
        "requested_sync_mode": sync_mode,
        "reset_s": 0.0,
        "request_setup_s": request_setup_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "request_total_s": request_setup_s + total_s,
        "lifecycle_total_s": request_setup_s + total_s,
        "prefill_tps": input_len / prefill_s,
        "decode_tps": decode_steps / decode_s if decode_steps else 0.0,
        "public_generation_tps": output_len / decode_s if decode_s else 0.0,
        "decode_steps": decode_steps,
        "decode_step_s": inter_yield_s,
        "output_ids": output_ids,
        "cache_layers": len(cache),
        "decode_cache_offsets": cache_offsets,
        "lookahead_forwards": 1,
        "mlx_memory_bytes": memory,
        "system_swap_before": trial_swap_before,
        "system_swap_after": trial_swap_after,
        "system_swap_delta": _swap_delta(trial_swap_before, trial_swap_after),
    }
    del cache, last_logprobs, prompt
    mx.synchronize(generation_stream)
    return result


def _run_mlx_worker(args: argparse.Namespace, manifest: dict[str, Any]) -> dict:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    mlx_lm_source = Path(mlx_lm.__file__).resolve()
    expected_checkout = Path(args.mlx_lm_path).expanduser().resolve()
    if expected_checkout not in mlx_lm_source.parents:
        raise RuntimeError(
            f"mlx_lm imported from {mlx_lm_source}, outside requested checkout "
            f"{expected_checkout}"
        )
    model, model_config = load_model(Path(manifest["model_path"]), lazy=False)
    if int(model_config["vocab_size"]) != int(manifest["vocab_size"]):
        raise RuntimeError("native MLX loaded a different vocabulary")
    mx.random.seed(int(manifest["seed"]))
    trial_fn = (
        _run_mlx_public_trial
        if args.worker_engine == "mlx_lm_public"
        else _run_mlx_model_core_trial
    )
    with wired_limit(model, [generation_stream]):
        warmups = []
        reference_output_ids = None
        for _ in range(args.warmup_trials):
            trial = trial_fn(
                model,
                manifest,
                args.sync_mode,
                generation_stream=generation_stream,
                make_prompt_cache=make_prompt_cache,
            )
            warmups.append(trial)
            if reference_output_ids is None:
                reference_output_ids = trial["output_ids"]
            elif trial["output_ids"] != reference_output_ids:
                raise RuntimeError("native MLX warmup output IDs are not deterministic")

        def run_timed_trial() -> dict[str, Any]:
            return trial_fn(
                model,
                manifest,
                args.sync_mode,
                generation_stream=generation_stream,
                make_prompt_cache=make_prompt_cache,
            )

        def validate_timed_trial(trial: dict[str, Any]) -> None:
            if trial["output_ids"] != reference_output_ids:
                raise RuntimeError("native MLX timed output IDs differ from warmup")

        _publish_worker_phase(args, "timing")
        stable = _collect_stable_trials(
            run_timed_trial,
            requested_trials=args.trials,
            max_attempts=_resolved_max_trial_attempts(args),
            trial_delay=args.trial_delay,
            swap_limits=args,
            validate_trial=validate_timed_trial,
        )
        _require_complete_stable_trials(
            stable,
            args=args,
            engine=args.worker_engine,
            profile=args.worker_profile,
        )

    result = {
        "status": "completed",
        "engine": args.worker_engine,
        "profile": args.worker_profile,
        "sync_mode": args.sync_mode,
        "warmup_trials": args.warmup_trials,
        "timed_trials": args.trials,
        "stable_trial_policy": _stable_trial_policy(args),
        **_stable_collection_fields(stable),
        "reference_output_ids": reference_output_ids,
        "trials": stable.trials,
        "summary": _summarize_trials(stable.trials),
        "versions": {
            "python": sys.version,
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm", mlx_lm),
            "mlx_lm_source": str(mlx_lm_source),
            "transformers": _package_version("transformers"),
        },
        "performance_environment": _performance_environment(),
    }
    mx.synchronize(generation_stream)
    mx.clear_cache()
    return result


def _worker_main(args: argparse.Namespace) -> None:
    _publish_worker_phase(args, "setup")
    manifest = _read_json(Path(args.worker_manifest))
    result_path = Path(args.worker_result)
    try:
        if args.worker_engine == "sglang":
            result = _run_sglang_worker(args, manifest)
        else:
            result = _run_mlx_worker(args, manifest)
    except _StableTrialExhausted as exc:
        # Publish the accepted partial trials and every rejected attempt before
        # preserving the nonzero worker exit for the parent process.
        _write_json(result_path, exc.artifact)
        raise
    _write_json(result_path, result)


def _git_metadata(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
        tracked_status = run("status", "--short", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
            "tracked_dirty": None,
            "status_sha256": None,
        }
    return {
        "commit": commit,
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status_line_count": len(status.splitlines()),
        "diff_sha256": hashlib.sha256(
            run("diff", "--binary", "HEAD").encode()
        ).hexdigest(),
    }


def _running_benchmark_processes() -> list[dict[str, Any]]:
    import psutil

    current_pid = os.getpid()
    excluded_pids = {current_pid}
    with contextlib.suppress(psutil.Error):
        excluded_pids.update(parent.pid for parent in psutil.Process().parents())
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        with contextlib.suppress(psutil.Error):
            if process.pid in excluded_pids:
                continue
            command = " ".join(process.info["cmdline"] or [])
            if (
                "sglang.benchmark.mps_qwen3" in command
                or "sglang.benchmark.one_batch" in command
                or "sglang.launch_server" in command
            ):
                matches.append({"pid": process.pid, "command": command})
    return matches


@contextlib.contextmanager
def _exclusive_benchmark_lock():
    lock_path = Path(tempfile.gettempdir()) / "sglang-mps-qwen3.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another MPS benchmark owns the process lock {lock_path}"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextlib.contextmanager
def _controlled_termination():
    watched = (signal.SIGTERM, signal.SIGHUP)
    previous = {item: signal.getsignal(item) for item in watched}

    def request_stop(signum, _frame):
        raise RuntimeError(
            f"benchmark parent received {signal.Signals(signum).name}; "
            "stopping the active worker before exiting"
        )

    try:
        for item in watched:
            signal.signal(item, request_stop)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def _process_tree_memory(process: Any) -> dict[str, Optional[int]]:
    import psutil

    processes = [process]
    with contextlib.suppress(psutil.Error):
        processes.extend(process.children(recursive=True))
    rss = 0
    uss = 0
    has_uss = False
    seen = set()
    for item in processes:
        if item.pid in seen:
            continue
        seen.add(item.pid)
        with contextlib.suppress(psutil.Error):
            rss += int(item.memory_info().rss)
        with contextlib.suppress(psutil.Error, AttributeError):
            value = getattr(item.memory_full_info(), "uss")
            uss += int(value)
            has_uss = True
    return {"rss": rss, "uss": uss if has_uss else None}


def _swap_snapshot() -> dict[str, int]:
    import psutil

    swap = psutil.swap_memory()
    return {
        "total": int(swap.total),
        "used": int(swap.used),
        "free": int(swap.free),
        "sin": int(getattr(swap, "sin", 0)),
        "sout": int(getattr(swap, "sout", 0)),
    }


def _swap_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "used_growth": max(0, after["used"] - before["used"]),
        "in": max(0, after["sin"] - before["sin"]),
        "out": max(0, after["sout"] - before["sout"]),
    }


_TRIAL_SWAP_DELTA_KEYS = ("in", "out", "used_growth")


def _validated_trial_swap_delta(trial: dict[str, Any]) -> dict[str, int]:
    """Return complete nonnegative byte counters or reject the telemetry."""
    if not isinstance(trial, dict):
        raise ValueError("trial must be a dictionary")
    delta = trial.get("system_swap_delta")
    if not isinstance(delta, dict):
        raise ValueError("trial is missing system_swap_delta telemetry")
    missing = [key for key in _TRIAL_SWAP_DELTA_KEYS if key not in delta]
    if missing:
        raise ValueError(f"system_swap_delta is missing fields: {missing}")

    validated = {}
    for key in _TRIAL_SWAP_DELTA_KEYS:
        value = delta[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"system_swap_delta[{key!r}] must be an integer")
        if value < 0:
            raise ValueError(f"system_swap_delta[{key!r}] cannot be negative")
        validated[key] = value
    return validated


def _resolved_max_trial_attempts(args: argparse.Namespace) -> int:
    return int(args.max_trial_attempts or int(args.trials) * 3)


def _stable_trial_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_attempts": _resolved_max_trial_attempts(args),
        "swap_limits_mib": {
            "in": float(args.max_trial_swap_in_mb),
            "out": float(args.max_trial_swap_out_mb),
            "used_growth": float(args.max_trial_swap_growth_mb),
        },
        "profile_hard_limits_mib": {
            "out": float(args.max_profile_swap_out_mb),
            "used_growth": float(args.max_profile_swap_growth_mb),
        },
    }


def _trial_swap_rejection(
    trial: dict[str, Any], args: argparse.Namespace
) -> Optional[dict[str, Any]]:
    """Describe a system-wide swap-contaminated trial, failing closed."""
    try:
        delta = _validated_trial_swap_delta(trial)
    except ValueError as exc:
        return {"reason": "invalid_swap_telemetry", "error": str(exc)}

    values_mib = {
        "in": delta["in"] / 2**20,
        "out": delta["out"] / 2**20,
        "used_growth": delta["used_growth"] / 2**20,
    }
    limits_mib = _stable_trial_policy(args)["swap_limits_mib"]
    exceeded = [
        key for key in _TRIAL_SWAP_DELTA_KEYS if values_mib[key] > limits_mib[key]
    ]
    if not exceeded:
        return None
    return {
        "reason": "swap_threshold_exceeded",
        "exceeded": exceeded,
        "system_swap_delta": delta,
        "swap_mib": values_mib,
        "limits_mib": limits_mib,
    }


def _collect_stable_trials(
    run_trial: Callable[[], dict[str, Any]],
    *,
    requested_trials: int,
    max_attempts: int,
    trial_delay: float,
    swap_limits: argparse.Namespace,
    validate_trial: Optional[Callable[[dict[str, Any]], None]] = None,
) -> _StableTrialCollection:
    """Collect only accepted trials while retaining one-based provenance."""
    if requested_trials < 1:
        raise ValueError("requested_trials must be positive")
    if max_attempts < requested_trials:
        raise ValueError("max_attempts cannot be smaller than requested_trials")

    trials = []
    accepted_attempts = []
    rejected_trials = []
    attempted = 0
    while len(trials) < requested_trials and attempted < max_attempts:
        if trial_delay:
            time.sleep(trial_delay)
        attempted += 1
        trial = run_trial()
        if validate_trial is not None:
            validate_trial(trial)
        rejection = _trial_swap_rejection(trial, swap_limits)
        if rejection is not None:
            rejected_trials.append({"attempt": attempted, **rejection})
            continue
        accepted_attempts.append(attempted)
        trials.append(trial)

    return _StableTrialCollection(
        trials=trials,
        attempted=attempted,
        accepted_attempts=accepted_attempts,
        rejected_trials=rejected_trials,
    )


def _stable_collection_fields(
    collection: _StableTrialCollection,
) -> dict[str, Any]:
    return {
        "attempted_timed_trials": collection.attempted,
        "accepted_timed_attempts": collection.accepted_attempts,
        "rejected_timed_trials": collection.rejected_trials,
    }


def _require_complete_stable_trials(
    collection: _StableTrialCollection,
    *,
    args: argparse.Namespace,
    engine: str,
    profile: str,
) -> None:
    requested = int(args.trials)
    if len(collection.trials) == requested:
        return
    message = (
        f"{profile} collected only {len(collection.trials)}/{requested} stable "
        f"timed trials after {collection.attempted} attempts"
    )
    artifact = {
        "schema_version": 1,
        "status": "insufficient_stable_trials",
        "error_type": "StableTrialExhausted",
        "error": message,
        "engine": engine,
        "profile": profile,
        "requested_timed_trials": requested,
        "stable_trial_policy": _stable_trial_policy(args),
        **_stable_collection_fields(collection),
        "accepted_partial_trials": collection.trials,
    }
    raise _StableTrialExhausted(message, artifact)


def _validate_worker_stable_trials(
    result: dict[str, Any],
    args: argparse.Namespace,
    *,
    profile_name: Optional[str] = None,
) -> None:
    """Revalidate a successful worker artifact before using its summary."""
    requested = int(args.trials)
    if result.get("status") != "completed":
        raise RuntimeError(
            f"worker result has non-completed status: {result.get('status')!r}"
        )
    if result.get("stable_trial_policy") != _stable_trial_policy(args):
        raise RuntimeError("worker stable-trial policy differs from parent arguments")
    if result.get("timed_trials") != requested:
        raise RuntimeError("worker timed_trials differs from the requested trial count")

    trials = result.get("trials")
    rejected = result.get("rejected_timed_trials")
    accepted_attempts = result.get("accepted_timed_attempts")
    attempted = result.get("attempted_timed_trials")
    if not isinstance(trials, list) or len(trials) != requested:
        raise RuntimeError(
            "worker artifact does not contain the requested stable trials"
        )
    if not isinstance(rejected, list) or not isinstance(accepted_attempts, list):
        raise RuntimeError("worker artifact has invalid attempt provenance")
    if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted < 0:
        raise RuntimeError(
            "worker attempted_timed_trials must be a nonnegative integer"
        )
    if len(accepted_attempts) != len(trials):
        raise RuntimeError("accepted attempt count does not match stable trial count")

    def validate_attempt(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"{label} must contain positive integer attempts")
        return value

    accepted_numbers = [
        validate_attempt(value, "accepted_timed_attempts")
        for value in accepted_attempts
    ]
    rejected_numbers = []
    for rejection in rejected:
        if not isinstance(rejection, dict):
            raise RuntimeError("rejected_timed_trials entries must be dictionaries")
        rejected_numbers.append(
            validate_attempt(rejection.get("attempt"), "rejected_timed_trials")
        )
    if accepted_numbers != sorted(accepted_numbers) or rejected_numbers != sorted(
        rejected_numbers
    ):
        raise RuntimeError("worker attempt provenance is not ordered")
    if attempted != len(accepted_numbers) + len(rejected_numbers):
        raise RuntimeError("worker attempted/accepted/rejected counts are inconsistent")
    if sorted(accepted_numbers + rejected_numbers) != list(range(1, attempted + 1)):
        raise RuntimeError("worker attempt provenance is not a complete partition")
    if attempted > _resolved_max_trial_attempts(args):
        raise RuntimeError("worker exceeded max_trial_attempts")

    for index, trial in enumerate(trials):
        rejection = _trial_swap_rejection(trial, args)
        if rejection is not None:
            raise RuntimeError(
                f"accepted trial {index} fails strict swap validation: {rejection}"
            )
    recomputed_summary = _summarize_trials(trials)
    if result.get("summary") != recomputed_summary:
        raise RuntimeError("worker summary was not computed from accepted trials only")
    _validate_worker_diagnostics(result, args, profile_name=profile_name)


def _validate_profile_swap(
    profile_name: str,
    before: dict[str, int],
    current: dict[str, int],
    *,
    max_growth_bytes: float,
    max_out_bytes: float,
) -> dict[str, int]:
    """Reject system swap pressure early enough to protect a 16 GiB host."""
    delta = _swap_delta(before, current)
    if delta["used_growth"] > max_growth_bytes:
        raise RuntimeError(
            f"{profile_name} grew swap by {delta['used_growth'] / 2**20:.1f} MiB; "
            "terminating the model process before it can create more pressure"
        )
    if delta["out"] > max_out_bytes:
        raise RuntimeError(
            f"{profile_name} swapped out {delta['out'] / 2**20:.1f} MiB; "
            "terminating the model process before it can continue thrashing"
        )
    return delta


def _validate_swap_headroom(
    label: str, current: dict[str, int], *, minimum_free_bytes: float
) -> None:
    """Keep a reserve before macOS exhausts its configured swap space."""
    total = int(current.get("total", 0))
    free = int(current.get("free", 0))
    if total and free < minimum_free_bytes:
        raise RuntimeError(
            f"{label} left only {free / 2**20:.1f} MiB free swap; "
            f"at least {minimum_free_bytes / 2**20:.1f} MiB is required"
        )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    if process.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)

    deadline = time.monotonic() + 5
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_exists(process.pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
    if _process_group_exists(process.pid):
        raise RuntimeError(
            f"benchmark process group {process.pid} survived SIGTERM and SIGKILL"
        )


def _wait_for_available_memory(minimum_bytes: int, timeout: float) -> int:
    import psutil

    deadline = time.monotonic() + timeout
    while True:
        available = int(psutil.virtual_memory().available)
        if available >= minimum_bytes:
            return available
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"available memory recovered only to {available / 2**30:.2f} GiB; "
                f"required {minimum_bytes / 2**30:.2f} GiB before the next profile"
            )
        time.sleep(0.5)


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _read_optional_worker_artifact(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        artifact = _read_json(path)
    except Exception as exc:
        return {
            "status": "invalid_worker_artifact",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
    if not isinstance(artifact, dict):
        return {
            "status": "invalid_worker_artifact",
            "error": "worker artifact root must be a dictionary",
            "path": str(path),
        }
    return artifact


def _memory_guard_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_start_available_gib": float(args.min_start_available_gb),
        "hard_min_available_gib": float(args.hard_min_available_gb),
        "soft_min_available_gib": float(args.min_available_gb),
        "setup_grace_s": float(args.setup_memory_grace_s),
        "low_memory_sustain_s": float(args.low_memory_sustain_s),
        "recovery_margin_gib": float(args.available_recovery_margin_gb),
    }


def _validate_memory_guard_args(args: argparse.Namespace) -> None:
    positive_gib = {
        "hard_min_available_gb": args.hard_min_available_gb,
        "min_available_gb": args.min_available_gb,
        "min_start_available_gb": args.min_start_available_gb,
    }
    invalid_positive = {
        name: value
        for name, value in positive_gib.items()
        if not math.isfinite(value) or value <= 0
    }
    nonnegative = {
        "setup_memory_grace_s": args.setup_memory_grace_s,
        "low_memory_sustain_s": args.low_memory_sustain_s,
        "available_recovery_margin_gb": args.available_recovery_margin_gb,
    }
    invalid_nonnegative = {
        name: value
        for name, value in nonnegative.items()
        if not math.isfinite(value) or value < 0
    }
    if invalid_positive or invalid_nonnegative:
        raise ValueError(
            "memory guard values must be finite with positive GiB floors and "
            "nonnegative durations/margin: "
            f"{invalid_positive | invalid_nonnegative}"
        )
    if args.hard_min_available_gb >= args.min_available_gb:
        raise ValueError("hard_min_available_gb must be smaller than min_available_gb")
    if args.min_start_available_gb < args.min_available_gb:
        raise ValueError(
            "min_start_available_gb cannot be smaller than min_available_gb"
        )


def _memory_guard_failure_artifact(
    *,
    profile_name: str,
    profile: Profile,
    violation: _ParentMemoryGuardViolation,
    policy: dict[str, float],
    phase: str,
    elapsed_s: float,
    pid: int,
    rss_bytes: int,
    uss_bytes: Optional[int],
    available_before_bytes: int,
    minimum_available_bytes: int,
    swap_before: dict[str, int],
    swap_current: dict[str, int],
    log_path: Path,
    phase_path: Path,
    previous_worker_artifact: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the result a killed worker can no longer publish itself."""
    canonical_phase = str(violation.details.get("phase", phase))
    artifact = {
        "schema_version": 1,
        "status": "terminated_by_memory_guard",
        "error_type": type(violation).__name__,
        "error": str(violation),
        "engine": profile.engine,
        "profile": profile_name,
        "phase": canonical_phase,
        "memory_guard_policy": policy,
        "memory_guard_violation": violation.details,
        "process": {
            "pid": pid,
            "elapsed_s": elapsed_s,
            "rss_bytes": rss_bytes,
            "uss_bytes": uss_bytes,
            "available_memory_before_bytes": available_before_bytes,
            "minimum_available_memory_bytes": minimum_available_bytes,
            "worker_log": str(log_path),
            "worker_phase": str(phase_path),
        },
        "system_swap_before": swap_before,
        "system_swap_at_violation": swap_current,
        "system_swap_delta": _swap_delta(swap_before, swap_current),
    }
    if previous_worker_artifact is not None:
        artifact["worker_artifact_before_termination"] = previous_worker_artifact
    return artifact


def _prepare_profile_launch(args: argparse.Namespace) -> None:
    """Re-establish the 16 GiB host preconditions immediately before spawn."""
    remaining = _running_benchmark_processes()
    if remaining:
        raise RuntimeError(
            "a benchmark process remained before profile launch; refusing "
            f"to overlap it: {remaining}"
        )
    _wait_for_available_memory(
        int(args.min_start_available_gb * 2**30),
        args.memory_recovery_timeout,
    )
    remaining = _running_benchmark_processes()
    if remaining:
        raise RuntimeError(
            "a benchmark process remained before profile launch; refusing "
            f"to overlap it: {remaining}"
        )


def _build_worker_command(
    args: argparse.Namespace,
    profile_name: str,
    profile: Profile,
    manifest_path: Path,
    result_path: Path,
) -> list[str]:
    """Build the worker CLI, including every stable-trial policy input."""
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.mps_qwen3",
        "--worker-engine",
        profile.engine,
        "--worker-profile",
        profile_name,
        "--worker-manifest",
        str(manifest_path),
        "--worker-result",
        str(result_path),
        "--worker-phase",
        str(_worker_phase_path(result_path)),
        "--warmup-trials",
        str(args.warmup_trials),
        "--trials",
        str(args.trials),
        "--sync-mode",
        args.sync_mode,
        "--trial-delay",
        str(args.trial_delay),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-trial-swap-growth-mb",
        str(args.max_trial_swap_growth_mb),
        "--max-trial-swap-out-mb",
        str(args.max_trial_swap_out_mb),
        "--max-trial-swap-in-mb",
        str(args.max_trial_swap_in_mb),
        "--max-profile-swap-growth-mb",
        str(args.max_profile_swap_growth_mb),
        "--max-profile-swap-out-mb",
        str(args.max_profile_swap_out_mb),
        "--max-trial-attempts",
        str(args.max_trial_attempts),
        "--mlx-lm-path",
        str(Path(args.mlx_lm_path).expanduser().resolve()),
    ]
    if getattr(args, "collect_mlx_phase_timing", False):
        command.append("--collect-mlx-phase-timing")
    return command


def _run_profile(
    args: argparse.Namespace,
    profile_name: str,
    profile: Profile,
    manifest_path: Path,
    directory: Path,
) -> dict[str, Any]:
    import psutil

    result_path = directory / f"{profile_name}.json"
    phase_path = _worker_phase_path(result_path)
    log_path = directory / f"{profile_name}.log"
    # A failed worker is allowed to publish a structured failure at this path.
    # Remove an older run first so a non-writing crash can never look current.
    result_path.unlink(missing_ok=True)
    phase_path.unlink(missing_ok=True)
    command = _build_worker_command(
        args, profile_name, profile, manifest_path, result_path
    )
    environment = os.environ.copy()
    for name in _PROVIDER_ENV_VARS:
        environment.pop(name, None)
    environment.update(profile.environment)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    if profile.engine.startswith("mlx_lm"):
        mlx_lm_path = Path(args.mlx_lm_path).expanduser().resolve()
        if not (mlx_lm_path / "mlx_lm").is_dir():
            raise ValueError(f"invalid mlx-lm source checkout: {mlx_lm_path}")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(mlx_lm_path)
            if not existing
            else os.pathsep.join((str(mlx_lm_path), existing))
        )

    print(f"Starting {profile_name}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    memory_policy = _memory_guard_policy(args)
    available_guard = _AvailableMemoryGuard(
        hard_min_bytes=int(args.hard_min_available_gb * 2**30),
        soft_min_bytes=int(args.min_available_gb * 2**30),
        setup_grace_s=float(args.setup_memory_grace_s),
        low_memory_sustain_s=float(args.low_memory_sustain_s),
        recovery_margin_bytes=int(args.available_recovery_margin_gb * 2**30),
        started_at=started,
    )
    available_before = int(psutil.virtual_memory().available)
    swap_before = _swap_snapshot()
    _validate_swap_headroom(
        profile_name,
        swap_before,
        minimum_free_bytes=args.min_swap_free_mb * 2**20,
    )
    peak_rss = 0
    peak_uss: Optional[int] = None
    minimum_available = available_before
    last_report = 0.0
    last_phase = "setup"
    elapsed = 0.0
    rss = 0
    uss: Optional[int] = None
    available = available_before
    current_swap = swap_before
    with log_path.open("w") as log_file:
        process: Optional[subprocess.Popen] = None
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            ps_process = psutil.Process(process.pid)
            while process.poll() is None:
                now = time.monotonic()
                elapsed = now - started
                last_phase = _advance_worker_phase(
                    last_phase, _read_worker_phase(phase_path)
                )
                process_memory = _process_tree_memory(ps_process)
                rss = int(process_memory["rss"] or 0)
                uss = process_memory["uss"]
                available = int(psutil.virtual_memory().available)
                current_swap = _swap_snapshot()
                peak_rss = max(peak_rss, rss)
                if uss is not None:
                    peak_uss = max(peak_uss or 0, uss)
                minimum_available = min(minimum_available, available)
                try:
                    _validate_profile_swap(
                        profile_name,
                        swap_before,
                        current_swap,
                        max_growth_bytes=args.max_profile_swap_growth_mb * 2**20,
                        max_out_bytes=args.max_profile_swap_out_mb * 2**20,
                    )
                except RuntimeError as memory_exc:
                    raise _ParentMemoryGuardViolation(
                        str(memory_exc),
                        {
                            "reason": "profile_swap_limit",
                            "phase": last_phase,
                            "elapsed_s": elapsed,
                        },
                    ) from memory_exc
                try:
                    _validate_swap_headroom(
                        profile_name,
                        current_swap,
                        minimum_free_bytes=args.min_swap_free_mb * 2**20,
                    )
                except RuntimeError as memory_exc:
                    raise _ParentMemoryGuardViolation(
                        str(memory_exc),
                        {
                            "reason": "swap_headroom",
                            "phase": last_phase,
                            "elapsed_s": elapsed,
                        },
                    ) from memory_exc
                if rss > args.max_rss_gb * 2**30:
                    raise _ParentMemoryGuardViolation(
                        f"{profile_name} exceeded max RSS {args.max_rss_gb:.2f} GiB",
                        {
                            "reason": "max_rss",
                            "phase": last_phase,
                            "elapsed_s": elapsed,
                            "rss_bytes": rss,
                            "max_rss_bytes": int(args.max_rss_gb * 2**30),
                        },
                    )
                available_violation = available_guard.observe(
                    now=now,
                    available_bytes=available,
                    phase=last_phase,
                )
                if available_violation is not None:
                    minimum_available = int(
                        available_guard.minimum_available_bytes or minimum_available
                    )
                    reason = available_violation["reason"]
                    violation_phase = str(available_violation["phase"])
                    if reason == "hard_min_available":
                        message = (
                            f"{profile_name} left only {available / 2**30:.2f} GiB "
                            "available system memory, below the immediate hard floor "
                            f"of {args.hard_min_available_gb:.2f} GiB"
                        )
                    else:
                        message = (
                            f"{profile_name} remained in an unrecovered memory-pressure "
                            f"episode for {available_violation['pressure_duration_s']:.2f}s "
                            f"during {violation_phase}; current available memory "
                            f"{available / 2**30:.2f} GiB is below the soft floor "
                            f"{args.min_available_gb:.2f} GiB and recovery requires "
                            f"{args.min_available_gb + args.available_recovery_margin_gb:.2f} GiB"
                        )
                    raise _ParentMemoryGuardViolation(
                        message,
                        available_violation,
                    )
                if elapsed > args.timeout:
                    raise TimeoutError(
                        f"{profile_name} exceeded timeout {args.timeout:.0f}s"
                    )
                if elapsed - last_report >= 10:
                    print(
                        f"  {profile_name}: elapsed={elapsed:.0f}s "
                        f"rss={rss / 2**30:.2f} GiB "
                        f"available={available / 2**30:.2f} GiB",
                        flush=True,
                    )
                    last_report = elapsed
                time.sleep(0.25)
        except BaseException as exc:
            termination_error = None
            if process is not None:
                try:
                    _terminate_process_group(process)
                except BaseException as termination_exc:
                    termination_error = termination_exc
            log_file.flush()
            previous_worker_artifact = _read_optional_worker_artifact(result_path)
            if isinstance(exc, _ParentMemoryGuardViolation):
                assert process is not None
                violation_phase = str(exc.details.get("phase", last_phase))
                worker_artifact = _memory_guard_failure_artifact(
                    profile_name=profile_name,
                    profile=profile,
                    violation=exc,
                    policy=memory_policy,
                    phase=violation_phase,
                    elapsed_s=elapsed,
                    pid=process.pid,
                    rss_bytes=rss,
                    uss_bytes=uss,
                    available_before_bytes=available_before,
                    minimum_available_bytes=minimum_available,
                    swap_before=swap_before,
                    swap_current=current_swap,
                    log_path=log_path,
                    phase_path=phase_path,
                    previous_worker_artifact=previous_worker_artifact,
                )
                if termination_error is not None:
                    worker_artifact["termination_error"] = str(termination_error)
                _write_json(result_path, worker_artifact)
            else:
                worker_artifact = previous_worker_artifact
            termination_suffix = (
                "\n--- termination error ---\n" + str(termination_error)
                if termination_error is not None
                else ""
            )
            raise _WorkerProfileError(
                f"{exc}{termination_suffix}\n--- worker log tail ---\n"
                f"{_log_tail(log_path)}",
                worker_artifact,
            ) from exc
        assert process is not None
        _terminate_process_group(process)
        last_phase = _advance_worker_phase(last_phase, _read_worker_phase(phase_path))
    elapsed = time.monotonic() - started
    available_after = int(psutil.virtual_memory().available)
    swap_after = _swap_snapshot()
    worker_artifact = _read_optional_worker_artifact(result_path)
    if process.returncode != 0:
        raise _WorkerProfileError(
            f"{profile_name} worker exited with {process.returncode}\n"
            f"--- worker log tail ---\n{_log_tail(log_path)}",
            worker_artifact,
        )
    if worker_artifact is None:
        raise _WorkerProfileError(
            f"{profile_name} worker produced no result\n"
            f"--- worker log tail ---\n{_log_tail(log_path)}",
            None,
        )
    swap_delta = _validate_profile_swap(
        profile_name,
        swap_before,
        swap_after,
        max_growth_bytes=args.max_profile_swap_growth_mb * 2**20,
        max_out_bytes=args.max_profile_swap_out_mb * 2**20,
    )
    swap_growth = swap_delta["used_growth"]
    swap_in = max(0, swap_after["sin"] - swap_before["sin"])
    swap_out = swap_delta["out"]
    result = worker_artifact
    try:
        _validate_worker_stable_trials(result, args, profile_name=profile_name)
    except Exception as exc:
        raise _WorkerProfileError(
            f"{profile_name} worker artifact failed strict validation: {exc}",
            worker_artifact,
        ) from exc
    result["process"] = {
        "worker_log": str(log_path),
        "worker_result": str(result_path),
        "worker_phase": str(phase_path),
        "last_worker_phase": last_phase,
        "memory_guard_policy": memory_policy,
        "elapsed_s": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_uss_bytes": peak_uss,
        "available_memory_before_bytes": available_before,
        "available_memory_after_bytes": available_after,
        "minimum_available_memory_bytes": minimum_available,
        "swap_before": swap_before,
        "swap_after": swap_after,
        "swap_growth_bytes": swap_growth,
        "swap_in_bytes": swap_in,
        "swap_out_bytes": swap_out,
    }
    median_prefill = result["summary"]["prefill_s"]["median"]
    median_decode_tps = result["summary"]["decode_tps"]["median"]
    if profile.engine == "mlx_lm_public":
        median_public_tps = result["summary"]["public_generation_tps"]["median"]
        throughput = (
            f"public_generation={median_public_tps:.2f} tok/s "
            f"remaining_decode={median_decode_tps:.2f} tok/s"
        )
    else:
        throughput = f"decode={median_decode_tps:.2f} tok/s"
    print(
        f"Finished {profile_name}: prefill={median_prefill:.4f}s "
        f"{throughput} peak_rss={peak_rss / 2**30:.2f} GiB",
        flush=True,
    )
    return result


def _first_token_id_mismatch(expected: list[int], actual: list[int]) -> Optional[dict]:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        return {"index": min(len(expected), len(actual)), "length_mismatch": True}
    return None


def _token_ids_are_strictly_comparable(result: dict[str, Any]) -> bool:
    return result.get("engine") != "mlx_lm_public"


def _classify_token_id_parity(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify final benchmark parity and annotate each result.

    The returned status is written directly to the final benchmark artifact, so
    a provider mismatch cannot be represented as a completed comparison even
    when the caller explicitly keeps the diagnostic results.
    """
    by_name = {result["profile"]: result for result in results}
    reference_name, reference_validated = _select_reference_profile(results)
    reference_ids = by_name[reference_name]["reference_output_ids"]
    mismatches = {}
    noncomparable_anchor_token_ids = {}
    for profile_name, result in by_name.items():
        mismatch = _first_token_id_mismatch(
            reference_ids, result["reference_output_ids"]
        )
        is_comparable = _token_ids_are_strictly_comparable(result)
        result["token_ids_comparable_to_reference"] = is_comparable
        if not is_comparable:
            result["token_ids_match_reference"] = None
            noncomparable_anchor_token_ids[profile_name] = {
                "reason": (
                    "public generate_step uses prefix-last execution, log-softmax, "
                    "and one-token look-ahead rather than the matched full-prompt graph"
                ),
                "difference": mismatch,
            }
            continue

        result["token_ids_match_reference"] = mismatch is None
        if mismatch is not None:
            mismatches[profile_name] = mismatch

    status = "invalid_token_id_mismatch" if mismatches else "completed"
    if not mismatches and not reference_validated:
        status = "completed_without_torch_reference"
    return {
        "status": status,
        "reference_profile": reference_name,
        "reference_validated": reference_validated,
        "reference_validation_reason": (
            "greedy token-ID parity was validated against sglang-torch"
            if reference_validated
            else "sglang-torch was not included; token-ID parity is unvalidated"
        ),
        "token_id_mismatches": mismatches,
        "noncomparable_anchor_token_ids": noncomparable_anchor_token_ids,
    }


def _source_fingerprints(repository: Path) -> dict[str, str]:
    relative_paths = (
        "python/sglang/benchmark/mps_qwen3.py",
        "python/sglang/benchmark/one_batch.py",
        "python/sglang/srt/environ.py",
        "python/sglang/srt/hardware_backend/mps/model_ops/selection.py",
        "python/sglang/srt/hardware_backend/mps/model_ops/plan.py",
        "python/sglang/srt/hardware_backend/mps/model_ops/qwen3.py",
        "python/sglang/srt/hardware_backend/mps/model_ops/qwen3_mlx.py",
        "python/sglang/srt/layers/attention/mps_backend.py",
        "python/sglang/srt/models/qwen3.py",
        "python/sglang/srt/utils/_phase_timing.py",
        "python/sglang/srt/utils/tensor_bridge.py",
        "python/sglang/kernels/ops/layernorm/__init__.py",
        "python/sglang/kernels/ops/layernorm/_qwen3_rmsnorm_mlx.py",
        "python/sglang/kernels/ops/attention/qwen3_mps.py",
        "python/sglang/kernels/ops/attention/qwen3_mlx.py",
        "python/sglang/kernels/ops/attention/_qwen3_mlx_metal.py",
        "python/sglang/kernels/ops/attention/_qwen3_qkv_mlx_metal.py",
        "python/sglang/kernels/ops/kvcache/qwen3.py",
        "python/sglang/kernels/ops/kvcache/_qwen3_deferred_kv_commit_metal_jit.py",
    )
    return {
        relative_path: _sha256_file(repository / relative_path)
        for relative_path in relative_paths
    }


def _benchmark_arguments(args: argparse.Namespace) -> dict[str, Any]:
    hidden = {
        "worker_engine",
        "worker_profile",
        "worker_manifest",
        "worker_result",
        "worker_phase",
    }
    return {name: value for name, value in vars(args).items() if name not in hidden}


def _parent_main(args: argparse.Namespace) -> None:
    import psutil

    existing = _running_benchmark_processes()
    if existing:
        raise RuntimeError(
            "another SGLang/model benchmark process is already running; refusing "
            f"to overlap it on a 16 GiB machine: {existing}"
        )
    available = int(psutil.virtual_memory().available)
    if available < args.min_start_available_gb * 2**30:
        raise RuntimeError(
            f"only {available / 2**30:.2f} GiB is currently available; "
            f"at least {args.min_start_available_gb:.2f} GiB is required before "
            "loading a model"
        )
    _validate_swap_headroom(
        "benchmark startup",
        _swap_snapshot(),
        minimum_free_bytes=args.min_swap_free_mb * 2**20,
    )
    manifest = _make_manifest(
        Path(args.model_path),
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
    )
    profile_names = list(args.profiles)
    if args.reverse_order:
        profile_names.reverse()
    if "MLX_DISABLE_COMPILE" in os.environ:
        raise RuntimeError(
            "MLX_DISABLE_COMPILE is set; refusing to benchmark a configuration "
            "that silently disables the compiled MLX execution path"
        )
    whole_model_profiles = [
        name
        for name in profile_names
        if _first_provider(
            PROFILES[name].environment,
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
        )
        == "mlx"
    ]
    if whole_model_profiles and not 512 <= args.input_len <= 1024:
        raise ValueError(
            "whole-model MLX cold prefill is benchmarkable only for the static "
            f"512..1024 token contract; input_len={args.input_len}, "
            f"profiles={whole_model_profiles}"
        )

    repository = Path(__file__).resolve().parents[3]
    output_path = Path(args.output).expanduser().resolve()
    directory = output_path.with_name(f"{output_path.name}.artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sglang_git": _git_metadata(repository),
        "mlx_lm_git": _git_metadata(Path(args.mlx_lm_path).expanduser().resolve()),
        "source_sha256": _source_fingerprints(repository),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "profiles": profile_names,
        "arguments": _benchmark_arguments(args),
        "memory_guard_policy": _memory_guard_policy(args),
        "performance_environment": _performance_environment(),
        "artifact_directory": str(directory),
    }
    manifest_path = directory / "manifest.json"
    checkpoint_path = directory / "checkpoint.json"
    _write_json(manifest_path, manifest)
    results = []
    try:
        for profile_name in profile_names:
            # Metadata and content hashing can take long enough for unrelated
            # applications to consume the memory checked at process startup.
            # Re-establish both launch preconditions immediately before every
            # worker, including the first profile.
            _prepare_profile_launch(args)
            results.append(
                _run_profile(
                    args,
                    profile_name,
                    PROFILES[profile_name],
                    manifest_path,
                    directory,
                )
            )
            pool_contract = _validate_sglang_pool_capacities(results, manifest)
            _write_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "manifest": manifest,
                    "metadata": metadata,
                    "completed_profiles": [item["profile"] for item in results],
                    "sglang_kv_pool_contract": pool_contract,
                    "results": results,
                },
            )
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest": manifest,
            "metadata": metadata,
            "sync_mode": args.sync_mode,
            "results": results,
        }
        worker_artifact = getattr(exc, "worker_artifact", None)
        if worker_artifact is not None:
            failure["failed_profile"] = worker_artifact.get("profile", profile_name)
            failure["failed_worker_artifact"] = worker_artifact
        _write_json(checkpoint_path, failure)
        _write_json(output_path, failure)
        print(f"Wrote partial benchmark results to {output_path}", flush=True)
        raise

    parity = _classify_token_id_parity(results)
    payload = {
        "schema_version": 1,
        "status": parity["status"],
        "manifest": manifest,
        "metadata": metadata,
        "sync_mode": args.sync_mode,
        "reference_profile": parity["reference_profile"],
        "reference_validated": parity["reference_validated"],
        "reference_validation_reason": parity["reference_validation_reason"],
        "parity_scope": "greedy_token_ids_only",
        "numerical_parity_validated": False,
        "sglang_kv_pool_contract": _validate_sglang_pool_capacities(results, manifest),
        "token_id_mismatches": parity["token_id_mismatches"],
        "noncomparable_anchor_token_ids": parity["noncomparable_anchor_token_ids"],
        "results": results,
    }
    _write_json(checkpoint_path, payload)
    _write_json(output_path, payload)
    print(f"Wrote benchmark results to {output_path}", flush=True)
    if parity["token_id_mismatches"] and not args.allow_output_mismatch:
        raise RuntimeError(
            "provider token IDs differ from "
            f"{parity['reference_profile']}: {parity['token_id_mismatches']}; "
            "results were saved but cannot be used as token-ID parity evidence"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--mlx-lm-path", default="mlx-lm")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=_DEFAULT_PROFILES,
    )
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-trials", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--sync-mode", choices=("each", "aggregate"), default="each")
    parser.add_argument(
        "--collect-mlx-phase-timing",
        action="store_true",
        help=(
            "Record bridge/provider phase timings for whole-MLX trials. "
            "This adds diagnostic overhead and is excluded from normal perf runs."
        ),
    )
    parser.add_argument("--trial-delay", type=float, default=1.0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-rss-gb", type=float, default=12.0)
    parser.add_argument("--min-start-available-gb", type=float, default=5.0)
    parser.add_argument("--min-available-gb", type=float, default=2.5)
    parser.add_argument("--hard-min-available-gb", type=float, default=2.0)
    parser.add_argument("--setup-memory-grace-s", type=float, default=120.0)
    parser.add_argument("--low-memory-sustain-s", type=float, default=5.0)
    parser.add_argument("--available-recovery-margin-gb", type=float, default=0.25)
    parser.add_argument("--min-swap-free-mb", type=float, default=256.0)
    parser.add_argument("--max-trial-swap-growth-mb", type=float, default=128.0)
    parser.add_argument("--max-trial-swap-out-mb", type=float, default=16.0)
    parser.add_argument("--max-trial-swap-in-mb", type=float, default=64.0)
    # These are process-safety caps, intentionally looser than the default
    # per-trial filters. Preserve the original 512/64 MiB hard-stop defaults.
    parser.add_argument("--max-profile-swap-growth-mb", type=float, default=512.0)
    parser.add_argument("--max-profile-swap-out-mb", type=float, default=64.0)
    parser.add_argument(
        "--max-trial-attempts",
        type=int,
        default=0,
        help="Maximum timed attempts used to collect stable trials; 0 means 3x trials.",
    )
    parser.add_argument("--memory-recovery-timeout", type=float, default=30.0)
    parser.add_argument("--allow-output-mismatch", action="store_true")
    parser.add_argument("--output", default="/tmp/sglang-mps-qwen3.json")

    parser.add_argument(
        "--worker-engine",
        choices=("sglang", "mlx_lm_core", "mlx_lm_public"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-profile", choices=tuple(PROFILES), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", help=argparse.SUPPRESS)
    return parser


def cli_main() -> None:
    args = _parser().parse_args()
    if args.warmup_trials < 1:
        raise ValueError("warmup_trials must be at least one")
    if args.trials < 1:
        raise ValueError("trials must be at least one")
    if args.max_trial_attempts < 0:
        raise ValueError("max_trial_attempts cannot be negative")
    if args.max_trial_attempts and args.max_trial_attempts < args.trials:
        raise ValueError("max_trial_attempts cannot be smaller than trials")
    _validate_memory_guard_args(args)
    swap_limits = {
        "max_trial_swap_in_mb": args.max_trial_swap_in_mb,
        "max_trial_swap_out_mb": args.max_trial_swap_out_mb,
        "max_trial_swap_growth_mb": args.max_trial_swap_growth_mb,
        "max_profile_swap_out_mb": args.max_profile_swap_out_mb,
        "max_profile_swap_growth_mb": args.max_profile_swap_growth_mb,
    }
    invalid_swap_limits = {
        name: value
        for name, value in swap_limits.items()
        if not math.isfinite(value) or value < 0
    }
    if invalid_swap_limits:
        raise ValueError(
            f"swap limits must be finite and nonnegative: {invalid_swap_limits}"
        )
    if args.max_trial_swap_out_mb >= args.max_profile_swap_out_mb:
        raise ValueError(
            "max_trial_swap_out_mb must be smaller than max_profile_swap_out_mb"
        )
    if args.max_trial_swap_growth_mb >= args.max_profile_swap_growth_mb:
        raise ValueError(
            "max_trial_swap_growth_mb must be smaller than max_profile_swap_growth_mb"
        )
    if args.worker_engine:
        required = (
            args.worker_profile,
            args.worker_manifest,
            args.worker_result,
            args.worker_phase,
        )
        if not all(required):
            raise ValueError(
                "worker profile, manifest, result, and phase sidecar are required"
            )
        _worker_main(args)
    else:
        if not args.model_path:
            raise ValueError("--model-path is required")
        with _exclusive_benchmark_lock():
            with _controlled_termination():
                _parent_main(args)


if __name__ == "__main__":
    cli_main()
