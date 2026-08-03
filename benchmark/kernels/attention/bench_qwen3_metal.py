"""A/B benchmark for Qwen3-0.6B Torch MPS Metal implementations.

This is a model-free, low-memory benchmark. It compares the packaged AOT
metallib with the same canonical MSL compiled through Torch's Metal JIT. Add
``--include-torch`` to include the correctness reference; no threshold is
enforced because MPS timing varies with thermals and system load.

Run::

    python benchmark/kernels/attention/bench_qwen3_metal.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from sglang.kernels.ops.attention.qwen3_mps import (
    QWEN3_06B_METAL_SPEC,
    is_qwen3_metal_aot_available,
    qwen3_qknorm_rope_store,
    qwen3_radix_decode,
    warmup_qwen3_mps_kernels,
)
from sglang.kernels.spec import KernelBackend


def _time_rounds(
    call: Callable[[KernelBackend], None],
    backends: Sequence[KernelBackend],
    *,
    repeats: int,
    rounds: int,
) -> dict[KernelBackend, list[float]]:
    timings = {backend: [] for backend in backends}
    for round_index in range(rounds):
        # Rotate both the forward and reverse orders.  With three providers a
        # simple forward/reverse alternation leaves the middle provider in the
        # middle for every round, making it vulnerable to systematic thermal
        # or command-buffer-position bias.
        direction = (
            tuple(backends)
            if (round_index // len(backends)) % 2 == 0
            else tuple(reversed(backends))
        )
        offset = round_index % len(direction)
        order = direction[offset:] + direction[:offset]
        for backend in order:
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            for _ in range(repeats):
                call(backend)
            torch.mps.synchronize()
            timings[backend].append((time.perf_counter_ns() - start) / repeats / 1_000)
    return timings


def _print_timings(label: str, timings: dict[KernelBackend, list[float]]) -> None:
    print(label)
    for backend, values in timings.items():
        print(
            f"  {backend.value:9s} median={statistics.median(values):8.2f} us "
            f"best={min(values):8.2f} us worst={max(values):8.2f} us"
        )


def _make_rope_cache(max_position: int) -> torch.Tensor:
    spec = QWEN3_06B_METAL_SPEC
    inverse_frequency = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, spec.head_dim, 2, dtype=torch.float32) / spec.head_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32)
    frequency = torch.einsum("i,j->ij", positions, inverse_frequency)
    return torch.cat((frequency.cos(), frequency.sin()), dim=-1).to(
        device="mps", dtype=torch.bfloat16
    )


def _benchmark_qkv(
    num_tokens: int,
    backends: Sequence[KernelBackend],
    *,
    repeats: int,
    rounds: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    spec = QWEN3_06B_METAL_SPEC
    qkv = torch.randn(
        num_tokens,
        spec.qkv_width,
        device="mps",
        dtype=torch.bfloat16,
    )
    q_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
    k_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device="mps", dtype=torch.int64)
    slots = torch.arange(num_tokens, device="mps", dtype=torch.int64)
    pool_size = max(num_tokens + 16, 128)
    k_pool = torch.empty(
        pool_size,
        spec.num_kv_heads,
        spec.head_dim,
        device="mps",
        dtype=torch.bfloat16,
    )
    v_pool = torch.empty_like(k_pool)
    q_out = torch.empty(
        num_tokens,
        spec.num_q_heads,
        spec.head_dim,
        device="mps",
        dtype=torch.bfloat16,
    )

    def call(backend: KernelBackend) -> None:
        qwen3_qknorm_rope_store(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            slots,
            q_out,
            k_pool,
            v_pool,
            epsilon=1e-6,
            backend=backend,
        )

    for backend in backends:
        call(backend)
    _print_timings(
        f"QK norm + RoPE + KV store, tokens={num_tokens}",
        _time_rounds(call, backends, repeats=repeats, rounds=rounds),
    )


def _benchmark_decode(
    sequence_length: int,
    backends: Sequence[KernelBackend],
    *,
    repeats: int,
    rounds: int,
) -> None:
    spec = QWEN3_06B_METAL_SPEC
    pool_size = sequence_length + 256
    table_stride = sequence_length + 128
    q = torch.randn(
        1,
        spec.num_q_heads,
        spec.head_dim,
        device="mps",
        dtype=torch.bfloat16,
    )
    k_pool = torch.randn(
        pool_size,
        spec.num_kv_heads,
        spec.head_dim,
        device="mps",
        dtype=torch.bfloat16,
    )
    v_pool = torch.randn_like(k_pool)
    req_to_token = torch.zeros(2, table_stride, device="mps", dtype=torch.int32)
    req_to_token[1, :sequence_length] = torch.randperm(pool_size, device="mps")[
        :sequence_length
    ].to(torch.int32)
    req_pool_indices = torch.ones(1, device="mps", dtype=torch.int64)
    seq_lens = torch.full((1,), sequence_length, device="mps", dtype=torch.int64)
    out = torch.empty_like(q)

    def call(backend: KernelBackend) -> None:
        qwen3_radix_decode(
            q,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            out,
            scale=spec.attention_scale,
            backend=backend,
        )

    for backend in backends:
        call(backend)
    _print_timings(
        f"Radix decode, batch=1, sequence_length={sequence_length}",
        _time_rounds(call, backends, repeats=repeats, rounds=rounds),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--qkv-repeats", type=int, default=100)
    parser.add_argument("--decode-repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--include-torch", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS is required")
    if not is_qwen3_metal_aot_available():
        raise SystemExit(
            "packaged Metal AOT is required; build and install setup_metal.py"
        )

    torch.manual_seed(2026)
    torch.empty(1, device="mps")
    torch.mps.synchronize()
    cold = {}
    for backend in (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT):
        start = time.perf_counter_ns()
        # This benchmark compares one provider end to end.  Use the explicit
        # compatibility argument so QKV and decode cannot accidentally warm
        # different providers when their independent defaults change.
        warmup_qwen3_mps_kernels(backend=backend)
        torch.mps.synchronize()
        cold[backend] = (time.perf_counter_ns() - start) / 1_000_000
    print(
        "Process-local warmup after MPS initialization: "
        + ", ".join(
            f"{backend.value}={milliseconds:.3f} ms"
            for backend, milliseconds in cold.items()
        )
    )

    backends = [KernelBackend.METAL_AOT, KernelBackend.METAL_JIT]
    if args.include_torch:
        backends.append(KernelBackend.TORCH)
    backends = tuple(backends)
    cos_sin_cache = _make_rope_cache(max(max(args.tokens) + 1, 2048))
    for num_tokens in args.tokens:
        _benchmark_qkv(
            num_tokens,
            backends,
            repeats=args.qkv_repeats,
            rounds=args.rounds,
            cos_sin_cache=cos_sin_cache,
        )
    for sequence_length in args.sequence_lengths:
        _benchmark_decode(
            sequence_length,
            backends,
            repeats=args.decode_repeats,
            rounds=args.rounds,
        )


if __name__ == "__main__":
    main()
