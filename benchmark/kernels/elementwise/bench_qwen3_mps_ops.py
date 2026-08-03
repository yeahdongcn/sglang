"""Low-memory A/B benchmark for Qwen3-0.6B MPS elementwise providers.

The benchmark keeps the exact production tensor contracts and synchronizes
the MPS device around every repeated block.  Provider order is rotated in
both directions so no implementation is always measured in the same thermal
or command-buffer position.

Run::

    python benchmark/kernels/elementwise/bench_qwen3_mps_ops.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation._silu_and_mul_metal_jit import (
    silu_and_mul as metal_silu_and_mul,
)
from sglang.kernels.ops.activation._silu_and_mul_metal_jit import (
    warmup_silu_and_mul_metal_kernel,
)
from sglang.kernels.ops.layernorm._rmsnorm_metal_jit import (
    mps_fused_add_rmsnorm,
    mps_rmsnorm,
    warmup_mps_rmsnorm_kernels,
)

_HIDDEN_SIZE = 1024
_INTERMEDIATE_SIZE = 3072
_EPSILON = 1e-6


def _ordered_labels(labels: tuple[str, ...], round_index: int) -> tuple[str, ...]:
    direction = (
        labels if (round_index // len(labels)) % 2 == 0 else tuple(reversed(labels))
    )
    offset = round_index % len(direction)
    return direction[offset:] + direction[:offset]


def _time_calls(
    calls: dict[str, Callable[[], None]], *, repeats: int, rounds: int
) -> dict[str, list[float]]:
    labels = tuple(calls)
    timings = {label: [] for label in labels}
    for round_index in range(rounds):
        for label in _ordered_labels(labels, round_index):
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            for _ in range(repeats):
                calls[label]()
            torch.mps.synchronize()
            timings[label].append((time.perf_counter_ns() - start) / repeats / 1_000)
    return timings


def _print_timings(label: str, timings: dict[str, list[float]]) -> None:
    print(label)
    baseline = statistics.median(timings["torch_semantic"])
    for provider, values in timings.items():
        median = statistics.median(values)
        print(
            f"  {provider:16s} median={median:8.2f} us "
            f"best={min(values):8.2f} us worst={max(values):8.2f} us "
            f"vs_semantic={baseline / median:6.2f}x"
        )


def _torch_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    value = input.to(torch.float32)
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    value = value * torch.rsqrt(variance + eps)
    return (value * weight).to(input.dtype)


def _torch_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    value = input.to(torch.float32) + residual.to(torch.float32)
    residual.copy_(value.to(residual.dtype))
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    input.copy_((normalized * weight).to(input.dtype))


def _benchmark_rmsnorm(rows: int, *, repeats: int, rounds: int) -> None:
    input = torch.randn(rows, _HIDDEN_SIZE, device="mps", dtype=torch.bfloat16)
    weight = torch.randn(_HIDDEN_SIZE, device="mps", dtype=torch.bfloat16)
    expected = _torch_rmsnorm(input, weight, _EPSILON)
    metal = mps_rmsnorm(input, weight, _EPSILON)
    torch_fused = F.rms_norm(input, (_HIDDEN_SIZE,), weight, _EPSILON)
    torch.mps.synchronize()
    torch.testing.assert_close(metal.cpu(), expected.cpu(), atol=0.03125, rtol=0.01)
    torch.testing.assert_close(
        torch_fused.cpu(), expected.cpu(), atol=0.03125, rtol=0.01
    )

    calls = {
        "torch_semantic": lambda: _torch_rmsnorm(input, weight, _EPSILON),
        "torch_F": lambda: F.rms_norm(input, (_HIDDEN_SIZE,), weight, _EPSILON),
        "metal_jit": lambda: mps_rmsnorm(input, weight, _EPSILON),
    }
    for call in calls.values():
        call()
    _print_timings(
        f"RMSNorm hidden=1024, rows={rows}",
        _time_calls(calls, repeats=repeats, rounds=rounds),
    )

    semantic_out = torch.empty_like(input)
    torch_f_out = torch.empty_like(input)
    metal_out = torch.empty_like(input)
    out_calls = {
        "torch_semantic": lambda: semantic_out.copy_(
            _torch_rmsnorm(input, weight, _EPSILON)
        ),
        "torch_F_copy": lambda: torch_f_out.copy_(
            F.rms_norm(input, (_HIDDEN_SIZE,), weight, _EPSILON)
        ),
        "metal_direct": lambda: mps_rmsnorm(input, weight, _EPSILON, out=metal_out),
    }
    for call in out_calls.values():
        call()
    torch.mps.synchronize()
    torch.testing.assert_close(
        torch_f_out.cpu(), expected.cpu(), atol=0.03125, rtol=0.01
    )
    torch.testing.assert_close(metal_out.cpu(), expected.cpu(), atol=0.03125, rtol=0.01)
    _print_timings(
        f"RMSNorm caller-owned out hidden=1024, rows={rows}",
        _time_calls(out_calls, repeats=repeats, rounds=rounds),
    )


def _benchmark_fused_add_rmsnorm(rows: int, *, repeats: int, rounds: int) -> None:
    # Zero inputs stay stable across repeated in-place calls, while the device
    # still executes the same loads, reductions, stores, and kernel launches.
    torch_input = torch.zeros(rows, _HIDDEN_SIZE, device="mps", dtype=torch.bfloat16)
    torch_residual = torch.zeros_like(torch_input)
    metal_input = torch.zeros_like(torch_input)
    metal_residual = torch.zeros_like(torch_input)
    weight = torch.randn(_HIDDEN_SIZE, device="mps", dtype=torch.bfloat16)
    calls = {
        "torch_semantic": lambda: _torch_fused_add_rmsnorm(
            torch_input, torch_residual, weight, _EPSILON
        ),
        "metal_jit": lambda: mps_fused_add_rmsnorm(
            metal_input, metal_residual, weight, _EPSILON
        ),
    }
    for call in calls.values():
        call()
    _print_timings(
        f"Fused add + RMSNorm hidden=1024, rows={rows}",
        _time_calls(calls, repeats=repeats, rounds=rounds),
    )


def _benchmark_silu_and_mul(rows: int, *, repeats: int, rounds: int) -> None:
    input = torch.randn(
        rows,
        2 * _INTERMEDIATE_SIZE,
        device="mps",
        dtype=torch.bfloat16,
    )

    def torch_semantic() -> torch.Tensor:
        gate, value = input.chunk(2, dim=-1)
        return F.silu(gate) * value

    expected = torch_semantic()
    metal = metal_silu_and_mul(input)
    torch.mps.synchronize()
    torch.testing.assert_close(metal.cpu(), expected.cpu(), atol=0.03125, rtol=0.01)
    calls = {
        "torch_semantic": torch_semantic,
        "metal_jit": lambda: metal_silu_and_mul(input),
    }
    for call in calls.values():
        call()
    _print_timings(
        f"SiLU and multiply intermediate=3072, rows={rows}",
        _time_calls(calls, repeats=repeats, rounds=rounds),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 8, 64, 512))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=11)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS is required")
    torch.manual_seed(2026)
    torch.empty(1, device="mps")
    warmup_mps_rmsnorm_kernels()
    warmup_silu_and_mul_metal_kernel()
    torch.mps.synchronize()
    with torch.inference_mode():
        for rows in args.rows:
            _benchmark_rmsnorm(rows, repeats=args.repeats, rounds=args.rounds)
            _benchmark_fused_add_rmsnorm(rows, repeats=args.repeats, rounds=args.rounds)
            _benchmark_silu_and_mul(rows, repeats=args.repeats, rounds=args.rounds)


if __name__ == "__main__":
    main()
