"""Torch-owned Metal JIT kernels for Qwen3-0.6B RMSNorm.

The initial contract is deliberately narrow: inference-only, contiguous bf16
2-D tensors with hidden size 1024.  Keeping the contract explicit lets the
caller fall back to the native Torch implementation without inserting hidden
layout or dtype conversions on the decode hot path.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch

_HIDDEN_SIZE = 1024
_NUM_THREADS = 256
_SIMD_SIZE = 32
_NUM_SIMDGROUPS = _NUM_THREADS // _SIMD_SIZE
_VALUES_PER_THREAD = _HIDDEN_SIZE // _NUM_THREADS

_METAL_SOURCE = f"""
#include <metal_stdlib>
using namespace metal;

#define HIDDEN_SIZE {_HIDDEN_SIZE}
#define NUM_THREADS {_NUM_THREADS}
#define SIMD_SIZE {_SIMD_SIZE}
#define NUM_SIMDGROUPS {_NUM_SIMDGROUPS}
#define VALUES_PER_THREAD {_VALUES_PER_THREAD}

inline float rmsnorm_simd_sum(float value) {{
  value += simd_shuffle_xor(value, ushort(16));
  value += simd_shuffle_xor(value, ushort(8));
  value += simd_shuffle_xor(value, ushort(4));
  value += simd_shuffle_xor(value, ushort(2));
  return value + simd_shuffle_xor(value, ushort(1));
}}

kernel void rmsnorm_1024_bf16(
    const device bfloat* input [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {{
  threadgroup float simdgroup_sums[NUM_SIMDGROUPS];
  threadgroup float inverse_rms;

  const ulong row_base = ulong(group.y) * HIDDEN_SIZE;
  const uint dim = tid * VALUES_PER_THREAD;
  float values[VALUES_PER_THREAD];
  float square_sum = 0.0f;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {{
    const float value = float(input[row_base + dim + index]);
    values[index] = value;
    square_sum += value * value;
  }}

  const float simdgroup_sum = rmsnorm_simd_sum(square_sum);
  if (lane == 0) {{
    simdgroup_sums[simdgroup] = simdgroup_sum;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup == 0) {{
    const float partial = lane < NUM_SIMDGROUPS ? simdgroup_sums[lane] : 0.0f;
    const float total = rmsnorm_simd_sum(partial);
    if (lane == 0) {{
      inverse_rms = metal::precise::rsqrt(
          total / float(HIDDEN_SIZE) + epsilon);
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {{
    const ulong offset = row_base + dim + index;
    output[offset] = bfloat(
        values[index] * inverse_rms * float(weight[dim + index]));
  }}
}}

kernel void fused_add_rmsnorm_1024_bf16(
    device bfloat* input [[buffer(0)]],
    device bfloat* residual [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {{
  threadgroup float simdgroup_sums[NUM_SIMDGROUPS];
  threadgroup float inverse_rms;

  const ulong row_base = ulong(group.y) * HIDDEN_SIZE;
  const uint dim = tid * VALUES_PER_THREAD;
  float values[VALUES_PER_THREAD];
  float square_sum = 0.0f;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {{
    const ulong offset = row_base + dim + index;
    // Preserve the fp32 sum in registers for RMSNorm.  The residual output is
    // narrowed independently, matching RMSNorm.forward_native.
    const float value = float(input[offset]) + float(residual[offset]);
    values[index] = value;
    square_sum += value * value;
    residual[offset] = bfloat(value);
  }}

  const float simdgroup_sum = rmsnorm_simd_sum(square_sum);
  if (lane == 0) {{
    simdgroup_sums[simdgroup] = simdgroup_sum;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup == 0) {{
    const float partial = lane < NUM_SIMDGROUPS ? simdgroup_sums[lane] : 0.0f;
    const float total = rmsnorm_simd_sum(partial);
    if (lane == 0) {{
      inverse_rms = metal::precise::rsqrt(
          total / float(HIDDEN_SIZE) + epsilon);
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {{
    const ulong offset = row_base + dim + index;
    input[offset] = bfloat(
        values[index] * inverse_rms * float(weight[dim + index]));
  }}
}}
"""


@lru_cache(maxsize=1)
def _compile_rmsnorm_library():
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if not callable(compile_shader):
        raise RuntimeError(
            "MPS RMSNorm Metal JIT requires torch.mps.compile_shader from Torch 2.13"
        )
    return compile_shader(_METAL_SOURCE)


def _is_mps_bf16_contiguous(tensor: object) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "mps"
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
    )


def can_use_mps_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Return whether the specialized non-residual kernel owns this call."""
    if not (_is_mps_bf16_contiguous(input) and _is_mps_bf16_contiguous(weight)):
        return False
    if input.ndim != 2 or tuple(input.shape)[1:] != (_HIDDEN_SIZE,):
        return False
    if tuple(weight.shape) != (_HIDDEN_SIZE,) or input.shape[0] == 0:
        return False
    if not math.isfinite(eps) or eps <= 0:
        return False
    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        return False
    if out is not None:
        if not _is_mps_bf16_contiguous(out) or tuple(out.shape) != tuple(input.shape):
            return False
        if torch.is_grad_enabled() and out.requires_grad:
            return False
    return callable(getattr(torch.mps, "compile_shader", None))


def can_use_mps_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> bool:
    """Return whether the specialized in-place residual kernel owns this call."""
    if not can_use_mps_rmsnorm(input, weight, eps):
        return False
    if not _is_mps_bf16_contiguous(residual):
        return False
    if tuple(residual.shape) != tuple(input.shape):
        return False
    if torch.is_grad_enabled() and residual.requires_grad:
        return False
    # Two distinct outputs cannot be represented when both arguments alias.
    return residual.data_ptr() != input.data_ptr()


def mps_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run bf16 RMSNorm on the current Torch MPS stream."""
    if not can_use_mps_rmsnorm(input, weight, eps, out=out):
        raise RuntimeError(
            "MPS RMSNorm requires contiguous bf16 [rows, 1024] input/output "
            "and contiguous bf16 [1024] weight"
        )
    if out is None:
        out = torch.empty_like(input)
    kernel = _compile_rmsnorm_library().rmsnorm_1024_bf16
    kernel(
        input,
        weight,
        out,
        float(eps),
        threads=(_NUM_THREADS, int(input.shape[0]), 1),
        group_size=(_NUM_THREADS, 1, 1),
    )
    return out


def mps_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """In-place ``residual += input; input = RMSNorm(residual) * weight``."""
    if not can_use_mps_fused_add_rmsnorm(input, residual, weight, eps):
        raise RuntimeError(
            "MPS fused-add RMSNorm requires distinct contiguous bf16 "
            "[rows, 1024] input/residual and contiguous bf16 [1024] weight"
        )
    kernel = _compile_rmsnorm_library().fused_add_rmsnorm_1024_bf16
    kernel(
        input,
        residual,
        weight,
        float(eps),
        threads=(_NUM_THREADS, int(input.shape[0]), 1),
        group_size=(_NUM_THREADS, 1, 1),
    )


def warmup_mps_rmsnorm_kernels(
    *, rmsnorm: bool = True, fused_add_rmsnorm: bool = True
) -> None:
    """Compile and resolve selected RMSNorm pipelines without launching work."""
    if not rmsnorm and not fused_add_rmsnorm:
        return
    library = _compile_rmsnorm_library()
    if rmsnorm:
        library.rmsnorm_1024_bf16
    if fused_add_rmsnorm:
        library.fused_add_rmsnorm_1024_bf16


__all__ = [
    "can_use_mps_fused_add_rmsnorm",
    "can_use_mps_rmsnorm",
    "mps_fused_add_rmsnorm",
    "mps_rmsnorm",
    "warmup_mps_rmsnorm_kernels",
]
