"""Torch MPS Metal JIT implementation of fused SiLU-and-mul.

The kernel deliberately has a small, explicit contract: contiguous MPS
``bfloat16`` input with rank two and an even last dimension.  The first half
of every row is the SiLU gate and the second half is multiplied with it.  The
wrapper validates the contract before launching so callers can keep a native
Torch fallback for arbitrary layouts and dtypes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

_THREADS_PER_GROUP = 256


def _metal_source() -> str:
    return r"""
#include <metal_stdlib>
using namespace metal;

kernel void silu_and_mul_bf16(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& intermediate [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= total_elements) {
    return;
  }

  const uint column = index % intermediate;
  const uint row = index / intermediate;
  const uint input_row_base = row * (2 * intermediate);
  const float gate = float(input[input_row_base + column]);
  const float value = float(input[input_row_base + intermediate + column]);
  const float sigmoid = 1.0f / (1.0f + metal::fast::exp(-gate));
  output[index] = bfloat(gate * sigmoid * value);
}
"""


@lru_cache(maxsize=1)
def _compile_silu_and_mul_library():
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if not callable(compile_shader):
        raise RuntimeError(
            "MPS fused SiLU-and-mul requires torch.mps.compile_shader from Torch 2.13"
        )
    return compile_shader(_metal_source())


def warmup_silu_and_mul_metal_kernel() -> None:
    """Compile and resolve the specialized entry point before serving traffic."""

    _compile_silu_and_mul_library().silu_and_mul_bf16


def _require_contract(input: torch.Tensor, out: torch.Tensor) -> tuple[int, int]:
    if not isinstance(input, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError("MPS fused SiLU-and-mul expects Torch tensors")
    if input.device.type != "mps" or out.device.type != "mps":
        raise RuntimeError(
            "MPS fused SiLU-and-mul expects MPS tensors; "
            f"got input={input.device}, out={out.device}"
        )
    if input.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise RuntimeError(
            "MPS fused SiLU-and-mul expects bfloat16 tensors; "
            f"got input={input.dtype}, out={out.dtype}"
        )
    if input.ndim != 2 or not input.is_contiguous():
        raise RuntimeError(
            "MPS fused SiLU-and-mul expects a contiguous 2-D input; "
            f"got shape={tuple(input.shape)}, strides={input.stride()}"
        )
    if input.shape[1] == 0 or input.shape[1] % 2:
        raise RuntimeError(
            "MPS fused SiLU-and-mul expects an even, non-zero input width; "
            f"got shape={tuple(input.shape)}"
        )
    rows, intermediate = (int(input.shape[0]), int(input.shape[1]) // 2)
    if tuple(out.shape) != (rows, intermediate) or not out.is_contiguous():
        raise RuntimeError(
            "MPS fused SiLU-and-mul expects a contiguous output shaped "
            f"[{rows}, {intermediate}]; got shape={tuple(out.shape)}, "
            f"strides={out.stride()}"
        )
    return rows, intermediate


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Run fused SiLU-and-mul on the Torch MPS stream.

    ``input`` has shape ``[rows, 2 * intermediate]`` and ``out`` has shape
    ``[rows, intermediate]``.  No synchronization or implicit copies are
    performed here; the caller owns the normal MPS stream boundary.
    """

    if out is None:
        if not isinstance(input, torch.Tensor):
            raise TypeError("MPS fused SiLU-and-mul expects a Torch tensor")
        if input.ndim != 2 or input.shape[1] == 0 or input.shape[1] % 2:
            raise RuntimeError(
                "MPS fused SiLU-and-mul requires a 2-D input with an even width"
            )
        out = torch.empty(
            (input.shape[0], input.shape[1] // 2),
            dtype=input.dtype,
            device=input.device,
        )

    rows, intermediate = _require_contract(input, out)
    if rows == 0:
        return out

    kernel = _compile_silu_and_mul_library().silu_and_mul_bf16
    kernel(
        input,
        out,
        int(intermediate),
        int(rows * intermediate),
        threads=(rows * intermediate,),
        group_size=(_THREADS_PER_GROUP,),
    )
    return out


__all__ = [
    "silu_and_mul",
    "warmup_silu_and_mul_metal_kernel",
]
