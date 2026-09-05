# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.utils import direct_register_custom_op


@triton.jit
def _magi2_swiglu7_kernel(
    input_ptr,
    output_ptr,
    num_outputs,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_outputs
    gate = tl.load(input_ptr + 2 * offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + 2 * offsets + 1, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, 7.0)
    up = tl.maximum(tl.minimum(up, 7.0), -7.0)
    output = gate * tl.sigmoid(1.702 * gate) * (up + 1.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def _magi2_swiglu7(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"expected an even last dimension, got {x.shape[-1]}")
    output = torch.empty(
        *x.shape[:-1], x.shape[-1] // 2, device=x.device, dtype=x.dtype
    )
    _magi2_swiglu7_kernel[(triton.cdiv(output.numel(), 256),)](
        x,
        output,
        output.numel(),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


def _magi2_swiglu7_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(*x.shape[:-1], x.shape[-1] // 2, device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="magi2_swiglu7",
    op_func=_magi2_swiglu7,
    mutates_args=[],
    fake_impl=_magi2_swiglu7_fake,
)

magi2_swiglu7 = torch.ops.sglang.magi2_swiglu7
