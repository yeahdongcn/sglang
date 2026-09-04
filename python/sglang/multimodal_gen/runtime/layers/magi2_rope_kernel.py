# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.utils import direct_register_custom_op


@triton.jit
def _magi2_partial_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads,
    head_dim,
    rotary_dim,
    BLOCK_HALF: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    """Apply MAGI-2's tiled (non-interleaved) partial rotary embedding."""
    pid = tl.program_id(0)
    token = pid // num_heads
    head = pid - token * num_heads
    row = (token * num_heads + head) * head_dim
    half = rotary_dim // 2

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half
    x_first = tl.load(x_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)
    x_second = tl.load(x_ptr + row + half + offs, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + token * half + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * half + offs, mask=mask, other=0.0).to(tl.float32)
    first_out = x_first * cos - x_second * sin
    second_out = x_second * cos + x_first * sin
    tl.store(out_ptr + row + offs, first_out, mask=mask)
    tl.store(out_ptr + row + half + offs, second_out, mask=mask)

    tail = rotary_dim + tl.arange(0, BLOCK_TAIL)
    tail_mask = tail < head_dim
    tail_value = tl.load(x_ptr + row + tail, mask=tail_mask, other=0.0)
    tl.store(out_ptr + row + tail, tail_value, mask=tail_mask)


def _magi2_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected x [tokens, heads, head_dim], got {tuple(x.shape)}")
    if cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("cos and sin must have the same [tokens, rotary_half] shape")
    tokens, heads, head_dim = x.shape
    rotary_dim = cos.shape[-1] * 2
    if cos.shape[0] != tokens or rotary_dim > head_dim:
        raise ValueError(
            f"incompatible rope shapes x={tuple(x.shape)} cos={tuple(cos.shape)}"
        )
    out = torch.empty_like(x)
    block_half = triton.next_power_of_2(cos.shape[-1])
    block_tail = triton.next_power_of_2(max(1, head_dim - rotary_dim))
    _magi2_partial_rope_kernel[(tokens * heads,)](
        x,
        cos,
        sin,
        out,
        heads,
        head_dim,
        rotary_dim,
        BLOCK_HALF=block_half,
        BLOCK_TAIL=block_tail,
        num_warps=4,
    )
    return out


def _magi2_partial_rope_fake(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="magi2_partial_rope",
    op_func=_magi2_partial_rope,
    mutates_args=[],
    fake_impl=_magi2_partial_rope_fake,
)

magi2_partial_rope = torch.ops.sglang.magi2_partial_rope
