# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Torch fallbacks for packed grammar bitmask operations.

Sets or clears specific bits in an int32 bitmask by token ID.  The token list
is typically tiny (< 10 entries); aggregation is done in Python with the actual
bitmask operations using torch tensor indexing.
"""

import ctypes
from typing import List

import torch


def apply_token_bitmask_inplace_torch(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
) -> None:
    """Apply xgrammar's packed int32 bitmask with ordinary Torch operations.

    A set bit keeps a token and a cleared bit masks it to negative infinity.
    This path is intended for devices without the CUDA/Triton kernel, including
    MPS.  It performs no host readback and keeps the complete mask expansion on
    the logits device.
    """
    if bitmask.dtype != torch.int32:
        raise TypeError(f"bitmask must use torch.int32, found {bitmask.dtype}")
    if logits.ndim not in (1, 2) or bitmask.ndim not in (1, 2):
        raise ValueError(
            "grammar logits and bitmask must each have one or two dimensions; "
            f"found logits={tuple(logits.shape)}, bitmask={tuple(bitmask.shape)}"
        )
    if logits.device != bitmask.device:
        raise ValueError(
            "grammar logits and bitmask must be on the same device; "
            f"found logits={logits.device}, bitmask={bitmask.device}"
        )

    logits_2d = logits.unsqueeze(0) if logits.ndim == 1 else logits
    bitmask_2d = bitmask.unsqueeze(0) if bitmask.ndim == 1 else bitmask
    if logits_2d.shape[0] != bitmask_2d.shape[0]:
        raise ValueError(
            "grammar logits and bitmask batch sizes differ: "
            f"{logits_2d.shape[0]} vs {bitmask_2d.shape[0]}"
        )

    required_width = (int(logits_2d.shape[1]) + 31) // 32
    if int(bitmask_2d.shape[1]) > required_width:
        raise ValueError(
            "grammar bitmask is wider than the logits vocabulary: "
            f"{bitmask_2d.shape[1]} int32 words for {logits_2d.shape[1]} logits"
        )
    vocab_size = min(int(logits_2d.shape[1]), int(bitmask_2d.shape[1]) * 32)
    if vocab_size == 0:
        return

    token_indices = torch.arange(
        vocab_size,
        dtype=torch.int32,
        device=logits.device,
    )
    word_indices = torch.div(token_indices, 32, rounding_mode="floor").to(torch.long)
    bit_indices = torch.remainder(token_indices, 32)
    packed_words = bitmask_2d[:, word_indices]
    allowed = torch.bitwise_and(
        torch.bitwise_right_shift(packed_words, bit_indices),
        1,
    ).to(torch.bool)
    logits_2d[:, :vocab_size].masked_fill_(~allowed, float("-inf"))


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if reset_vocab_mask:
        vocab_mask[batch_idx].fill_(-1 if (not is_allowed) else 0)

    if not token_ids:
        return

    # Aggregate bit masks per int32 element to handle duplicate indices.
    aggregated: dict[int, int] = {}
    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        aggregated[element_idx] = aggregated.get(element_idx, 0) | (1 << bit_idx)

    row = vocab_mask[batch_idx]
    element_indices = torch.tensor(
        list(aggregated.keys()), dtype=torch.long, device=row.device
    )
    bitmasks = torch.tensor(
        [
            ctypes.c_int32(mask if is_allowed else ~mask).value
            for mask in aggregated.values()
        ],
        dtype=row.dtype,
        device=row.device,
    )

    if is_allowed:
        row[element_indices] = torch.bitwise_or(row[element_indices], bitmasks)
    else:
        row[element_indices] = torch.bitwise_and(row[element_indices], bitmasks)
