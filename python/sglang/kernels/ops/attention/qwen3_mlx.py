"""Public semantic entry points for Qwen3 attention inside an MLX island.

Unlike the Torch-stream fused operators in :mod:`qwen3_mps`, these callables
have an MLX-array contract and intentionally do not use ``BaseFusedOp``. The
public wrapper keeps model/runtime code independent of the private Metal
implementation and imports MLX only when the primitive is selected.
"""

from __future__ import annotations


def qwen3_radix_decode_deferred(*args, **kwargs):
    """Decode against Torch-owned Radix storage before deferred KV commit."""
    from sglang.kernels.ops.attention._qwen3_mlx_metal import (
        qwen3_radix_decode_deferred as implementation,
    )

    return implementation(*args, **kwargs)


def warmup_qwen3_radix_decode_deferred(
    *,
    request_rows: int = 1,
    table_stride: int = 1,
    pool_slots: int = 1,
) -> None:
    """Compile and resolve the fixed Qwen3-0.6B MLX decode kernel."""
    from sglang.kernels.ops.attention._qwen3_mlx_metal import (
        warmup_qwen3_radix_decode_deferred as implementation,
    )

    implementation(
        request_rows=request_rows,
        table_stride=table_stride,
        pool_slots=pool_slots,
    )


__all__ = [
    "qwen3_radix_decode_deferred",
    "warmup_qwen3_radix_decode_deferred",
]
