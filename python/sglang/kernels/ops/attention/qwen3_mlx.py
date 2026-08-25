"""Public semantic entry points for Qwen3 attention inside an MLX island.

Unlike the Torch-stream fused operators in :mod:`qwen3_mps`, this callable has
an MLX-array contract and intentionally has no Torch ``BaseFusedOp`` reference.
The module is still the public ``kernels.ops`` boundary: model/runtime code
must not import the private Metal implementation directly.
"""

from __future__ import annotations


def qwen3_radix_decode_deferred(*args, **kwargs):
    """Decode against Torch-owned Radix storage before deferred KV commit."""
    from sglang.kernels.ops.attention._qwen3_mlx_metal import (
        qwen3_radix_decode_deferred as implementation,
    )

    return implementation(*args, **kwargs)


def qwen3_qkv_prepare_deferred(*args, **kwargs):
    """Prepare dense Q/K/V inside MLX without committing the Torch KV pool."""
    from sglang.kernels.ops.attention._qwen3_qkv_mlx_metal import (
        qwen3_qkv_prepare_deferred as implementation,
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


def warmup_qwen3_qkv_prepare_deferred(epsilon: float = 1e-6) -> None:
    """Compile and resolve the fixed Qwen3-0.6B MLX QKV kernel."""
    from sglang.kernels.ops.attention._qwen3_qkv_mlx_metal import (
        warmup_qwen3_qkv_prepare_deferred as implementation,
    )

    implementation(epsilon)


__all__ = [
    "qwen3_qkv_prepare_deferred",
    "qwen3_radix_decode_deferred",
    "warmup_qwen3_qkv_prepare_deferred",
    "warmup_qwen3_radix_decode_deferred",
]
