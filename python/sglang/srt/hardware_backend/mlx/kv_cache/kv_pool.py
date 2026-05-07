"""MLX/Metal KV cache dtype helpers."""

from __future__ import annotations

import mlx.core as mx


def normalize_mlx_metal_dtype(dtype: mx.Dtype | None = None) -> mx.Dtype:
    if dtype is None:
        return mx.float16
    if dtype == mx.float16 or dtype == mx.float32:
        return dtype
    return mx.float16
