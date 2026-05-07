"""Paged attention wrapper for the MLX/Metal backend."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.paged_context import (
    PagedAttentionContext,
    get_paged_context,
)


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention for Metal-backed cached attention."""

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        paged_ctx = get_paged_context()
        if paged_ctx is not None:
            return self._paged_attention(x, paged_ctx, cache)
        return self._inner(x, mask=mask, cache=cache)

    def _paged_attention(
        self, x: mx.array, ctx: PagedAttentionContext, cache: Any = None
    ) -> mx.array:
        if x.ndim != 3:
            raise ValueError("MLX paged attention expects input shape (B, L, D)")
        if ctx.kv_pool is None:
            raise ValueError("MLX paged attention requires a KV pool")
        if ctx.block_tables is None:
            raise ValueError("MLX paged attention requires block tables")
        if ctx.context_lens is None:
            raise ValueError("MLX paged attention requires context lengths")
        if ctx.offsets is None:
            raise ValueError("MLX paged attention requires offsets")
        if ctx.is_prefill:
            if ctx.cu_seqlens is None:
                raise ValueError("MLX paged prefill requires cu_seqlens")
            if ctx.radix_prefix_lens is None:
                raise ValueError("MLX paged prefill requires radix prefix lengths")

        inner = self._inner
        layer_idx = self._layer_idx
        B, L, _ = x.shape
        if B != ctx.batch_size:
            raise ValueError("paged attention batch size must match context metadata")

        from sgl_kernel.metal import paged_kv_scatter

        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = queries.shape[-1] // inner.n_heads
        queries = queries.reshape(B, L, inner.n_heads, head_dim)
        keys = keys.reshape(B, L, inner.n_kv_heads, head_dim)
        values = values.reshape(B, L, inner.n_kv_heads, head_dim)

        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = inner.rope(queries, offset=ctx.offsets)
        keys = inner.rope(keys, offset=ctx.offsets)

        k_cache = ctx.kv_pool.k_buffer[layer_idx]
        v_cache = ctx.kv_pool.v_buffer[layer_idx]
        kv_dtype = k_cache.dtype
        if queries.dtype != kv_dtype:
            queries = queries.astype(kv_dtype)
        if keys.dtype != kv_dtype:
            keys = keys.astype(kv_dtype)
        if values.dtype != kv_dtype:
            values = values.astype(kv_dtype)
        if k_cache.ndim == 3:
            k_cache = k_cache.reshape(
                k_cache.shape[0], 1, k_cache.shape[1], k_cache.shape[2]
            )
            v_cache = v_cache.reshape(
                v_cache.shape[0], 1, v_cache.shape[1], v_cache.shape[2]
            )

        k_scatter = keys.transpose(0, 2, 1, 3).reshape(
            B * L, inner.n_kv_heads, head_dim
        )
        v_scatter = values.transpose(0, 2, 1, 3).reshape(
            B * L, inner.n_kv_heads, head_dim
        )
        paged_kv_scatter(k_scatter, v_scatter, k_cache, v_cache, ctx.slot_mapping)
        ctx.mark_kv_scattered(layer_idx)

        if ctx.is_prefill:
            from sgl_kernel.metal import prefill_attention_paged

            q_packed = mx.contiguous(
                queries.transpose(0, 2, 1, 3).reshape(B * L, inner.n_heads, head_dim)
            )
            k_packed = mx.contiguous(k_scatter)
            v_packed = mx.contiguous(v_scatter)
            output_packed = prefill_attention_paged(
                q_packed,
                k_packed,
                v_packed,
                k_cache,
                v_cache,
                ctx.block_tables,
                ctx.radix_prefix_lens,
                ctx.cu_seqlens,
                inner.scale,
                causal=True,
            )
            output = output_packed.reshape(B, L, inner.n_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        else:
            from sgl_kernel.metal import decode_attention_paged

            output = decode_attention_paged(
                queries,
                k_cache,
                v_cache,
                ctx.block_tables,
                ctx.context_lens,
                inner.scale,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return inner.o_proj(output)
