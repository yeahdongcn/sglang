"""Paged attention wrapper for the MLX/Metal backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache
from sglang.srt.hardware_backend.mlx.kv_cache.paged_context import (
    PagedAttentionContext,
    get_paged_context,
)

_thread_local = threading.local()


@dataclass
class BatchedDecodeContext:
    """Context set before batched contiguous-cache decode."""

    batch_size: int
    seq_lens: list[int]
    layer_caches: list[list[ContiguousKVCache]]
    offsets: mx.array = field(init=False)
    max_len: int = field(init=False)
    valid_lens: mx.array = field(init=False)
    needs_padding: bool = field(init=False)
    pad_sizes: list[int] = field(init=False)
    positions: Optional[mx.array] = field(init=False)

    def __post_init__(self) -> None:
        seq_lens = self.seq_lens
        max_seq_len = max(seq_lens)
        if len(seq_lens) == 1:
            self.offsets = None
            self.max_len = max_seq_len + 1
            self.valid_lens = None
            self.needs_padding = False
            self.pad_sizes = [0]
            self.positions = None
            return
        self.offsets = mx.array(seq_lens, dtype=mx.int32)
        self.max_len = max_seq_len + 1
        self.valid_lens = self.offsets + 1
        self.needs_padding = min(seq_lens) < max_seq_len
        self.pad_sizes = [max_seq_len - seq_len for seq_len in seq_lens]
        self.positions = mx.arange(self.max_len) if self.needs_padding else None


def set_context(ctx: BatchedDecodeContext | None) -> None:
    _thread_local.batched_ctx = ctx


def get_context() -> BatchedDecodeContext | None:
    return getattr(_thread_local, "batched_ctx", None)


def clear_context() -> None:
    _thread_local.batched_ctx = None


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention for Metal-backed cached attention."""

    def __init__(self, inner: nn.Module, layer_idx: int, *, enable_paged: bool = True):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)
        object.__setattr__(self, "_enable_paged", enable_paged)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        if self._enable_paged:
            paged_ctx = get_paged_context()
            if paged_ctx is not None:
                return self._paged_attention(x, paged_ctx, mask=mask, cache=cache)
        batched_ctx = get_context()
        if batched_ctx is not None:
            return self._batched_decode(x, batched_ctx)
        return self._inner(x, mask=mask, cache=cache)

    def _batched_decode(self, x: mx.array, ctx: BatchedDecodeContext) -> mx.array:
        inner = self._inner
        layer_idx = self._layer_idx
        B = ctx.batch_size

        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = queries.shape[-1] // inner.n_heads
        queries = queries.reshape(B, 1, inner.n_heads, head_dim)
        keys = keys.reshape(B, 1, inner.n_kv_heads, head_dim)
        values = values.reshape(B, 1, inner.n_kv_heads, head_dim)

        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if B == 1:
            offset = int(ctx.seq_lens[0])
            queries = inner.rope(queries, offset=offset)
            keys = inner.rope(keys, offset=offset)

            layer_cache = ctx.layer_caches[layer_idx][0]
            layer_cache.write_token(keys, values)
            keys_b, values_b = layer_cache.get_kv()

            output = mx.fast.scaled_dot_product_attention(
                queries, keys_b, values_b, scale=inner.scale, mask=None
            )
            output = output.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            return inner.o_proj(output)

        queries = inner.rope(queries, offset=ctx.offsets)
        keys = inner.rope(keys, offset=ctx.offsets)

        layer_caches = ctx.layer_caches[layer_idx]
        all_k = []
        all_v = []

        for i in range(B):
            layer_caches[i].write_token(keys[i : i + 1], values[i : i + 1])
            k_all, v_all = layer_caches[i].get_kv()

            pad = ctx.pad_sizes[i]
            if pad > 0:
                k_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=k_all.dtype
                )
                v_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=v_all.dtype
                )
                k_all = mx.concatenate([k_all, k_pad], axis=2)
                v_all = mx.concatenate([v_all, v_pad], axis=2)

            all_k.append(k_all)
            all_v.append(v_all)

        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        attn_mask = None
        if ctx.needs_padding:
            mask_bool = ctx.positions[None, :] >= ctx.valid_lens[:, None]
            attn_mask = mx.where(
                mask_bool[:, None, None, :],
                mx.array(mx.finfo(queries.dtype).min, dtype=queries.dtype),
                mx.array(0.0, dtype=queries.dtype),
            )

        output = mx.fast.scaled_dot_product_attention(
            queries, keys_b, values_b, scale=inner.scale, mask=attn_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return inner.o_proj(output)

    def _paged_attention(
        self,
        x: mx.array,
        ctx: PagedAttentionContext,
        mask: Any = None,
        cache: Any = None,
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

        queries = _apply_rope_with_offsets(inner.rope, queries, ctx.offset_values)
        keys = _apply_rope_with_offsets(inner.rope, keys, ctx.offset_values)

        k_cache = ctx.kv_pool.k_buffer[layer_idx]
        v_cache = ctx.kv_pool.v_buffer[layer_idx]
        kv_dtype = k_cache.dtype
        if queries.dtype != kv_dtype:
            queries = queries.astype(kv_dtype)
        if keys.dtype != kv_dtype:
            keys = keys.astype(kv_dtype)
        if values.dtype != kv_dtype:
            values = values.astype(kv_dtype)
        k_scatter = keys.transpose(0, 2, 1, 3).reshape(
            B * L, inner.n_kv_heads, head_dim
        )
        v_scatter = values.transpose(0, 2, 1, 3).reshape(
            B * L, inner.n_kv_heads, head_dim
        )
        if kv_dtype in (mx.float16, mx.float32):
            from sgl_kernel.metal import paged_kv_scatter

            paged_kv_scatter(k_scatter, v_scatter, k_cache, v_cache, ctx.slot_mapping)
        else:
            ctx.kv_pool.set_kv(layer_idx, ctx.slot_mapping, k_scatter, v_scatter)
        ctx.mark_kv_scattered(layer_idx)

        if ctx.is_prefill:
            if not ctx.has_radix_prefix:
                from mlx_lm.models.base import scaled_dot_product_attention

                dense_mask = "causal" if mask is None and L > 1 else mask
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=inner.scale,
                    mask=dense_mask,
                )
            else:
                if kv_dtype in (mx.float16, mx.float32):
                    from sgl_kernel.metal import prefill_attention_paged

                    q_packed = mx.contiguous(
                        queries.transpose(0, 2, 1, 3).reshape(
                            B * L, inner.n_heads, head_dim
                        )
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
                    output = output_packed.reshape(
                        B, L, inner.n_heads, head_dim
                    ).transpose(0, 2, 1, 3)
                else:
                    output = _prefill_attention_p1_mlx_fast(
                        queries,
                        k_cache,
                        v_cache,
                        ctx,
                        inner.scale,
                    )
        else:
            output = _decode_attention_mlx_fast_paged(
                queries,
                k_cache,
                v_cache,
                ctx,
                inner.scale,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return inner.o_proj(output)


class MLXBatchedAttentionWrapper(MLXAttentionWrapper):
    """Wrapper variant for contiguous-cache runtime paths without paged dispatch."""

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__(inner, layer_idx, enable_paged=False)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        return self._batched_decode(x, ctx)


def _apply_rope_with_offsets(rope: Any, x: mx.array, offsets: Any) -> mx.array:
    offset_values = offsets
    if not isinstance(offset_values, list):
        return rope(x, offset=int(offset_values))
    if len(offset_values) == 1 or len(set(offset_values)) == 1:
        return rope(x, offset=int(offset_values[0]))
    rows = [
        rope(x[i : i + 1], offset=int(offset)) for i, offset in enumerate(offset_values)
    ]
    return mx.concatenate(rows, axis=0)


def _decode_attention_mlx_fast_paged(
    queries: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    ctx: PagedAttentionContext,
    scale: float,
) -> mx.array:
    if k_cache.ndim not in (3, 4) or v_cache.ndim != k_cache.ndim:
        raise ValueError("MLX paged decode requires 4-D paged or p1 3-D KV caches")
    if k_cache.shape != v_cache.shape:
        raise ValueError("MLX paged decode requires matching K/V cache shapes")

    batch = queries.shape[0]
    if k_cache.ndim == 4:
        block_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
    else:
        block_size = 1
        num_kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[2]
    max_blocks = ctx.block_tables.shape[1]
    max_tokens = max_blocks * block_size
    if max_tokens <= 0:
        raise ValueError("MLX paged decode requires non-empty block tables")
    if ctx.context_len_values and max(ctx.context_len_values) > max_tokens:
        raise ValueError("MLX paged decode context lengths exceed block tables")

    if (
        queries.shape[2] == 1
        and queries.dtype == mx.float16
        and k_cache.dtype == mx.float16
        and v_cache.dtype == mx.float16
        and block_size == 16
        and head_dim == 128
        and (batch > 1 or max(ctx.context_len_values or [max_tokens]) > block_size)
    ):
        from sgl_kernel.metal import decode_attention_paged_lazy_unchecked

        return decode_attention_paged_lazy_unchecked(
            queries,
            k_cache,
            v_cache,
            ctx.block_tables,
            ctx.context_lens,
            scale,
        )

    if (
        queries.shape[2] == 1
        and queries.dtype in (mx.float16, mx.bfloat16)
        and k_cache.dtype == queries.dtype
        and v_cache.dtype == queries.dtype
        and block_size == 1
        and head_dim == 128
        and (batch > 1 or max(ctx.context_len_values or [max_tokens]) > 16)
    ):
        from sgl_kernel.metal import decode_attention_paged_p1_lazy_unchecked

        return decode_attention_paged_p1_lazy_unchecked(
            queries,
            k_cache,
            v_cache,
            ctx.block_tables,
            ctx.context_lens,
            scale,
        )

    k_blocks = k_cache[ctx.block_tables]
    v_blocks = v_cache[ctx.block_tables]
    if block_size == 1 and k_cache.ndim == 3:
        k_blocks = k_blocks[:, :, None]
        v_blocks = v_blocks[:, :, None]
    k_dense = mx.contiguous(
        k_blocks.reshape(batch, max_tokens, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )
    )
    v_dense = mx.contiguous(
        v_blocks.reshape(batch, max_tokens, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )
    )

    mask = None
    if not ctx.context_len_values or any(
        context_len != max_tokens for context_len in ctx.context_len_values
    ):
        mask = _decode_padding_mask(ctx.context_lens, max_tokens, queries.dtype)

    return mx.fast.scaled_dot_product_attention(
        queries,
        k_dense,
        v_dense,
        scale=scale,
        mask=mask,
    )


def _prefill_attention_p1_mlx_fast(
    queries: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    ctx: PagedAttentionContext,
    scale: float,
) -> mx.array:
    """Dense MLX fallback for p1 paged-prefix prefill with unsupported dtypes."""
    if getattr(ctx.kv_pool, "block_size", 1) != 1:
        raise TypeError("BF16 paged-prefix prefill fallback requires block_size=1")
    B, _, q_len, _ = queries.shape
    max_context_len = max(ctx.context_len_values)
    key_positions = mx.arange(max_context_len)
    keys = []
    values = []
    masks = []

    for row_idx, context_len in enumerate(ctx.context_len_values):
        slot_ids = ctx.block_tables[row_idx, :context_len]
        k_row = k_cache[slot_ids, 0].transpose(1, 0, 2)
        v_row = v_cache[slot_ids, 0].transpose(1, 0, 2)
        if context_len < max_context_len:
            pad = max_context_len - context_len
            k_pad = mx.zeros((k_row.shape[0], pad, k_row.shape[2]), dtype=k_row.dtype)
            v_pad = mx.zeros((v_row.shape[0], pad, v_row.shape[2]), dtype=v_row.dtype)
            k_row = mx.concatenate([k_row, k_pad], axis=1)
            v_row = mx.concatenate([v_row, v_pad], axis=1)
        keys.append(k_row)
        values.append(v_row)

        prefix_len = ctx.radix_prefix_len_values[row_idx]
        query_positions = prefix_len + mx.arange(q_len)
        invalid = (key_positions[None, :] > query_positions[:, None]) | (
            key_positions[None, :] >= context_len
        )
        masks.append(invalid)

    k_all = mx.stack(keys, axis=0)
    v_all = mx.stack(values, axis=0)
    if B != k_all.shape[0]:
        raise ValueError("paged prefill fallback batch size mismatch")
    mask_bool = mx.stack(masks, axis=0)
    mask = mx.where(
        mask_bool[:, None, :, :],
        mx.array(mx.finfo(queries.dtype).min, dtype=queries.dtype),
        mx.array(0.0, dtype=queries.dtype),
    )
    return mx.fast.scaled_dot_product_attention(
        queries,
        k_all,
        v_all,
        scale=scale,
        mask=mask,
    )


def _decode_padding_mask(
    context_lens: mx.array, max_tokens: int, dtype: mx.Dtype
) -> mx.array:
    positions = mx.arange(max_tokens, dtype=mx.int32)
    invalid = positions[None, None, None, :] >= context_lens[:, None, None, None]
    return mx.where(
        invalid,
        mx.array(mx.finfo(dtype).min, dtype=dtype),
        mx.array(0.0, dtype=dtype),
    )
