"""MLXAttentionWrapper and batched decode context for MLX backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool

_thread_local = threading.local()


@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # Per-request token count before the new token
    # layer_caches[layer_idx][req_idx] = ContiguousKVCache
    layer_caches: list[list[ContiguousKVCache]]


@dataclass
class RadixDecodeContext:
    """Context for radix-cache-backed decode.

    Instead of per-request contiguous caches, references the shared
    ``MlxKVPool`` and per-request slot-ID lists.
    """

    kv_pool: MlxKVPool
    batch_size: int
    seq_lens: list[int]  # tokens already cached (before new decode token)
    # Per-request slot IDs for historical tokens (excluding the new token)
    req_slot_ids: list[list[int]]
    # Slot ID allocated for each request's new decode token
    new_slot_ids: list[int]


_ContextType = Union[BatchedDecodeContext, RadixDecodeContext, None]


def set_context(ctx: _ContextType) -> None:
    _thread_local.batched_ctx = ctx


def get_context() -> _ContextType:
    return getattr(_thread_local, "batched_ctx", None)


def clear_context() -> None:
    _thread_local.batched_ctx = None


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention module for batched decode.

    When BatchedDecodeContext is set (BS>1), performs:
      1. Q/K/V projection
      2. Per-request RoPE with correct offsets
      3. Write new K/V token to contiguous cache (slice assignment)
      4. Gather all K/V, run batched SDPA
      5. Output projection

    When no context is set, delegates to inner module (fallback).
    In practice, the model is unpatched for BS=1 so this fallback
    path is never hit — ensuring zero overhead for single requests.
    """

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        if isinstance(ctx, RadixDecodeContext):
            return self._radix_decode(x, ctx)
        return self._batched_decode(x, ctx)

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

        # Per-head norms (Qwen3)
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        # Transpose to (B, heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Per-request RoPE + cache write + gather
        layer_caches = ctx.layer_caches[layer_idx]
        max_len = max(ctx.seq_lens) + 1  # +1 for new token

        q_parts = []
        all_k = []
        all_v = []

        for i in range(B):
            offset = ctx.seq_lens[i]
            q_i = inner.rope(queries[i : i + 1], offset=offset)
            k_i = inner.rope(keys[i : i + 1], offset=offset)
            v_i = values[i : i + 1]

            layer_caches[i].write_token(k_i, v_i)

            q_parts.append(q_i)

            k_all, v_all = layer_caches[i].get_kv()
            curr_len = layer_caches[i].offset

            if curr_len < max_len:
                pad = max_len - curr_len
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

        queries_b = mx.concatenate(q_parts, axis=0)
        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        # Build padding mask for variable-length sequences
        attn_mask = None
        seq_lens_plus1 = [s + 1 for s in ctx.seq_lens]
        if min(seq_lens_plus1) < max_len:
            attn_mask = mx.zeros((B, 1, 1, max_len), dtype=queries_b.dtype)
            for i in range(B):
                valid = seq_lens_plus1[i]
                if valid < max_len:
                    attn_mask[i, 0, 0, valid:] = -1e9

        output = mx.fast.scaled_dot_product_attention(
            queries_b, keys_b, values_b, scale=inner.scale, mask=attn_mask
        )

        # (B, heads, 1, head_dim) -> (B, 1, D)
        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return inner.o_proj(output)

    def _radix_decode(self, x: mx.array, ctx: RadixDecodeContext) -> mx.array:
        """Decode using the radix-cache KV pool.

        For each request:
          1. Project Q/K/V for the new token
          2. Apply RoPE at the correct offset
          3. Write new K/V to pool at the allocated slot
          4. Gather all historical K/V from pool (including new slot)
          5. Run batched SDPA
        """
        inner = self._inner
        layer_idx = self._layer_idx
        B = ctx.batch_size
        pool = ctx.kv_pool

        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = queries.shape[-1] // inner.n_heads
        queries = queries.reshape(B, 1, inner.n_heads, head_dim)
        keys = keys.reshape(B, 1, inner.n_kv_heads, head_dim)
        values = values.reshape(B, 1, inner.n_kv_heads, head_dim)

        # Per-head norms (Qwen3)
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        # Transpose to (B, heads, 1, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Per-request RoPE + pool write + pool gather
        max_len = max(ctx.seq_lens) + 1  # +1 for new token

        q_parts = []
        all_k = []
        all_v = []

        for i in range(B):
            offset = ctx.seq_lens[i]
            q_i = inner.rope(queries[i : i + 1], offset=offset)
            k_i = inner.rope(keys[i : i + 1], offset=offset)
            v_i = values[i : i + 1]

            # k_i/v_i are (1, n_kv_heads, 1, head_dim) — squeeze to (1, n_kv_heads, head_dim)
            k_flat = k_i.squeeze(2)  # (1, n_kv_heads, head_dim)
            v_flat = v_i.squeeze(2)

            # Write to pool
            new_slot = mx.array([ctx.new_slot_ids[i]], dtype=mx.int32)
            pool.set_kv(layer_idx, new_slot, k_flat, v_flat)

            # Gather all historical K/V (old slots + new slot)
            all_slots = ctx.req_slot_ids[i] + [ctx.new_slot_ids[i]]
            slot_ids = mx.array(all_slots, dtype=mx.int32)
            k_hist, v_hist = pool.get_kv(layer_idx, slot_ids)
            # k_hist: (S, n_kv_heads, head_dim) → (1, n_kv_heads, S, head_dim)
            k_hist = k_hist.transpose(1, 0, 2)[None]
            v_hist = v_hist.transpose(1, 0, 2)[None]

            curr_len = len(all_slots)
            if curr_len < max_len:
                pad = max_len - curr_len
                k_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=k_hist.dtype
                )
                v_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=v_hist.dtype
                )
                k_hist = mx.concatenate([k_hist, k_pad], axis=2)
                v_hist = mx.concatenate([v_hist, v_pad], axis=2)

            q_parts.append(q_i)
            all_k.append(k_hist)
            all_v.append(v_hist)

        queries_b = mx.concatenate(q_parts, axis=0)
        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        # Build padding mask for variable-length sequences
        attn_mask = None
        seq_lens_plus1 = [s + 1 for s in ctx.seq_lens]
        if min(seq_lens_plus1) < max_len:
            attn_mask = mx.zeros((B, 1, 1, max_len), dtype=queries_b.dtype)
            for i in range(B):
                valid = seq_lens_plus1[i]
                if valid < max_len:
                    attn_mask[i, 0, 0, valid:] = -1e9

        output = mx.fast.scaled_dot_product_attention(
            queries_b, keys_b, values_b, scale=inner.scale, mask=attn_mask
        )

        # (B, heads, 1, head_dim) -> (B, 1, D)
        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return inner.o_proj(output)
