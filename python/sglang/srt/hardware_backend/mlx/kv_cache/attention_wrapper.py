"""MLXAttentionWrapper and batched decode context for MLX backend."""

import threading
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache

_thread_local = threading.local()


@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # Per-request token count before the new token
    # layer_caches[layer_idx][req_idx] = ContiguousKVCache
    layer_caches: list[list[ContiguousKVCache]]


def set_context(ctx: BatchedDecodeContext) -> None:
    _thread_local.batched_ctx = ctx


def get_context() -> Optional[BatchedDecodeContext]:
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
