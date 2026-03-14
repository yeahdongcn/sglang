"""Per-request state for MLX inference."""

from dataclasses import dataclass

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache


@dataclass
class MlxRequestState:
    """Per-request state for MLX inference.

    The cache is always ``list[ContiguousKVCache]``.
    """

    token_ids: list[int]
    cache: list[ContiguousKVCache]
    generated_tokens: int = 0


def extract_kv_cache(
    batch_caches: list[ContiguousKVCache], idx: int
) -> list[ContiguousKVCache]:
    """Extract a single request's cache from batched caches after batched prefill.

    After batched prefill, each ContiguousKVCache has keys/values of shape
    (B, H, max_seq_len, D). We slice out the idx-th request into a fresh
    single-request ContiguousKVCache.
    """
    extracted: list[ContiguousKVCache] = []
    for cache in batch_caches:
        new_cache = ContiguousKVCache()
        src_keys = mx.contiguous(cache.keys[idx : idx + 1, :, : cache.offset, :])
        src_values = mx.contiguous(cache.values[idx : idx + 1, :, : cache.offset, :])
        new_cache.update_and_fetch(src_keys, src_values)
        extracted.append(new_cache)
    return extracted
