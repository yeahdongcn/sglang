"""Unified KV cache module for MLX backend.

Uses ContiguousKVCache for all batch sizes with a shared KV pool and
trie-based prefix matching (radix cache).
"""

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    BatchedDecodeContext,
    MLXAttentionWrapper,
    RadixDecodeContext,
    clear_context,
    get_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import (
    ContiguousKVCache,
    OffsetCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.radix_trie import MlxRadixTrie

__all__ = [
    "BatchedDecodeContext",
    "clear_context",
    "ContiguousKVCache",
    "find_attention_layers",
    "get_context",
    "get_num_layers",
    "MLXAttentionWrapper",
    "MlxKVPool",
    "MlxRadixTrie",
    "OffsetCache",
    "patch_model_attention",
    "RadixDecodeContext",
    "set_context",
]
