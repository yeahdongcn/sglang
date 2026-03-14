"""Unified KV cache module for MLX backend.

Uses ContiguousKVCache for all batch sizes, eliminating cache format
conversions during batch size transitions.
"""

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    BatchedDecodeContext,
    MLXAttentionWrapper,
    clear_context,
    get_context,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import (
    ContiguousKVCache,
    OffsetCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.model_patching import (
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.request_state import (
    MlxRequestState,
    extract_kv_cache,
)

__all__ = [
    "BatchedDecodeContext",
    "clear_context",
    "ContiguousKVCache",
    "extract_kv_cache",
    "find_attention_layers",
    "get_context",
    "get_num_layers",
    "MLXAttentionWrapper",
    "MlxRequestState",
    "OffsetCache",
    "patch_model_attention",
    "set_context",
]
