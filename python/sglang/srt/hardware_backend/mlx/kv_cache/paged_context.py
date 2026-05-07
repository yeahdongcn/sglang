"""Paged attention context for the MLX/Metal backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional Apple Silicon dependency
    mx = None


_thread_local = threading.local()


def _as_int32_array(value: Any | None):
    if value is None or mx is None:
        return value
    if isinstance(value, mx.array):
        return value.astype(mx.int32) if value.dtype != mx.int32 else value
    return mx.array(value, dtype=mx.int32)


@dataclass
class PagedAttentionContext:
    """Scheduler-owned paged-attention metadata for one MLX forward pass."""

    is_prefill: bool
    slot_mapping: Any
    block_tables: Any | None = None
    context_lens: Any | None = None
    offsets: Any | None = None
    cu_seqlens: Any | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    radix_prefix_lens: Any | None = None
    kv_pool: Any | None = None
    kv_scatter_layer_ids: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.slot_mapping = _as_int32_array(self.slot_mapping)
        self.block_tables = _as_int32_array(self.block_tables)
        self.context_lens = _as_int32_array(self.context_lens)
        self.offsets = _as_int32_array(self.offsets)
        self.cu_seqlens = _as_int32_array(self.cu_seqlens)
        self.radix_prefix_lens = _as_int32_array(self.radix_prefix_lens)

    def mark_kv_scattered(self, layer_idx: int) -> None:
        self.kv_scatter_layer_ids.add(layer_idx)

    def has_scattered_all_layers(self, num_layers: int) -> bool:
        return all(
            layer_idx in self.kv_scatter_layer_ids for layer_idx in range(num_layers)
        )

    @property
    def batch_size(self) -> int:
        if self.context_lens is not None:
            return int(self.context_lens.shape[0])
        if self.block_tables is not None:
            return int(self.block_tables.shape[0])
        if self.offsets is not None:
            return int(self.offsets.shape[0])
        return 1


def set_paged_context(ctx: PagedAttentionContext | None) -> None:
    _thread_local.paged_ctx = ctx


def get_paged_context() -> PagedAttentionContext | None:
    return getattr(_thread_local, "paged_ctx", None)


def clear_paged_context() -> None:
    _thread_local.paged_ctx = None
