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


def _as_int_list(value: Any | None) -> list[int]:
    if value is None:
        return []
    if mx is not None and isinstance(value, mx.array):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [int(item) for item in value]


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
    offset_values: list[int] = field(default_factory=list, init=False)
    context_len_values: list[int] = field(default_factory=list, init=False)
    radix_prefix_len_values: list[int] = field(default_factory=list, init=False)
    has_radix_prefix: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.offset_values = _as_int_list(self.offsets)
        self.context_len_values = _as_int_list(self.context_lens)
        self.radix_prefix_len_values = _as_int_list(self.radix_prefix_lens)
        self.has_radix_prefix = any(value > 0 for value in self.radix_prefix_len_values)
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
