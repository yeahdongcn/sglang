"""ContiguousKVCache, PoolBackedCache and OffsetCache for MLX backend."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx


class LayerKVStore(Protocol):
    def get_kv(self, layer_id: int, slots: mx.array) -> tuple[mx.array, mx.array]: ...

    def get_kv_slot_range(
        self, layer_id: int, start: int, end: int
    ) -> tuple[mx.array, mx.array]: ...

    def get_kv_slot_ranges(
        self, layer_id: int, ranges: list[tuple[int, int]]
    ) -> tuple[mx.array, mx.array]: ...

    def get_kv_blocks(
        self, layer_id: int, block_ids: mx.array
    ) -> tuple[mx.array, mx.array]: ...


class OffsetCache:
    """Data-free shim satisfying mlx-lm's cache protocol.

    Provides ``make_mask`` and ``state`` without storing actual K/V.
    """

    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty — safe for mx.eval unpacking

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(self, keys, values):
        raise RuntimeError("OffsetCache should not store data")


_DEFAULT_MAX_SEQ_LEN = 4096


class ContiguousKVCache:
    """Pre-allocated KV buffer for one request × one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.
    """

    __slots__ = ("keys", "values", "offset", "max_seq_len")

    def __init__(
        self,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        dtype: mx.Dtype | None = None,
    ):
        if n_kv_heads is not None and head_dim is not None and dtype is not None:
            self.keys = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
            self.values = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        else:
            self.keys = None
            self.values = None
        self.offset = 0
        self.max_seq_len = max_seq_len

    def _allocate(self, keys: mx.array) -> None:
        """Allocate buffers matching the first key tensor's shape."""
        B, n_kv_heads, _, head_dim = keys.shape
        self.keys = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )
        self.values = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )

    @property
    def state(self):
        """Arrays for ``mx.eval`` unpacking."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def _grow(self, required: int) -> None:
        """Double the buffer until it can hold *required* tokens."""
        new_max = self.max_seq_len
        while new_max < required:
            new_max *= 2
        B, n_kv_heads, _, head_dim = self.keys.shape
        new_k = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.values.dtype)
        if self.offset > 0:
            new_k[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
            new_v[:, :, : self.offset, :] = self.values[:, :, : self.offset, :]
        self.keys = new_k
        self.values = new_v
        self.max_seq_len = new_max

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append K/V and return all valid K/V up to current offset."""
        if self.keys is None:
            self._allocate(keys)
        S = keys.shape[2]
        end = self.offset + S
        if end > self.keys.shape[2]:
            self.max_seq_len = max(self.max_seq_len, self.keys.shape[2])
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        if self.keys is None:
            self._allocate(k)
        end = self.offset + 1
        if end > self.keys.shape[2]:
            self.max_seq_len = max(self.max_seq_len, self.keys.shape[2])
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.offset += 1

    def get_kv(self) -> tuple[mx.array, mx.array]:
        """Return valid K/V: (1, n_kv_heads, offset, head_dim)."""
        return self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :]


class BatchedContiguousKVCache:
    """Batch view over per-request contiguous caches with a shared offset."""

    __slots__ = ("keys", "values", "offset", "_sources")

    def __init__(self, sources: list[ContiguousKVCache]):
        if not sources:
            raise ValueError("BatchedContiguousKVCache requires source caches")
        offset = sources[0].offset
        if any(source.offset != offset for source in sources):
            raise ValueError("BatchedContiguousKVCache requires equal offsets")
        if any(source.keys is None or source.values is None for source in sources):
            raise ValueError("BatchedContiguousKVCache requires populated sources")
        self.keys = mx.concatenate(
            [source.keys[:, :, :offset, :] for source in sources], axis=0
        )
        self.values = mx.concatenate(
            [source.values[:, :, :offset, :] for source in sources], axis=0
        )
        self.offset = offset
        self._sources = sources

    @property
    def state(self):
        return (self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :])

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        S = keys.shape[2]
        end = self.offset + S
        if end > self.keys.shape[2]:
            new_max = self.keys.shape[2]
            while new_max < end:
                new_max *= 2
            new_k = mx.zeros(
                (self.keys.shape[0], self.keys.shape[1], new_max, self.keys.shape[3]),
                dtype=self.keys.dtype,
            )
            new_v = mx.zeros(
                (
                    self.values.shape[0],
                    self.values.shape[1],
                    new_max,
                    self.values.shape[3],
                ),
                dtype=self.values.dtype,
            )
            if self.offset > 0:
                new_k[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
                new_v[:, :, : self.offset, :] = self.values[:, :, : self.offset, :]
            self.keys = new_k
            self.values = new_v
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        for row, source in enumerate(self._sources):
            source.update_and_fetch(keys[row : row + 1], values[row : row + 1])
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]


class PoolBackedCache:
    """Lazily gathers cached KV from the shared pool during forward pass.

    Each ``update_and_fetch`` gathers this layer's prefix from the pool
    on demand, keeping operations in the lazy compute graph.  Convert to
    ``ContiguousKVCache`` via ``to_contiguous`` after the forward pass.
    """

    __slots__ = (
        "_pool",
        "_layer_idx",
        "_slots",
        "offset",
        "_full_keys",
        "_full_values",
        "_new_keys",
        "_new_values",
        "_full_block_ids",
        "_slot_range",
        "_slot_ranges",
    )

    def __init__(
        self,
        pool: LayerKVStore,
        layer_idx: int,
        slots: mx.array,
        prefix_len: int,
        full_block_ids: mx.array | None = None,
        slot_range: tuple[int, int] | None = None,
        slot_ranges: list[tuple[int, int]] | None = None,
    ):
        self._pool = pool
        self._layer_idx = layer_idx
        self._slots = slots
        self.offset = prefix_len
        self._full_keys: mx.array | None = None
        self._full_values: mx.array | None = None
        self._new_keys: mx.array | None = None
        self._new_values: mx.array | None = None
        self._full_block_ids = full_block_ids
        self._slot_range = slot_range
        self._slot_ranges = slot_ranges

    @staticmethod
    def _trim_slot_ranges(
        slot_ranges: list[tuple[int, int]], token_count: int
    ) -> list[tuple[int, int]] | None:
        remaining = int(token_count)
        trimmed: list[tuple[int, int]] = []
        for start, end in slot_ranges:
            if remaining <= 0:
                break
            length = int(end) - int(start)
            if length <= 0:
                continue
            take = min(length, remaining)
            trimmed.append((int(start), int(start) + take))
            remaining -= take
        if remaining != 0:
            return None
        return trimmed

    @property
    def keys(self) -> mx.array | None:
        return self._full_keys

    @property
    def values(self) -> mx.array | None:
        return self._full_values

    @property
    def state(self):
        if self._full_keys is not None:
            return (self._full_keys, self._full_values)
        return ()

    def make_mask(self, N, **kwargs):
        return None if N == 1 else "causal"

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Gather cached prefix from pool, concatenate with new K/V."""
        S = keys.shape[2]

        if self.offset > 0:
            if (
                self._full_block_ids is not None
                and self.offset
                == self._full_block_ids.size * int(getattr(self._pool, "block_size", 1))
                and hasattr(self._pool, "get_kv_blocks")
            ):
                k_cached, v_cached = self._pool.get_kv_blocks(
                    self._layer_idx, self._full_block_ids
                )
            elif self._slot_range is not None and hasattr(
                self._pool, "get_kv_slot_range"
            ):
                start, _ = self._slot_range
                k_cached, v_cached = self._pool.get_kv_slot_range(
                    self._layer_idx, start, start + self.offset
                )
            elif self._slot_ranges is not None and hasattr(
                self._pool, "get_kv_slot_ranges"
            ):
                slot_ranges = self._trim_slot_ranges(self._slot_ranges, self.offset)
                if slot_ranges is None:
                    get_kv = getattr(self._pool, "get_kv_unchecked", self._pool.get_kv)
                    k_cached, v_cached = get_kv(
                        self._layer_idx, self._slots[: self.offset]
                    )
                else:
                    k_cached, v_cached = self._pool.get_kv_slot_ranges(
                        self._layer_idx, slot_ranges
                    )
            else:
                get_kv = getattr(self._pool, "get_kv_unchecked", self._pool.get_kv)
                k_cached, v_cached = get_kv(self._layer_idx, self._slots[: self.offset])
            # Pool layout (S, n_kv_heads, head_dim) → cache (1, n_kv_heads, S, head_dim)
            k_cached = k_cached.transpose(1, 0, 2)[None]
            v_cached = v_cached.transpose(1, 0, 2)[None]
            if k_cached.dtype != keys.dtype:
                k_cached = k_cached.astype(keys.dtype)
            if v_cached.dtype != values.dtype:
                v_cached = v_cached.astype(values.dtype)
            k_all = mx.concatenate([k_cached, keys], axis=2)
            v_all = mx.concatenate([v_cached, values], axis=2)
        else:
            k_all = keys
            v_all = values

        self.offset += S
        self._full_keys = k_all
        self._full_values = v_all
        self._new_keys = keys
        self._new_values = values
        return k_all, v_all

    def to_contiguous(self, max_seq_len: int = 4096) -> ContiguousKVCache:
        """Convert to ContiguousKVCache reusing forward-pass arrays."""
        cache = ContiguousKVCache(max_seq_len=max_seq_len)
        if self._full_keys is not None:
            cache.update_and_fetch(self._full_keys, self._full_values)
        return cache
