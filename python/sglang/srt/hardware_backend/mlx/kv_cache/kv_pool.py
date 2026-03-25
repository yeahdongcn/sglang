"""Flat KV pool for MLX radix cache.

Pre-allocates per-layer K and V buffers as flat ``mx.array`` tensors
of shape ``(pool_size, n_kv_heads, head_dim)``, indexed by integer
slot IDs managed by a simple free-list allocator.

The pool layout uses token-first ordering ``(pool_size, n_kv_heads,
head_dim)`` for efficient leading-dimension scatter (simple index
assignment).  Callers transpose the small gathered slices when
loading into ``ContiguousKVCache``'s ``(1, n_kv_heads, S, head_dim)``
layout.
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


class SlotAllocator:
    """Integer free-list slot allocator.

    Manages a pool of ``pool_size`` integer slot IDs (0 .. pool_size-1).
    Alloc pops from the free list; free pushes back.
    """

    __slots__ = ("_free", "_capacity")

    def __init__(self, pool_size: int):
        self._capacity = pool_size
        self._free: list[int] = list(range(pool_size))

    @property
    def available(self) -> int:
        return len(self._free)

    @property
    def capacity(self) -> int:
        return self._capacity

    def alloc(self, n: int) -> Optional[list[int]]:
        """Allocate *n* slot IDs.  Returns ``None`` if not enough free."""
        if n > len(self._free):
            return None
        # Pop from the end — O(1) per pop, avoids O(len) list copies
        free = self._free
        start = len(free) - n
        slots = free[start:]
        del free[start:]
        return slots

    def free(self, slots: list[int]) -> None:
        """Return *slots* to the free list."""
        self._free.extend(slots)

    def clear(self) -> None:
        """Reset — mark all slots as free."""
        self._free = list(range(self._capacity))


class MlxKVPool:
    """Flat, pre-allocated KV cache pool for MLX.

    Stores K and V for all layers in flat arrays indexed by slot ID.

    Parameters
    ----------
    pool_size : int
        Number of token slots in the pool.
    num_layers : int
        Number of transformer layers.
    n_kv_heads : int
        Number of KV attention heads.
    head_dim : int
        Dimension per head.
    dtype : mx.Dtype
        Data type for KV buffers (e.g. ``mx.float16``).
    """

    def __init__(
        self,
        pool_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.allocator = SlotAllocator(pool_size)

        # Per-layer buffers: shape (pool_size, n_kv_heads, head_dim)
        # Token-first layout enables simple leading-dim scatter:
        #   buffer[slots] = data  (no read-modify-write needed)
        self.k_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]

        mem_mb = (pool_size * n_kv_heads * head_dim * 2 * num_layers * dtype.size) / (
            1024 * 1024
        )
        logger.info(
            f"MlxKVPool: {pool_size} slots × {num_layers} layers "
            f"× {n_kv_heads} heads × {head_dim} dim, "
            f"dtype={dtype}, ~{mem_mb:.1f} MB"
        )

    # --- scatter / gather ---

    def set_kv(
        self,
        layer_id: int,
        slots: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> None:
        """Write K/V tokens into the pool at the given slot IDs.

        Args:
            layer_id: Layer index.
            slots: 1-D int32 array of slot IDs, shape ``(S,)``.
            k: ``(S, n_kv_heads, head_dim)`` — token-first layout.
            v: ``(S, n_kv_heads, head_dim)``
        """
        # Leading-dim scatter — simple indexed assignment
        self.k_buffer[layer_id] = (
            self.k_buffer[layer_id].at[slots].add(k - self.k_buffer[layer_id][slots])
        )
        self.v_buffer[layer_id] = (
            self.v_buffer[layer_id].at[slots].add(v - self.v_buffer[layer_id][slots])
        )

    def get_kv(
        self,
        layer_id: int,
        slots: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Gather K/V tokens from the pool.

        Args:
            layer_id: Layer index.
            slots: 1-D int32 array of slot IDs, shape ``(S,)``.

        Returns:
            ``(k, v)`` each of shape ``(S, n_kv_heads, head_dim)``.
        """
        return (
            self.k_buffer[layer_id][slots],
            self.v_buffer[layer_id][slots],
        )

    def get_kv_all_layers(
        self,
        slots: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Gather K/V for ALL layers in one batched operation.

        Args:
            slots: 1-D int32 array of slot IDs, shape ``(S,)``.

        Returns:
            ``(k, v)`` each of shape ``(num_layers, S, n_kv_heads, head_dim)``.
        """
        k_all = mx.stack([self.k_buffer[i][slots] for i in range(self.num_layers)])
        v_all = mx.stack([self.v_buffer[i][slots] for i in range(self.num_layers)])
        return k_all, v_all

    def set_kv_all_layers(
        self,
        slots: mx.array,
        k_all: mx.array,
        v_all: mx.array,
    ) -> None:
        """Scatter K/V for ALL layers.

        Args:
            slots: 1-D int32 array of slot IDs, shape ``(S,)``.
            k_all: ``(num_layers, S, n_kv_heads, head_dim)``
            v_all: ``(num_layers, S, n_kv_heads, head_dim)``
        """
        for i in range(self.num_layers):
            self.set_kv(i, slots, k_all[i], v_all[i])

    def all_buffers(self) -> list[mx.array]:
        """Return all buffer arrays (for ``mx.eval``)."""
        return self.k_buffer + self.v_buffer

    def clear(self) -> None:
        """Reset allocator and zero all buffers."""
        self.allocator.clear()
        shape = (self.pool_size, self.n_kv_heads, self.head_dim)
        for i in range(self.num_layers):
            self.k_buffer[i] = mx.zeros(shape, dtype=self.dtype)
            self.v_buffer[i] = mx.zeros(shape, dtype=self.dtype)
