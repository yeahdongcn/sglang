"""Block-paged KV cache primitives for the MLX/Metal backend."""

from __future__ import annotations

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional Apple Silicon dependency
    mx = None

from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import normalize_mlx_metal_dtype


class MlxPagedKVCache:
    """Per-layer block-paged KV storage with scheduler-owned slot IDs."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype=None,
        normalize_dtype: bool = True,
    ) -> None:
        if mx is None:
            raise ImportError("MLX is required for MlxPagedKVCache")
        if min(num_layers, num_blocks, block_size, n_kv_heads, head_dim) <= 0:
            raise ValueError("paged KV cache dimensions must be positive")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = normalize_mlx_metal_dtype(dtype) if normalize_dtype else dtype
        if self.dtype is None:
            self.dtype = mx.float16
        self.capacity = num_blocks * block_size

        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        self.k_buffer = [
            mx.contiguous(mx.zeros(shape, dtype=self.dtype)) for _ in range(num_layers)
        ]
        self.v_buffer = [
            mx.contiguous(mx.zeros(shape, dtype=self.dtype)) for _ in range(num_layers)
        ]

    def slot_to_block_offset(self, slots):
        slots = self._as_slots(slots)
        return slots // self.block_size, slots % self.block_size

    def set_kv(self, layer_id: int, slots, k, v, *, eager: bool = False) -> None:
        self._check_layer(layer_id)
        slots = self._as_slots(slots)
        self._check_kv_shape(k, slots.size)
        self._check_kv_shape(v, slots.size)
        if self.block_size == 1:
            self.k_buffer[layer_id][slots, 0] = k
            self.v_buffer[layer_id][slots, 0] = v
            if eager and slots.size:
                mx.eval(
                    self.k_buffer[layer_id][slots, 0],
                    self.v_buffer[layer_id][slots, 0],
                )
            return
        block_ids = slots // self.block_size
        block_offsets = slots % self.block_size
        self.k_buffer[layer_id][block_ids, block_offsets] = k
        self.v_buffer[layer_id][block_ids, block_offsets] = v
        if eager and slots.size:
            mx.eval(
                self.k_buffer[layer_id][block_ids, block_offsets],
                self.v_buffer[layer_id][block_ids, block_offsets],
            )

    def get_kv(self, layer_id: int, slots):
        self._check_layer(layer_id)
        slots = self._as_slots(slots)
        return self.get_kv_unchecked(layer_id, slots)

    def get_kv_unchecked(self, layer_id: int, slots):
        self._check_layer(layer_id)
        if self.block_size == 1:
            return (
                self._flat_k_buffer(layer_id)[slots],
                self._flat_v_buffer(layer_id)[slots],
            )
        block_ids = slots // self.block_size
        block_offsets = slots % self.block_size
        return (
            self.k_buffer[layer_id][block_ids, block_offsets],
            self.v_buffer[layer_id][block_ids, block_offsets],
        )

    def get_kv_slot_range(self, layer_id: int, start: int, end: int):
        self._check_layer(layer_id)
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.capacity:
            raise ValueError("slot range must be within paged KV cache capacity")
        if self.block_size == 1:
            return (
                self._flat_k_buffer(layer_id)[start:end],
                self._flat_v_buffer(layer_id)[start:end],
            )
        slots = mx.arange(start, end, dtype=mx.int32)
        return self.get_kv_unchecked(layer_id, slots)

    def get_kv_slot_ranges(self, layer_id: int, ranges: list[tuple[int, int]]):
        self._check_layer(layer_id)
        ranges = [(int(start), int(end)) for start, end in ranges]
        if not ranges:
            empty = mx.zeros((0, self.n_kv_heads, self.head_dim), dtype=self.dtype)
            return empty, empty
        if len(ranges) == 1:
            return self.get_kv_slot_range(layer_id, ranges[0][0], ranges[0][1])

        k_parts = []
        v_parts = []
        for start, end in ranges:
            k_part, v_part = self.get_kv_slot_range(layer_id, start, end)
            k_parts.append(k_part)
            v_parts.append(v_part)
        return mx.concatenate(k_parts, axis=0), mx.concatenate(v_parts, axis=0)

    def get_kv_blocks(self, layer_id: int, block_ids):
        self._check_layer(layer_id)
        block_ids = self._as_block_ids(block_ids)
        return self.get_kv_blocks_unchecked(layer_id, block_ids)

    def get_kv_blocks_unchecked(self, layer_id: int, block_ids):
        self._check_layer(layer_id)
        if self.block_size == 1:
            return (
                self._flat_k_buffer(layer_id)[block_ids],
                self._flat_v_buffer(layer_id)[block_ids],
            )
        k_blocks = self.k_buffer[layer_id][block_ids]
        v_blocks = self.v_buffer[layer_id][block_ids]
        token_count = block_ids.size * self.block_size
        return (
            k_blocks.reshape(token_count, self.n_kv_heads, self.head_dim),
            v_blocks.reshape(token_count, self.n_kv_heads, self.head_dim),
        )

    def gather_blocks(self, layer_id: int, block_tables):
        self._check_layer(layer_id)
        block_tables = self._as_block_tables(block_tables)
        return (
            self.k_buffer[layer_id][block_tables],
            self.v_buffer[layer_id][block_tables],
        )

    def gather_block_table_tokens(self, layer_id: int, block_tables, context_lens):
        self._check_layer(layer_id)
        block_tables = self._as_block_tables(block_tables)
        max_tokens = block_tables.shape[1] * self.block_size
        context_lens = self._as_context_lens(
            context_lens, block_tables.shape[0], max_tokens
        )
        k_blocks, v_blocks = self.gather_blocks(layer_id, block_tables)
        offsets = mx.arange(block_tables.shape[1] * self.block_size, dtype=mx.int32)
        offsets = offsets.reshape(block_tables.shape[1], self.block_size)
        valid_mask = offsets[None] < context_lens[:, None, None]
        return (
            k_blocks,
            v_blocks,
            valid_mask.reshape(
                block_tables.shape[0], block_tables.shape[1] * self.block_size
            ),
        )

    def set_kv_all_layers(
        self, slots, k_all, v_all, *, eager: bool = False, use_metal: bool = False
    ) -> None:
        slots = self._as_slots(slots)
        if len(k_all) != self.num_layers or len(v_all) != self.num_layers:
            raise ValueError(
                "all-layer KV tensors must have num_layers as the first dim"
            )
        for layer_id in range(self.num_layers):
            self._check_kv_shape(k_all[layer_id], slots.size)
            self._check_kv_shape(v_all[layer_id], slots.size)
        if slots.size == 0:
            return
        if use_metal:
            try:
                from sgl_kernel.metal import paged_kv_scatter_all_layers_unchecked
            except ImportError:
                pass
            else:
                paged_kv_scatter_all_layers_unchecked(
                    k_all,
                    v_all,
                    self.k_buffer,
                    self.v_buffer,
                    slots,
                    eager=eager,
                )
                return
        if self.block_size == 1:
            eager_refs = []
            for layer_id in range(self.num_layers):
                self.k_buffer[layer_id][slots, 0] = k_all[layer_id]
                self.v_buffer[layer_id][slots, 0] = v_all[layer_id]
                if eager:
                    eager_refs.extend(
                        [
                            self.k_buffer[layer_id][slots, 0],
                            self.v_buffer[layer_id][slots, 0],
                        ]
                    )
            if eager_refs:
                mx.eval(*eager_refs)
            return
        block_ids = slots // self.block_size
        block_offsets = slots % self.block_size
        eager_refs = []
        for layer_id in range(self.num_layers):
            self.k_buffer[layer_id][block_ids, block_offsets] = k_all[layer_id]
            self.v_buffer[layer_id][block_ids, block_offsets] = v_all[layer_id]
            if eager:
                eager_refs.extend(
                    [
                        self.k_buffer[layer_id][block_ids, block_offsets],
                        self.v_buffer[layer_id][block_ids, block_offsets],
                    ]
                )
        if eager_refs:
            mx.eval(*eager_refs)

    def set_kv_all_layers_slot_range(
        self, start: int, end: int, k_all, v_all, *, eager: bool = False
    ) -> None:
        if self.block_size != 1:
            raise ValueError("slot-range KV writes require block_size=1")
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.capacity:
            raise ValueError("slot range must be within paged KV cache capacity")
        tokens = end - start
        if len(k_all) != self.num_layers or len(v_all) != self.num_layers:
            raise ValueError(
                "all-layer KV tensors must have num_layers as the first dim"
            )
        for layer_id in range(self.num_layers):
            self._check_kv_shape(k_all[layer_id], tokens)
            self._check_kv_shape(v_all[layer_id], tokens)
        if tokens == 0:
            return

        eager_refs = []
        for layer_id in range(self.num_layers):
            k_layer = k_all[layer_id]
            v_layer = v_all[layer_id]
            if k_layer.dtype != self.dtype:
                k_layer = k_layer.astype(self.dtype)
            if v_layer.dtype != self.dtype:
                v_layer = v_layer.astype(self.dtype)
            self.k_buffer[layer_id][start:end, 0] = k_layer
            self.v_buffer[layer_id][start:end, 0] = v_layer
            if eager:
                eager_refs.extend(
                    [
                        self.k_buffer[layer_id][start:end, 0],
                        self.v_buffer[layer_id][start:end, 0],
                    ]
                )
        if eager_refs:
            mx.eval(*eager_refs)

    def _contiguous_slot_range(self, slots) -> tuple[int, int] | None:
        if slots.size == 0:
            return None
        start = int(slots[0].item())
        end = start + int(slots.size)
        if end > self.capacity:
            return None
        expected = mx.arange(start, end, dtype=mx.int32)
        if bool(mx.all(slots == expected).item()):
            return start, end
        return None

    def _flat_k_buffer(self, layer_id: int):
        if self.k_buffer[layer_id].ndim == 3:
            return self.k_buffer[layer_id]
        return self.k_buffer[layer_id].reshape(
            self.num_blocks, self.n_kv_heads, self.head_dim
        )

    def _flat_v_buffer(self, layer_id: int):
        if self.v_buffer[layer_id].ndim == 3:
            return self.v_buffer[layer_id]
        return self.v_buffer[layer_id].reshape(
            self.num_blocks, self.n_kv_heads, self.head_dim
        )

    def reset_slots(self, slots) -> None:
        slots = self._as_slots(slots)
        if slots.size == 0:
            return
        block_ids = slots // self.block_size
        block_offsets = slots % self.block_size
        zeros = mx.zeros((slots.size, self.n_kv_heads, self.head_dim), dtype=self.dtype)
        for layer_id in range(self.num_layers):
            self.k_buffer[layer_id][block_ids, block_offsets] = zeros
            self.v_buffer[layer_id][block_ids, block_offsets] = zeros

    def reset_blocks(self, block_ids) -> None:
        block_ids = self._as_block_ids(block_ids)
        if block_ids.size == 0:
            return
        zeros = mx.zeros(
            (block_ids.size, self.block_size, self.n_kv_heads, self.head_dim),
            dtype=self.dtype,
        )
        for layer_id in range(self.num_layers):
            self.k_buffer[layer_id][block_ids] = zeros
            self.v_buffer[layer_id][block_ids] = zeros

    def clear(self) -> None:
        shape = (self.num_blocks, self.block_size, self.n_kv_heads, self.head_dim)
        self.k_buffer = [
            mx.contiguous(mx.zeros(shape, dtype=self.dtype))
            for _ in range(self.num_layers)
        ]
        self.v_buffer = [
            mx.contiguous(mx.zeros(shape, dtype=self.dtype))
            for _ in range(self.num_layers)
        ]

    @property
    def state(self) -> list:
        return [
            buf
            for pair in zip(self.k_buffer, self.v_buffer, strict=True)
            for buf in pair
        ]

    def _check_layer(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.num_layers:
            raise IndexError(f"layer_id {layer_id} is outside [0, {self.num_layers})")

    def _as_slots(self, slots):
        if not isinstance(slots, mx.array):
            slots = mx.array(slots, dtype=mx.int32)
        if slots.dtype != mx.int32:
            slots = slots.astype(mx.int32)
        if slots.ndim != 1:
            raise ValueError("slot IDs must be a 1-D array")
        if slots.size == 0:
            return slots
        if int(mx.min(slots).item()) < 0 or int(mx.max(slots).item()) >= self.capacity:
            raise ValueError("slot IDs must be within paged KV cache capacity")
        return slots

    def _as_block_tables(self, block_tables):
        if not isinstance(block_tables, mx.array):
            block_tables = mx.array(block_tables, dtype=mx.int32)
        if block_tables.dtype != mx.int32:
            block_tables = block_tables.astype(mx.int32)
        if block_tables.ndim != 2:
            raise ValueError("block tables must be a 2-D array")
        if block_tables.size == 0:
            return block_tables
        if (
            int(mx.min(block_tables).item()) < 0
            or int(mx.max(block_tables).item()) >= self.num_blocks
        ):
            raise ValueError("block IDs must be within paged KV cache block capacity")
        return block_tables

    def _as_block_ids(self, block_ids):
        if not isinstance(block_ids, mx.array):
            block_ids = mx.array(block_ids, dtype=mx.int32)
        if block_ids.dtype != mx.int32:
            block_ids = block_ids.astype(mx.int32)
        if block_ids.ndim != 1:
            raise ValueError("block IDs must be a 1-D array")
        if block_ids.size == 0:
            return block_ids
        if (
            int(mx.min(block_ids).item()) < 0
            or int(mx.max(block_ids).item()) >= self.num_blocks
        ):
            raise ValueError("block IDs must be within paged KV cache block capacity")
        return block_ids

    @staticmethod
    def _as_context_lens(context_lens, batch_size: int, max_tokens: int):
        if not isinstance(context_lens, mx.array):
            context_lens = mx.array(context_lens, dtype=mx.int32)
        if context_lens.dtype != mx.int32:
            context_lens = context_lens.astype(mx.int32)
        if context_lens.ndim != 1:
            raise ValueError("context lengths must be a 1-D array")
        if context_lens.shape[0] != batch_size:
            raise ValueError("context lengths must match block table batch size")
        if context_lens.size and int(mx.min(context_lens).item()) < 0:
            raise ValueError("context lengths must be non-negative")
        if context_lens.size and int(mx.max(context_lens).item()) > max_tokens:
            raise ValueError("context lengths must fit within gathered block tables")
        return context_lens

    def _check_kv_shape(self, tensor, num_tokens: int) -> None:
        if tensor.ndim != 3:
            raise ValueError(
                "KV tensor must have shape (num_tokens, n_kv_heads, head_dim)"
            )
        if tensor.shape[1:] != (self.n_kv_heads, self.head_dim):
            raise ValueError(
                "KV tensor trailing shape must match (n_kv_heads, head_dim)"
            )
        if tensor.shape[0] != num_tokens:
            raise ValueError("KV tensor token count must match slot count")
