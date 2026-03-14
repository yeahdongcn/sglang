"""ContiguousKVCache and OffsetCache for MLX backend."""

import mlx.core as mx


class OffsetCache:
    """Fake cache that stores no data — only satisfies mlx-lm's protocol.

    During batched decode, real K/V lives in ContiguousKVCache objects,
    not in these shims. OffsetCache is needed so that:
    - create_attention_mask(h, cache) works (calls cache[0].make_mask(N))
    - The model can iterate over cache objects per layer
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
    """Pre-allocated KV buffer for a single request at a single layer.

    Shape: (1, n_kv_heads, max_seq_len, head_dim)
    Avoids expensive mx.concatenate by using slice assignment.

    Implements the mlx-lm cache protocol (update_and_fetch, state,
    make_mask) so it can be used as a drop-in replacement for KVCache
    in both prefill and decode paths.

    Supports lazy initialization: create with ``ContiguousKVCache()``
    and the buffer is allocated on the first ``update_and_fetch`` call
    when the actual dimensions and dtype are known.
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
        """Allocate buffers based on the first key tensor seen."""
        B, n_kv_heads, _, head_dim = keys.shape
        self.keys = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )
        self.values = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )

    # --- mlx-lm cache protocol ---

    @property
    def state(self):
        """Return arrays that need to be evaluated (for mx.eval unpacking)."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def make_mask(self, N, **kwargs):
        """Create attention mask for mlx-lm's create_attention_mask."""
        return None if N == 1 else "causal"

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Write new K/V tokens and return all valid K/V.

        On the first call, allocates the full contiguous buffer based on
        the shape and dtype of the incoming keys.

        Args:
            keys: (1, n_kv_heads, S, head_dim) — new tokens
            values: (1, n_kv_heads, S, head_dim) — new tokens

        Returns:
            Tuple of (all_keys, all_values) up to current offset.
        """
        if self.keys is None:
            self._allocate(keys)
        S = keys.shape[2]
        end = self.offset + S
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    # --- Batched decode helpers ---

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        self.keys[:, :, self.offset : self.offset + 1, :] = k
        self.values[:, :, self.offset : self.offset + 1, :] = v
        self.offset += 1

    def get_kv(self) -> tuple[mx.array, mx.array]:
        """Return valid K/V: (1, n_kv_heads, offset, head_dim)."""
        return self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :]
