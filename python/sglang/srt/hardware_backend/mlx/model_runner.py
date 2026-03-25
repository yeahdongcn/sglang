"""End-to-end MLX model runner for Apple Silicon.

Runs the entire model within MLX, bypassing PyTorch MPS entirely.
KV data is stored in a shared flat pool (``MlxKVPool``) indexed by a
radix trie.  When ``disable_radix_cache=False`` (default), prefix
sharing across requests is enabled.  When ``disable_radix_cache=True``,
the pool and trie still exist but prefix matching and trie insertion
are skipped — matching the CUDA backend's behavior.
"""

import logging
import time

import mlx.core as mx
from mlx_lm import load as mlx_lm_load

from sglang.srt.hardware_backend.mlx.kv_cache import (
    BatchedDecodeContext,
    ContiguousKVCache,
    MLXAttentionWrapper,
    OffsetCache,
    clear_context,
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.hardware_backend.mlx.kv_cache.radix_trie import MlxRadixTrie

logger = logging.getLogger(__name__)

# Default pool size (number of token slots) for radix cache
_DEFAULT_POOL_SIZE = 65536


class MlxModelRunner:
    """Model runner that executes the entire model in MLX.

    This avoids the MPS<->MLX tensor bridge overhead by keeping all
    computation within MLX.

    When ``disable_radix_cache`` is *False* (default), KV data is stored
    in a shared flat pool indexed by a radix trie for prefix sharing.
    When *True*, the pool still exists but prefix matching and trie
    insertion are skipped — matching the CUDA backend's behavior.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int = _DEFAULT_POOL_SIZE,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache

        self._load_model()
        patch_model_attention(self.model)

        # Cache layer count to avoid repeated model traversal
        self._num_layers = get_num_layers(self.model)

        # Initial buffer size for ContiguousKVCache.  Kept small so
        # that short sequences (the common case & benchmark path) don't
        # over-allocate.  The cache will dynamically double its buffer
        # whenever a sequence exceeds this limit (see ContiguousKVCache._grow).
        self._max_seq_len = 4096

        # Per-request contiguous caches (used in both modes)
        self._req_caches: dict[str, list[ContiguousKVCache]] = {}
        # req_id → list of token IDs
        self._req_token_ids: dict[str, list[int]] = {}

        # Reusable cache object pool to avoid per-request allocation
        self._cache_pool: list[list[ContiguousKVCache]] = []

        # Radix cache state (only used when radix cache is enabled)
        self._kv_pool: MlxKVPool | None = None
        self._radix_trie: MlxRadixTrie | None = None
        # req_id → list of slot IDs (all tokens including generated)
        self._req_slot_ids: dict[str, list[int]] = {}
        # req_ids whose KV data has NOT yet been synced to the pool
        self._pool_dirty: set[str] = set()
        # req_id → trie node locked by this request (for dec_ref on removal)
        self._req_last_node: dict[str, object | None] = {}
        # req_id → number of prefix tokens loaded from pool (skip sync for these)
        self._req_prefix_len: dict[str, int] = {}

        self._init_radix_cache(pool_size)

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _acquire_cache(self) -> list[ContiguousKVCache]:
        """Get a reusable cache list from the pool, or create a new one."""
        if self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                c.offset = 0
            return cache
        return [
            ContiguousKVCache(max_seq_len=self._max_seq_len)
            for _ in range(self._num_layers)
        ]

    def _release_cache(self, cache: list[ContiguousKVCache]) -> None:
        """Return a cache list to the pool for reuse."""
        self._cache_pool.append(cache)

    @staticmethod
    def _eval_with_cache(
        token_result: mx.array, cache: list[ContiguousKVCache]
    ) -> None:
        """Evaluate token result and all cache states in a single mx.eval call.

        Avoids the overhead of building a list of tuples from cache.state
        and unpacking it on every call.
        """
        eval_args = [token_result]
        for c in cache:
            if c.keys is not None:
                eval_args.append(c.keys)
                eval_args.append(c.values)
        mx.eval(*eval_args)

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def _init_radix_cache(self, pool_size: int) -> None:
        """Initialize the radix cache pool and trie based on model dimensions."""
        num_layers = self._num_layers

        # Probe model dimensions from attention layer
        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            raise RuntimeError("Cannot init radix cache: no attention layers found")

        # Drill into the transformer block to find the actual attention module
        sample_block = layer_list[0]
        sample_attn = getattr(sample_block, attn_attr)
        # If already wrapped, get the inner module
        if isinstance(sample_attn, MLXAttentionWrapper):
            sample_attn = sample_attn._inner

        n_kv_heads = sample_attn.n_kv_heads

        # head_dim may not be stored directly; derive from k_proj weight shape
        if hasattr(sample_attn, "head_dim"):
            head_dim = sample_attn.head_dim
        elif hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            # k_proj.weight shape: (n_kv_heads * head_dim, hidden_size)
            head_dim = sample_attn.k_proj.weight.shape[0] // n_kv_heads
        else:
            raise RuntimeError("Cannot determine head_dim from attention module")

        # Determine dtype from model weights
        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype

        self._kv_pool = MlxKVPool(
            pool_size=pool_size,
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self._radix_trie = MlxRadixTrie(pool_capacity=pool_size)
        logger.info(
            f"Radix cache initialized: {pool_size} slots, "
            f"{num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Prefill using radix cache pool with optional prefix reuse.

        When radix cache is enabled, checks the trie for a cached prefix.
        If found, cached KV data is loaded from the pool and only the
        *new* tokens are forwarded — saving compute proportional to
        prefix length.  When ``disable_radix_cache`` is True, prefix
        matching and trie insertion are skipped and pool slot allocation
        and sync are bypassed entirely (matching CUDA behavior).

        Returns:
            Tuple of (next_token_id, prefix_len) where prefix_len is the
            number of tokens that were served from the radix cache.
        """
        num_layers = self._num_layers
        num_tokens = len(token_ids)

        # --- Fast path: radix cache disabled ---
        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            next_token_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            self._eval_with_cache(next_token_mlx, cache)
            next_token = int(next_token_mlx.item())

            self._req_token_ids[req_id] = list(token_ids) + [next_token]
            self._req_caches[req_id] = cache
            # Empty slot list — no pool involvement
            self._req_slot_ids[req_id] = []
            return next_token, 0

        # --- Radix cache enabled path ---
        assert self._kv_pool is not None and self._radix_trie is not None

        match = self._radix_trie.match_prefix(token_ids)
        prefix_len = match.prefix_len
        matched_node = match.last_node

        # Lock the matched node so it won't be evicted while we use it
        if prefix_len > 0:
            self._radix_trie.inc_ref(matched_node)

        # Only allocate slots for the NEW (non-cached) tokens
        new_token_count = num_tokens - prefix_len
        if new_token_count > 0:
            new_slots = self._kv_pool.allocator.alloc(new_token_count)
            if new_slots is None:
                freed = self._radix_trie.evict(new_token_count)
                if freed:
                    self._kv_pool.allocator.free(freed)
                new_slots = self._kv_pool.allocator.alloc(new_token_count)
                if new_slots is None:
                    if prefix_len > 0:
                        self._radix_trie.dec_ref(matched_node)
                    raise RuntimeError(
                        f"KV pool exhausted: need {new_token_count} slots, "
                        f"only {self._kv_pool.allocator.available} available"
                    )
        else:
            new_slots = []

        # All slot IDs for this request: cached prefix + newly allocated
        if prefix_len > 0:
            all_slots = match.slot_ids + new_slots
        else:
            all_slots = new_slots

        # Build contiguous caches
        cache = self._acquire_cache()

        # If we have a cached prefix, pre-populate the contiguous caches
        if prefix_len > 0:
            # Flush any pending syncs so pool data is valid before reading
            self._flush_pending_syncs()
            slot_ids_mx = mx.array(match.slot_ids, dtype=mx.int32)
            k_all, v_all = self._kv_pool.get_kv_all_layers(slot_ids_mx)
            for layer_idx in range(num_layers):
                k_t = k_all[layer_idx].transpose(1, 0, 2)[None]
                v_t = v_all[layer_idx].transpose(1, 0, 2)[None]
                cache[layer_idx].update_and_fetch(k_t, v_t)

            if new_token_count > 0:
                logger.info(
                    f"Prefix reuse: {prefix_len}/{num_tokens} tokens cached, "
                    f"computing {new_token_count} new tokens"
                )

        # Run model on remaining (non-cached) tokens only
        if new_token_count > 0:
            extend_tokens = token_ids[prefix_len:]
        else:
            extend_tokens = token_ids[-1:]
            for c in cache:
                c.offset = max(c.offset - 1, 0)

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        # Eval model output only — pool sync is deferred to remove_request
        self._eval_with_cache(next_token_mlx, cache)

        next_token = int(next_token_mlx.item())

        # Insert new tokens into trie
        self._radix_trie.insert(token_ids, all_slots)

        # Track per-request state
        self._req_slot_ids[req_id] = all_slots
        self._req_token_ids[req_id] = list(token_ids) + [next_token]
        self._req_caches[req_id] = cache
        self._pool_dirty.add(req_id)
        self._req_last_node[req_id] = matched_node if prefix_len > 0 else None
        self._req_prefix_len[req_id] = prefix_len

        return next_token, prefix_len

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.

        Processes additional prompt tokens for an existing request whose
        KV cache already contains earlier chunks.  Allocates new pool
        slots for the incoming tokens, runs the model forward using the
        existing contiguous cache, and updates all bookkeeping.

        Returns:
            Next token ID (greedy sampled)
        """
        assert req_id in self._req_caches, f"extend called for unknown request {req_id}"

        cache = self._req_caches[req_id]
        num_new = len(new_token_ids)

        # Allocate new pool slots (skip when radix cache is disabled)
        if not self.disable_radix_cache:
            new_slots = self._kv_pool.allocator.alloc(num_new)
            if new_slots is None:
                freed = self._radix_trie.evict(num_new)
                if freed:
                    self._kv_pool.allocator.free(freed)
                new_slots = self._kv_pool.allocator.alloc(num_new)
                if new_slots is None:
                    raise RuntimeError(
                        f"KV pool exhausted: need {num_new} slots, "
                        f"only {self._kv_pool.allocator.available} available"
                    )

        # Run model forward
        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)
        self._eval_with_cache(next_token_mlx, cache)
        next_token = int(next_token_mlx.item())

        # Update bookkeeping
        prev_tokens = self._req_token_ids[req_id]
        if prev_tokens:
            prev_tokens.pop()  # remove stale intermediate token
        prev_tokens.extend(new_token_ids)
        prev_tokens.append(next_token)

        if not self.disable_radix_cache:
            self._req_slot_ids[req_id].extend(new_slots)
            self._pool_dirty.add(req_id)
            full_prompt = prev_tokens[:-1]
            self._radix_trie.insert(full_prompt, self._req_slot_ids[req_id])

        logger.info(
            f"Extend req {req_id}: +{num_new} tokens, "
            f"total cache offset {cache[0].offset}"
        )

        return next_token

    def _flush_pending_syncs(self) -> None:
        """Flush all pending KV syncs to the pool.

        Called before any pool READ (cache hit path) to ensure data
        consistency.  Cold prefills (no cache hit) never call this,
        so they pay zero sync cost.
        """
        if not self._pool_dirty:
            return
        for rid in list(self._pool_dirty):
            self._sync_request_to_pool(rid)

    def _sync_request_to_pool(self, req_id: str) -> None:
        """Sync a request's KV data from contiguous cache to the pool.

        Only writes the *new* (non-prefix) tokens to the pool — prefix
        tokens are already in the pool from a previous request.  Called
        lazily when pool data is actually needed (e.g., request completion).
        """
        if req_id not in self._pool_dirty:
            return
        cache = self._req_caches.get(req_id)
        slots = self._req_slot_ids.get(req_id)
        if cache is None or slots is None:
            self._pool_dirty.discard(req_id)
            return

        num_layers = len(cache)
        num_prefill_slots = len(slots)
        prefill_len = min(num_prefill_slots, cache[0].offset)

        # Skip the prefix portion that was loaded from the pool
        cached_prefix_len = self._req_prefix_len.get(req_id, 0)
        sync_start = cached_prefix_len
        sync_len = prefill_len - sync_start
        if sync_len > 0:
            sync_slots_mx = mx.array(slots[sync_start:prefill_len], dtype=mx.int32)
            # Cache layout: (1, n_kv_heads, S, head_dim)
            # Pool layout:  (pool_size, n_kv_heads, head_dim)
            # Transpose (n_kv_heads, S, head_dim) → (S, n_kv_heads, head_dim)
            k_all = mx.stack(
                [
                    cache[i].keys[0, :, sync_start:prefill_len, :].transpose(1, 0, 2)
                    for i in range(num_layers)
                ]
            )
            v_all = mx.stack(
                [
                    cache[i].values[0, :, sync_start:prefill_len, :].transpose(1, 0, 2)
                    for i in range(num_layers)
                ]
            )
            self._kv_pool.set_kv_all_layers(sync_slots_mx, k_all, v_all)

            mx.eval(*self._kv_pool.all_buffers())

        self._pool_dirty.discard(req_id)

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Run batched prefill for multiple requests.

        When all sequences have the same length and the same prefix
        match length, they are batched into a single forward pass.
        Otherwise falls back to serial prefill.

        Returns:
            List of (next_token_id, prefix_len) tuples per request.
        """
        if len(req_ids) == 1:
            return [self.prefill(req_ids[0], token_ids_list[0])]

        # Check if all sequences have the same length
        lengths = [len(tids) for tids in token_ids_list]
        if len(set(lengths)) != 1:
            return [
                self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)
            ]

        num_tokens = lengths[0]
        batch_size = len(req_ids)

        # Use serial prefills to avoid the expensive _extract_kv_cache
        # overhead that comes with batched forward passes.  On Apple Silicon,
        # serial BS=1 forwards are comparable in speed to a single BS=B
        # forward, and they avoid 28*B mx.contiguous copy operations needed
        # to split the batched cache into per-request caches.
        return [self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)]

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Decode using radix cache pool.

        Uses per-request contiguous caches for the model forward pass —
        identical to the legacy path.  Pool sync for decode tokens is
        deferred (the trie only indexes prefill tokens, so decode K/V
        is not needed in the pool during generation).
        """
        batch_size = len(req_ids)
        num_layers = self._num_layers

        # Retrieve per-request contiguous caches
        caches = [self._req_caches[rid] for rid in req_ids]
        seq_lens = [caches[i][0].offset for i in range(batch_size)]

        # --- Forward pass using contiguous caches (same as legacy) ---
        if batch_size == 1:
            # BS=1: native attention, no wrapper overhead
            cache = caches[0]
            last_token = self._req_token_ids[req_ids[0]][-1]
            input_ids = mx.array([[last_token]], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            self._eval_with_cache(next_tokens_mlx, cache)
        else:
            # BS>1: use BatchedDecodeContext (same as legacy batched decode)
            layer_caches = [
                [caches[i][layer_idx] for i in range(batch_size)]
                for layer_idx in range(num_layers)
            ]
            ctx = BatchedDecodeContext(
                batch_size=batch_size,
                seq_lens=seq_lens,
                layer_caches=layer_caches,
            )
            set_context(ctx)
            try:
                max_offset = max(seq_lens)
                shim_cache = [OffsetCache(offset=max_offset) for _ in range(num_layers)]
                last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
                batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
                model_output = self.model(batched_input, cache=shim_cache)
                logits = self._extract_logits(model_output)
                next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)

                eval_targets = [next_tokens_mlx]
                for c_list in caches:
                    for c in c_list:
                        eval_targets.append(c.keys)
                        eval_targets.append(c.values)
                mx.eval(*eval_targets)
            finally:
                clear_context()

        next_tokens = next_tokens_mlx.tolist()

        # Update per-request state (no pool sync needed during decode)
        for i, rid in enumerate(req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        return req_id in self._req_slot_ids or req_id in self._req_caches

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        if not self.disable_radix_cache:
            # Sync KV to pool so the trie can serve this prefix to future requests
            self._sync_request_to_pool(req_id)

            # Release the lock on the matched trie node
            last_node = self._req_last_node.pop(req_id, None)
            if last_node is not None:
                self._radix_trie.dec_ref(last_node)

        self._req_slot_ids.pop(req_id, None)
        self._req_token_ids.pop(req_id, None)
        cache = self._req_caches.pop(req_id, None)
        if cache is not None:
            self._release_cache(cache)
        self._pool_dirty.discard(req_id)
        self._req_prefix_len.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._req_slot_ids.clear()
        self._req_token_ids.clear()
        # Return all active caches to the pool
        for cache in self._req_caches.values():
            self._cache_pool.append(cache)
        self._req_caches.clear()
        self._pool_dirty.clear()
        self._req_last_node.clear()
        self._req_prefix_len.clear()
        if self._radix_trie is not None:
            freed = self._radix_trie.reset()
            if freed and self._kv_pool is not None:
                self._kv_pool.allocator.free(freed)
        if self._kv_pool is not None:
            self._kv_pool.clear()
