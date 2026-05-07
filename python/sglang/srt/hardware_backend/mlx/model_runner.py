"""MLX model runner for Apple Silicon.

Slot allocation and radix-trie prefix matching are handled by the
scheduler (``TokenToKVPoolAllocator`` / ``RadixCache``).  This runner
stores scheduler-visible KV in ``MlxPagedKVCache`` so the Metal attention
backend can read reused prefixes and decode state directly from paged blocks.
"""

import importlib
import logging
import platform
import sys
import time

import mlx.core as mx
import psutil
from mlx_lm import load as mlx_lm_load

from sglang.srt.hardware_backend.mlx.kv_cache import (
    MLXAttentionWrapper,
    OffsetCache,
    PagedAttentionContext,
    clear_paged_context,
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
    set_paged_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.paged_cache import MlxPagedKVCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

_REQUIRED_METAL_APIS = (
    "paged_kv_scatter",
    "prefill_attention_paged",
    "decode_attention_paged",
)


def _check_mlx_metal_backend_available() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX Metal backend requires Apple Silicon")
    try:
        metal = importlib.import_module("sgl_kernel.metal")
    except ImportError as exc:
        raise RuntimeError("MLX Metal backend requires sgl_kernel.metal") from exc
    missing_apis = [name for name in _REQUIRED_METAL_APIS if not hasattr(metal, name)]
    if missing_apis:
        raise RuntimeError(
            "MLX Metal backend requires sgl_kernel.metal APIs: "
            + ", ".join(missing_apis)
        )
    require_extension = getattr(metal, "_require_metal_extension", None)
    try:
        if require_extension is not None:
            require_extension()
        elif getattr(metal, "_metal", None) is None:
            raise ImportError("sgl_kernel._metal is not available")
    except ImportError as exc:
        raise RuntimeError(
            "MLX Metal backend requires the native sgl_kernel._metal extension"
        ) from exc


class MlxModelRunner:
    """MLX model runner with radix-cache prefix sharing."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
    ):
        _check_mlx_metal_backend_available()

        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static

        self._load_model()

        # Pin MLX allocations to prevent OS paging
        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        self._num_layers = get_num_layers(self.model)
        self._max_seq_len = 4096  # doubles on overflow

        self._req_token_ids: dict[str, list[int]] = {}

        self._paged_kv_cache: MlxPagedKVCache | None = None
        self._req_to_token_pool: ReqToTokenPool | None = None
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}

        self._pool_size = self._compute_pool_size(pool_size)
        self._paged_attention_block_size = 1

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    @staticmethod
    def _eval_with_cache(token_result: mx.array, cache: list[OffsetCache]) -> None:
        """Evaluate token result and cache protocol state in one mx.eval call."""
        mx.eval(token_result, *[s for c in cache for s in c.state])

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )
        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def _get_attn_config(self) -> tuple[int, int, mx.Dtype]:
        """Return (n_kv_heads, head_dim, dtype) from the model."""
        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            raise RuntimeError("Cannot determine attention config: no layers found")
        sample_attn = getattr(layer_list[0], attn_attr)
        if isinstance(sample_attn, MLXAttentionWrapper):
            sample_attn = sample_attn._inner
        n_kv_heads = sample_attn.n_kv_heads
        if hasattr(sample_attn, "head_dim"):
            head_dim = sample_attn.head_dim
        elif hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            head_dim = sample_attn.k_proj.weight.shape[0] // n_kv_heads
        else:
            raise RuntimeError("Cannot determine head_dim from attention module")
        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype
        return n_kv_heads, head_dim, dtype

    def _compute_pool_size(self, explicit_size: int | None) -> int:
        """Determine pool slot count (auto-size from available memory if needed)."""
        if explicit_size is not None:
            return explicit_size
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        num_layers = self._num_layers
        sys_available = psutil.virtual_memory().available
        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()
        mlx_usable = int(mlx_limit * self._mem_fraction_static)
        kv_budget = min(
            max(mlx_usable - mlx_used, 0),
            int(sys_available * self._mem_fraction_static),
        )
        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size
        pool_size = max(kv_budget // bytes_per_slot, 256)
        logger.info(
            f"Auto-sized KV pool: "
            f"sys_available={sys_available / (1024**3):.2f} GB, "
            f"mlx_limit={mlx_limit / (1024**3):.1f} GB, "
            f"mlx_used={mlx_used / (1024**3):.2f} GB, "
            f"kv_budget={kv_budget / (1024**3):.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}"
        )
        return pool_size

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def init_kv_pool(self, req_to_token_pool: ReqToTokenPool) -> None:
        """Create paged KV cache storage and wire scheduler pools."""
        self._req_to_token_pool = req_to_token_pool
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        block_size = self._get_paged_attention_block_size()
        capacity = self._pool_size + 1
        num_blocks = (capacity + block_size - 1) // block_size
        self._paged_kv_cache = MlxPagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=self._num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        logger.info(
            f"Paged KV cache initialized: pool_size={self._pool_size} "
            f"(capacity {self._paged_kv_cache.capacity} incl. padding slot 0), "
            f"block_size={block_size}, num_blocks={num_blocks}, "
            f"{self._num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def _make_cache_shim(self, offset: int) -> list[OffsetCache]:
        return [OffsetCache(offset=offset) for _ in range(self._num_layers)]

    def _require_paged_kv_scatter(self, ctx: PagedAttentionContext) -> None:
        if not ctx.has_scattered_all_layers(self._num_layers):
            raise RuntimeError("MLX paged attention did not scatter KV for all layers")

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> int:
        """Prefill a request.  Returns next_token_id."""
        if self._paged_kv_cache is None:
            raise RuntimeError("MLX runner requires paged KV cache for prefill")

        prefix_len = len(prefix_slot_ids)
        new_token_count = len(new_token_ids)
        if new_token_count > 0:
            extend_tokens = new_token_ids
            cache_offset = prefix_len
            synced_offset = prefix_len + new_token_count
        else:
            extend_tokens = full_token_ids[-1:]
            cache_offset = max(prefix_len - 1, 0)
            synced_offset = prefix_len

        paged_ctx = self._build_paged_prefill_context(prefix_slot_ids, new_slot_ids)
        if paged_ctx is None:
            raise RuntimeError("MLX runner requires paged KV cache for prefill")

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        cache = self._make_cache_shim(cache_offset)
        set_paged_context(paged_ctx)
        try:
            model_output = self.model(input_ids, cache=cache)
        finally:
            clear_paged_context()
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)
        self._eval_with_cache(next_token_mlx, cache)
        self._require_paged_kv_scatter(paged_ctx)
        next_token = int(next_token_mlx.item())

        self._req_token_ids[req_id] = list(full_token_ids) + [next_token]
        self._req_pool_idx[req_id] = req_pool_idx
        self._req_synced_offset[req_id] = synced_offset

        return next_token

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.  Returns next_token_id."""
        assert (
            req_id in self._req_token_ids
        ), f"extend called for unknown request {req_id}"

        paged_ctx = self._build_paged_extend_context(req_id, new_slot_ids)
        if paged_ctx is None:
            raise RuntimeError("MLX runner requires paged KV cache for extend")

        offset = int(self._req_synced_offset.get(req_id, 0))
        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        cache = self._make_cache_shim(offset)
        set_paged_context(paged_ctx)
        try:
            model_output = self.model(input_ids, cache=cache)
        finally:
            clear_paged_context()
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)
        self._eval_with_cache(next_token_mlx, cache)
        self._require_paged_kv_scatter(paged_ctx)
        next_token = int(next_token_mlx.item())

        prev_tokens = self._req_token_ids[req_id]
        if prev_tokens:
            prev_tokens.pop()
        prev_tokens.extend(new_token_ids)
        prev_tokens.append(next_token)
        self._req_synced_offset[req_id] = offset + len(new_slot_ids)

        return next_token

    def flush_all_decode_kv(self) -> None:
        """No-op because native paged attention scatters decode KV directly."""
        return

    def _get_paged_attention_block_size(self) -> int:
        paged_cache = getattr(self, "_paged_kv_cache", None)
        block_size = getattr(paged_cache, "block_size", None)
        if block_size is None:
            block_size = getattr(self, "_paged_attention_block_size", 1)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("paged attention block size must be positive")
        return block_size

    @staticmethod
    def _make_cu_seqlens(seq_lens: list[int]) -> list[int]:
        cu_seqlens = [0]
        for seq_len in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        return cu_seqlens

    def _build_block_tables(
        self, slot_rows: list[list[int]], context_lens: list[int]
    ) -> tuple[list[list[int]], int]:
        block_size = self._get_paged_attention_block_size()
        max_context_len = max(context_lens, default=0)
        max_blocks = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for row, context_len in zip(slot_rows, context_lens, strict=True):
            blocks = [
                row[start] // block_size for start in range(0, context_len, block_size)
            ]
            blocks.extend([0] * (max_blocks - len(blocks)))
            block_tables.append(blocks)
        return block_tables, max_context_len

    def _build_paged_prefill_context(
        self, prefix_slot_ids: list[int], new_slot_ids: list[int]
    ) -> PagedAttentionContext | None:
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None:
            return None
        prefix_slot_ids = list(prefix_slot_ids)
        new_slot_ids = list(new_slot_ids)
        if new_slot_ids:
            q_len = len(new_slot_ids)
            offset = len(prefix_slot_ids)
            context_len = offset + q_len
            slot_mapping = new_slot_ids
            radix_prefix_len = offset
            slot_row = prefix_slot_ids + new_slot_ids
        elif prefix_slot_ids:
            q_len = 1
            context_len = len(prefix_slot_ids)
            offset = context_len - 1
            slot_mapping = [prefix_slot_ids[-1]]
            radix_prefix_len = offset
            slot_row = prefix_slot_ids
        else:
            return None

        block_tables, max_context_len = self._build_block_tables(
            [slot_row], [context_len]
        )
        return PagedAttentionContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=[context_len],
            offsets=[offset],
            cu_seqlens=self._make_cu_seqlens([q_len]),
            max_seqlen_q=q_len,
            max_seqlen_k=max_context_len,
            radix_prefix_lens=[radix_prefix_len],
            kv_pool=paged_cache,
        )

    def _build_paged_extend_context(
        self, req_id: str, new_slot_ids: list[int]
    ) -> PagedAttentionContext | None:
        req_to_token_pool = getattr(self, "_req_to_token_pool", None)
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or req_to_token_pool is None or not new_slot_ids:
            return None
        req_pool_idx = self._req_pool_idx.get(req_id)
        if req_pool_idx is None:
            return None

        q_len = len(new_slot_ids)
        offset = int(self._req_synced_offset.get(req_id, 0))
        context_len = offset + q_len
        slot_rows = req_to_token_pool.req_to_token[[req_pool_idx], :context_len].to(
            dtype=int
        )
        slot_rows = slot_rows.tolist()
        block_tables, max_context_len = self._build_block_tables(
            slot_rows, [context_len]
        )

        return PagedAttentionContext(
            is_prefill=True,
            slot_mapping=list(new_slot_ids),
            block_tables=block_tables,
            context_lens=[context_len],
            offsets=[offset],
            cu_seqlens=self._make_cu_seqlens([q_len]),
            max_seqlen_q=q_len,
            max_seqlen_k=max_context_len,
            radix_prefix_lens=[offset],
            kv_pool=paged_cache,
        )

    def _build_paged_decode_context(
        self, req_ids: list[str], seq_lens: list[int]
    ) -> PagedAttentionContext | None:
        req_to_token_pool = getattr(self, "_req_to_token_pool", None)
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or req_to_token_pool is None or not req_ids:
            return None

        max_context_len = max(seq_lens) + 1
        req_pool_indices = [self._req_pool_idx[rid] for rid in req_ids]
        slot_rows = req_to_token_pool.req_to_token[
            req_pool_indices, :max_context_len
        ].to(dtype=int)
        slot_rows = slot_rows.tolist()

        context_lens = [seq_len + 1 for seq_len in seq_lens]
        slot_mapping = [
            row[seq_len] for row, seq_len in zip(slot_rows, seq_lens, strict=True)
        ]
        block_tables, max_context_len = self._build_block_tables(
            slot_rows, context_lens
        )

        synced_offsets = getattr(self, "_req_synced_offset", {})
        radix_prefix_lens = [
            min(int(synced_offsets.get(rid, 0)), seq_len)
            for rid, seq_len in zip(req_ids, seq_lens, strict=True)
        ]

        return PagedAttentionContext(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            offsets=seq_lens,
            cu_seqlens=self._make_cu_seqlens([1] * len(req_ids)),
            max_seqlen_q=1,
            max_seqlen_k=max_context_len,
            radix_prefix_lens=radix_prefix_lens,
            kv_pool=paged_cache,
        )

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Decode one token per request."""
        if not req_ids:
            return []

        seq_lens = [len(self._req_token_ids[rid]) - 1 for rid in req_ids]
        paged_ctx = self._build_paged_decode_context(req_ids, seq_lens)
        if paged_ctx is None:
            raise RuntimeError("MLX runner requires paged KV cache for decode")

        max_offset = max(seq_lens)
        cache = self._make_cache_shim(max_offset)
        last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        set_paged_context(paged_ctx)
        try:
            model_output = self.model(batched_input, cache=cache)
        finally:
            clear_paged_context()
        logits = self._extract_logits(model_output)
        next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
        self._eval_with_cache(next_tokens_mlx, cache)
        self._require_paged_kv_scatter(paged_ctx)

        next_tokens = next_tokens_mlx.tolist()
        for i, rid in enumerate(req_ids):
            self._req_token_ids[rid].append(next_tokens[i])
            self._req_synced_offset[rid] = seq_lens[i] + 1

        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        return req_id in self._req_token_ids

    def remove_request(self, req_id: str):
        """Release request state."""
        self._req_token_ids.pop(req_id, None)
        self._req_pool_idx.pop(req_id, None)
        self._req_synced_offset.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._req_token_ids.clear()
        self._req_pool_idx.clear()
        self._req_synced_offset.clear()
        if self._paged_kv_cache is not None:
            self._paged_kv_cache.clear()
