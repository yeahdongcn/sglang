"""MLX model runner for Apple Silicon.

Slot allocation and radix-trie prefix matching are handled by the
scheduler (``TokenToKVPoolAllocator`` / ``RadixCache``).  This runner
stores scheduler-visible KV in ``MlxPagedKVCache`` so the Metal attention
backend can read reused prefixes and decode state directly from paged blocks.
"""

import importlib
import logging
import os
import platform
import sys
import time

import psutil
from mlx_lm import load as mlx_lm_load

import mlx.core as mx
from sglang.srt.hardware_backend.mlx.kv_cache import (
    BatchedDecodeContext,
    ContiguousKVCache,
    MLXAttentionWrapper,
    OffsetCache,
    PagedAttentionContext,
    PoolBackedCache,
    clear_context,
    clear_paged_context,
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
    set_context,
    set_paged_context,
    unpatch_model_attention,
)
from sglang.srt.hardware_backend.mlx.kv_cache.paged_cache import MlxPagedKVCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

_REQUIRED_METAL_APIS = (
    "paged_kv_scatter",
    "prefill_attention_paged",
    "decode_attention_paged",
    "decode_attention_paged_lazy_unchecked",
    "decode_attention_paged_p1_lazy_unchecked",
    "decode_attention_paged_unchecked",
)

_DEFAULT_PAGED_ATTENTION_BLOCK_SIZE = 1
_RADIX_RECOMPUTE_PREFIX_MAX_RATIO = 0.5
_RADIX_RECOMPUTE_PREFIX_MAX_TOKENS = 128
_RADIX_PAGED_PREFILL_MIN_NEW_TOKENS = 384
_RADIX_PAGED_PREFILL_MIN_PREFIX_TOKENS = 384
_RADIX_PAGED_PREFILL_LONG_SUFFIX_TOKENS = 768
_RADIX_RANGE_GATHER_MAX_PREFIX_TOKENS = 1024
_RADIX_MULTI_RANGE_GATHER_MAX_RANGES = 4
_RADIX_FOLDED_RANGE_SYNC_MAX_TOKENS = 64
_RADIX_RIGHT_SIZE_POOL_CACHE_MAX_PREFIX_TOKENS = 1024
_RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS = 384
_RADIX_FULL_STATE_DECODE_MIN_TOKENS = 1024
_RADIX_BATCHED_WRAPPER_B1_DECODE_MAX_TOKENS = 512
_RADIX_BATCHED_WRAPPER_B1_FULL_STATE_MAX_CAPACITY = 512
_RADIX_IDLE_PAGED_PREFILL_AFTER_S = 1.0
_RADIX_IDLE_PAGED_PREFILL_MAX_NEW_TOKENS = 16
_RADIX_IDLE_PAGED_PREFILL_MIN_PREFIX_TOKENS = 128
_CONTIGUOUS_CACHE_MIN_CAPACITY = 16
_CONTIGUOUS_CACHE_DECODE_SLACK = 32
_PROFILE_TIMING_ENV = "SGLANG_MLX_PROFILE_TIMING"
_RADIX_KERNEL_WARMUP_ENV = "SGLANG_MLX_DISABLE_RADIX_KERNEL_WARMUP"
_RADIX_IDLE_PAGED_PREFILL_ENABLE_ENV = "SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL"
_RADIX_BF16_PAGED_DECODE_ENABLE_ENV = "SGLANG_MLX_ENABLE_BF16_PAGED_DECODE"
_RADIX_BF16_METAL_SCATTER_ENABLE_ENV = "SGLANG_MLX_ENABLE_BF16_METAL_SCATTER"
_NO_RADIX_SINGLE_PREFILL_MIN_CAPACITY = 384
_RADIX_KERNEL_WARMUP_PREFIX_TOKENS = 128


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
        page_size: int | None = None,
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

        # Radix runs need the paged wrapper for the large-suffix prefix-hit fast
        # path. No-radix installs the lean batched wrapper only around batched
        # decode so prefill and single-request decode stay on the raw mlx-lm
        # modules.
        if not disable_radix_cache:
            patch_model_attention(self.model, enable_paged=True)

        self._num_layers = get_num_layers(self.model)
        self._max_seq_len = 4096  # doubles on overflow

        self._req_caches: dict[str, list[ContiguousKVCache | PoolBackedCache]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._cache_pool: list[list[ContiguousKVCache]] = []
        self._no_radix_batched_wrapper_installed = False

        self._paged_kv_cache: MlxPagedKVCache | None = None
        self._req_to_token_pool: ReqToTokenPool | None = None
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}
        self._last_idle_at: float | None = None
        self._last_forward_done_at: float | None = None

        self._paged_attention_block_size = self._normalize_page_size(page_size)
        self._pool_size = self._align_pool_size_to_pages(
            self._compute_pool_size(pool_size)
        )

    @staticmethod
    def _normalize_page_size(page_size: int | None) -> int:
        if page_size is None:
            return _DEFAULT_PAGED_ATTENTION_BLOCK_SIZE
        page_size = int(page_size)
        if page_size <= 0:
            raise ValueError("MLX paged attention page_size must be positive")
        return page_size

    def _align_pool_size_to_pages(self, pool_size: int) -> int:
        """Keep scheduler-visible MLX pool capacity page-accountable."""
        pool_size = int(pool_size)
        block_size = int(getattr(self, "_paged_attention_block_size", 1))
        if block_size <= 1:
            return pool_size
        if pool_size < block_size:
            aligned = block_size
        else:
            aligned = pool_size - (pool_size % block_size)
        if aligned != pool_size:
            logger.info(
                "Aligning MLX KV pool_size from %s to %s for page_size=%s",
                pool_size,
                aligned,
                block_size,
            )
        return aligned

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    @staticmethod
    def _round_cache_capacity(required: int) -> int:
        required = max(int(required), _CONTIGUOUS_CACHE_MIN_CAPACITY)
        alignment = 32
        return ((required + alignment - 1) // alignment) * alignment

    def _acquire_cache(
        self, initial_capacity: int | None = None
    ) -> list[ContiguousKVCache]:
        """Get a reusable contiguous cache list, or create one."""
        cache_pool = getattr(self, "_cache_pool", None)
        if cache_pool is None:
            self._cache_pool = []
            cache_pool = self._cache_pool
        max_seq_len = (
            int(getattr(self, "_max_seq_len", 4096))
            if initial_capacity is None
            else self._round_cache_capacity(initial_capacity)
        )
        if not self.disable_radix_cache:
            max_seq_len = max(max_seq_len, int(getattr(self, "_max_seq_len", 4096)))
        cache_idx = None
        for idx in range(len(cache_pool) - 1, -1, -1):
            candidate = cache_pool[idx]
            existing_capacity = max(
                (
                    int(item.keys.shape[2])
                    if item.keys is not None
                    else int(item.max_seq_len)
                )
                for item in candidate
            )
            if (
                existing_capacity >= max_seq_len
                and existing_capacity <= max_seq_len * 2
            ):
                cache_idx = idx
                break
        if cache_idx is not None:
            cache = cache_pool.pop(cache_idx)
            for item in cache:
                item.offset = 0
                item.max_seq_len = max_seq_len
            return cache
        return [
            ContiguousKVCache(max_seq_len=max_seq_len) for _ in range(self._num_layers)
        ]

    def _release_cache(self, cache: list[ContiguousKVCache]) -> None:
        cache_pool = getattr(self, "_cache_pool", None)
        if cache_pool is None:
            self._cache_pool = []
            cache_pool = self._cache_pool
        if not self.disable_radix_cache:
            max_capacity = max(
                (
                    int(item.keys.shape[2])
                    if item.keys is not None
                    else int(item.max_seq_len)
                )
                for item in cache
            )
            if max_capacity < int(getattr(self, "_max_seq_len", 4096)):
                return
        cache_pool.append(cache)

    def _ensure_no_radix_batched_wrapper(self) -> None:
        if getattr(self, "_no_radix_batched_wrapper_installed", False):
            return
        patch_model_attention(self.model, enable_paged=False)
        self._no_radix_batched_wrapper_installed = True

    def _remove_no_radix_batched_wrapper(self) -> None:
        if not getattr(self, "_no_radix_batched_wrapper_installed", False):
            return
        if self.disable_radix_cache:
            unpatch_model_attention(self.model)
        else:
            patch_model_attention(self.model, enable_paged=True)
        self._no_radix_batched_wrapper_installed = False

    def _ensure_radix_paged_wrapper(self) -> None:
        if self.disable_radix_cache or not getattr(
            self, "_no_radix_batched_wrapper_installed", False
        ):
            return
        patch_model_attention(self.model, enable_paged=True)
        self._no_radix_batched_wrapper_installed = False

    @staticmethod
    def _cache_state_for_eval(
        item: OffsetCache | ContiguousKVCache | PoolBackedCache,
        *,
        compact: bool = False,
    ) -> tuple[mx.array, ...]:
        if compact and isinstance(item, ContiguousKVCache) and item.keys is not None:
            return item.get_kv()
        return item.state

    @staticmethod
    def _eval_with_cache(
        token_result: mx.array,
        cache: list[OffsetCache | ContiguousKVCache | PoolBackedCache],
        *,
        compact: bool = False,
        extra: list[mx.array] | None = None,
    ) -> None:
        """Evaluate token result and cache protocol state in one mx.eval call."""
        extra = extra or []
        if not compact:
            mx.eval(
                token_result,
                *[state for item in cache for state in item.state],
                *extra,
            )
            return
        states = []
        for item in cache:
            states.extend(MlxModelRunner._cache_state_for_eval(item, compact=compact))
        mx.eval(token_result, *states, *extra)

    def _require_cache_range(
        self,
        cache: list[ContiguousKVCache],
        end: int,
        action: str,
    ) -> None:
        """Validate that the model populated K/V through ``end`` tokens."""
        for layer_idx, layer_cache in enumerate(cache):
            if (
                layer_cache.keys is None
                or layer_cache.values is None
                or layer_cache.offset < end
            ):
                raise RuntimeError(
                    "MLX runner requires populated contiguous KV cache for "
                    f"{action} (layer {layer_idx})"
                )

    def _materialize_pool_backed_cache(
        self,
        cache: list[PoolBackedCache],
        *,
        right_size: bool = False,
    ) -> list[ContiguousKVCache]:
        """Convert one forward pass of pool-backed cache state to contiguous."""
        token_count = 0
        for pool_cache in cache:
            if pool_cache._full_keys is not None:
                token_count = max(token_count, int(pool_cache._full_keys.shape[2]))
        if right_size:
            capacity = self._round_cache_capacity(
                token_count + _CONTIGUOUS_CACHE_DECODE_SLACK
            )
            contiguous_cache = [ContiguousKVCache(max_seq_len=capacity) for _ in cache]
        else:
            contiguous_cache = self._acquire_cache(
                token_count + _CONTIGUOUS_CACHE_DECODE_SLACK
            )
            capacity = max(
                (
                    int(item.keys.shape[2])
                    if item.keys is not None
                    else int(item.max_seq_len)
                )
                for item in contiguous_cache
            )
        for layer_idx, pool_cache in enumerate(cache):
            if pool_cache._full_keys is None or pool_cache._full_values is None:
                raise RuntimeError(
                    "MLX runner requires populated pool-backed KV cache for "
                    f"prefill (layer {layer_idx})"
                )
            if not right_size:
                contiguous_cache[layer_idx].update_and_fetch(
                    pool_cache._full_keys, pool_cache._full_values
                )
                continue
            token_count = int(pool_cache._full_keys.shape[2])
            pad = capacity - token_count
            if pad > 0:
                pad_shape = (
                    pool_cache._full_keys.shape[0],
                    pool_cache._full_keys.shape[1],
                    pad,
                    pool_cache._full_keys.shape[3],
                )
                contiguous_cache[layer_idx].keys = mx.concatenate(
                    [
                        pool_cache._full_keys,
                        mx.zeros(pad_shape, dtype=pool_cache._full_keys.dtype),
                    ],
                    axis=2,
                )
                contiguous_cache[layer_idx].values = mx.concatenate(
                    [
                        pool_cache._full_values,
                        mx.zeros(pad_shape, dtype=pool_cache._full_values.dtype),
                    ],
                    axis=2,
                )
            else:
                contiguous_cache[layer_idx].keys = pool_cache._full_keys
                contiguous_cache[layer_idx].values = pool_cache._full_values
            contiguous_cache[layer_idx].offset = token_count
        return contiguous_cache

    def _materialize_paged_slots_to_contiguous(
        self,
        slot_ids: list[int],
    ) -> list[ContiguousKVCache]:
        """Gather scheduler-visible paged slots into a request-local cache."""
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None:
            raise RuntimeError("MLX runner requires paged KV cache for materialize")
        if not slot_ids:
            raise RuntimeError("MLX runner requires slots for paged materialize")

        slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)
        token_count = len(slot_ids)
        slot_range = self._slot_range_for_slots(slot_ids)
        slot_ranges = (
            self._slot_ranges_for_slots(
                slot_ids, max_ranges=_RADIX_MULTI_RANGE_GATHER_MAX_RANGES
            )
            if slot_range is None
            else None
        )
        contiguous_cache = self._acquire_cache(
            token_count + _CONTIGUOUS_CACHE_DECODE_SLACK
        )
        for layer_idx in range(self._num_layers):
            if slot_range is not None:
                k_cached, v_cached = paged_cache.get_kv_slot_range(
                    layer_idx, *slot_range
                )
            elif slot_ranges is not None and hasattr(paged_cache, "get_kv_slot_ranges"):
                k_cached, v_cached = paged_cache.get_kv_slot_ranges(
                    layer_idx, slot_ranges
                )
            else:
                get_kv = getattr(paged_cache, "get_kv_unchecked", paged_cache.get_kv)
                k_cached, v_cached = get_kv(layer_idx, slot_ids_mx)
            contiguous_cache[layer_idx].update_and_fetch(
                mx.contiguous(k_cached.transpose(1, 0, 2)[None]),
                mx.contiguous(v_cached.transpose(1, 0, 2)[None]),
            )
        return contiguous_cache

    @staticmethod
    def _should_recompute_small_radix_prefix(
        prefix_len: int, new_token_count: int
    ) -> bool:
        if prefix_len <= 0 or new_token_count <= 0:
            return False
        return prefix_len <= _RADIX_RECOMPUTE_PREFIX_MAX_TOKENS and prefix_len <= int(
            new_token_count * _RADIX_RECOMPUTE_PREFIX_MAX_RATIO
        )

    @staticmethod
    def _should_use_paged_radix_prefill(prefix_len: int, new_token_count: int) -> bool:
        if prefix_len <= 0 or new_token_count <= 0:
            return False
        if new_token_count >= _RADIX_PAGED_PREFILL_LONG_SUFFIX_TOKENS:
            return True
        return (
            prefix_len >= _RADIX_PAGED_PREFILL_MIN_PREFIX_TOKENS
            and new_token_count >= _RADIX_PAGED_PREFILL_MIN_NEW_TOKENS
        )

    def mark_idle(self) -> None:
        """Record a scheduler-idle boundary for radix prefill routing."""
        self._last_idle_at = time.perf_counter()

    def _mark_forward_done(self) -> None:
        self._last_forward_done_at = time.perf_counter()

    def _should_use_idle_paged_radix_prefill(
        self, prefix_len: int, new_token_count: int
    ) -> bool:
        if os.environ.get(_RADIX_IDLE_PAGED_PREFILL_ENABLE_ENV, "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        if prefix_len < _RADIX_IDLE_PAGED_PREFILL_MIN_PREFIX_TOKENS:
            return False
        if (
            new_token_count <= 0
            or new_token_count > _RADIX_IDLE_PAGED_PREFILL_MAX_NEW_TOKENS
        ):
            return False
        last_idle_at = getattr(self, "_last_idle_at", None)
        last_forward_done_at = getattr(self, "_last_forward_done_at", None)
        last_activity_at = max(
            (at for at in (last_idle_at, last_forward_done_at) if at is not None),
            default=None,
        )
        if last_activity_at is None:
            return False
        return (
            time.perf_counter() - last_activity_at
        ) >= _RADIX_IDLE_PAGED_PREFILL_AFTER_S

    def _full_block_ids_for_slots(self, slot_ids: list[int]) -> mx.array | None:
        block_size = self._get_paged_attention_block_size()
        if block_size <= 1 or not slot_ids or len(slot_ids) % block_size != 0:
            return None
        block_ids = []
        for start in range(0, len(slot_ids), block_size):
            block_start = int(slot_ids[start])
            if block_start % block_size != 0:
                return None
            for offset, slot_id in enumerate(slot_ids[start : start + block_size]):
                if int(slot_id) != block_start + offset:
                    return None
            block_ids.append(block_start // block_size)
        return mx.array(block_ids, dtype=mx.int32)

    def _slot_range_for_slots(self, slot_ids: list[int]) -> tuple[int, int] | None:
        if self._get_paged_attention_block_size() != 1 or not slot_ids:
            return None
        start = int(slot_ids[0])
        for offset, slot_id in enumerate(slot_ids):
            if int(slot_id) != start + offset:
                return None
        return start, start + len(slot_ids)

    def _slot_ranges_for_slots(
        self, slot_ids: list[int], *, max_ranges: int
    ) -> list[tuple[int, int]] | None:
        if (
            self._get_paged_attention_block_size() != 1
            or not slot_ids
            or max_ranges <= 0
        ):
            return None
        ranges: list[tuple[int, int]] = []
        start = int(slot_ids[0])
        prev = start
        for slot_id in slot_ids[1:]:
            slot = int(slot_id)
            if slot == prev + 1:
                prev = slot
                continue
            ranges.append((start, prev + 1))
            if len(ranges) >= max_ranges:
                return None
            start = slot
            prev = slot
        ranges.append((start, prev + 1))
        if len(ranges) > max_ranges:
            return None
        return ranges

    @staticmethod
    def _profile_timing_enabled() -> bool:
        return os.environ.get(_PROFILE_TIMING_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _paged_cache_supports_metal(cache) -> bool:
        return getattr(cache, "dtype", mx.float16) in (mx.float16, mx.float32)

    @staticmethod
    def _paged_cache_supports_metal_scatter(cache) -> bool:
        cache_dtype = getattr(cache, "dtype", mx.float16)
        if cache_dtype in (mx.float16, mx.float32):
            return True
        if cache_dtype != mx.bfloat16:
            return False
        return os.environ.get(_RADIX_BF16_METAL_SCATTER_ENABLE_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _paged_cache_supports_idle_prefill(cache) -> bool:
        return MlxModelRunner._paged_cache_supports_metal(cache) or (
            getattr(cache, "block_size", 1) == 1
        )

    def _should_use_paged_radix_decode(
        self, batch_size: int, seq_lens: list[int]
    ) -> bool:
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or getattr(self, "_req_to_token_pool", None) is None:
            return False
        if getattr(paged_cache, "block_size", None) != 1:
            return False
        cache_dtype = getattr(paged_cache, "dtype", None)
        if cache_dtype not in (mx.float16, mx.bfloat16):
            return False
        if cache_dtype == mx.bfloat16 and os.environ.get(
            _RADIX_BF16_PAGED_DECODE_ENABLE_ENV, ""
        ).lower() not in {"1", "true", "yes", "on"}:
            return False
        if getattr(paged_cache, "head_dim", None) != 128:
            return False
        if batch_size <= 0 or not seq_lens:
            return False
        # The raw p1 lazy decode microbench regresses for tiny B1 contexts.
        return batch_size > 1 or max(seq_lens) + 1 > 16

    def _can_sync_or_use_paged_decode_state(
        self,
        req_id: str,
        cache: list[ContiguousKVCache],
        seq_len: int,
    ) -> bool:
        cache_offset = int(cache[0].offset)
        synced_offset = int(self._req_synced_offset.get(req_id, 0))
        if cache_offset >= seq_len:
            self._sync_pending_decode_kv_to_pool(req_id, cache)
            return int(self._req_synced_offset.get(req_id, 0)) >= seq_len
        return synced_offset >= seq_len

    def _materialize_synced_paged_cache_if_needed(
        self, req_id: str, cache: list[ContiguousKVCache]
    ) -> list[ContiguousKVCache]:
        synced = int(self._req_synced_offset.get(req_id, 0))
        if synced <= 0 or int(cache[0].offset) >= synced:
            return cache
        req_pool_idx = self._req_pool_idx.get(req_id)
        req_to_token_pool = getattr(self, "_req_to_token_pool", None)
        if req_pool_idx is None or req_to_token_pool is None:
            raise RuntimeError("MLX runner requires scheduler slots to materialize KV")
        slot_ids = (
            req_to_token_pool.req_to_token[req_pool_idx, :synced].to(dtype=int).tolist()
        )
        materialized = self._materialize_paged_slots_to_contiguous(slot_ids)
        self._release_cache(cache)
        self._req_caches[req_id] = materialized
        return materialized

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

    def _get_num_attention_heads(self, default: int) -> int:
        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            return default
        sample_attn = getattr(layer_list[0], attn_attr)
        if isinstance(sample_attn, MLXAttentionWrapper):
            sample_attn = sample_attn._inner
        return int(getattr(sample_attn, "n_heads", default))

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
        if self.disable_radix_cache:
            return
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
            normalize_dtype=block_size != 1,
        )
        logger.info(
            f"Paged KV cache initialized: pool_size={self._pool_size} "
            f"(capacity {self._paged_kv_cache.capacity} incl. padding slot 0), "
            f"block_size={block_size}, num_blocks={num_blocks}, "
            f"{self._num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )
        self._warm_radix_metal_kernels(n_kv_heads, head_dim)

    @staticmethod
    def _radix_kernel_warmup_enabled() -> bool:
        return os.environ.get(_RADIX_KERNEL_WARMUP_ENV, "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _warm_radix_metal_kernels(self, n_kv_heads: int, head_dim: int) -> None:
        """Precompile radix scatter/prefix kernels before the first user request."""
        paged_cache = self._paged_kv_cache
        if (
            paged_cache is None
            or not self._radix_kernel_warmup_enabled()
            or paged_cache.capacity < 2
            or not self._paged_cache_supports_metal_scatter(paged_cache)
        ):
            return

        try:
            cache_dtype = paged_cache.dtype
            n_q_heads = max(
                int(n_kv_heads),
                self._get_num_attention_heads(default=int(n_kv_heads)),
            )
            q = mx.zeros((1, n_q_heads, head_dim), dtype=cache_dtype)
            k = mx.zeros((1, n_kv_heads, head_dim), dtype=cache_dtype)
            v = mx.zeros((1, n_kv_heads, head_dim), dtype=cache_dtype)

            slot = mx.array([1], dtype=mx.int32)
            paged_cache.set_kv_all_layers(
                slot,
                [k] * self._num_layers,
                [v] * self._num_layers,
                eager=True,
                use_metal=True,
            )

            if not self._paged_cache_supports_metal(paged_cache):
                mx.synchronize()
                logger.info(
                    "Warmed MLX radix Metal scatter kernel: q_heads=%s "
                    "kv_heads=%s head_dim=%s dtype=%s",
                    n_q_heads,
                    n_kv_heads,
                    head_dim,
                    cache_dtype,
                )
                return

            from sgl_kernel.metal import prefill_attention_paged

            prefix_len = min(
                _RADIX_KERNEL_WARMUP_PREFIX_TOKENS,
                max(int(paged_cache.capacity) - 2, 0),
            )
            if prefix_len <= 0:
                return
            slot_row = list(range(1, prefix_len + 2))
            block_tables, _ = self._build_block_tables([slot_row], [prefix_len + 1])
            out = prefill_attention_paged(
                q,
                k,
                v,
                paged_cache.k_buffer[0],
                paged_cache.v_buffer[0],
                mx.array(block_tables, dtype=mx.int32),
                mx.array([prefix_len], dtype=mx.int32),
                mx.array([0, 1], dtype=mx.int32),
                head_dim**-0.5,
                causal=True,
            )
            mx.eval(out)
            mx.synchronize()
            logger.info(
                "Warmed MLX radix Metal kernels: prefix_len=%s q_heads=%s "
                "kv_heads=%s head_dim=%s",
                prefix_len,
                n_q_heads,
                n_kv_heads,
                head_dim,
            )
        except Exception:
            logger.warning("Failed to warm MLX radix Metal kernels", exc_info=True)

    def _make_cache_shim(self, offset: int) -> list[OffsetCache]:
        return [OffsetCache(offset=offset) for _ in range(self._num_layers)]

    def _require_paged_kv_scatter(self, ctx: PagedAttentionContext) -> None:
        if not ctx.has_scattered_all_layers(self._num_layers):
            raise RuntimeError("MLX paged attention did not scatter KV for all layers")

    def _sync_new_kv_to_pool(
        self,
        cache: list[ContiguousKVCache],
        cache_start: int,
        slot_ids: list[int],
        *,
        eager: bool = True,
    ) -> None:
        """Sync a contiguous-cache token range into the radix-visible paged pool."""
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or not slot_ids:
            return

        cache_start = int(cache_start)
        end = cache_start + len(slot_ids)
        self._require_cache_range(cache, end, "radix sync")
        slot_range = self._slot_range_for_slots(slot_ids)
        k_all = [
            mx.contiguous(
                cache[layer_idx].keys[0, :, cache_start:end, :].transpose(1, 0, 2)
            )
            for layer_idx in range(self._num_layers)
        ]
        v_all = [
            mx.contiguous(
                cache[layer_idx].values[0, :, cache_start:end, :].transpose(1, 0, 2)
            )
            for layer_idx in range(self._num_layers)
        ]
        if slot_range is not None:
            paged_cache.set_kv_all_layers_slot_range(
                slot_range[0], slot_range[1], k_all, v_all, eager=eager
            )
        else:
            slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)
            paged_cache.set_kv_all_layers(
                slot_ids_mx,
                k_all,
                v_all,
                eager=eager,
                use_metal=self._paged_cache_supports_metal_scatter(paged_cache),
            )

    def _prepare_new_kv_range_sync_to_pool(
        self,
        cache: list[ContiguousKVCache],
        cache_start: int,
        slot_ids: list[int],
    ) -> list[mx.array] | None:
        """Stage a p1 contiguous side-store sync for the caller's mx.eval.

        Returns destination slices that must be included in the same evaluation
        as the token/cache result.  Non-contiguous slots return ``None`` so the
        caller can keep the existing eager Metal scatter path.
        """
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or not slot_ids:
            return []

        slot_range = self._slot_range_for_slots(slot_ids)
        if slot_range is None:
            return None

        cache_start = int(cache_start)
        end = cache_start + len(slot_ids)
        self._require_cache_range(cache, end, "radix sync")
        k_all = [
            mx.contiguous(
                cache[layer_idx].keys[0, :, cache_start:end, :].transpose(1, 0, 2)
            )
            for layer_idx in range(self._num_layers)
        ]
        v_all = [
            mx.contiguous(
                cache[layer_idx].values[0, :, cache_start:end, :].transpose(1, 0, 2)
            )
            for layer_idx in range(self._num_layers)
        ]
        start, stop = slot_range
        paged_cache.set_kv_all_layers_slot_range(start, stop, k_all, v_all, eager=False)
        refs: list[mx.array] = []
        for layer_idx in range(self._num_layers):
            refs.extend(
                [
                    paged_cache.k_buffer[layer_idx][start:stop, 0],
                    paged_cache.v_buffer[layer_idx][start:stop, 0],
                ]
            )
        return refs

    def _sync_decode_kv_to_pool_batch(
        self,
        caches: list[list[ContiguousKVCache]],
        cache_starts: list[int],
        slot_ids: list[int],
    ) -> None:
        """Sync one newly decoded token per request into the paged pool."""
        paged_cache = getattr(self, "_paged_kv_cache", None)
        if paged_cache is None or not slot_ids:
            return

        for req_cache, start in zip(caches, cache_starts, strict=True):
            self._require_cache_range(req_cache, int(start) + 1, "decode radix sync")

        slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)
        k_layers = []
        v_layers = []
        if len(caches) == 1:
            cache = caches[0]
            start = int(cache_starts[0])
            for layer_idx in range(self._num_layers):
                k_layers.append(
                    mx.contiguous(
                        cache[layer_idx]
                        .keys[0, :, start : start + 1, :]
                        .transpose(1, 0, 2)
                    )
                )
                v_layers.append(
                    mx.contiguous(
                        cache[layer_idx]
                        .values[0, :, start : start + 1, :]
                        .transpose(1, 0, 2)
                    )
                )
        else:
            for layer_idx in range(self._num_layers):
                k_tokens = []
                v_tokens = []
                for req_cache, start in zip(caches, cache_starts, strict=True):
                    start = int(start)
                    k_tokens.append(
                        req_cache[layer_idx]
                        .keys[0, :, start : start + 1, :]
                        .transpose(1, 0, 2)
                    )
                    v_tokens.append(
                        req_cache[layer_idx]
                        .values[0, :, start : start + 1, :]
                        .transpose(1, 0, 2)
                    )
                k_layers.append(mx.contiguous(mx.concatenate(k_tokens, axis=0)))
                v_layers.append(mx.contiguous(mx.concatenate(v_tokens, axis=0)))
        paged_cache.set_kv_all_layers(
            slot_ids_mx,
            k_layers,
            v_layers,
            eager=True,
            use_metal=self._paged_cache_supports_metal_scatter(paged_cache),
        )

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
        profile_timing = self._profile_timing_enabled()
        profile_start = time.perf_counter() if profile_timing else 0.0
        if self.disable_radix_cache:
            self._remove_no_radix_batched_wrapper()
            if not hasattr(self, "_req_caches"):
                self._req_caches = {}
            cache = self._acquire_cache(
                max(
                    len(new_token_ids) + _CONTIGUOUS_CACHE_DECODE_SLACK,
                    _NO_RADIX_SINGLE_PREFILL_MIN_CAPACITY,
                )
            )
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            model_start = time.perf_counter() if profile_timing else 0.0
            model_output = self.model(input_ids, cache=cache)
            model_done = time.perf_counter() if profile_timing else 0.0
            logits = self._extract_logits(model_output)
            next_token_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            eval_start = time.perf_counter() if profile_timing else 0.0
            self._eval_with_cache(next_token_mlx, cache)
            eval_done = time.perf_counter() if profile_timing else 0.0
            next_token = int(next_token_mlx.item())

            self._req_token_ids[req_id] = list(full_token_ids) + [next_token]
            self._req_caches[req_id] = cache
            self._req_pool_idx[req_id] = req_pool_idx
            self._req_synced_offset[req_id] = 0
            if profile_timing:
                logger.info(
                    "MLX prefill timing no_radix rid=%s new=%s "
                    "model_ms=%.2f eval_ms=%.2f total_ms=%.2f",
                    req_id,
                    len(new_token_ids),
                    (model_done - model_start) * 1000,
                    (eval_done - eval_start) * 1000,
                    (eval_done - profile_start) * 1000,
                )
            self._mark_forward_done()
            return next_token

        if self._paged_kv_cache is None:
            raise RuntimeError("MLX runner requires paged KV cache for prefill")

        if not hasattr(self, "_req_caches"):
            self._req_caches = {}
        prefix_len = len(prefix_slot_ids)
        new_token_count = len(new_token_ids)
        if new_token_count > 0:
            extend_tokens = new_token_ids
            synced_offset = prefix_len + new_token_count
        else:
            extend_tokens = full_token_ids[-1:]
            synced_offset = prefix_len

        recompute_prefix = self._should_recompute_small_radix_prefix(
            prefix_len, new_token_count
        )
        use_paged_prefill = (
            not recompute_prefix
            and self._should_use_paged_radix_prefill(prefix_len, new_token_count)
        )
        if use_paged_prefill and not self._paged_cache_supports_metal(
            self._paged_kv_cache
        ):
            use_paged_prefill = False
        use_idle_paged_prefill = False
        if (
            not recompute_prefix
            and not use_paged_prefill
            and self._paged_cache_supports_idle_prefill(self._paged_kv_cache)
            and self._should_use_idle_paged_radix_prefill(prefix_len, new_token_count)
        ):
            use_paged_prefill = True
            use_idle_paged_prefill = True
        if recompute_prefix:
            extend_tokens = full_token_ids[: prefix_len + new_token_count]
            pool_cache = None
            cache = self._acquire_cache(
                len(extend_tokens) + _CONTIGUOUS_CACHE_DECODE_SLACK
            )
        elif prefix_len > 0 and not use_paged_prefill:
            cache = None
            slot_ids_mx = mx.array(prefix_slot_ids, dtype=mx.int32)
            full_block_ids = self._full_block_ids_for_slots(prefix_slot_ids)
            slot_range = (
                self._slot_range_for_slots(prefix_slot_ids)
                if prefix_len <= _RADIX_RANGE_GATHER_MAX_PREFIX_TOKENS
                else None
            )
            slot_ranges = (
                self._slot_ranges_for_slots(
                    prefix_slot_ids,
                    max_ranges=_RADIX_MULTI_RANGE_GATHER_MAX_RANGES,
                )
                if slot_range is None
                and prefix_len > _RADIX_RANGE_GATHER_MAX_PREFIX_TOKENS
                else None
            )
            pool_cache = [
                PoolBackedCache(
                    self._paged_kv_cache,
                    layer_idx,
                    slot_ids_mx,
                    prefix_len,
                    full_block_ids,
                    slot_range,
                    slot_ranges,
                )
                for layer_idx in range(self._num_layers)
            ]
            if new_token_count == 0:
                for item in pool_cache:
                    item.offset = max(item.offset - 1, 0)
        elif prefix_len > 0:
            pool_cache = None
            cache = None
        else:
            if not new_token_ids:
                raise RuntimeError("MLX runner requires tokens for prefill")
            pool_cache = None
            cache = self._acquire_cache(
                len(new_token_ids) + _CONTIGUOUS_CACHE_DECODE_SLACK
            )

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        model_cache = (
            pool_cache
            if (
                prefix_len > 0
                and not recompute_prefix
                and not use_paged_prefill
                and pool_cache is not None
            )
            else cache
        )
        model_start = time.perf_counter() if profile_timing else 0.0
        if use_paged_prefill:
            self._ensure_radix_paged_wrapper()
            paged_ctx = self._build_paged_prefill_context(
                prefix_slot_ids, list(new_slot_ids)
            )
            if paged_ctx is None:
                raise RuntimeError("MLX runner requires paged prefill context")
            set_paged_context(paged_ctx)
            try:
                model_output = self.model(
                    input_ids,
                    cache=self._make_cache_shim(paged_ctx.offset_values[0]),
                )
            finally:
                clear_paged_context()
            self._require_paged_kv_scatter(paged_ctx)
            mx.synchronize()
        else:
            model_output = self.model(input_ids, cache=model_cache)
        model_done = time.perf_counter() if profile_timing else 0.0
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        materialize_start = time.perf_counter() if profile_timing else 0.0
        if use_paged_prefill:
            cache = self._materialize_paged_slots_to_contiguous(
                list(prefix_slot_ids) + list(new_slot_ids)
            )
        elif prefix_len > 0 and not recompute_prefix and pool_cache is not None:
            cache = self._materialize_pool_backed_cache(
                pool_cache,
                right_size=prefix_len <= _RADIX_RIGHT_SIZE_POOL_CACHE_MAX_PREFIX_TOKENS,
            )
        materialize_done = time.perf_counter() if profile_timing else 0.0

        eval_start = time.perf_counter() if profile_timing else 0.0
        dense_recompute_cache = (
            not use_paged_prefill
            and (prefix_len == 0 or recompute_prefix)
            and new_slot_ids
        )
        sync_refs: list[mx.array] = []
        sync_eager_after_eval = False
        sync_lazy_after_eval = False
        if not use_paged_prefill and new_slot_ids:
            if len(new_slot_ids) > _RADIX_FOLDED_RANGE_SYNC_MAX_TOKENS:
                sync_lazy_after_eval = (
                    self._slot_range_for_slots(list(new_slot_ids)) is not None
                )
                sync_eager_after_eval = not sync_lazy_after_eval
            else:
                prepared_refs = self._prepare_new_kv_range_sync_to_pool(
                    cache, prefix_len, list(new_slot_ids)
                )
                if prepared_refs is None:
                    sync_eager_after_eval = True
                else:
                    sync_refs = prepared_refs
        compact_eval = not (
            prefix_len == 0
            or new_token_count >= _RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS
        )
        if dense_recompute_cache and not sync_lazy_after_eval:
            mx.eval(next_token_mlx, *sync_refs)
        else:
            self._eval_with_cache(
                next_token_mlx, cache, compact=compact_eval, extra=sync_refs
            )
        eval_done = time.perf_counter() if profile_timing else 0.0
        next_token = int(next_token_mlx.item())

        sync_start = time.perf_counter() if profile_timing else 0.0
        if sync_eager_after_eval:
            self._sync_new_kv_to_pool(cache, prefix_len, list(new_slot_ids))
        elif sync_lazy_after_eval:
            self._sync_new_kv_to_pool(
                cache, prefix_len, list(new_slot_ids), eager=False
            )
        sync_done = time.perf_counter() if profile_timing else 0.0
        self._req_token_ids[req_id] = list(full_token_ids) + [next_token]
        self._req_caches[req_id] = cache
        self._req_pool_idx[req_id] = req_pool_idx
        self._req_synced_offset[req_id] = synced_offset

        if profile_timing:
            logger.info(
                "MLX prefill timing rid=%s prefix=%s new=%s extend=%s "
                "recompute=%s materialized=%s paged=%s idle_paged=%s "
                "model_ms=%.2f materialize_ms=%.2f "
                "eval_ms=%.2f sync_ms=%.2f total_ms=%.2f",
                req_id,
                prefix_len,
                new_token_count,
                len(extend_tokens),
                recompute_prefix,
                False,
                use_paged_prefill,
                use_idle_paged_prefill,
                (model_done - model_start) * 1000,
                (materialize_done - materialize_start) * 1000,
                (eval_done - eval_start) * 1000,
                (sync_done - sync_start) * 1000,
                (sync_done - profile_start) * 1000,
            )

        self._mark_forward_done()
        return next_token

    def prefill_batch_no_radix(
        self,
        req_ids: list[str],
        token_ids: list[list[int]],
    ) -> list[int]:
        """Prefill equal-length no-radix requests in one MLX forward pass."""
        if not self.disable_radix_cache:
            raise RuntimeError("batched no-radix prefill requires radix cache disabled")
        if not req_ids:
            return []
        if len(req_ids) != len(token_ids):
            raise ValueError("req_ids and token_ids must have the same length")
        seq_len = len(token_ids[0])
        if seq_len <= 0:
            raise ValueError("batched no-radix prefill requires non-empty token IDs")
        if any(len(tokens) != seq_len for tokens in token_ids):
            raise ValueError("batched no-radix prefill requires equal sequence lengths")

        self._remove_no_radix_batched_wrapper()
        if not hasattr(self, "_req_caches"):
            self._req_caches = {}
        capacity = self._round_cache_capacity(seq_len + _CONTIGUOUS_CACHE_DECODE_SLACK)
        cache = [
            ContiguousKVCache(max_seq_len=capacity) for _ in range(self._num_layers)
        ]
        input_ids = mx.array(token_ids, dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)
        next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
        self._eval_with_cache(next_tokens_mlx, cache, compact=True)
        next_tokens = next_tokens_mlx.tolist()

        row_caches: list[list[ContiguousKVCache]] = [[] for _ in range(len(req_ids))]
        for layer_cache in cache:
            for row_idx, req_cache in enumerate(row_caches):
                row_cache = ContiguousKVCache(max_seq_len=capacity)
                row_cache.keys = mx.contiguous(layer_cache.keys[row_idx : row_idx + 1])
                row_cache.values = mx.contiguous(
                    layer_cache.values[row_idx : row_idx + 1]
                )
                row_cache.offset = layer_cache.offset
                row_cache.max_seq_len = capacity
                req_cache.append(row_cache)

        mx.eval(
            *[
                state
                for req_cache in row_caches
                for layer_cache in req_cache
                for state in layer_cache.state
            ]
        )

        for req_id, full_tokens, next_token, req_cache in zip(
            req_ids, token_ids, next_tokens, row_caches, strict=True
        ):
            self._req_token_ids[req_id] = list(full_tokens) + [next_token]
            self._req_caches[req_id] = req_cache
            self._req_pool_idx[req_id] = 0
            self._req_synced_offset[req_id] = 0

        self._mark_forward_done()
        return next_tokens

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.  Returns next_token_id."""
        if self.disable_radix_cache:
            if not hasattr(self, "_req_caches"):
                self._req_caches = {}
            assert (
                req_id in self._req_caches
            ), f"extend called for unknown request {req_id}"

            cache = self._req_caches[req_id]
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)

            last_logits = logits[:, -1, :]
            next_token_mlx = mx.argmax(last_logits, axis=-1)
            self._eval_with_cache(next_token_mlx, cache, compact=True)
            next_token = int(next_token_mlx.item())

            prev_tokens = self._req_token_ids[req_id]
            if prev_tokens:
                prev_tokens.pop()
            prev_tokens.extend(new_token_ids)
            prev_tokens.append(next_token)
            self._mark_forward_done()
            return next_token

        assert (
            req_id in self._req_token_ids
        ), f"extend called for unknown request {req_id}"
        if not hasattr(self, "_req_caches") or req_id not in self._req_caches:
            raise RuntimeError("MLX runner requires contiguous KV cache for extend")

        cache = self._req_caches[req_id]
        cache = self._materialize_synced_paged_cache_if_needed(req_id, cache)
        synced = int(self._req_synced_offset.get(req_id, 0))
        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)
        self._eval_with_cache(next_token_mlx, cache, compact=True)
        next_token = int(next_token_mlx.item())

        self._sync_new_kv_to_pool(cache, synced, list(new_slot_ids))
        self._req_synced_offset[req_id] = synced + len(new_slot_ids)

        prev_tokens = self._req_token_ids[req_id]
        if prev_tokens:
            prev_tokens.pop()
        prev_tokens.extend(new_token_ids)
        prev_tokens.append(next_token)

        self._mark_forward_done()
        return next_token

    def flush_all_decode_kv(self) -> None:
        """Sync any decoded contiguous-cache KV into radix-visible paged storage."""
        if self.disable_radix_cache or self._paged_kv_cache is None:
            return
        for req_id, cache in list(getattr(self, "_req_caches", {}).items()):
            self._sync_pending_decode_kv_to_pool(req_id, cache)

    def _sync_pending_decode_kv_to_pool(
        self,
        req_id: str,
        cache: list[ContiguousKVCache] | None = None,
    ) -> None:
        """Flush decoded-but-unsynced request-local KV into the radix side-store."""
        if self.disable_radix_cache or self._paged_kv_cache is None:
            return
        if self._req_to_token_pool is None:
            return
        if cache is None:
            cache = getattr(self, "_req_caches", {}).get(req_id)
        if cache is None:
            return

        current_offset = cache[0].offset
        synced_offset = int(self._req_synced_offset.get(req_id, 0))
        if current_offset <= synced_offset:
            return
        req_pool_idx = self._req_pool_idx.get(req_id)
        if req_pool_idx is None:
            return
        slot_ids = (
            self._req_to_token_pool.req_to_token[
                req_pool_idx, synced_offset:current_offset
            ]
            .to(dtype=int)
            .tolist()
        )
        self._sync_new_kv_to_pool(cache, synced_offset, slot_ids)
        self._req_synced_offset[req_id] = current_offset

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
        self,
        req_ids: list[str],
        seq_lens: list[int],
        decode_slot_ids: list[int] | None = None,
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
        if decode_slot_ids is not None:
            decode_slot_ids = list(decode_slot_ids)
            if len(decode_slot_ids) != len(req_ids):
                raise ValueError("decode_slot_ids length must match req_ids")
            for row, seq_len, slot_id in zip(
                slot_rows, seq_lens, decode_slot_ids, strict=True
            ):
                row[seq_len] = int(slot_id)
            slot_mapping = decode_slot_ids
        else:
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
        decode_slot_ids: list[int] | None = None,
    ) -> list[int]:
        """Decode one token per request."""
        if not req_ids:
            return []
        profile_timing = self._profile_timing_enabled()

        if self.disable_radix_cache:
            if not hasattr(self, "_req_caches"):
                self._req_caches = {}
            profile_start = time.perf_counter() if profile_timing else 0.0
            model_start = profile_start
            model_done = profile_start
            eval_start = profile_start
            eval_done = profile_start
            batch_size = len(req_ids)
            caches = [self._req_caches[rid] for rid in req_ids]

            if batch_size == 1:
                self._remove_no_radix_batched_wrapper()
                cache = caches[0]
                last_token = self._req_token_ids[req_ids[0]][-1]
                input_ids = mx.array([[last_token]], dtype=mx.int32)
                model_start = time.perf_counter() if profile_timing else 0.0
                model_output = self.model(input_ids, cache=cache)
                model_done = time.perf_counter() if profile_timing else 0.0
                logits = self._extract_logits(model_output)
                next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
                eval_start = time.perf_counter() if profile_timing else 0.0
                self._eval_with_cache(next_tokens_mlx, cache)
                eval_done = time.perf_counter() if profile_timing else 0.0
            else:
                seq_lens = [caches[i][0].offset for i in range(batch_size)]
                layer_caches = [
                    [caches[i][layer_idx] for i in range(batch_size)]
                    for layer_idx in range(self._num_layers)
                ]
                ctx = BatchedDecodeContext(
                    batch_size=batch_size,
                    seq_lens=seq_lens,
                    layer_caches=layer_caches,
                )
                self._ensure_no_radix_batched_wrapper()
                set_context(ctx)
                try:
                    max_offset = max(seq_lens)
                    shim_cache = self._make_cache_shim(max_offset)
                    last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
                    batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
                    model_start = time.perf_counter() if profile_timing else 0.0
                    model_output = self.model(batched_input, cache=shim_cache)
                    model_done = time.perf_counter() if profile_timing else 0.0
                    logits = self._extract_logits(model_output)
                    next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)

                    eval_targets = [next_tokens_mlx]
                    for req_cache in caches:
                        for layer_cache in req_cache:
                            eval_targets.extend(layer_cache.state)
                    eval_start = time.perf_counter() if profile_timing else 0.0
                    mx.eval(*eval_targets)
                    eval_done = time.perf_counter() if profile_timing else 0.0
                finally:
                    clear_context()

            next_tokens = next_tokens_mlx.tolist()
            for i, rid in enumerate(req_ids):
                self._req_token_ids[rid].append(next_tokens[i])
            if profile_timing:
                seq_lens_for_log = [caches[i][0].offset for i in range(batch_size)]
                logger.info(
                    "MLX decode timing no_radix batch=%s seq_min=%s seq_max=%s "
                    "model_ms=%.2f eval_ms=%.2f total_ms=%.2f",
                    batch_size,
                    min(seq_lens_for_log),
                    max(seq_lens_for_log),
                    (model_done - model_start) * 1000,
                    (eval_done - eval_start) * 1000,
                    (eval_done - profile_start) * 1000,
                )
            self._mark_forward_done()
            return next_tokens

        if not hasattr(self, "_req_caches"):
            self._req_caches = {}
        batch_size = len(req_ids)
        missing = [rid for rid in req_ids if rid not in self._req_caches]
        if missing:
            raise RuntimeError(
                "MLX runner requires contiguous KV cache for decode: "
                + ", ".join(missing)
            )
        caches = [self._req_caches[rid] for rid in req_ids]
        history_lens = [len(self._req_token_ids[rid]) - 1 for rid in req_ids]
        profile_start = time.perf_counter() if profile_timing else 0.0
        model_start = profile_start
        model_done = profile_start
        eval_start = profile_start
        eval_done = profile_start

        paged_ctx = None
        if self._should_use_paged_radix_decode(batch_size, history_lens):
            can_use_paged = all(
                self._can_sync_or_use_paged_decode_state(rid, cache, seq_len)
                for rid, cache, seq_len in zip(
                    req_ids, caches, history_lens, strict=True
                )
            )
            if can_use_paged:
                paged_ctx = self._build_paged_decode_context(
                    req_ids, history_lens, decode_slot_ids
                )

        if paged_ctx is not None:
            self._ensure_radix_paged_wrapper()
            last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
            input_ids = mx.array(last_tokens, dtype=mx.int32)[:, None]
            set_paged_context(paged_ctx)
            try:
                model_start = time.perf_counter() if profile_timing else 0.0
                model_output = self.model(
                    input_ids,
                    cache=self._make_cache_shim(max(history_lens)),
                )
                model_done = time.perf_counter() if profile_timing else 0.0
            finally:
                clear_paged_context()
            self._require_paged_kv_scatter(paged_ctx)
            logits = self._extract_logits(model_output)
            next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            eval_start = time.perf_counter() if profile_timing else 0.0
            mx.eval(next_tokens_mlx)
            eval_done = time.perf_counter() if profile_timing else 0.0

            next_tokens = next_tokens_mlx.tolist()
            for i, rid in enumerate(req_ids):
                self._req_token_ids[rid].append(next_tokens[i])
                self._req_synced_offset[rid] = history_lens[i] + 1

            if profile_timing:
                logger.info(
                    "MLX decode timing radix_paged batch=%s seq_min=%s seq_max=%s "
                    "model_ms=%.2f eval_ms=%.2f total_ms=%.2f",
                    batch_size,
                    min(history_lens),
                    max(history_lens),
                    (model_done - model_start) * 1000,
                    (eval_done - eval_start) * 1000,
                    (eval_done - profile_start) * 1000,
                )

            self._mark_forward_done()
            return next_tokens

        for i, rid in enumerate(req_ids):
            caches[i] = self._materialize_synced_paged_cache_if_needed(rid, caches[i])
        seq_lens = [caches[i][0].offset for i in range(batch_size)]

        if (
            batch_size == 1
            and seq_lens[0] < _RADIX_FULL_STATE_DECODE_MIN_TOKENS
            and seq_lens[0] <= _RADIX_BATCHED_WRAPPER_B1_DECODE_MAX_TOKENS
        ):
            layer_caches = [
                [caches[0][layer_idx]] for layer_idx in range(self._num_layers)
            ]
            ctx = BatchedDecodeContext(
                batch_size=1,
                seq_lens=seq_lens,
                layer_caches=layer_caches,
            )
            full_state_eval = all(
                layer_cache.keys is not None
                and int(layer_cache.keys.shape[2])
                <= _RADIX_BATCHED_WRAPPER_B1_FULL_STATE_MAX_CAPACITY
                for layer_cache in caches[0]
            )
            set_context(ctx)
            try:
                last_token = self._req_token_ids[req_ids[0]][-1]
                input_ids = mx.array([[last_token]], dtype=mx.int32)
                model_start = time.perf_counter() if profile_timing else 0.0
                model_output = self.model(
                    input_ids,
                    cache=self._make_cache_shim(seq_lens[0]),
                )
                model_done = time.perf_counter() if profile_timing else 0.0
                logits = self._extract_logits(model_output)
                next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)

                eval_targets = [next_tokens_mlx]
                for layer_cache in caches[0]:
                    if full_state_eval:
                        eval_targets.extend(layer_cache.state)
                    else:
                        eval_targets.extend(
                            self._cache_state_for_eval(layer_cache, compact=True)
                        )
                eval_start = time.perf_counter() if profile_timing else 0.0
                mx.eval(*eval_targets)
                eval_done = time.perf_counter() if profile_timing else 0.0
            finally:
                clear_context()
        elif batch_size == 1:
            cache = caches[0]
            req_id = req_ids[0]
            last_token = self._req_token_ids[req_ids[0]][-1]
            input_ids = mx.array([[last_token]], dtype=mx.int32)
            model_start = time.perf_counter() if profile_timing else 0.0
            model_output = self.model(input_ids, cache=cache)
            model_done = time.perf_counter() if profile_timing else 0.0
            logits = self._extract_logits(model_output)
            next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)
            eval_start = time.perf_counter() if profile_timing else 0.0
            compact_decode_eval = seq_lens[0] < _RADIX_FULL_STATE_DECODE_MIN_TOKENS
            self._eval_with_cache(next_tokens_mlx, cache, compact=compact_decode_eval)
            eval_done = time.perf_counter() if profile_timing else 0.0
        else:
            layer_caches = [
                [caches[i][layer_idx] for i in range(batch_size)]
                for layer_idx in range(self._num_layers)
            ]
            ctx = BatchedDecodeContext(
                batch_size=batch_size,
                seq_lens=seq_lens,
                layer_caches=layer_caches,
            )
            self._ensure_no_radix_batched_wrapper()
            set_context(ctx)
            try:
                max_offset = max(seq_lens)
                shim_cache = self._make_cache_shim(max_offset)
                last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
                batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
                model_start = time.perf_counter() if profile_timing else 0.0
                model_output = self.model(batched_input, cache=shim_cache)
                model_done = time.perf_counter() if profile_timing else 0.0
                logits = self._extract_logits(model_output)
                next_tokens_mlx = mx.argmax(logits[:, -1, :], axis=-1)

                eval_targets = [next_tokens_mlx]
                for req_cache in caches:
                    for layer_cache in req_cache:
                        eval_targets.extend(
                            self._cache_state_for_eval(layer_cache, compact=True)
                        )
                eval_start = time.perf_counter() if profile_timing else 0.0
                mx.eval(*eval_targets)
                eval_done = time.perf_counter() if profile_timing else 0.0
            finally:
                clear_context()

        next_tokens = next_tokens_mlx.tolist()
        for req_cache, seq_len in zip(caches, seq_lens, strict=True):
            self._require_cache_range(req_cache, int(seq_len) + 1, "decode")
        for i, rid in enumerate(req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        if profile_timing:
            logger.info(
                "MLX decode timing radix batch=%s seq_min=%s seq_max=%s "
                "model_ms=%.2f eval_ms=%.2f total_ms=%.2f",
                batch_size,
                min(seq_lens),
                max(seq_lens),
                (model_done - model_start) * 1000,
                (eval_done - eval_start) * 1000,
                (eval_done - profile_start) * 1000,
            )

        self._mark_forward_done()
        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        if self.disable_radix_cache:
            return req_id in getattr(self, "_req_caches", {})
        return req_id in self._req_token_ids

    def remove_request(self, req_id: str):
        """Release request state."""
        cache = getattr(self, "_req_caches", {}).get(req_id)
        self._sync_pending_decode_kv_to_pool(req_id, cache)
        self._req_token_ids.pop(req_id, None)
        cache = getattr(self, "_req_caches", {}).pop(req_id, None)
        if cache is not None:
            self._release_cache(cache)
        self._req_pool_idx.pop(req_id, None)
        self._req_synced_offset.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        if self.disable_radix_cache:
            self._remove_no_radix_batched_wrapper()
        self._req_token_ids.clear()
        cache_pool = getattr(self, "_cache_pool", None)
        if cache_pool is None:
            self._cache_pool = []
            cache_pool = self._cache_pool
        req_caches = getattr(self, "_req_caches", {})
        for cache in req_caches.values():
            if all(isinstance(item, ContiguousKVCache) for item in cache):
                cache_pool.append(cache)
        req_caches.clear()
        self._req_pool_idx.clear()
        self._req_synced_offset.clear()
        if self._paged_kv_cache is not None:
            self._paged_kv_cache.clear()
