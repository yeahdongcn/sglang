"""Unit tests for MLX model runner paged attention routing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - platform-dependent optional dependency
    mx = None

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@unittest.skipIf(mx is None, "MLX is not available")
class TestMlxModelRunnerDecodeRouting(CustomTestCase):
    def _make_runner(self, req_token_ids=None):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        req_token_ids = {"req": [7]} if req_token_ids is None else req_token_ids
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._num_layers = 2
        runner._req_token_ids = {
            rid: list(tokens) for rid, tokens in req_token_ids.items()
        }
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        runner._req_caches = {}
        runner._cache_pool = []
        runner._max_seq_len = 16
        runner._paged_kv_cache = None
        runner._req_to_token_pool = None
        runner._paged_attention_block_size = 2
        runner.disable_radix_cache = False
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )
        return runner

    def _make_paged_cache(self, block_size=2, dtype=None, head_dim=2):
        dtype = mx.float32 if dtype is None else dtype

        class FakePagedCache:
            def __init__(self):
                self.block_size = block_size
                self.dtype = dtype
                self.head_dim = head_dim
                self.n_kv_heads = 1
                self.k_buffer = [
                    mx.zeros((64, block_size, 1, head_dim), dtype=dtype)
                    for _ in range(2)
                ]
                self.v_buffer = [
                    mx.zeros((64, block_size, 1, head_dim), dtype=dtype)
                    for _ in range(2)
                ]
                self.set_kv = MagicMock()
                self.set_kv_all_layers = MagicMock()
                self.set_kv_all_layers_slot_range = MagicMock()
                self.clear = MagicMock()

            def get_kv(self, layer_id, slots):
                token_count = slots.size if isinstance(slots, mx.array) else len(slots)
                k = mx.ones((token_count, 1, head_dim), dtype=dtype) * (layer_id + 1)
                return k, k + 10

            def get_kv_slot_range(self, layer_id, start, end):
                token_count = end - start
                k = mx.ones((token_count, 1, head_dim), dtype=dtype) * (layer_id + 1)
                return k, k + 10

        return FakePagedCache()

    def _make_populating_model(self, logits):
        def model(input_ids, cache=None):
            from sglang.srt.hardware_backend.mlx.kv_cache import (
                get_context,
                get_paged_context,
            )

            paged_ctx = get_paged_context()
            ctx = get_context()
            if paged_ctx is not None:
                for layer_idx in range(2):
                    paged_ctx.mark_kv_scattered(layer_idx)
            elif ctx is not None:
                for layer_idx, layer_caches in enumerate(ctx.layer_caches):
                    for row_idx, layer_cache in enumerate(layer_caches):
                        k = mx.ones((1, 1, 1, 2), dtype=mx.float32) * (
                            100 + layer_idx * 10 + row_idx
                        )
                        layer_cache.write_token(k, k + 1000)
            elif cache is not None:
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                for layer_idx, layer_cache in enumerate(cache):
                    k = mx.ones((batch_size, 1, seq_len, 2), dtype=mx.float32) * (
                        200 + layer_idx
                    )
                    layer_cache.update_and_fetch(k, k + 1000)
            return logits

        return MagicMock(side_effect=model)

    def _install_contiguous_caches(self, runner, offsets):
        from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousKVCache

        runner._req_caches = {}
        for req_id, offset in offsets.items():
            caches = []
            for layer_idx in range(runner._num_layers):
                cache = ContiguousKVCache(
                    n_kv_heads=1,
                    head_dim=2,
                    max_seq_len=16,
                    dtype=mx.float32,
                )
                if offset:
                    k = mx.ones((1, 1, offset, 2), dtype=mx.float32) * (10 + layer_idx)
                    cache.update_and_fetch(k, k + 100)
                    mx.eval(*cache.state)
                caches.append(cache)
            runner._req_caches[req_id] = caches

    def test_contiguous_cache_write_token_grows_when_full(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousKVCache

        cache = ContiguousKVCache(
            n_kv_heads=1,
            head_dim=2,
            max_seq_len=1,
            dtype=mx.float32,
        )
        first = mx.ones((1, 1, 1, 2), dtype=mx.float32)
        second = first * 2

        cache.write_token(first, first + 10)
        cache.write_token(second, second + 10)

        self.assertEqual(cache.offset, 2)
        self.assertGreaterEqual(cache.keys.shape[2], 2)
        self.assertTrue(mx.allclose(cache.keys[:, :, :1, :], first))
        self.assertTrue(mx.allclose(cache.keys[:, :, 1:2, :], second))

    def test_contiguous_cache_update_grows_from_actual_buffer_capacity(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousKVCache

        cache = ContiguousKVCache(
            n_kv_heads=1,
            head_dim=2,
            max_seq_len=2,
            dtype=mx.float32,
        )
        cache.max_seq_len = 4
        values = mx.ones((1, 1, 4, 2), dtype=mx.float32)

        cache.update_and_fetch(values, values + 10)

        self.assertEqual(cache.offset, 4)
        self.assertGreaterEqual(cache.keys.shape[2], 4)
        self.assertTrue(mx.allclose(cache.keys[:, :, :4, :], values))

    def _install_scheduler_slots(self, runner, rows, pool_indices):
        runner._paged_kv_cache = self._make_paged_cache()
        runner._req_to_token_pool = MagicMock()
        runner._req_to_token_pool.req_to_token = FakeReqToToken(rows)
        runner._req_pool_idx = pool_indices

    def _make_worker_batch(self, forward_mode, seq_len=1, decoding_req_ids=None):
        req = SimpleNamespace(
            rid="req",
            prefix_indices=torch.tensor([]),
            fill_ids=[7],
            req_pool_idx=1,
        )
        input_ids = list(range(10, 10 + seq_len))
        out_cache_loc = list(range(6, 6 + seq_len))
        decoding_req_ids = set() if decoding_req_ids is None else decoding_req_ids
        return SimpleNamespace(
            forward_mode=forward_mode,
            reqs=[req],
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            out_cache_loc=torch.tensor(out_cache_loc, dtype=torch.long),
            extend_seq_lens=[seq_len],
            decoding_reqs=[req] if req.rid in decoding_req_ids else None,
        )

    def test_worker_routes_one_token_extend_to_extend_outside_mixed(self):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_active_rids = set()
        worker._mlx_runner = MagicMock()
        worker._mlx_runner.has_request.return_value = True
        worker._mlx_runner.extend.return_value = 23
        batch = self._make_worker_batch(ForwardMode.EXTEND)

        result = MlxTpModelWorker._forward_batch_generation_mlx(worker, batch)

        worker._mlx_runner.extend.assert_called_once_with("req", [10], [6])
        worker._mlx_runner.decode_batch.assert_not_called()
        self.assertEqual(result.next_token_ids.tolist(), [23])

    def test_worker_releases_stale_requests_before_extend(self):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_active_rids = {"old_req"}
        worker._mlx_runner = MagicMock()
        worker._mlx_runner._profile_timing_enabled.return_value = False
        worker._mlx_runner.disable_radix_cache = True
        worker._mlx_runner.has_request.return_value = False
        worker._mlx_runner.prefill.return_value = 23
        batch = self._make_worker_batch(ForwardMode.EXTEND)

        result = MlxTpModelWorker._forward_batch_generation_mlx(worker, batch)

        worker._mlx_runner.remove_request.assert_called_once_with("old_req")
        worker._mlx_runner.prefill.assert_called_once()
        self.assertEqual(worker._mlx_active_rids, {"req"})
        self.assertEqual(result.next_token_ids.tolist(), [23])

    def test_worker_routes_one_token_mixed_existing_request_to_decode(self):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_active_rids = set()
        worker._mlx_runner = MagicMock()
        worker._mlx_runner.has_request.return_value = True
        worker._mlx_runner.decode_batch.return_value = [24]
        batch = self._make_worker_batch(ForwardMode.MIXED, decoding_req_ids={"req"})

        result = MlxTpModelWorker._forward_batch_generation_mlx(worker, batch)

        worker._mlx_runner.decode_batch.assert_called_once_with(["req"], [6])
        worker._mlx_runner.extend.assert_not_called()
        self.assertEqual(result.next_token_ids.tolist(), [24])

    def test_worker_routes_one_token_mixed_existing_non_decode_to_extend(self):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_active_rids = set()
        worker._mlx_runner = MagicMock()
        worker._mlx_runner.has_request.return_value = True
        worker._mlx_runner.extend.return_value = 25
        batch = self._make_worker_batch(ForwardMode.MIXED)

        result = MlxTpModelWorker._forward_batch_generation_mlx(worker, batch)

        worker._mlx_runner.extend.assert_called_once_with("req", [10], [6])
        worker._mlx_runner.decode_batch.assert_not_called()
        self.assertEqual(result.next_token_ids.tolist(), [25])

    def test_worker_releases_active_mlx_requests_on_idle(self):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_active_rids = {"req0", "req1"}
        worker._mlx_runner = MagicMock()
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE, reqs=[])

        result = MlxTpModelWorker._forward_batch_generation_mlx(worker, batch)

        self.assertIsNone(result.next_token_ids)
        self.assertEqual(worker._mlx_active_rids, set())
        self.assertEqual(worker._mlx_runner.remove_request.call_count, 2)
        removed = {
            call.args[0] for call in worker._mlx_runner.remove_request.call_args_list
        }
        self.assertEqual(removed, {"req0", "req1"})
        worker._mlx_runner.mark_idle.assert_called_once()
        worker._mlx_runner.clear.assert_not_called()

    def test_startup_guard_requires_apple_silicon(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "linux"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="x86_64",
        ):
            with self.assertRaisesRegex(RuntimeError, "requires Apple Silicon"):
                _check_mlx_metal_backend_available()

    def test_startup_guard_requires_metal_extension(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            side_effect=ImportError("missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires sgl_kernel.metal"):
                _check_mlx_metal_backend_available()

    def test_startup_guard_requires_paged_metal_apis(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()
        del metal_module.prefill_attention_paged

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            with self.assertRaisesRegex(RuntimeError, "prefill_attention_paged"):
                _check_mlx_metal_backend_available()

    def test_startup_guard_requires_unchecked_paged_decode_api(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()
        del metal_module.decode_attention_paged_unchecked

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "decode_attention_paged_unchecked"
            ):
                _check_mlx_metal_backend_available()

    def test_startup_guard_requires_lazy_paged_decode_api(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()
        del metal_module.decode_attention_paged_lazy_unchecked

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "decode_attention_paged_lazy_unchecked"
            ):
                _check_mlx_metal_backend_available()

    def test_startup_guard_requires_p1_lazy_paged_decode_api(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()
        del metal_module.decode_attention_paged_p1_lazy_unchecked

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "decode_attention_paged_p1_lazy_unchecked"
            ):
                _check_mlx_metal_backend_available()

    def test_model_runner_defaults_to_page_sized_blocks(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self.assertEqual(MlxModelRunner._normalize_page_size(None), 1)
        self.assertEqual(MlxModelRunner._normalize_page_size(4), 4)
        with self.assertRaisesRegex(ValueError, "page_size must be positive"):
            MlxModelRunner._normalize_page_size(0)

    def test_model_runner_aligns_pool_size_to_page_size(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._paged_attention_block_size = 16

        self.assertEqual(runner._align_pool_size_to_pages(31003), 30992)
        self.assertEqual(runner._align_pool_size_to_pages(15), 16)

        runner._paged_attention_block_size = 1
        self.assertEqual(runner._align_pool_size_to_pages(31003), 31003)

    def test_radix_paged_prefill_threshold_uses_prefix_and_suffix_size(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self.assertFalse(MlxModelRunner._should_use_paged_radix_prefill(432, 2))
        self.assertFalse(MlxModelRunner._should_use_paged_radix_prefill(256, 512))
        self.assertTrue(MlxModelRunner._should_use_paged_radix_prefill(432, 384))
        self.assertTrue(MlxModelRunner._should_use_paged_radix_prefill(256, 768))
        self.assertTrue(MlxModelRunner._should_use_paged_radix_prefill(434, 1728))
        self.assertTrue(MlxModelRunner._should_use_paged_radix_prefill(1392, 1781))

    def test_small_radix_prefix_recompute_is_capped(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self.assertTrue(MlxModelRunner._should_recompute_small_radix_prefix(2, 5))
        self.assertFalse(MlxModelRunner._should_recompute_small_radix_prefix(433, 1729))

    def test_idle_radix_paged_prefill_threshold_uses_idle_boundary(self):
        runner = self._make_runner()
        runner._last_idle_at = None

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {"SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL": "1"},
            clear=True,
        ):
            self.assertFalse(runner._should_use_idle_paged_radix_prefill(432, 1))

            runner._last_idle_at = 10.0
            with patch(
                "sglang.srt.hardware_backend.mlx.model_runner.time.perf_counter",
                return_value=10.5,
            ):
                self.assertFalse(runner._should_use_idle_paged_radix_prefill(432, 1))

            with patch(
                "sglang.srt.hardware_backend.mlx.model_runner.time.perf_counter",
                return_value=11.1,
            ):
                self.assertTrue(runner._should_use_idle_paged_radix_prefill(432, 1))
                self.assertFalse(runner._should_use_idle_paged_radix_prefill(64, 1))
                self.assertFalse(runner._should_use_idle_paged_radix_prefill(432, 17))

    def test_idle_radix_paged_prefill_is_opt_in(self):
        runner = self._make_runner()
        runner._last_idle_at = 10.0

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {},
            clear=True,
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.time.perf_counter",
            return_value=11.1,
        ):
            self.assertFalse(runner._should_use_idle_paged_radix_prefill(432, 1))

    def test_full_block_ids_for_slots_accepts_whole_pages_only(self):
        runner = self._make_runner()
        runner._paged_attention_block_size = 4

        block_ids = runner._full_block_ids_for_slots([4, 5, 6, 7, 12, 13, 14, 15])

        self.assertEqual(block_ids.tolist(), [1, 3])
        self.assertIsNone(runner._full_block_ids_for_slots([4, 5, 6]))
        self.assertIsNone(runner._full_block_ids_for_slots([5, 6, 7, 8]))
        self.assertIsNone(runner._full_block_ids_for_slots([4, 5, 7, 6]))

    def test_slot_ranges_for_slots_accepts_bounded_p1_runs(self):
        runner = self._make_runner()
        runner._paged_attention_block_size = 1

        self.assertEqual(
            runner._slot_ranges_for_slots([1, 2, 3, 10, 11], max_ranges=2),
            [(1, 4), (10, 12)],
        )
        self.assertIsNone(
            runner._slot_ranges_for_slots([1, 2, 10, 11, 20], max_ranges=2)
        )

        runner._paged_attention_block_size = 2
        self.assertIsNone(runner._slot_ranges_for_slots([1, 2, 3], max_ranges=2))

    def test_startup_guard_requires_native_metal_extension(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()
        metal_module._require_metal_extension.side_effect = ImportError(
            "sgl_kernel._metal is not available"
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            with self.assertRaisesRegex(RuntimeError, "native sgl_kernel._metal"):
                _check_mlx_metal_backend_available()

    def test_startup_guard_accepts_required_paged_metal_apis(self):
        from sglang.srt.hardware_backend.mlx.model_runner import (
            _check_mlx_metal_backend_available,
        )

        metal_module = MagicMock()

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.sys.platform", "darwin"
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.platform.machine",
            return_value="arm64",
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.importlib.import_module",
            return_value=metal_module,
        ):
            self.assertIsNone(_check_mlx_metal_backend_available())

        runner = self._make_runner({"req0": [7, 8], "req1": [9, 10, 11]})
        self._install_scheduler_slots(
            runner,
            {
                1: [0, 1, 2, 3, 4, 5],
                3: [8, 9, 10, 11, 12, 13],
            },
            {"req0": 1, "req1": 3},
        )
        runner._req_synced_offset = {"req0": 5, "req1": 2}

        ctx = runner._build_paged_decode_context(["req0", "req1"], [2, 3])

        self.assertFalse(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.dtype, mx.int32)
        self.assertEqual(ctx.block_tables.dtype, mx.int32)
        self.assertEqual(ctx.context_lens.dtype, mx.int32)
        self.assertEqual(ctx.offsets.dtype, mx.int32)
        self.assertEqual(ctx.radix_prefix_lens.dtype, mx.int32)
        self.assertEqual(ctx.slot_mapping.tolist(), [2, 11])
        self.assertEqual(ctx.block_tables.tolist(), [[0, 1], [4, 5]])
        self.assertEqual(ctx.context_lens.tolist(), [3, 4])
        self.assertEqual(ctx.offsets.tolist(), [2, 3])
        self.assertEqual(ctx.cu_seqlens.tolist(), [0, 1, 2])
        self.assertEqual(ctx.max_seqlen_q, 1)
        self.assertEqual(ctx.max_seqlen_k, 4)
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [2, 2])
        self.assertIs(ctx.kv_pool, runner._paged_kv_cache)

    def test_build_paged_decode_context_uses_explicit_decode_slots(self):
        runner = self._make_runner({"req0": [7, 8, 9], "req1": [10, 11, 12, 13, 14]})
        self._install_scheduler_slots(
            runner,
            {
                1: [0, 1, 2, 3, 4, 5],
                3: [8, 9, 10, 11, 12, 13],
            },
            {"req0": 1, "req1": 3},
        )
        runner._req_synced_offset = {"req0": 2, "req1": 4}

        ctx = runner._build_paged_decode_context(
            ["req0", "req1"], [2, 4], decode_slot_ids=[7, 21]
        )

        self.assertFalse(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.tolist(), [7, 21])
        self.assertEqual(ctx.block_tables.tolist(), [[0, 3, 0], [4, 5, 10]])
        self.assertEqual(ctx.context_lens.tolist(), [3, 5])
        self.assertEqual(ctx.offsets.tolist(), [2, 4])
        self.assertEqual(ctx.max_seqlen_k, 5)
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [2, 4])

    def test_build_paged_decode_context_validates_decode_slot_count(self):
        runner = self._make_runner({"req0": [7, 8, 9], "req1": [10, 11, 12]})
        self._install_scheduler_slots(
            runner,
            {
                1: [0, 1, 2, 3],
                3: [8, 9, 10, 11],
            },
            {"req0": 1, "req1": 3},
        )

        with self.assertRaisesRegex(
            ValueError, "decode_slot_ids length must match req_ids"
        ):
            runner._build_paged_decode_context(
                ["req0", "req1"], [2, 2], decode_slot_ids=[7]
            )

    def test_decode_batch_accepts_explicit_decode_slots_and_defers_sync(self):
        runner = self._make_runner({"req0": [7, 8, 9], "req1": [10, 11, 12, 13, 14]})
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]], dtype=mx.float32)
        )
        self._install_scheduler_slots(
            runner,
            {
                1: [0, 1, 2, 3, 4, 5],
                3: [8, 9, 10, 11, 12, 13],
            },
            {"req0": 1, "req1": 3},
        )
        self._install_contiguous_caches(runner, {"req0": 2, "req1": 4})
        runner._req_synced_offset = {"req0": 2, "req1": 4}

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_tokens = runner.decode_batch(["req0", "req1"], [7, 21])

        self.assertEqual(next_tokens, [2, 0])
        set_paged_context.assert_not_called()
        clear_paged_context.assert_not_called()
        runner._paged_kv_cache.set_kv.assert_not_called()
        runner._paged_kv_cache.set_kv_all_layers.assert_not_called()
        self.assertEqual(
            runner._req_token_ids,
            {"req0": [7, 8, 9, 2], "req1": [10, 11, 12, 13, 14, 0]},
        )
        self.assertEqual(runner._req_synced_offset, {"req0": 2, "req1": 4})

    def test_build_paged_prefill_context_from_prefix_and_new_slots(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()

        ctx = runner._build_paged_prefill_context([4, 5, 6], [7, 8])

        self.assertTrue(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.tolist(), [7, 8])
        self.assertEqual(ctx.block_tables.tolist(), [[2, 3, 4]])
        self.assertEqual(ctx.context_lens.tolist(), [5])
        self.assertEqual(ctx.offsets.tolist(), [3])
        self.assertEqual(ctx.cu_seqlens.tolist(), [0, 2])
        self.assertEqual(ctx.max_seqlen_q, 2)
        self.assertEqual(ctx.max_seqlen_k, 5)
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [3])
        self.assertIs(ctx.kv_pool, runner._paged_kv_cache)

    def test_build_paged_prefill_context_for_full_prefix_hit(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()

        ctx = runner._build_paged_prefill_context([4, 5, 6], [])

        self.assertTrue(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.tolist(), [6])
        self.assertEqual(ctx.block_tables.tolist(), [[2, 3]])
        self.assertEqual(ctx.context_lens.tolist(), [3])
        self.assertEqual(ctx.offsets.tolist(), [2])
        self.assertEqual(ctx.cu_seqlens.tolist(), [0, 1])
        self.assertEqual(ctx.max_seqlen_q, 1)
        self.assertEqual(ctx.max_seqlen_k, 3)
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [2])
        self.assertIs(ctx.kv_pool, runner._paged_kv_cache)

    def test_build_paged_prefill_context_returns_none_without_slots(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()

        ctx = runner._build_paged_prefill_context([], [])

        self.assertIsNone(ctx)

    def test_build_paged_extend_context_from_scheduler_slots(self):
        runner = self._make_runner({"req": [7, 8, 9]})
        self._install_scheduler_slots(runner, {1: [4, 5, 6, 7, 8]}, {"req": 1})
        runner._req_synced_offset = {"req": 2}

        ctx = runner._build_paged_extend_context("req", [6, 7])

        self.assertTrue(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.tolist(), [6, 7])
        self.assertEqual(ctx.block_tables.tolist(), [[2, 3]])
        self.assertEqual(ctx.context_lens.tolist(), [4])
        self.assertEqual(ctx.offsets.tolist(), [2])
        self.assertEqual(ctx.cu_seqlens.tolist(), [0, 2])
        self.assertEqual(ctx.max_seqlen_q, 2)
        self.assertEqual(ctx.max_seqlen_k, 4)
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [2])
        self.assertIs(ctx.kv_pool, runner._paged_kv_cache)

    def test_build_paged_extend_context_returns_none_without_pool(self):
        runner = self._make_runner({"req": [7]})

        ctx = runner._build_paged_extend_context("req", [4])

        self.assertIsNone(ctx)

    def test_build_paged_decode_context_returns_none_without_pool(self):
        runner = self._make_runner()

        ctx = runner._build_paged_decode_context(["req"], [1])

        self.assertIsNone(ctx)

    def test_decode_batch_requires_contiguous_cache(self):
        runner = self._make_runner({"req": [7]})

        with self.assertRaisesRegex(RuntimeError, "requires contiguous KV cache"):
            runner.decode_batch(["req"])

    def test_decode_batch_uses_contiguous_cache_and_defers_paged_pool_sync(self):
        runner = self._make_runner({"req0": [7], "req1": [8, 9]})
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]], dtype=mx.float32)
        )
        self._install_scheduler_slots(
            runner,
            {
                1: [11, 12, 13, 14],
                3: [31, 32, 33, 34],
            },
            {"req0": 1, "req1": 3},
        )
        self._install_contiguous_caches(runner, {"req0": 0, "req1": 1})
        runner._req_synced_offset = {"req0": 0, "req1": 1}

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_tokens = runner.decode_batch(["req0", "req1"])

        self.assertEqual(next_tokens, [2, 0])
        set_paged_context.assert_not_called()
        clear_paged_context.assert_not_called()
        runner._paged_kv_cache.set_kv.assert_not_called()
        runner._paged_kv_cache.set_kv_all_layers.assert_not_called()
        shim_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in shim_cache], [1, 1])
        self.assertEqual(runner._req_token_ids, {"req0": [7, 2], "req1": [8, 9, 0]})
        self.assertEqual(runner._req_synced_offset, {"req0": 0, "req1": 1})

    def test_radix_single_decode_uses_full_state_for_long_contexts(self):
        runner = self._make_runner({"req": [7, 8, 9, 10]})
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )
        self._install_scheduler_slots(runner, {1: [11, 12, 13, 14, 15]}, {"req": 1})
        self._install_contiguous_caches(runner, {"req": 3})
        runner._req_synced_offset = {"req": 3}

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner._RADIX_FULL_STATE_DECODE_MIN_TOKENS",
            3,
        ), patch.object(
            runner, "_eval_with_cache", wraps=runner._eval_with_cache
        ) as eval_with_cache:
            next_tokens = runner.decode_batch(["req"])

        self.assertEqual(next_tokens, [2])
        self.assertFalse(eval_with_cache.call_args.kwargs["compact"])

    def test_radix_p1_decode_uses_paged_context_and_updates_synced_offset(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import get_paged_context

        tokens = list(range(21))
        runner = self._make_runner({"req": tokens})
        runner._paged_attention_block_size = 1
        runner._paged_kv_cache = self._make_paged_cache(
            block_size=1, dtype=mx.bfloat16, head_dim=128
        )
        self._install_scheduler_slots(
            runner,
            {1: list(range(64))},
            {"req": 1},
        )
        runner._paged_kv_cache = self._make_paged_cache(
            block_size=1, dtype=mx.bfloat16, head_dim=128
        )
        self._install_contiguous_caches(runner, {"req": 20})
        runner._req_synced_offset = {"req": 20}

        seen_ctx = {}

        def model(input_ids, cache=None):
            ctx = get_paged_context()
            self.assertIsNotNone(ctx)
            seen_ctx["ctx"] = ctx
            self.assertEqual(input_ids.tolist(), [[20]])
            self.assertEqual([item.offset for item in cache], [20, 20])
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)
            return mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)

        runner.model = MagicMock(side_effect=model)

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {"SGLANG_MLX_ENABLE_BF16_PAGED_DECODE": "1"},
            clear=True,
        ):
            next_tokens = runner.decode_batch(["req"], [42])

        self.assertEqual(next_tokens, [2])
        ctx = seen_ctx["ctx"]
        self.assertFalse(ctx.is_prefill)
        self.assertEqual(ctx.slot_mapping.tolist(), [42])
        self.assertEqual(ctx.context_lens.tolist(), [21])
        self.assertEqual(ctx.offsets.tolist(), [20])
        self.assertEqual(runner._req_token_ids["req"], tokens + [2])
        self.assertEqual(runner._req_synced_offset, {"req": 21})
        self.assertEqual(runner._req_caches["req"][0].offset, 20)

    def test_radix_bf16_p1_decode_requires_explicit_opt_in(self):
        runner = self._make_runner({"req": list(range(21))})
        runner._paged_attention_block_size = 1
        runner._paged_kv_cache = self._make_paged_cache(
            block_size=1, dtype=mx.bfloat16, head_dim=128
        )
        self._install_scheduler_slots(runner, {1: list(range(64))}, {"req": 1})
        runner._paged_kv_cache = self._make_paged_cache(
            block_size=1, dtype=mx.bfloat16, head_dim=128
        )

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {},
            clear=True,
        ):
            self.assertFalse(runner._should_use_paged_radix_decode(1, [20]))

    def test_bf16_paged_cache_keeps_metal_scatter_opt_in(self):
        runner = self._make_runner()
        cache = self._make_paged_cache(block_size=1, dtype=mx.bfloat16, head_dim=128)

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {},
            clear=True,
        ):
            self.assertFalse(runner._paged_cache_supports_metal_scatter(cache))
            self.assertFalse(runner._paged_cache_supports_metal(cache))
        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {"SGLANG_MLX_ENABLE_BF16_METAL_SCATTER": "1"},
            clear=True,
        ):
            self.assertTrue(runner._paged_cache_supports_metal_scatter(cache))
            self.assertFalse(runner._paged_cache_supports_metal(cache))

    def test_radix_batched_decode_evaluates_only_valid_cache_slices(self):
        runner = self._make_runner({"req0": [7], "req1": [8, 9, 10, 11]})
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]], dtype=mx.float32)
        )
        self._install_scheduler_slots(
            runner,
            {
                1: [11, 12, 13, 14, 15, 16],
                3: [31, 32, 33, 34, 35, 36],
            },
            {"req0": 1, "req1": 3},
        )
        self._install_contiguous_caches(runner, {"req0": 0, "req1": 3})
        runner._req_synced_offset = {"req0": 0, "req1": 3}

        with patch("sglang.srt.hardware_backend.mlx.model_runner.mx.eval") as eval_mock:
            next_tokens = runner.decode_batch(["req0", "req1"])

        self.assertEqual(next_tokens, [2, 0])
        eval_args = max((call.args for call in eval_mock.call_args_list), key=len)
        cache_seq_lens = [
            arg.shape[2] for arg in eval_args[1:] if getattr(arg, "ndim", 0) == 4
        ]
        self.assertEqual(cache_seq_lens, [1, 1, 1, 1, 4, 4, 4, 4])

    def test_decode_batch_raises_if_model_does_not_populate_kv(self):
        runner = self._make_runner({"req": [7]})
        self._install_scheduler_slots(runner, {1: [11, 12]}, {"req": 1})
        self._install_contiguous_caches(runner, {"req": 0})
        runner._req_synced_offset = {"req": 0}
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with self.assertRaisesRegex(RuntimeError, "populated contiguous KV cache"):
            runner.decode_batch(["req"])

        self.assertEqual(runner._req_token_ids, {"req": [7]})
        self.assertEqual(runner._req_synced_offset, {"req": 0})

    def test_decode_batch_clears_contiguous_context_after_model_error(self):
        runner = self._make_runner({"req0": [7], "req1": [8, 9]})
        runner.model = MagicMock(side_effect=RuntimeError("decode failed"))
        self._install_scheduler_slots(
            runner,
            {
                1: [11, 12, 13, 14],
                3: [31, 32, 33, 34],
            },
            {"req0": 1, "req1": 3},
        )
        self._install_contiguous_caches(runner, {"req0": 0, "req1": 1})

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_context"
        ) as clear_context:
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                runner.decode_batch(["req0", "req1"])

        clear_context.assert_called_once()

    def test_prefill_uses_pool_backed_prefix_and_syncs_new_slots(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_token = runner.prefill(
                req_id="prefill_req",
                new_token_ids=[8, 9],
                full_token_ids=[7, 8, 9],
                prefix_slot_ids=[4, 5],
                new_slot_ids=[6, 7],
                req_pool_idx=1,
            )

        self.assertEqual(next_token, 2)
        set_paged_context.assert_not_called()
        clear_paged_context.assert_not_called()
        pool_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in pool_cache], [4, 4])
        runner._paged_kv_cache.set_kv.assert_not_called()
        slots_arg = runner._paged_kv_cache.set_kv_all_layers.call_args.args[0]
        self.assertEqual(slots_arg.tolist(), [6, 7])
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["use_metal"]
        )
        self.assertEqual(runner._req_caches["prefill_req"][0].offset, 4)
        self.assertEqual(runner._req_token_ids["prefill_req"], [7, 8, 9, 2])
        self.assertEqual(runner._req_pool_idx["prefill_req"], 1)
        self.assertEqual(runner._req_synced_offset["prefill_req"], 4)

    def test_prefill_uses_multi_range_prefix_hint_for_long_p1_hits(self):
        runner = self._make_runner()
        runner._paged_attention_block_size = 1

        class FakePagedCache:
            block_size = 1

            def __init__(self):
                self.get_kv = MagicMock()
                self.get_kv_slot_ranges = MagicMock(side_effect=self._get_slot_ranges)
                self.set_kv_all_layers = MagicMock()

            def _get_slot_ranges(self, layer_id, ranges):
                token_count = sum(end - start for start, end in ranges)
                k = mx.ones((token_count, 1, 2), dtype=mx.float32) * (layer_id + 3)
                return k, k + 10

        runner._paged_kv_cache = FakePagedCache()
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )
        prefix_slot_ids = list(range(1, 434)) + list(range(1000, 1594))
        expected_ranges = [(1, 434), (1000, 1593)]

        next_token = runner.prefill(
            req_id="full_hit_req",
            new_token_ids=[],
            full_token_ids=list(range(len(prefix_slot_ids))),
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=[],
            req_pool_idx=1,
        )

        self.assertEqual(next_token, 2)
        self.assertEqual(
            runner._paged_kv_cache.get_kv_slot_ranges.call_args_list[0].args,
            (0, expected_ranges),
        )
        self.assertEqual(
            runner._paged_kv_cache.get_kv_slot_ranges.call_args_list[1].args,
            (1, expected_ranges),
        )
        runner._paged_kv_cache.get_kv.assert_not_called()
        self.assertEqual(runner._req_caches["full_hit_req"][0].offset, 1027)
        self.assertEqual(runner._req_synced_offset["full_hit_req"], 1027)

    def test_prefill_uses_paged_context_for_large_radix_suffix(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import (
            OffsetCache,
            get_paged_context,
        )

        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        prefix_slot_ids = list(range(4, 1396))
        new_slot_ids = list(range(1396, 1796))

        def model(input_ids, cache=None):
            ctx = get_paged_context()
            self.assertIsNotNone(ctx)
            self.assertEqual(input_ids.shape[1], len(new_slot_ids))
            self.assertTrue(all(isinstance(item, OffsetCache) for item in cache))
            self.assertEqual([item.offset for item in cache], [1392, 1392])
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)
            return mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)

        runner.model = MagicMock(side_effect=model)

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.mx.synchronize"
        ) as synchronize:
            next_token = runner.prefill(
                req_id="prefill_req",
                new_token_ids=list(range(400)),
                full_token_ids=list(range(1792)),
                prefix_slot_ids=prefix_slot_ids,
                new_slot_ids=new_slot_ids,
                req_pool_idx=1,
            )

        self.assertEqual(next_token, 2)
        synchronize.assert_called_once()
        runner._paged_kv_cache.set_kv.assert_not_called()
        runner._paged_kv_cache.set_kv_all_layers.assert_not_called()
        self.assertEqual(runner._req_caches["prefill_req"][0].offset, 1792)
        self.assertEqual(runner._req_token_ids["prefill_req"], list(range(1792)) + [2])
        self.assertEqual(runner._req_pool_idx["prefill_req"], 1)
        self.assertEqual(runner._req_synced_offset["prefill_req"], 1792)

    def test_prefill_uses_paged_context_for_idle_short_radix_hit(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import (
            OffsetCache,
            get_paged_context,
        )

        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner._last_idle_at = 0.0
        prefix_slot_ids = list(range(4, 204))
        new_slot_ids = [204]

        def model(input_ids, cache=None):
            ctx = get_paged_context()
            self.assertIsNotNone(ctx)
            self.assertEqual(input_ids.shape[1], 1)
            self.assertTrue(all(isinstance(item, OffsetCache) for item in cache))
            self.assertEqual([item.offset for item in cache], [200, 200])
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)
            return mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)

        runner.model = MagicMock(side_effect=model)

        with patch.dict(
            "sglang.srt.hardware_backend.mlx.model_runner.os.environ",
            {"SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL": "1"},
            clear=True,
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.time.perf_counter",
            return_value=2.0,
        ), patch(
            "sglang.srt.hardware_backend.mlx.model_runner.mx.synchronize"
        ) as synchronize:
            next_token = runner.prefill(
                req_id="prefill_req",
                new_token_ids=[10],
                full_token_ids=list(range(201)),
                prefix_slot_ids=prefix_slot_ids,
                new_slot_ids=new_slot_ids,
                req_pool_idx=1,
            )

        self.assertEqual(next_token, 2)
        synchronize.assert_called_once()
        runner._paged_kv_cache.set_kv_all_layers.assert_not_called()
        self.assertEqual(runner._req_caches["prefill_req"][0].offset, 201)
        self.assertEqual(runner._req_synced_offset["prefill_req"], 201)

    def test_prefill_recomputes_small_partial_prefix(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousKVCache

        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        next_token = runner.prefill(
            req_id="prefill_req",
            new_token_ids=[10, 11, 12, 13, 14],
            full_token_ids=[8, 9, 10, 11, 12, 13, 14],
            prefix_slot_ids=[4, 5],
            new_slot_ids=[6, 7, 8, 9, 10],
            req_pool_idx=1,
        )

        self.assertEqual(next_token, 2)
        input_ids = runner.model.call_args.args[0]
        self.assertEqual(input_ids.tolist(), [[8, 9, 10, 11, 12, 13, 14]])
        model_cache = runner.model.call_args.kwargs["cache"]
        self.assertTrue(
            all(isinstance(cache, ContiguousKVCache) for cache in model_cache)
        )
        self.assertEqual([cache.offset for cache in model_cache], [7, 7])
        runner._paged_kv_cache.set_kv.assert_not_called()
        slots_arg, k_arg, v_arg = (
            runner._paged_kv_cache.set_kv_all_layers.call_args.args
        )
        self.assertEqual(slots_arg.tolist(), [6, 7, 8, 9, 10])
        self.assertEqual(len(k_arg), runner._num_layers)
        self.assertEqual(len(v_arg), runner._num_layers)
        self.assertEqual(k_arg[0].shape, (5, 1, 2))
        self.assertEqual(v_arg[0].shape, (5, 1, 2))
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["eager"]
        )
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["use_metal"]
        )
        self.assertEqual(runner._req_caches["prefill_req"][0].offset, 7)
        self.assertEqual(
            runner._req_token_ids["prefill_req"], [8, 9, 10, 11, 12, 13, 14, 2]
        )
        self.assertEqual(runner._req_synced_offset["prefill_req"], 7)

    def test_prefill_batch_no_radix_splits_row_caches(self):
        runner = self._make_runner({"r0": [1], "r1": [2]})
        runner.disable_radix_cache = True
        runner.model = self._make_populating_model(
            mx.array(
                [[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]],
                dtype=mx.float32,
            )
        )

        next_tokens = runner.prefill_batch_no_radix(["r0", "r1"], [[10, 11], [20, 21]])

        self.assertEqual(next_tokens, [2, 0])
        self.assertEqual(runner.model.call_args.args[0].tolist(), [[10, 11], [20, 21]])
        self.assertEqual(runner._req_token_ids["r0"], [10, 11, 2])
        self.assertEqual(runner._req_token_ids["r1"], [20, 21, 0])
        self.assertEqual(runner._req_caches["r0"][0].offset, 2)
        self.assertEqual(runner._req_caches["r1"][0].offset, 2)
        self.assertEqual(runner._req_caches["r0"][0].keys.shape[0], 1)
        self.assertEqual(runner._req_caches["r1"][0].keys.shape[0], 1)

    def _make_stub_for_allocator(self, mlx_page_size, mlx_pool_size=16):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub

        stub = MlxModelRunnerStub.__new__(MlxModelRunnerStub)
        stub._mlx_pool_size = mlx_pool_size
        stub._mlx_page_size = mlx_page_size
        stub.server_args = SimpleNamespace(enable_memory_saver=False)
        stub.model_config = SimpleNamespace(
            dtype=torch.float16,
            is_hybrid_swa=False,
            sliding_window_size=None,
            attention_chunk_size=None,
            num_hidden_layers=1,
            num_attention_layers=1,
            context_len=16,
        )
        return stub

    def test_model_runner_stub_uses_token_allocator_for_default_page_size(self):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        stub = self._make_stub_for_allocator(1)

        MlxModelRunnerStub.initialize(stub, pre_model_load_memory=0)

        self.assertIsInstance(stub.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
        self.assertEqual(stub.token_to_kv_pool.page_size, 1)
        self.assertEqual(stub.token_to_kv_pool_allocator.page_size, 1)

    def test_model_runner_stub_uses_paged_allocator_for_larger_page_size(self):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        stub = self._make_stub_for_allocator(4)

        MlxModelRunnerStub.initialize(stub, pre_model_load_memory=0)

        allocator = stub.token_to_kv_pool_allocator
        self.assertIsInstance(allocator, PagedTokenToKVPoolAllocator)
        self.assertEqual(stub.token_to_kv_pool.page_size, 4)
        self.assertEqual(allocator.page_size, 4)
        self.assertEqual(allocator.alloc(4).tolist(), [4, 5, 6, 7])
        self.assertEqual(allocator.alloc(8).tolist(), [8, 9, 10, 11, 12, 13, 14, 15])

    def test_model_runner_stub_aligns_paged_pool_for_idle_accounting(self):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub

        stub = self._make_stub_for_allocator(4, mlx_pool_size=18)

        MlxModelRunnerStub.initialize(stub, pre_model_load_memory=0)

        self.assertEqual(stub.max_total_num_tokens, 16)
        self.assertEqual(stub.token_to_kv_pool.size, 16)
        self.assertEqual(stub.token_to_kv_pool_allocator.available_size(), 16)

    def test_paged_allocator_supports_cpu_extend_and_decode(self):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub

        stub = self._make_stub_for_allocator(4)
        MlxModelRunnerStub.initialize(stub, pre_model_load_memory=0)
        allocator = stub.token_to_kv_pool_allocator

        self.assertEqual(allocator.alloc(4).tolist(), [4, 5, 6, 7])
        prefix_lens = torch.tensor([3], dtype=torch.int64)
        seq_lens = torch.tensor([6], dtype=torch.int64)
        out = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            last_loc=torch.tensor([6], dtype=torch.int64),
            extend_num_tokens=3,
        )
        self.assertEqual(out.tolist(), [7, 8, 9])

        allocator.clear()
        seq_lens = torch.tensor([1, 2, 5], dtype=torch.int64)
        out = allocator.alloc_decode(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            last_loc=torch.tensor([0, 4, 7], dtype=torch.int64),
        )
        self.assertEqual(out.tolist(), [4, 5, 8])

    def test_init_kv_pool_skips_paged_cache_when_radix_cache_disabled(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = self._make_runner()
        runner.disable_radix_cache = True
        runner._pool_size = 15
        runner._paged_attention_block_size = 2
        runner._get_attn_config = MagicMock(return_value=(1, 4, mx.float16))
        req_to_token_pool = MagicMock()

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.MlxPagedKVCache"
        ) as paged_cache_cls:
            paged_cache = MagicMock()
            paged_cache.capacity = 16
            paged_cache_cls.return_value = paged_cache
            MlxModelRunner.init_kv_pool(runner, req_to_token_pool)

        paged_cache_cls.assert_not_called()
        self.assertIs(runner._req_to_token_pool, req_to_token_pool)
        self.assertIsNone(runner._paged_kv_cache)

    def test_prefill_uses_contiguous_cache_when_radix_cache_disabled(self):
        runner = self._make_runner()
        runner.disable_radix_cache = True
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context:
            next_token = runner.prefill("req", [8], [7, 8], [], [4], req_pool_idx=1)

        self.assertEqual(next_token, 2)
        self.assertEqual(runner._req_token_ids["req"], [7, 8, 2])
        self.assertIn("req", runner._req_caches)
        set_paged_context.assert_not_called()

    def test_no_radix_multi_request_decode_uses_batched_wrapper(self):
        runner = self._make_runner({"req0": [7], "req1": [8]})
        runner.disable_radix_cache = True
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]], dtype=mx.float32)
        )
        self._install_contiguous_caches(runner, {"req0": 1, "req1": 1})

        with patch.object(
            runner,
            "_ensure_no_radix_batched_wrapper",
            wraps=runner._ensure_no_radix_batched_wrapper,
        ) as ensure_wrapper:
            next_tokens = runner.decode_batch(["req0", "req1"])

        self.assertEqual(next_tokens, [2, 0])
        ensure_wrapper.assert_called_once()
        model_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in model_cache], [1, 1])
        self.assertEqual(
            [runner._req_caches["req0"][i].offset for i in range(runner._num_layers)],
            [2, 2],
        )
        self.assertEqual(
            [runner._req_caches["req1"][i].offset for i in range(runner._num_layers)],
            [2, 2],
        )

    def test_prefill_raises_if_model_does_not_populate_kv(self):
        runner = self._make_runner({})
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with self.assertRaisesRegex(RuntimeError, "populated contiguous KV cache"):
            runner.prefill("prefill_req", [8], [7, 8], [], [4], req_pool_idx=1)

        self.assertFalse(runner.has_request("prefill_req"))

    def test_prefill_does_not_register_request_after_model_error(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(side_effect=RuntimeError("prefill failed"))

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context:
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                runner.prefill(
                    req_id="prefill_req",
                    new_token_ids=[8],
                    full_token_ids=[7, 8],
                    prefix_slot_ids=[4],
                    new_slot_ids=[5],
                    req_pool_idx=1,
                )

        set_paged_context.assert_not_called()
        runner._paged_kv_cache.set_kv.assert_not_called()
        runner._paged_kv_cache.set_kv_all_layers.assert_not_called()
        self.assertFalse(runner.has_request("prefill_req"))

    def test_extend_uses_contiguous_cache_and_syncs_new_slots(self):
        runner = self._make_runner({"req": [7, 8, 9]})
        self._install_scheduler_slots(runner, {1: [4, 5, 6, 7]}, {"req": 1})
        self._install_contiguous_caches(runner, {"req": 2})
        runner._req_synced_offset = {"req": 2}
        runner.model = self._make_populating_model(
            mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_token = runner.extend("req", [10, 11], [6, 7])

        self.assertEqual(next_token, 2)
        set_paged_context.assert_not_called()
        clear_paged_context.assert_not_called()
        model_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in model_cache], [4, 4])
        runner._paged_kv_cache.set_kv.assert_not_called()
        slots_arg = runner._paged_kv_cache.set_kv_all_layers.call_args.args[0]
        self.assertEqual(slots_arg.tolist(), [6, 7])
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["use_metal"]
        )
        self.assertEqual(runner._req_token_ids["req"], [7, 8, 10, 11, 2])
        self.assertEqual(runner._req_synced_offset["req"], 4)

    def test_extend_raises_if_model_does_not_populate_kv(self):
        runner = self._make_runner({"req": [7, 8, 9]})
        self._install_scheduler_slots(runner, {1: [4, 5, 6, 7]}, {"req": 1})
        self._install_contiguous_caches(runner, {"req": 2})
        runner._req_synced_offset = {"req": 2}
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with self.assertRaisesRegex(RuntimeError, "populated contiguous KV cache"):
            runner.extend("req", [10, 11], [6, 7])

        self.assertEqual(runner._req_token_ids["req"], [7, 8, 9])
        self.assertEqual(runner._req_synced_offset["req"], 2)

    def test_extend_requires_contiguous_cache(self):
        runner = self._make_runner({"req": [7, 8, 9]})

        with self.assertRaisesRegex(RuntimeError, "requires contiguous KV cache"):
            runner.extend("req", [10, 11], [6, 7])

    def test_has_remove_and_clear_track_token_state_only(self):
        runner = self._make_runner({"req0": [7], "req1": [8]})
        runner._req_pool_idx = {"req0": 1, "req1": 2}
        runner._req_synced_offset = {"req0": 1, "req1": 1}
        runner._paged_kv_cache = self._make_paged_cache()

        self.assertTrue(runner.has_request("req0"))
        runner.remove_request("req0")
        self.assertFalse(runner.has_request("req0"))
        self.assertEqual(runner._req_token_ids, {"req1": [8]})
        self.assertEqual(runner._req_pool_idx, {"req1": 2})
        self.assertEqual(runner._req_synced_offset, {"req1": 1})

        runner.clear()
        self.assertEqual(runner._req_token_ids, {})
        self.assertEqual(runner._req_pool_idx, {})
        self.assertEqual(runner._req_synced_offset, {})
        runner._paged_kv_cache.clear.assert_called_once()

    def test_flush_all_decode_kv_syncs_pending_decode_range(self):
        runner = self._make_runner({"req": [7, 8, 9, 10]})
        self._install_scheduler_slots(runner, {1: [11, 12, 13, 14]}, {"req": 1})
        self._install_contiguous_caches(runner, {"req": 3})
        runner._req_synced_offset = {"req": 1}

        runner.flush_all_decode_kv()

        slots_arg, k_arg, v_arg = (
            runner._paged_kv_cache.set_kv_all_layers.call_args.args
        )
        self.assertEqual(slots_arg.tolist(), [12, 13])
        self.assertEqual(len(k_arg), runner._num_layers)
        self.assertEqual(len(v_arg), runner._num_layers)
        self.assertEqual(k_arg[0].shape, (2, 1, 2))
        self.assertEqual(v_arg[0].shape, (2, 1, 2))
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["eager"]
        )
        self.assertTrue(
            runner._paged_kv_cache.set_kv_all_layers.call_args.kwargs["use_metal"]
        )
        self.assertEqual(runner._req_synced_offset, {"req": 3})

    def test_flush_all_decode_kv_is_noop_without_pool(self):
        runner = self._make_runner({"req": [7]})

        self.assertIsNone(runner.flush_all_decode_kv())


class FakeReqToToken:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        row_indices, col_indices = key
        if isinstance(row_indices, int):
            if isinstance(col_indices, slice):
                data = self.rows[row_indices][col_indices]
            else:
                data = self.rows[row_indices][col_indices]
            return FakeReqToTokenTensor(data)
        if isinstance(col_indices, slice):
            data = [self.rows[row][col_indices] for row in row_indices]
        else:
            data = [
                self.rows[row][col]
                for row, col in zip(row_indices, col_indices, strict=True)
            ]
        return FakeReqToTokenTensor(data)


class FakeReqToTokenTensor:
    def __init__(self, data):
        self.data = data

    def to(self, dtype=None):
        return self

    def tolist(self):
        return self.data


if __name__ == "__main__":
    unittest.main()
