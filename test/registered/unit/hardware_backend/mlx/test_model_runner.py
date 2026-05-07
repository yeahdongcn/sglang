"""Unit tests for MLX model runner paged attention routing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

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
        runner._paged_kv_cache = None
        runner._req_to_token_pool = None
        runner._paged_attention_block_size = 2
        runner.disable_radix_cache = False
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )
        return runner

    def _make_paged_cache(self, block_size=2):
        paged_cache = MagicMock()
        paged_cache.block_size = block_size
        return paged_cache

    def _install_scheduler_slots(self, runner, rows, pool_indices):
        runner._paged_kv_cache = self._make_paged_cache()
        runner._req_to_token_pool = MagicMock()
        runner._req_to_token_pool.req_to_token = FakeReqToToken(rows)
        runner._req_pool_idx = pool_indices

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

    def test_decode_batch_requires_paged_context(self):
        runner = self._make_runner({"req": [7]})

        with self.assertRaisesRegex(RuntimeError, "requires paged KV cache for decode"):
            runner.decode_batch(["req"])

    def test_decode_batch_sets_and_clears_paged_context(self):
        runner = self._make_runner({"req0": [7], "req1": [8, 9]})
        runner.model = MagicMock(
            return_value=mx.array(
                [[[0.0, 1.0, 2.0]], [[3.0, 2.0, 1.0]]], dtype=mx.float32
            )
        )
        self._install_scheduler_slots(
            runner,
            {
                1: [11, 12, 13, 14],
                3: [31, 32, 33, 34],
            },
            {"req0": 1, "req1": 3},
        )
        runner._req_synced_offset = {"req0": 1, "req1": 1}

        def mark_all_layers(ctx):
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context",
            side_effect=mark_all_layers,
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_tokens = runner.decode_batch(["req0", "req1"])

        self.assertEqual(next_tokens, [2, 0])
        set_paged_context.assert_called_once()
        paged_ctx = set_paged_context.call_args.args[0]
        self.assertEqual(paged_ctx.slot_mapping.tolist(), [11, 32])
        self.assertEqual(paged_ctx.block_tables.tolist(), [[5], [15]])
        self.assertEqual(paged_ctx.context_lens.tolist(), [1, 2])
        self.assertEqual(paged_ctx.offsets.tolist(), [0, 1])
        self.assertEqual(paged_ctx.radix_prefix_lens.tolist(), [0, 1])
        clear_paged_context.assert_called_once()
        shim_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in shim_cache], [1, 1])
        self.assertEqual(runner._req_token_ids, {"req0": [7, 2], "req1": [8, 9, 0]})
        self.assertEqual(runner._req_synced_offset, {"req0": 1, "req1": 2})

    def test_decode_batch_raises_if_native_scatter_missing(self):
        runner = self._make_runner({"req": [7]})
        self._install_scheduler_slots(runner, {1: [11, 12]}, {"req": 1})
        runner._req_synced_offset = {"req": 0}

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ), patch("sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"):
            with self.assertRaisesRegex(RuntimeError, "did not scatter KV"):
                runner.decode_batch(["req"])

        self.assertEqual(runner._req_token_ids, {"req": [7]})
        self.assertEqual(runner._req_synced_offset, {"req": 0})

    def test_decode_batch_clears_paged_context_after_model_error(self):
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

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                runner.decode_batch(["req0", "req1"])

        clear_paged_context.assert_called_once()

    def test_prefill_sets_and_clears_paged_context(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        def mark_all_layers(ctx):
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context",
            side_effect=mark_all_layers,
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
        set_paged_context.assert_called_once()
        paged_ctx = set_paged_context.call_args.args[0]
        self.assertEqual(paged_ctx.slot_mapping.tolist(), [6, 7])
        self.assertEqual(paged_ctx.block_tables.tolist(), [[2, 3]])
        self.assertEqual(paged_ctx.context_lens.tolist(), [4])
        self.assertEqual(paged_ctx.offsets.tolist(), [2])
        self.assertEqual(paged_ctx.radix_prefix_lens.tolist(), [2])
        clear_paged_context.assert_called_once()
        shim_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in shim_cache], [2, 2])
        self.assertEqual(runner._req_token_ids["prefill_req"], [7, 8, 9, 2])
        self.assertEqual(runner._req_pool_idx["prefill_req"], 1)
        self.assertEqual(runner._req_synced_offset["prefill_req"], 4)

    def test_init_kv_pool_creates_paged_cache_when_radix_cache_disabled(self):
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

        paged_cache_cls.assert_called_once_with(
            num_blocks=8,
            block_size=2,
            num_layers=2,
            n_kv_heads=1,
            head_dim=4,
            dtype=mx.float16,
        )
        self.assertIs(runner._req_to_token_pool, req_to_token_pool)
        self.assertIs(runner._paged_kv_cache, paged_cache)

    def test_prefill_uses_paged_cache_when_radix_cache_disabled(self):
        runner = self._make_runner()
        runner.disable_radix_cache = True
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        def mark_all_layers(ctx):
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context",
            side_effect=mark_all_layers,
        ), patch("sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"):
            next_token = runner.prefill("req", [8], [7, 8], [], [4], req_pool_idx=1)

        self.assertEqual(next_token, 2)
        self.assertEqual(runner._req_token_ids["req"], [7, 8, 2])

    def test_prefill_raises_if_native_scatter_missing(self):
        runner = self._make_runner({})
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ), patch("sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"):
            with self.assertRaisesRegex(RuntimeError, "did not scatter KV"):
                runner.prefill("prefill_req", [8], [7, 8], [], [4], req_pool_idx=1)

        self.assertFalse(runner.has_request("prefill_req"))

    def test_prefill_clears_paged_context_after_model_error(self):
        runner = self._make_runner()
        runner._paged_kv_cache = self._make_paged_cache()
        runner.model = MagicMock(side_effect=RuntimeError("prefill failed"))

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                runner.prefill(
                    req_id="prefill_req",
                    new_token_ids=[8],
                    full_token_ids=[7, 8],
                    prefix_slot_ids=[4],
                    new_slot_ids=[5],
                    req_pool_idx=1,
                )

        clear_paged_context.assert_called_once()

    def test_extend_sets_and_clears_paged_context(self):
        runner = self._make_runner({"req": [7, 8, 9]})
        self._install_scheduler_slots(runner, {1: [4, 5, 6, 7]}, {"req": 1})
        runner._req_synced_offset = {"req": 2}
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        def mark_all_layers(ctx):
            for layer_idx in range(runner._num_layers):
                ctx.mark_kv_scattered(layer_idx)

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context",
            side_effect=mark_all_layers,
        ) as set_paged_context, patch(
            "sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"
        ) as clear_paged_context:
            next_token = runner.extend("req", [10, 11], [6, 7])

        self.assertEqual(next_token, 2)
        set_paged_context.assert_called_once()
        paged_ctx = set_paged_context.call_args.args[0]
        self.assertEqual(paged_ctx.slot_mapping.tolist(), [6, 7])
        self.assertEqual(paged_ctx.block_tables.tolist(), [[2, 3]])
        self.assertEqual(paged_ctx.context_lens.tolist(), [4])
        self.assertEqual(paged_ctx.offsets.tolist(), [2])
        self.assertEqual(paged_ctx.radix_prefix_lens.tolist(), [2])
        clear_paged_context.assert_called_once()
        shim_cache = runner.model.call_args.kwargs["cache"]
        self.assertEqual([cache.offset for cache in shim_cache], [2, 2])
        self.assertEqual(runner._req_token_ids["req"], [7, 8, 10, 11, 2])
        self.assertEqual(runner._req_synced_offset["req"], 4)

    def test_extend_raises_if_native_scatter_missing(self):
        runner = self._make_runner({"req": [7, 8, 9]})
        self._install_scheduler_slots(runner, {1: [4, 5, 6, 7]}, {"req": 1})
        runner._req_synced_offset = {"req": 2}
        runner.model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0]]], dtype=mx.float32)
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.set_paged_context"
        ), patch("sglang.srt.hardware_backend.mlx.model_runner.clear_paged_context"):
            with self.assertRaisesRegex(RuntimeError, "did not scatter KV"):
                runner.extend("req", [10, 11], [6, 7])

        self.assertEqual(runner._req_token_ids["req"], [7, 8, 9])
        self.assertEqual(runner._req_synced_offset["req"], 2)

    def test_extend_requires_paged_context(self):
        runner = self._make_runner({"req": [7, 8, 9]})

        with self.assertRaisesRegex(RuntimeError, "requires paged KV cache for extend"):
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

    def test_flush_all_decode_kv_is_noop(self):
        runner = self._make_runner({"req": [7]})

        self.assertIsNone(runner.flush_all_decode_kv())


class FakeReqToToken:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        row_indices, col_indices = key
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
