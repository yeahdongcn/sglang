"""Unit tests for MLX block-paged KV cache primitives."""

from __future__ import annotations

import unittest

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - platform-dependent optional dependency
    mx = None

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@unittest.skipIf(mx is None, "MLX is not available")
class TestMlxPagedKVCache(CustomTestCase):
    def test_allocates_block_paged_buffers(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=2,
            num_blocks=3,
            block_size=4,
            n_kv_heads=2,
            head_dim=8,
            dtype=mx.float32,
        )

        self.assertEqual(cache.capacity, 12)
        self.assertEqual(cache.k_buffer[0].shape, (3, 4, 2, 8))
        self.assertEqual(cache.v_buffer[1].shape, (3, 4, 2, 8))
        self.assertEqual(len(cache.state), 4)

    def test_normalizes_unsupported_dtype_to_float16(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import (
            MlxPagedKVCache,
            normalize_mlx_metal_dtype,
        )

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.bfloat16,
        )

        self.assertEqual(normalize_mlx_metal_dtype(mx.bfloat16), mx.float16)
        self.assertEqual(cache.dtype, mx.float16)
        self.assertEqual(cache.k_buffer[0].dtype, mx.float16)

    def test_slot_mapping_writes_and_reads_across_blocks(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=2,
            block_size=4,
            n_kv_heads=1,
            head_dim=2,
            dtype=mx.float32,
        )
        slots = mx.array([0, 3, 4, 7], dtype=mx.int32)
        k = mx.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]])
        v = k + 10

        cache.set_kv(0, slots, k, v)
        out_k, out_v = cache.get_kv(0, slots)

        self.assertTrue(mx.allclose(out_k, k))
        self.assertTrue(mx.allclose(out_v, v))
        block_ids, block_offsets = cache.slot_to_block_offset(slots)
        self.assertEqual(block_ids.tolist(), [0, 0, 1, 1])
        self.assertEqual(block_offsets.tolist(), [0, 3, 0, 3])

    def test_set_kv_all_layers(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=2,
            num_blocks=2,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([1, 2], dtype=mx.int32)
        k_all = mx.array([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
        v_all = k_all + 20

        cache.set_kv_all_layers(slots, k_all, v_all)

        self.assertTrue(mx.allclose(cache.get_kv(0, slots)[0], k_all[0]))
        self.assertTrue(mx.allclose(cache.get_kv(1, slots)[1], v_all[1]))

    def test_gather_blocks_from_block_tables(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=3,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int32)
        k = mx.arange(6, dtype=mx.float32).reshape(6, 1, 1)
        v = k + 10
        block_tables = mx.array([[0, 2], [1, 0]], dtype=mx.int32)

        cache.set_kv(0, slots, k, v)
        k_blocks, v_blocks = cache.gather_blocks(0, block_tables)

        self.assertEqual(k_blocks.shape, (2, 2, 2, 1, 1))
        self.assertTrue(mx.allclose(k_blocks[0], cache.k_buffer[0][[0, 2]]))
        self.assertTrue(mx.allclose(v_blocks[1], cache.v_buffer[0][[1, 0]]))

    def test_gather_block_table_tokens_returns_valid_mask(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=3,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int32)
        k = mx.arange(6, dtype=mx.float32).reshape(6, 1, 1)
        v = k + 10
        block_tables = mx.array([[0, 2], [1, 0]], dtype=mx.int32)

        cache.set_kv(0, slots, k, v)
        k_blocks, v_blocks, valid_mask = cache.gather_block_table_tokens(
            0, block_tables, [3, 1]
        )

        self.assertEqual(k_blocks.shape, (2, 2, 2, 1, 1))
        self.assertEqual(v_blocks.shape, (2, 2, 2, 1, 1))
        self.assertEqual(
            k_blocks.reshape(2, 4, 1, 1)[:, :, 0, 0].tolist(),
            [[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 0.0, 1.0]],
        )
        self.assertEqual(
            v_blocks.reshape(2, 4, 1, 1)[:, :, 0, 0].tolist(),
            [[10.0, 11.0, 14.0, 15.0], [12.0, 13.0, 10.0, 11.0]],
        )
        self.assertEqual(
            valid_mask.tolist(),
            [[True, True, True, False], [True, False, False, False]],
        )

    def test_rejects_context_lens_exceeding_block_table_capacity(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=2,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )

        with self.assertRaisesRegex(ValueError, "fit within gathered block tables"):
            cache.gather_block_table_tokens(0, mx.array([[0, 1]], dtype=mx.int32), [5])

    def test_reset_slots_clears_only_selected_slots(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=2,
            num_blocks=2,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([0, 1, 2, 3], dtype=mx.int32)
        k_all = mx.array(
            [[[[1.0]], [[2.0]], [[3.0]], [[4.0]]], [[[5.0]], [[6.0]], [[7.0]], [[8.0]]]]
        )
        v_all = k_all + 10

        cache.set_kv_all_layers(slots, k_all, v_all)
        cache.reset_slots(mx.array([1, 2], dtype=mx.int32))

        self.assertEqual(
            cache.get_kv(0, slots)[0][:, 0, 0].tolist(), [1.0, 0.0, 0.0, 4.0]
        )
        self.assertEqual(
            cache.get_kv(0, slots)[1][:, 0, 0].tolist(), [11.0, 0.0, 0.0, 14.0]
        )
        self.assertEqual(
            cache.get_kv(1, slots)[0][:, 0, 0].tolist(), [5.0, 0.0, 0.0, 8.0]
        )
        self.assertEqual(
            cache.get_kv(1, slots)[1][:, 0, 0].tolist(), [15.0, 0.0, 0.0, 18.0]
        )

    def test_reset_blocks_clears_whole_blocks(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=3,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int32)
        k = mx.arange(1, 7, dtype=mx.float32).reshape(6, 1, 1)
        v = k + 10

        cache.set_kv(0, slots, k, v)
        cache.reset_blocks(mx.array([1], dtype=mx.int32))

        self.assertEqual(
            cache.get_kv(0, slots)[0][:, 0, 0].tolist(), [1.0, 2.0, 0.0, 0.0, 5.0, 6.0]
        )
        self.assertEqual(
            cache.get_kv(0, slots)[1][:, 0, 0].tolist(),
            [11.0, 12.0, 0.0, 0.0, 15.0, 16.0],
        )

    def test_reset_empty_slots_and_blocks_is_noop(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
            dtype=mx.float32,
        )
        slots = mx.array([0], dtype=mx.int32)
        k = mx.array([[[3.0]]], dtype=mx.float32)

        cache.set_kv(0, slots, k, k)
        cache.reset_slots(mx.array([], dtype=mx.int32))
        cache.reset_blocks(mx.array([], dtype=mx.int32))

        self.assertEqual(cache.get_kv(0, slots)[0][:, 0, 0].tolist(), [3.0])

    def test_rejects_non_1d_reset_blocks(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )

        with self.assertRaisesRegex(ValueError, "block IDs must be a 1-D array"):
            cache.reset_blocks(mx.array([[0]], dtype=mx.int32))

    def test_rejects_out_of_range_reset_blocks(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )

        with self.assertRaisesRegex(ValueError, "within paged KV cache block capacity"):
            cache.reset_blocks(mx.array([1], dtype=mx.int32))

    def test_rejects_out_of_range_block_tables(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=2,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )

        with self.assertRaisesRegex(ValueError, "within paged KV cache block capacity"):
            cache.gather_blocks(0, mx.array([[0, 2]], dtype=mx.int32))

    def test_rejects_context_lens_batch_mismatch(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=2,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )

        with self.assertRaisesRegex(ValueError, "context lengths must match"):
            cache.gather_block_table_tokens(
                0, mx.array([[0], [1]], dtype=mx.int32), [1]
            )

    def test_rejects_out_of_range_slots(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            n_kv_heads=1,
            head_dim=1,
        )
        k = mx.ones((1, 1, 1), dtype=mx.float16)

        with self.assertRaisesRegex(ValueError, "within paged KV cache capacity"):
            cache.set_kv(0, mx.array([2], dtype=mx.int32), k, k)

    def test_rejects_singleton_kv_for_multiple_slots(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=4,
            n_kv_heads=1,
            head_dim=1,
        )
        slots = mx.array([0, 1], dtype=mx.int32)
        k = mx.ones((1, 1, 1), dtype=mx.float16)

        with self.assertRaisesRegex(ValueError, "token count must match slot count"):
            cache.set_kv(0, slots, k, k)

    def test_rejects_non_1d_slots(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=4,
            n_kv_heads=1,
            head_dim=1,
        )
        slots = mx.array([[0, 1]], dtype=mx.int32)
        k = mx.ones((2, 1, 1), dtype=mx.float16)

        with self.assertRaisesRegex(ValueError, "slot IDs must be a 1-D array"):
            cache.set_kv(0, slots, k, k)

    def test_rejects_mismatched_k_and_v_token_counts(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache

        cache = MlxPagedKVCache(
            num_layers=1,
            num_blocks=1,
            block_size=4,
            n_kv_heads=1,
            head_dim=1,
        )
        slots = mx.array([0, 1], dtype=mx.int32)
        k = mx.ones((2, 1, 1), dtype=mx.float16)
        v = mx.ones((1, 1, 1), dtype=mx.float16)

        with self.assertRaisesRegex(ValueError, "token count must match slot count"):
            cache.set_kv(0, slots, k, v)


if __name__ == "__main__":
    unittest.main()
