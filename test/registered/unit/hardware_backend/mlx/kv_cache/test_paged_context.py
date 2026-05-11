"""Unit tests for MLX paged attention context primitives."""

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
class TestPagedAttentionContext(CustomTestCase):
    def tearDown(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import clear_paged_context

        clear_paged_context()

    def test_context_normalizes_scheduler_metadata(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import PagedAttentionContext

        ctx = PagedAttentionContext(
            is_prefill=True,
            slot_mapping=[3, 4, 8],
            block_tables=[[0, 1], [2, 3]],
            context_lens=[2, 3],
            offsets=[0, 2],
            cu_seqlens=[0, 2, 5],
            max_seqlen_q=3,
            max_seqlen_k=5,
            radix_prefix_lens=[1, 2],
        )

        self.assertTrue(ctx.is_prefill)
        self.assertEqual(ctx.batch_size, 2)
        self.assertEqual(ctx.slot_mapping.dtype, mx.int32)
        self.assertEqual(ctx.block_tables.tolist(), [[0, 1], [2, 3]])
        self.assertEqual(ctx.context_lens.tolist(), [2, 3])
        self.assertEqual(ctx.offsets.tolist(), [0, 2])
        self.assertEqual(ctx.cu_seqlens.tolist(), [0, 2, 5])
        self.assertEqual(ctx.radix_prefix_lens.tolist(), [1, 2])
        self.assertEqual(ctx.max_seqlen_q, 3)
        self.assertEqual(ctx.max_seqlen_k, 5)

    def test_scatter_tracking_records_layers(self):
        from sglang.srt.hardware_backend.mlx.kv_cache import PagedAttentionContext

        ctx = PagedAttentionContext(
            is_prefill=False, slot_mapping=[1], context_lens=[2]
        )

        self.assertFalse(ctx.has_scattered_all_layers(2))
        ctx.mark_kv_scattered(0)
        self.assertEqual(ctx.kv_scatter_layer_ids, {0})
        self.assertFalse(ctx.has_scattered_all_layers(2))
        ctx.mark_kv_scattered(1)
        self.assertTrue(ctx.has_scattered_all_layers(2))

        from sglang.srt.hardware_backend.mlx.kv_cache import (
            PagedAttentionContext,
            clear_paged_context,
            get_paged_context,
            set_paged_context,
        )

        ctx = PagedAttentionContext(
            is_prefill=False, slot_mapping=[1], context_lens=[4]
        )

        self.assertIsNone(get_paged_context())
        set_paged_context(ctx)
        self.assertIs(get_paged_context(), ctx)
        clear_paged_context()
        self.assertIsNone(get_paged_context())


if __name__ == "__main__":
    unittest.main()
