"""Unit tests for MLX paged attention routing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - platform-dependent optional dependency
    mx = None
    nn = None

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@unittest.skipIf(mx is None, "MLX is not available")
class TestMLXAttentionWrapper(CustomTestCase):
    def _make_cached_wrapper(self, seq_len=1):
        from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
            MLXAttentionWrapper,
        )

        B = 1
        n_heads = 2
        n_kv_heads = 1
        head_dim = 4
        dtype = mx.float32

        class Inner(nn.Module):
            n_heads = 2
            n_kv_heads = 1
            scale = 0.5

            def __init__(self):
                super().__init__()
                self.calls = []

            def __call__(self, x, mask=None, cache=None):
                self.calls.append((x, mask, cache))
                return mx.ones((B, seq_len, n_heads * head_dim), dtype=dtype) * 13

            def q_proj(self, x):
                return mx.ones((B, seq_len, n_heads * head_dim), dtype=dtype)

            def k_proj(self, x):
                return mx.ones((B, seq_len, n_kv_heads * head_dim), dtype=dtype) * 2

            def v_proj(self, x):
                return mx.ones((B, seq_len, n_kv_heads * head_dim), dtype=dtype) * 3

            def rope(self, x, offset):
                return x

            def o_proj(self, x):
                return x

        inner = Inner()
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        cache = object()
        x = mx.zeros((B, seq_len, 1), dtype=dtype)
        return wrapper, cache, x, inner

    def _make_paged_wrapper(
        self,
        batch_size=2,
        seq_len=1,
        is_prefill=False,
        projection_dtype=None,
        cache_dtype=None,
    ):
        from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
            MLXAttentionWrapper,
        )
        from sglang.srt.hardware_backend.mlx.kv_cache.paged_context import (
            PagedAttentionContext,
        )

        B = batch_size
        L = seq_len
        n_heads = 2
        n_kv_heads = 1
        head_dim = 4
        dtype = mx.float32 if projection_dtype is None else projection_dtype
        cache_dtype = dtype if cache_dtype is None else cache_dtype

        class Inner(nn.Module):
            n_heads = 2
            n_kv_heads = 1
            scale = 0.5

            def q_proj(self, x):
                return mx.ones((B, L, n_heads * head_dim), dtype=dtype)

            def k_proj(self, x):
                return mx.ones((B, L, n_kv_heads * head_dim), dtype=dtype) * 2

            def v_proj(self, x):
                return mx.ones((B, L, n_kv_heads * head_dim), dtype=dtype) * 3

            def rope(self, x, offset):
                return x

            def o_proj(self, x):
                return x

        class KVPool:
            k_buffer = [mx.zeros((16, n_kv_heads, head_dim), dtype=cache_dtype)]
            v_buffer = [mx.zeros((16, n_kv_heads, head_dim), dtype=cache_dtype)]

        if is_prefill:
            ctx = PagedAttentionContext(
                is_prefill=True,
                slot_mapping=list(range(1, B * L + 1)),
                block_tables=[list(range(L + 1)) for _ in range(B)],
                context_lens=[L + 1] * B,
                offsets=[1] * B,
                cu_seqlens=[i * L for i in range(B + 1)],
                max_seqlen_q=L,
                max_seqlen_k=L + 1,
                radix_prefix_lens=[1] * B,
                kv_pool=KVPool(),
            )
        else:
            context_lens = [2, 3][:B]
            ctx = PagedAttentionContext(
                is_prefill=False,
                slot_mapping=[1, 4][:B],
                block_tables=[[0, 1, 0], [2, 3, 4]][:B],
                context_lens=context_lens,
                offsets=[1, 2][:B],
                cu_seqlens=list(range(B + 1)),
                max_seqlen_q=1,
                max_seqlen_k=max(context_lens),
                radix_prefix_lens=[1, 2][:B],
                kv_pool=KVPool(),
            )

        wrapper = MLXAttentionWrapper(Inner(), layer_idx=0)
        x = mx.zeros((B, L, 1), dtype=dtype)
        return wrapper, ctx, x

    def test_paged_decode_path_uses_paged_kernel(self):
        wrapper, ctx, x = self._make_paged_wrapper()
        sentinel = mx.ones((2, 2, 1, 4), dtype=mx.float32) * 12
        metal_module = MagicMock()
        metal_module.paged_kv_scatter = MagicMock()
        metal_module.decode_attention_paged = MagicMock(return_value=sentinel)

        with patch.dict("sys.modules", {"sgl_kernel.metal": metal_module}):
            out = wrapper._paged_attention(x, ctx)

        expected = sentinel.transpose(0, 2, 1, 3).reshape(2, 1, -1)
        self.assertTrue(mx.allclose(out, expected))
        metal_module.paged_kv_scatter.assert_called_once()
        k_arg, v_arg, k_cache_arg, v_cache_arg, slots_arg = (
            metal_module.paged_kv_scatter.call_args.args
        )
        self.assertEqual(k_arg.shape, (2, 1, 4))
        self.assertEqual(v_arg.shape, (2, 1, 4))
        self.assertEqual(k_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(v_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(slots_arg.tolist(), [1, 4])
        self.assertEqual(ctx.kv_scatter_layer_ids, {0})
        self.assertTrue(ctx.has_scattered_all_layers(1))
        metal_module.decode_attention_paged.assert_called_once()
        (
            query_arg,
            k_cache_arg,
            v_cache_arg,
            block_tables_arg,
            context_lens_arg,
            scale_arg,
        ) = metal_module.decode_attention_paged.call_args.args
        self.assertEqual(query_arg.shape, (2, 2, 1, 4))
        self.assertEqual(k_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(v_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(block_tables_arg.tolist(), [[0, 1, 0], [2, 3, 4]])
        self.assertEqual(context_lens_arg.tolist(), [2, 3])
        self.assertEqual(scale_arg, 0.5)

    def test_paged_prefill_path_uses_paged_kernel(self):
        wrapper, ctx, x = self._make_paged_wrapper(
            batch_size=1, seq_len=3, is_prefill=True
        )
        sentinel = mx.ones((3, 2, 4), dtype=mx.float32) * 14
        metal_module = MagicMock()
        metal_module.paged_kv_scatter = MagicMock()
        metal_module.prefill_attention_paged = MagicMock(return_value=sentinel)

        with patch.dict("sys.modules", {"sgl_kernel.metal": metal_module}):
            out = wrapper._paged_attention(x, ctx)

        expected = (
            sentinel.reshape(1, 3, 2, 4)
            .transpose(0, 2, 1, 3)
            .transpose(0, 2, 1, 3)
            .reshape(1, 3, -1)
        )
        self.assertTrue(mx.allclose(out, expected))
        metal_module.paged_kv_scatter.assert_called_once()
        k_arg, v_arg, k_cache_arg, v_cache_arg, slots_arg = (
            metal_module.paged_kv_scatter.call_args.args
        )
        self.assertEqual(k_arg.shape, (3, 1, 4))
        self.assertEqual(v_arg.shape, (3, 1, 4))
        self.assertEqual(k_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(v_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(slots_arg.tolist(), [1, 2, 3])
        self.assertEqual(ctx.kv_scatter_layer_ids, {0})
        self.assertTrue(ctx.has_scattered_all_layers(1))
        metal_module.prefill_attention_paged.assert_called_once()
        (
            q_arg,
            k_arg,
            v_arg,
            k_cache_arg,
            v_cache_arg,
            block_tables_arg,
            prefix_lens_arg,
            cu_seqlens_arg,
            scale_arg,
        ) = metal_module.prefill_attention_paged.call_args.args
        self.assertEqual(q_arg.shape, (3, 2, 4))
        self.assertEqual(k_arg.shape, (3, 1, 4))
        self.assertEqual(v_arg.shape, (3, 1, 4))
        self.assertEqual(k_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(v_cache_arg.shape, (16, 1, 1, 4))
        self.assertEqual(block_tables_arg.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(prefix_lens_arg.tolist(), [1])
        self.assertEqual(cu_seqlens_arg.tolist(), [0, 3])
        self.assertEqual(scale_arg, 0.5)
        self.assertTrue(metal_module.prefill_attention_paged.call_args.kwargs["causal"])

    def test_paged_attention_casts_projection_tensors_to_cache_dtype(self):
        wrapper, ctx, x = self._make_paged_wrapper(
            batch_size=1, projection_dtype=mx.float32, cache_dtype=mx.float16
        )
        sentinel = mx.ones((1, 2, 1, 4), dtype=mx.float16) * 12
        metal_module = MagicMock()
        metal_module.paged_kv_scatter = MagicMock()
        metal_module.decode_attention_paged = MagicMock(return_value=sentinel)

        with patch.dict("sys.modules", {"sgl_kernel.metal": metal_module}):
            wrapper._paged_attention(x, ctx)

        k_arg, v_arg, k_cache_arg, v_cache_arg, _ = (
            metal_module.paged_kv_scatter.call_args.args
        )
        self.assertEqual(k_arg.dtype, mx.float16)
        self.assertEqual(v_arg.dtype, mx.float16)
        self.assertEqual(k_cache_arg.dtype, mx.float16)
        self.assertEqual(v_cache_arg.dtype, mx.float16)
        query_arg = metal_module.decode_attention_paged.call_args.args[0]
        self.assertEqual(query_arg.dtype, mx.float16)

    def test_paged_attention_does_not_mark_scatter_before_scatter_succeeds(self):
        wrapper, ctx, x = self._make_paged_wrapper()
        metal_module = MagicMock()
        metal_module.paged_kv_scatter = MagicMock(
            side_effect=RuntimeError("scatter failed")
        )
        metal_module.decode_attention_paged = MagicMock()

        with patch.dict("sys.modules", {"sgl_kernel.metal": metal_module}):
            with self.assertRaisesRegex(RuntimeError, "scatter failed"):
                wrapper._paged_attention(x, ctx)

        self.assertEqual(ctx.kv_scatter_layer_ids, set())
        metal_module.decode_attention_paged.assert_not_called()

    def test_paged_attention_ignores_legacy_cache_argument(self):
        wrapper, ctx, x = self._make_paged_wrapper(batch_size=1)
        legacy_cache = MagicMock()
        legacy_cache.offset = 5
        sentinel = mx.ones((1, 2, 1, 4), dtype=mx.float32) * 12
        metal_module = MagicMock()
        metal_module.paged_kv_scatter = MagicMock()
        metal_module.decode_attention_paged = MagicMock(return_value=sentinel)

        with patch.dict("sys.modules", {"sgl_kernel.metal": metal_module}):
            wrapper._paged_attention(x, ctx, cache=legacy_cache)

        self.assertEqual(legacy_cache.offset, 5)
        legacy_cache.update_and_fetch.assert_not_called()
        metal_module.paged_kv_scatter.assert_called_once()

    def test_paged_attention_requires_kv_pool(self):
        wrapper, ctx, x = self._make_paged_wrapper()
        ctx.kv_pool = None

        with self.assertRaisesRegex(ValueError, "requires a KV pool"):
            wrapper._paged_attention(x, ctx)

    def test_paged_attention_requires_decode_metadata(self):
        wrapper, ctx, x = self._make_paged_wrapper()
        ctx.block_tables = None

        with self.assertRaisesRegex(ValueError, "requires block tables"):
            wrapper._paged_attention(x, ctx)

    def test_paged_attention_requires_prefill_metadata(self):
        wrapper, ctx, x = self._make_paged_wrapper(
            batch_size=1, seq_len=3, is_prefill=True
        )
        ctx.radix_prefix_lens = None

        with self.assertRaisesRegex(ValueError, "requires radix prefix lengths"):
            wrapper._paged_attention(x, ctx)

    def test_paged_attention_rejects_non_3d_input(self):
        wrapper, ctx, _ = self._make_paged_wrapper()
        x = mx.zeros((2, 1), dtype=mx.float32)

        with self.assertRaisesRegex(ValueError, "expects input shape"):
            wrapper._paged_attention(x, ctx)

    def test_paged_attention_rejects_mismatched_batch_size(self):
        wrapper, ctx, _ = self._make_paged_wrapper()
        x = mx.zeros((1, 1, 1), dtype=mx.float32)

        with self.assertRaisesRegex(ValueError, "batch size must match"):
            wrapper._paged_attention(x, ctx)

    def test_call_uses_paged_context_without_env_flag(self):
        from sglang.srt.hardware_backend.mlx.kv_cache.paged_context import (
            clear_paged_context,
            set_paged_context,
        )

        wrapper, ctx, x = self._make_paged_wrapper(batch_size=1)
        sentinel = mx.ones((1, 1, 8), dtype=mx.float32) * 17

        try:
            set_paged_context(ctx)
            with patch.object(
                wrapper, "_paged_attention", return_value=sentinel
            ) as paged_attention:
                out = wrapper(x, cache="cache")
        finally:
            clear_paged_context()

        self.assertTrue(mx.allclose(out, sentinel))
        paged_attention.assert_called_once_with(x, ctx, "cache")

    def test_non_paged_cached_call_delegates_to_inner(self):
        wrapper, cache, x, inner = self._make_cached_wrapper(seq_len=1)

        out = wrapper(x, cache=cache)

        self.assertTrue(mx.allclose(out, mx.ones((1, 1, 8), dtype=mx.float32) * 13))
        self.assertEqual(len(inner.calls), 1)
        self.assertIs(inner.calls[0][2], cache)


if __name__ == "__main__":
    unittest.main()
