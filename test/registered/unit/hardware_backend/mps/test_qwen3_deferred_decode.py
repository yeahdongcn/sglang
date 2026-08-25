"""Small-memory checks for MLX deferred-commit Qwen3 decode attention."""

from __future__ import annotations

import gc
import importlib.util
import unittest

import torch
from packaging.version import Version

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=8, suite="stage-a-unit-test-mps")

_HAS_SUPPORTED_RUNTIME = (
    importlib.util.find_spec("mlx") is not None
    and torch.backends.mps.is_available()
    and Version(torch.__version__) >= Version("2.13.0")
)


@unittest.skipUnless(_HAS_SUPPORTED_RUNTIME, "requires Torch 2.13 and MLX on MPS")
class TestQwen3MlxDeferredDecode(unittest.TestCase):
    @staticmethod
    def _release_case_memory():
        """Retire both framework queues/caches between long-context cases."""
        import mlx.core as mx

        mx.synchronize()
        torch.mps.synchronize()
        gc.collect()
        mx.clear_cache()
        torch.mps.empty_cache()

    def _assert_cold_prefill_causal_gqa(self, token_count: int):
        from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (
            _mlx_causal_gqa,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20260802 + token_count)
        q = torch.randn(token_count, 16, 128, device="mps", dtype=torch.bfloat16)
        k = torch.randn(token_count, 8, 128, device="mps", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        output = mlx_call(_mlx_causal_gqa, q, k, v, device="mps")
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None, ...],
            k.transpose(0, 1)[None, ...],
            v.transpose(0, 1)[None, ...],
            scale=128**-0.5,
            is_causal=True,
            enable_gqa=True,
        )[0].transpose(0, 1)
        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0.008, rtol=0.03)

    def test_production_dimension_warmup_executes_the_specialized_kernel(self):
        from sglang.kernels.ops.attention.qwen3_mlx import (
            warmup_qwen3_radix_decode_deferred,
        )

        try:
            # These nontrivial constants exercise the same specialization path
            # used by provider.start without allocating production-sized table
            # or pool storage.
            warmup_qwen3_radix_decode_deferred(
                request_rows=8,
                table_stride=32,
                pool_slots=1024,
            )
        finally:
            self._release_case_memory()

    def test_cold_prefill_causal_gqa_matches_torch(self):
        # The 511/512/513 boundary covers the production cold-prefill cutoff,
        # while 1024 protects the same primitive if that cutoff is raised.
        for token_count in (1, 2, 17, 511, 512, 513, 1024):
            with self.subTest(token_count=token_count):
                try:
                    self._assert_cold_prefill_causal_gqa(token_count)
                finally:
                    self._release_case_memory()

    def _assert_long_deferred_decode(self, sequence_length: int):
        from sglang.kernels.ops.attention import qwen3_radix_decode_deferred
        from sglang.srt.utils.tensor_bridge import (
            borrow_torch_tensors,
            mlx_to_torch,
        )

        torch.manual_seed(20260801 + sequence_length)
        pool_slots = sequence_length + 37
        prefix_length = sequence_length - 1
        q = torch.randn(1, 16, 128, device="mps", dtype=torch.bfloat16)
        current_k = torch.randn(1, 8, 128, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(pool_slots, 8, 128, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)

        # Use a non-contiguous logical-to-physical mapping and poison the final
        # table entry. The deferred kernel must gather the prefix and use the
        # uncommitted current K/V for the last logical token.
        slots_cpu = torch.randperm(pool_slots)[:prefix_length]
        req_to_token = torch.full(
            (1, sequence_length), -1, device="mps", dtype=torch.int32
        )
        req_to_token[0, :prefix_length] = slots_cpu.to(device="mps", dtype=torch.int32)
        req_to_token[0, -1] = pool_slots + 11
        req_pool_indices = torch.zeros(1, device="mps", dtype=torch.int64)
        seq_lens = torch.tensor([sequence_length], device="mps", dtype=torch.int64)

        torch.mps.synchronize()
        views = borrow_torch_tensors(
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            synchronize=False,
        )
        output = mlx_to_torch(
            qwen3_radix_decode_deferred(*(view.array for view in views)),
            device="mps",
        )

        slots = slots_cpu.to(device="mps", dtype=torch.int64)
        keys = torch.cat((k_pool[slots], current_k), dim=0).float()
        values = torch.cat((v_pool[slots], current_v), dim=0).float()
        grouped_q = q[0].float().reshape(8, 2, 128)
        logits = torch.einsum("hgd,thd->hgt", grouped_q, keys) * (128**-0.5)
        probabilities = torch.softmax(logits, dim=-1)
        reference = (
            torch.einsum("hgt,thd->hgd", probabilities, values)
            .reshape(1, 16, 128)
            .to(torch.bfloat16)
        )

        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0.008, rtol=0.03)

    def test_long_context_deferred_decode_matches_fp32_torch(self):
        for sequence_length in (512, 513, 1024):
            with self.subTest(sequence_length=sequence_length):
                try:
                    self._assert_long_deferred_decode(sequence_length)
                finally:
                    self._release_case_memory()

    def test_qk_rmsnorm_rope_neox_boundary_positions_match_torch(self):
        from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (
            _rms_norm,
            _rope_neox,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call_multi

        epsilon = 1e-6
        positions_cpu = torch.tensor([0, 511, 512, 1023], dtype=torch.int64)
        inv_freq = 1.0 / (
            1_000_000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
        )
        frequencies = torch.outer(torch.arange(1024, dtype=torch.float32), inv_freq)
        cos_sin_cpu = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(
            torch.bfloat16
        )

        torch.manual_seed(20260803)
        q_cpu = torch.randn(4, 16, 128).to(torch.bfloat16)
        k_cpu = torch.randn(4, 8, 128).to(torch.bfloat16)
        q_weight_cpu = (1 + 0.05 * torch.randn(128)).to(torch.bfloat16)
        k_weight_cpu = (1 + 0.05 * torch.randn(128)).to(torch.bfloat16)
        q = q_cpu.to("mps")
        k = k_cpu.to("mps")
        q_weight = q_weight_cpu.to("mps")
        k_weight = k_weight_cpu.to("mps")
        cos_sin = cos_sin_cpu.to("mps")
        positions = positions_cpu.to("mps")

        def torch_reference(value, weight):
            value_fp32 = value.float()
            inverse_rms = torch.rsqrt(
                value_fp32.square().mean(dim=-1, keepdim=True) + epsilon
            )
            normalized = (value_fp32 * inverse_rms * weight.float()).to(torch.bfloat16)
            selected = cos_sin.index_select(0, positions)
            cosine, sine = selected.chunk(2, dim=-1)
            first, second = normalized.chunk(2, dim=-1)
            return torch.cat(
                (
                    first * cosine[:, None, :] - second * sine[:, None, :],
                    second * cosine[:, None, :] + first * sine[:, None, :],
                ),
                dim=-1,
            ).to(torch.bfloat16)

        q_reference = torch_reference(q, q_weight)
        k_reference = torch_reference(k, k_weight)

        def mlx_qk_norm_rope(
            q_array, k_array, q_weight_array, k_weight_array, cache, pos
        ):
            q_normed = _rms_norm(q_array, q_weight_array, epsilon)
            k_normed = _rms_norm(k_array, k_weight_array, epsilon)
            return (
                _rope_neox(q_normed, cache, pos),
                _rope_neox(k_normed, cache, pos),
            )

        try:
            q_output, k_output = mlx_call_multi(
                mlx_qk_norm_rope,
                q,
                k,
                q_weight,
                k_weight,
                cos_sin,
                positions,
                device="mps",
            )
            torch.mps.synchronize()
            torch.testing.assert_close(
                q_output.cpu(), q_reference.cpu(), atol=0.008, rtol=0.03
            )
            torch.testing.assert_close(
                k_output.cpu(), k_reference.cpu(), atol=0.008, rtol=0.03
            )
        finally:
            self._release_case_memory()

    def test_current_token_bypasses_uncommitted_pool_entry(self):
        from sglang.kernels.ops.attention import (
            qwen3_radix_decode_deferred,
        )
        from sglang.srt.utils.tensor_bridge import (
            borrow_torch_tensors,
            mlx_to_torch,
        )

        torch.manual_seed(20260801)
        batch_size = 4
        pool_slots = 31
        table_stride = 8
        # Every adjacent pair is owned by one KV head.  Make the sibling
        # queries deliberately different so a GQA kernel cannot accidentally
        # reuse the first query's online-softmax state for the second.
        q_per_kv_head = torch.randn(
            batch_size, 8, 128, device="mps", dtype=torch.bfloat16
        )
        q = torch.stack((q_per_kv_head, -q_per_kv_head), dim=2).reshape(
            batch_size, 16, 128
        )
        current_k = torch.randn(batch_size, 8, 128, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(pool_slots, 8, 128, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)
        req_to_token = torch.zeros(3, table_stride, device="mps", dtype=torch.int32)
        req_to_token[1] = torch.tensor(
            [-17, 21, 7, 18, 2, 29, 11, 5], device="mps", dtype=torch.int32
        )
        req_to_token[2] = torch.tensor(
            [14, 4, 23, 1, 27, 8, 38, 6], device="mps", dtype=torch.int32
        )
        req_pool_indices = torch.tensor([1, 2, -1, 0], device="mps", dtype=torch.int64)
        seq_lens = torch.tensor([1, 7, 4, 0], device="mps", dtype=torch.int64)

        original_k_pool = k_pool.clone()
        original_v_pool = v_pool.clone()
        torch.mps.synchronize()
        views = borrow_torch_tensors(
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            synchronize=False,
        )
        result = qwen3_radix_decode_deferred(*(view.array for view in views))
        output = mlx_to_torch(result, device="mps")

        references = []
        scale = 128**-0.5
        for batch_index, sequence_length in enumerate((1, 7)):
            request = int(req_pool_indices[batch_index].item())
            slots = req_to_token[request, : sequence_length - 1].long()
            keys = torch.cat(
                (k_pool[slots], current_k[batch_index : batch_index + 1]), dim=0
            )
            values = torch.cat(
                (v_pool[slots], current_v[batch_index : batch_index + 1]), dim=0
            )
            keys = keys.repeat_interleave(2, dim=1)
            values = values.repeat_interleave(2, dim=1)
            logits = (
                torch.einsum("hd,thd->ht", q[batch_index].float(), keys.float()) * scale
            )
            probabilities = torch.softmax(logits, dim=-1)
            references.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        # Invalid request metadata and a non-positive sequence length are
        # threadgroup-uniform early exits.  Both query heads paired with every
        # KV head must be initialized, not merely the first sibling.
        references.extend(
            torch.zeros(16, 128, device="mps", dtype=torch.bfloat16) for _ in range(2)
        )
        reference = torch.stack(references)

        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0.004, rtol=0.02)
        torch.testing.assert_close(
            output[2:].cpu(),
            torch.zeros_like(output[2:]).cpu(),
            atol=0.0,
            rtol=0.0,
        )
        # The final logical entries deliberately contain invalid/stale pool
        # indices.  The kernel must use current K/V instead of reading those
        # entries, including the no-prefix sequence_length == 1 case.
        torch.testing.assert_close(k_pool.cpu(), original_k_pool.cpu())
        torch.testing.assert_close(v_pool.cpu(), original_v_pool.cpu())


if __name__ == "__main__":
    unittest.main()
