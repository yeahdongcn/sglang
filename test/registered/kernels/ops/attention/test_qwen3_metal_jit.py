"""Correctness and ownership tests for Qwen3 Torch-stream Metal JIT kernels."""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest import mock

import torch
from packaging.version import Version

from sglang.kernels import metal
from sglang.kernels.ops.attention.qwen3_mps import (
    QWEN3_06B_METAL_SPEC,
    qwen3_qknorm_rope_store,
    qwen3_radix_decode,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_SUPPORTED_RUNTIME = torch.backends.mps.is_available() and Version(
    torch.__version__
) >= Version("2.13.0")


class _TrackingLibrary:
    def __init__(self):
        self.lookups = []

    def __getattr__(self, name):
        self.lookups.append(name)
        return object()


class TestQwen3MetalJitWarmup(unittest.TestCase):
    def test_attention_namespace_import_is_compile_and_routing_lazy(self):
        program = r"""
import sys
from unittest import mock

import torch

with (
    mock.patch.object(torch.mps, "compile_shader", side_effect=AssertionError, create=True),
    mock.patch.object(torch.mps, "load_metallib", side_effect=AssertionError, create=True),
):
    import sglang.kernels.ops.attention as attention

assert attention.qwen3_qknorm_rope_store is not None
assert "sglang.kernels.ops.attention._qwen3_metal_jit" not in sys.modules
assert "sglang.srt.hardware_backend.mps.model_ops.plan" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_attention_namespace_registers_torch_and_jit_only(self):
        import sglang.kernels.ops.attention as attention
        from sglang.kernels import registry

        for name in (
            "Qwen3QKNormRopeStoreOp",
            "Qwen3RadixDecodeOp",
            "qwen3_qknorm_rope_store",
            "qwen3_radix_decode",
        ):
            self.assertIn(name, attention.__all__)
            self.assertTrue(hasattr(attention, name))

        expected = {
            KernelBackend.TORCH,
            KernelBackend.TORCH_COMPILE,
            KernelBackend.METAL_JIT,
        }
        for op in (
            "attention.qwen3_qknorm_rope_store",
            "attention.qwen3_radix_decode",
        ):
            self.assertEqual({spec.backend for spec in registry.get(op)}, expected)

    def test_per_op_warmups_share_one_compiled_library(self):
        from sglang.kernels.ops.attention import _qwen3_metal_jit as jit

        library = _TrackingLibrary()
        compile_library = mock.Mock(return_value=library)
        metal.clear_metal_library_caches()
        try:
            with mock.patch.object(
                metal, "_torch_mps_function", return_value=compile_library
            ):
                jit.warmup_qwen3_metal_qknorm_rope_store()
                jit.warmup_qwen3_metal_radix_decode()

            compile_library.assert_called_once()
            self.assertEqual(
                library.lookups,
                [
                    "qwen3_qknorm_rope_store_bf16",
                    "qwen3_radix_decode_bf16",
                ],
            )
        finally:
            metal.clear_metal_library_caches()


@unittest.skipUnless(_HAS_SUPPORTED_RUNTIME, "requires Torch 2.13 MPS")
class TestQwen3TorchMetalKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = QWEN3_06B_METAL_SPEC
        head_dim = cls.spec.head_dim
        inverse_frequency = 1.0 / (
            1_000_000.0
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(2048, dtype=torch.float32)
        frequency = torch.einsum("i,j->ij", positions, inverse_frequency)
        cls.cos_sin_cache = torch.cat((frequency.cos(), frequency.sin()), dim=-1).to(
            device="mps", dtype=torch.bfloat16
        )

    def test_fused_qknorm_rope_store_matches_staged_torch(self):
        spec = self.spec
        torch.manual_seed(41)
        num_tokens = 19
        qkv = torch.randn(
            num_tokens,
            spec.qkv_width,
            device="mps",
            dtype=torch.bfloat16,
        )
        q_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
        k_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
        positions = torch.arange(11, 11 + num_tokens, device="mps", dtype=torch.int64)
        slots = torch.randperm(num_tokens, device="mps").to(torch.int64) + 2
        pool_shape = (num_tokens + 4, spec.num_kv_heads, spec.head_dim)
        k_pool = torch.full(pool_shape, -7, device="mps", dtype=torch.bfloat16)
        v_pool = torch.full_like(k_pool, -7)
        q_out = torch.empty(
            num_tokens,
            spec.num_q_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        reference_k_pool = k_pool.clone()
        reference_v_pool = v_pool.clone()
        k_pool_pointer = k_pool.data_ptr()
        v_pool_pointer = v_pool.data_ptr()

        with mock.patch.object(
            torch.mps, "synchronize", wraps=torch.mps.synchronize
        ) as synchronize:
            qwen3_qknorm_rope_store(
                qkv,
                q_weight,
                k_weight,
                self.cos_sin_cache,
                positions,
                slots,
                q_out,
                k_pool,
                v_pool,
                epsilon=1e-6,
                backend=KernelBackend.METAL_JIT,
            )
        self.assertEqual(synchronize.call_count, 0)
        self.assertEqual(k_pool.data_ptr(), k_pool_pointer)
        self.assertEqual(v_pool.data_ptr(), v_pool_pointer)

        q, k, v = qkv.split(
            [
                spec.num_q_heads * spec.head_dim,
                spec.num_kv_heads * spec.head_dim,
                spec.num_kv_heads * spec.head_dim,
            ],
            dim=-1,
        )
        q = q.reshape(num_tokens, spec.num_q_heads, spec.head_dim)
        k = k.reshape(num_tokens, spec.num_kv_heads, spec.head_dim)
        v = v.reshape(num_tokens, spec.num_kv_heads, spec.head_dim)

        def rms_norm(value, weight):
            normalized = value.float() * torch.rsqrt(
                value.float().square().mean(dim=-1, keepdim=True) + 1e-6
            )
            return (normalized * weight.float()).to(torch.bfloat16)

        q = rms_norm(q, q_weight)
        k = rms_norm(k, k_weight)
        cosine, sine = self.cos_sin_cache[positions.long()].chunk(2, dim=-1)
        cosine = cosine[:, None, :]
        sine = sine[:, None, :]

        def rope(value):
            first, second = value.chunk(2, dim=-1)
            return torch.cat(
                (first * cosine - second * sine, second * cosine + first * sine),
                dim=-1,
            )

        reference_q = rope(q)
        reference_k_pool[slots] = rope(k)
        reference_v_pool[slots] = v
        torch.mps.synchronize()

        torch.testing.assert_close(
            q_out.cpu(), reference_q.cpu(), atol=0.0625, rtol=0.02
        )
        torch.testing.assert_close(
            k_pool.cpu(), reference_k_pool.cpu(), atol=0.0625, rtol=0.02
        )
        torch.testing.assert_close(v_pool.cpu(), reference_v_pool.cpu())

    def test_radix_decode_reads_random_physical_slots(self):
        spec = self.spec
        torch.manual_seed(73)
        batch_size = 2
        sequence_length = 513
        table_stride = 640
        pool_size = 2 * sequence_length + 32
        q = torch.randn(
            batch_size,
            spec.num_q_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        k_pool = torch.randn(
            pool_size,
            spec.num_kv_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        v_pool = torch.randn_like(k_pool)
        req_to_token = torch.zeros(
            batch_size + 1,
            table_stride,
            device="mps",
            dtype=torch.int32,
        )
        permutation = torch.randperm(pool_size, device="mps", dtype=torch.int64)
        req_to_token[1, :sequence_length] = permutation[:sequence_length].to(
            torch.int32
        )
        req_to_token[2, :sequence_length] = permutation[
            sequence_length : 2 * sequence_length
        ].to(torch.int32)
        req_pool_indices = torch.tensor([1, 2], device="mps", dtype=torch.int64)
        seq_lens = torch.full(
            (batch_size,), sequence_length, device="mps", dtype=torch.int64
        )
        out = torch.empty_like(q)

        with mock.patch.object(
            torch.mps, "synchronize", wraps=torch.mps.synchronize
        ) as synchronize:
            qwen3_radix_decode(
                q,
                k_pool,
                v_pool,
                req_to_token,
                req_pool_indices,
                seq_lens,
                out,
                scale=spec.attention_scale,
                backend=KernelBackend.METAL_JIT,
            )
        self.assertEqual(synchronize.call_count, 0)

        references = []
        for batch_index in range(batch_size):
            slots = req_to_token[req_pool_indices[batch_index], :sequence_length].long()
            key = k_pool[slots].movedim(0, 1)
            value = v_pool[slots].movedim(0, 1)
            references.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[batch_index][:, None, :].unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    enable_gqa=True,
                    scale=spec.attention_scale,
                    is_causal=False,
                )
                .squeeze(0)
                .squeeze(1)
            )
        reference = torch.stack(references)
        torch.mps.synchronize()
        torch.testing.assert_close(out.cpu(), reference.cpu(), atol=0.002, rtol=0.02)

    def test_layout_drift_fails_instead_of_copying(self):
        spec = self.spec
        q = torch.empty(
            1,
            spec.num_q_heads,
            spec.head_dim * 2,
            device="mps",
            dtype=torch.bfloat16,
        )[..., ::2]
        pool = torch.empty(
            4,
            spec.num_kv_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        req_to_token = torch.zeros(2, 4, device="mps", dtype=torch.int32)
        req_pool_indices = torch.ones(1, device="mps", dtype=torch.int64)
        seq_lens = torch.ones(1, device="mps", dtype=torch.int64)
        out = torch.empty_like(q).contiguous()
        with self.assertRaisesRegex(RuntimeError, "implicit Metal-path copies"):
            qwen3_radix_decode(
                q,
                pool,
                pool,
                req_to_token,
                req_pool_indices,
                seq_lens,
                out,
                scale=spec.attention_scale,
                backend=KernelBackend.METAL_JIT,
            )

    def test_qkv_shader_guards_invalid_position_and_slot(self):
        spec = self.spec
        qkv = torch.randn(2, spec.qkv_width, device="mps", dtype=torch.bfloat16)
        weight = torch.ones(spec.head_dim, device="mps", dtype=torch.bfloat16)
        positions = torch.tensor(
            [self.cos_sin_cache.shape[0], 0], device="mps", dtype=torch.int64
        )
        pool = torch.full(
            (4, spec.num_kv_heads, spec.head_dim),
            -3,
            device="mps",
            dtype=torch.bfloat16,
        )
        slots = torch.tensor([0, pool.shape[0]], device="mps", dtype=torch.int64)
        q_out = torch.full(
            (2, spec.num_q_heads, spec.head_dim),
            9,
            device="mps",
            dtype=torch.bfloat16,
        )

        qwen3_qknorm_rope_store(
            qkv,
            weight,
            weight,
            self.cos_sin_cache,
            positions,
            slots,
            q_out,
            pool,
            pool,
            epsilon=1e-6,
            backend=KernelBackend.METAL_JIT,
        )
        torch.mps.synchronize()
        self.assertEqual(torch.count_nonzero(q_out[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(q_out[1]).item(), 0)
        torch.testing.assert_close(pool.cpu(), torch.full_like(pool, -3).cpu())

    def test_decode_shader_guards_invalid_request_metadata(self):
        spec = self.spec
        q = torch.randn(
            1,
            spec.num_q_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        pool = torch.randn(
            4,
            spec.num_kv_heads,
            spec.head_dim,
            device="mps",
            dtype=torch.bfloat16,
        )
        req_to_token = torch.zeros(1, 2, device="mps", dtype=torch.int32)
        req_pool_indices = torch.ones(1, device="mps", dtype=torch.int64)
        seq_lens = torch.full((1,), 3, device="mps", dtype=torch.int64)
        out = torch.full_like(q, 7)

        qwen3_radix_decode(
            q,
            pool,
            pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            out,
            scale=spec.attention_scale,
            backend=KernelBackend.METAL_JIT,
        )
        torch.mps.synchronize()
        self.assertEqual(torch.count_nonzero(out).item(), 0)


if __name__ == "__main__":
    unittest.main()
