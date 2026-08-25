"""Correctness and ownership tests for MPS Torch-stream Metal kernels."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from packaging.version import Version

from sglang.kernels.ops.attention.qwen3_mps import (
    QWEN3_06B_METAL_SPEC,
    is_qwen3_metal_aot_available,
    qwen3_qknorm_rope_store,
    qwen3_radix_decode,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.qwen3 import (
    validate_qwen3_attention_module,
)
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_SUPPORTED_RUNTIME = (
    importlib.util.find_spec("mlx") is not None
    and torch.backends.mps.is_available()
    and Version(torch.__version__) >= Version("2.13.0")
)
_HAS_METAL_AOT = _HAS_SUPPORTED_RUNTIME and is_qwen3_metal_aot_available()


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

    def test_static_contract_rejects_mismatched_qk_rms_epsilons(self):
        spec = self.spec
        weight = torch.ones(spec.head_dim, device="mps", dtype=torch.bfloat16)
        module = SimpleNamespace(
            num_heads=spec.num_q_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            q_norm=SimpleNamespace(weight=weight, variance_epsilon=1e-6),
            k_norm=SimpleNamespace(weight=weight.clone(), variance_epsilon=1e-5),
            rotary_emb=SimpleNamespace(
                cos_sin_cache=self.cos_sin_cache,
                is_neox_style=True,
                rotary_dim=spec.head_dim,
            ),
            attn=SimpleNamespace(scaling=spec.attention_scale),
        )

        with self.assertRaisesRegex(RuntimeError, "matching finite positive"):
            validate_qwen3_attention_module(module)

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

    @unittest.skipUnless(_HAS_METAL_AOT, "requires packaged Qwen3 Metal AOT")
    def test_qknorm_rope_store_aot_and_jit_are_bitwise_identical(self):
        spec = self.spec
        torch.manual_seed(91)
        num_tokens = 5
        qkv = torch.randn(
            num_tokens,
            spec.qkv_width,
            device="mps",
            dtype=torch.bfloat16,
        )
        q_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
        k_weight = torch.randn(spec.head_dim, device="mps", dtype=torch.bfloat16)
        positions = torch.arange(23, 23 + num_tokens, device="mps", dtype=torch.int64)
        slots = torch.tensor([7, 2, 9, 1, 5], device="mps", dtype=torch.int64)
        pool_shape = (12, spec.num_kv_heads, spec.head_dim)
        base_k_pool = torch.full(pool_shape, -4, device="mps", dtype=torch.bfloat16)
        base_v_pool = torch.full_like(base_k_pool, -4)
        results = {}

        for backend in (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT):
            q_out = torch.empty(
                num_tokens,
                spec.num_q_heads,
                spec.head_dim,
                device="mps",
                dtype=torch.bfloat16,
            )
            k_pool = base_k_pool.clone()
            v_pool = base_v_pool.clone()
            k_pointer = k_pool.data_ptr()
            v_pointer = v_pool.data_ptr()
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
                    backend=backend,
                )
            self.assertEqual(synchronize.call_count, 0)
            self.assertEqual(k_pool.data_ptr(), k_pointer)
            self.assertEqual(v_pool.data_ptr(), v_pointer)
            results[backend] = (q_out, k_pool, v_pool)

        torch.mps.synchronize()
        for aot, jit in zip(
            results[KernelBackend.METAL_AOT],
            results[KernelBackend.METAL_JIT],
        ):
            torch.testing.assert_close(aot.cpu(), jit.cpu(), atol=0, rtol=0)

    @unittest.skipUnless(_HAS_METAL_AOT, "requires packaged Qwen3 Metal AOT")
    def test_radix_decode_aot_and_jit_are_bitwise_identical(self):
        spec = self.spec
        torch.manual_seed(92)
        sequence_length = 67
        pool_size = 96
        q = torch.randn(
            1,
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
        req_to_token = torch.zeros(2, 80, device="mps", dtype=torch.int32)
        req_to_token[1, :sequence_length] = torch.randperm(
            pool_size, device="mps", dtype=torch.int64
        )[:sequence_length].to(torch.int32)
        req_pool_indices = torch.ones(1, device="mps", dtype=torch.int64)
        seq_lens = torch.full((1,), sequence_length, device="mps", dtype=torch.int64)
        outputs = {}

        for backend in (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT):
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
                    backend=backend,
                )
            self.assertEqual(synchronize.call_count, 0)
            outputs[backend] = out

        torch.mps.synchronize()
        torch.testing.assert_close(
            outputs[KernelBackend.METAL_AOT].cpu(),
            outputs[KernelBackend.METAL_JIT].cpu(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
