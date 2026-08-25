"""Correctness and dispatch tests for Torch-stream Metal RMSNorm kernels."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM
from sglang.kernels.ops.layernorm._rmsnorm_metal_jit import (
    mps_fused_add_rmsnorm,
    mps_rmsnorm,
    warmup_mps_rmsnorm_kernels,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_SUPPORTED_RUNTIME = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)


def _reference_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    value_fp32 = value.float()
    inverse_rms = torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
    return (value_fp32 * inverse_rms * weight.float()).to(torch.bfloat16)


@unittest.skipUnless(_HAS_SUPPORTED_RUNTIME, "requires Torch 2.13 MPS")
class TestMpsRMSNormMetalJit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warmup_mps_rmsnorm_kernels()

    def _inputs(self, rows: int):
        torch.manual_seed(1900 + rows)
        x = torch.randn(rows, 1024).to(torch.bfloat16)
        residual = torch.randn(rows, 1024).to(torch.bfloat16)
        weight = torch.randn(1024).to(torch.bfloat16)
        return x, residual, weight

    def test_plain_and_fused_match_fp32_accumulation(self):
        epsilon = 1e-6
        for rows in (1, 4, 17):
            with self.subTest(rows=rows):
                x_cpu, residual_cpu, weight_cpu = self._inputs(rows)
                x = x_cpu.to("mps")
                residual = residual_cpu.to("mps")
                weight = weight_cpu.to("mps")

                with mock.patch.object(
                    torch.mps, "synchronize", wraps=torch.mps.synchronize
                ) as synchronize:
                    output = mps_rmsnorm(x, weight, epsilon)
                self.assertEqual(synchronize.call_count, 0)
                self.assertNotEqual(output.data_ptr(), x.data_ptr())

                fused_input = x.clone()
                fused_residual = residual.clone()
                input_pointer = fused_input.data_ptr()
                residual_pointer = fused_residual.data_ptr()
                with mock.patch.object(
                    torch.mps, "synchronize", wraps=torch.mps.synchronize
                ) as synchronize:
                    mps_fused_add_rmsnorm(fused_input, fused_residual, weight, epsilon)
                self.assertEqual(synchronize.call_count, 0)

                torch.mps.synchronize()
                expected_output = _reference_rmsnorm(x_cpu, weight_cpu, epsilon)
                residual_sum = x_cpu.float() + residual_cpu.float()
                expected_fused_output = _reference_rmsnorm(
                    residual_sum, weight_cpu, epsilon
                )
                expected_residual = residual_sum.to(torch.bfloat16)

                torch.testing.assert_close(
                    output.cpu(), expected_output, atol=0.03125, rtol=0.01
                )
                torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)
                torch.testing.assert_close(
                    fused_input.cpu(),
                    expected_fused_output,
                    atol=0.03125,
                    rtol=0.01,
                )
                torch.testing.assert_close(
                    fused_residual.cpu(), expected_residual, atol=0, rtol=0
                )
                self.assertEqual(fused_input.data_ptr(), input_pointer)
                self.assertEqual(fused_residual.data_ptr(), residual_pointer)

    def test_unified_selector_uses_metal_jit_for_the_narrow_contract(self):
        x_cpu, residual_cpu, weight_cpu = self._inputs(1)
        x = x_cpu.to("mps")
        residual = residual_cpu.to("mps")
        weight = weight_cpu.to("mps")
        with torch.inference_mode():
            self.assertIs(
                _RMSNORM._resolve_backend(x, weight, 1e-6),
                KernelBackend.METAL_JIT,
            )
            self.assertIs(
                _FUSED_ADD_RMSNORM._resolve_backend(x, residual, weight, 1e-6),
                KernelBackend.METAL_JIT,
            )

            strided = torch.empty(1, 2048, device="mps", dtype=torch.bfloat16)[:, ::2]
            self.assertIs(
                _RMSNORM._resolve_backend(strided, weight, 1e-6),
                KernelBackend.TORCH,
            )

    def test_torch_provider_preserves_reference_semantics_and_out(self):
        epsilon = 1e-6
        x_cpu, _, weight_cpu = self._inputs(8)
        x = x_cpu.to("mps")
        weight = weight_cpu.to("mps")
        expected = _reference_rmsnorm(x_cpu, weight_cpu, epsilon)

        with (
            mock.patch.object(F, "rms_norm", wraps=F.rms_norm) as fused,
            mock.patch.object(
                torch.mps, "synchronize", wraps=torch.mps.synchronize
            ) as synchronize,
        ):
            allocated = _RMSNORM.forward_native(x, weight, epsilon)
            contiguous_out = torch.empty_like(x)
            contiguous_pointer = contiguous_out.data_ptr()
            contiguous = _RMSNORM.forward_native(x, weight, epsilon, out=contiguous_out)
            backing = torch.empty(8, 2048, device="mps", dtype=torch.bfloat16)
            strided_out = backing[:, ::2]
            strided_pointer = strided_out.data_ptr()
            strided = _RMSNORM.forward_native(x, weight, epsilon, out=strided_out)
            alias = x.clone()
            alias_pointer = alias.data_ptr()
            alias_result = _RMSNORM.forward_native(alias, weight, epsilon, out=alias)

        # The fused Torch 2.13 MPS path has a different rounding contract and
        # changes long-context Qwen3 greedy output. Keep this semantic provider
        # on the explicit FP32 reference even though F.rms_norm is available.
        self.assertEqual(fused.call_count, 0)
        self.assertEqual(synchronize.call_count, 0)
        self.assertNotEqual(allocated.data_ptr(), x.data_ptr())
        self.assertIs(contiguous, contiguous_out)
        self.assertEqual(contiguous.data_ptr(), contiguous_pointer)
        self.assertNotEqual(contiguous.data_ptr(), x.data_ptr())
        self.assertIs(strided, strided_out)
        self.assertEqual(strided.data_ptr(), strided_pointer)
        self.assertFalse(strided.is_contiguous())
        self.assertIs(alias_result, alias)
        self.assertEqual(alias_result.data_ptr(), alias_pointer)
        torch.mps.synchronize()
        for result in (allocated, contiguous, strided, alias_result):
            torch.testing.assert_close(result.cpu(), expected, atol=0.03125, rtol=0.01)
        torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)

    def test_rmsnorm_layer_returns_both_outputs_and_falls_back_cleanly(self):
        epsilon = 1e-6
        x_cpu, residual_cpu, weight_cpu = self._inputs(1)
        layer = RMSNorm(1024, eps=epsilon).to(device="mps", dtype=torch.bfloat16)
        layer.weight.data.copy_(weight_cpu.to("mps"))
        x = x_cpu.to("mps")
        residual = residual_cpu.to("mps")

        with (
            torch.inference_mode(),
            mock.patch.object(
                _RMSNORM,
                "forward_metal_jit",
                wraps=_RMSNORM.forward_metal_jit,
            ) as plain_kernel,
            mock.patch.object(
                _FUSED_ADD_RMSNORM,
                "forward_metal_jit",
                wraps=_FUSED_ADD_RMSNORM.forward_metal_jit,
            ) as fused_kernel,
        ):
            output = layer.forward_mps(x)
            fused_output, residual_output = layer.forward_mps(
                x.clone(), residual.clone()
            )
        self.assertEqual(plain_kernel.call_count, 1)
        self.assertEqual(fused_kernel.call_count, 1)

        torch.mps.synchronize()
        expected_output = _reference_rmsnorm(x_cpu, weight_cpu, epsilon)
        residual_sum = x_cpu.float() + residual_cpu.float()
        expected_fused_output = _reference_rmsnorm(residual_sum, weight_cpu, epsilon)
        torch.testing.assert_close(
            output.cpu(), expected_output, atol=0.03125, rtol=0.01
        )
        torch.testing.assert_close(
            fused_output.cpu(), expected_fused_output, atol=0.03125, rtol=0.01
        )
        torch.testing.assert_close(
            residual_output.cpu(), residual_sum.to(torch.bfloat16), atol=0, rtol=0
        )

        strided = torch.empty(1, 2048, device="mps", dtype=torch.bfloat16)[:, ::2]
        with (
            torch.inference_mode(),
            mock.patch.object(_RMSNORM, "forward_metal_jit") as metal_kernel,
            mock.patch.object(
                layer, "forward_native", wraps=layer.forward_native
            ) as native_kernel,
        ):
            fallback = layer.forward_mps(strided)
        metal_kernel.assert_not_called()
        native_kernel.assert_called_once()
        self.assertEqual(tuple(fallback.shape), (1, 1024))
        self.assertTrue(torch.isfinite(fallback).all().item())


if __name__ == "__main__":
    unittest.main()
