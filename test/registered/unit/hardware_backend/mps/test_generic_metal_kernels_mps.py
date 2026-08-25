"""Small real-device parity checks for generic Torch-stream Metal kernels."""

import sys
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.fused_op import set_fused_op_backend
from sglang.kernels.ops.activation import _SILU_AND_MUL
from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM
from sglang.srt.hardware_backend.mps.generic_kernels import (
    clear_mps_generic_kernel_configuration,
    configure_mps_generic_kernels,
)
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_MPS_JIT = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)
pytestmark = pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch MPS Metal JIT")


@pytest.fixture(autouse=True)
def _enable_all_generic_metal_ops(monkeypatch):
    set_fused_op_backend(None)
    clear_mps_generic_kernel_configuration()
    for name in (
        "SGLANG_MPS_RMSNORM",
        "SGLANG_MPS_FUSED_ADD_RMSNORM",
        "SGLANG_MPS_SILU_AND_MUL",
    ):
        monkeypatch.setenv(name, "metal_jit,torch")
    configure_mps_generic_kernels()
    yield
    clear_mps_generic_kernel_configuration()
    set_fused_op_backend(None)


def _reference_rmsnorm(
    value: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    value_fp32 = value.float()
    inverse_rms = torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
    return (value_fp32 * inverse_rms * weight.float()).to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 8, 17])
def test_rmsnorm_and_fused_add_match_fp32_reference_without_hidden_sync(rows):
    torch.manual_seed(1900 + rows)
    epsilon = 1e-6
    x_cpu = torch.randn(rows, 1024).to(torch.bfloat16)
    residual_cpu = torch.randn(rows, 1024).to(torch.bfloat16)
    weight_cpu = torch.randn(1024).to(torch.bfloat16)
    x = x_cpu.to("mps")
    residual = residual_cpu.to("mps")
    weight = weight_cpu.to("mps")

    with mock.patch.object(
        torch.mps, "synchronize", wraps=torch.mps.synchronize
    ) as synchronize:
        output = _RMSNORM(x, weight, epsilon)
        fused_input = x.clone()
        fused_residual = residual.clone()
        _FUSED_ADD_RMSNORM(fused_input, fused_residual, weight, epsilon)
    assert synchronize.call_count == 0

    torch.mps.synchronize()
    residual_sum = x_cpu.float() + residual_cpu.float()
    torch.testing.assert_close(
        output.cpu(),
        _reference_rmsnorm(x_cpu, weight_cpu, epsilon),
        atol=0.03125,
        rtol=0.01,
    )
    torch.testing.assert_close(
        fused_input.cpu(),
        _reference_rmsnorm(residual_sum, weight_cpu, epsilon),
        atol=0.03125,
        rtol=0.01,
    )
    torch.testing.assert_close(
        fused_residual.cpu(), residual_sum.to(torch.bfloat16), atol=0, rtol=0
    )


@pytest.mark.parametrize("rows", [8, 17])
def test_silu_and_mul_matches_torch_without_hidden_sync(rows):
    torch.manual_seed(37 + rows)
    x = torch.randn(rows, 6144, device="mps", dtype=torch.bfloat16)
    reference = F.silu(x[:, :3072]) * x[:, 3072:]

    with mock.patch.object(
        torch.mps, "synchronize", wraps=torch.mps.synchronize
    ) as synchronize:
        result = _SILU_AND_MUL(x)
    assert synchronize.call_count == 0

    torch.mps.synchronize()
    torch.testing.assert_close(result.cpu(), reference.cpu(), atol=0.03125, rtol=0.02)


def test_unsupported_shapes_fall_back_to_torch_instead_of_launching_metal():
    rms_input = torch.randn(2, 512, device="mps", dtype=torch.bfloat16)
    rms_weight = torch.randn(512, device="mps", dtype=torch.bfloat16)
    alias_input = torch.randn(2, 1024, device="mps", dtype=torch.bfloat16)
    alias_weight = alias_input[1]
    small_silu = torch.randn(1, 64, device="mps", dtype=torch.bfloat16)

    with (
        mock.patch.object(_RMSNORM, "forward_metal_jit") as rms_metal,
        mock.patch.object(_SILU_AND_MUL, "forward_metal_jit") as silu_metal,
    ):
        rms_result = _RMSNORM(rms_input, rms_weight)
        alias_result = _RMSNORM(alias_input, alias_weight)
        silu_result = _SILU_AND_MUL(small_silu)

    rms_metal.assert_not_called()
    silu_metal.assert_not_called()
    torch.mps.synchronize()
    assert rms_result.shape == rms_input.shape
    assert alias_result.shape == alias_input.shape
    torch.testing.assert_close(
        silu_result.cpu(),
        (F.silu(small_silu[:, :32]) * small_silu[:, 32:]).cpu(),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
