"""Numerical contract for the gated Qwen3 MLX RMSNorm candidate."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.utils.tensor_bridge import mlx_to_torch
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")

mx = pytest.importorskip("mlx.core")

from sglang.kernels.ops.layernorm._qwen3_rmsnorm_mlx import (  # noqa: E402
    add_rms_norm,
    rms_norm,
    warmup,
)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX RMSNorm requires Metal"
)
@pytest.mark.parametrize("rows", [1, 8, 512])
def test_qwen3_mlx_rmsnorm_candidate_matches_fp32_reference(rows):
    mx.random.seed(23 + rows)
    value = mx.random.normal((rows, 1024)).astype(mx.bfloat16)
    residual = mx.random.normal((rows, 1024)).astype(mx.bfloat16)
    weight = mx.random.normal((1024,)).astype(mx.bfloat16)
    epsilon = 1e-6

    warmup(epsilon)
    actual = rms_norm(value, weight, epsilon)
    actual_add, actual_residual = add_rms_norm(value, residual, weight, epsilon)
    reference = mx.fast.rms_norm(
        value.astype(mx.float32), weight.astype(mx.float32), epsilon
    ).astype(mx.bfloat16)
    summed = value.astype(mx.float32) + residual.astype(mx.float32)
    reference_add = mx.fast.rms_norm(summed, weight.astype(mx.float32), epsilon).astype(
        mx.bfloat16
    )
    reference_residual = summed.astype(mx.bfloat16)
    mx.eval(
        actual,
        actual_add,
        actual_residual,
        reference,
        reference_add,
        reference_residual,
    )

    # Match MLX 0.32's fp32 RMS reduction order exactly while fusing the casts
    # (and residual add for the second form) into one Metal launch.
    assert bool(mx.all(actual == reference).item())
    assert bool(mx.all(actual_add == reference_add).item())
    assert bool(mx.all(actual_residual == reference_residual).item())

    # Torch/MPS uses a different fp32 reduction tree for the 512-row case.
    # Check the direct framework contract within the observed bf16 rounding
    # boundary without misrepresenting it as bitwise parity.
    torch_value = mlx_to_torch(value, device="mps").to(torch.float32)
    torch_residual = mlx_to_torch(residual, device="mps").to(torch.float32)
    torch_weight = mlx_to_torch(weight, device="mps").to(torch.float32)
    torch_actual = mlx_to_torch(actual, device="mps")
    torch_actual_add = mlx_to_torch(actual_add, device="mps")
    torch_actual_residual = mlx_to_torch(actual_residual, device="mps")

    torch_variance = torch_value.pow(2).mean(dim=-1, keepdim=True)
    torch_reference = (
        torch_value * torch.rsqrt(torch_variance + epsilon) * torch_weight
    ).to(torch.bfloat16)
    torch_summed = torch_value + torch_residual
    torch_add_variance = torch_summed.pow(2).mean(dim=-1, keepdim=True)
    torch_reference_add = (
        torch_summed * torch.rsqrt(torch_add_variance + epsilon) * torch_weight
    ).to(torch.bfloat16)

    torch.testing.assert_close(torch_actual, torch_reference, rtol=0, atol=0.002)
    torch.testing.assert_close(
        torch_actual_add, torch_reference_add, rtol=0, atol=0.002
    )
    assert torch.equal(torch_actual_residual, torch_summed.to(torch.bfloat16))


def test_qwen3_mlx_rmsnorm_rejects_wrong_hidden_size():
    value = mx.zeros((1, 16), dtype=mx.bfloat16)
    weight = mx.ones((16,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="hidden size 1024"):
        rms_norm(value, weight, 1e-6)


def test_qwen3_mlx_rmsnorm_rejects_scalar_input_cleanly():
    value = mx.zeros((), dtype=mx.bfloat16)
    weight = mx.ones((1024,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="at least one dimension"):
        rms_norm(value, weight, 1e-6)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX RMSNorm requires Metal"
)
def test_qwen3_mlx_rmsnorm_allows_read_only_input_aliases():
    value = mx.random.normal((2, 1024)).astype(mx.bfloat16)
    weight = mx.ones((1024,), dtype=mx.bfloat16)
    actual, residual_output = add_rms_norm(value, value, weight, 1e-6)
    summed = value.astype(mx.float32) + value.astype(mx.float32)
    expected = mx.fast.rms_norm(summed, weight.astype(mx.float32), 1e-6).astype(
        mx.bfloat16
    )
    mx.eval(actual, residual_output, expected, summed)
    assert bool(mx.all(actual == expected).item())
    assert bool(mx.all(residual_output == summed.astype(mx.bfloat16)).item())


def test_qwen3_mlx_rmsnorm_empty_add_keeps_norm_and_residual_contract():
    value = mx.zeros((0, 1024), dtype=mx.bfloat16)
    residual = mx.zeros_like(value)
    weight = mx.ones((1024,), dtype=mx.bfloat16)

    normalized, residual_output = add_rms_norm(value, residual, weight, 1e-6)
    mx.eval(normalized, residual_output)

    assert normalized.shape == (0, 1024)
    assert residual_output.shape == (0, 1024)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX RMSNorm requires Metal"
)
def test_qwen3_mlx_rmsnorm_materializes_strided_rows_exactly():
    mx.random.seed(47)
    value = mx.random.normal((3, 2048)).astype(mx.bfloat16)[:, ::2]
    residual = mx.random.normal((3, 2048)).astype(mx.bfloat16)[:, ::2]
    weight = mx.random.normal((1024,)).astype(mx.bfloat16)
    summed = value.astype(mx.float32) + residual.astype(mx.float32)
    expected = mx.fast.rms_norm(
        value.astype(mx.float32), weight.astype(mx.float32), 1e-6
    ).astype(mx.bfloat16)
    expected_add = mx.fast.rms_norm(summed, weight.astype(mx.float32), 1e-6).astype(
        mx.bfloat16
    )
    expected_residual = summed.astype(mx.bfloat16)

    actual = rms_norm(value, weight, 1e-6)
    actual_add, actual_residual = add_rms_norm(value, residual, weight, 1e-6)
    mx.eval(
        expected,
        expected_add,
        expected_residual,
        actual,
        actual_add,
        actual_residual,
    )

    assert bool(mx.all(actual == expected).item())
    assert bool(mx.all(actual_add == expected_add).item())
    assert bool(mx.all(actual_residual == expected_residual).item())


@pytest.mark.parametrize("epsilon", [0.0, -1e-6, float("nan"), float("inf")])
def test_qwen3_mlx_rmsnorm_rejects_invalid_epsilon(epsilon):
    value = mx.zeros((1, 1024), dtype=mx.bfloat16)
    weight = mx.ones((1024,), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="positive epsilon"):
        rms_norm(value, weight, epsilon)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX RMSNorm requires Metal"
)
def test_qwen3_mlx_rmsnorm_accepts_positive_exponent_kernel_name():
    value = mx.zeros((1, 1024), dtype=mx.bfloat16)
    weight = mx.ones((1024,), dtype=mx.bfloat16)
    output = rms_norm(value, weight, 1.0)
    mx.eval(output)
    assert output.shape == value.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
