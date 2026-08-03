"""Correctness and fallback tests for the MPS fused SiLU-and-mul kernel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.fused_op import (
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_trace,
)
from sglang.kernels.ops.activation import _SILU_AND_MUL, silu_and_mul
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.activation import SiluAndMul
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_MPS_JIT = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)


def _reference(x: torch.Tensor) -> torch.Tensor:
    intermediate = x.shape[-1] // 2
    return F.silu(x[..., :intermediate]) * x[..., intermediate:]


def _make_layer() -> SiluAndMul:
    with mock.patch(
        "sglang.srt.layers.activation.get_exec",
        return_value=SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        ),
    ):
        return SiluAndMul()


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
@pytest.mark.parametrize("rows", [8, 17])
def test_qwen3_decode_shape_matches_torch_without_host_sync(rows):
    torch.manual_seed(37 + rows)
    intermediate = 3072
    x = torch.randn(
        rows,
        2 * intermediate,
        device="mps",
        dtype=torch.bfloat16,
    )
    reference = _reference(x)
    layer = _make_layer()
    assert layer._forward_method is None

    with mock.patch.object(
        torch.mps, "synchronize", wraps=torch.mps.synchronize
    ) as synchronize:
        result = layer(x)

    assert layer._forward_method.__func__ is SiluAndMul.forward_mps
    assert synchronize.call_count == 0
    assert result.shape == (rows, intermediate)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "mps"
    assert result.is_contiguous()
    torch.mps.synchronize()
    torch.testing.assert_close(result.cpu(), reference.cpu(), atol=0.03125, rtol=0.02)


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
@pytest.mark.parametrize("rows", [1, 7])
def test_small_decode_shapes_stay_on_faster_torch_path(rows):
    x = torch.randn(rows, 2 * 3072, device="mps", dtype=torch.bfloat16)
    clear_fused_op_trace()
    enable_fused_op_trace()
    try:
        result = _make_layer().forward_mps(x)
    finally:
        disable_fused_op_trace()
    records = get_fused_op_trace()

    assert records[-1].op == "activation.silu_and_mul"
    assert records[-1].backend == KernelBackend.TORCH.value
    clear_fused_op_trace()
    torch.mps.synchronize()
    torch.testing.assert_close(result.cpu(), _reference(x).cpu())


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_unified_op_honors_preallocated_output():
    torch.manual_seed(83)
    x = torch.randn(3, 2 * 3072, device="mps", dtype=torch.bfloat16)
    out = torch.empty(3, 3072, device="mps", dtype=torch.bfloat16)
    pointer = out.data_ptr()

    returned = silu_and_mul(x, out)
    torch.mps.synchronize()

    assert returned is out
    assert out.data_ptr() == pointer
    torch.testing.assert_close(out.cpu(), _reference(x).cpu(), atol=0.03125, rtol=0.02)


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
@pytest.mark.parametrize("case", ["float32", "rank3", "noncontiguous"])
def test_non_contract_input_uses_torch_reference(case):
    if case == "float32":
        x = torch.randn(2, 32, device="mps", dtype=torch.float32)
    elif case == "rank3":
        x = torch.randn(2, 3, 32, device="mps", dtype=torch.bfloat16)
    elif case == "noncontiguous":
        x = torch.randn(2, 64, device="mps", dtype=torch.bfloat16)[:, ::2]
        assert not x.is_contiguous()
    clear_fused_op_trace()
    enable_fused_op_trace()
    try:
        result = _make_layer().forward_mps(x)
    finally:
        disable_fused_op_trace()
    records = get_fused_op_trace()

    assert records[-1].op == "activation.silu_and_mul"
    assert records[-1].backend == KernelBackend.TORCH.value
    clear_fused_op_trace()
    torch.mps.synchronize()
    torch.testing.assert_close(result.cpu(), _reference(x).cpu())


def test_namespace_registers_mps_backend_without_changing_existing_priority():
    assert KernelBackend.METAL_JIT in _SILU_AND_MUL.available_backends()
    assert _SILU_AND_MUL.priority == (
        KernelBackend.JIT,
        KernelBackend.AOT,
        KernelBackend.AITER,
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
