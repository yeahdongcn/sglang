"""Model-free tests for generic MPS Metal kernel selection and warmup."""

import sys
from unittest import mock

import pytest

from sglang.kernels.fused_op import set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.generic_kernels import (
    MpsGenericKernelSelection,
    clear_mps_generic_kernel_configuration,
    configure_mps_generic_kernels,
)
from sglang.srt.platforms.mps import MpsSRTPlatform
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")

_ENV_NAMES = (
    "SGLANG_MPS_RMSNORM",
    "SGLANG_MPS_FUSED_ADD_RMSNORM",
    "SGLANG_MPS_SILU_AND_MUL",
)


@pytest.fixture(autouse=True)
def _reset_configuration(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    set_fused_op_backend(None)
    clear_mps_generic_kernel_configuration()
    yield
    set_fused_op_backend(None)
    clear_mps_generic_kernel_configuration()


def test_default_selection_keeps_every_op_on_torch():
    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM

    with mock.patch(
        "sglang.kernels.metal.compile_metal_library",
        side_effect=AssertionError("default Torch selection must not compile Metal"),
    ):
        selection = configure_mps_generic_kernels()

    assert selection.metal_jit_ops() == ()
    assert _RMSNORM.get_priority() == (KernelBackend.TORCH,)
    assert _FUSED_ADD_RMSNORM.get_priority() == (KernelBackend.TORCH,)
    assert _SILU_AND_MUL.get_priority() == (KernelBackend.TORCH,)


def test_each_gate_warms_and_publishes_independently(monkeypatch):
    monkeypatch.setenv("SGLANG_MPS_RMSNORM", "metal_jit,torch")
    monkeypatch.setenv("SGLANG_MPS_FUSED_ADD_RMSNORM", "torch")
    monkeypatch.setenv("SGLANG_MPS_SILU_AND_MUL", "metal_jit,torch")

    with (
        mock.patch("sglang.kernels.metal.is_metal_jit_available", return_value=True),
        mock.patch(
            "sglang.kernels.ops.layernorm._rmsnorm_metal_jit.warmup_mps_rmsnorm_kernels"
        ) as warm_rms,
        mock.patch(
            "sglang.kernels.ops.activation._silu_and_mul_metal_jit."
            "warmup_silu_and_mul_metal_kernel"
        ) as warm_silu,
    ):
        selection = configure_mps_generic_kernels()

    assert selection.metal_jit_ops() == ("rmsnorm", "silu_and_mul")
    warm_rms.assert_called_once_with(rmsnorm=True, fused_add_rmsnorm=False)
    warm_silu.assert_called_once_with()

    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM

    assert _RMSNORM.get_priority() == (
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )
    assert _FUSED_ADD_RMSNORM.get_priority() == (KernelBackend.TORCH,)
    assert _SILU_AND_MUL.get_priority() == (
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )


def test_warmup_failure_aborts_before_priority_publication(monkeypatch):
    monkeypatch.setenv("SGLANG_MPS_RMSNORM", "metal_jit,torch")
    from sglang.kernels.ops.layernorm import _RMSNORM

    original_priority = _RMSNORM.get_priority()
    with (
        mock.patch("sglang.kernels.metal.is_metal_jit_available", return_value=True),
        mock.patch(
            "sglang.kernels.ops.layernorm._rmsnorm_metal_jit."
            "warmup_mps_rmsnorm_kernels",
            side_effect=RuntimeError("compiler failed"),
        ),
        pytest.raises(RuntimeError, match="compiler failed"),
    ):
        configure_mps_generic_kernels()

    assert _RMSNORM.get_priority() == original_priority


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("SGLANG_MPS_RMSNORM", "metal_jit", "must end with 'torch'"),
        ("SGLANG_MPS_FUSED_ADD_RMSNORM", "triton,torch", "unsupported"),
        ("SGLANG_MPS_SILU_AND_MUL", "torch,torch", "duplicate"),
    ],
)
def test_selection_rejects_invalid_priorities(monkeypatch, name, value, error):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=error):
        MpsGenericKernelSelection.from_env()


def test_platform_init_validates_runtime_before_generic_warmup():
    order = []
    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.runtime.validate_mps_runtime",
            side_effect=lambda: order.append("runtime"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.generic_kernels."
            "configure_mps_generic_kernels",
            side_effect=lambda: order.append("kernels"),
        ),
    ):
        MpsSRTPlatform().init_backend()
    assert order == ["runtime", "kernels"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
