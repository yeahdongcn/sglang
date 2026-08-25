"""CPU-safe tests for independent Qwen3 Metal attention gates."""

import os
from unittest import mock

import pytest

from sglang.kernels.ops.attention.qwen3_mps import (
    Qwen3QKNormRopeStoreOp,
    Qwen3RadixDecodeOp,
    is_qwen3_metal_aot_available,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    Qwen3MetalAttentionSelection,
    choose_kernel_backend,
    choose_model_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _selection(**overrides: str) -> Qwen3MetalAttentionSelection:
    values = {
        "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE": "torch",
        "SGLANG_MPS_QWEN3_RADIX_DECODE": "torch",
    }
    values.update(overrides)
    with mock.patch.dict(os.environ, values):
        return Qwen3MetalAttentionSelection.from_env()


def test_defaults_keep_both_semantic_ops_on_torch(monkeypatch):
    monkeypatch.delenv("SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE", raising=False)
    monkeypatch.delenv("SGLANG_MPS_QWEN3_RADIX_DECODE", raising=False)

    selection = Qwen3MetalAttentionSelection.from_env()

    assert selection.qknorm_rope_store == (KernelBackend.TORCH,)
    assert selection.radix_decode == (KernelBackend.TORCH,)
    assert selection.model_forward == ("torch",)
    assert selection.deferred_kv_commit == (KernelBackend.TORCH,)


def test_aot_is_not_advertised_without_a_torch_data_only_artifact():
    assert not is_qwen3_metal_aot_available()
    assert KernelBackend.METAL_AOT not in Qwen3QKNormRopeStoreOp().available_backends()
    assert KernelBackend.METAL_AOT not in Qwen3RadixDecodeOp().available_backends()


def test_gates_are_independent_and_ordered():
    selection = _selection(
        SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE="metal_aot,metal_jit,torch",
        SGLANG_MPS_QWEN3_RADIX_DECODE="metal_jit,torch",
    )

    assert selection.qknorm_rope_store == (
        KernelBackend.METAL_AOT,
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )
    assert selection.radix_decode == (
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        (
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
            "triton,torch",
            "unsupported provider",
        ),
        (
            "SGLANG_MPS_QWEN3_RADIX_DECODE",
            "metal_jit,metal_jit,torch",
            "duplicate providers",
        ),
        (
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
            "metal_jit",
            "must end with 'torch'",
        ),
    ],
)
def test_invalid_priorities_fail_at_startup(field, value, error):
    with pytest.raises(RuntimeError, match=error):
        _selection(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
            "mlx",
            "must end with 'torch'",
        ),
        (
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
            "cuda,torch",
            "unsupported provider",
        ),
        (
            "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
            "metal_aot,torch",
            "unsupported provider",
        ),
    ],
)
def test_model_and_deferred_priorities_fail_closed(field, value, error):
    with pytest.raises(RuntimeError, match=error):
        _selection(**{field: value})


def test_availability_falls_through_in_declared_order():
    priority = (
        KernelBackend.METAL_AOT,
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )
    assert (
        choose_kernel_backend(
            priority,
            aot_available=False,
            jit_available=True,
        )
        is KernelBackend.METAL_JIT
    )
    assert (
        choose_kernel_backend(
            priority,
            aot_available=False,
            jit_available=False,
        )
        is KernelBackend.TORCH
    )


def test_model_priority_falls_back_to_torch_when_mlx_is_unavailable():
    assert choose_model_backend(("mlx", "torch"), mlx_available=False) == "torch"
    assert choose_model_backend(("mlx", "torch"), mlx_available=True) == "mlx"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
