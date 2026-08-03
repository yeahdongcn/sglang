"""CPU-safe tests for startup-time MPS semantic-operator selection."""

import os
from unittest import mock

import pytest

from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    MpsGenericOperatorSelection,
    MpsOperatorSelection,
    choose_kernel_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


_SELECTION_ENVS = (
    envs.SGLANG_MPS_QWEN3_MODEL_FORWARD,
    envs.SGLANG_MPS_QWEN3_GREEDY_TAIL,
    envs.SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE,
    envs.SGLANG_MPS_QWEN3_RADIX_DECODE,
    envs.SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT,
    envs.SGLANG_MPS_RMSNORM,
    envs.SGLANG_MPS_FUSED_ADD_RMSNORM,
    envs.SGLANG_MPS_SILU_AND_MUL,
)


def _read_selection(**overrides: str) -> MpsOperatorSelection:
    """Read one isolated env configuration without touching MPS hardware."""
    values = {field.name: "torch" for field in _SELECTION_ENVS}
    values.update(overrides)
    with mock.patch.dict(os.environ, values):
        return MpsOperatorSelection.from_env()


def test_selection_defaults_every_semantic_op_to_torch(monkeypatch):
    # Exercise the EnvTuple declarations themselves rather than spelling the
    # default values through the helper below.
    for field in _SELECTION_ENVS:
        monkeypatch.delenv(field.name, raising=False)
    selection = MpsOperatorSelection.from_env()

    assert selection.model_forward == ("torch",)
    assert selection.greedy_tail == ("torch",)
    assert selection.qknorm_rope_store == (KernelBackend.TORCH,)
    assert selection.radix_decode == (KernelBackend.TORCH,)
    assert selection.deferred_kv_commit == (KernelBackend.TORCH,)
    assert selection.rmsnorm == (KernelBackend.TORCH,)
    assert selection.fused_add_rmsnorm == (KernelBackend.TORCH,)
    assert selection.silu_and_mul == (KernelBackend.TORCH,)


def test_generic_selection_does_not_parse_family_specific_priorities(monkeypatch):
    monkeypatch.setenv("SGLANG_MPS_QWEN3_MODEL_FORWARD", "not-a-provider")
    monkeypatch.setenv("SGLANG_MPS_QWEN3_GREEDY_TAIL", "also-invalid")
    for field in (
        envs.SGLANG_MPS_RMSNORM,
        envs.SGLANG_MPS_FUSED_ADD_RMSNORM,
        envs.SGLANG_MPS_SILU_AND_MUL,
    ):
        monkeypatch.setenv(field.name, "torch")

    selection = MpsGenericOperatorSelection.from_env()

    assert selection.as_state() == {
        "rmsnorm": ["torch"],
        "fused_add_rmsnorm": ["torch"],
        "silu_and_mul": ["torch"],
    }


def test_selection_keeps_per_op_priorities_independent():
    selection = _read_selection(
        SGLANG_MPS_QWEN3_GREEDY_TAIL="mlx,torch",
        SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE="metal_aot,metal_jit,torch",
        SGLANG_MPS_QWEN3_RADIX_DECODE="metal_jit,torch",
        SGLANG_MPS_RMSNORM="metal_jit,torch",
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
    assert selection.rmsnorm == (
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    )
    # A sibling not mentioned by the override remains independently gated.
    assert selection.silu_and_mul == (KernelBackend.TORCH,)
    assert selection.greedy_tail == ("mlx", "torch")
    assert selection.as_state()["greedy_tail"] == ["mlx", "torch"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        (
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
            "triton,torch",
            "not supported by this operation",
        ),
        (
            "SGLANG_MPS_RMSNORM",
            "not_a_provider,torch",
            "unsupported provider",
        ),
        (
            "SGLANG_MPS_QWEN3_RADIX_DECODE",
            "metal_jit,metal_jit,torch",
            "duplicate providers",
        ),
        (
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
            "mlx,mlx,torch",
            "duplicate providers",
        ),
        (
            "SGLANG_MPS_QWEN3_MODEL_FORWARD",
            "mlx",
            "must end with 'torch'",
        ),
        (
            "SGLANG_MPS_QWEN3_GREEDY_TAIL",
            "mlx",
            "must end with 'torch'",
        ),
        (
            "SGLANG_MPS_QWEN3_GREEDY_TAIL",
            "metal_jit,torch",
            "unsupported provider",
        ),
        (
            "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
            "metal_jit",
            "must end with 'torch'",
        ),
        (
            "SGLANG_MPS_RMSNORM",
            "metal_jit",
            "must end with 'torch'",
        ),
    ],
)
def test_selection_rejects_invalid_or_duplicate_provider(field, value, error):
    with pytest.raises(RuntimeError, match=error):
        _read_selection(**{field: value})


@pytest.mark.parametrize(
    "priority,expected",
    [
        (
            (
                KernelBackend.METAL_AOT,
                KernelBackend.METAL_JIT,
                KernelBackend.TORCH,
            ),
            KernelBackend.METAL_JIT,
        ),
        (
            (KernelBackend.METAL_AOT, KernelBackend.TORCH),
            KernelBackend.TORCH,
        ),
    ],
)
def test_missing_aot_falls_back_to_the_next_listed_provider(priority, expected):
    assert (
        choose_kernel_backend(
            priority,
            op_name="attention.qwen3_qknorm_rope_store",
            aot_available=False,
        )
        is expected
    )


def test_missing_aot_fails_when_it_is_the_only_allowed_provider():
    with pytest.raises(RuntimeError, match="no statically available provider"):
        choose_kernel_backend(
            (KernelBackend.METAL_AOT,),
            op_name="attention.qwen3_qknorm_rope_store",
            aot_available=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
