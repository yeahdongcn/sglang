"""CPU-safe tests for model-neutral MPS operator routing."""

from __future__ import annotations

import subprocess
import sys
import threading
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.hardware_backend.mps.model_ops.registry import MpsModelOperatorSpec
from sglang.srt.hardware_backend.mps.model_ops.router import install_mps_operators
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _config(architecture: str):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
        hf_text_config=None,
    )


def test_unknown_model_route_does_not_import_family_provider():
    script = """
import os
import sys
from types import SimpleNamespace
from unittest import mock

from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops import router

os.environ['SGLANG_MPS_QWEN3_MODEL_FORWARD'] = 'not-a-provider'
os.environ['SGLANG_MPS_QWEN3_GREEDY_TAIL'] = 'also-invalid'
os.environ['SGLANG_MPS_RMSNORM'] = 'torch'
os.environ['SGLANG_MPS_FUSED_ADD_RMSNORM'] = 'torch'
os.environ['SGLANG_MPS_SILU_AND_MUL'] = 'torch'
generic = {
    'rmsnorm': KernelBackend.TORCH,
    'fused_add_rmsnorm': KernelBackend.TORCH,
    'silu_and_mul': KernelBackend.TORCH,
}
config = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=['UnknownForCausalLM']),
    hf_text_config=None,
)
with (
    mock.patch.object(router, 'get_fused_op_backend', return_value=None),
    mock.patch.object(router, '_configure_generic_mps_ops', return_value=generic),
):
    plan = router.install_mps_operators(object(), config, object())

assert plan.get_state()['model'] == 'UnknownForCausalLM'
assert plan.get_state()['generic_kernel_backends']['rmsnorm'] == 'torch'
assert 'model_forward' not in plan.get_state()['provider_priorities']
assert 'sglang.srt.hardware_backend.mps.model_ops.qwen3' not in sys.modules
assert 'sglang.srt.hardware_backend.mps.model_ops.plan' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_router_rejects_invalid_plan_and_closes_it(monkeypatch):
    module_name = "test_invalid_mps_family_installer"
    module = types.ModuleType(module_name)

    class InvalidPlan:
        model = "InvalidForCausalLM"
        enabled = True
        forward_lock = threading.RLock()

        def __init__(self):
            self.closed = False

        def invalidate_views(self):
            return

        def close(self):
            self.closed = True

    invalid_plan = InvalidPlan()
    module.install = mock.Mock(return_value=invalid_plan)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec = MpsModelOperatorSpec(
        name="invalid_family",
        architectures=frozenset({"InvalidForCausalLM"}),
        installer_path=f"{module_name}:install",
    )

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.router."
            "MPS_MODEL_OPERATOR_REGISTRY.resolve",
            return_value=spec,
        ),
        pytest.raises(TypeError) as exc_info,
    ):
        install_mps_operators(object(), _config("InvalidForCausalLM"), object())

    message = str(exc_info.value)
    assert "invalid_family" in message
    assert f"{module_name}:install" in message
    assert "get_state" in message
    assert invalid_plan.closed


@pytest.mark.parametrize("lock_factory", [threading.RLock, threading.Lock])
def test_router_requires_the_shared_reentrant_serving_lock(monkeypatch, lock_factory):
    module_name = "test_wrong_lock_mps_family_installer"
    module = types.ModuleType(module_name)

    class WrongLockPlan:
        model = "WrongLockForCausalLM"
        enabled = True

        def __init__(self):
            self.forward_lock = lock_factory()
            self.closed = False

        def invalidate_views(self):
            return

        def close(self):
            self.closed = True

        def get_state(self):
            return {"enabled": True}

    plan = WrongLockPlan()
    module.install = mock.Mock(return_value=plan)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec = MpsModelOperatorSpec(
        name="wrong_lock_family",
        architectures=frozenset({"WrongLockForCausalLM"}),
        installer_path=f"{module_name}:install",
    )

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.router."
            "MPS_MODEL_OPERATOR_REGISTRY.resolve",
            return_value=spec,
        ),
        pytest.raises(TypeError, match="shared MPS_OPERATOR_FORWARD_LOCK"),
    ):
        install_mps_operators(object(), _config("WrongLockForCausalLM"), object())

    assert plan.closed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
