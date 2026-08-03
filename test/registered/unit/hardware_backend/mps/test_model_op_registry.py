"""CPU-safe contracts for lazy MPS model-operator registration."""

import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MPS_MODEL_OPERATOR_REGISTRY,
    MpsModelOperatorRegistry,
    MpsModelOperatorSpec,
    model_architectures,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _config(*architectures: str):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=list(architectures)),
        hf_text_config=None,
    )


def test_builtin_registry_routes_qwen3_without_importing_provider_module():
    spec = MPS_MODEL_OPERATOR_REGISTRY.resolve(_config("Qwen3ForCausalLM"))

    assert spec is not None
    assert spec.name == "qwen3_dense"
    assert spec.installer_path.endswith("plan:install_qwen3_operators")


def test_registry_import_does_not_eagerly_import_qwen_provider():
    script = """
import importlib
import sys

importlib.import_module('sglang.srt.hardware_backend.mps.model_ops.registry')
assert 'sglang.srt.hardware_backend.mps.model_ops.qwen3' not in sys.modules
assert 'sglang.srt.hardware_backend.mps.model_ops.plan' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_unknown_architecture_keeps_the_generic_torch_path():
    assert MPS_MODEL_OPERATOR_REGISTRY.resolve(_config("LlamaForCausalLM")) is None


def test_architecture_collection_deduplicates_text_and_outer_configs():
    config = SimpleNamespace(
        hf_config={"architectures": ["Qwen3ForCausalLM"]},
        hf_text_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
    )

    assert model_architectures(config) == frozenset({"Qwen3ForCausalLM"})


def test_registry_rejects_duplicate_names_and_architecture_owners():
    first = MpsModelOperatorSpec(
        name="first",
        architectures=frozenset({"ArchA"}),
        installer_path="package.module:install_a",
    )
    registry = MpsModelOperatorRegistry((first,))

    with pytest.raises(ValueError, match="duplicate.*name"):
        registry.register(
            MpsModelOperatorSpec(
                name="first",
                architectures=frozenset({"ArchB"}),
                installer_path="package.module:install_b",
            )
        )
    with pytest.raises(ValueError, match="duplicate.*architecture"):
        registry.register(
            MpsModelOperatorSpec(
                name="second",
                architectures=frozenset({"ArchA"}),
                installer_path="package.module:install_b",
            )
        )


def test_installer_is_imported_only_when_explicitly_loaded(monkeypatch):
    module_name = "test_mps_lazy_provider"
    module = types.ModuleType(module_name)
    module.install = lambda: "installed"
    monkeypatch.setitem(sys.modules, module_name, module)
    spec = MpsModelOperatorSpec(
        name="lazy",
        architectures=frozenset({"LazyArchitecture"}),
        installer_path=f"{module_name}:install",
    )
    registry = MpsModelOperatorRegistry((spec,))

    resolved = registry.resolve(_config("LazyArchitecture"))
    assert resolved is spec
    assert resolved.load_installer()() == "installed"


def test_installer_load_error_reports_spec_and_path():
    spec = MpsModelOperatorSpec(
        name="broken_family",
        architectures=frozenset({"BrokenArchitecture"}),
        installer_path="missing_mps_family_module:install",
    )

    with pytest.raises(RuntimeError) as exc_info:
        spec.load_installer()

    message = str(exc_info.value)
    assert "broken_family" in message
    assert "missing_mps_family_module:install" in message
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_multiple_architecture_hints_cannot_select_multiple_specs():
    registry = MpsModelOperatorRegistry(
        (
            MpsModelOperatorSpec(
                name="first",
                architectures=frozenset({"ArchA"}),
                installer_path="package.module:install_a",
            ),
            MpsModelOperatorSpec(
                name="second",
                architectures=frozenset({"ArchB"}),
                installer_path="package.module:install_b",
            ),
        )
    )

    with pytest.raises(RuntimeError, match="multiple MPS model operator specs"):
        registry.resolve(_config("ArchA", "ArchB"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
