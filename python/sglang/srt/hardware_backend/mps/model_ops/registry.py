"""Declarative registry for model-specific MPS operator installers.

Registry entries contain import strings rather than imported functions. This
keeps model-specific modules lazy and lets the MPS platform remain neutral as
new model families gain optimized operators.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def model_architectures(model_config: Any) -> tuple[str, ...]:
    """Return unique Hugging Face architecture names in declaration order."""
    architectures: list[str] = []
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if config is None:
            continue
        for architecture in _config_value(config, "architectures", None) or ():
            name = str(architecture).strip()
            if name and name not in architectures:
                architectures.append(name)
    return tuple(architectures)


@dataclass(frozen=True, slots=True)
class MpsModelOperatorSpec:
    """One lazy model-family installer registration."""

    name: str
    architectures: frozenset[str]
    installer_path: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MPS model-operator spec name must not be empty")
        if not self.architectures or any(
            not item.strip() for item in self.architectures
        ):
            raise ValueError(
                "MPS model-operator spec must declare non-empty architectures"
            )
        module_name, separator, attribute = self.installer_path.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(
                "MPS model-operator installer path must use 'module:callable'"
            )

    def load_installer(self) -> Callable[..., object | None]:
        module_name, _, attribute = self.installer_path.partition(":")
        installer = getattr(importlib.import_module(module_name), attribute)
        if not callable(installer):
            raise TypeError(
                f"MPS model-operator installer {self.installer_path!r} is not callable"
            )
        return installer


class MpsModelOperatorRegistry:
    """Resolve one installer from the model's declared architecture."""

    def __init__(self, specs: Iterable[MpsModelOperatorSpec] = ()) -> None:
        self._specs_by_architecture: dict[str, MpsModelOperatorSpec] = {}
        self._names: set[str] = set()
        for spec in specs:
            self.register(spec)

    def register(self, spec: MpsModelOperatorSpec) -> None:
        if spec.name in self._names:
            raise ValueError(f"duplicate MPS model-operator spec name {spec.name!r}")
        conflicts = sorted(
            architecture
            for architecture in spec.architectures
            if architecture in self._specs_by_architecture
        )
        if conflicts:
            raise ValueError(
                f"MPS model-operator architectures already registered: {conflicts!r}"
            )
        self._names.add(spec.name)
        for architecture in spec.architectures:
            self._specs_by_architecture[architecture] = spec

    def resolve(self, model_config: Any) -> MpsModelOperatorSpec | None:
        matches = {
            self._specs_by_architecture[architecture]
            for architecture in model_architectures(model_config)
            if architecture in self._specs_by_architecture
        }
        if len(matches) > 1:
            names = sorted(spec.name for spec in matches)
            raise RuntimeError(
                f"model config matches multiple MPS model-operator specs: {names!r}"
            )
        return next(iter(matches), None)


# Model-specific slices add declarative entries here. Import strings keep their
# implementations unloaded until an MPS ModelRunner actually selects them.
MPS_MODEL_OPERATOR_REGISTRY = MpsModelOperatorRegistry(
    [
        MpsModelOperatorSpec(
            name="qwen3_dense_attention",
            architectures=frozenset({"Qwen3ForCausalLM"}),
            installer_path=(
                "sglang.srt.hardware_backend.mps.model_ops.plan:"
                "install_qwen3_metal_attention"
            ),
        )
    ]
)


__all__ = [
    "MPS_MODEL_OPERATOR_REGISTRY",
    "MpsModelOperatorRegistry",
    "MpsModelOperatorSpec",
    "model_architectures",
]
