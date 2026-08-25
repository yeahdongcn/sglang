"""Lazy architecture routing for model-specific MPS operator plans.

Architecture names are only a cheap import-routing hint.  A selected
installer remains responsible for exact model type, tensor shape, dtype,
layout, server-mode, and storage-lifetime validation before publication.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sglang.srt.hardware_backend.mps.model_ops.base import (
    MpsModelOperatorInstaller,
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def model_architectures(model_config: Any) -> frozenset[str]:
    """Return the unique HF architecture hints visible to the runner."""
    architectures: set[str] = set()
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if config is None:
            continue
        values = _config_value(config, "architectures", None) or []
        if isinstance(values, str):
            values = (values,)
        architectures.update(str(item) for item in values)
    return frozenset(architectures)


@dataclass(frozen=True, slots=True)
class MpsModelOperatorSpec:
    """One lazily imported model-family operator-plan installer."""

    name: str
    architectures: frozenset[str]
    installer_path: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an MPS model operator spec requires a name")
        if not self.architectures:
            raise ValueError(
                f"MPS model operator spec {self.name!r} requires an architecture"
            )
        module_name, separator, attribute = self.installer_path.partition(":")
        if not module_name or separator != ":" or not attribute:
            raise ValueError(
                "an MPS installer path must have the form 'module:attribute'; "
                f"found {self.installer_path!r}"
            )

    def load_installer(self) -> MpsModelOperatorInstaller:
        module_name, _, attribute = self.installer_path.partition(":")
        try:
            module = importlib.import_module(module_name)
            installer = getattr(module, attribute)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MPS model operator installer for spec "
                f"{self.name!r} at {self.installer_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(installer):
            raise TypeError(
                f"MPS model operator installer for spec {self.name!r} at "
                f"{self.installer_path!r} is not callable"
            )
        return installer


class MpsModelOperatorRegistry:
    """Immutable-by-convention registry with explicit duplicate checks."""

    def __init__(self, specs: Iterable[MpsModelOperatorSpec] = ()) -> None:
        self._specs: list[MpsModelOperatorSpec] = []
        self._by_architecture: dict[str, MpsModelOperatorSpec] = {}
        self._names: set[str] = set()
        for spec in specs:
            self.register(spec)

    def register(self, spec: MpsModelOperatorSpec) -> None:
        if spec.name in self._names:
            raise ValueError(f"duplicate MPS model operator spec name {spec.name!r}")
        duplicates = sorted(
            architecture
            for architecture in spec.architectures
            if architecture in self._by_architecture
        )
        if duplicates:
            owners = {
                architecture: self._by_architecture[architecture].name
                for architecture in duplicates
            }
            raise ValueError(
                "duplicate MPS model operator architecture registration: "
                f"{owners!r} conflicts with {spec.name!r}"
            )
        self._specs.append(spec)
        self._names.add(spec.name)
        for architecture in spec.architectures:
            self._by_architecture[architecture] = spec

    def resolve(self, model_config: Any) -> Optional[MpsModelOperatorSpec]:
        matches = {
            self._by_architecture[architecture]
            for architecture in model_architectures(model_config)
            if architecture in self._by_architecture
        }
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                "HF model configuration matched multiple MPS model operator specs: "
                f"{sorted(spec.name for spec in matches)!r}"
            )
        return next(iter(matches))

    @property
    def specs(self) -> tuple[MpsModelOperatorSpec, ...]:
        return tuple(self._specs)


MPS_MODEL_OPERATOR_REGISTRY = MpsModelOperatorRegistry(
    (
        MpsModelOperatorSpec(
            name="qwen3_dense",
            architectures=frozenset({"Qwen3ForCausalLM"}),
            installer_path=(
                "sglang.srt.hardware_backend.mps.model_ops.plan:install_qwen3_operators"
            ),
        ),
    )
)


__all__ = [
    "MPS_MODEL_OPERATOR_REGISTRY",
    "MpsModelOperatorRegistry",
    "MpsModelOperatorSpec",
    "model_architectures",
]
