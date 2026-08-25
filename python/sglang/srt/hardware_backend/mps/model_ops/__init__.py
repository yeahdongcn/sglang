"""Lazy model-operator routing for the Torch-owned MPS runtime."""

from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MPS_MODEL_OPERATOR_REGISTRY,
    MpsModelOperatorRegistry,
    MpsModelOperatorSpec,
)
from sglang.srt.hardware_backend.mps.model_ops.router import (
    install_mps_model_operators,
)

__all__ = [
    "MPS_MODEL_OPERATOR_REGISTRY",
    "MpsModelOperatorRegistry",
    "MpsModelOperatorSpec",
    "install_mps_model_operators",
]
