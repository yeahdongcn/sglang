"""Model-neutral routing for lazily imported MPS operator plans."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.srt.hardware_backend.mps.model_ops.base import (
    GenericMpsOperatorPlan,
    MpsOperatorPlanProtocol,
    configure_generic_mps_ops,
    validate_mps_operator_plan,
)
from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MPS_MODEL_OPERATOR_REGISTRY,
    model_architectures,
)
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    MpsGenericOperatorSelection,
)

logger = logging.getLogger(__name__)


def _configure_generic_mps_ops(selection, forced_backend):
    """Patchable routing seam for model-independent semantic operators."""

    return configure_generic_mps_ops(selection, forced_backend)


def _close_invalid_plan(plan: Any) -> None:
    close = getattr(plan, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.exception("Failed to close an invalid MPS model operator plan")


def install_mps_operators(
    model: torch.nn.Module,
    model_config: Any,
    server_args: Any,
    *,
    req_to_token_pool: Any = None,
    token_to_kv_pool: Any = None,
) -> MpsOperatorPlanProtocol:
    """Route one model to a family installer or the generic Torch plan.

    Architecture registration is only an import-routing hint. A family
    installer remains responsible for exact model, tensor, storage, and
    server-mode validation. Unknown models never import a family provider.
    """

    spec = MPS_MODEL_OPERATOR_REGISTRY.resolve(model_config)
    if spec is not None:
        installer = spec.load_installer()
        plan = installer(
            model,
            model_config,
            server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
        )
        try:
            return validate_mps_operator_plan(
                plan,
                spec_name=spec.name,
                installer_path=spec.installer_path,
            )
        except Exception:
            _close_invalid_plan(plan)
            raise

    # Unknown models must not parse family-specific gates. A stale or forced
    # Qwen3 priority is irrelevant to their model-neutral semantic operators.
    selection = MpsGenericOperatorSelection.from_env()
    forced_backend = get_fused_op_backend()
    generic_backends = _configure_generic_mps_ops(selection, forced_backend)
    architectures = model_architectures(model_config)
    model_name = (
        next(iter(architectures)) if len(architectures) == 1 else type(model).__name__
    )
    reason = "model has no registered MPS model-operator spec"
    logger.info("MPS model-op plan falls back to Torch native: %s", reason)
    return GenericMpsOperatorPlan(
        model=model_name,
        fallback_reason=reason,
        provider_priorities=selection.as_state(),
        generic_kernel_backends={
            name: backend.value for name, backend in generic_backends.items()
        },
    )


__all__ = ["install_mps_operators"]
