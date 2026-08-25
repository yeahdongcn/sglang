"""Model-neutral entry point for installing MPS runtime operators."""

from __future__ import annotations

from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MPS_MODEL_OPERATOR_REGISTRY,
    MpsModelOperatorRegistry,
)


def install_mps_model_operators(
    model,
    model_config,
    server_args,
    *,
    req_to_token_pool,
    token_to_kv_pool,
    registry: MpsModelOperatorRegistry = MPS_MODEL_OPERATOR_REGISTRY,
) -> object | None:
    """Install the lazily selected model-family plan, if one is registered."""
    spec = registry.resolve(model_config)
    if spec is None:
        return None

    plan = spec.load_installer()(
        model,
        model_config,
        server_args,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
    )
    if plan is None:
        return None
    if not callable(getattr(plan, "close", None)):
        raise TypeError(
            f"MPS model-operator installer {spec.installer_path!r} returned "
            f"{type(plan).__name__}, which does not implement close()"
        )
    return plan


__all__ = ["install_mps_model_operators"]
