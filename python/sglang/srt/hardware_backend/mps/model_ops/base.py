"""Shared contracts for model-specific MPS runtime operators.

The standard Torch ``ModelRunner`` owns model parameters, scheduler state,
request tables, and KV storage.  A model provider may only borrow that state
for an explicitly selected semantic operation.  These types keep the
selection and lifecycle boundary independent of MLX or any one model family.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from sglang.kernels.spec import KernelBackend

logger = logging.getLogger(__name__)

# ModelRunner requires one stable lock identity across an online replacement,
# including transitions between a family-specific plan and the generic fallback.
MPS_OPERATOR_FORWARD_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class MpsGateDecision:
    """A synchronization-free provider eligibility result."""

    eligible: bool
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.eligible and self.reason is not None:
            raise ValueError("an eligible MPS gate decision cannot have a reason")
        if not self.eligible and not self.reason:
            raise ValueError("an ineligible MPS gate decision requires a reason")


class MpsModelForwardProvider(Protocol):
    """Lifecycle contract for a provider bound to one Torch model instance.

    ``should_run`` must make its decision before launching device work and
    must not synchronize.  Once ``forward`` starts, an exception is fatal;
    providers must never partially mutate KV state and then fall back to
    Torch.
    """

    def should_run(self, forward_batch: Any, **kwargs: Any) -> bool: ...

    def forward(self, model: Any, *args: Any, **kwargs: Any) -> Any: ...

    def invalidate_views(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class MpsOperatorPlanProtocol(Protocol):
    """Lifecycle surface consumed by the backend-neutral ``ModelRunner``."""

    model: str
    enabled: bool
    forward_lock: Any

    def invalidate_views(self) -> None: ...

    def close(self) -> None: ...

    def get_state(self) -> Mapping[str, Any]: ...


class MpsModelOperatorInstaller(Protocol):
    """Signature implemented by one lazily imported model-family installer."""

    def __call__(
        self,
        model: Any,
        model_config: Any,
        server_args: Any,
        *,
        req_to_token_pool: Any = None,
        token_to_kv_pool: Any = None,
    ) -> MpsOperatorPlanProtocol: ...


def validate_mps_operator_plan(
    plan: Any,
    *,
    spec_name: str,
    installer_path: str,
) -> MpsOperatorPlanProtocol:
    """Validate a contributor plan before ModelRunner publishes its lifecycle."""

    missing_attributes = [
        name for name in ("model", "enabled", "forward_lock") if not hasattr(plan, name)
    ]
    missing_methods = [
        name
        for name in ("invalidate_views", "close", "get_state")
        if not callable(getattr(plan, name, None))
    ]
    lock = getattr(plan, "forward_lock", None)
    invalid_lock = lock is not MPS_OPERATOR_FORWARD_LOCK

    state_error: Optional[str] = None
    if not missing_methods and not missing_attributes and not invalid_lock:
        try:
            state = plan.get_state()
        except Exception as exc:
            state_error = f"get_state() raised {type(exc).__name__}: {exc}"
        else:
            if not isinstance(state, Mapping):
                state_error = (
                    f"get_state() must return a mapping; found {type(state).__name__}"
                )

    if missing_attributes or missing_methods or invalid_lock or state_error is not None:
        details = []
        if missing_attributes:
            details.append(f"missing attributes {missing_attributes!r}")
        if missing_methods:
            details.append(f"missing methods {missing_methods!r}")
        if invalid_lock:
            details.append("forward_lock is not the shared MPS_OPERATOR_FORWARD_LOCK")
        if state_error is not None:
            details.append(state_error)
        raise TypeError(
            f"MPS model operator installer for spec {spec_name!r} at "
            f"{installer_path!r} returned {type(plan).__name__}, which does not "
            f"satisfy the plan lifecycle contract: {'; '.join(details)}"
        )
    return plan


@dataclass(frozen=True, slots=True)
class _PublishedBinding:
    target: Any
    attribute: str
    provider: Any
    previous: Any
    attribute_existed: bool


class MpsBindingPublication:
    """Publish provider attributes as one rollback-capable transaction."""

    def __init__(self) -> None:
        self._bindings: list[_PublishedBinding] = []
        self._committed = False

    def __enter__(self) -> MpsBindingPublication:
        return self

    def bind(self, target: Any, attribute: str, provider: Any) -> None:
        if self._committed:
            raise RuntimeError("cannot add an MPS binding after publication commit")
        attribute_existed = hasattr(target, attribute)
        previous = getattr(target, attribute, None)
        binding = _PublishedBinding(
            target=target,
            attribute=attribute,
            provider=provider,
            previous=previous,
            attribute_existed=attribute_existed,
        )
        # Record before calling a contributor-owned setter: a setter may mutate
        # its target and then raise, in which case rollback still has ownership.
        self._bindings.append(binding)
        setattr(target, attribute, provider)

    def commit(self) -> None:
        self._committed = True

    def rollback(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for binding in reversed(self._bindings):
            try:
                if (
                    getattr(binding.target, binding.attribute, None)
                    is not binding.provider
                ):
                    continue
                if binding.attribute_existed:
                    setattr(binding.target, binding.attribute, binding.previous)
                else:
                    delattr(binding.target, binding.attribute)
            except BaseException as exc:
                # Continue restoring older bindings. Startup is already fatal,
                # but retaining an earlier compiled provider would otherwise
                # leak memory and obscure the original publication error.
                failures.append(exc)
        self._bindings.clear()
        return failures

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._committed:
            failures = self.rollback()
            if failures:
                if exc_type is None:
                    raise RuntimeError(
                        "failed to roll back an unpublished MPS provider binding"
                    ) from failures[0]
                for failure in failures:
                    logger.error(
                        "Failed to roll back an MPS provider binding while "
                        "preserving the original publication error",
                        exc_info=(
                            type(failure),
                            failure,
                            failure.__traceback__,
                        ),
                    )
        return False


@dataclass
class GenericMpsOperatorPlan:
    """Model-neutral plan returned when no family installer is registered."""

    model: str
    fallback_reason: str
    provider_priorities: dict[str, Any] = field(default_factory=dict)
    generic_kernel_backends: dict[str, str] = field(default_factory=dict)
    provider_spec: Optional[str] = None
    forward_lock: Any = field(
        default_factory=lambda: MPS_OPERATOR_FORWARD_LOCK,
        repr=False,
        compare=False,
    )
    enabled: bool = field(init=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.enabled = any(
            backend != KernelBackend.TORCH.value
            for backend in self.generic_kernel_backends.values()
        )

    def invalidate_views(self) -> None:
        return

    def close(self) -> None:
        self._closed = True

    def get_state(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled and not self._closed,
            "model": self.model,
            "provider_spec": self.provider_spec,
            "provider_priorities": self.provider_priorities,
            "generic_kernel_backends": self.generic_kernel_backends,
            "attention_backend": "torch_native",
            "qkv_kernel_backend": "torch",
            "decode_kernel_backend": "torch",
            "deferred_kv_commit_backend": "off",
            "qkv_fallback_reason": self.fallback_reason,
            "decode_fallback_reason": self.fallback_reason,
            "whole_model_backend": "off",
            "whole_model_fallback_reason": self.fallback_reason,
            "patched_attention_modules": 0,
            "patched_qkv_modules": 0,
            "patched_decode_modules": 0,
            "attention_qkv_call_count": 0,
            "attention_qkv_fallback_count": 0,
            "attention_decode_call_count": 0,
            "attention_decode_fallback_count": 0,
            "whole_model_call_count": 0,
            "whole_model_decode_call_count": 0,
            "whole_model_max_decode_batch_size": 0,
            "whole_model_prefill_call_count": 0,
            "whole_model_selector_call_count": 0,
            "whole_model_selector_fallback_count": 0,
            "whole_model_last_fallback_reason": None,
            "whole_model_greedy_tail_enabled": False,
            "whole_model_greedy_tail_backend": "off",
            "whole_model_greedy_tail_static_fallback_reason": self.fallback_reason,
            "whole_model_greedy_tail_call_count": 0,
            "whole_model_greedy_tail_torch_call_count": 0,
            "whole_model_greedy_tail_fallback_count": 0,
            "whole_model_greedy_tail_last_fallback_reason": None,
            "whole_model_compile_enabled": False,
            "whole_model_compile_total_enabled": False,
            "whole_model_compile_primary_variant": "off",
            "whole_model_compile_warmup_count": 0,
            "whole_model_compile_call_count": 0,
            "whole_model_compile_total_warmup_count": 0,
            "whole_model_compile_total_call_count": 0,
            "whole_model_compile_fallback_count": 0,
            "whole_model_greedy_compile_enabled": False,
            "whole_model_greedy_compile_warmup_count": 0,
            "whole_model_greedy_compile_call_count": 0,
        }


def configure_generic_mps_ops(
    selection: Any,
    forced_backend: Optional[KernelBackend],
) -> dict[str, KernelBackend]:
    """Pin and warm model-independent MPS semantic operators."""

    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM
    from sglang.srt.hardware_backend.mps.model_ops.selection import (
        choose_kernel_backend,
    )

    _RMSNORM.set_priority(selection.rmsnorm)
    _FUSED_ADD_RMSNORM.set_priority(selection.fused_add_rmsnorm)
    _SILU_AND_MUL.set_priority(selection.silu_and_mul)

    if forced_backend is None:
        rms_backend = choose_kernel_backend(
            selection.rmsnorm,
            op_name="SGLANG_MPS_RMSNORM",
        )
        fused_backend = choose_kernel_backend(
            selection.fused_add_rmsnorm,
            op_name="SGLANG_MPS_FUSED_ADD_RMSNORM",
        )
        silu_backend = choose_kernel_backend(
            selection.silu_and_mul,
            op_name="SGLANG_MPS_SILU_AND_MUL",
        )
    elif forced_backend in {KernelBackend.TORCH, KernelBackend.METAL_JIT}:
        rms_backend = fused_backend = silu_backend = forced_backend
    else:
        raise RuntimeError(
            "The MPS RMSNorm/SiLU operators do not support the globally forced "
            f"backend {forced_backend.value!r}; use 'torch' or 'metal_jit'"
        )

    if (
        rms_backend is KernelBackend.METAL_JIT
        or fused_backend is KernelBackend.METAL_JIT
    ):
        from sglang.kernels.ops.layernorm._rmsnorm_metal_jit import (
            warmup_mps_rmsnorm_kernels,
        )

        warmup_mps_rmsnorm_kernels(
            rmsnorm=rms_backend is KernelBackend.METAL_JIT,
            fused_add_rmsnorm=fused_backend is KernelBackend.METAL_JIT,
        )
    if silu_backend is KernelBackend.METAL_JIT:
        from sglang.kernels.ops.activation._silu_and_mul_metal_jit import (
            warmup_silu_and_mul_metal_kernel,
        )

        warmup_silu_and_mul_metal_kernel()
    return {
        "rmsnorm": rms_backend,
        "fused_add_rmsnorm": fused_backend,
        "silu_and_mul": silu_backend,
    }


__all__ = [
    "GenericMpsOperatorPlan",
    "MPS_OPERATOR_FORWARD_LOCK",
    "MpsBindingPublication",
    "MpsGateDecision",
    "MpsModelForwardProvider",
    "MpsModelOperatorInstaller",
    "MpsOperatorPlanProtocol",
    "configure_generic_mps_ops",
    "validate_mps_operator_plan",
]
