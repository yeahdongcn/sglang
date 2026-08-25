"""Startup selection for model-independent MPS Metal kernels.

The standard Torch ModelRunner remains the model and storage owner.  This
module only pins three semantic fused ops to an ordered implementation list.
Environment values are resolved once during MPS backend initialization; the
request path neither reads configuration nor compiles Metal source.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ALLOWED_BACKENDS = frozenset({KernelBackend.METAL_JIT, KernelBackend.TORCH})


def _parse_priority(name: str, value: Iterable[str] | str) -> tuple[KernelBackend, ...]:
    raw = (value,) if isinstance(value, str) else tuple(value)
    if not raw:
        raise RuntimeError(f"{name} must contain at least one backend")

    priority = []
    for item in raw:
        try:
            backend = (
                item
                if isinstance(item, KernelBackend)
                else KernelBackend(str(item).strip().lower())
            )
        except ValueError as exc:
            raise RuntimeError(f"{name} contains unsupported backend {item!r}") from exc
        if backend not in _ALLOWED_BACKENDS:
            allowed = ", ".join(sorted(item.value for item in _ALLOWED_BACKENDS))
            raise RuntimeError(
                f"{name} backend {backend.value!r} is unsupported; "
                f"allowed values are {allowed}"
            )
        if backend in priority:
            raise RuntimeError(f"{name} contains duplicate backends: {raw!r}")
        priority.append(backend)

    if priority[-1] is not KernelBackend.TORCH:
        raise RuntimeError(
            f"{name} must end with 'torch' as its explicit correctness fallback"
        )
    return tuple(priority)


@dataclass(frozen=True, slots=True)
class MpsGenericKernelSelection:
    rmsnorm: tuple[KernelBackend, ...]
    fused_add_rmsnorm: tuple[KernelBackend, ...]
    silu_and_mul: tuple[KernelBackend, ...]

    @classmethod
    def from_env(cls) -> MpsGenericKernelSelection:
        return cls(
            rmsnorm=_parse_priority(
                envs.SGLANG_MPS_RMSNORM.name,
                envs.SGLANG_MPS_RMSNORM.get(),
            ),
            fused_add_rmsnorm=_parse_priority(
                envs.SGLANG_MPS_FUSED_ADD_RMSNORM.name,
                envs.SGLANG_MPS_FUSED_ADD_RMSNORM.get(),
            ),
            silu_and_mul=_parse_priority(
                envs.SGLANG_MPS_SILU_AND_MUL.name,
                envs.SGLANG_MPS_SILU_AND_MUL.get(),
            ),
        )

    def metal_jit_ops(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, priority in (
                ("rmsnorm", self.rmsnorm),
                ("fused_add_rmsnorm", self.fused_add_rmsnorm),
                ("silu_and_mul", self.silu_and_mul),
            )
            if KernelBackend.METAL_JIT in priority
        )


@lru_cache(maxsize=1)
def configure_mps_generic_kernels() -> MpsGenericKernelSelection:
    """Validate, warm, then atomically publish the selected per-op priorities."""
    selection = MpsGenericKernelSelection.from_env()

    from sglang.kernels.fused_op import get_fused_op_backend

    forced_backend = get_fused_op_backend()
    if forced_backend not in (None, KernelBackend.TORCH, KernelBackend.METAL_JIT):
        raise RuntimeError(
            "MPS generic fused ops support only 'torch' and 'metal_jit'; "
            f"SGLANG_FORCE_FUSED_OP_BACKEND={forced_backend.value!r} is incompatible"
        )

    selected_metal_ops = set(selection.metal_jit_ops())
    if forced_backend is KernelBackend.METAL_JIT:
        selected_metal_ops.update({"rmsnorm", "fused_add_rmsnorm", "silu_and_mul"})

    if selected_metal_ops:
        from sglang.kernels.metal import is_metal_jit_available

        if not is_metal_jit_available():
            raise RuntimeError(
                "Metal JIT was selected for MPS generic ops, but "
                "torch.mps.compile_shader is unavailable"
            )

        started = time.perf_counter()
        if {"rmsnorm", "fused_add_rmsnorm"} & selected_metal_ops:
            from sglang.kernels.ops.layernorm._rmsnorm_metal_jit import (
                warmup_mps_rmsnorm_kernels,
            )

            warmup_mps_rmsnorm_kernels(
                rmsnorm="rmsnorm" in selected_metal_ops,
                fused_add_rmsnorm="fused_add_rmsnorm" in selected_metal_ops,
            )
        if "silu_and_mul" in selected_metal_ops:
            from sglang.kernels.ops.activation._silu_and_mul_metal_jit import (
                warmup_silu_and_mul_metal_kernel,
            )

            warmup_silu_and_mul_metal_kernel()
        logger.info(
            "Warmed MPS Metal JIT ops %s in %.3f seconds",
            sorted(selected_metal_ops),
            time.perf_counter() - started,
        )

    # Publish only after every selected library and entry point resolves. A
    # compile/load failure therefore aborts startup without a half-enabled set.
    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM

    _RMSNORM.set_priority(selection.rmsnorm)
    _FUSED_ADD_RMSNORM.set_priority(selection.fused_add_rmsnorm)
    _SILU_AND_MUL.set_priority(selection.silu_and_mul)
    return selection


def clear_mps_generic_kernel_configuration() -> None:
    """Restore class priorities and clear startup state for isolated tests."""
    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM

    for op in (_RMSNORM, _FUSED_ADD_RMSNORM, _SILU_AND_MUL):
        op.set_priority(None)
    configure_mps_generic_kernels.cache_clear()


__all__ = [
    "MpsGenericKernelSelection",
    "clear_mps_generic_kernel_configuration",
    "configure_mps_generic_kernels",
]
