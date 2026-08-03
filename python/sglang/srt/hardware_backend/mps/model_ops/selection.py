"""Startup-time provider selection for the Torch-owned MPS model path.

The MPS backend has one runner and one Torch storage lifecycle.  This module
only turns the per-operator environment declarations into immutable startup
configuration.  Dispatching a request never reads an environment variable or
re-runs this validation; the selected provider is pinned by the model-op plan.

The shape is intentionally close to the useful part of vLLM's kernel
selection model: every semantic operation has an ordered provider list, and a
provider is selected independently from its siblings.  It is not an FX/IR
lowering layer -- the current MPS compile backend is eager, so introducing a
second graph compiler here would obscure ownership without removing a runtime
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs

_MODEL_BACKENDS = frozenset({"torch", "mlx"})
_QWEN3_BACKENDS = frozenset(
    {KernelBackend.METAL_AOT, KernelBackend.METAL_JIT, KernelBackend.TORCH}
)
_GENERIC_MPS_BACKENDS = frozenset({KernelBackend.METAL_JIT, KernelBackend.TORCH})


def _as_tuple(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _read_model_priority(
    name: str, value: Iterable[str] | str | None
) -> tuple[str, ...]:
    priority = _as_tuple(value)
    if not priority:
        raise RuntimeError(f"{name} must contain at least one provider")
    unknown = [item for item in priority if item not in _MODEL_BACKENDS]
    if unknown:
        allowed = ", ".join(sorted(_MODEL_BACKENDS))
        raise RuntimeError(
            f"{name} contains unsupported provider(s) {unknown!r}; "
            f"allowed values are {allowed}"
        )
    if len(set(priority)) != len(priority):
        raise RuntimeError(f"{name} contains duplicate providers: {priority!r}")
    if priority[-1] != "torch":
        raise RuntimeError(
            f"{name} must end with 'torch' as its explicit correctness fallback; "
            f"found {priority!r}"
        )
    return priority


def _read_kernel_priority(
    name: str,
    value: Iterable[str] | str | None,
    allowed: frozenset[KernelBackend],
) -> tuple[KernelBackend, ...]:
    raw = _as_tuple(value)
    if not raw:
        raise RuntimeError(f"{name} must contain at least one provider")
    priority = []
    for item in raw:
        try:
            backend = KernelBackend(item)
        except ValueError:
            allowed_names = ", ".join(sorted(backend.value for backend in allowed))
            raise RuntimeError(
                f"{name} contains unsupported provider {item!r}; "
                f"allowed values are {allowed_names}"
            ) from None
        if backend not in allowed:
            allowed_names = ", ".join(sorted(item.value for item in allowed))
            raise RuntimeError(
                f"{name} contains provider {item!r}, which is not supported by "
                f"this operation; allowed values are {allowed_names}"
            )
        if backend in priority:
            raise RuntimeError(f"{name} contains duplicate providers: {raw!r}")
        priority.append(backend)
    if priority[-1] is not KernelBackend.TORCH:
        raise RuntimeError(
            f"{name} must end with 'torch' as its explicit correctness fallback; "
            f"found {raw!r}"
        )
    return tuple(priority)


@dataclass(frozen=True)
class MpsGenericOperatorSelection:
    """Model-neutral provider priorities safe to parse for every model."""

    rmsnorm: tuple[KernelBackend, ...]
    fused_add_rmsnorm: tuple[KernelBackend, ...]
    silu_and_mul: tuple[KernelBackend, ...]

    @classmethod
    def from_env(cls) -> MpsGenericOperatorSelection:
        return cls(
            rmsnorm=_read_kernel_priority(
                "SGLANG_MPS_RMSNORM",
                envs.SGLANG_MPS_RMSNORM.get(),
                _GENERIC_MPS_BACKENDS,
            ),
            fused_add_rmsnorm=_read_kernel_priority(
                "SGLANG_MPS_FUSED_ADD_RMSNORM",
                envs.SGLANG_MPS_FUSED_ADD_RMSNORM.get(),
                _GENERIC_MPS_BACKENDS,
            ),
            silu_and_mul=_read_kernel_priority(
                "SGLANG_MPS_SILU_AND_MUL",
                envs.SGLANG_MPS_SILU_AND_MUL.get(),
                _GENERIC_MPS_BACKENDS,
            ),
        )

    def as_state(self) -> dict[str, Any]:
        return {
            "rmsnorm": [item.value for item in self.rmsnorm],
            "fused_add_rmsnorm": [item.value for item in self.fused_add_rmsnorm],
            "silu_and_mul": [item.value for item in self.silu_and_mul],
        }


@dataclass(frozen=True)
class MpsOperatorSelection:
    """Validated, immutable provider priorities for one ModelRunner."""

    model_forward: tuple[str, ...]
    greedy_tail: tuple[str, ...]
    qknorm_rope_store: tuple[KernelBackend, ...]
    radix_decode: tuple[KernelBackend, ...]
    deferred_kv_commit: tuple[KernelBackend, ...]
    rmsnorm: tuple[KernelBackend, ...]
    fused_add_rmsnorm: tuple[KernelBackend, ...]
    silu_and_mul: tuple[KernelBackend, ...]

    @classmethod
    def from_env(cls) -> MpsOperatorSelection:
        """Read and validate all MPS selections once at worker startup."""
        generic = MpsGenericOperatorSelection.from_env()
        return cls(
            model_forward=_read_model_priority(
                "SGLANG_MPS_QWEN3_MODEL_FORWARD",
                envs.SGLANG_MPS_QWEN3_MODEL_FORWARD.get(),
            ),
            greedy_tail=_read_model_priority(
                "SGLANG_MPS_QWEN3_GREEDY_TAIL",
                envs.SGLANG_MPS_QWEN3_GREEDY_TAIL.get(),
            ),
            qknorm_rope_store=_read_kernel_priority(
                "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
                envs.SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE.get(),
                _QWEN3_BACKENDS,
            ),
            radix_decode=_read_kernel_priority(
                "SGLANG_MPS_QWEN3_RADIX_DECODE",
                envs.SGLANG_MPS_QWEN3_RADIX_DECODE.get(),
                _QWEN3_BACKENDS,
            ),
            deferred_kv_commit=_read_kernel_priority(
                "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
                envs.SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT.get(),
                _GENERIC_MPS_BACKENDS,
            ),
            rmsnorm=generic.rmsnorm,
            fused_add_rmsnorm=generic.fused_add_rmsnorm,
            silu_and_mul=generic.silu_and_mul,
        )

    def as_state(self) -> dict[str, Any]:
        """Return JSON-safe priorities for server-info observability."""
        return {
            "model_forward": list(self.model_forward),
            "greedy_tail": list(self.greedy_tail),
            "qknorm_rope_store": [item.value for item in self.qknorm_rope_store],
            "radix_decode": [item.value for item in self.radix_decode],
            "deferred_kv_commit": [item.value for item in self.deferred_kv_commit],
            "rmsnorm": [item.value for item in self.rmsnorm],
            "fused_add_rmsnorm": [item.value for item in self.fused_add_rmsnorm],
            "silu_and_mul": [item.value for item in self.silu_and_mul],
        }


def choose_kernel_backend(
    priority: tuple[KernelBackend, ...],
    *,
    op_name: str,
    aot_available: Optional[bool] = None,
) -> KernelBackend:
    """Choose the first statically available provider in ``priority``.

    A missing optional AOT wheel is a normal selector miss when a later
    provider is listed.  Compilation/load failures after this function returns
    are intentionally not swallowed by the plan: startup should fail rather
    than silently changing the benchmarked provider.
    """
    for backend in priority:
        if backend is KernelBackend.METAL_AOT:
            if aot_available is False:
                continue
            if aot_available is None:
                raise ValueError(
                    "aot_available must be supplied when selecting a Metal AOT op"
                )
        return backend
    raise RuntimeError(
        f"{op_name} has no statically available provider in priority "
        f"{[item.value for item in priority]!r}; add 'torch' or install the "
        "requested kernel provider"
    )


def choose_model_backend(
    priority: tuple[str, ...],
    *,
    mlx_available: bool,
    op_name: str = "SGLANG_MPS_QWEN3_MODEL_FORWARD",
) -> str:
    """Choose the first available whole-model implementation."""
    for backend in priority:
        if backend == "mlx" and not mlx_available:
            continue
        return backend
    raise RuntimeError(
        f"{op_name} requested providers that are not available: {priority!r}"
    )


__all__ = [
    "MpsGenericOperatorSelection",
    "MpsOperatorSelection",
    "choose_kernel_backend",
    "choose_model_backend",
]
