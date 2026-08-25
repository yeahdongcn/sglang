"""Startup-time selection for Qwen3 Torch-stream Metal attention kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sglang.kernels.metal import is_metal_jit_available
from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs

_ALLOWED = frozenset(
    {KernelBackend.METAL_AOT, KernelBackend.METAL_JIT, KernelBackend.TORCH}
)
_DEFERRED_KV_ALLOWED = frozenset({KernelBackend.METAL_JIT, KernelBackend.TORCH})
_MODEL_BACKENDS = frozenset({"mlx", "torch"})


def _normalized_priority(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = (value,)
    else:
        raw = tuple(value or ())
    return tuple(str(item).strip().lower() for item in raw if str(item).strip())


def _read_model_priority(
    name: str, value: Iterable[str] | str | None
) -> tuple[str, ...]:
    priority = _normalized_priority(value)
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
            f"{name} must end with 'torch' as its correctness fallback; "
            f"found {priority!r}"
        )
    return priority


def _read_priority(
    name: str,
    value: Iterable[str] | str | None,
    *,
    allowed: frozenset[KernelBackend] = _ALLOWED,
) -> tuple[KernelBackend, ...]:
    normalized = _normalized_priority(value)
    if not normalized:
        raise RuntimeError(f"{name} must contain at least one provider")

    priority = []
    for item in normalized:
        try:
            backend = KernelBackend(item)
        except ValueError:
            allowed_names = ", ".join(sorted(candidate.value for candidate in allowed))
            raise RuntimeError(
                f"{name} contains unsupported provider {item!r}; "
                f"allowed values are {allowed_names}"
            ) from None
        if backend not in allowed:
            allowed_names = ", ".join(sorted(candidate.value for candidate in allowed))
            raise RuntimeError(
                f"{name} contains unsupported provider {item!r}; "
                f"allowed values are {allowed_names}"
            )
        if backend in priority:
            raise RuntimeError(f"{name} contains duplicate providers: {normalized!r}")
        priority.append(backend)

    if priority[-1] is not KernelBackend.TORCH:
        raise RuntimeError(
            f"{name} must end with 'torch' as its correctness fallback; "
            f"found {normalized!r}"
        )
    return tuple(priority)


@dataclass(frozen=True, slots=True)
class Qwen3MetalAttentionSelection:
    qknorm_rope_store: tuple[KernelBackend, ...]
    radix_decode: tuple[KernelBackend, ...]
    # Keep the original two-field constructor usable for callers that only
    # select the attention operators.  Whole-model decode and deferred KV
    # commit are opt-in extensions and default to their Torch fallbacks.
    model_forward: tuple[str, ...] = ("torch",)
    deferred_kv_commit: tuple[KernelBackend, ...] = (KernelBackend.TORCH,)

    @classmethod
    def from_env(cls) -> Qwen3MetalAttentionSelection:
        return cls(
            model_forward=_read_model_priority(
                "SGLANG_MPS_QWEN3_MODEL_FORWARD",
                envs.SGLANG_MPS_QWEN3_MODEL_FORWARD.get(),
            ),
            qknorm_rope_store=_read_priority(
                "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
                envs.SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE.get(),
            ),
            radix_decode=_read_priority(
                "SGLANG_MPS_QWEN3_RADIX_DECODE",
                envs.SGLANG_MPS_QWEN3_RADIX_DECODE.get(),
            ),
            deferred_kv_commit=_read_priority(
                "SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
                envs.SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT.get(),
                allowed=_DEFERRED_KV_ALLOWED,
            ),
        )

    def as_state(self) -> dict[str, Any]:
        return {
            "model_forward": list(self.model_forward),
            "qknorm_rope_store": [item.value for item in self.qknorm_rope_store],
            "radix_decode": [item.value for item in self.radix_decode],
            "deferred_kv_commit": [item.value for item in self.deferred_kv_commit],
        }


def choose_kernel_backend(
    priority: tuple[KernelBackend, ...],
    *,
    aot_available: bool,
    jit_available: bool | None = None,
) -> KernelBackend:
    """Choose the first statically available provider from one kernel gate."""
    if jit_available is None:
        jit_available = is_metal_jit_available()
    for backend in priority:
        if backend is KernelBackend.METAL_AOT and not aot_available:
            continue
        if backend is KernelBackend.METAL_JIT and not jit_available:
            continue
        return backend
    raise RuntimeError(
        "Qwen3 MPS attention has no available provider in priority "
        f"{[item.value for item in priority]!r}"
    )


def choose_model_backend(
    priority: tuple[str, ...],
    *,
    mlx_available: bool,
) -> str:
    """Choose the first statically available whole-model implementation."""
    for backend in priority:
        if backend == "mlx" and not mlx_available:
            continue
        return backend
    raise RuntimeError(
        f"Qwen3 MPS model-forward providers are unavailable for priority {priority!r}"
    )


__all__ = [
    "Qwen3MetalAttentionSelection",
    "choose_kernel_backend",
    "choose_model_backend",
]
