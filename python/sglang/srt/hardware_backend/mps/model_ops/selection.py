"""Startup-time selection for Qwen3 Torch-stream Metal attention kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sglang.kernels.metal import is_metal_jit_available
from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs

_ALLOWED = frozenset(
    {KernelBackend.METAL_AOT, KernelBackend.METAL_JIT, KernelBackend.TORCH}
)


def _read_priority(
    name: str, value: Iterable[str] | str | None
) -> tuple[KernelBackend, ...]:
    if isinstance(value, str):
        raw = (value,)
    else:
        raw = tuple(value or ())
    normalized = tuple(str(item).strip().lower() for item in raw if str(item).strip())
    if not normalized:
        raise RuntimeError(f"{name} must contain at least one provider")

    priority = []
    for item in normalized:
        try:
            backend = KernelBackend(item)
        except ValueError:
            allowed = ", ".join(sorted(candidate.value for candidate in _ALLOWED))
            raise RuntimeError(
                f"{name} contains unsupported provider {item!r}; "
                f"allowed values are {allowed}"
            ) from None
        if backend not in _ALLOWED:
            allowed = ", ".join(sorted(candidate.value for candidate in _ALLOWED))
            raise RuntimeError(
                f"{name} contains unsupported provider {item!r}; "
                f"allowed values are {allowed}"
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

    @classmethod
    def from_env(cls) -> Qwen3MetalAttentionSelection:
        return cls(
            qknorm_rope_store=_read_priority(
                "SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
                envs.SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE.get(),
            ),
            radix_decode=_read_priority(
                "SGLANG_MPS_QWEN3_RADIX_DECODE",
                envs.SGLANG_MPS_QWEN3_RADIX_DECODE.get(),
            ),
        )

    def as_state(self) -> dict[str, list[str]]:
        return {
            "qknorm_rope_store": [item.value for item in self.qknorm_rope_store],
            "radix_decode": [item.value for item in self.radix_decode],
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


__all__ = [
    "Qwen3MetalAttentionSelection",
    "choose_kernel_backend",
]
