"""Lazy, operator-agnostic helpers for Torch MPS Metal libraries.

This module owns the host-runtime boundary shared by Metal kernels.  It does
not know about model families, tensor layouts, or kernel entry-point names:
semantic operator modules keep those contracts and call these helpers only
when their selected backend is Metal.

Torch runtime lookups stay inside the helper calls; importing this module never
compiles Metal source, loads a ``.metallib``, or synchronizes MPS.  Callers
control JIT/AOT timing explicitly by calling :func:`compile_metal_library`,
:func:`load_metal_library`, and :func:`resolve_metal_entry_points` during their
own warmup phase.
"""

from __future__ import annotations

import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Tuple


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _torch_mps_function(name: str):
    if not _is_apple_silicon():
        raise RuntimeError("Metal kernels require macOS on Apple Silicon")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Metal kernels require a Torch build with MPS support"
        ) from exc

    function = getattr(torch.mps, name, None)
    if not callable(function):
        raise RuntimeError(f"Metal kernels require torch.mps.{name}")
    return function


def is_metal_jit_available() -> bool:
    """Whether the current process exposes Torch's inline Metal compiler."""
    if not _is_apple_silicon():
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(
        torch.backends.mps.is_available()
        and callable(getattr(torch.mps, "compile_shader", None))
    )


def is_metal_aot_available(path: str | Path) -> bool:
    """Whether ``path`` exists and Torch can load Metal libraries here."""
    if not _is_apple_silicon() or not Path(path).is_file():
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(
        torch.backends.mps.is_available()
        and callable(getattr(torch.mps, "load_metallib", None))
    )


@lru_cache(maxsize=None)
def compile_metal_library(source: str) -> Any:
    """Compile one MSL source string once and return its shader library.

    Compilation is lazy and cached by the complete source string.  Merely
    importing a semantic operator therefore cannot trigger JIT compilation;
    a runtime may call this function during explicit warmup instead of paying
    the cost on the first request.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Metal JIT source must be a non-empty string")
    return _torch_mps_function("compile_shader")(source)


@lru_cache(maxsize=None)
def _load_metal_library(path: str) -> Any:
    return _torch_mps_function("load_metallib")(Path(path))


def load_metal_library(path: str | Path) -> Any:
    """Load one precompiled ``.metallib`` once and return its shader library."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Metal library is missing: {resolved}")
    return _load_metal_library(str(resolved))


def resolve_metal_entry_points(
    library: Any, entry_points: Iterable[str]
) -> Tuple[Any, ...]:
    """Resolve named pipelines without assuming any operator-specific names.

    Torch creates the callable pipeline when the attribute is resolved.  This
    helper lets a provider make that cost part of an explicit warmup while
    keeping the common substrate independent of model and operator contracts.
    """
    resolved = []
    for name in entry_points:
        if not isinstance(name, str) or not name:
            raise ValueError("Metal entry-point names must be non-empty strings")
        resolved.append(getattr(library, name))
    return tuple(resolved)


def clear_metal_library_caches() -> None:
    """Forget JIT/AOT libraries, primarily for tests and controlled reloads."""
    compile_metal_library.cache_clear()
    _load_metal_library.cache_clear()


__all__ = [
    "clear_metal_library_caches",
    "compile_metal_library",
    "is_metal_aot_available",
    "is_metal_jit_available",
    "load_metal_library",
    "resolve_metal_entry_points",
]
