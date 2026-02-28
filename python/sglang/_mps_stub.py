"""Stub implementations for APIs missing from ``torch.mps``.

``torch.mps`` lacks several APIs that ``torch.cuda`` provides (``Stream``,
``set_device``, ``get_device_properties``, …).  Rather than scattering
``hasattr`` / ``getattr`` guards throughout the codebase, we monkey-patch
``torch.mps`` once at startup so that generic device-agnostic code paths
just work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Stream – a no-op placeholder (MPS serialises everything on one stream)
# ---------------------------------------------------------------------------
class Stream:
    """Minimal stand-in for ``torch.cuda.Stream``.

    MPS does not expose user-visible streams.  Every method is a no-op so
    that code written for CUDA's multi-stream model still runs.
    """

    def __init__(self, device: Any = None, priority: int = 0) -> None:
        pass

    # -- public API expected by the codebase ---------------------------------
    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        return None

    def query(self) -> bool:
        return True

    # context-manager protocol (``with stream:``)
    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# set_device – MPS has exactly one device; this is a no-op.
# ---------------------------------------------------------------------------
def set_device(device: Any) -> None:  # noqa: ARG001
    pass


# ---------------------------------------------------------------------------
# current_device / device_count – trivial for single-device MPS
# ---------------------------------------------------------------------------
def current_device() -> int:
    return 0


def device_count() -> int:
    return 1


# ---------------------------------------------------------------------------
# get_device_properties – returns a lightweight dataclass
# ---------------------------------------------------------------------------
@dataclass
class _MPSDeviceProperties:
    """Mimics the object returned by ``torch.cuda.get_device_properties``."""

    name: str = "Apple MPS"
    total_memory: int = 0  # populated at install time
    multi_processor_count: int = 0
    warp_size: int = 32
    is_integrated: bool = True
    major: int = 0
    minor: int = 0
    # Extra attrs some callers inspect
    _extra: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        # Return a safe default for any attribute we didn't anticipate
        try:
            return self._extra[name]
        except KeyError:
            return None


_cached_props: _MPSDeviceProperties | None = None


def get_device_properties(device: Any = 0) -> _MPSDeviceProperties:  # noqa: ARG001
    global _cached_props
    if _cached_props is None:
        import psutil

        _cached_props = _MPSDeviceProperties(
            total_memory=psutil.virtual_memory().total,
        )
    return _cached_props


# ---------------------------------------------------------------------------
# install – monkey-patch torch.mps
# ---------------------------------------------------------------------------
_installed = False


def install() -> None:
    """Patch ``torch.mps`` with the stubs above.  Safe to call multiple times."""
    global _installed
    if _installed:
        return

    import torch

    mps = torch.mps
    # Only patch attributes that are actually missing
    for name, obj in [
        ("Stream", Stream),
        ("set_device", set_device),
        ("current_device", current_device),
        ("device_count", device_count),
        ("get_device_properties", get_device_properties),
    ]:
        if not hasattr(mps, name):
            setattr(mps, name, obj)

    _installed = True
