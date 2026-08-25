"""Optional, context-local phase timing for benchmark diagnostics.

The serving path does not enable a recorder.  In that normal case callers only
pay for a cheap ``ContextVar.get`` at the boundary where they may be timed;
they do not call ``perf_counter`` or allocate timing events.  The benchmark
installs a recorder around one forward call and removes it on scope exit.

This module is intentionally private.  It is a small instrumentation seam,
not a public timing API or an environment-controlled serving feature.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, TypeVar

PhaseRecorder = Callable[[str, float], None]
_T = TypeVar("_T")

_CURRENT_PHASE_RECORDER: ContextVar[PhaseRecorder | None] = ContextVar(
    "sglang_mlx_phase_recorder", default=None
)


@contextmanager
def phase_recorder(recorder: PhaseRecorder | None) -> Iterator[None]:
    """Temporarily route measured phase durations to ``recorder``.

    The context is deliberately nestable and context-local.  A ``None``
    recorder is a no-op, which lets benchmark call sites share one wrapper
    without changing production behavior.
    """

    if recorder is None:
        yield
        return
    if not callable(recorder):
        raise TypeError("phase recorder must be callable or None")
    token = _CURRENT_PHASE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _CURRENT_PHASE_RECORDER.reset(token)


def current_phase_recorder() -> PhaseRecorder | None:
    """Return the recorder active in the current context, if any."""

    return _CURRENT_PHASE_RECORDER.get()


def measure_phase(
    recorder: PhaseRecorder | None,
    name: str,
    operation: Callable[[], _T],
) -> _T:
    """Run ``operation`` and report its host duration when recording.

    Exceptions retain their original semantics.  The duration is reported in
    a ``finally`` block so a benchmark can diagnose a failed phase without
    changing the exception that caused the failure.
    """

    if recorder is None:
        return operation()
    started = time.perf_counter()
    try:
        return operation()
    finally:
        # Diagnostics must never change the serving/benchmark operation's
        # exception semantics.  The built-in recorder is intentionally tiny,
        # but this guard also protects future artifact sinks.
        try:
            recorder(name, max(0.0, time.perf_counter() - started))
        except Exception:
            pass


__all__ = ["PhaseRecorder", "current_phase_recorder", "measure_phase", "phase_recorder"]
