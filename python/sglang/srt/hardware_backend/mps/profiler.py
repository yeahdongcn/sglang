from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from sglang.srt.managers.io_struct import ProfileReqOutput

logger = logging.getLogger(__name__)


class _EmptyEventList(list):
    """Small EventList-compatible fallback for a Metal-only capture."""

    def table(self, *args, **kwargs):
        return ""


@dataclass
class MetalCaptureProfiler:
    label: str
    trace_path: Path
    stop_capture: Callable[[], None]
    standalone: bool
    _stopped: bool = False

    @classmethod
    def start_mlx(cls, trace_path: Path):
        """Start a process-level Metal capture for the mixed MPS/MLX path.

        MLX's capture descriptor targets the Metal device rather than an MLX
        command queue.  That is important for the current backend: Torch owns
        the model and normally submits most work through MPS, while selected
        operator islands submit work through MLX.  Starting the MLX capture
        here therefore captures both kinds of command buffers without trying
        to nest two mutually-exclusive ``MTLCaptureManager`` sessions.
        """
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx

            mx.metal.start_capture(str(trace_path))
        except Exception as e:
            return None, _capture_error("MLX", e)

        return cls._started(
            label="MLX",
            trace_path=trace_path,
            stop_capture=mx.metal.stop_capture,
            standalone=True,
        )

    @classmethod
    def start_mps(cls, trace_path: Path):
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Torch 2.13 treats this argument as a basename, prepends a four-digit
        # capture sequence, and appends ``.gputrace`` itself.  Passing an
        # absolute path therefore produces an invalid name such as
        # ``0000-/tmp/...gputrace.gputrace``.  Start with a collision-safe
        # basename in the process working directory, then move the generated
        # package to our requested path when capture stops.
        capture_dir = Path.cwd()
        capture_basename = trace_path.stem
        existing = set(_mps_capture_candidates(capture_dir, capture_basename))

        try:
            if not hasattr(torch, "mps") or not hasattr(torch.mps, "profiler"):
                raise RuntimeError("torch.mps.profiler is not available")
            context = torch.mps.profiler.metal_capture(capture_basename)
            context.__enter__()
        except Exception as e:
            return None, _capture_error("MPS", e)

        def stop_capture() -> None:
            context.__exit__(None, None, None)
            created = [
                path
                for path in _mps_capture_candidates(capture_dir, capture_basename)
                if path not in existing
            ]
            if len(created) != 1:
                raise RuntimeError(
                    "Torch MPS capture stopped but its generated .gputrace "
                    f"package could not be identified in {capture_dir}: {created}"
                )
            shutil.move(str(created[0]), str(trace_path))

        return cls._started(
            label="MPS",
            trace_path=trace_path,
            stop_capture=stop_capture,
            standalone=False,
        )

    @classmethod
    def _started(
        cls,
        *,
        label: str,
        trace_path: Path,
        stop_capture: Callable[[], None],
        standalone: bool,
    ):
        profiler = cls(
            label=label,
            trace_path=trace_path,
            stop_capture=stop_capture,
            standalone=standalone,
        )
        logger.info("%s Metal capture started, saving to %s", label, trace_path)
        return profiler, ProfileReqOutput(success=True, message="Succeeded")

    def stop(self) -> str:
        if not self._stopped:
            self._stopped = True
            # Mark the native capture as consumed before calling into the
            # platform API.  If stopping or moving the package raises, a
            # retry must not invoke ``MTLCaptureManager`` a second time.
            self.stop_capture()

        logger.info(
            "%s Metal capture stopped. Trace saved to: %s",
            self.label,
            self.trace_path,
        )
        return f" Metal trace: {self.trace_path}"


def _capture_error(label: str, error: Exception) -> ProfileReqOutput:
    return ProfileReqOutput(
        success=False,
        message=(
            f"Failed to start {label} Metal capture: {error}. "
            "Set MTL_CAPTURE_ENABLED=1 in the server's environment "
            "before launching to enable GPU trace capture."
        ),
    )


class MetalTorchProfiler:
    def __init__(
        self,
        *,
        start_metal_capture: Callable[[Path], tuple[Any, ProfileReqOutput]],
        torch_profiler: Optional[Any] = None,
    ):
        self.start_metal_capture = start_metal_capture
        self.torch_profiler = torch_profiler
        self.metal_profiler = None
        self.metal_trace_path: Optional[Path] = None
        self._started = False

    def start(self):
        if self._started:
            return self
        trace_path = _new_temp_gputrace_path()
        metal_profiler, result = self.start_metal_capture(trace_path)
        if not result.success:
            raise RuntimeError(result.message)
        self.metal_profiler = metal_profiler
        try:
            if self.torch_profiler is not None:
                self.torch_profiler.start()
        except Exception:
            try:
                self.metal_profiler.stop()
            except Exception:
                logger.exception(
                    "Failed to stop Metal capture after Torch profiler startup failed"
                )
            finally:
                self.metal_profiler = None
            raise
        self._started = True
        return self

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def stop(self):
        if not self._started:
            return
        try:
            if self.torch_profiler is not None:
                self.torch_profiler.stop()
        finally:
            try:
                if self.metal_profiler is not None:
                    self.metal_trace_path = self.metal_profiler.trace_path
                    self.metal_profiler.stop()
            finally:
                # Allow callers to recover from a failed stop without
                # accidentally issuing a second native stop on retry.
                self._started = False

    def step(self):
        """Preserve the profiler interface for schedule-driven callers.

        A Metal capture has no Torch Kineto step state when the caller asked
        for GPU-only activities.  In mixed CPU/GPU mode, forward the step to
        the retained Torch CPU profiler.
        """
        if self.torch_profiler is not None:
            return self.torch_profiler.step()
        return None

    def key_averages(self, *args, **kwargs):
        """Expose Torch summaries when available; Metal traces are sidecars."""
        if self.torch_profiler is not None:
            return self.torch_profiler.key_averages(*args, **kwargs)
        return _EmptyEventList()

    def events(self):
        if self.torch_profiler is not None:
            return self.torch_profiler.events()
        return []

    def __getattr__(self, name):
        # Keep less common read-only profiler attributes available to callers
        # in mixed CPU/GPU mode without making the wrapper own Torch state.
        torch_profiler = self.__dict__.get("torch_profiler")
        if torch_profiler is not None:
            return getattr(torch_profiler, name)
        raise AttributeError(name)

    def export_chrome_trace(self, path: str):
        """Export the Torch trace and retain the Metal trace as a sidecar.

        Chrome trace files cannot embed Apple's ``.gputrace`` package.  Keep
        the historical Torch output path for API compatibility, and move the
        Metal package beside it with a collision-safe name.  The returned path
        lets callers surface the sidecar to users (the server profiler manager
        may also use ``metal_trace_path`` after export).
        """
        path_obj = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if self.torch_profiler is not None:
            self.torch_profiler.export_chrome_trace(str(path_obj))
        else:
            _write_empty_chrome_trace(str(path_obj))

        if self.metal_profiler is None:
            return None

        final_path = _unique_gputrace_path_for_chrome_trace(str(path_obj))
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metal_profiler.trace_path.exists():
            shutil.move(str(self.metal_profiler.trace_path), str(final_path))
            self.metal_trace_path = final_path
            logger.info("Metal trace saved to: %s", final_path)
            return final_path
        return None


def apply_metal_profiler_patches() -> None:
    if getattr(torch.profiler.profile, "_sglang_metal_patched", False):
        return

    original_profile = torch.profiler.profile

    def profile(*args, **kwargs):
        activities = _get_activities(args, kwargs)
        if activities is not None and not isinstance(activities, (list, tuple)):
            activities = list(activities)
            args, kwargs = _replace_activities(args, kwargs, activities)
        if not _has_cuda_activity(activities):
            return original_profile(*args, **kwargs)

        torch_activities = []
        for activity in activities:
            activity_without_cuda = _without_cuda_activity(activity)
            if activity_without_cuda is not None:
                torch_activities.append(activity_without_cuda)
        # Keep a CPU Kineto object even for a GPU-only request.  Metal captures
        # have no EventList of their own, while this preserves schedule,
        # on_trace_ready, step(), and summary APIs used by generic callers.
        if not torch_activities:
            torch_activities = [torch.profiler.ProfilerActivity.CPU]
        patched_args, patched_kwargs = _replace_activities(
            args, kwargs, torch_activities
        )
        torch_profiler = original_profile(*patched_args, **patched_kwargs)

        return MetalTorchProfiler(
            # MLX captures the process-level Metal device, covering both the
            # Torch MPS queue and automatically selected MLX custom ops in the
            # single MPS backend without trying to nest capture sessions.
            start_metal_capture=MetalCaptureProfiler.start_mlx,
            torch_profiler=torch_profiler,
        )

    profile._sglang_metal_patched = True
    profile._sglang_original_profile = original_profile
    torch.profiler.profile = profile


def _get_activities(args, kwargs):
    if "activities" in kwargs:
        return kwargs["activities"]
    if args:
        return args[0]
    return None


def _replace_activities(args, kwargs, activities):
    kwargs = dict(kwargs)
    if "activities" in kwargs:
        kwargs["activities"] = activities
        return args, kwargs

    if args:
        args = list(args)
        args[0] = activities
        return tuple(args), kwargs

    kwargs["activities"] = activities
    return args, kwargs


def _has_cuda_activity(activities) -> bool:
    if activities is None:
        return False
    return any(_is_cuda_activity(activity) for activity in activities)


def _is_cuda_activity(activity) -> bool:
    if activity == torch.profiler.ProfilerActivity.CUDA:
        return True
    if isinstance(activity, dict):
        return any(_is_cuda_activity(key) for key in activity)
    return False


def _without_cuda_activity(activity):
    """Remove CUDA entries while preserving other activity metadata."""
    if isinstance(activity, dict):
        filtered = {
            key: value for key, value in activity.items() if not _is_cuda_activity(key)
        }
        return filtered or None
    return None if _is_cuda_activity(activity) else activity


def _new_temp_gputrace_path() -> Path:
    output_dir = Path(os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        candidate = (
            output_dir / f"sglang-metal-{os.getpid()}-{time.time_ns()}-{i}.gputrace"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an unused Metal trace path in {output_dir}")


def _mps_capture_candidates(directory: Path, basename: str) -> list[Path]:
    """Return Torch 2.13 MPS capture packages for one unique basename."""
    candidates = []
    suffix = f"-{basename}.gputrace"
    for path in directory.iterdir():
        if not path.name.endswith(suffix):
            continue
        prefix = path.name.removesuffix(suffix)
        if prefix.isdigit():
            candidates.append(path)
    return candidates


def _unique_gputrace_path_for_chrome_trace(path: str) -> Path:
    chrome_path = Path(path).expanduser()
    name = chrome_path.name
    if name.endswith(".trace.json.gz"):
        name = name[: -len(".trace.json.gz")] + ".gputrace"
    else:
        name = chrome_path.stem + ".gputrace"

    base = chrome_path.with_name(name)
    if not base.exists():
        return base

    stem = base.name[: -len(".gputrace")]
    for i in range(100):
        candidate = base.with_name(f"{stem}-{time.time_ns()}-{i}.gputrace")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an unused Metal trace path for {base}")


def _write_empty_chrome_trace(path: str):
    trace = {"traceEvents": []}
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
    else:
        with open(path, "w") as f:
            json.dump(trace, f)
