import gzip
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.mlx import profiler as metal_profiler
from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler
from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeMlxMetal:
    def __init__(self):
        self.started_path = None
        self.stopped = False

    def start_capture(self, path):
        self.started_path = path

    def stop_capture(self):
        self.stopped = True


class _FakeMpsCaptureContext:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True


class _FakeTorchProfiler:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.exported = None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        self.exported = path
        Path(path).write_text("{}")


class _FakeMetalProfiler:
    def __init__(self, trace_path):
        self.trace_path = trace_path

    def stop(self):
        self.trace_path.mkdir(parents=True, exist_ok=False)
        return f" Metal trace: {self.trace_path}"


class TestMetalCaptureProfiler(unittest.TestCase):
    def test_mlx_capture_uses_mlx_metal_api(self):
        fake_metal = _FakeMlxMetal()
        mlx_module = types.ModuleType("mlx")
        mlx_core_module = types.ModuleType("mlx.core")
        mlx_core_module.metal = fake_metal
        mlx_module.core = mlx_core_module

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "profile.gputrace"
            with patch.dict(
                sys.modules, {"mlx": mlx_module, "mlx.core": mlx_core_module}
            ):
                profiler, result = MetalCaptureProfiler.start_mlx(trace_path)
                message = profiler.stop()

        self.assertTrue(result.success)
        self.assertEqual(profiler.label, "MLX")
        self.assertTrue(profiler.standalone)
        self.assertEqual(fake_metal.started_path, str(trace_path))
        self.assertTrue(fake_metal.stopped)
        self.assertEqual(message, f" Metal trace: {trace_path}")

    def test_mps_capture_uses_torch_mps_profiler_context(self):
        context = _FakeMpsCaptureContext()
        fake_torch_mps = SimpleNamespace(
            profiler=SimpleNamespace(metal_capture=lambda path: context)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "profile.gputrace"
            with patch.object(metal_profiler.torch, "mps", fake_torch_mps, create=True):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)
                message = profiler.stop()

        self.assertTrue(result.success)
        self.assertEqual(profiler.label, "MPS")
        self.assertFalse(profiler.standalone)
        self.assertTrue(context.entered)
        self.assertTrue(context.exited)
        self.assertEqual(message, f" Metal trace: {trace_path}")

    def test_capture_start_failure_returns_actionable_error(self):
        def fail_start_capture(path):
            raise RuntimeError("Capture layer is not inserted")

        fake_torch_mps = SimpleNamespace(
            profiler=SimpleNamespace(metal_capture=fail_start_capture)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "profile.gputrace"
            with patch.object(metal_profiler.torch, "mps", fake_torch_mps, create=True):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        self.assertIsNone(profiler)
        self.assertFalse(result.success)
        self.assertIn("MTL_CAPTURE_ENABLED=1", result.message)

    def test_patch_routes_mlx_gpu_profile_to_metal_capture_only(self):
        original_called = False

        def original_profile(*args, **kwargs):
            nonlocal original_called
            original_called = True
            return _FakeTorchProfiler()

        def start_mlx(trace_path):
            return _FakeMetalProfiler(trace_path), ProfileReqOutput(
                success=True, message="Succeeded"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.trace.json.gz"
            with (
                patch.dict(os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmpdir}),
                patch.object(metal_profiler, "use_mlx", return_value=True),
                patch.object(MetalCaptureProfiler, "start_mlx", side_effect=start_mlx),
                patch.object(
                    metal_profiler.torch.profiler, "profile", original_profile
                ),
            ):
                metal_profiler.apply_metal_profiler_patches()
                profiler = metal_profiler.torch.profiler.profile(
                    activities=[
                        metal_profiler.torch.profiler.ProfilerActivity.CPU,
                        metal_profiler.torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                profiler.start()
                profiler.stop()
                profiler.export_chrome_trace(str(output_path))

            self.assertFalse(original_called)
            self.assertTrue(output_path.exists())
            with gzip.open(output_path, "rt") as f:
                self.assertEqual(json.load(f), {"traceEvents": []})
            self.assertTrue((Path(tmpdir) / "profile.gputrace").exists())

    def test_patch_keeps_torch_cpu_profile_for_mps_gpu_request(self):
        fake_torch_profiler = _FakeTorchProfiler()
        captured_activities = None

        def original_profile(*args, **kwargs):
            nonlocal captured_activities
            captured_activities = kwargs["activities"]
            return fake_torch_profiler

        def start_mps(trace_path):
            return _FakeMetalProfiler(trace_path), ProfileReqOutput(
                success=True, message="Succeeded"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.trace.json.gz"
            with (
                patch.dict(os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmpdir}),
                patch.object(metal_profiler, "use_mlx", return_value=False),
                patch.object(MetalCaptureProfiler, "start_mps", side_effect=start_mps),
                patch.object(
                    metal_profiler.torch.profiler, "profile", original_profile
                ),
            ):
                metal_profiler.apply_metal_profiler_patches()
                profiler = metal_profiler.torch.profiler.profile(
                    activities=[
                        metal_profiler.torch.profiler.ProfilerActivity.CPU,
                        metal_profiler.torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                profiler.start()
                profiler.stop()
                profiler.export_chrome_trace(str(output_path))

            self.assertEqual(
                captured_activities,
                [metal_profiler.torch.profiler.ProfilerActivity.CPU],
            )
            self.assertTrue(fake_torch_profiler.started)
            self.assertTrue(fake_torch_profiler.stopped)
            self.assertEqual(fake_torch_profiler.exported, str(output_path))
            self.assertTrue((Path(tmpdir) / "profile.gputrace").exists())


if __name__ == "__main__":
    unittest.main()
