"""Unit tests for the Torch/MPS/MLX profiling adapter.

The production execution path is a standard Torch ``ModelRunner`` with
optional MLX operator islands.  These tests deliberately exercise the adapter
with mocks, so the contract is checked on CPU CI as well as Apple Silicon:

* CPU-only requests remain ordinary Torch profiler requests.
* GPU requests keep the non-CUDA Torch activities (normally CPU) and add one
  Metal capture session.
* MPS uses one device-scoped MLX capture, which can cover both Torch MPS and
  MLX command queues; it does not drop the Torch CPU profiler.
* Metal traces are emitted as collision-safe ``.gputrace`` sidecars next to
  the historical Chrome trace.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.mps import profiler as profiler_module
from sglang.srt.hardware_backend.mps.profiler import (
    MetalCaptureProfiler,
    MetalTorchProfiler,
    apply_metal_profiler_patches,
)
from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mps_ci(est_time=5, suite="stage-a-unit-test-mps")


def _success_capture_factory(label: str = "MLX"):
    """Return a fake capture factory that leaves a trace package on disk."""

    stop = MagicMock()

    def start(trace_path: Path):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(b"fake-gputrace")
        return (
            MetalCaptureProfiler(
                label=label,
                trace_path=trace_path,
                stop_capture=stop,
                standalone=label == "MLX",
            ),
            ProfileReqOutput(success=True, message="Succeeded"),
        )

    return start, stop


class _ProfilerPatchTestCase(unittest.TestCase):
    def setUp(self):
        # A different test module (notably SchedulerProfilerManager) may have
        # installed the global patch before this test runs.  Temporarily use
        # the true implementation so each test can install one fresh wrapper,
        # then restore the exact prior object in tearDown.
        self._previous_profile = torch.profiler.profile
        torch.profiler.profile = getattr(
            self._previous_profile,
            "_sglang_original_profile",
            self._previous_profile,
        )

    def tearDown(self):
        torch.profiler.profile = self._previous_profile


class TestApplyMetalProfilerPatches(_ProfilerPatchTestCase):
    def test_cpu_only_request_keeps_original_profile(self):
        apply_metal_profiler_patches()
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )

        self.assertNotIsInstance(profile, MetalTorchProfiler)

    def test_mps_gpu_request_keeps_torch_cpu_profiler(self):
        apply_metal_profiler_patches()
        profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )

        self.assertIsInstance(profile, MetalTorchProfiler)
        self.assertIsNotNone(profile.torch_profiler)
        self.assertIs(
            profile.start_metal_capture.__func__,
            MetalCaptureProfiler.start_mlx.__func__,
        )

    def test_gpu_only_request_preserves_schedule_and_event_list_api(self):
        apply_metal_profiler_patches()
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        )

        self.assertIsInstance(profile, MetalTorchProfiler)
        self.assertIsNotNone(profile.torch_profiler)
        profile.torch_profiler.start()
        try:
            event_list = profile.key_averages()
        finally:
            profile.torch_profiler.stop()
        self.assertTrue(hasattr(event_list, "table"))

    def test_original_kwargs_are_forwarded_after_cuda_is_removed(self):
        base_profile = MagicMock(name="base_profile")
        base_profile._sglang_metal_patched = False
        torch.profiler.profile = base_profile

        apply_metal_profiler_patches()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        wrapped = torch.profiler.profile(
            activities=activities,
            with_stack=False,
            record_shapes=True,
        )

        self.assertIsInstance(wrapped, MetalTorchProfiler)
        base_profile.assert_called_once()
        call = base_profile.call_args
        self.assertEqual(
            call.kwargs["activities"], [torch.profiler.ProfilerActivity.CPU]
        )
        self.assertFalse(call.kwargs["with_stack"])
        self.assertTrue(call.kwargs["record_shapes"])

    def test_activity_mapping_keeps_non_cuda_entries(self):
        base_profile = MagicMock(name="base_profile")
        base_profile._sglang_metal_patched = False
        torch.profiler.profile = base_profile

        apply_metal_profiler_patches()
        activities = [
            {
                torch.profiler.ProfilerActivity.CPU: {"detail": "cpu"},
                torch.profiler.ProfilerActivity.CUDA: {"detail": "gpu"},
            }
        ]
        torch.profiler.profile(activities=activities)

        self.assertEqual(
            base_profile.call_args.kwargs["activities"],
            [{torch.profiler.ProfilerActivity.CPU: {"detail": "cpu"}}],
        )


class TestMetalCaptureProfiler(unittest.TestCase):
    def test_start_mlx_imports_lazily_and_reports_success(self):
        fake_metal = types.ModuleType("mlx.core.metal")
        fake_metal.start_capture = MagicMock()
        fake_metal.stop_capture = MagicMock()
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = fake_metal
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core

        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                capture, result = MetalCaptureProfiler.start_mlx(
                    Path(tmp) / "capture.gputrace"
                )

        self.assertTrue(result.success)
        self.assertEqual(capture.label, "MLX")
        fake_metal.start_capture.assert_called_once()

    def test_start_mlx_failure_is_a_profile_result(self):
        fake_metal = types.ModuleType("mlx.core.metal")
        fake_metal.start_capture = MagicMock(
            side_effect=RuntimeError("Capture layer is not inserted")
        )
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = fake_metal
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core

        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                capture, result = MetalCaptureProfiler.start_mlx(
                    Path(tmp) / "capture.gputrace"
                )

        self.assertIsNone(capture)
        self.assertFalse(result.success)
        self.assertIn("MTL_CAPTURE_ENABLED", result.message)

    def test_start_mps_wraps_context_manager(self):
        context = MagicMock()
        context.__enter__.return_value = context
        metal_capture = MagicMock(return_value=context)

        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "torch-output"
            capture_dir.mkdir()
            target_dir = Path(tmp) / "requested-output"
            target_dir.mkdir()

            def finish_capture(*args):
                package = capture_dir / "0007-capture.gputrace"
                package.mkdir()
                (package / "trace.bin").write_bytes(b"capture")
                return False

            context.__exit__.side_effect = finish_capture
            with (
                patch.object(
                    torch.mps.profiler,
                    "metal_capture",
                    metal_capture,
                    create=True,
                ),
                patch.object(profiler_module.Path, "cwd", return_value=capture_dir),
            ):
                capture, result = MetalCaptureProfiler.start_mps(
                    target_dir / "capture.gputrace"
                )
                self.assertTrue(result.success)
                capture.stop()
                self.assertTrue(
                    (target_dir / "capture.gputrace" / "trace.bin").exists()
                )

        metal_capture.assert_called_once_with("capture")
        context.__enter__.assert_called_once()
        context.__exit__.assert_called_once_with(None, None, None)

    def test_stop_is_idempotent(self):
        stop = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            capture = MetalCaptureProfiler(
                label="MLX",
                trace_path=Path(tmp) / "capture.gputrace",
                stop_capture=stop,
                standalone=True,
            )
            capture.stop()
            capture.stop()

        stop.assert_called_once()

    def test_failed_capture_stop_is_not_retried(self):
        stop = MagicMock(side_effect=RuntimeError("stop failed"))
        with tempfile.TemporaryDirectory() as tmp:
            capture = MetalCaptureProfiler(
                label="MPS",
                trace_path=Path(tmp) / "capture.gputrace",
                stop_capture=stop,
                standalone=False,
            )
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                capture.stop()
            capture.stop()

        stop.assert_called_once()


class TestMetalTorchProfiler(unittest.TestCase):
    def test_mixed_start_starts_torch_and_metal(self):
        start_capture, stop_capture = _success_capture_factory()
        torch_profiler = MagicMock()
        profiler = MetalTorchProfiler(
            start_metal_capture=start_capture,
            torch_profiler=torch_profiler,
        )

        profiler.start()
        profiler.stop()

        torch_profiler.start.assert_called_once()
        torch_profiler.stop.assert_called_once()
        stop_capture.assert_called_once()

    def test_torch_start_failure_stops_metal_capture(self):
        start_capture, stop_capture = _success_capture_factory()
        torch_profiler = MagicMock()
        torch_profiler.start.side_effect = RuntimeError("torch profiler failed")
        profiler = MetalTorchProfiler(
            start_metal_capture=start_capture,
            torch_profiler=torch_profiler,
        )

        with self.assertRaisesRegex(RuntimeError, "torch profiler failed"):
            profiler.start()

        stop_capture.assert_called_once()
        self.assertIsNone(profiler.metal_profiler)

    def test_failed_stop_marks_wrapper_inactive(self):
        start_capture, stop_capture = _success_capture_factory()
        stop_capture.side_effect = RuntimeError("metal stop failed")
        profiler = MetalTorchProfiler(start_metal_capture=start_capture)

        profiler.start()
        with self.assertRaisesRegex(RuntimeError, "metal stop failed"):
            profiler.stop()
        profiler.stop()

        self.assertFalse(profiler._started)
        stop_capture.assert_called_once()

    def test_export_writes_chrome_and_unique_gputrace_sidecar(self):
        start_capture, _ = _success_capture_factory()
        torch_profiler = MagicMock()
        torch_profiler.export_chrome_trace.side_effect = lambda path: Path(
            path
        ).write_text("{}")
        profiler = MetalTorchProfiler(
            start_metal_capture=start_capture,
            torch_profiler=torch_profiler,
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "profile.trace.json.gz"
            profiler.start()
            profiler.stop()
            sidecar = profiler.export_chrome_trace(str(trace))

            self.assertTrue(trace.exists())
            self.assertIsNotNone(sidecar)
            self.assertTrue(sidecar.exists())
            self.assertEqual(sidecar.suffix, ".gputrace")
            torch_profiler.export_chrome_trace.assert_called_once_with(str(trace))

    def test_gpu_only_export_writes_empty_chrome_trace(self):
        start_capture, _ = _success_capture_factory()
        profiler = MetalTorchProfiler(start_metal_capture=start_capture)

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "profile.trace.json.gz"
            profiler.start()
            profiler.stop()
            profiler.export_chrome_trace(str(trace))

            self.assertTrue(trace.exists())

    def test_wrapper_supports_context_manager_and_step_contract(self):
        start_capture, _ = _success_capture_factory()
        torch_profiler = MagicMock()
        profiler = MetalTorchProfiler(
            start_metal_capture=start_capture,
            torch_profiler=torch_profiler,
        )

        with profiler as active:
            self.assertIs(active, profiler)
            profiler.step()
            profiler.key_averages(group_by_input_shape=True)

        torch_profiler.start.assert_called_once()
        torch_profiler.step.assert_called_once()
        torch_profiler.key_averages.assert_called_once_with(group_by_input_shape=True)
        torch_profiler.stop.assert_called_once()


class TestSchedulerProfilerCleanup(unittest.TestCase):
    def test_stop_or_export_failure_clears_scheduler_profile_state(self):
        from sglang.srt.managers.scheduler_components.profiler_manager import (
            SchedulerProfilerManager,
        )

        ps = types.SimpleNamespace(
            tp_rank=0,
            dp_size=1,
            dp_rank=0,
            pp_size=1,
            pp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
        )
        for failure in ("stop", "export"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                with patch(
                    "sglang.srt.managers.scheduler_components.profiler_manager.envs.SGLANG_PROFILE_V2.get",
                    return_value=False,
                ):
                    manager = SchedulerProfilerManager(
                        ps=ps,
                        dp_tp_cpu_group=None,
                        get_forward_ct=lambda: 0,
                    )
                manager.profile_in_progress = True
                manager.profiler_start_forward_ct = 1
                manager.torch_profiler_output_dir = Path(tmp)
                manager.profile_prefix = ""
                manager.profile_id = "cleanup"
                manager.profiler_activities = ["GPU"]
                manager.torch_profiler = MagicMock()
                failure_method = (
                    "export_chrome_trace" if failure == "export" else "stop"
                )
                getattr(manager.torch_profiler, failure_method).side_effect = (
                    RuntimeError(f"{failure} failed")
                )

                with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                    manager._stop_profile()

                self.assertFalse(manager.profile_in_progress)
                self.assertIsNone(manager.torch_profiler)
                self.assertIsNone(manager.profiler_start_forward_ct)


if __name__ == "__main__":
    unittest.main()
