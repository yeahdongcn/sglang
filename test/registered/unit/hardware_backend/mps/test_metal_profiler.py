"""Unit tests for the Torch MPS Metal profiling adapter."""

from __future__ import annotations

import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=5, suite="stage-a-unit-test-mps")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_SKIP_REASON = "requires Apple Silicon"


@unittest.skipUnless(_IS_APPLE_SILICON, _SKIP_REASON)
class TestApplyMetalProfilerPatches(unittest.TestCase):
    def setUp(self):
        self._original_profile = getattr(
            torch.profiler.profile, "_sglang_original_profile", None
        )

    def tearDown(self):
        if self._original_profile is not None:
            torch.profiler.profile = self._original_profile

    def test_patch_replaces_profile(self):
        from sglang.srt.hardware_backend.mps.profiler import (
            MetalTorchProfiler,
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        self.assertTrue(getattr(torch.profiler.profile, "_sglang_metal_patched", False))
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        )
        self.assertIsInstance(profiler, MetalTorchProfiler)

    def test_patch_is_idempotent(self):
        from sglang.srt.hardware_backend.mps.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        first = torch.profiler.profile
        apply_metal_profiler_patches()
        self.assertIs(torch.profiler.profile, first)

    def test_no_cuda_activity_uses_original(self):
        from sglang.srt.hardware_backend.mps.profiler import (
            MetalTorchProfiler,
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )
        self.assertNotIsInstance(profiler, MetalTorchProfiler)


@unittest.skipUnless(_IS_APPLE_SILICON, _SKIP_REASON)
class TestMetalCaptureProfilerMPS(unittest.TestCase):
    def test_start_mps_success(self):
        from sglang.srt.hardware_backend.mps.profiler import MetalCaptureProfiler

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(
                torch.mps.profiler, "metal_capture", return_value=mock_context
            ):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        self.assertTrue(result.success)
        self.assertIsNotNone(profiler)
        self.assertEqual(profiler.label, "MPS")
        self.assertFalse(profiler.standalone)

    def test_start_mps_runtime_error_returns_failure(self):
        from sglang.srt.hardware_backend.mps.profiler import MetalCaptureProfiler

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(
                torch.mps.profiler,
                "metal_capture",
                side_effect=RuntimeError("MPS profiler unavailable"),
            ):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        self.assertIsNone(profiler)
        self.assertFalse(result.success)
        self.assertIn("MTL_CAPTURE_ENABLED", result.message)


@unittest.skipUnless(_IS_APPLE_SILICON, _SKIP_REASON)
class TestSchedulerProfilerManagerMPS(unittest.TestCase):
    def _make_manager(self, output_dir):
        from sglang.srt.managers.scheduler_components.profiler_manager import (
            SchedulerProfilerManager,
        )

        class FakePS:
            tp_rank = dp_rank = pp_rank = moe_ep_rank = 0
            dp_size = pp_size = moe_ep_size = 1
            gpu_id = 0

        manager = SchedulerProfilerManager(
            ps=FakePS(), dp_tp_cpu_group=None, get_forward_ct=lambda: 0
        )
        manager._init_profile(output_dir, None, None, None, None, None, False, "test")
        return manager

    def test_start_profile_failure_does_not_crash(self):
        from sglang.srt.hardware_backend.mps.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._make_manager(tmp)
            with patch.object(
                torch.mps.profiler,
                "metal_capture",
                side_effect=RuntimeError("Capture layer is not inserted"),
            ):
                result = manager._start_profile()

        self.assertFalse(result.success)
        self.assertIn("Capture layer is not inserted", result.message)
        self.assertFalse(manager.profile_in_progress)
        self.assertIsNone(manager.torch_profiler)

    def test_start_profile_success_with_mock_capture(self):
        from sglang.srt.hardware_backend.mps.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._make_manager(tmp)
            capture_context = MagicMock()
            with patch.object(
                torch.mps.profiler,
                "metal_capture",
                return_value=capture_context,
            ), patch("torch.distributed.barrier"):
                result = manager._start_profile()
                self.assertTrue(result.success, result.message)
                self.assertTrue(manager.profile_in_progress)
                capture_context.__enter__.assert_called_once()
                manager._stop_profile()
                self.assertFalse(manager.profile_in_progress)
                capture_context.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
