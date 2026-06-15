import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components import profiler_manager as pm
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


class TestSchedulerProfilerManager(unittest.TestCase):
    def _manager(self, output_dir, ps=None):
        manager = SchedulerProfilerManager(
            ps=ps
            or SimpleNamespace(
                tp_rank=0,
                dp_rank=0,
                pp_rank=0,
                moe_ep_rank=0,
                dp_size=1,
                pp_size=1,
                moe_ep_size=1,
                gpu_id=0,
            ),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )
        manager._init_profile(
            output_dir=output_dir,
            start_step=None,
            num_steps=None,
            activities=["CPU", "GPU"],
            with_stack=False,
            record_shapes=False,
            profile_by_stage=False,
            profile_id="profile",
            profile_prefix="pref",
        )
        return manager

    def test_start_profile_uses_torch_profiler_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(tmpdir)
            fake_profiler = _FakeTorchProfiler()

            with patch.object(
                pm.torch.profiler, "profile", return_value=fake_profiler
            ) as mock_profile:
                result = manager._start_profile()

            self.assertTrue(result.success)
            self.assertTrue(manager.profile_in_progress)
            self.assertTrue(fake_profiler.started)
            mock_profile.assert_called_once()

    def test_stop_profile_exports_rank_and_stage_trace_path(self):
        ps = SimpleNamespace(
            tp_rank=1,
            dp_rank=2,
            pp_rank=3,
            moe_ep_rank=4,
            dp_size=4,
            pp_size=4,
            moe_ep_size=8,
            gpu_id=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(tmpdir, ps=ps)
            fake_profiler = _FakeTorchProfiler()
            manager.torch_profiler = fake_profiler
            manager.profile_in_progress = True

            with patch.object(pm.torch.distributed, "barrier") as mock_barrier:
                result = manager._stop_profile(ForwardMode.DECODE)

            self.assertTrue(result.success)
            self.assertFalse(manager.profile_in_progress)
            self.assertTrue(fake_profiler.stopped)
            self.assertTrue(
                fake_profiler.exported.endswith(
                    "pref-profile-TP-1-DP-2-PP-3-EP-4-DECODE.trace.json.gz"
                )
            )
            mock_barrier.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
