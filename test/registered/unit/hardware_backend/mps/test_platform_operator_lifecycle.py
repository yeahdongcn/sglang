"""Backend-neutral lifecycle tests for the MPS semantic-operator plan."""

from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")


class _RecordingLock:
    def __init__(self, events):
        self.events = events
        self.active = False

    def __enter__(self):
        self.active = True
        self.events.append("enter")

    def __exit__(self, *args):
        self.active = False
        self.events.append("exit")


class TestPlatformOperatorLifecycle(unittest.TestCase):
    def test_model_runner_serializes_platform_operator_forward(self):
        events = []
        lock = _RecordingLock(events)
        expect_locked = True

        def forward_unlocked(*args):
            self.assertEqual(lock.active, expect_locked)
            events.append("forward")
            return "output"

        runner = types.SimpleNamespace(
            _platform_forward_lock=lock,
            _forward_raw_unlocked=forward_unlocked,
        )
        self.assertEqual(ModelRunner._forward_raw(runner, object(), None), "output")
        self.assertEqual(events, ["enter", "forward", "exit"])

        events.clear()
        expect_locked = False
        runner._platform_forward_lock = None
        self.assertEqual(ModelRunner._forward_raw(runner, object(), None), "output")
        self.assertEqual(events, ["forward"])

        events.clear()
        expect_locked = True
        runner._platform_forward_lock = lock
        runner._forward_split_prefill_unlocked = forward_unlocked
        self.assertEqual(ModelRunner.forward_split_prefill(runner, object()), "output")
        self.assertEqual(events, ["enter", "forward", "exit"])

    def test_model_runner_routes_post_pool_binding_through_platform(self):
        first_pool = object()
        second_pool = object()
        req_pool = object()
        platform = mock.MagicMock(device_type="mps")
        platform.bind_model_runtime_operators.side_effect = ["first", "second"]
        runner = types.SimpleNamespace(
            device="mps",
            model_config=object(),
            server_args=object(),
            req_to_token_pool=req_pool,
            token_to_kv_pool=first_pool,
        )
        model = nn.Module()

        with mock.patch(
            "sglang.srt.model_executor.model_runner.current_platform", platform
        ):
            self.assertEqual(
                ModelRunner._bind_platform_runtime_operators(runner, model), "first"
            )
            runner.token_to_kv_pool = second_pool
            self.assertEqual(
                ModelRunner._bind_platform_runtime_operators(runner, model), "second"
            )

        self.assertEqual(
            [
                call.kwargs["token_to_kv_pool"]
                for call in platform.bind_model_runtime_operators.call_args_list
            ],
            [first_pool, second_pool],
        )
        self.assertTrue(
            all(
                call.kwargs["req_to_token_pool"] is req_pool
                for call in platform.bind_model_runtime_operators.call_args_list
            )
        )

    def test_mps_memory_pool_allocation_is_explicitly_one_shot(self):
        runner = types.SimpleNamespace(
            device="mps",
            _platform_memory_pool_allocation_started=True,
        )
        platform = mock.MagicMock(device_type="mps")
        platform.supports_memory_pool_reallocation.return_value = False

        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.current_platform", platform
            ),
            self.assertRaisesRegex(RuntimeError, "one-shot"),
        ):
            ModelRunner.alloc_memory_pool(runner)

    def test_pool_one_shot_capability_ignores_mismatched_runner_device(self):
        runner = types.SimpleNamespace(
            device="cpu",
            _platform_memory_pool_allocation_started=True,
            init_kv_cache_configurator=mock.Mock(
                side_effect=ValueError("continued past platform guard")
            ),
        )
        platform = mock.MagicMock(device_type="mps")
        platform.supports_memory_pool_reallocation.return_value = False

        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.current_platform", platform
            ),
            self.assertRaisesRegex(ValueError, "continued past platform guard"),
        ):
            ModelRunner.alloc_memory_pool(runner)

    def test_model_runner_skips_binding_for_device_mismatch(self):
        platform = mock.MagicMock(device_type="mps")
        runner = types.SimpleNamespace(device="cpu")
        with mock.patch(
            "sglang.srt.model_executor.model_runner.current_platform", platform
        ):
            self.assertIsNone(
                ModelRunner._bind_platform_runtime_operators(runner, nn.Module())
            )
        platform.bind_model_runtime_operators.assert_not_called()

    def test_replacement_failure_keeps_model_with_safe_torch_fallback(self):
        old_model = nn.Module()
        old_plan = mock.MagicMock()
        runner = object.__new__(ModelRunner)
        runner.model = old_model
        runner.model_config = object()
        runner.server_args = types.SimpleNamespace(
            model_path="original",
            load_format="auto",
        )
        runner.is_draft_worker = False
        runner.platform_operator_plan = old_plan
        runner._platform_forward_lock = None
        runtime_context = mock.Mock()
        runtime_context.resolved_server_args_dict.return_value = {
            "model_path": "original",
            "load_format": "auto",
        }

        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_context",
                return_value=runtime_context,
            ),
            mock.patch.object(
                ModelRunner,
                "_bind_platform_runtime_operators",
                side_effect=RuntimeError("pool contract failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "pool contract failed"),
        ):
            ModelRunner.update_model_fields(
                runner,
                nn.Module(),
                model_path="replacement",
                load_format="auto",
                load_config=object(),
            )

        self.assertIs(runner.model, old_model)
        self.assertIsNone(runner.platform_operator_plan)
        old_plan.close.assert_called_once_with()
        self.assertEqual(
            runtime_context.override.call_args_list,
            [
                mock.call(
                    "model_runner.update_model_fields",
                    model_path="replacement",
                    load_format="auto",
                ),
                mock.call(
                    "model_runner.update_model_fields.rollback",
                    model_path="original",
                    load_format="auto",
                ),
            ],
        )

    def test_replacement_keeps_one_serving_lock_during_build_and_publication(self):
        old_model = nn.Module()
        new_model = nn.Module()
        serving_lock = threading.RLock()
        old_plan = mock.MagicMock(forward_lock=serving_lock)
        new_plan = mock.MagicMock(forward_lock=serving_lock)
        runner = object.__new__(ModelRunner)
        runner.model = old_model
        runner.model_config = object()
        runner.server_args = types.SimpleNamespace(
            model_path="original",
            load_format="auto",
        )
        runner.is_draft_worker = False
        runner.platform_operator_plan = old_plan
        runner._platform_forward_lock = serving_lock
        runtime_context = mock.Mock()
        runtime_context.resolved_server_args_dict.return_value = {
            "model_path": "original",
            "load_format": "auto",
        }

        def build_plan(model):
            self.assertIs(model, new_model)
            self.assertIs(runner._platform_forward_lock, serving_lock)
            self.assertIsNone(runner.platform_operator_plan)
            return new_plan

        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_context",
                return_value=runtime_context,
            ),
            mock.patch.object(
                ModelRunner,
                "_bind_platform_runtime_operators",
                side_effect=build_plan,
            ),
        ):
            ModelRunner.update_model_fields(
                runner,
                new_model,
                model_path="replacement",
                load_format="auto",
                load_config="new-load-config",
            )

        old_plan.close.assert_called_once_with()
        self.assertIs(runner.model, new_model)
        self.assertIs(runner.platform_operator_plan, new_plan)
        self.assertIs(runner._platform_forward_lock, serving_lock)
        self.assertEqual(runner.load_config, "new-load-config")

    def test_replacement_blocks_forward_until_the_new_plan_is_published(self):
        old_model = nn.Module()
        new_model = nn.Module()
        serving_lock = threading.RLock()
        old_plan = mock.MagicMock(forward_lock=serving_lock)
        new_plan = mock.MagicMock(forward_lock=serving_lock)
        runner = object.__new__(ModelRunner)
        runner.model = old_model
        runner.model_config = object()
        runner.server_args = types.SimpleNamespace(
            model_path="original",
            load_format="auto",
        )
        runner.is_draft_worker = False
        runner.platform_operator_plan = old_plan
        runner._platform_forward_lock = serving_lock
        runtime_context = mock.Mock()
        runtime_context.resolved_server_args_dict.return_value = {
            "model_path": "original",
            "load_format": "auto",
        }

        build_entered = threading.Event()
        release_build = threading.Event()
        forward_attempted = threading.Event()
        forward_entered = threading.Event()
        observed_models = []
        errors = []

        def build_plan(_model):
            build_entered.set()
            if not release_build.wait(timeout=2):
                raise TimeoutError("test did not release provider build")
            return new_plan

        def replace_model():
            try:
                ModelRunner.update_model_fields(
                    runner,
                    new_model,
                    model_path="replacement",
                    load_format="auto",
                    load_config=object(),
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def forward_unlocked(*_args):
            observed_models.append(runner.model)
            forward_entered.set()
            return "output"

        def run_forward():
            try:
                forward_attempted.set()
                ModelRunner._forward_raw(runner, object(), None)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        runner._forward_raw_unlocked = forward_unlocked
        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_context",
                return_value=runtime_context,
            ),
            mock.patch.object(
                ModelRunner,
                "_bind_platform_runtime_operators",
                side_effect=build_plan,
            ),
        ):
            replacement_thread = threading.Thread(target=replace_model)
            replacement_thread.start()
            self.assertTrue(build_entered.wait(timeout=2))

            forward_thread = threading.Thread(target=run_forward)
            forward_thread.start()
            self.assertTrue(forward_attempted.wait(timeout=2))
            self.assertFalse(forward_entered.wait(timeout=0.1))

            release_build.set()
            replacement_thread.join(timeout=2)
            forward_thread.join(timeout=2)

        self.assertFalse(replacement_thread.is_alive())
        self.assertFalse(forward_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observed_models, [new_model])

    def test_model_runner_delegates_state_and_lifecycle_to_plan(self):
        state = {"enabled": True, "attention_backend": "metal_jit"}
        plan = mock.MagicMock()
        plan.get_state.return_value = state
        runner = types.SimpleNamespace(
            platform_operator_plan=plan,
            _platform_forward_lock=None,
        )

        self.assertIs(ModelRunner.get_platform_operator_state(runner), state)
        ModelRunner.invalidate_platform_operator_views(runner)
        ModelRunner.close_platform_operators(runner)

        plan.get_state.assert_called_once_with()
        plan.invalidate_views.assert_called_once_with()
        plan.close.assert_called_once_with()

    def test_worker_boundary_delegates_platform_operator_state(self):
        from sglang.srt.managers.tp_worker import BaseTpWorker

        state = {"enabled": True, "attention_backend": "metal_jit"}
        worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                get_platform_operator_state=lambda: state
            )
        )

        self.assertIs(BaseTpWorker.get_platform_operator_state(worker), state)

    def test_graceful_shutdown_closes_plan_before_cache_owners(self):
        from sglang.srt.managers.scheduler import Scheduler

        events = []
        scheduler = types.SimpleNamespace(
            tp_worker=types.SimpleNamespace(
                close_platform_operators=lambda: events.append("target_plan")
            ),
            draft_worker=types.SimpleNamespace(
                close_platform_operators=lambda: events.append("draft_plan")
            ),
            hisparse_coordinator=None,
            tree_cache=types.SimpleNamespace(
                release_host_resources=lambda: events.append("tree_cache")
            ),
            decode_offload_manager=None,
        )

        Scheduler.release_host_resources(scheduler)

        self.assertEqual(events, ["target_plan", "draft_plan", "tree_cache"])

    def test_tp_worker_closes_each_owned_runner_once(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        first = mock.MagicMock()
        second = mock.MagicMock()
        worker = types.SimpleNamespace(model_runner_list=[first, second, first])

        TpModelWorker.close_platform_operators(worker)

        first.close_platform_operators.assert_called_once_with()
        second.close_platform_operators.assert_called_once_with()

    def test_spec_workers_delegate_platform_plan_close_to_draft_worker(self):
        from sglang.srt.speculative.base_spec_worker import (
            BaseSpecWorker,
            EagleDraftWorkerBase,
        )

        base_draft = mock.MagicMock()
        eagle_draft = mock.MagicMock()

        BaseSpecWorker.close_platform_operators(
            types.SimpleNamespace(draft_worker=base_draft)
        )
        EagleDraftWorkerBase.close_platform_operators(
            types.SimpleNamespace(draft_worker=eagle_draft)
        )

        base_draft.close_platform_operators.assert_called_once_with()
        eagle_draft.close_platform_operators.assert_called_once_with()

    def test_mps_checkpoint_ipc_rejects_before_cuda_access(self):
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(weight_cache_mode="off")
        )
        updater = self._weight_updater(nn.Module(), runner)

        success, message = updater.update_weights_from_ipc(
            types.SimpleNamespace(zmq_handles={})
        )

        self.assertFalse(success)
        self.assertIn("require a CUDA-compatible device", message)

    def test_mps_distributed_updates_reject_before_collectives(self):
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(weight_cache_mode="off")
        )
        updater = self._weight_updater(nn.Module(), runner)

        with mock.patch.object(torch.distributed, "is_initialized") as initialized:
            success, message = updater.init_weights_update_group(
                "127.0.0.1", 12345, 0, 1, "mps-update", backend="gloo"
            )
        self.assertFalse(success)
        self.assertIn("cannot broadcast MPS tensors", message)
        initialized.assert_not_called()

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            for load_format in (None, "flattened_bucket"):
                with self.subTest(load_format=load_format):
                    success, message = updater.update_weights_from_distributed(
                        [], [], [], "mps-update", load_format=load_format
                    )
                    self.assertFalse(success)
                    self.assertIn("cannot broadcast MPS tensors", message)
        broadcast.assert_not_called()

    @staticmethod
    def _weight_updater(model, runner):
        return WeightUpdater(
            tp_rank=0,
            device="mps",
            gpu_id=0,
            model_config=types.SimpleNamespace(),
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: runner,
        )


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
class TestPlatformOperatorMpsWeightUpdates(unittest.TestCase):
    def test_tensor_update_uses_explicit_mps_device(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.zeros(2, device="mps"), requires_grad=False
                )

            def load_weights(self, named_tensors):
                for name, tensor in named_tensors:
                    self.get_parameter(name).data.copy_(tensor)

        model = TinyModel()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(weight_cache_mode="off")
        )
        updater = TestPlatformOperatorLifecycle._weight_updater(model, runner)

        success, message = updater.update_weights_from_tensor(
            [("weight", torch.ones(2))]
        )

        self.assertTrue(success, message)
        torch.testing.assert_close(model.weight.cpu(), torch.ones(2))

    def test_weight_update_holds_platform_lock_during_mutation(self):
        events = []
        lock = _RecordingLock(events)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, device="mps"))

            def load_weights(self, named_tensors):
                events.append(("load", lock.active))
                for name, tensor in named_tensors:
                    self.get_parameter(name).data.copy_(tensor)

        model = TinyModel()
        runner = types.SimpleNamespace(
            _platform_forward_lock=lock,
            server_args=types.SimpleNamespace(weight_cache_mode="off"),
            invalidate_platform_operator_views=lambda: events.append(
                ("invalidate", lock.active)
            ),
        )
        updater = TestPlatformOperatorLifecycle._weight_updater(model, runner)

        success, message = updater.update_weights_from_tensor(
            [("weight", torch.ones(2))]
        )

        self.assertTrue(success, message)
        self.assertEqual(events[0], "enter")
        self.assertIn(("load", True), events)
        self.assertIn(("invalidate", True), events)
        self.assertEqual(events[-1], "exit")


if __name__ == "__main__":
    unittest.main()
