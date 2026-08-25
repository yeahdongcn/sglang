"""CPU-safe lifecycle contracts for MPS operator plans."""

import threading
import types
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")


class _RecordingLock:
    def __init__(self, events):
        self.events = events
        self.active = False

    def __enter__(self):
        self.active = True
        self.events.append("enter")
        return self

    def __exit__(self, *_args):
        self.active = False
        self.events.append("exit")


def test_model_runner_serializes_raw_forward_when_plan_has_a_lock():
    events = []
    lock = _RecordingLock(events)

    def forward_unlocked(*_args):
        assert lock.active
        events.append("forward")
        return "hidden"

    runner = types.SimpleNamespace(
        _platform_forward_lock=lock,
        _forward_raw_unlocked=forward_unlocked,
    )
    assert ModelRunner._forward_raw(runner, object(), None) == "hidden"
    assert events == ["enter", "forward", "exit"]


def test_model_runner_builds_and_publishes_one_platform_plan():
    plan = mock.Mock(forward_lock=threading.RLock())
    platform = mock.MagicMock(device_type="mps")
    platform.bind_model_runtime_operators.return_value = plan
    runner = object.__new__(ModelRunner)
    runner.model = nn.Module()
    runner.model_config = object()
    runner.server_args = object()
    runner.req_to_token_pool = object()
    runner.token_to_kv_pool = object()
    runner.platform_operator_plan = None
    runner._platform_forward_lock = None

    with mock.patch(
        "sglang.srt.model_executor.model_runner.current_platform", platform
    ):
        ModelRunner._bind_platform_runtime_operators(runner)

    assert runner.platform_operator_plan is plan
    assert runner._platform_forward_lock is plan.forward_lock
    plan.close.assert_not_called()
    platform.bind_model_runtime_operators.assert_called_once()


def test_model_runner_delegates_close_invalidate_and_telemetry():
    state = {"enabled": True, "whole_model_backend": "mlx"}
    plan = mock.MagicMock()
    plan.get_state.return_value = state
    runner = types.SimpleNamespace(
        platform_operator_plan=plan,
        _platform_forward_lock=None,
    )

    assert ModelRunner.get_platform_operator_state(runner) is state
    ModelRunner.invalidate_platform_operator_views(runner)
    ModelRunner.close_platform_operators(runner)
    plan.get_state.assert_called_once_with()
    plan.invalidate_views.assert_called_once_with()
    plan.close.assert_called_once_with()


def test_model_replacement_rolls_back_context_and_leaves_old_torch_model():
    old_model = nn.Module()
    old_plan = mock.MagicMock()
    runner = object.__new__(ModelRunner)
    runner.model = old_model
    runner.model_config = object()
    runner.server_args = types.SimpleNamespace(model_path="old", load_format="auto")
    runner.is_draft_worker = False
    runner.platform_operator_plan = old_plan
    runner._platform_forward_lock = None
    context = mock.Mock()
    context.resolved_server_args_dict.return_value = {
        "model_path": "old",
        "load_format": "auto",
    }
    replacement = nn.Module()

    with (
        mock.patch(
            "sglang.srt.model_executor.model_runner.get_context",
            return_value=context,
        ),
        mock.patch.object(
            ModelRunner,
            "_build_platform_runtime_operator_plan",
            side_effect=RuntimeError("provider contract failed"),
        ),
        pytest.raises(RuntimeError, match="provider contract failed"),
    ):
        ModelRunner.update_model_fields(
            runner,
            replacement,
            model_path="new",
            load_format="auto",
            load_config=object(),
        )

    assert runner.model is old_model
    assert runner.platform_operator_plan is None
    old_plan.close.assert_called_once_with()
    assert context.override.call_args_list[-1] == mock.call(
        "model_runner.update_model_fields.rollback",
        model_path="old",
        load_format="auto",
    )


def test_tp_worker_closes_each_owned_runner_once():
    from sglang.srt.managers.tp_worker import TpModelWorker

    first = mock.MagicMock()
    second = mock.MagicMock()
    worker = types.SimpleNamespace(model_runner_list=[first, second, first])

    TpModelWorker.close_platform_operators(worker)

    first.close_platform_operators.assert_called_once_with()
    second.close_platform_operators.assert_called_once_with()


def test_weight_update_invalidates_borrowed_views_while_holding_runner_lock():
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        WeightUpdater,
    )

    events = []
    lock = _RecordingLock(events)

    class TinyModel:
        def load_weights(self, named_tensors):
            assert lock.active
            events.append(("load", named_tensors))

    model = TinyModel()
    runner = types.SimpleNamespace(
        _platform_forward_lock=lock,
        server_args=types.SimpleNamespace(weight_cache_mode="off"),
        invalidate_platform_operator_views=lambda: events.append(
            ("invalidate", lock.active)
        ),
    )
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=types.SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
    )

    success, message = updater.update_weights_from_tensor([("weight", torch.ones(2))])

    assert success, message
    assert events[0] == "enter"
    assert events[1][0] == "load"
    assert events[2] == ("invalidate", True)
    assert events[-1] == "exit"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
