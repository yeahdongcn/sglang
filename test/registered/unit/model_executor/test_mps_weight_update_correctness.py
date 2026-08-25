"""CPU-safe coverage for standard-Torch online weight updates on MPS."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _get_weight_update_device,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _updater(*, device: str = "mps", model=None) -> WeightUpdater:
    if model is None:
        model = mock.Mock()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(weight_cache_mode="off"),
    )
    return WeightUpdater(
        tp_rank=0,
        device=device,
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
    )


def test_weight_update_device_falls_back_to_explicit_mps_device():
    with mock.patch.object(
        torch,
        "get_device_module",
        return_value=SimpleNamespace(),
    ):
        assert _get_weight_update_device("mps") == torch.device("mps")


def test_weight_update_device_preserves_cuda_current_device():
    with mock.patch.object(
        torch,
        "get_device_module",
        return_value=SimpleNamespace(current_device=lambda: 3),
    ):
        assert _get_weight_update_device("cuda") == 3


def test_tensor_update_passes_explicit_mps_device_to_deserializer():
    model = mock.Mock()
    updater = _updater(model=model)
    source = torch.ones(2)
    moved = object()

    with (
        mock.patch.object(weight_updater, "monkey_patch_torch_reductions"),
        mock.patch.object(
            weight_updater,
            "_unsupported_derived_weight_cache_error",
            return_value=None,
        ),
        mock.patch.object(
            torch,
            "get_device_module",
            return_value=SimpleNamespace(),
        ),
        mock.patch.object(
            weight_updater, "_unwrap_tensor", return_value=moved
        ) as unwrap,
    ):
        success, message = updater.update_weights_from_tensor([("weight", source)])

    assert success, message
    unwrap.assert_called_once_with(source, tp_rank=0, device=torch.device("mps"))
    model.load_weights.assert_called_once_with([("weight", moved)])


def test_mps_update_group_rejects_before_distributed_initialization():
    updater = _updater()

    with mock.patch.object(torch.distributed, "is_initialized") as initialized:
        success, message = updater.init_weights_update_group(
            "127.0.0.1",
            12345,
            0,
            1,
            "mps-update",
            backend="gloo",
        )

    assert not success
    assert "cannot broadcast MPS tensors" in message
    initialized.assert_not_called()


@pytest.mark.parametrize("load_format", [None, "flattened_bucket"])
def test_mps_distributed_update_rejects_before_broadcast(load_format):
    updater = _updater()

    with mock.patch.object(torch.distributed, "broadcast") as broadcast:
        success, message = updater.update_weights_from_distributed(
            [],
            [],
            [],
            "mps-update",
            load_format=load_format,
        )

    assert not success
    assert "cannot broadcast MPS tensors" in message
    broadcast.assert_not_called()


def test_mps_checkpoint_ipc_rejects_before_checkpoint_worker_import():
    updater = _updater()
    module_name = "sglang.srt.checkpoint_engine.checkpoint_engine_worker"

    with (
        mock.patch.object(
            weight_updater,
            "_unsupported_derived_weight_cache_error",
            return_value=None,
        ),
        mock.patch.dict(sys.modules, {module_name: None}),
    ):
        success, message = updater.update_weights_from_ipc(
            SimpleNamespace(zmq_handles={})
        )

    assert not success
    assert "require a CUDA-compatible device" in message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
