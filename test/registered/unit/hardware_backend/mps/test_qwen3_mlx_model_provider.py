"""CPU-safe contracts for the decode-only whole-model Qwen3 MLX island."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx as qwen3_mlx
from sglang.kernels.spec import KernelBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")


class _FakeMlxArray:
    def __init__(self, name, shape=(1,), dtype="mlx.core.bfloat16"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


def _fake_decode_views(prefix="view"):
    arrays = iter(
        _FakeMlxArray(f"{prefix}-{index}")
        for index in range(2 + qwen3_mlx.QWEN3_06B_NUM_LAYERS * 11)
    )

    def ref():
        return SimpleNamespace(array=next(arrays))

    embedding = ref()
    layers = tuple(
        qwen3_mlx._LayerViews(
            *(ref() for _ in range(11)),
            input_epsilon=1e-6,
            qk_epsilon=1e-6,
            post_attention_epsilon=1e-6,
        )
        for _ in range(qwen3_mlx.QWEN3_06B_NUM_LAYERS)
    )
    return qwen3_mlx._DecodeViews(
        embedding=embedding,
        layers=layers,
        final_norm=ref(),
        final_epsilon=1e-6,
        pool_identity=7,
        pool_slots=2048,
    )


def _decode_batch(**overrides):
    values = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        out_cache_loc=torch.tensor([7], dtype=torch.int64),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compiled_decode_capture_order_is_stable_and_refreshable():
    old_views = _fake_decode_views("old")
    new_views = _fake_decode_views("new")
    compiled_fn = mock.Mock(return_value=(object(), object(), object()))

    with mock.patch("mlx.core.compile", return_value=compiled_fn) as compile_fn:
        bundle = qwen3_mlx._CompiledBs1Decode.create(old_views)

    compile_fn.assert_called_once()
    assert len(bundle.captures) == 2 + qwen3_mlx.QWEN3_06B_NUM_LAYERS * 11
    captures_identity = id(bundle.captures)
    bundle.refresh(new_views)
    assert id(bundle.captures) == captures_identity
    assert bundle.owner is new_views


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(ForwardMode.DECODE, True), (ForwardMode.EXTEND, False)],
)
def test_provider_is_decode_only(mode, expected):
    provider = qwen3_mlx.Qwen3MlxModelProvider(views=None, started=True)
    batch = SimpleNamespace(forward_mode=mode)
    if expected:
        with mock.patch.object(provider, "_decode_fallback_reason", return_value=None):
            assert provider.should_run(batch)
    else:
        assert not provider.should_run(batch)
        assert provider.last_selector_fallback_reason == "forward_mode"


def test_provider_forward_returns_hidden_and_commits_deferred_kv_once():
    hidden = torch.empty(1, 1024)
    new_k = torch.empty(28, 1, 8, 128)
    new_v = torch.empty_like(new_k)
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    views = SimpleNamespace(layers=(layer,), pool_slots=2048)
    provider = qwen3_mlx.Qwen3MlxModelProvider(
        views=views,
        started=True,
        kv_commit_backend=KernelBackend.TORCH,
    )
    graph_result = object()
    req_to_token = torch.empty(2, 2048, dtype=torch.int32)
    batch = _decode_batch()

    def bridge(operation, *inputs, device, after_mps_fence):
        assert after_mps_fence is None
        assert operation(*(["mlx"] * len(inputs))) is graph_result
        assert device == inputs[0].device
        return hidden, new_k, new_v

    model = SimpleNamespace(layers_to_capture=[])
    with (
        mock.patch.object(
            provider,
            "_validate_decode_inputs",
            return_value=(object(), req_to_token),
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch.object(qwen3_mlx, "_mlx_decode_graph", return_value=graph_result),
        mock.patch.object(qwen3_mlx, "mlx_call_multi", side_effect=bridge),
        mock.patch.object(qwen3_mlx, "qwen3_commit_deferred_kv") as commit,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_in_closed_range"),
    ):
        result = provider.forward(
            model,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([7], dtype=torch.int64),
            batch,
        )

    assert result is hidden
    commit.assert_called_once_with(
        new_k,
        new_v,
        batch.out_cache_loc,
        (layer.k_pool.torch_tensor,),
        (layer.v_pool.torch_tensor,),
        backend=KernelBackend.TORCH,
    )
    assert provider.call_count == 1
    assert provider.decode_call_count == 1
    assert provider._pending_commit_sources == (new_k, new_v)


def test_provider_invalidation_drops_views_and_close_is_idempotent():
    provider = qwen3_mlx.Qwen3MlxModelProvider(
        views=_fake_decode_views(),
        started=True,
    )
    compiled = mock.Mock()
    provider._compiled_decode = compiled

    provider.invalidate_views()
    assert provider.views is None
    provider.close()
    provider.close()

    compiled.close.assert_called_once_with()
    assert provider.closed
    assert provider._compiled_decode is None


def test_qwen3_model_dispatches_provider_and_falls_back_to_torch():
    model = object.__new__(Qwen3Model)
    torch.nn.Module.__init__(model)
    provider = mock.Mock()
    provider.should_run.return_value = True
    provider.forward.return_value = "hidden-from-mlx"
    model.model_forward_provider = provider
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    input_ids = torch.tensor([1])
    positions = torch.tensor([0])

    assert Qwen3Model.forward(model, input_ids, positions, batch) == "hidden-from-mlx"
    provider.forward.assert_called_once()

    provider.should_run.return_value = False
    with mock.patch.object(Qwen2Model, "forward", return_value="torch-hidden"):
        assert Qwen3Model.forward(model, input_ids, positions, batch) == "torch-hidden"


def test_causal_lm_disables_whole_model_provider_for_aux_hidden_capture():
    causal_lm = object.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(causal_lm)
    causal_lm.capture_aux_hidden_states = True
    causal_lm.pp_group = SimpleNamespace(is_last_rank=False)
    causal_lm.model = mock.Mock(return_value=("hidden", []))

    result = Qwen3ForCausalLM.forward(
        causal_lm,
        torch.tensor([1]),
        torch.tensor([0]),
        SimpleNamespace(forward_mode=ForwardMode.DECODE),
    )

    assert result == "hidden"
    assert causal_lm.model.call_args.kwargs["allow_model_forward_provider"] is False


def test_provider_rejects_non_qwen3_model_at_static_validation():
    with pytest.raises(RuntimeError, match="only Qwen3ForCausalLM"):
        qwen3_mlx.validate_qwen3_mlx_static_contract(
            torch.nn.Linear(1, 1), object(), object()
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
