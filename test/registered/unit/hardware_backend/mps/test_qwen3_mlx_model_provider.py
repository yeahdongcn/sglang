"""CPU/mock contracts for the whole-model Qwen3 MLX model-forward hook.

Numeric Metal coverage lives with the focused attention/commit kernels.  These
tests intentionally allocate no model weights or KV cache.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx as qwen3_mlx
from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (
    QWEN3_COLD_PREFILL_MAX_TOKENS,
    QWEN3_COLD_PREFILL_MIN_TOKENS,
    Qwen3MlxModelProvider,
    create_qwen3_mlx_model_provider,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


class _FakeMlxArray:
    def __init__(self, name, shape=(1,), dtype="mlx.core.bfloat16"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


def _fake_decode_views(prefix="old"):
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
    final_norm = ref()
    return qwen3_mlx._DecodeViews(
        embedding=embedding,
        layers=layers,
        final_norm=final_norm,
        final_epsilon=1e-6,
        pool_identity=7,
        pool_slots=2048,
    )


def test_compiled_decode_flatten_round_trip_has_stable_310_leaf_order():
    views = _fake_decode_views()
    flattened = qwen3_mlx._flatten_decode_arrays(views)
    rebuilt = qwen3_mlx._unflatten_decode_arrays(
        flattened, qwen3_mlx._decode_static(views)
    )

    assert len(flattened) == 310
    assert qwen3_mlx._flatten_decode_arrays(rebuilt) == flattened


def test_rope_cache_bound_is_validated_once_at_provider_installation():
    assert qwen3_mlx._require_equal_rope_cache_length(None, 2048, 0) == 2048
    assert qwen3_mlx._require_equal_rope_cache_length(2048, 2048, 1) == 2048
    with pytest.raises(RuntimeError, match="equal RoPE cache lengths"):
        qwen3_mlx._require_equal_rope_cache_length(2048, 1024, 2)
    with pytest.raises(RuntimeError, match="non-empty RoPE cache"):
        qwen3_mlx._require_equal_rope_cache_length(None, 0, 0)


def test_compiled_decode_uses_mutable_inputs_and_refreshes_in_place():
    old_views = _fake_decode_views("old")
    new_views = _fake_decode_views("new")
    compiled_fn = mock.Mock(return_value=object())
    with mock.patch("mlx.core.compile", return_value=compiled_fn) as compile_fn:
        bundle = qwen3_mlx._CompiledBs1Decode.create(old_views)

    compile_fn.assert_called_once()
    assert compile_fn.call_args.kwargs == {
        "inputs": bundle.captures,
        "shapeless": False,
    }
    captures_identity = id(bundle.captures)
    old_capture = bundle.captures[0]

    bundle.refresh(new_views)

    assert id(bundle.captures) == captures_identity
    assert bundle.captures[0] is not old_capture
    assert bundle.owner is new_views


def test_compiled_decode_is_bs1_only_and_reuses_one_signature():
    views = _fake_decode_views()
    compiled_fn = mock.Mock(return_value=object())
    with mock.patch("mlx.core.compile", return_value=compiled_fn):
        bundle = qwen3_mlx._CompiledBs1Decode.create(views)

    bs1 = (
        _FakeMlxArray("ids", dtype="mlx.core.int64"),
        _FakeMlxArray("pos", dtype="mlx.core.int64"),
        _FakeMlxArray("table", shape=(8, 32), dtype="mlx.core.int32"),
        _FakeMlxArray("req", dtype="mlx.core.int64"),
        _FakeMlxArray("lens", dtype="mlx.core.int64"),
    )
    assert bundle.can_run(bs1)
    bundle.warmup(*bs1)
    bundle(*bs1)
    assert bundle.can_run(bs1)
    assert not bundle.can_run(
        (_FakeMlxArray("ids", shape=(2,), dtype="mlx.core.int64"), *bs1[1:])
    )
    assert bundle.warmup_count == 1
    assert bundle.call_count == 1
    assert compiled_fn.call_args_list == [mock.call(*bs1), mock.call(*bs1)]
    provider = Qwen3MlxModelProvider(views=views)
    provider._compiled_decode = bundle
    state = provider.get_compiled_decode_state()
    assert state["enabled"]
    assert state["total_enabled"]
    assert state["primary_variant"] == "hidden"
    assert state["warmup_count"] == 1
    assert state["call_count"] == 1
    assert state["total_warmup_count"] == 1
    assert state["total_call_count"] == 1


def test_compiled_decode_greedy_variant_is_the_only_resident_graph():
    views = _fake_decode_views()
    greedy_compiled = mock.Mock(return_value=(object(), object(), object()))
    with mock.patch("mlx.core.compile", return_value=greedy_compiled) as compile_fn:
        bundle = qwen3_mlx._CompiledBs1Decode.create(views, compile_greedy_tail=True)

    assert compile_fn.call_count == 1
    assert compile_fn.call_args_list[0].kwargs["inputs"] is bundle.captures
    assert bundle.compiled is None
    assert bundle.compiled_greedy is greedy_compiled
    bs1 = (
        _FakeMlxArray("ids", dtype="mlx.core.int64"),
        _FakeMlxArray("pos", dtype="mlx.core.int64"),
        _FakeMlxArray("table", shape=(8, 32), dtype="mlx.core.int32"),
        _FakeMlxArray("req", dtype="mlx.core.int64"),
        _FakeMlxArray("lens", dtype="mlx.core.int64"),
    )

    assert not bundle.can_run_hidden(bs1)
    assert bundle.can_run_greedy(bs1)
    bundle.warmup_greedy(*bs1)
    bundle.greedy(*bs1)

    assert bundle.greedy_warmup_count == 1
    assert bundle.greedy_call_count == 1
    assert greedy_compiled.call_args_list == [mock.call(*bs1), mock.call(*bs1)]
    provider = Qwen3MlxModelProvider(views=views)
    provider._compiled_decode = bundle
    state = provider.get_compiled_decode_state()
    assert not state["enabled"]
    assert state["total_enabled"]
    assert state["primary_variant"] == "greedy"
    assert state["warmup_count"] == 0
    assert state["call_count"] == 0
    assert state["total_warmup_count"] == 1
    assert state["total_call_count"] == 1


def test_greedy_graph_projects_all_decode_rows_but_only_last_prefill_row():
    views = object()
    hidden = object()
    new_k = object()
    new_v = object()
    token_ids = object()
    with (
        mock.patch.object(
            qwen3_mlx,
            "_mlx_decode_graph",
            return_value=(hidden, new_k, new_v),
        ) as decode,
        mock.patch.object(
            qwen3_mlx, "_mlx_greedy_token", return_value=token_ids
        ) as project,
    ):
        assert qwen3_mlx._mlx_decode_greedy_graph(
            views, "ids", "pos", "table", "req", "lens"
        ) == (token_ids, new_k, new_v)
    decode.assert_called_once_with(views, "ids", "pos", "table", "req", "lens")
    project.assert_called_once_with(views, hidden, last_only=False)

    with (
        mock.patch.object(
            qwen3_mlx,
            "_mlx_cold_prefill_graph",
            return_value=(hidden, new_k, new_v),
        ) as prefill,
        mock.patch.object(
            qwen3_mlx, "_mlx_greedy_token", return_value=token_ids
        ) as project,
    ):
        assert qwen3_mlx._mlx_cold_prefill_greedy_graph(views, "ids", "pos") == (
            token_ids,
            new_k,
            new_v,
        )
    prefill.assert_called_once_with(views, "ids", "pos")
    project.assert_called_once_with(views, hidden, last_only=True)


def test_greedy_tail_static_contract_requires_exact_tied_torch_semantics():
    embedding = torch.nn.Embedding(10, 1024)
    logits_processor = SimpleNamespace(
        logit_scale=None,
        final_logit_softcapping=None,
        use_fp32_lm_head=False,
        do_tensor_parallel_all_gather=False,
        do_tensor_parallel_all_gather_dp_attn=False,
        return_full_logits=False,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=True, vocab_size=10),
        model=SimpleNamespace(embed_tokens=embedding),
        lm_head=embedding,
        logits_processor=logits_processor,
    )
    server_args = SimpleNamespace(
        sampling_backend="pytorch",
        enable_fp32_lm_head=False,
        enable_dp_lm_head=False,
    )

    assert (
        qwen3_mlx._greedy_tail_static_fallback_reason(model, server_args, enabled=True)
        is None
    )
    assert (
        qwen3_mlx._greedy_tail_static_fallback_reason(model, server_args, enabled=False)
        == "disabled by SGLANG_MPS_QWEN3_GREEDY_TAIL priority"
    )
    model.lm_head = torch.nn.Embedding(10, 1024)
    assert (
        qwen3_mlx._greedy_tail_static_fallback_reason(model, server_args, enabled=True)
        == "LM head and token embedding are different modules"
    )


def test_provider_constructor_enforces_greedy_backend_invariant():
    torch_tail = Qwen3MlxModelProvider(views=None)
    assert (
        torch_tail.greedy_tail_static_fallback_reason
        == "disabled by SGLANG_MPS_QWEN3_GREEDY_TAIL priority"
    )

    mlx_tail = Qwen3MlxModelProvider(views=None, greedy_tail_backend="mlx")
    assert mlx_tail.greedy_tail_static_fallback_reason is None

    with pytest.raises(ValueError, match="must be 'mlx' or 'torch'"):
        Qwen3MlxModelProvider(views=None, greedy_tail_backend="metal")


@pytest.mark.parametrize(
    ("observer_active", "expected_reason"),
    [
        (False, None),
        (True, "sampling_observer"),
        (None, "sampling_observer_state"),
    ],
)
def test_greedy_tail_requires_resolved_inactive_sampling_observer(
    observer_active,
    expected_reason,
):
    provider = Qwen3MlxModelProvider(views=None, greedy_tail_backend="mlx")
    sampling_info = SimpleNamespace(
        is_all_greedy=True,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=sampling_info,
        sampling_observer_active=observer_active,
    )

    with (
        mock.patch.object(envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=False),
        mock.patch.object(envs.SGLANG_SANITIZE_NAN_LOGITS, "get", return_value=False),
    ):
        assert provider._greedy_tail_fallback_reason(batch) == expected_reason


def test_provider_refreshes_compiled_captures_only_after_fence():
    old_views = _fake_decode_views("old")
    new_views = _fake_decode_views("new")
    bundle = mock.Mock(owner=old_views)
    provider = Qwen3MlxModelProvider(views=new_views)
    provider._compiled_decode = bundle
    old_sources = (object(), object())
    provider._pending_commit_sources = old_sources

    provider._after_mps_fence(new_views)

    bundle.refresh.assert_called_once_with(new_views)
    assert provider._pending_commit_sources is None


def _empty_module(cls):
    module = object.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


@pytest.mark.parametrize(
    "mode, expected",
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.EXTEND, False),
        (ForwardMode.SPLIT_PREFILL, False),
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DRAFT_EXTEND_V2, False),
        (ForwardMode.IDLE, False),
    ],
)
def test_provider_selects_exact_decode_without_prefill_metadata(mode, expected):
    provider = Qwen3MlxModelProvider(views=None, started=True)
    if mode is ForwardMode.DECODE:
        with mock.patch.object(provider, "_decode_fallback_reason", return_value=None):
            assert provider.should_run(SimpleNamespace(forward_mode=mode)) is expected
    else:
        assert provider.should_run(SimpleNamespace(forward_mode=mode)) is expected


def _cold_batch(token_count=QWEN3_COLD_PREFILL_MIN_TOKENS, **overrides):
    values = dict(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        contains_last_prefill_chunk=True,
        extend_num_tokens=token_count,
        extend_seq_lens_cpu=[token_count],
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=torch.tensor([token_count]),
        num_token_non_padded_cpu=token_count,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        lora_ids=[None],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_provider_selects_only_bounded_complete_cold_prefill():
    provider = Qwen3MlxModelProvider(views=None, started=True)
    model = SimpleNamespace(layers_to_capture=[])
    with mock.patch.object(qwen3_mlx, "_is_mps_vector", return_value=True):
        for token_count in (
            QWEN3_COLD_PREFILL_MIN_TOKENS,
            QWEN3_COLD_PREFILL_MAX_TOKENS,
        ):
            input_ids = torch.zeros(token_count, dtype=torch.int64)
            positions = torch.arange(token_count, dtype=torch.int64)
            assert provider.should_run(
                _cold_batch(token_count),
                model=model,
                input_ids=input_ids,
                positions=positions,
            )

    cases = (
        (_cold_batch(contains_last_prefill_chunk=False), {}),
        (_cold_batch(extend_prefix_lens_cpu=[1]), {}),
        (_cold_batch(batch_size=2), {}),
        (_cold_batch(forward_mode=ForwardMode.SPLIT_PREFILL), {}),
        (_cold_batch(capture_hidden_mode=CaptureHiddenMode.FULL), {}),
        (_cold_batch(spec_info=object()), {}),
        (_cold_batch(spec_algorithm=SpeculativeAlgorithm.EAGLE), {}),
        (_cold_batch(lora_ids=["adapter"]), {}),
        (_cold_batch(), {"input_embeds": torch.zeros(1, 1024)}),
        (_cold_batch(), {"pp_proxy_tensors": object()}),
    )
    input_ids = torch.zeros(QWEN3_COLD_PREFILL_MIN_TOKENS, dtype=torch.int64)
    for batch, kwargs in cases:
        assert not provider.should_run(
            batch,
            model=model,
            input_ids=input_ids,
            **kwargs,
        )

    too_small = QWEN3_COLD_PREFILL_MIN_TOKENS - 1
    assert not provider.should_run(
        _cold_batch(too_small),
        model=model,
        input_ids=torch.zeros(too_small, dtype=torch.int64),
    )


def test_qwen3_model_typed_hook_dispatches_decode():
    model = _empty_module(Qwen3Model)
    provider = mock.Mock()
    provider.should_run.return_value = True
    sentinel = object()
    provider.forward.return_value = sentinel
    model.model_forward_provider = provider
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    input_ids = torch.tensor([1])
    positions = torch.tensor([0])

    result = Qwen3Model.forward(model, input_ids, positions, batch)

    assert result is sentinel
    provider.should_run.assert_called_once_with(
        batch,
        model=model,
        input_ids=input_ids,
        positions=positions,
        input_embeds=None,
        pp_proxy_tensors=None,
    )
    provider.forward.assert_called_once_with(
        model,
        input_ids,
        positions,
        batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    )


def test_qwen3_causal_lm_passes_precomputed_token_output_through():
    causal_lm = _empty_module(Qwen3ForCausalLM)
    qwen3_model = _empty_module(Qwen3Model)
    provider = mock.Mock()
    provider.should_run.return_value = True
    token_ids = torch.tensor([5], dtype=torch.int64)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=token_ids,
    )
    provider.forward.return_value = output
    qwen3_model.model_forward_provider = provider
    causal_lm.model = qwen3_model
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    result = Qwen3ForCausalLM.forward(
        causal_lm,
        torch.tensor([1]),
        torch.tensor([0]),
        batch,
    )

    assert result is output


def test_qwen3_embedding_bypasses_model_forward_provider():
    causal_lm = _empty_module(Qwen3ForCausalLM)
    qwen3_model = _empty_module(Qwen3Model)
    provider = mock.Mock()
    qwen3_model.model_forward_provider = provider
    causal_lm.model = qwen3_model
    causal_lm.capture_aux_hidden_states = False
    causal_lm.pp_group = SimpleNamespace(is_last_rank=True)
    pooled = object()
    causal_lm.pooler = mock.Mock(return_value=pooled)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    hidden_states = torch.zeros(1, 1024)

    with mock.patch.object(Qwen2Model, "forward", return_value=hidden_states) as native:
        result = Qwen3ForCausalLM.forward(
            causal_lm,
            torch.tensor([1]),
            torch.tensor([0]),
            batch,
            get_embedding=True,
        )

    assert result is pooled
    provider.should_run.assert_not_called()
    provider.forward.assert_not_called()
    native.assert_called_once()
    causal_lm.pooler.assert_called_once_with(hidden_states, batch)


def test_qwen3_model_non_decode_keeps_standard_forward():
    model = _empty_module(Qwen3Model)
    provider = mock.Mock()
    provider.should_run.return_value = False
    model.model_forward_provider = provider
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    sentinel = object()

    with mock.patch.object(Qwen2Model, "forward", return_value=sentinel) as native:
        result = Qwen3Model.forward(
            model,
            torch.tensor([1]),
            torch.tensor([0]),
            batch,
        )

    assert result is sentinel
    native.assert_called_once()
    provider.should_run.assert_called_once()
    provider.forward.assert_not_called()


def test_provider_rejects_direct_non_decode_invocation_before_bridge():
    provider = Qwen3MlxModelProvider(views=None)
    with pytest.raises(RuntimeError, match="forward_mode"):
        provider._validate_decode_inputs(
            SimpleNamespace(layers_to_capture=[]),
            torch.tensor([1]),
            torch.tensor([0]),
            SimpleNamespace(forward_mode=ForwardMode.EXTEND),
            None,
            None,
        )


def test_cold_prefill_forward_uses_one_bridge_and_commits_all_token_rows():
    token_count = QWEN3_COLD_PREFILL_MIN_TOKENS
    hidden = torch.empty(token_count, 1024)
    new_k = torch.empty(28, token_count, 8, 128)
    new_v = torch.empty_like(new_k)
    slots = torch.arange(token_count, dtype=torch.int64)
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    views = SimpleNamespace(layers=(layer,), pool_slots=2048)
    provider = Qwen3MlxModelProvider(
        views=views,
        started=True,
        greedy_tail_backend="mlx",
        kv_commit_backend=KernelBackend.METAL_JIT,
    )
    batch = _cold_batch(
        token_count,
        out_cache_loc=slots,
        sampling_observer_active=True,
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    graph_result = object()

    def bridge(
        operation,
        input_ids,
        positions,
        *,
        device,
        after_mps_fence,
    ):
        assert after_mps_fence is None
        assert operation("mlx ids", "mlx positions") is graph_result
        assert input_ids.shape == (token_count,)
        assert positions.shape == (token_count,)
        assert device == input_ids.device
        return hidden, new_k, new_v

    with (
        mock.patch.object(
            provider, "_validate_cold_prefill_inputs", return_value=object()
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx._mlx_cold_prefill_graph",
            return_value=graph_result,
        ) as cold_graph,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.mlx_call_multi",
            side_effect=bridge,
        ) as mlx_bridge,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "qwen3_commit_deferred_kv"
        ) as commit,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
    ):
        result = provider.forward(
            SimpleNamespace(layers_to_capture=[]),
            torch.zeros(token_count, dtype=torch.int64),
            torch.arange(token_count, dtype=torch.int64),
            batch,
        )

    assert result is hidden
    mlx_bridge.assert_called_once()
    cold_graph.assert_called_once_with(views, "mlx ids", "mlx positions")
    commit.assert_called_once_with(
        new_k,
        new_v,
        slots,
        (layer.k_pool.torch_tensor,),
        (layer.v_pool.torch_tensor,),
        backend=KernelBackend.METAL_JIT,
    )
    assert provider._pending_commit_sources == (new_k, new_v)
    assert provider.call_count == 1
    assert provider.prefill_call_count == 1
    assert provider.decode_call_count == 0
    assert provider.greedy_tail_torch_call_count == 1
    assert provider.greedy_tail_fallback_count == 1
    assert provider.last_greedy_tail_fallback_reason == "sampling_observer"


def test_decode_forward_retires_old_sources_at_bridge_fence():
    hidden = torch.empty(1, 1024)
    new_k = torch.empty(28, 1, 8, 128)
    new_v = torch.empty_like(new_k)
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    views = SimpleNamespace(layers=(layer,), pool_slots=2048)
    provider = Qwen3MlxModelProvider(
        views=views,
        started=True,
        kv_commit_backend=KernelBackend.METAL_JIT,
    )
    provider._pending_commit_sources = (object(), object())
    req_to_token = torch.empty(2, 2048, dtype=torch.int32)
    out_cache_loc = torch.tensor([7], dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        out_cache_loc=out_cache_loc,
    )
    graph_result = object()

    def bridge(operation, *inputs, device, after_mps_fence):
        assert after_mps_fence is not None
        after_mps_fence()
        assert provider._pending_commit_sources is None
        assert operation(*("mlx input" for _ in inputs)) is graph_result
        assert device == inputs[0].device
        return hidden, new_k, new_v

    with (
        mock.patch.object(
            provider,
            "_validate_decode_inputs",
            return_value=(object(), object(), req_to_token),
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx._mlx_decode_graph",
            return_value=graph_result,
        ) as decode_graph,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.mlx_call_multi",
            side_effect=bridge,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "qwen3_commit_deferred_kv"
        ) as commit,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_in_closed_range"),
    ):
        result = provider.forward(
            SimpleNamespace(layers_to_capture=[]),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([7], dtype=torch.int64),
            batch,
        )

    assert result is hidden
    decode_graph.assert_called_once_with(
        views,
        "mlx input",
        "mlx input",
        "mlx input",
        "mlx input",
        "mlx input",
    )
    commit.assert_called_once_with(
        new_k,
        new_v,
        out_cache_loc,
        (layer.k_pool.torch_tensor,),
        (layer.v_pool.torch_tensor,),
        backend=KernelBackend.METAL_JIT,
    )
    assert provider._pending_commit_sources == (new_k, new_v)
    assert provider.call_count == 1
    assert provider.prefill_call_count == 0
    assert provider.decode_call_count == 1
    assert provider.greedy_tail_torch_call_count == 1
    assert provider.greedy_tail_fallback_count == 0


def test_decode_greedy_tail_returns_generic_precomputed_token_output():
    token_ids = torch.tensor([9], dtype=torch.int64)
    new_k = torch.empty(28, 1, 8, 128)
    new_v = torch.empty_like(new_k)
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    views = SimpleNamespace(layers=(layer,), pool_slots=2048)
    provider = Qwen3MlxModelProvider(
        views=views,
        started=True,
        greedy_tail_backend="mlx",
        kv_commit_backend=KernelBackend.METAL_JIT,
    )
    req_to_token = torch.empty(2, 2048, dtype=torch.int32)
    out_cache_loc = torch.tensor([7], dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        out_cache_loc=out_cache_loc,
    )
    graph_result = object()

    def bridge(operation, *inputs, device, after_mps_fence):
        assert operation(*("mlx input" for _ in inputs)) is graph_result
        assert device == inputs[0].device
        assert after_mps_fence is None
        return token_ids, new_k, new_v

    with (
        mock.patch.object(
            provider,
            "_validate_decode_inputs",
            return_value=(object(), object(), req_to_token),
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch.object(provider, "_greedy_tail_fallback_reason", return_value=None),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "_mlx_decode_greedy_graph",
            return_value=graph_result,
        ) as greedy_graph,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.mlx_call_multi",
            side_effect=bridge,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "qwen3_commit_deferred_kv"
        ) as commit,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_in_closed_range"),
    ):
        result = provider.forward(
            SimpleNamespace(layers_to_capture=[]),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([7], dtype=torch.int64),
            batch,
        )

    assert isinstance(result, LogitsProcessorOutput)
    assert result.next_token_logits is None
    assert result.precomputed_greedy_token_ids is token_ids
    greedy_graph.assert_called_once()
    commit.assert_called_once()
    assert provider.greedy_tail_call_count == 1
    assert provider.greedy_tail_fallback_count == 0
    assert provider.last_greedy_tail_fallback_reason is None


@pytest.mark.parametrize("fence_succeeds", [False, True])
def test_bridge_failure_retires_old_sources_only_after_successful_fence(
    fence_succeeds,
):
    token_count = QWEN3_COLD_PREFILL_MIN_TOKENS
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    provider = Qwen3MlxModelProvider(
        views=SimpleNamespace(layers=(layer,), pool_slots=2048), started=True
    )
    old_sources = (object(), object())
    provider._pending_commit_sources = old_sources
    batch = _cold_batch(
        token_count,
        out_cache_loc=torch.arange(token_count, dtype=torch.int64),
    )

    def failing_bridge(*_args, after_mps_fence, **_kwargs):
        if fence_succeeds:
            after_mps_fence()
            assert provider._pending_commit_sources is None
        raise RuntimeError("bridge failed")

    with (
        mock.patch.object(
            provider, "_validate_cold_prefill_inputs", return_value=object()
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.mlx_call_multi",
            side_effect=failing_bridge,
        ),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        pytest.raises(RuntimeError, match="bridge failed"),
    ):
        provider.forward(
            SimpleNamespace(layers_to_capture=[]),
            torch.zeros(token_count, dtype=torch.int64),
            torch.arange(token_count, dtype=torch.int64),
            batch,
        )

    expected = None if fence_succeeds else old_sources
    assert provider._pending_commit_sources == expected


def test_commit_failure_retains_exported_sources_and_does_not_count_call():
    token_count = QWEN3_COLD_PREFILL_MIN_TOKENS
    new_k = torch.empty(28, token_count, 8, 128)
    new_v = torch.empty_like(new_k)
    layer = SimpleNamespace(
        rope_cache=SimpleNamespace(torch_tensor=torch.empty(2048, 128)),
        k_pool=SimpleNamespace(torch_tensor=object()),
        v_pool=SimpleNamespace(torch_tensor=object()),
    )
    provider = Qwen3MlxModelProvider(
        views=SimpleNamespace(layers=(layer,), pool_slots=2048), started=True
    )
    batch = _cold_batch(
        token_count,
        out_cache_loc=torch.arange(token_count, dtype=torch.int64),
    )
    phase_events = []

    with (
        mock.patch.object(
            provider, "_validate_cold_prefill_inputs", return_value=object()
        ),
        mock.patch.object(provider, "_validate_pool_storage"),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.mlx_call_multi",
            return_value=(torch.empty(token_count, 1024), new_k, new_v),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "qwen3_commit_deferred_kv",
            side_effect=RuntimeError("second launch failed"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.current_phase_recorder",
            return_value=lambda name, duration: phase_events.append((name, duration)),
        ),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        pytest.raises(RuntimeError, match="second launch failed"),
    ):
        provider.forward(
            SimpleNamespace(layers_to_capture=[]),
            torch.zeros(token_count, dtype=torch.int64),
            torch.arange(token_count, dtype=torch.int64),
            batch,
        )

    assert provider._pending_commit_sources == (new_k, new_v)
    assert [name for name, _ in phase_events] == ["kv_commit_submit"]
    assert all(duration >= 0.0 for _, duration in phase_events)
    assert provider.call_count == 0
    assert provider.prefill_call_count == 0
    assert provider.decode_call_count == 0


def test_provider_start_warms_owned_pipelines_before_serving():
    req_pool = object()
    req_to_token = mock.MagicMock()
    req_to_token.data_ptr.return_value = 123
    req_to_token.shape = (8, 32)
    provider = Qwen3MlxModelProvider(
        views=SimpleNamespace(pool_slots=123),
        kv_commit_backend=KernelBackend.METAL_JIT,
    )

    with (
        mock.patch.object(
            provider, "_resolve_req_to_token", return_value=req_to_token
        ) as resolve,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx._configure_mlx_memory_cache"
        ) as configure_cache,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.warmup_qwen3_kv_commit"
        ) as warmup_commit,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.warmup_qwen3_radix_decode_deferred"
        ) as warmup_attention,
        mock.patch.object(provider, "enable_compiled_decode") as enable_compile,
        mock.patch.object(provider, "warmup_compiled_decode") as warmup_compile,
    ):
        provider.start(req_pool)

    resolve.assert_called_once_with(req_pool)
    configure_cache.assert_called_once_with()
    warmup_commit.assert_called_once_with(pool_slots=123)
    warmup_attention.assert_called_once_with(
        request_rows=8,
        table_stride=32,
        pool_slots=123,
    )
    enable_compile.assert_called_once_with()
    warmup_compile.assert_called_once_with(req_to_token)
    assert provider.started
    assert provider.req_pool_identity == id(req_pool)
    assert provider.req_to_token_data_ptr == 123
    assert provider.req_to_token_shape == (8, 32)


def test_create_returns_started_provider_without_publishing():
    causal_lm = _empty_module(Qwen3ForCausalLM)
    qwen3_model = _empty_module(Qwen3Model)
    qwen3_model.model_forward_provider = None
    causal_lm.model = qwen3_model
    views = SimpleNamespace(pool_slots=123)
    req_pool = object()

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx._build_views",
            return_value=views,
        ) as build,
        mock.patch.object(Qwen3MlxModelProvider, "start") as start,
    ):
        provider = create_qwen3_mlx_model_provider(causal_lm, object(), req_pool)

    build.assert_called_once()
    start.assert_called_once_with(req_pool)
    assert qwen3_model.model_forward_provider is None
    assert provider.views is views
    assert provider.greedy_tail_backend == "torch"
    assert (
        provider.greedy_tail_static_fallback_reason
        == "disabled by SGLANG_MPS_QWEN3_GREEDY_TAIL priority"
    )


def test_create_failure_does_not_publish_provider():
    causal_lm = _empty_module(Qwen3ForCausalLM)
    qwen3_model = _empty_module(Qwen3Model)
    qwen3_model.model_forward_provider = None
    causal_lm.model = qwen3_model
    req_pool = object()

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx._build_views",
            return_value=SimpleNamespace(pool_slots=123),
        ),
        mock.patch.object(
            Qwen3MlxModelProvider,
            "start",
            side_effect=RuntimeError("compile failed"),
        ),
        mock.patch.object(Qwen3MlxModelProvider, "close") as close,
        pytest.raises(RuntimeError, match="compile failed"),
    ):
        create_qwen3_mlx_model_provider(causal_lm, object(), req_pool)

    close.assert_called_once_with()
    assert qwen3_model.model_forward_provider is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
