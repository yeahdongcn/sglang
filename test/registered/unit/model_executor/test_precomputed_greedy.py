import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    require_graph_compatible_logits_output,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    precomputed_greedy_fallback_reason,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _sampling_info(**overrides):
    values = dict(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        custom_logit_processor=None,
        logit_bias=None,
        grammars=None,
        grammar_mask=None,
        return_sampling_masks=[False],
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        device="cpu",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(**overrides):
    values = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        contains_last_prefill_chunk=True,
        is_prefill_only=False,
        return_pooled_hidden_states=False,
        return_logprob=False,
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
        multi_item_delimiter_indices=None,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
        sampling_observer_active=False,
        sampling_info=_sampling_info(),
        positions=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _without_nan_diagnostics():
    return (
        mock.patch.object(envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=False),
        mock.patch.object(envs.SGLANG_SANITIZE_NAN_LOGITS, "get", return_value=False),
    )


def test_precomputed_greedy_contract_is_strict_and_sync_free():
    async_assert, sanitize = _without_nan_diagnostics()
    with async_assert, sanitize:
        assert precomputed_greedy_fallback_reason(_batch()) is None

        cases = (
            (
                _batch(
                    forward_mode=ForwardMode.EXTEND,
                    contains_last_prefill_chunk=False,
                ),
                "incomplete_prefill_chunk",
            ),
            (_batch(is_prefill_only=True), "prefill_only"),
            (_batch(return_pooled_hidden_states=True), "pooled_hidden_states"),
            (_batch(return_logprob=True), "return_logprob"),
            (_batch(top_logprobs_nums=[1]), "top_logprobs"),
            (_batch(token_ids_logprobs=[[7]]), "token_ids_logprobs"),
            (
                _batch(multi_item_delimiter_indices=torch.tensor([0])),
                "multi_item_scoring",
            ),
            (_batch(spec_info=object()), "speculative_decoding"),
            (
                _batch(capture_hidden_mode=CaptureHiddenMode.LAST),
                "hidden_capture",
            ),
            (
                _batch(return_hidden_states_before_norm=True),
                "pre_norm_hidden_capture",
            ),
            (
                _batch(sampling_info=_sampling_info(is_all_greedy=False)),
                "non_greedy_sampling",
            ),
            (
                _batch(sampling_info=_sampling_info(has_custom_logit_processor=True)),
                "custom_logit_processor",
            ),
            (
                _batch(
                    sampling_info=_sampling_info(custom_logit_processor={1: object()})
                ),
                "custom_logit_processor",
            ),
            (
                _batch(sampling_info=_sampling_info(logit_bias=torch.zeros(1))),
                "logit_bias",
            ),
            (
                _batch(sampling_info=_sampling_info(grammars=[object()])),
                "grammar",
            ),
            (
                _batch(sampling_info=_sampling_info(grammar_mask=object())),
                "grammar_mask",
            ),
            (
                _batch(sampling_info=_sampling_info(return_sampling_masks=[True])),
                "sampling_mask",
            ),
            (
                _batch(
                    sampling_info=_sampling_info(acc_additive_penalties=torch.zeros(1))
                ),
                "additive_penalties",
            ),
            (
                _batch(
                    sampling_info=_sampling_info(acc_scaling_penalties=torch.ones(1))
                ),
                "scaling_penalties",
            ),
            (
                _batch(
                    sampling_info=_sampling_info(
                        penalizer_orchestrator=SimpleNamespace(is_required=True)
                    )
                ),
                "penalties",
            ),
        )
        for batch, expected in cases:
            assert precomputed_greedy_fallback_reason(batch) == expected

        unresolved = _batch(sampling_observer_active=None)
        assert (
            precomputed_greedy_fallback_reason(unresolved) == "sampling_observer_state"
        )
        assert (
            precomputed_greedy_fallback_reason(
                unresolved,
                sampling_observer=SimpleNamespace(is_active=lambda _: True),
            )
            == "sampling_observer"
        )

    with (
        mock.patch.object(envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=True),
        mock.patch.object(envs.SGLANG_SANITIZE_NAN_LOGITS, "get", return_value=False),
    ):
        assert precomputed_greedy_fallback_reason(_batch()) == "async_assert"

    with (
        mock.patch.object(envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=False),
        mock.patch.object(envs.SGLANG_SANITIZE_NAN_LOGITS, "get", return_value=True),
    ):
        assert precomputed_greedy_fallback_reason(_batch()) == "nan_sanitization"


def test_logits_output_rejects_tokens_together_with_logits():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 4),
            precomputed_greedy_token_ids=torch.zeros(1, dtype=torch.int64),
        )


def test_precomputed_greedy_payload_is_explicitly_eager_only():
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=torch.tensor([1], dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="static graph buffers are borrowed"):
        require_graph_compatible_logits_output(output, "TestGraphRunner")

    require_graph_compatible_logits_output(
        LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4)),
        "TestGraphRunner",
    )


def test_cpu_graph_exact_capture_rejects_precomputed_greedy_payload():
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=torch.tensor([1], dtype=torch.int64),
    )
    graph = mock.Mock(return_value=output)
    prepared_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
    )
    runner = object.__new__(CPUGraphRunner)
    runner.is_encoder_decoder = False
    runner.graphs = {1: graph}
    runner.graphs_cross = {1: graph}
    runner._get_skip_cross_attention = mock.Mock(return_value=False)
    runner.prepare_replay = mock.Mock(return_value=prepared_batch)

    with pytest.raises(RuntimeError, match="static graph buffers are borrowed"):
        CPUGraphRunner.execute(runner, SimpleNamespace(batch_size=1))


def test_model_runner_fast_path_keeps_finalizer_and_ngram_hook():
    runner = object.__new__(ModelRunner)
    runner.sampler = mock.MagicMock()
    runner.ngram_embedding_manager = mock.MagicMock()
    runner._preprocess_logits = mock.MagicMock()
    runner._sampling_observer = None
    token_ids = torch.tensor([7], dtype=torch.int64)
    runner.sampler.finalize_precomputed_greedy_token_ids.return_value = token_ids
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=token_ids,
    )
    batch = _batch()

    async_assert, sanitize = _without_nan_diagnostics()
    with async_assert, sanitize:
        result = ModelRunner.sample(runner, output, batch)

    assert result is token_ids
    assert output.precomputed_greedy_token_ids is None
    runner._preprocess_logits.assert_not_called()
    runner.sampler.assert_not_called()
    runner.sampler.finalize_precomputed_greedy_token_ids.assert_called_once_with(
        token_ids,
        batch.sampling_info,
        batch_size=1,
    )
    runner.ngram_embedding_manager.update_after_decode.assert_called_once_with(
        next_token_ids=token_ids,
        forward_batch=batch,
    )


def test_model_runner_rejects_an_ineligible_precomputed_output():
    runner = object.__new__(ModelRunner)
    runner.sampler = mock.MagicMock()
    runner.ngram_embedding_manager = mock.MagicMock()
    runner._preprocess_logits = mock.MagicMock()
    runner._sampling_observer = None
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=torch.tensor([7]),
    )

    async_assert, sanitize = _without_nan_diagnostics()
    with async_assert, sanitize, pytest.raises(RuntimeError, match="return_logprob"):
        ModelRunner.sample(runner, output, _batch(return_logprob=True))

    runner.sampler.finalize_precomputed_greedy_token_ids.assert_not_called()
    runner.ngram_embedding_manager.update_after_decode.assert_not_called()


@pytest.mark.parametrize("observer_active", [False, True])
def test_model_runner_resolves_unknown_observer_before_precomputed_fast_path(
    observer_active,
):
    observer = SimpleNamespace(is_active=mock.Mock(return_value=observer_active))
    runner = object.__new__(ModelRunner)
    runner._sampling_observer = observer
    runner.sampler = mock.MagicMock()
    runner.ngram_embedding_manager = mock.MagicMock()
    token_ids = torch.tensor([7], dtype=torch.int64)
    runner.sampler.finalize_precomputed_greedy_token_ids.return_value = token_ids
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=token_ids,
    )
    batch = _batch(sampling_observer_active=None)

    async_assert, sanitize = _without_nan_diagnostics()
    with async_assert, sanitize:
        if observer_active:
            with pytest.raises(RuntimeError, match="sampling_observer"):
                ModelRunner.sample(runner, output, batch)
        else:
            assert ModelRunner.sample(runner, output, batch) is token_ids

    observer.is_active.assert_called_once_with(batch.sampling_info)
    if observer_active:
        runner.sampler.finalize_precomputed_greedy_token_ids.assert_not_called()
        runner.ngram_embedding_manager.update_after_decode.assert_not_called()
    else:
        runner.sampler.finalize_precomputed_greedy_token_ids.assert_called_once()
        runner.ngram_embedding_manager.update_after_decode.assert_called_once()


def test_prepared_inactive_observer_keeps_precomputed_fast_path_without_recheck():
    observer = SimpleNamespace(is_active=mock.Mock(return_value=False))
    runner = object.__new__(ModelRunner)
    runner._sampling_observer = observer
    runner.sampler = mock.MagicMock()
    runner.ngram_embedding_manager = mock.MagicMock()
    token_ids = torch.tensor([7], dtype=torch.int64)
    runner.sampler.finalize_precomputed_greedy_token_ids.return_value = token_ids
    output = LogitsProcessorOutput(
        next_token_logits=None,
        precomputed_greedy_token_ids=token_ids,
    )
    batch = _batch(sampling_observer_active=None)

    ModelRunner._prepare_sampling_observer_state(runner, batch)
    async_assert, sanitize = _without_nan_diagnostics()
    with async_assert, sanitize:
        assert ModelRunner.sample(runner, output, batch) is token_ids

    assert batch.sampling_observer_active is False
    observer.is_active.assert_called_once_with(batch.sampling_info)
    runner.sampler.finalize_precomputed_greedy_token_ids.assert_called_once()


def test_standard_logits_sampling_path_is_unchanged():
    runner = object.__new__(ModelRunner)
    runner.sampler = mock.MagicMock(return_value=torch.tensor([3]))
    runner.ngram_embedding_manager = mock.MagicMock()
    runner._preprocess_logits = mock.MagicMock(return_value=None)
    runner._sampling_observer = None
    output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    batch = _batch()

    result = ModelRunner.sample(runner, output, batch)

    runner._preprocess_logits.assert_called_once_with(output, batch.sampling_info)
    runner.sampler.assert_called_once()
    assert result.tolist() == [3]


def test_sampler_validates_and_finalizes_precomputed_tokens():
    sampler = object.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampling_info = _sampling_info()
    token_ids = torch.tensor([2, 4], dtype=torch.int64)

    with mock.patch.object(sampler, "_sync_token_ids_across_tp") as synchronize:
        assert (
            sampler.finalize_precomputed_greedy_token_ids(
                token_ids, sampling_info, batch_size=2
            )
            is token_ids
        )
    synchronize.assert_called_once_with(token_ids, sampling_info)

    with pytest.raises(RuntimeError, match="dtype"):
        sampler.finalize_precomputed_greedy_token_ids(
            token_ids.to(torch.int32), sampling_info, batch_size=2
        )
    with pytest.raises(RuntimeError, match="shape"):
        sampler.finalize_precomputed_greedy_token_ids(
            token_ids[:1], sampling_info, batch_size=2
        )


def test_sampler_skips_grammar_collective_for_single_tp_rank():
    sampler = object.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.tp_sync_world_size = 1
    sampler.tp_sync_group = object()
    sampling_info = SimpleNamespace(grammars=[object()])

    with mock.patch("torch.distributed.all_reduce") as all_reduce:
        sampler._sync_token_ids_across_tp(torch.tensor([3]), sampling_info)

    all_reduce.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
