import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.benchmark.mps_qwen3 import (
    _COUNTER_KEYS,
    _MLX_PHASE_TIMING_CONTRACT,
    PROFILES,
    _advance_worker_phase,
    _AvailableMemoryGuard,
    _build_worker_command,
    _classify_token_id_parity,
    _collect_stable_trials,
    _decode_tail_metrics,
    _expected_mlx_benchmark_variants,
    _first_token_id_mismatch,
    _install_sglang_mlx_benchmark_variant,
    _make_manifest,
    _memory_guard_failure_artifact,
    _memory_guard_policy,
    _mlx_cache_growth_events,
    _mlx_cache_snapshot,
    _mlx_decode_fence,
    _mlx_last_token_logits,
    _ParentMemoryGuardViolation,
    _parser,
    _prepare_profile_launch,
    _publish_worker_phase,
    _read_worker_phase,
    _require_complete_stable_trials,
    _run_profile,
    _run_sglang_forward_with_phase_timing,
    _run_sglang_worker,
    _select_reference_profile,
    _stable_trial_policy,
    _StableTrialExhausted,
    _state_delta,
    _sum_mlx_phase_samples,
    _summarize_trials,
    _swap_delta,
    _token_ids_are_strictly_comparable,
    _trial_swap_rejection,
    _validate_memory_guard_args,
    _validate_profile_swap,
    _validate_sglang_pool_capacities,
    _validate_sglang_provider_state,
    _validate_sglang_trace,
    _validate_swap_headroom,
    _validate_worker_diagnostics,
    _validate_worker_stable_trials,
    _worker_main,
    _worker_phase_path,
    _WorkerProfileError,
)
from sglang.srt.utils._phase_timing import current_phase_recorder
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")

_HAS_MLX = importlib.util.find_spec("mlx") is not None


def _state(**updates):
    state = {
        "qkv_kernel_backend": "torch",
        "decode_kernel_backend": "torch",
        "whole_model_backend": "torch",
        "whole_model_greedy_tail_backend": "off",
        "deferred_kv_commit_backend": "off",
        "patched_qkv_modules": 0,
        "patched_decode_modules": 0,
        "generic_kernel_backends": {
            "rmsnorm": "torch",
            "fused_add_rmsnorm": "torch",
            "silu_and_mul": "torch",
        },
    }
    state.update({key: 0 for key in _COUNTER_KEYS})
    state.update(updates)
    return state


def _trial_args(**updates):
    values = {
        "trials": 2,
        "max_trial_attempts": 4,
        "max_trial_swap_in_mb": 64.0,
        "max_trial_swap_out_mb": 16.0,
        "max_trial_swap_growth_mb": 128.0,
        "max_profile_swap_out_mb": 64.0,
        "max_profile_swap_growth_mb": 512.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _benchmark_trial(value, *, swap_in=0, swap_out=0, swap_growth=0):
    return {
        "request_setup_s": value / 10,
        "prefill_s": value,
        "decode_s": value * 2,
        "total_s": value * 3,
        "request_total_s": value * 3.1,
        "prefill_tps": 100 / value,
        "decode_tps": 10 / value,
        "system_swap_delta": {
            "in": swap_in,
            "out": swap_out,
            "used_growth": swap_growth,
        },
    }


class TestQwen3MpsBenchmark(unittest.TestCase):
    def test_mlx_phase_samples_are_added_by_name(self):
        self.assertEqual(
            _sum_mlx_phase_samples(
                [
                    {"producer_fence": 0.25, "graph_build": 1.0},
                    {"producer_fence": 0.5, "dlpack_export": 0.125},
                ]
            ),
            {
                "dlpack_export": 0.125,
                "graph_build": 1.0,
                "producer_fence": 0.75,
            },
        )

    def test_forward_phase_scope_is_opt_in_and_resets_after_call(self):
        runner = mock.Mock()
        runner.forward.return_value = ("token", "aux")

        result, phases = _run_sglang_forward_with_phase_timing(
            runner,
            object(),
            enabled=False,
        )
        self.assertEqual(result, ("token", "aux"))
        self.assertEqual(phases, {})
        runner.forward.assert_called_once()

    def test_forward_phase_scope_is_active_only_during_instrumented_call(self):
        recorder_seen = []
        runner = mock.Mock()

        def forward(_batch):
            recorder = current_phase_recorder()
            recorder_seen.append(recorder is not None)
            assert recorder is not None
            recorder("inside_forward", 0.25)
            return "result"

        runner.forward.side_effect = forward
        result, phases = _run_sglang_forward_with_phase_timing(
            runner,
            object(),
            enabled=True,
        )

        self.assertEqual(result, "result")
        self.assertEqual(phases, {"inside_forward": 0.25})
        self.assertEqual(recorder_seen, [True])
        self.assertIsNone(current_phase_recorder())

    def test_decode_tail_excludes_exactly_the_first_synchronized_step(self):
        self.assertEqual(_decode_tail_metrics([]), {})
        self.assertEqual(_decode_tail_metrics([0.25]), {})
        self.assertEqual(
            _decode_tail_metrics([0.25, 0.5, 1.0]),
            {
                "decode_tail_s": 1.5,
                "decode_tail_tps": 2 / 1.5,
                "decode_tail_steps": 2,
                "decode_tail_excluded_steps": 1,
            },
        )

    def test_mlx_decode_fence_count_matches_harness_mode(self):
        for sync_mode, expected in (("each", 3), ("aggregate", 1)):
            mlx = mock.MagicMock()
            stream = object()
            for _ in range(3):
                _mlx_decode_fence(mlx, stream, sync_mode, final=False)
            _mlx_decode_fence(mlx, stream, sync_mode, final=True)

            self.assertEqual(mlx.synchronize.call_count, expected)
            mlx.synchronize.assert_called_with(stream)

    def test_mlx_cache_snapshot_and_growth_are_strict_and_outside_timing(self):
        KVCache = type("KVCache", (), {})

        def cache(offset, capacity, *, value_capacity=None):
            item = KVCache()
            item.offset = offset
            item.step = 256
            item.keys = SimpleNamespace(shape=(1, 8, capacity, 128), dtype="bfloat16")
            item.values = SimpleNamespace(
                shape=(1, 8, value_capacity or capacity, 128), dtype="bfloat16"
            )
            return item

        prefill = _mlx_cache_snapshot([cache(512, 512) for _ in range(28)])
        final = _mlx_cache_snapshot([cache(639, 768) for _ in range(28)])
        events = _mlx_cache_growth_events(prefill, final, decode_steps=127)

        self.assertEqual(prefill[0]["key_shape"], (1, 8, 512, 128))
        self.assertEqual(prefill[0]["value_shape"], (1, 8, 512, 128))
        self.assertEqual(
            events,
            [
                {
                    "decode_index": 0,
                    "decode_step": 1,
                    "offset_before": 512,
                    "offset_after": 513,
                    "capacity_before": 512,
                    "capacity_after": 768,
                    "grew_layers": list(range(28)),
                    "source": "inferred_from_prefill_final_and_step",
                }
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "mismatched K/V"):
            _mlx_cache_snapshot([cache(512, 512, value_capacity=768)])
        one_sided = cache(0, 512)
        one_sided.values = None
        with self.assertRaisesRegex(RuntimeError, "only one of K/V"):
            _mlx_cache_snapshot([one_sided])
        wrong_rank = cache(512, 512)
        wrong_rank.keys.shape = (1, 512, 128)
        wrong_rank.values.shape = (1, 512, 128)
        with self.assertRaisesRegex(RuntimeError, "not rank-4"):
            _mlx_cache_snapshot([wrong_rank])
        invalid = cache(513, 512)
        with self.assertRaisesRegex(RuntimeError, "exceeds capacity"):
            _mlx_cache_snapshot([invalid])

    def test_sglang_pool_contract_requires_equal_adequate_actual_capacity(self):
        manifest = {"input_len": 512, "output_len": 33}
        results = [
            {
                "engine": "sglang",
                "profile": "sglang-torch",
                "max_total_num_tokens": 1024,
            },
            {
                "engine": "sglang",
                "profile": "sglang-whole-mlx",
                "max_total_num_tokens": 1024,
            },
            {"engine": "mlx_lm_core", "profile": "mlx-lm-model-core"},
        ]

        self.assertEqual(
            _validate_sglang_pool_capacities(results, manifest),
            {
                "required_tokens": 544,
                "capacity_by_profile": {
                    "sglang-torch": 1024,
                    "sglang-whole-mlx": 1024,
                },
                "common_capacity": 1024,
                "consistent": True,
            },
        )
        unequal = json.loads(json.dumps(results))
        unequal[1]["max_total_num_tokens"] = 2048
        with self.assertRaisesRegex(RuntimeError, "different actual KV-pool"):
            _validate_sglang_pool_capacities(unequal, manifest)
        too_small = json.loads(json.dumps(results[:1]))
        too_small[0]["max_total_num_tokens"] = 543
        with self.assertRaisesRegex(RuntimeError, "fewer than"):
            _validate_sglang_pool_capacities(too_small, manifest)

    def test_torch_reference_is_required_for_validated_parity(self):
        native = {"profile": "mlx-lm-model-core", "engine": "mlx_lm_core"}
        whole = {"profile": "sglang-whole-mlx", "engine": "sglang"}
        torch = {"profile": "sglang-torch", "engine": "sglang"}

        self.assertEqual(
            _select_reference_profile([native, whole]),
            ("mlx-lm-model-core", False),
        )
        self.assertEqual(
            _select_reference_profile([native, torch, whole]),
            ("sglang-torch", True),
        )

    def test_native_mlx_profiles_keep_matched_and_public_contracts_separate(self):
        self.assertEqual(PROFILES["mlx-lm-model-core"].engine, "mlx_lm_core")
        self.assertEqual(PROFILES["mlx-lm-public"].engine, "mlx_lm_public")
        self.assertTrue(_token_ids_are_strictly_comparable({"engine": "mlx_lm_core"}))
        self.assertFalse(
            _token_ids_are_strictly_comparable({"engine": "mlx_lm_public"})
        )

    def test_final_parity_classification_fails_closed_on_provider_mismatch(self):
        torch = {
            "profile": "sglang-torch",
            "engine": "sglang",
            "reference_output_ids": [198, 522],
        }
        whole_mlx = {
            "profile": "sglang-whole-mlx",
            "engine": "sglang",
            "reference_output_ids": [198, 197],
        }
        public_mlx = {
            "profile": "mlx-lm-public",
            "engine": "mlx_lm_public",
            "reference_output_ids": [198, 197],
        }

        parity = _classify_token_id_parity([torch, whole_mlx, public_mlx])

        self.assertEqual(parity["status"], "invalid_token_id_mismatch")
        self.assertTrue(parity["reference_validated"])
        self.assertEqual(parity["reference_profile"], "sglang-torch")
        self.assertEqual(
            parity["token_id_mismatches"],
            {"sglang-whole-mlx": {"index": 1, "expected": 522, "actual": 197}},
        )
        self.assertTrue(torch["token_ids_match_reference"])
        self.assertFalse(whole_mlx["token_ids_match_reference"])
        self.assertFalse(public_mlx["token_ids_comparable_to_reference"])
        self.assertIsNone(public_mlx["token_ids_match_reference"])

    def test_native_mlx_projects_only_the_last_prompt_hidden_state(self):
        hidden = mock.MagicMock()
        last_hidden = object()
        hidden.__getitem__.return_value = last_hidden
        core = mock.MagicMock(return_value=hidden)
        embedding = mock.MagicMock()
        logits = object()
        embedding.as_linear.return_value = logits
        model = SimpleNamespace(
            args=SimpleNamespace(tie_word_embeddings=True),
            model=core,
        )
        model.model.embed_tokens = embedding
        inputs = object()
        cache = object()

        self.assertIs(_mlx_last_token_logits(model, inputs, cache), logits)
        core.assert_called_once_with(inputs, cache=cache)
        hidden.__getitem__.assert_called_once_with(
            (slice(None), slice(-1, None), slice(None))
        )
        embedding.as_linear.assert_called_once_with(last_hidden)

    def test_manifest_is_literal_reproducible_and_uses_full_vocabulary(self):
        config = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "config.json").write_text(json.dumps(config))
            (model_path / "model.safetensors").write_bytes(b"weights")

            first = _make_manifest(
                model_path,
                input_len=64,
                output_len=8,
                seed=2026,
            )
            second = _make_manifest(
                model_path,
                input_len=64,
                output_len=8,
                seed=2026,
            )

        self.assertEqual(first["prompt_token_ids"], second["prompt_token_ids"])
        self.assertEqual(len(first["prompt_token_ids"]), 64)
        self.assertTrue(all(0 <= token < 151936 for token in first["prompt_token_ids"]))
        self.assertTrue(any(token >= 10000 for token in first["prompt_token_ids"]))
        self.assertEqual(len(first["weight_files"]), 1)

    def test_manifest_rejects_multi_shard_checkpoint_for_matched_loader(self):
        config = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "config.json").write_text(json.dumps(config))
            (model_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (model_path / "model-00002-of-00002.safetensors").write_bytes(b"two")

            with self.assertRaisesRegex(ValueError, "exactly one"):
                _make_manifest(
                    model_path,
                    input_len=64,
                    output_len=8,
                    seed=2026,
                )

    def test_summary_keeps_raw_trials_and_uses_interpolated_percentiles(self):
        trials = [
            {
                "request_setup_s": value / 10,
                "prefill_s": value,
                "decode_s": value * 2,
                "total_s": value * 3,
                "request_total_s": value * 3.1,
                "prefill_tps": 100 / value,
                "decode_tps": 10 / value,
            }
            for value in (1.0, 2.0, 3.0)
        ]

        summary = _summarize_trials(trials)

        self.assertEqual(summary["prefill_s"]["median"], 2.0)
        self.assertEqual(summary["prefill_s"]["p10"], 1.2)
        self.assertEqual(summary["prefill_s"]["p90"], 2.8)
        self.assertEqual(summary["prefill_s"]["values"], [1.0, 2.0, 3.0])
        public_trials = [
            {**trial, "public_generation_tps": 20.0 + index}
            for index, trial in enumerate(trials)
        ]
        public_summary = _summarize_trials(public_trials)
        self.assertEqual(public_summary["public_generation_tps"]["median"], 21.0)
        with self.assertRaisesRegex(ValueError, "missing from some trials"):
            _summarize_trials(
                [public_trials[0], {**public_trials[1], "public_generation_tps": 21.0}]
                + [
                    {
                        key: value
                        for key, value in public_trials[2].items()
                        if key != "public_generation_tps"
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "missing summary metrics"):
            _summarize_trials([{"prefill_s": 1.0}])

    def test_attention_profile_requires_real_calls_without_fallback(self):
        before = _state(
            qkv_kernel_backend="metal_aot",
            decode_kernel_backend="metal_aot",
            patched_qkv_modules=28,
            patched_decode_modules=28,
        )
        after = _state(
            qkv_kernel_backend="metal_aot",
            decode_kernel_backend="metal_aot",
            patched_qkv_modules=28,
            patched_decode_modules=28,
            attention_qkv_call_count=84,
            attention_decode_call_count=56,
        )

        delta = _validate_sglang_provider_state(
            "sglang-aot-aot",
            before,
            after,
            output_len=3,
        )

        self.assertEqual(delta["attention_qkv_call_count"], 84)
        self.assertEqual(delta["attention_decode_call_count"], 56)

        after["attention_decode_fallback_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "runtime fallback"):
            _validate_sglang_provider_state(
                "sglang-aot-aot",
                before,
                after,
                output_len=3,
            )

    def test_whole_model_torch_tail_profile_counts_without_greedy_graph(self):
        before = _state(
            whole_model_backend="mlx",
            whole_model_greedy_tail_backend="torch",
            deferred_kv_commit_backend="torch",
        )
        after = _state(
            whole_model_backend="mlx",
            whole_model_greedy_tail_backend="torch",
            deferred_kv_commit_backend="torch",
            whole_model_call_count=5,
            whole_model_prefill_call_count=1,
            whole_model_decode_call_count=4,
            whole_model_selector_call_count=5,
            whole_model_compile_call_count=4,
            whole_model_compile_total_call_count=4,
            whole_model_greedy_tail_torch_call_count=5,
        )

        delta = _validate_sglang_provider_state(
            "sglang-whole-mlx",
            before,
            after,
            output_len=5,
        )

        self.assertEqual(delta["whole_model_prefill_call_count"], 1)
        self.assertEqual(delta["whole_model_decode_call_count"], 4)
        self.assertEqual(delta["whole_model_greedy_tail_call_count"], 0)
        self.assertEqual(delta["whole_model_greedy_tail_fallback_count"], 0)

    def test_whole_model_greedy_profile_counts_exact_fast_path(self):
        before = _state(
            whole_model_backend="mlx",
            whole_model_greedy_tail_backend="mlx",
            deferred_kv_commit_backend="torch",
        )
        after = _state(
            whole_model_backend="mlx",
            whole_model_greedy_tail_backend="mlx",
            deferred_kv_commit_backend="torch",
            whole_model_call_count=5,
            whole_model_prefill_call_count=1,
            whole_model_decode_call_count=4,
            whole_model_selector_call_count=5,
            whole_model_compile_total_call_count=4,
            whole_model_greedy_tail_call_count=5,
            whole_model_greedy_compile_call_count=4,
        )

        delta = _validate_sglang_provider_state(
            "sglang-whole-mlx-greedy",
            before,
            after,
            output_len=5,
        )

        self.assertEqual(delta["whole_model_greedy_tail_call_count"], 5)
        self.assertEqual(delta["whole_model_greedy_compile_call_count"], 4)

    def test_whole_model_greedy_profile_differs_only_by_tail_gate(self):
        baseline = PROFILES["sglang-whole-mlx-metal-commit"].environment
        candidate = PROFILES["sglang-whole-mlx-greedy-metal-commit"].environment
        gate = "SGLANG_MPS_QWEN3_GREEDY_TAIL"

        self.assertEqual(baseline[gate], "torch")
        self.assertEqual(candidate[gate], "mlx,torch")
        self.assertEqual(
            {key: value for key, value in baseline.items() if key != gate},
            {key: value for key, value in candidate.items() if key != gate},
        )
        self.assertEqual(candidate["SGLANG_MPS_QWEN3_GREEDY_TAIL"], "mlx,torch")

    def test_best_profile_proves_each_requested_generic_provider_executed(self):
        generic_backends = {
            "rmsnorm": "metal_jit",
            "fused_add_rmsnorm": "metal_jit",
            "silu_and_mul": "metal_jit",
        }
        before = _state(
            qkv_kernel_backend="metal_aot",
            decode_kernel_backend="metal_jit",
            patched_qkv_modules=28,
            patched_decode_modules=28,
            generic_kernel_backends=generic_backends,
        )
        after = _state(
            qkv_kernel_backend="metal_aot",
            decode_kernel_backend="metal_jit",
            patched_qkv_modules=28,
            patched_decode_modules=28,
            generic_kernel_backends=generic_backends,
            attention_qkv_call_count=56,
            attention_decode_call_count=28,
        )
        _validate_sglang_provider_state(
            "sglang-best-metal",
            before,
            after,
            output_len=2,
        )
        after["attention_qkv_call_count"] = 55
        with self.assertRaisesRegex(RuntimeError, "QKV calls"):
            _validate_sglang_provider_state(
                "sglang-best-metal",
                before,
                after,
                output_len=2,
            )
        after["attention_qkv_call_count"] = 56
        trace = [
            {
                "op": "attention.qwen3_qknorm_rope_store",
                "backend": "metal_aot",
                "calls": 56,
            },
            {
                "op": "attention.qwen3_radix_decode",
                "backend": "metal_jit",
                "calls": 28,
            },
            {
                "op": "layernorm.rmsnorm",
                "backend": "metal_jit",
                "calls": 2,
            },
            {
                "op": "layernorm.fused_add_rmsnorm",
                "backend": "metal_jit",
                "calls": 112,
            },
            {
                "op": "activation.silu_and_mul",
                "backend": "metal_jit",
                "calls": 28,
            },
            {
                "op": "activation.silu_and_mul",
                "backend": "torch",
                "calls": 28,
            },
        ]
        _validate_sglang_trace(
            "sglang-best-metal",
            trace,
            input_len=512,
            output_len=2,
        )

        with self.assertRaisesRegex(RuntimeError, "did not match"):
            _validate_sglang_trace(
                "sglang-best-metal",
                trace[:1],
                input_len=512,
                output_len=2,
            )

        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _validate_sglang_trace(
                "sglang-best-metal",
                trace
                + [
                    {
                        "op": "attention.unplanned_provider",
                        "backend": "metal_jit",
                        "calls": 1,
                    }
                ],
                input_len=512,
                output_len=2,
            )

        _validate_sglang_trace(
            "sglang-whole-mlx-metal-commit",
            [
                {
                    "op": "kvcache.qwen3_deferred_kv_commit",
                    "backend": "metal_jit",
                    "calls": 2,
                }
            ],
            input_len=512,
            output_len=2,
        )

    def test_state_delta_and_token_id_mismatch_are_explicit(self):
        before = _state(attention_qkv_call_count=2)
        after = _state(attention_qkv_call_count=5)

        self.assertEqual(_state_delta(before, after)["attention_qkv_call_count"], 3)
        self.assertIsNone(_first_token_id_mismatch([1, 2], [1, 2]))
        self.assertEqual(
            _first_token_id_mismatch([1, 2], [1, 3]),
            {"index": 1, "expected": 2, "actual": 3},
        )
        self.assertEqual(
            _swap_delta(
                {"used": 20, "sin": 100, "sout": 50},
                {"used": 15, "sin": 140, "sout": 70},
            ),
            {"used_growth": 0, "in": 40, "out": 20},
        )
        self.assertEqual(
            _validate_profile_swap(
                "safe",
                {"used": 20, "sin": 100, "sout": 50},
                {"used": 25, "sin": 140, "sout": 70},
                max_growth_bytes=10,
                max_out_bytes=30,
            ),
            {"used_growth": 5, "in": 40, "out": 20},
        )
        with self.assertRaisesRegex(RuntimeError, "before it can continue thrashing"):
            _validate_profile_swap(
                "unsafe",
                {"used": 20, "sin": 100, "sout": 50},
                {"used": 25, "sin": 140, "sout": 81},
                max_growth_bytes=10,
                max_out_bytes=30,
            )
        _validate_swap_headroom(
            "no-swap",
            {"total": 0, "free": 0},
            minimum_free_bytes=256,
        )
        with self.assertRaisesRegex(RuntimeError, "free swap"):
            _validate_swap_headroom(
                "unsafe",
                {"total": 1024, "free": 255},
                minimum_free_bytes=256,
            )

        limits = _trial_args()
        self.assertIsNone(
            _trial_swap_rejection(
                {
                    "system_swap_delta": {
                        "in": 64 * 2**20,
                        "out": 16 * 2**20,
                        "used_growth": 128 * 2**20,
                    }
                },
                limits,
            )
        )
        self.assertEqual(
            _trial_swap_rejection(
                {
                    "system_swap_delta": {
                        "in": 65 * 2**20,
                        "out": 0,
                        "used_growth": 0,
                    }
                },
                limits,
            ),
            {
                "reason": "swap_threshold_exceeded",
                "exceeded": ["in"],
                "system_swap_delta": {
                    "in": 65 * 2**20,
                    "out": 0,
                    "used_growth": 0,
                },
                "swap_mib": {"in": 65.0, "out": 0.0, "used_growth": 0.0},
                "limits_mib": {"in": 64.0, "out": 16.0, "used_growth": 128.0},
            },
        )

    def test_trial_swap_telemetry_fails_closed(self):
        limits = _trial_args()
        for trial in (
            {},
            {"system_swap_delta": {}},
            {"system_swap_delta": {"in": -1, "out": 0, "used_growth": 0}},
            {"system_swap_delta": {"in": 0.0, "out": 0, "used_growth": 0}},
        ):
            rejection = _trial_swap_rejection(trial, limits)
            self.assertEqual(rejection["reason"], "invalid_swap_telemetry")

    def test_available_memory_guard_separates_setup_dip_from_sustained_pressure(
        self,
    ):
        gib = 2**30
        guard = _AvailableMemoryGuard(
            hard_min_bytes=2 * gib,
            soft_min_bytes=int(3.5 * gib),
            setup_grace_s=120,
            low_memory_sustain_s=5,
            recovery_margin_bytes=int(0.25 * gib),
            started_at=0,
        )

        self.assertIsNone(
            guard.observe(now=1, available_bytes=int(5.43 * gib), phase="setup")
        )
        self.assertIsNone(
            guard.observe(now=10, available_bytes=int(2.87 * gib), phase="setup")
        )
        self.assertIsNone(
            guard.observe(now=119, available_bytes=int(2.87 * gib), phase="setup")
        )
        # The setup grace expires at 120s; the soft floor still has to remain
        # violated for the independent five-second sustain window.
        self.assertIsNone(
            guard.observe(now=120, available_bytes=int(2.87 * gib), phase="setup")
        )
        violation = guard.observe(
            now=125,
            available_bytes=int(2.87 * gib),
            phase="setup",
        )
        self.assertEqual(violation["reason"], "sustained_unrecovered_memory_pressure")
        self.assertEqual(violation["pressure_duration_s"], 5)
        self.assertEqual(violation["phase"], "setup")

    def test_available_memory_guard_timing_hysteresis_and_hard_floor(self):
        gib = 2**30

        hard = _AvailableMemoryGuard(
            hard_min_bytes=2 * gib,
            soft_min_bytes=int(3.5 * gib),
            setup_grace_s=120,
            low_memory_sustain_s=5,
            recovery_margin_bytes=int(0.25 * gib),
            started_at=0,
        )
        violation = hard.observe(
            now=1,
            available_bytes=int(1.99 * gib),
            phase="setup",
        )
        self.assertEqual(violation["reason"], "hard_min_available")

        timing = _AvailableMemoryGuard(
            hard_min_bytes=2 * gib,
            soft_min_bytes=int(3.5 * gib),
            setup_grace_s=120,
            low_memory_sustain_s=5,
            recovery_margin_bytes=int(0.25 * gib),
            started_at=0,
        )
        self.assertIsNone(
            timing.observe(
                now=10,
                available_bytes=int(3.4 * gib),
                phase="timing",
            )
        )
        # Crossing the soft floor without the recovery margin keeps the same
        # unrecovered pressure episode latched, but cannot itself trigger a
        # failure while the current sample is above the floor.
        self.assertIsNone(
            timing.observe(
                now=13,
                available_bytes=int(3.6 * gib),
                phase="timing",
            )
        )
        violation = timing.observe(
            now=15,
            available_bytes=int(3.4 * gib),
            phase="timing",
        )
        self.assertEqual(violation["reason"], "sustained_unrecovered_memory_pressure")
        self.assertEqual(violation["pressure_duration_s"], 5)

        recovered = _AvailableMemoryGuard(
            hard_min_bytes=2 * gib,
            soft_min_bytes=int(3.5 * gib),
            setup_grace_s=120,
            low_memory_sustain_s=5,
            recovery_margin_bytes=int(0.25 * gib),
            started_at=0,
        )
        self.assertIsNone(
            recovered.observe(
                now=10,
                available_bytes=int(3.4 * gib),
                phase="timing",
            )
        )
        self.assertIsNone(
            recovered.observe(
                now=12,
                available_bytes=int(3.8 * gib),
                phase="timing",
            )
        )
        self.assertIsNone(
            recovered.observe(
                now=15,
                available_bytes=int(3.4 * gib),
                phase="setup",
            )
        )
        # Once timing has been observed, a stale/regressed setup sidecar cannot
        # reopen the 120-second setup grace.
        violation = recovered.observe(
            now=20,
            available_bytes=int(3.4 * gib),
            phase="setup",
        )
        self.assertEqual(violation["phase"], "timing")

    def test_worker_phase_sidecar_is_atomic_and_memory_artifact_is_parent_owned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "sglang-torch.json"
            phase_path = _worker_phase_path(result_path)
            args = SimpleNamespace(
                worker_phase=str(phase_path),
                worker_profile="sglang-torch",
                worker_engine="sglang",
            )

            self.assertEqual(_read_worker_phase(phase_path), "setup")
            _publish_worker_phase(args, "setup")
            self.assertEqual(_read_worker_phase(phase_path), "setup")
            _publish_worker_phase(args, "timing")
            self.assertEqual(_read_worker_phase(phase_path), "timing")
            self.assertEqual(_advance_worker_phase("setup", "setup"), "setup")
            self.assertEqual(_advance_worker_phase("setup", "timing"), "timing")
            self.assertEqual(_advance_worker_phase("timing", "setup"), "timing")
            self.assertEqual(list(root.glob(".*.tmp")), [])

            violation = _ParentMemoryGuardViolation(
                "hard floor",
                {
                    "reason": "hard_min_available",
                    "phase": "timing",
                    "available_bytes": 1,
                },
            )
            artifact = _memory_guard_failure_artifact(
                profile_name="sglang-torch",
                profile=PROFILES["sglang-torch"],
                violation=violation,
                policy={
                    "min_start_available_gib": 5.0,
                    "hard_min_available_gib": 2.0,
                    "soft_min_available_gib": 3.5,
                    "setup_grace_s": 120.0,
                    "low_memory_sustain_s": 5.0,
                    "recovery_margin_gib": 0.25,
                },
                # The nested violation is canonical if a stale sidecar regresses.
                phase="setup",
                elapsed_s=1.5,
                pid=123,
                rss_bytes=456,
                uss_bytes=400,
                available_before_bytes=1000,
                minimum_available_bytes=1,
                swap_before={"used": 0, "sin": 0, "sout": 0},
                swap_current={"used": 2, "sin": 3, "sout": 4},
                log_path=root / "worker.log",
                phase_path=phase_path,
                previous_worker_artifact=None,
            )
            self.assertEqual(artifact["status"], "terminated_by_memory_guard")
            self.assertEqual(artifact["phase"], "timing")
            self.assertEqual(artifact["memory_guard_violation"]["phase"], "timing")
            self.assertEqual(
                artifact["memory_guard_violation"]["reason"],
                "hard_min_available",
            )
            self.assertEqual(
                artifact["system_swap_delta"],
                {"used_growth": 2, "in": 3, "out": 4},
            )

    def test_memory_guard_policy_validation_and_metadata_are_complete(self):
        defaults = _parser().parse_args(["--model-path", "/unused/Qwen3-0.6B"])
        self.assertEqual(defaults.hard_min_available_gb, 2.0)
        self.assertEqual(defaults.min_available_gb, 2.5)
        self.assertEqual(defaults.min_start_available_gb, 5.0)
        self.assertEqual(defaults.setup_memory_grace_s, 120.0)
        self.assertEqual(defaults.low_memory_sustain_s, 5.0)
        self.assertEqual(defaults.available_recovery_margin_gb, 0.25)

        args = SimpleNamespace(
            hard_min_available_gb=2.0,
            min_available_gb=3.5,
            min_start_available_gb=5.0,
            setup_memory_grace_s=120.0,
            low_memory_sustain_s=5.0,
            available_recovery_margin_gb=0.25,
        )
        _validate_memory_guard_args(args)
        self.assertEqual(
            _memory_guard_policy(args),
            {
                "min_start_available_gib": 5.0,
                "hard_min_available_gib": 2.0,
                "soft_min_available_gib": 3.5,
                "setup_grace_s": 120.0,
                "low_memory_sustain_s": 5.0,
                "recovery_margin_gib": 0.25,
            },
        )

        args.hard_min_available_gb = 3.5
        with self.assertRaisesRegex(ValueError, "hard_min_available_gb"):
            _validate_memory_guard_args(args)
        args.hard_min_available_gb = 2.0
        args.min_start_available_gb = 3.0
        with self.assertRaisesRegex(ValueError, "min_start_available_gb"):
            _validate_memory_guard_args(args)

    def test_profile_launch_rechecks_memory_and_process_exclusivity(self):
        args = SimpleNamespace(
            min_start_available_gb=5.0,
            memory_recovery_timeout=30.0,
        )
        with (
            mock.patch(
                "sglang.benchmark.mps_qwen3._wait_for_available_memory"
            ) as wait_for_memory,
            mock.patch(
                "sglang.benchmark.mps_qwen3._running_benchmark_processes",
                side_effect=([], []),
            ) as running_processes,
        ):
            _prepare_profile_launch(args)

        wait_for_memory.assert_called_once_with(5 * 2**30, 30.0)
        self.assertEqual(running_processes.call_count, 2)

        with (
            mock.patch(
                "sglang.benchmark.mps_qwen3._wait_for_available_memory"
            ) as wait_for_memory,
            mock.patch(
                "sglang.benchmark.mps_qwen3._running_benchmark_processes",
                return_value=[{"pid": 123}],
            ),
            self.assertRaisesRegex(RuntimeError, "before profile launch"),
        ):
            _prepare_profile_launch(args)
        wait_for_memory.assert_not_called()

        with (
            mock.patch("sglang.benchmark.mps_qwen3._wait_for_available_memory"),
            mock.patch(
                "sglang.benchmark.mps_qwen3._running_benchmark_processes",
                side_effect=([], [{"pid": 456}]),
            ),
            self.assertRaisesRegex(RuntimeError, "before profile launch"),
        ):
            _prepare_profile_launch(args)

    def test_profile_cleans_process_group_when_monitor_setup_fails(self):
        args = SimpleNamespace(
            hard_min_available_gb=2.0,
            min_available_gb=3.5,
            setup_memory_grace_s=120.0,
            low_memory_sustain_s=5.0,
            available_recovery_margin_gb=0.25,
            min_swap_free_mb=512.0,
        )
        process = SimpleNamespace(pid=123)
        swap = {"total": 0, "used": 0, "free": 0, "sin": 0, "sout": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch(
                    "sglang.benchmark.mps_qwen3._build_worker_command",
                    return_value=["worker"],
                ),
                mock.patch(
                    "sglang.benchmark.mps_qwen3._memory_guard_policy",
                    return_value={},
                ),
                mock.patch(
                    "sglang.benchmark.mps_qwen3._swap_snapshot", return_value=swap
                ),
                mock.patch("sglang.benchmark.mps_qwen3._validate_swap_headroom"),
                mock.patch(
                    "sglang.benchmark.mps_qwen3.subprocess.Popen",
                    return_value=process,
                ) as popen,
                mock.patch(
                    "psutil.Process", side_effect=RuntimeError("monitor failed")
                ),
                mock.patch(
                    "sglang.benchmark.mps_qwen3._terminate_process_group"
                ) as terminate,
                self.assertRaisesRegex(_WorkerProfileError, "monitor failed"),
            ):
                _run_profile(
                    args,
                    "sglang-torch",
                    PROFILES["sglang-torch"],
                    root / "manifest.json",
                    root,
                )

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        terminate.assert_called_once_with(process)

    def test_stable_trial_collector_is_accepted_only_and_one_based(self):
        limits = _trial_args()
        contaminated_out = _benchmark_trial(1.0, swap_out=17 * 2**20)
        first = _benchmark_trial(2.0)
        contaminated_growth = _benchmark_trial(3.0, swap_growth=129 * 2**20)
        second = _benchmark_trial(4.0)
        candidates = iter((contaminated_out, first, contaminated_growth, second))

        collection = _collect_stable_trials(
            lambda: next(candidates),
            requested_trials=2,
            max_attempts=4,
            trial_delay=0,
            swap_limits=limits,
        )

        self.assertEqual(collection.trials, [first, second])
        self.assertEqual(collection.attempted, 4)
        self.assertEqual(collection.accepted_attempts, [2, 4])
        self.assertEqual(
            [item["attempt"] for item in collection.rejected_trials], [1, 3]
        )

        calls = 0

        def always_contaminated():
            nonlocal calls
            calls += 1
            return contaminated_out

        exhausted = _collect_stable_trials(
            always_contaminated,
            requested_trials=2,
            max_attempts=3,
            trial_delay=0,
            swap_limits=limits,
        )
        self.assertEqual(calls, 3)
        self.assertEqual(exhausted.attempted, 3)
        self.assertEqual(exhausted.trials, [])
        with self.assertRaises(_StableTrialExhausted) as raised:
            _require_complete_stable_trials(
                exhausted,
                args=limits,
                engine="sglang",
                profile="sglang-torch",
            )
        self.assertEqual(
            raised.exception.artifact["status"], "insufficient_stable_trials"
        )
        self.assertEqual(
            raised.exception.artifact["rejected_timed_trials"][0]["attempt"], 1
        )

    def test_parent_revalidates_counts_growth_and_accepted_only_summary(self):
        args = _trial_args()
        first = _benchmark_trial(1.0)
        second = _benchmark_trial(2.0)
        rejected_trial = _benchmark_trial(9.0, swap_out=17 * 2**20)
        rejection = _trial_swap_rejection(rejected_trial, args)
        result = {
            "status": "completed",
            "timed_trials": 2,
            "stable_trial_policy": _stable_trial_policy(args),
            "attempted_timed_trials": 3,
            "accepted_timed_attempts": [1, 3],
            "rejected_timed_trials": [{"attempt": 2, **rejection}],
            "trials": [first, second],
            "summary": _summarize_trials([first, second]),
        }
        _validate_worker_stable_trials(result, args)

        unstable = json.loads(json.dumps(result))
        unstable["trials"][0]["system_swap_delta"]["used_growth"] = 129 * 2**20
        with self.assertRaisesRegex(RuntimeError, "accepted trial"):
            _validate_worker_stable_trials(unstable, args)

        polluted = json.loads(json.dumps(result))
        polluted["summary"]["prefill_s"]["values"].append(9.0)
        with self.assertRaisesRegex(RuntimeError, "accepted trials only"):
            _validate_worker_stable_trials(polluted, args)

        inconsistent = json.loads(json.dumps(result))
        inconsistent["attempted_timed_trials"] = 4
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            _validate_worker_stable_trials(inconsistent, args)

        mismatched_policy = json.loads(json.dumps(result))
        mismatched_policy["stable_trial_policy"]["profile_hard_limits_mib"][
            "out"
        ] = 65.0
        with self.assertRaisesRegex(RuntimeError, "policy differs"):
            _validate_worker_stable_trials(mismatched_policy, args)

    def test_worker_diagnostic_artifact_is_fail_closed(self):
        self.assertEqual(
            _expected_mlx_benchmark_variants(
                "sglang-whole-mlx-fused-qkv-norm-metal-commit"
            ),
            ("fused_qkv", "fused_norm"),
        )
        self.assertEqual(
            _expected_mlx_benchmark_variants(
                "sglang-whole-mlx-fused-qkv-norm-swiglu-metal-commit"
            ),
            ("fused_qkv", "fused_norm", "fused_swiglu"),
        )
        args = _trial_args(trials=1, collect_mlx_phase_timing=True)
        first = _benchmark_trial(1.0)
        required = {
            "producer_fence": 0.1,
            "input_import": 0.2,
            "graph_build": 0.3,
            "prepare_eval": 0.4,
            "dlpack_export": 0.5,
            "kv_commit_submit": 0.6,
        }
        first["decode_steps"] = 1
        first["mlx_phase_timing"] = {
            "enabled": True,
            "contract": _MLX_PHASE_TIMING_CONTRACT,
            "prefill": dict(required),
            "decode": dict(required),
            "decode_steps": [dict(required)],
        }
        result = {
            "status": "completed",
            "engine": "sglang",
            "profile": "sglang-whole-mlx-fused-qkv-metal-commit",
            "timed_trials": 1,
            "stable_trial_policy": _stable_trial_policy(args),
            "attempted_timed_trials": 1,
            "accepted_timed_attempts": [1],
            "rejected_timed_trials": [],
            "trials": [first],
            "summary": _summarize_trials([first]),
            "mlx_benchmark_variants": list(
                _expected_mlx_benchmark_variants(
                    "sglang-whole-mlx-fused-qkv-metal-commit"
                )
            ),
            "timing_instrumentation": {
                "mlx_phase_timing": True,
                "summary_includes_instrumentation": True,
            },
        }
        _validate_worker_stable_trials(result, args)

        missing = json.loads(json.dumps(result))
        del missing["trials"][0]["mlx_phase_timing"]["prefill"]["graph_build"]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            _validate_worker_stable_trials(missing, args)

        wrong_variant = json.loads(json.dumps(result))
        wrong_variant["mlx_benchmark_variants"] = []
        with self.assertRaisesRegex(RuntimeError, "variants"):
            _validate_worker_diagnostics(wrong_variant, args)

        wrong_duration = json.loads(json.dumps(result))
        wrong_duration["trials"][0]["mlx_phase_timing"]["prefill"]["prepare_eval"] = (
            float("nan")
        )
        with self.assertRaisesRegex(RuntimeError, "finite"):
            _validate_worker_diagnostics(wrong_duration, args)

    @unittest.skipUnless(_HAS_MLX, "requires MLX")
    def test_swiglu_benchmark_variant_matches_the_normal_range(self):
        import mlx.core as mx

        import sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx as qwen3_mlx

        original = qwen3_mlx._swiglu
        try:
            with mock.patch.dict(
                os.environ, {"SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_swiglu"}
            ):
                self.assertEqual(
                    _install_sglang_mlx_benchmark_variant(), ("fused_swiglu",)
                )
            mx.random.seed(20260803)
            gate = mx.random.normal((8, 3072)).astype(mx.bfloat16)
            up = mx.random.normal((8, 3072)).astype(mx.bfloat16)
            actual = qwen3_mlx._swiglu(gate, up)
            expected = (mx.sigmoid(gate) * gate) * up
            mx.eval(actual, expected)
            self.assertTrue(bool(mx.all(actual == expected).item()))
        finally:
            qwen3_mlx._swiglu = original

    @unittest.skipUnless(_HAS_MLX, "requires MLX")
    def test_fused_norm_variant_keeps_qk_width_on_staged_path(self):
        import mlx.core as mx

        import sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx as qwen3_mlx

        original_rms = qwen3_mlx._rms_norm
        original_add_rms = qwen3_mlx._add_rms_norm
        try:
            with mock.patch.dict(
                os.environ, {"SGLANG_MPS_QWEN3_BENCH_VARIANT": "fused_norm"}
            ):
                self.assertEqual(
                    _install_sglang_mlx_benchmark_variant(), ("fused_norm",)
                )

            mx.random.seed(20260803)
            qk = mx.random.normal((1, 16, 128)).astype(mx.bfloat16)
            qk_weight = mx.random.normal((128,)).astype(mx.bfloat16)
            qk_actual = qwen3_mlx._rms_norm(qk, qk_weight, 1e-6)
            qk_expected = original_rms(qk, qk_weight, 1e-6)

            hidden = mx.random.normal((1, 1024)).astype(mx.bfloat16)
            residual = mx.random.normal((1, 1024)).astype(mx.bfloat16)
            hidden_weight = mx.random.normal((1024,)).astype(mx.bfloat16)
            hidden_actual = qwen3_mlx._add_rms_norm(
                hidden, residual, hidden_weight, 1e-6
            )
            hidden_expected = original_add_rms(hidden, residual, hidden_weight, 1e-6)
            mx.eval(qk_actual, qk_expected, *hidden_actual, *hidden_expected)
            self.assertTrue(bool(mx.all(qk_actual == qk_expected).item()))
            self.assertTrue(
                all(
                    bool(mx.all(actual == expected).item())
                    for actual, expected in zip(hidden_actual, hidden_expected)
                )
            )
        finally:
            qwen3_mlx._rms_norm = original_rms
            qwen3_mlx._add_rms_norm = original_add_rms

    def test_worker_writes_structured_stable_trial_failure_atomically(self):
        artifact = {
            "schema_version": 1,
            "status": "insufficient_stable_trials",
            "profile": "sglang-torch",
            "attempted_timed_trials": 3,
            "accepted_timed_attempts": [2],
            "rejected_timed_trials": [{"attempt": 1}, {"attempt": 3}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            result_path = root / "worker.json"
            manifest_path.write_text("{}")
            args = SimpleNamespace(
                worker_manifest=str(manifest_path),
                worker_result=str(result_path),
                worker_phase=str(root / "worker.phase.json"),
                worker_engine="sglang",
                worker_profile="sglang-torch",
            )
            error = _StableTrialExhausted("not enough stable trials", artifact)
            with mock.patch(
                "sglang.benchmark.mps_qwen3._run_sglang_worker",
                side_effect=error,
            ):
                with self.assertRaises(_StableTrialExhausted):
                    _worker_main(args)

            self.assertEqual(json.loads(result_path.read_text()), artifact)
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_sglang_worker_disables_single_shard_multithread_loader(self):
        # Import the module before replacing ServerArgs. Upstream's tokenizer
        # manager inspects the real ServerArgs dataclass at import time, while
        # this test only needs to replace the constructor used by the worker.
        from sglang.benchmark import one_batch  # noqa: F401

        args = SimpleNamespace(
            mem_fraction_static=0.55,
            worker_profile="sglang-torch",
            sync_mode="each",
        )
        manifest = {
            "model_path": "/unused/Qwen3-0.6B",
            "input_len": 512,
            "output_len": 17,
            "seed": 2026,
        }
        server_args = SimpleNamespace(cuda_graph_config=None)
        with (
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch(
                "sglang.srt.server_args.ServerArgs",
                return_value=server_args,
            ) as constructor,
            mock.patch(
                "sglang.srt.server_args.PortArgs.init_new",
                return_value=object(),
            ),
            mock.patch("sglang.benchmark.one_batch._set_envs_and_config"),
            mock.patch("sglang.benchmark.one_batch.initialize_moe_config"),
            mock.patch("sglang.benchmark.one_batch.initialize_fp8_gemm_config"),
            mock.patch("sglang.benchmark.one_batch.initialize_fp4_gemm_config"),
            mock.patch(
                "sglang.benchmark.one_batch.load_model",
                side_effect=RuntimeError("stop before model loading"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                _run_sglang_worker(args, manifest)

        loader_config = constructor.call_args.kwargs["model_loader_extra_config"]
        self.assertEqual(loader_config, {"enable_multithread_load": False})
        self.assertNotIn("disable_mmap", loader_config)
        self.assertNotIn("drop_cache", loader_config)

    def test_worker_command_forwards_separate_trial_and_profile_limits(self):
        args = _trial_args(
            warmup_trials=2,
            trial_delay=1.0,
            sync_mode="each",
            mem_fraction_static=0.55,
            mlx_lm_path="mlx-lm",
        )
        command = _build_worker_command(
            args,
            "sglang-torch",
            PROFILES["sglang-torch"],
            Path("manifest.json"),
            Path("result.json"),
        )

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value("--max-trial-swap-in-mb"), "64.0")
        self.assertEqual(value("--max-trial-swap-out-mb"), "16.0")
        self.assertEqual(value("--max-trial-swap-growth-mb"), "128.0")
        self.assertEqual(value("--max-profile-swap-out-mb"), "64.0")
        self.assertEqual(value("--max-profile-swap-growth-mb"), "512.0")
        self.assertEqual(value("--max-trial-attempts"), "4")
        self.assertEqual(value("--worker-phase"), "result.phase.json")
        self.assertNotIn("--max-swap-out-mb", command)

    def test_worker_command_keeps_phase_timing_opt_in(self):
        args = _trial_args(
            warmup_trials=1,
            trial_delay=0.0,
            sync_mode="each",
            mem_fraction_static=0.35,
            mlx_lm_path="mlx-lm",
        )
        command = _build_worker_command(
            args,
            "sglang-whole-mlx-metal-commit",
            PROFILES["sglang-whole-mlx-metal-commit"],
            Path("manifest.json"),
            Path("result.json"),
        )
        self.assertNotIn("--collect-mlx-phase-timing", command)
        args.collect_mlx_phase_timing = True
        command = _build_worker_command(
            args,
            "sglang-whole-mlx-metal-commit",
            PROFILES["sglang-whole-mlx-metal-commit"],
            Path("manifest.json"),
            Path("result.json"),
        )
        self.assertIn("--collect-mlx-phase-timing", command)


if __name__ == "__main__":
    unittest.main()
