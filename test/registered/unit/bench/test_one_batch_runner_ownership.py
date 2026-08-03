import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.benchmark import one_batch
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


class TestOneBatchRunnerOwnership(unittest.TestCase):
    def test_synthetic_request_uses_the_measured_output_length(self):
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1,
            3,
            [[11, 12, 13]],
            output_len=7,
        )

        self.assertEqual(reqs[0].sampling_params.max_new_tokens, 7)

    def test_split_prepare_and_forward_keep_extend_decode_contracts(self):
        model_runner = object()
        reqs = object()
        batch = object()
        prefill_forward_batch = object()
        decode_forward_batch = object()
        prefill_tokens = object()
        prefill_logits = object()
        decode_tokens = object()
        decode_logits = object()

        with (
            mock.patch.object(
                one_batch,
                "prepare_extend_forward_batch",
                return_value=(prefill_forward_batch, batch),
            ) as prepare_extend,
            mock.patch.object(
                one_batch,
                "run_forward_batch",
                return_value=(prefill_tokens, prefill_logits),
            ) as run_prefill,
        ):
            self.assertEqual(
                one_batch.extend(reqs, model_runner),
                (prefill_tokens, prefill_logits, batch),
            )
        prepare_extend.assert_called_once_with(reqs, model_runner)
        run_prefill.assert_called_once_with(prefill_forward_batch, model_runner)

        with (
            mock.patch.object(
                one_batch,
                "prepare_decode_forward_batch",
                return_value=decode_forward_batch,
            ) as prepare_decode,
            mock.patch.object(
                one_batch,
                "run_forward_batch",
                return_value=(decode_tokens, decode_logits),
            ) as run_decode,
        ):
            self.assertEqual(
                one_batch.decode(prefill_tokens, batch, model_runner),
                (decode_tokens, decode_logits),
            )
        prepare_decode.assert_called_once_with(prefill_tokens, batch, model_runner)
        run_decode.assert_called_once_with(decode_forward_batch, model_runner)

    def test_zero_correctness_cut_constructs_one_full_prefill(self):
        tokenizer = SimpleNamespace(encode=lambda _prompt: [11, 12, 13, 14])
        bench_args = SimpleNamespace(batch_size=(1,), cut_len=0, output_len=(7,))

        input_ids, reqs = one_batch.prepare_inputs_for_correctness_test(
            bench_args,
            tokenizer,
            ["full prompt"],
        )

        self.assertEqual(input_ids, [[11, 12, 13, 14]])
        self.assertEqual(list(reqs[0].origin_input_ids), input_ids[0])
        self.assertEqual(reqs[0].extend_range.start, 0)
        self.assertEqual(reqs[0].extend_range.end, len(input_ids[0]))
        self.assertEqual(reqs[0].sampling_params.max_new_tokens, 7)

    def test_mps_constructs_standard_model_runner(self):
        server_args = SimpleNamespace(
            tp_size=1,
            ep_size=1,
            enable_dp_attention=False,
            dp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            dcp_size=1,
            mem_fraction_static=0.7,
            tokenizer_path="test-tokenizer",
            tokenizer_mode="auto",
            trust_remote_code=False,
            is_startup_weight_load_overlap=False,
        )
        port_args = SimpleNamespace(nccl_port=12345)
        model_config = object()
        parallel_state = object()
        tokenizer = object()
        torch_runner = mock.MagicMock(max_total_num_tokens=1024)

        with (
            mock.patch.object(
                one_batch.ModelConfig,
                "from_server_args",
                return_value=model_config,
            ),
            mock.patch.object(
                one_batch,
                "compute_dp_attention_world_info",
                return_value=(0, 1, 0, 1),
            ),
            mock.patch.object(one_batch, "ParallelState", return_value=parallel_state),
            mock.patch.object(
                one_batch, "ModelRunner", return_value=torch_runner
            ) as model_runner_cls,
            mock.patch.object(one_batch, "get_tokenizer", return_value=tokenizer),
            mock.patch.object(one_batch, "suppress_other_loggers"),
        ):
            runner, loaded_tokenizer = one_batch.load_model(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
            )

        model_runner_cls.assert_called_once_with(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            ps=parallel_state,
            nccl_port=port_args.nccl_port,
            server_args=server_args,
        )
        torch_runner.alloc_memory_pool.assert_called_once_with()
        torch_runner.init_attention_backends.assert_called_once_with()
        torch_runner.init_cuda_graphs.assert_called_once_with()
        self.assertIsInstance(runner, one_batch._TorchBenchRunner)
        self.assertIs(runner.torch_runner, torch_runner)
        self.assertIs(loaded_tokenizer, tokenizer)

    def test_model_core_benchmark_can_skip_tokenizer_loading(self):
        server_args = SimpleNamespace(
            tp_size=1,
            ep_size=1,
            enable_dp_attention=False,
            dp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            dcp_size=1,
            mem_fraction_static=0.7,
            tokenizer_path="unused-tokenizer",
            tokenizer_mode="auto",
            trust_remote_code=False,
            is_startup_weight_load_overlap=False,
        )
        port_args = SimpleNamespace(nccl_port=12345)
        torch_runner = mock.MagicMock(max_total_num_tokens=1024)

        with (
            mock.patch.object(one_batch.ModelConfig, "from_server_args"),
            mock.patch.object(
                one_batch,
                "compute_dp_attention_world_info",
                return_value=(0, 1, 0, 1),
            ),
            mock.patch.object(one_batch, "ParallelState"),
            mock.patch.object(one_batch, "ModelRunner", return_value=torch_runner),
            mock.patch.object(one_batch, "get_tokenizer") as get_tokenizer,
            mock.patch.object(one_batch, "suppress_other_loggers"),
        ):
            runner, tokenizer = one_batch.load_model(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
                load_tokenizer=False,
            )

        get_tokenizer.assert_not_called()
        self.assertIsInstance(runner, one_batch._TorchBenchRunner)
        self.assertIsNone(tokenizer)


class TestOneBatchMetalProfiling(unittest.TestCase):
    def test_mps_profile_uses_unified_torch_profiler_path(self):
        profiler = mock.MagicMock()

        with (
            mock.patch.object(
                torch.backends.mps,
                "is_available",
                return_value=True,
            ),
            mock.patch.object(one_batch, "apply_metal_profiler_patches") as apply_patch,
            mock.patch.object(
                torch.profiler,
                "profile",
                return_value=profiler,
            ) as torch_profile,
        ):
            result = one_batch.start_profile(
                ("CPU", "GPU"),
                profile_record_shapes=True,
            )

        self.assertIs(result, profiler)
        apply_patch.assert_called_once_with()
        torch_profile.assert_called_once_with(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=True,
        )
        profiler.start.assert_called_once_with()

    def test_stop_profile_exports_wrapper_trace_normally(self):
        torch_profiler = mock.MagicMock()
        key_averages = mock.MagicMock()
        torch_profiler.key_averages.return_value = key_averages
        profiler = SimpleNamespace(
            torch_profiler=torch_profiler,
            stop=mock.MagicMock(),
            export_chrome_trace=mock.MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "profile.trace.json.gz"
            with mock.patch("builtins.print"):
                one_batch.stop_profile(
                    profiler,
                    ("CPU", "GPU"),
                    save_trace=True,
                    trace_filename=str(trace_path),
                    stage="prefill",
                )

        profiler.stop.assert_called_once_with()
        profiler.export_chrome_trace.assert_called_once_with(str(trace_path))
        torch_profiler.key_averages.assert_called_once_with(group_by_input_shape=True)
        key_averages.table.assert_called_once_with(sort_by="self_cpu_time_total")

    def test_gpu_only_metal_wrapper_does_not_require_key_averages(self):
        profiler = SimpleNamespace(
            torch_profiler=None,
            export_chrome_trace=mock.MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            one_batch._save_profile_trace_results(
                profiler,
                ("GPU",),
                str(Path(tmp) / "profile.trace.json.gz"),
            )

        profiler.export_chrome_trace.assert_called_once()


if __name__ == "__main__":
    unittest.main()
