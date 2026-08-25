"""Torch-native LoRA metadata ownership on Apple MPS."""

import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
from sglang.srt.lora.layers import ParallelLMHeadWithLoRA
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _extend_batch(
    seq_lens: list[int],
    *,
    return_logprob: bool = False,
    logprob_start_lens: list[int] | None = None,
):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        extend_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=seq_lens,
        return_logprob=return_logprob,
        extend_logprob_start_lens_cpu=logprob_start_lens,
    )


def test_torch_native_backend_normalizes_string_device():
    backend = TorchNativeLoRABackend(max_loras_per_batch=1, device="cpu")

    assert backend.device == torch.device("cpu")


def test_torch_native_lm_head_uses_pruned_per_request_metadata():
    backend = TorchNativeLoRABackend(
        max_loras_per_batch=2,
        device=torch.device("cpu"),
    )
    backend.prepare_lora_batch(
        _extend_batch([3, 4]),
        weight_indices=[0, 1],
        lora_ranks=[1, 1],
        scalings=[1.0, 1.0],
        use_cuda_graph=False,
    )

    info = backend.lm_head_batch_info
    assert info is not None
    assert info.expected_tokens == 2
    torch.testing.assert_close(
        info.seg_lens_cpu, torch.tensor([1, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        info.weight_indices_cpu, torch.tensor([0, 1], dtype=torch.int32)
    )

    output = backend.run_lora_a_sgemm(
        torch.ones(2, 1),
        torch.tensor([[[1.0]], [[10.0]]]),
        pruned_batch_info=info,
    )
    torch.testing.assert_close(output, torch.tensor([[1.0], [10.0]]))

    layer = object.__new__(ParallelLMHeadWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.lora_backend = backend
    layer.lm_head_A_buffer = torch.tensor([[[1.0]], [[10.0]]])
    layer.lm_head_B_buffer = torch.ones(2, 1, 1)
    layer.output_offset = torch.tensor([0, 1], dtype=torch.int32)
    layer.output_offset_cpu = torch.tensor([0, 1], dtype=torch.int32)
    wrapped_output = ParallelLMHeadWithLoRA.apply_lora(
        layer,
        torch.zeros(2, 1),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(wrapped_output, torch.tensor([[1.0], [10.0]]))

    backend.prepare_lora_batch(
        SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=2),
        weight_indices=[0, 1],
        lora_ranks=[1, 1],
        scalings=[1.0, 1.0],
        use_cuda_graph=False,
    )
    assert backend.lm_head_batch_info is None
    assert backend.lm_head_pass_batch_infos is None


def test_torch_native_lm_head_builds_metadata_for_each_logprob_pass():
    backend = TorchNativeLoRABackend(
        max_loras_per_batch=2,
        device=torch.device("cpu"),
    )
    batch = _extend_batch(
        [3, 4],
        return_logprob=True,
        logprob_start_lens=[0, 1],
    )
    with (
        mock.patch(
            "sglang.srt.lora.backend.lmhead_mixing.envs."
            "SGLANG_ENABLE_LOGPROB_CHUNK.get",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.lora.backend.lmhead_mixing.envs."
            "SGLANG_LOGPROB_CHUNK_SIZE.get",
            return_value=2,
        ),
    ):
        backend.prepare_lora_batch(
            batch,
            weight_indices=[0, 1],
            lora_ranks=[1, 1],
            scalings=[1.0, 1.0],
            use_cuda_graph=False,
        )

    infos = backend.lm_head_pass_batch_infos
    assert infos is not None
    assert [info.expected_tokens for info in infos] == [2, 2, 2]
    assert [info.seg_lens_cpu.tolist() for info in infos] == [
        [2],
        [1, 1],
        [2],
    ]
    assert [info.weight_indices_cpu.tolist() for info in infos] == [
        [0],
        [0, 1],
        [1],
    ]


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
def test_torch_native_lora_uses_pageable_metadata_on_mps():
    backend = TorchNativeLoRABackend(
        max_loras_per_batch=2,
        device=torch.device("mps"),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=2)

    backend.prepare_lora_batch(
        batch,
        weight_indices=[0, 1],
        lora_ranks=[4, 8],
        scalings=[0.5, 1.0],
        use_cuda_graph=False,
    )

    info = backend.batch_info
    assert info.weight_indices.device.type == "cpu"
    assert info.seg_indptr.device.type == "cpu"
    assert info.lora_ranks.device.type == "cpu"
    assert info.weight_indices_cpu.device.type == "cpu"
    assert info.seg_indptr_cpu.device.type == "cpu"
    assert not info.weight_indices_cpu.is_pinned()
    assert not info.seg_indptr_cpu.is_pinned()
    assert info.weight_indices is info.weight_indices_cpu
    assert info.seg_indptr is info.seg_indptr_cpu
    assert info.seg_lens is info.seg_lens_cpu
    assert info.lora_ranks is info.lora_ranks_cpu
    assert info.scalings is info.scalings_cpu
    torch.testing.assert_close(
        info.weight_indices.cpu(), torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        info.seg_indptr.cpu(), torch.tensor([0, 1, 2], dtype=torch.int32)
    )

    backend.prepare_lora_batch(
        batch,
        weight_indices=[0, 1],
        lora_ranks=[1, 2],
        scalings=[0.5, 1.0],
        use_cuda_graph=False,
    )
    inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
    lora_a = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        device="mps",
    )
    a_output = backend.run_lora_a_sgemm(inputs, lora_a)
    lora_b = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        ],
        device="mps",
    )
    output = backend.run_lora_b_sgemm(
        a_output,
        lora_b,
        torch.tensor([0, 3], dtype=torch.int32),
    )
    torch.mps.synchronize()
    torch.testing.assert_close(a_output.cpu(), torch.tensor([[0.5, 0.0], [9.0, 6.0]]))
    torch.testing.assert_close(
        output.cpu(), torch.tensor([[0.5, 1.5, 2.5], [9.0, 6.0, 15.0]])
    )


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
def test_torch_native_lora_layer_offsets_do_not_request_pinned_memory_on_mps():
    # A pinned CPU allocation can terminate a Torch MPS process rather than
    # raising a catchable exception, so keep this regression in a subprocess.
    script = textwrap.dedent("""
        import torch
        from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
        from sglang.srt.lora.layers import ColumnParallelLinearWithLoRA

        class FakeColumnParallelLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(1, device="mps"))
                self.output_partition_sizes = [8]

        layer = ColumnParallelLinearWithLoRA(
            FakeColumnParallelLinear(),
            TorchNativeLoRABackend(2, torch.device("mps")),
        )
        assert layer.output_offset.device.type == "mps"
        assert layer.output_offset_cpu.device.type == "cpu"
        assert not layer.output_offset_cpu.is_pinned()
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (
        completed.returncode == 0
    ), f"stdout={completed.stdout}\nstderr={completed.stderr}"


def test_torch_native_lora_rejects_programmatic_mps_graph_initialization():
    backend = TorchNativeLoRABackend(
        max_loras_per_batch=2,
        device=torch.device("mps"),
    )
    with pytest.raises(RuntimeError, match="CUDA-only"):
        backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=2,
            num_tokens_per_req=1,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
