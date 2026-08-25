import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.torch_ops import (
    sgemm_lora_a_embedding_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    generate_sequence_lengths,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.platforms import current_platform


@dataclass
class TorchNativeLoRABatchInfo(LoRABatchInfo):
    # ranks of each lora adapter, in shape (lora_num,) placed on cpu device
    lora_ranks_cpu: Optional[torch.Tensor] = None

    # Indice pointers of each segment in shape (num_segments + 1, ) placed on cpu device
    seg_indptr_cpu: Optional[torch.Tensor] = None

    # Lengths of each segments in shape (num_segments,) placed on cpu device
    seg_lens_cpu: Optional[torch.Tensor] = None

    # The index of lora adapter used by each segment, in shape (num_segments,) placed on cpu device
    weight_indices_cpu: Optional[torch.Tensor] = None

    # Scaling factors for each lora adapter, in shape (lora_num,) placed on cpu device
    scalings_cpu: Optional[torch.Tensor] = None


class TorchNativeLoRABackend(BaseLoRABackend):
    name = "torch_native"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        # Direct backend users historically pass either ``"cpu"`` or a
        # torch.device. Normalize once before platform/lifecycle decisions.
        super().__init__(max_loras_per_batch, torch.device(device))

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert (
            extra_embeddings is None
        ), "Extra embeddings for lora a is not supported yet in chunked backend"
        output_tensor = sgemm_lora_a_embedding_fwd(
            inputs=input_ids,
            weights=weights,
            batch_info=self.batch_info,
            vocab_size=vocab_size,
        )

        return output_tensor

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        pruned_batch_info: TorchNativeLoRABatchInfo = None,
        stack_num: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )
        output_tensor = sgemm_lora_a_fwd(
            inputs=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=stack_num,
        )

        return output_tensor

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        output_offset_cpu: torch.Tensor = None,
        base_output: torch.Tensor = None,
        pruned_batch_info: TorchNativeLoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, weight_out_dim, _ = weights.shape
        if output_offset_cpu is None:
            output_offset_cpu = torch.tensor(
                [0, weight_out_dim], dtype=torch.int32, device="cpu"
            )
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )

        output_tensor = sgemm_lora_b_fwd(
            inputs=x,
            weights=weights,
            batch_info=batch_info,
            slice_offsets=output_offset_cpu,
            base_output=base_output,
        )

        return output_tensor

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        output_offset_cpu: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        n_slices: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        lora_a_output = sgemm_lora_a_fwd(
            inputs=x,
            weights=qkv_lora_a,
            batch_info=self.batch_info,
            num_slices=n_slices,
        )

        output_tensor = sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=qkv_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset_cpu,
            base_output=base_output,
        )

        return output_tensor

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset_cpu: torch.Tensor = None,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, weight_out_dim, _ = gate_up_lora_b.shape
        if output_offset_cpu is None:
            output_offset_cpu = torch.tensor(
                [0, weight_out_dim // 2, weight_out_dim],
                dtype=torch.int32,
                device="cpu",
            )
        num_slices = len(output_offset_cpu) - 1

        lora_a_output = sgemm_lora_a_fwd(
            inputs=x,
            weights=gate_up_lora_a,
            batch_info=self.batch_info,
            num_slices=num_slices,
        )

        output_tensor = sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=gate_up_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset_cpu,
            base_output=base_output,
        )

        return output_tensor

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_req: int,
    ):
        if self.device.type != "cuda":
            raise RuntimeError(
                "torch_native LoRA graph metadata is CUDA-only; graph capture "
                f"must remain disabled on device {self.device.type!r}"
            )
        with torch.device("cuda"):
            self.cuda_graph_batch_info = TorchNativeLoRABatchInfo(
                use_cuda_graph=True,
                bs=max_bs_in_cuda_graph,
                num_segments=self.max_loras_per_batch,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_req, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                permutation=None,
                max_len=num_tokens_per_req,
            )

            # Initialize seg_indptr for CUDA graph as they remain constant
            # across batches.
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
            )

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
        use_prefill_cuda_graph: bool = False,
    ):
        # Do not use merge optimization for graph mode
        # CUDA pins host metadata for non-blocking H2D copies. MPS has unified
        # memory and no pinned allocator, so retain ordinary CPU ownership.
        pin_memory = current_platform.is_pin_memory_available(self.device)
        non_blocking = pin_memory

        def maybe_pin(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.pin_memory() if pin_memory else tensor

        original_seq_lens_cpu = generate_sequence_lengths(forward_batch, device="cpu")
        if not use_cuda_graph:
            original_weight_indices_tensor = torch.tensor(
                weight_indices, dtype=torch.int32, device="cpu"
            )

            unique_weight_indices_tensor, inverse_weight_indices_tensor = (
                torch.unique_consecutive(
                    original_weight_indices_tensor, return_inverse=True
                )
            )

            seg_lens_cpu = maybe_pin(
                torch.zeros_like(
                    unique_weight_indices_tensor, dtype=torch.int32, device="cpu"
                ).scatter_add_(
                    0,
                    inverse_weight_indices_tensor,
                    original_seq_lens_cpu,
                )
            )

            weight_indices_tensor = maybe_pin(unique_weight_indices_tensor)
        else:
            weight_indices_tensor = maybe_pin(
                torch.repeat_interleave(
                    torch.tensor(weight_indices, dtype=torch.int32, device="cpu"),
                    original_seq_lens_cpu,
                )
            )
            seg_lens_cpu = maybe_pin(torch.ones_like(weight_indices_tensor))

        seg_indptr_cpu = torch.zeros(
            (len(seg_lens_cpu) + 1,), dtype=torch.int32, pin_memory=pin_memory
        )
        seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)
        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=pin_memory, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=pin_memory, device="cpu"
        )

        bs = forward_batch.batch_size
        num_segments = len(weight_indices_tensor)

        # The non-graph dense torch_native kernels read only the *_cpu fields.
        # Do not manufacture five unconsumed MPS mirrors: besides forcing a
        # command-buffer synchronization, those copies add roughly 1 ms to
        # every scheduler batch. Keep one CPU-owned metadata set and alias the
        # generic fields so lifecycle and dtype remain explicit.
        if self.device.type == "mps" and not use_cuda_graph and not self.is_moe_lora:
            batch_info = TorchNativeLoRABatchInfo(
                bs=bs,
                num_segments=num_segments,
                max_len=int(max(seg_lens_cpu)),
                use_cuda_graph=False,
                seg_lens=seg_lens_cpu,
                seg_indptr=seg_indptr_cpu,
                weight_indices=weight_indices_tensor,
                lora_ranks=lora_ranks_tensor,
                scalings=scalings_tensor,
                permutation=None,
                lora_ranks_cpu=lora_ranks_tensor,
                seg_indptr_cpu=seg_indptr_cpu,
                seg_lens_cpu=seg_lens_cpu,
                weight_indices_cpu=weight_indices_tensor,
                scalings_cpu=scalings_tensor,
            )
            self.batch_info = batch_info
            self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
                self._prepare_lm_head_batch_info(
                    forward_batch, weight_indices, batch_info
                )
            )
            return

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments
        else:
            max_len = max(seg_lens_cpu)

            batch_info = TorchNativeLoRABatchInfo(
                bs=forward_batch.batch_size,
                num_segments=num_segments,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=torch.empty((bs,), dtype=torch.int32, device=self.device),
                seg_indptr=torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=None,
            )

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=non_blocking
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=non_blocking
        )
        batch_info.weight_indices[:num_segments].copy_(
            weight_indices_tensor, non_blocking=non_blocking
        )
        batch_info.seg_indptr[: len(seg_indptr_cpu)].copy_(
            seg_indptr_cpu, non_blocking=non_blocking
        )
        batch_info.seg_lens[: len(seg_lens_cpu)].copy_(
            seg_lens_cpu, non_blocking=non_blocking
        )

        batch_info.lora_ranks_cpu = lora_ranks_tensor
        batch_info.seg_indptr_cpu = seg_indptr_cpu
        batch_info.seg_lens_cpu = seg_lens_cpu
        batch_info.weight_indices_cpu = weight_indices_tensor
        batch_info.scalings_cpu = scalings_tensor

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)
        self.batch_info = batch_info
        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)
        )

    def _prepare_lm_head_batch_info(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        batch_info: TorchNativeLoRABatchInfo,
    ) -> Tuple[
        Optional[TorchNativeLoRABatchInfo],
        Optional[List[TorchNativeLoRABatchInfo]],
    ]:
        """Build Torch-control metadata matching pruned lm_head inputs."""
        pruned_lens = get_lm_head_pruned_lens(forward_batch)
        if pruned_lens is None:
            return None, None

        pruned_total = sum(pruned_lens)
        lm_head_segments = merge_and_chunk_segments(
            weight_indices,
            pruned_lens,
            chunk_size=pruned_total,
        )
        lm_head_batch_info = self._build_lm_head_batch_info(
            lm_head_segments,
            batch_info,
            pruned_total,
        )

        lm_head_pass_batch_infos = None
        pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)
        if pass_segments is not None:
            lm_head_pass_batch_infos = []
            for pass_weight_indices, pass_lens in pass_segments:
                pass_total = sum(pass_lens)
                merged_segments = merge_and_chunk_segments(
                    pass_weight_indices,
                    pass_lens,
                    chunk_size=pass_total,
                )
                lm_head_pass_batch_infos.append(
                    self._build_lm_head_batch_info(
                        merged_segments,
                        batch_info,
                        pass_total,
                    )
                )

        return lm_head_batch_info, lm_head_pass_batch_infos

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: TorchNativeLoRABatchInfo,
        expected_tokens: int,
    ) -> TorchNativeLoRABatchInfo:
        weight_indices, seg_lens = lm_head_segments
        num_segments = len(weight_indices)
        pin_memory = current_platform.is_pin_memory_available(self.device)

        weight_indices_cpu = torch.tensor(
            weight_indices,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        seg_lens_cpu = torch.tensor(
            seg_lens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        seg_indptr_cpu = torch.zeros(
            num_segments + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)

        # Torch control-flow kernels consume the host fields. Keep generic
        # fields as aliases on CPU/MPS; CUDA retains device mirrors for the
        # common LoRABatchInfo contract even though this eager tail uses CPU
        # metadata.
        if self.device.type in {"cpu", "mps"}:
            weight_indices_device = weight_indices_cpu
            seg_lens_device = seg_lens_cpu
            seg_indptr_device = seg_indptr_cpu
        else:
            weight_indices_device = weight_indices_cpu.to(
                self.device, non_blocking=pin_memory
            )
            seg_lens_device = seg_lens_cpu.to(self.device, non_blocking=pin_memory)
            seg_indptr_device = seg_indptr_cpu.to(self.device, non_blocking=pin_memory)

        return dataclasses.replace(
            batch_info,
            use_cuda_graph=False,
            bs=num_segments,
            num_segments=num_segments,
            max_len=max(seg_lens),
            seg_lens=seg_lens_device,
            seg_indptr=seg_indptr_device,
            weight_indices=weight_indices_device,
            permutation=None,
            expected_tokens=expected_tokens,
            seg_lens_cpu=seg_lens_cpu,
            seg_indptr_cpu=seg_indptr_cpu,
            weight_indices_cpu=weight_indices_cpu,
        )
