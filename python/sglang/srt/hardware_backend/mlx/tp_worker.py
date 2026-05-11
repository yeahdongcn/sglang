"""MLX-specific TpModelWorker subclass for Apple Silicon.

Routes forward passes through the MLX model runner, bypassing PyTorch
MPS. A lightweight stub provides scheduler bookkeeping; scheduler-visible
KV data lives in the MLX paged KV cache.
"""

import logging
import time
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)


class MlxTpModelWorker(TpModelWorker):
    """A tensor parallel model worker that routes inference through MLX.

    Inherits from TpModelWorker for scheduler integration, but replaces
    the standard ModelRunner with MlxModelRunnerStub (no PyTorch weights,
    zero-memory KV cache) and delegates all forward passes to a native
    MlxModelRunner.
    """

    def _init_model_runner(self):
        """Create MLX runner first (auto-sizes pool), then stub with matching size."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        init_kwargs = dict(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            disable_radix_cache=self.server_args.disable_radix_cache,
            mem_fraction_static=self.server_args.mem_fraction_static,
            page_size=self.server_args.page_size,
        )
        if self.server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = self.server_args.max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            mlx_pool_size=self._mlx_runner.pool_size,
            mlx_page_size=self._mlx_runner._get_paged_attention_block_size(),
        )

        self._mlx_active_rids: set[str] = set()
        self._mlx_pool_initialized = False

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

    def _ensure_mlx_pool_initialized(self):
        """Lazily initialize MLX paged KV cache metadata after stub pools are ready."""
        if not self._mlx_pool_initialized:
            self._mlx_runner.init_kv_pool(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if model_worker_batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(model_worker_batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            model_worker_batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
        )

    def _forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Run forward pass through the MLX model runner (greedy only)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs
        profile_timing = self._mlx_runner._profile_timing_enabled()
        profile_start = time.perf_counter() if profile_timing else 0.0

        if forward_mode.is_idle():
            cleanup_start = time.perf_counter() if profile_timing else 0.0
            for rid in self._mlx_active_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids.clear()
            self._mlx_runner.mark_idle()
            cleanup_done = time.perf_counter() if profile_timing else 0.0
            if profile_timing:
                logger.info(
                    "MLX worker timing mode=idle cleanup_ms=%.2f total_ms=%.2f",
                    (cleanup_done - cleanup_start) * 1000,
                    (cleanup_done - profile_start) * 1000,
                )
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # Auto-cleanup: remove MLX state for requests no longer in the batch.
        # No-radix cleanup is just cache-pool release, so run it before new
        # extend batches as well. Radix cleanup can flush deferred decode K/V
        # into the side-store, so keep that on decode/idle boundaries.
        current_rids = {req.rid for req in reqs}
        if self._mlx_runner.disable_radix_cache or forward_mode.is_decode():
            stale_rids = self._mlx_active_rids - current_rids
            for rid in stale_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids = current_rids
        else:
            self._mlx_active_rids |= current_rids

        next_token_ids_list = []

        if forward_mode.is_extend():
            cpu_start = time.perf_counter() if profile_timing else 0.0
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            out_cache_loc_cpu = model_worker_batch.out_cache_loc.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens
            cpu_done = time.perf_counter() if profile_timing else 0.0

            offset = 0  # into input_ids_cpu
            slot_offset = 0  # into out_cache_loc_cpu
            prefill_rids = []
            extend_rids = []
            decode_rids = []
            decode_slots = []
            prefill_ms = 0.0
            extend_ms = 0.0
            decode_ms = 0.0

            decoding_req_ids = set()
            if forward_mode.is_mixed():
                decoding_reqs = getattr(model_worker_batch, "decoding_reqs", None) or []
                decoding_req_ids = {req.rid for req in decoding_reqs}

            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
                offset += seq_len
                slot_offset += seq_len

                if self._mlx_runner.has_request(req.rid):
                    if req.rid in decoding_req_ids:
                        decode_rids.append(req.rid)
                        decode_slots.append(req_new_slots[0])
                    else:
                        runner_start = time.perf_counter() if profile_timing else 0.0
                        next_token = self._mlx_runner.extend(
                            req.rid, req_token_ids, req_new_slots
                        )
                        runner_done = time.perf_counter() if profile_timing else 0.0
                        if profile_timing:
                            extend_ms += (runner_done - runner_start) * 1000
                        extend_rids.append((req.rid, next_token))
                else:
                    # New prefill
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.fill_ids)
                    runner_start = time.perf_counter() if profile_timing else 0.0
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                    )
                    runner_done = time.perf_counter() if profile_timing else 0.0
                    if profile_timing:
                        prefill_ms += (runner_done - runner_start) * 1000
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                runner_start = time.perf_counter() if profile_timing else 0.0
                decode_results = self._mlx_runner.decode_batch(
                    decode_rids, decode_slots
                )
                runner_done = time.perf_counter() if profile_timing else 0.0
                if profile_timing:
                    decode_ms += (runner_done - runner_start) * 1000
                decode_map = dict(zip(decode_rids, decode_results))
            else:
                decode_map = {}

            assemble_start = time.perf_counter() if profile_timing else 0.0
            prefill_map = dict(prefill_rids)
            extend_map = dict(extend_rids)

            for req in reqs:
                if req.rid in decode_map:
                    next_token_ids_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_token_ids_list.append(extend_map[req.rid])
                else:
                    next_token_ids_list.append(prefill_map[req.rid])
            assemble_done = time.perf_counter() if profile_timing else 0.0

            if profile_timing:
                logger.info(
                    "MLX worker timing mode=%s reqs=%s input_tokens=%s "
                    "prefills=%s extends=%s decodes=%s cpu_ms=%.2f "
                    "prefill_ms=%.2f extend_ms=%.2f decode_ms=%.2f "
                    "assemble_ms=%.2f total_before_tensor_ms=%.2f",
                    getattr(forward_mode, "name", str(forward_mode)),
                    len(reqs),
                    len(input_ids_cpu),
                    len(prefill_rids),
                    len(extend_rids),
                    len(decode_rids),
                    (cpu_done - cpu_start) * 1000,
                    prefill_ms,
                    extend_ms,
                    decode_ms,
                    (assemble_done - assemble_start) * 1000,
                    (assemble_done - profile_start) * 1000,
                )

        elif forward_mode.is_decode():
            cpu_start = time.perf_counter() if profile_timing else 0.0
            req_ids = [req.rid for req in reqs]
            decode_slot_ids = model_worker_batch.out_cache_loc.cpu().tolist()
            cpu_done = time.perf_counter() if profile_timing else 0.0
            runner_start = time.perf_counter() if profile_timing else 0.0
            next_token_ids_list = self._mlx_runner.decode_batch(
                req_ids, decode_slot_ids
            )
            runner_done = time.perf_counter() if profile_timing else 0.0
            if profile_timing:
                logger.info(
                    "MLX worker timing mode=decode reqs=%s cpu_ms=%.2f "
                    "runner_ms=%.2f total_before_tensor_ms=%.2f",
                    len(reqs),
                    (cpu_done - cpu_start) * 1000,
                    (runner_done - runner_start) * 1000,
                    (runner_done - profile_start) * 1000,
                )

        else:
            raise ValueError(
                f"MLX runner does not support forward mode: {forward_mode}"
            )

        tensor_start = time.perf_counter() if profile_timing else 0.0
        next_token_ids = torch.tensor(
            next_token_ids_list, dtype=torch.long, device="cpu"
        )
        tensor_done = time.perf_counter() if profile_timing else 0.0
        if profile_timing:
            logger.info(
                "MLX worker timing result_tensor_ms=%.2f total_ms=%.2f",
                (tensor_done - tensor_start) * 1000,
                (tensor_done - profile_start) * 1000,
            )

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
