from typing import TYPE_CHECKING

import mlx.core as mx
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.tensor_bridge import mlx_to_torch, sync_torch, torch_to_mlx

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class MlxAttnBackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def _run_mlx_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        causal=False,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        sync_torch()

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx].item()
            prefill_seq_len_q = extend_prefix_lens[seq_idx].item()
            seq_len_kv = seq_lens[seq_idx].item()

            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[start_q:end_q, :, :]
            per_req_query_redudant = torch.empty(
                (seq_len_kv, per_req_query.shape[1], per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redudant[prefill_seq_len_q:, :, :] = per_req_query

            req_pool_idx = req_pool_indices[seq_idx].item()
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]
            per_req_value = v_cache[per_req_tokens]

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            mq = mx.expand_dims(torch_to_mlx(per_req_query_redudant), 0).transpose(
                0, 2, 1, 3
            )  # [1, n_heads, seq_len, head_dim]
            mk = mx.expand_dims(torch_to_mlx(per_req_key), 0).transpose(
                0, 2, 1, 3
            )  # [1, n_heads, seq_len, head_dim]
            mv = mx.expand_dims(torch_to_mlx(per_req_value), 0).transpose(
                0, 2, 1, 3
            )  # [1, n_heads, seq_len, head_dim]

            mask = "causal" if causal else None

            mo = mx.fast.scaled_dot_product_attention(
                mq, mk, mv, scale=scaling if scaling is not None else 1.0, mask=mask
            )

            to = mlx_to_torch(
                mo.transpose(0, 2, 1, 3).squeeze(0), output.device
            )  # [seq_len, n_heads, head_dim]
            output[start_q:end_q, :, :] = to[prefill_seq_len_q:, :, :]

            start_q, start_kv = end_q, end_kv

        sync_torch()

        return output

    def _run_mlx_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
    ):
        sync_torch()

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx].item()
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[start_q:end_q, :, :]

            req_pool_idx = req_pool_indices[seq_idx].item()
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]
            per_req_value = v_cache[per_req_tokens]

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            mq = mx.expand_dims(torch_to_mlx(per_req_query), 0).transpose(
                0, 2, 1, 3
            )  # [1, n_heads, 1, head_dim]
            mk = mx.expand_dims(torch_to_mlx(per_req_key), 0).transpose(
                0, 2, 1, 3
            )  # [1, n_heads, seq_len_kv, head_dim]
            mv = mx.expand_dims(torch_to_mlx(per_req_value), 0).transpose(0, 2, 1, 3)

            mo = mx.fast.scaled_dot_product_attention(
                mq, mk, mv, scale=scaling if scaling is not None else 1.0, mask=None
            )

            to = mlx_to_torch(
                mo.transpose(0, 2, 1, 3).squeeze(0), output.device
            )  # [1, n_heads, head_dim]
            output[start_q:end_q, :, :] = to

            start_q, start_kv = end_q, end_kv

        sync_torch()

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_mlx_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_mlx_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
        )

        return o

    def support_triton(self):
        return False
