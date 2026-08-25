"""Torch-native attention with an optional Qwen3 Metal decode provider.

All metadata, Radix page-table ownership, and KV writes stay in the normal
``TorchNativeAttnBackend``.  This subclass only consumes a provider already
bound to ``RadixAttention`` by the model-specific plan; it never imports a
model implementation or checks the MLX environment on the hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MpsAttnBackend(TorchNativeAttnBackend):
    """Torch SDPA fallback plus an explicitly bound Metal decode provider."""

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        provider = layer.decode_provider
        if provider is None:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )

        # Match the normal backend's output allocation and cache-write
        # contract.  The Qwen3 provider writes K/V during model preparation,
        # so k/v are None and there is no duplicate store here.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        if save_kv_cache and k is not None and v is not None:
            from sglang.srt.mem_cache.memory_pool import KVWriteLoc

            cache_loc = (
                forward_batch.encoder_out_cache_loc
                if layer.is_cross_attention
                else forward_batch.out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, self.swa_out_cache_loc), k, v
            )

        provider.decode(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            scale=layer.scaling,
        )
        return o


__all__ = ["MpsAttnBackend"]
