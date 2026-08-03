"""CPU-safe dynamic fallback tests for the pinned Qwen3 MPS provider."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.qwen3 import (
    Qwen3MpsAttentionProvider,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _provider(*, forced_backend=None) -> Qwen3MpsAttentionProvider:
    contract = mock.MagicMock()
    contract.layer_id = 0
    contract.num_slots = 16
    with mock.patch(
        "sglang.kernels.fused_op.get_fused_op_backend",
        return_value=forced_backend,
    ):
        return Qwen3MpsAttentionProvider(
            pool_contract=contract,
            qkv_kernel_backend=KernelBackend.METAL_JIT,
            decode_kernel_backend=KernelBackend.METAL_JIT,
        )


def _qkv_inputs():
    spec = QWEN3_06B_METAL_SPEC
    qkv = torch.empty(1, spec.qkv_width, dtype=torch.bfloat16)
    pool = torch.empty(16, spec.num_kv_heads, spec.head_dim, dtype=torch.bfloat16)
    kv_pool = mock.MagicMock()
    kv_pool.get_kv_buffer.return_value = (pool, pool.clone())
    module = SimpleNamespace(
        qkv_proj=lambda _hidden: (qkv, None),
        q_norm=SimpleNamespace(
            weight=torch.ones(spec.head_dim, dtype=torch.bfloat16),
            variance_epsilon=1e-6,
        ),
        k_norm=SimpleNamespace(weight=torch.ones(spec.head_dim, dtype=torch.bfloat16)),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.empty(8, spec.head_dim, dtype=torch.bfloat16)
        ),
        attn=SimpleNamespace(layer_id=0),
    )
    forward_batch = SimpleNamespace(out_cache_loc=torch.tensor([3], dtype=torch.int64))
    return module, forward_batch, kv_pool


def test_qkv_dynamic_miss_uses_torch_and_counts_fallback():
    provider = _provider()
    module, forward_batch, kv_pool = _qkv_inputs()

    with (
        mock.patch(
            "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
            return_value=kv_pool,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3."
            "is_qwen3_qknorm_rope_store_backend_eligible",
            return_value=False,
        ) as eligible,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3.qwen3_qknorm_rope_store"
        ) as run,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
    ):
        provider.prepare_qkv(
            module, torch.tensor([0]), torch.empty(1, 1), forward_batch
        )

    eligible.assert_called_once()
    assert run.call_args.kwargs["backend"] is KernelBackend.TORCH
    assert provider.qkv_call_count == 1
    assert provider.qkv_fallback_count == 1


def test_decode_dynamic_miss_uses_torch_and_counts_fallback():
    provider = _provider()
    spec = QWEN3_06B_METAL_SPEC
    q = torch.empty(1, spec.num_q_heads, spec.head_dim, dtype=torch.bfloat16)
    pool = torch.empty(16, spec.num_kv_heads, spec.head_dim, dtype=torch.bfloat16)
    table = torch.zeros(2, 8, dtype=torch.int32)
    request = torch.tensor([1], dtype=torch.int64)
    lengths = torch.tensor([2], dtype=torch.int64)
    out = torch.empty_like(q)

    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3."
            "is_qwen3_radix_decode_backend_eligible",
            return_value=False,
        ) as eligible,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3.qwen3_radix_decode"
        ) as run,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_in_closed_range"),
    ):
        provider.decode(
            q,
            pool,
            pool.clone(),
            table,
            request,
            lengths,
            out,
            scale=spec.attention_scale,
        )

    eligible.assert_called_once()
    assert run.call_args.kwargs["backend"] is KernelBackend.TORCH
    assert provider.decode_call_count == 1
    assert provider.decode_fallback_count == 1


def test_global_force_keeps_provider_strict_without_dynamic_fallback():
    provider = _provider(forced_backend=KernelBackend.METAL_JIT)
    module, forward_batch, kv_pool = _qkv_inputs()

    with (
        mock.patch(
            "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
            return_value=kv_pool,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3."
            "is_qwen3_qknorm_rope_store_backend_eligible"
        ) as eligible,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3.qwen3_qknorm_rope_store"
        ) as run,
        mock.patch("sglang.srt.utils.async_probe.maybe_detect_oob"),
    ):
        provider.prepare_qkv(
            module, torch.tensor([0]), torch.empty(1, 1), forward_batch
        )

    eligible.assert_not_called()
    assert run.call_args.kwargs["backend"] is KernelBackend.METAL_JIT
    assert provider.qkv_call_count == 1
    assert provider.qkv_fallback_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
