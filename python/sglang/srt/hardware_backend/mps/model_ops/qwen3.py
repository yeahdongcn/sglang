"""Explicit Qwen3 attention providers for the Torch-owned MPS runner.

The providers are plain Python objects, not ``nn.Module`` instances.  They do
not own model parameters or cache storage; the model and standard ModelRunner
remain the owners of those lifecycles.  Installation binds one provider object
to each Qwen3 module, and the module's normal forward contract invokes it when
the selected plan permits the operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from sglang.kernels.ops.attention.qwen3_mps import (
    QWEN3_06B_METAL_SPEC,
    is_qwen3_attention_scale_supported,
    is_qwen3_qknorm_rope_store_backend_eligible,
    is_qwen3_radix_decode_backend_eligible,
    is_qwen3_rms_epsilon_supported,
    qwen3_qknorm_rope_store,
    qwen3_radix_decode,
)
from sglang.kernels.spec import KernelBackend


def validate_qwen3_attention_shape(module: Any) -> None:
    expected = QWEN3_06B_METAL_SPEC
    found = (
        int(getattr(module, "num_heads", -1)),
        int(getattr(module, "num_kv_heads", -1)),
        int(getattr(module, "head_dim", -1)),
    )
    expected_heads = (expected.num_q_heads, expected.num_kv_heads, expected.head_dim)
    if found != expected_heads:
        raise RuntimeError(
            "Qwen3 MPS Metal attention requires "
            f"(num_heads, num_kv_heads, head_dim)={expected_heads}; found {found}"
        )


def validate_qwen3_qkv_module(module: Any) -> None:
    """Validate contracts needed by QK norm/RoPE/KV-store only."""
    expected = QWEN3_06B_METAL_SPEC
    validate_qwen3_attention_shape(module)
    for name in ("q_norm", "k_norm"):
        weight = getattr(getattr(module, name, None), "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"Qwen3 MPS Metal attention requires {name}.weight")
        if (
            weight.device.type != "mps"
            or weight.dtype != torch.bfloat16
            or tuple(weight.shape) != (expected.head_dim,)
            or not weight.is_contiguous()
        ):
            raise RuntimeError(
                f"Qwen3 MPS Metal attention requires contiguous "
                f"MPS bf16 {name}.weight[{expected.head_dim}], found "
                f"device={weight.device}, dtype={weight.dtype}, shape={tuple(weight.shape)}"
            )
    q_epsilon = getattr(getattr(module, "q_norm", None), "variance_epsilon", None)
    k_epsilon = getattr(getattr(module, "k_norm", None), "variance_epsilon", None)
    if (
        not is_qwen3_rms_epsilon_supported(q_epsilon)
        or not is_qwen3_rms_epsilon_supported(k_epsilon)
        or float(q_epsilon) != float(k_epsilon)
    ):
        raise RuntimeError(
            "Qwen3 MPS Metal attention requires matching finite positive Q/K "
            f"RMS epsilons; found q={q_epsilon!r}, k={k_epsilon!r}"
        )
    rope = getattr(module, "rotary_emb", None)
    cache = getattr(rope, "cos_sin_cache", None)
    if (
        rope is None
        or not bool(getattr(rope, "is_neox_style", False))
        or int(getattr(rope, "rotary_dim", -1)) != expected.head_dim
        or not isinstance(cache, torch.Tensor)
        or cache.device.type != "mps"
        or cache.dtype != torch.bfloat16
        or cache.ndim != 2
        or cache.shape[1] != expected.head_dim
        or not cache.is_contiguous()
    ):
        raise RuntimeError(
            "Qwen3 MPS Metal attention requires a contiguous "
            "MPS bf16 full-dimension NeoX rotary cache"
        )


def validate_qwen3_decode_module(module: Any) -> None:
    """Validate contracts needed by Radix decode only."""
    expected = QWEN3_06B_METAL_SPEC
    validate_qwen3_attention_shape(module)
    scale = float(getattr(getattr(module, "attn", None), "scaling", 0.0))
    if not is_qwen3_attention_scale_supported(scale, expected):
        raise RuntimeError(
            "Qwen3 MPS Metal attention found an unsupported "
            f"attention scale {scale}; expected {expected.attention_scale}"
        )


def validate_qwen3_attention_module(module: Any) -> None:
    """Validate the combined QKV + decode contract for diagnostics/tests."""
    validate_qwen3_qkv_module(module)
    validate_qwen3_decode_module(module)


@dataclass
class Qwen3MpsKvPoolContract:
    """Validated physical KV storage bound to one attention layer.

    The provider deliberately stores only identity/shape metadata and raw
    pointer values, not a reference to the pool or its tensors.  Torch's
    ``ModelRunner`` therefore remains the sole owner of KV storage while a
    pool replacement is detected before a custom kernel can touch it.
    """

    pool_identity: int
    layer_id: int
    num_slots: int
    k_data_ptr: int
    v_data_ptr: int

    def validate(self, kv_pool: Any, layer_id: int, k_pool, v_pool) -> None:
        if id(kv_pool) != self.pool_identity:
            raise RuntimeError(
                "Qwen3 MPS attention detected a replaced KV pool; reinstall "
                "the model operator plan before serving requests"
            )
        if getattr(kv_pool, "kv_cache_layout", None) != "nhd":
            raise RuntimeError(
                "Qwen3 MPS attention requires the standard contiguous NHD KV "
                f"layout; found {getattr(kv_pool, 'kv_cache_layout', None)!r}"
            )
        self.validate_buffers(layer_id, k_pool, v_pool)

    def validate_buffers(self, layer_id: int, k_pool, v_pool) -> None:
        spec = QWEN3_06B_METAL_SPEC
        expected_tail = (spec.num_kv_heads, spec.head_dim)
        if int(layer_id) != self.layer_id:
            raise RuntimeError(
                "Qwen3 MPS attention provider was invoked for the wrong layer: "
                f"expected {self.layer_id}, found {layer_id}"
            )
        for name, tensor, expected_ptr in (
            ("k_pool", k_pool, self.k_data_ptr),
            ("v_pool", v_pool, self.v_data_ptr),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "mps"
                or tensor.dtype != torch.bfloat16
                or tensor.ndim != 3
                or tuple(tensor.shape) != (self.num_slots, *expected_tail)
                or not tensor.is_contiguous()
                or tensor.data_ptr() != expected_ptr
            ):
                raise RuntimeError(
                    "Qwen3 MPS attention KV storage changed after plan "
                    f"installation: {name} must be contiguous MPS bf16 "
                    f"[{self.num_slots}, {expected_tail[0]}, {expected_tail[1]}]"
                )


@dataclass
class Qwen3MpsAttentionProvider:
    """Torch-stream Metal provider for Qwen3 QKV preparation and decode."""

    pool_contract: Qwen3MpsKvPoolContract
    # QKV preparation and Radix decode are independent semantic operations.
    # ``None`` means the ordinary Torch ModelRunner path owns that operation;
    # non-None values are pinned after startup validation/warmup.
    qkv_kernel_backend: Optional[KernelBackend] = None
    decode_kernel_backend: Optional[KernelBackend] = None
    qkv_call_count: int = 0
    decode_call_count: int = 0
    qkv_fallback_count: int = 0
    decode_fallback_count: int = 0
    _forced_backend: Optional[KernelBackend] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Capture the diagnostic override at the same startup boundary where
        # the plan pins and warms this provider. Normal priorities retain their
        # explicit Torch fallback; a global force is deliberately strict.
        from sglang.kernels.fused_op import get_fused_op_backend

        self._forced_backend = get_fused_op_backend()
        if self._forced_backend is None:
            return
        for op_name, selected in (
            ("QK-norm/RoPE/store", self.qkv_kernel_backend),
            ("Radix decode", self.decode_kernel_backend),
        ):
            if selected is not None and selected is not self._forced_backend:
                raise RuntimeError(
                    f"Qwen3 MPS {op_name} provider was pinned to "
                    f"{selected.value!r}, but the global fused-op backend is "
                    f"{self._forced_backend.value!r}"
                )

    def _runtime_backend(
        self,
        preferred: Optional[KernelBackend],
        eligible: Callable[..., bool],
        *args,
        **kwargs,
    ) -> tuple[Optional[KernelBackend], bool]:
        """Return the pinned provider or its explicit Torch fallback.

        Static discovery and warmup stay outside the hot path. This checks only
        request-varying tensor/layout contracts. Global force skips eligibility
        so the requested adapter remains strict and raises on contract drift.
        """
        if (
            preferred is None
            or preferred is KernelBackend.TORCH
            or self._forced_backend is not None
        ):
            return preferred, False
        if eligible(preferred, *args, **kwargs):
            return preferred, False
        return KernelBackend.TORCH, True

    def prepare_qkv(self, module: Any, positions, hidden_states, forward_batch):
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        qkv, _ = module.qkv_proj(hidden_states)
        spec = QWEN3_06B_METAL_SPEC
        if qkv.ndim != 2 or qkv.shape[1] != spec.qkv_width:
            raise RuntimeError(
                "Qwen3 MPS Metal attention received an unexpected "
                f"QKV projection shape {tuple(qkv.shape)}"
            )
        kv_pool = get_token_to_kv_pool()
        k_pool, v_pool = kv_pool.get_kv_buffer(module.attn.layer_id)
        self.pool_contract.validate(kv_pool, module.attn.layer_id, k_pool, v_pool)
        num_tokens = qkv.shape[0]
        q_out = torch.empty(
            (num_tokens, spec.num_q_heads, spec.head_dim),
            dtype=qkv.dtype,
            device=qkv.device,
        )
        args = (
            qkv,
            module.q_norm.weight,
            module.k_norm.weight,
            module.rotary_emb.cos_sin_cache,
            positions,
            forward_batch.out_cache_loc,
            q_out,
            k_pool,
            v_pool,
        )
        kwargs = dict(
            epsilon=float(module.q_norm.variance_epsilon),
            spec=spec,
        )
        runtime_backend, used_fallback = self._runtime_backend(
            self.qkv_kernel_backend,
            is_qwen3_qknorm_rope_store_backend_eligible,
            *args,
            **kwargs,
        )
        if runtime_backend is not None:
            kwargs["backend"] = runtime_backend
        from sglang.srt.utils.async_probe import maybe_detect_oob

        maybe_detect_oob(
            positions,
            0,
            int(module.rotary_emb.cos_sin_cache.shape[0]),
            "Qwen3 MPS qknorm/rope positions",
        )
        maybe_detect_oob(
            forward_batch.out_cache_loc,
            0,
            self.pool_contract.num_slots,
            "Qwen3 MPS fused KV store slots",
        )
        qwen3_qknorm_rope_store(*args, **kwargs)
        if used_fallback:
            self.qkv_fallback_count += 1
        self.qkv_call_count += 1
        return q_out.reshape(num_tokens, -1), None, None

    def decode(
        self,
        q,
        k_pool,
        v_pool,
        req_to_token,
        req_pool_indices,
        seq_lens,
        out,
        *,
        scale,
    ):
        # The attention backend owns the pool reference; decode receives its
        # already-resolved layer buffers.  Validate their identity directly.
        self.pool_contract.validate_buffers(
            self.pool_contract.layer_id,
            k_pool,
            v_pool,
        )
        args = (
            q,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            out,
        )
        kwargs = dict(scale=scale)
        runtime_backend, used_fallback = self._runtime_backend(
            self.decode_kernel_backend,
            is_qwen3_radix_decode_backend_eligible,
            *args,
            **kwargs,
        )
        if runtime_backend is not None:
            kwargs["backend"] = runtime_backend
        from sglang.srt.utils.async_probe import (
            maybe_detect_in_closed_range,
            maybe_detect_oob,
        )

        maybe_detect_oob(
            req_pool_indices,
            0,
            int(req_to_token.shape[0]),
            "Qwen3 MPS Radix request rows",
        )
        maybe_detect_in_closed_range(
            seq_lens,
            1,
            int(req_to_token.shape[1]),
            "Qwen3 MPS Radix sequence lengths",
        )
        qwen3_radix_decode(*args, **kwargs)
        if used_fallback:
            self.decode_fallback_count += 1
        self.decode_call_count += 1


def warmup_qwen3_mps_provider(
    qkv_backend: Optional[KernelBackend] = None,
    decode_backend: Optional[KernelBackend] = None,
) -> tuple[KernelBackend, KernelBackend]:
    from sglang.kernels.ops.attention.qwen3_mps import warmup_qwen3_mps_kernels

    return warmup_qwen3_mps_kernels(qkv_backend, decode_backend)


__all__ = [
    "Qwen3MpsAttentionProvider",
    "Qwen3MpsKvPoolContract",
    "validate_qwen3_attention_shape",
    "validate_qwen3_qkv_module",
    "validate_qwen3_decode_module",
    "validate_qwen3_attention_module",
    "warmup_qwen3_mps_provider",
]
