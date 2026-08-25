"""Qwen3-0.6B Metal attention plan for the standard Torch ModelRunner.

The plan owns only provider selection and bindings. Torch continues to own the
model parameters, KV pools, request tables, and Radix cache. Both semantic
operators default to Torch and can be enabled independently at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.qwen3 import (
    QWEN3_06B_METAL_SPEC,
    Qwen3MpsAttentionProvider,
    Qwen3MpsKvPoolContract,
    validate_qwen3_decode_module,
    validate_qwen3_qkv_module,
    warmup_qwen3_mps_provider,
)
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    Qwen3MetalAttentionSelection,
    choose_kernel_backend,
)

logger = logging.getLogger(__name__)

QWEN3_MPS_MODEL = "Qwen3ForCausalLM"


@dataclass
class Qwen3MetalAttentionPlan:
    """Lifecycle and diagnostics for one runner's Qwen3 attention providers."""

    model: str = QWEN3_MPS_MODEL
    qkv_kernel_backend: str = "torch"
    decode_kernel_backend: str = "torch"
    qkv_fallback_reason: Optional[str] = None
    decode_fallback_reason: Optional[str] = None
    provider_priorities: dict[str, list[str]] = field(default_factory=dict)
    _bindings: tuple[tuple[Any, Qwen3MpsAttentionProvider], ...] = field(
        default=(), repr=False, compare=False
    )
    _closed: bool = field(default=False, repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return bool(self._bindings) and not self._closed

    def close(self) -> None:
        """Remove only bindings still owned by this plan."""
        if self._closed:
            return
        for module, provider in self._bindings:
            if getattr(module, "op_provider", None) is provider:
                module.op_provider = None
            attention = getattr(module, "attn", None)
            if getattr(attention, "decode_provider", None) is provider:
                attention.decode_provider = None
        self._bindings = ()
        self._closed = True

    def get_state(self) -> dict[str, Any]:
        providers = [provider for _, provider in self._bindings]
        return {
            "enabled": self.enabled,
            "model": self.model,
            "provider_priorities": self.provider_priorities,
            "qkv_kernel_backend": self.qkv_kernel_backend,
            "decode_kernel_backend": self.decode_kernel_backend,
            "qkv_fallback_reason": self.qkv_fallback_reason,
            "decode_fallback_reason": self.decode_fallback_reason,
            "patched_qkv_modules": sum(
                provider.qkv_kernel_backend is not None for provider in providers
            ),
            "patched_decode_modules": sum(
                provider.decode_kernel_backend is not None for provider in providers
            ),
            "qkv_call_count": sum(provider.qkv_call_count for provider in providers),
            "qkv_fallback_count": sum(
                provider.qkv_fallback_count for provider in providers
            ),
            "decode_call_count": sum(
                provider.decode_call_count for provider in providers
            ),
            "decode_fallback_count": sum(
                provider.decode_fallback_count for provider in providers
            ),
        }


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _hf_configs(model_config: Any):
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if config is not None:
            yield config


def model_is_qwen3_06b(model_config: Any) -> bool:
    architectures = {
        str(architecture)
        for config in _hf_configs(model_config)
        for architecture in (_config_value(config, "architectures", None) or ())
    }
    if architectures != {QWEN3_MPS_MODEL}:
        return False
    return any(
        int(_config_value(config, "hidden_size", -1)) == 1024
        and int(_config_value(config, "intermediate_size", -1)) == 3072
        for config in _hf_configs(model_config)
    )


def _model_num_hidden_layers(model_config: Any) -> Optional[int]:
    for config in _hf_configs(model_config):
        value = _config_value(config, "num_hidden_layers", None)
        if value is not None:
            return int(value)
    return None


def _validate_full_attention_config(model_config: Any) -> None:
    for config in _hf_configs(model_config):
        if bool(_config_value(config, "use_sliding_window", False)):
            raise RuntimeError("Qwen3 Metal attention requires full attention")
        for field_name in ("sliding_window", "attention_chunk_size", "window_size"):
            value = _config_value(config, field_name, None)
            if value not in (None, 0, -1):
                raise RuntimeError(
                    "Qwen3 Metal attention requires full attention; "
                    f"unsupported {field_name}={value!r}"
                )


def _qwen3_modules(model: torch.nn.Module) -> list[Any]:
    from sglang.srt.models.qwen3 import Qwen3Attention

    return [module for module in model.modules() if isinstance(module, Qwen3Attention)]


def _attention_backends(server_args: Any) -> tuple[str, str]:
    def normalize(value: Any) -> Optional[str]:
        value = getattr(value, "value", value)
        return None if value is None else str(value).lower()

    get_backends = getattr(server_args, "get_attention_backends", None)
    if callable(get_backends):
        prefill, decode = get_backends()
    else:
        common = getattr(server_args, "attention_backend", None)
        prefill = getattr(server_args, "prefill_attention_backend", None) or common
        decode = getattr(server_args, "decode_attention_backend", None) or common
    return normalize(prefill) or "mps", normalize(decode) or "mps"


def _validate_kv_pool_contract(
    token_to_kv_pool: Any, attentions: list[Any]
) -> list[Qwen3MpsKvPoolContract]:
    if token_to_kv_pool is None:
        raise RuntimeError("Qwen3 Metal attention requires an allocated Torch KV pool")
    if getattr(token_to_kv_pool, "kv_cache_layout", None) != "nhd":
        raise RuntimeError("Qwen3 Metal attention requires the contiguous NHD KV pool")
    if bool(getattr(token_to_kv_pool, "is_quantized_kv_cache", False)):
        raise RuntimeError("Qwen3 Metal attention does not support quantized KV cache")

    expected_tail = (
        QWEN3_06B_METAL_SPEC.num_kv_heads,
        QWEN3_06B_METAL_SPEC.head_dim,
    )
    contracts = []
    for module in attentions:
        layer_id = int(module.attn.layer_id)
        k_pool, v_pool = token_to_kv_pool.get_kv_buffer(layer_id)
        for name, tensor in (("K", k_pool), ("V", v_pool)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "mps"
                or tensor.dtype != torch.bfloat16
                or tensor.ndim != 3
                or tuple(tensor.shape[1:]) != expected_tail
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    "Qwen3 Metal attention requires contiguous MPS bf16 NHD "
                    f"storage; layer={layer_id} {name} has "
                    f"device={getattr(tensor, 'device', None)}, "
                    f"dtype={getattr(tensor, 'dtype', None)}, "
                    f"shape={tuple(getattr(tensor, 'shape', ()))}"
                )
        if tuple(v_pool.shape) != tuple(k_pool.shape):
            raise RuntimeError(
                f"Qwen3 Metal K/V pool shapes differ at layer {layer_id}: "
                f"{tuple(k_pool.shape)} vs {tuple(v_pool.shape)}"
            )
        contracts.append(
            Qwen3MpsKvPoolContract(
                pool_identity=id(token_to_kv_pool),
                layer_id=layer_id,
                num_slots=int(k_pool.shape[0]),
                k_data_ptr=int(k_pool.data_ptr()),
                v_data_ptr=int(v_pool.data_ptr()),
            )
        )
    return contracts


def _fallback_plan(
    reason: str,
    *,
    model: str = QWEN3_MPS_MODEL,
    selection: Qwen3MetalAttentionSelection | None = None,
) -> Qwen3MetalAttentionPlan:
    logger.info("Qwen3 Metal attention uses Torch: %s", reason)
    return Qwen3MetalAttentionPlan(
        model=model,
        qkv_fallback_reason=reason,
        decode_fallback_reason=reason,
        provider_priorities=selection.as_state() if selection else {},
    )


def _publish_bindings(
    attentions: list[Any], providers: list[Qwen3MpsAttentionProvider]
) -> None:
    published: list[tuple[Any, str, Any]] = []
    try:
        for module, provider in zip(attentions, providers):
            if provider.qkv_kernel_backend is not None:
                published.append((module, "op_provider", module.op_provider))
                module.op_provider = provider
            if provider.decode_kernel_backend is not None:
                published.append(
                    (module.attn, "decode_provider", module.attn.decode_provider)
                )
                module.attn.decode_provider = provider
    except Exception:
        for target, attribute, previous in reversed(published):
            try:
                setattr(target, attribute, previous)
            except Exception:
                logger.exception("Failed to roll back Qwen3 Metal provider binding")
        raise


def install_qwen3_metal_attention(
    model: torch.nn.Module,
    model_config: Any,
    server_args: Any,
    *,
    token_to_kv_pool: Any = None,
    **_: Any,
) -> Qwen3MetalAttentionPlan:
    """Validate, warm, and atomically bind the selected Qwen3 providers."""
    if not model_is_qwen3_06b(model_config):
        return _fallback_plan(
            "model is outside the dense Qwen3-0.6B Metal attention contract",
            model=type(model).__name__,
        )

    selection = Qwen3MetalAttentionSelection.from_env()
    from sglang.kernels.metal import is_metal_jit_available
    from sglang.kernels.ops.attention.qwen3_mps import (
        is_qwen3_metal_aot_available,
    )

    aot_available = is_qwen3_metal_aot_available()
    jit_available = is_metal_jit_available()
    selected_qkv = choose_kernel_backend(
        selection.qknorm_rope_store,
        aot_available=aot_available,
        jit_available=jit_available,
    )
    selected_decode = choose_kernel_backend(
        selection.radix_decode,
        aot_available=aot_available,
        jit_available=jit_available,
    )
    qkv_reason = (
        "requested Metal providers are unavailable"
        if selected_qkv is KernelBackend.TORCH
        and selection.qknorm_rope_store[0] is not KernelBackend.TORCH
        else None
    )
    decode_reason = (
        "requested Metal providers are unavailable"
        if selected_decode is KernelBackend.TORCH
        and selection.radix_decode[0] is not KernelBackend.TORCH
        else None
    )

    prefill_backend, decode_backend = _attention_backends(server_args)
    if prefill_backend not in {"mps", "torch_native"}:
        return _fallback_plan(
            f"unsupported MPS prefill attention backend {prefill_backend!r}",
            selection=selection,
        )
    if decode_backend not in {"mps", "torch_native"}:
        return _fallback_plan(
            f"unsupported MPS decode attention backend {decode_backend!r}",
            selection=selection,
        )
    if decode_backend == "torch_native" and selected_decode is not KernelBackend.TORCH:
        selected_decode = KernelBackend.TORCH
        decode_reason = (
            "decode attention backend 'torch_native' does not consume the "
            "Qwen3 Metal Radix decode provider"
        )

    if selected_qkv is KernelBackend.TORCH and selected_decode is KernelBackend.TORCH:
        return Qwen3MetalAttentionPlan(
            qkv_fallback_reason=qkv_reason,
            decode_fallback_reason=decode_reason,
            provider_priorities=selection.as_state(),
        )

    try:
        _validate_full_attention_config(model_config)
        attentions = _qwen3_modules(model)
        expected_layers = _model_num_hidden_layers(model_config)
        if not attentions:
            raise RuntimeError("no Qwen3Attention modules were found")
        if expected_layers is not None and len(attentions) != expected_layers:
            raise RuntimeError(
                f"expected {expected_layers} Qwen3Attention modules, "
                f"found {len(attentions)}"
            )
    except RuntimeError as exc:
        return _fallback_plan(str(exc), selection=selection)

    if selected_qkv is not KernelBackend.TORCH:
        try:
            for module in attentions:
                validate_qwen3_qkv_module(module)
        except RuntimeError as exc:
            selected_qkv = KernelBackend.TORCH
            qkv_reason = str(exc)
    if selected_decode is not KernelBackend.TORCH:
        try:
            for module in attentions:
                validate_qwen3_decode_module(module)
        except RuntimeError as exc:
            selected_decode = KernelBackend.TORCH
            decode_reason = str(exc)

    if selected_qkv is KernelBackend.TORCH and selected_decode is KernelBackend.TORCH:
        return Qwen3MetalAttentionPlan(
            qkv_fallback_reason=qkv_reason,
            decode_fallback_reason=decode_reason,
            provider_priorities=selection.as_state(),
        )

    try:
        contracts = _validate_kv_pool_contract(token_to_kv_pool, attentions)
    except RuntimeError as exc:
        if selected_qkv is not KernelBackend.TORCH:
            qkv_reason = str(exc)
        if selected_decode is not KernelBackend.TORCH:
            decode_reason = str(exc)
        return Qwen3MetalAttentionPlan(
            qkv_fallback_reason=qkv_reason,
            decode_fallback_reason=decode_reason,
            provider_priorities=selection.as_state(),
        )

    # Compilation/load failures after selection are startup errors. Do not
    # silently change a benchmarked provider to Torch.
    warmup_qwen3_mps_provider(selected_qkv, selected_decode)
    providers = [
        Qwen3MpsAttentionProvider(
            pool_contract=contract,
            qkv_kernel_backend=(
                selected_qkv if selected_qkv is not KernelBackend.TORCH else None
            ),
            decode_kernel_backend=(
                selected_decode if selected_decode is not KernelBackend.TORCH else None
            ),
        )
        for contract in contracts
    ]
    _publish_bindings(attentions, providers)

    plan = Qwen3MetalAttentionPlan(
        qkv_kernel_backend=selected_qkv.value,
        decode_kernel_backend=selected_decode.value,
        qkv_fallback_reason=qkv_reason,
        decode_fallback_reason=decode_reason,
        provider_priorities=selection.as_state(),
        _bindings=tuple(zip(attentions, providers)),
    )
    logger.info(
        "Installed Qwen3 Metal attention plan: qkv=%s decode=%s layers=%d; "
        "Torch owns weights, KV pools, and Radix cache",
        selected_qkv.value,
        selected_decode.value,
        len(providers),
    )
    return plan


__all__ = [
    "QWEN3_MPS_MODEL",
    "Qwen3MetalAttentionPlan",
    "install_qwen3_metal_attention",
    "model_is_qwen3_06b",
]
