"""Model-owned operator plans for the in-tree Torch MPS platform.

This module is deliberately independent of the MLX runtime.  It discovers the
typed Qwen3 model modules and binds Torch-stream Metal providers when their
static contract is supported.  The standard ``ModelRunner`` remains the owner
of weights, scheduler state, KV pools, and Radix prefix cache.

For the exact Qwen3-0.6B contract this plan can install independently selected
semantic providers and, when explicitly listed, one coarse whole-transformer
MLX island.  The default for every operation is Torch.  A custom provider is
validated and warmed once, then pinned; a later request never re-reads an
environment variable or crosses frameworks per layer.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.base import (
    MPS_OPERATOR_FORWARD_LOCK as _MPS_OPERATOR_FORWARD_LOCK,
)
from sglang.srt.hardware_backend.mps.model_ops.base import (
    MpsBindingPublication,
)
from sglang.srt.hardware_backend.mps.model_ops.base import (
    configure_generic_mps_ops as _configure_generic_mps_ops,
)
from sglang.srt.hardware_backend.mps.model_ops.base import (
    validate_mps_operator_plan,
)
from sglang.srt.hardware_backend.mps.model_ops.qwen3 import (
    QWEN3_06B_METAL_SPEC,
    Qwen3MpsAttentionProvider,
    Qwen3MpsKvPoolContract,
    validate_qwen3_decode_module,
    validate_qwen3_qkv_module,
    warmup_qwen3_mps_provider,
)
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    MpsOperatorSelection,
    choose_kernel_backend,
    choose_model_backend,
)

logger = logging.getLogger(__name__)

QWEN3_MPS_MODEL = "Qwen3ForCausalLM"
_QWEN3_PROVIDER_SPEC = "qwen3_dense"
_QWEN3_INSTALLER_PATH = f"{__name__}:install_qwen3_operators"


@dataclass
class MpsOperatorPlan:
    """Observable lifecycle owner for one model-specific MPS operator plan."""

    model: str = QWEN3_MPS_MODEL
    provider_spec: Optional[str] = None
    patched_attention_modules: int = 0
    patched_qkv_modules: int = 0
    patched_decode_modules: int = 0
    enabled: bool = False
    attention_backend: str = "torch_native"
    qkv_kernel_backend: str = "torch"
    decode_kernel_backend: str = "torch"
    deferred_kv_commit_backend: str = "torch"
    qkv_fallback_reason: Optional[str] = None
    decode_fallback_reason: Optional[str] = None
    whole_model_backend: str = "off"
    whole_model_fallback_reason: Optional[str] = None
    provider_priorities: dict[str, Any] = field(default_factory=dict)
    generic_kernel_backends: dict[str, str] = field(default_factory=dict)
    # Every MPS plan shares one stable lock, including a pure-Torch fallback
    # plan. ModelRunner can therefore keep the same serving-visible lock
    # across an online model replacement instead of opening an unlocked gap
    # while a newly selected MLX provider is compiled and warmed.
    forward_lock: threading.RLock = field(
        default_factory=lambda: _MPS_OPERATOR_FORWARD_LOCK,
        repr=False,
        compare=False,
    )
    _model: Optional[torch.nn.Module] = field(default=None, repr=False, compare=False)
    _attention_bindings: tuple[tuple[Any, Any], ...] = field(
        default=(), repr=False, compare=False
    )
    _model_forward_provider: Any = field(default=None, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    def invalidate_views(self) -> None:
        """Invalidate borrowed MLX storage after an in-place weight update."""
        if self._closed or self._model_forward_provider is None:
            return
        lock = self.forward_lock or _MPS_OPERATOR_FORWARD_LOCK
        with lock:
            self._model_forward_provider.invalidate_views()

    def close(self) -> None:
        """Release providers without clearing bindings installed by a new plan."""
        if self._closed:
            return
        lock = self.forward_lock or _MPS_OPERATOR_FORWARD_LOCK
        with lock:
            provider = self._model_forward_provider
            qwen3_model = getattr(self._model, "model", None)
            if provider is not None:
                if getattr(qwen3_model, "model_forward_provider", None) is provider:
                    qwen3_model.model_forward_provider = None
                provider.close()
            for module, attention_provider in self._attention_bindings:
                if getattr(module, "op_provider", None) is attention_provider:
                    module.op_provider = None
                attention = getattr(module, "attn", None)
                if getattr(attention, "decode_provider", None) is attention_provider:
                    attention.decode_provider = None
            self._model_forward_provider = None
            self._attention_bindings = ()
            self._closed = True

    def get_state(self) -> dict[str, Any]:
        """Return stable server-info fields without exposing provider objects."""
        provider = self._model_forward_provider
        compiled = (
            provider.get_compiled_decode_state()
            if provider is not None
            else {
                "enabled": False,
                "total_enabled": False,
                "primary_variant": "off",
                "warmup_count": 0,
                "call_count": 0,
                "total_warmup_count": 0,
                "total_call_count": 0,
                "fallback_count": 0,
                "greedy_enabled": False,
                "greedy_warmup_count": 0,
                "greedy_call_count": 0,
            }
        )
        attention_providers = [provider for _, provider in self._attention_bindings]
        return {
            "enabled": self.enabled and not self._closed,
            "model": self.model,
            "provider_spec": self.provider_spec,
            "provider_priorities": self.provider_priorities,
            "generic_kernel_backends": self.generic_kernel_backends,
            "attention_backend": self.attention_backend,
            "qkv_kernel_backend": self.qkv_kernel_backend,
            "decode_kernel_backend": self.decode_kernel_backend,
            "deferred_kv_commit_backend": self.deferred_kv_commit_backend,
            "patched_attention_modules": self.patched_attention_modules,
            "patched_qkv_modules": self.patched_qkv_modules,
            "patched_decode_modules": self.patched_decode_modules,
            "attention_qkv_call_count": sum(
                int(getattr(item, "qkv_call_count", 0)) for item in attention_providers
            ),
            "attention_qkv_fallback_count": sum(
                int(getattr(item, "qkv_fallback_count", 0))
                for item in attention_providers
            ),
            "attention_decode_call_count": sum(
                int(getattr(item, "decode_call_count", 0))
                for item in attention_providers
            ),
            "attention_decode_fallback_count": sum(
                int(getattr(item, "decode_fallback_count", 0))
                for item in attention_providers
            ),
            "qkv_fallback_reason": self.qkv_fallback_reason,
            "decode_fallback_reason": self.decode_fallback_reason,
            "whole_model_backend": self.whole_model_backend,
            "whole_model_fallback_reason": self.whole_model_fallback_reason,
            "whole_model_call_count": int(getattr(provider, "call_count", 0)),
            "whole_model_decode_call_count": int(
                getattr(provider, "decode_call_count", 0)
            ),
            "whole_model_max_decode_batch_size": int(
                getattr(provider, "max_decode_batch_size", 0)
            ),
            "whole_model_prefill_call_count": int(
                getattr(provider, "prefill_call_count", 0)
            ),
            "whole_model_selector_call_count": int(
                getattr(provider, "selector_call_count", 0)
            ),
            "whole_model_selector_fallback_count": int(
                getattr(provider, "selector_fallback_count", 0)
            ),
            "whole_model_last_fallback_reason": getattr(
                provider, "last_selector_fallback_reason", None
            ),
            "whole_model_greedy_tail_enabled": bool(
                provider is not None
                and getattr(provider, "greedy_tail_static_fallback_reason", None)
                is None
            ),
            "whole_model_greedy_tail_backend": (
                getattr(provider, "greedy_tail_backend", "torch")
                if provider is not None
                else "off"
            ),
            "whole_model_greedy_tail_static_fallback_reason": getattr(
                provider, "greedy_tail_static_fallback_reason", None
            ),
            "whole_model_greedy_tail_call_count": int(
                getattr(provider, "greedy_tail_call_count", 0)
            ),
            "whole_model_greedy_tail_torch_call_count": int(
                getattr(provider, "greedy_tail_torch_call_count", 0)
            ),
            "whole_model_greedy_tail_fallback_count": int(
                getattr(provider, "greedy_tail_fallback_count", 0)
            ),
            "whole_model_greedy_tail_last_fallback_reason": getattr(
                provider, "last_greedy_tail_fallback_reason", None
            ),
            "whole_model_compile_enabled": bool(compiled["enabled"]),
            "whole_model_compile_total_enabled": bool(compiled["total_enabled"]),
            "whole_model_compile_primary_variant": compiled["primary_variant"],
            "whole_model_compile_warmup_count": int(compiled["warmup_count"]),
            "whole_model_compile_call_count": int(compiled["call_count"]),
            "whole_model_compile_total_warmup_count": int(
                compiled["total_warmup_count"]
            ),
            "whole_model_compile_total_call_count": int(compiled["total_call_count"]),
            "whole_model_compile_fallback_count": int(compiled["fallback_count"]),
            "whole_model_greedy_compile_enabled": bool(compiled["greedy_enabled"]),
            "whole_model_greedy_compile_warmup_count": int(
                compiled["greedy_warmup_count"]
            ),
            "whole_model_greedy_compile_call_count": int(compiled["greedy_call_count"]),
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
    architectures = set()
    for config in _hf_configs(model_config):
        architectures.update(_config_value(config, "architectures", None) or [])
    if architectures != {QWEN3_MPS_MODEL}:
        return False
    for config in _hf_configs(model_config):
        hidden_size = _config_value(config, "hidden_size")
        intermediate_size = _config_value(config, "intermediate_size")
        if hidden_size is not None and intermediate_size is not None:
            return int(hidden_size) == 1024 and int(intermediate_size) == 3072
    return False


def _model_num_hidden_layers(model_config: Any) -> Optional[int]:
    for config in _hf_configs(model_config):
        value = _config_value(config, "num_hidden_layers")
        if value is not None:
            return int(value)
    return None


def _validate_full_attention_config(model_config: Any) -> None:
    for config in _hf_configs(model_config):
        if bool(_config_value(config, "use_sliding_window", False)):
            raise RuntimeError("Qwen3 MPS operators require full attention")
        for field_name in ("sliding_window", "attention_chunk_size", "window_size"):
            value = _config_value(config, field_name, None)
            if value not in (None, 0, -1):
                raise RuntimeError(
                    "Qwen3 MPS operators require full attention; "
                    f"unsupported {field_name}={value!r}"
                )


def _whole_model_mlx_fallback_reason(
    model_config: Any,
    server_args: Any,
    forced_backend: Optional[KernelBackend],
) -> Optional[str]:
    """Return a normal static selector miss for the coarse MLX island."""
    if forced_backend is not None:
        return (
            f"explicit fused-op backend {forced_backend.value!r} disables "
            "whole-model MLX selection"
        )
    # Dumper relies on nn.Module hooks registered before this plan is bound.
    # A whole-transformer island bypasses each decoder layer's __call__, so
    # selecting it would silently produce an incomplete dump. Keep the normal
    # Torch module path while Dumper can be enabled, including HTTP control.
    from sglang.srt.debug_utils.dumper import dumper

    if dumper.may_enable:
        return "Dumper observability requires the ordinary Torch module path"
    try:
        _validate_full_attention_config(model_config)
    except RuntimeError as exc:
        return str(exc)
    model_dtype = getattr(model_config, "dtype", None)
    if model_dtype is not None and model_dtype != torch.bfloat16:
        return f"model dtype {model_dtype} is not bfloat16"
    quantization = getattr(model_config, "quantization", None)
    if quantization not in (None, "unquant"):
        return f"quantization {quantization!r} is not supported"
    if not bool(getattr(server_args, "disable_overlap_schedule", True)):
        return "overlap scheduling has no cross-framework queue ownership contract"
    for field_name in (
        "enable_deterministic_inference",
        "enable_torch_compile",
        "enable_memory_saver",
        "enable_lora",
        "forward_hooks",
        "enable_layerwise_nvtx_marker",
        "debug_tensor_dump_output_folder",
        "msprobe_dump_config",
        "torchao_config",
    ):
        if getattr(server_args, field_name, None):
            return f"{field_name} requires the ordinary Torch module path"
    if getattr(server_args, "lora_paths", None):
        return "LoRA requires the ordinary Torch module path"
    if float(getattr(server_args, "cpu_offload_gb", 0) or 0) > 0:
        return "CPU-offloaded weights cannot be borrowed by MLX"
    if str(getattr(server_args, "weight_cache_mode", "off")) != "off":
        return "weight-cache storage can invalidate MLX borrows"
    return None


def _qwen3_modules(model: torch.nn.Module):
    """Discover the explicit model classes, never by class-name matching."""
    from sglang.srt.models.qwen3 import Qwen3Attention

    attentions = [
        module for module in model.modules() if isinstance(module, Qwen3Attention)
    ]
    return attentions


def _server_args_support_mps(server_args: Any) -> tuple[str, str]:
    """Validate only constraints needed by the Torch-stream provider."""
    device = getattr(server_args, "device", "mps")
    if device is not None and str(device).split(":", 1)[0].lower() != "mps":
        raise RuntimeError(f"Qwen3 MPS operators require an MPS device; got {device!r}")

    def normalize(value: Any) -> Optional[str]:
        value = getattr(value, "value", value)
        return None if value is None else str(value).lower()

    get_backends = getattr(server_args, "get_attention_backends", None)
    if callable(get_backends):
        prefill_backend, decode_backend = get_backends()
    else:
        common = getattr(server_args, "attention_backend", None)
        prefill_backend = getattr(server_args, "prefill_attention_backend", None)
        decode_backend = getattr(server_args, "decode_attention_backend", None)
        prefill_backend = prefill_backend or common
        decode_backend = decode_backend or common
    prefill_backend = normalize(prefill_backend) or "mps"
    decode_backend = normalize(decode_backend) or "mps"
    for phase, backend in (
        ("prefill", prefill_backend),
        ("decode", decode_backend),
    ):
        if backend not in {"mps", "torch_native"}:
            raise RuntimeError(
                "Qwen3 MPS operators require the mps or torch_native attention "
                f"backend; got {phase}={backend!r}"
            )
    if any(
        getattr(server_args, field, 1) != 1
        for field in ("tp_size", "pp_size", "dp_size")
    ):
        raise RuntimeError(
            "Qwen3 MPS operators currently require tp_size=pp_size=dp_size=1"
        )
    return prefill_backend, decode_backend


def _fallback(
    reason: str,
    *,
    model: str = QWEN3_MPS_MODEL,
    provider_spec: Optional[str] = None,
    selection: Optional[MpsOperatorSelection] = None,
    generic_backends: Optional[dict[str, KernelBackend]] = None,
) -> MpsOperatorPlan:
    logger.info("MPS model-op plan falls back to Torch native: %s", reason)
    generic_backends = generic_backends or {}
    return MpsOperatorPlan(
        model=model,
        provider_spec=provider_spec,
        enabled=any(
            backend is not KernelBackend.TORCH for backend in generic_backends.values()
        ),
        qkv_fallback_reason=reason,
        decode_fallback_reason=reason,
        whole_model_fallback_reason=reason,
        provider_priorities=(selection.as_state() if selection else {}),
        generic_kernel_backends={
            name: backend.value for name, backend in generic_backends.items()
        },
    )


def _validate_kv_pool_contract(token_to_kv_pool: Any, attentions: list[Any]):
    """Resolve the physical NHD buffers before publishing any provider."""
    if token_to_kv_pool is None:
        raise RuntimeError(
            "the Torch KV pool has not been allocated; install MPS operators "
            "after ModelRunner.alloc_memory_pool()"
        )
    layout = getattr(token_to_kv_pool, "kv_cache_layout", None)
    if layout != "nhd":
        raise RuntimeError(
            "Qwen3 MPS operators require the standard contiguous NHD KV pool; "
            f"found layout={layout!r}"
        )
    if bool(getattr(token_to_kv_pool, "is_quantized_kv_cache", False)):
        raise RuntimeError("Qwen3 MPS operators do not support a quantized KV pool")

    spec = QWEN3_06B_METAL_SPEC
    expected_tail = (spec.num_kv_heads, spec.head_dim)
    contracts = []
    for module in attentions:
        layer_id = int(module.attn.layer_id)
        k_pool, v_pool = token_to_kv_pool.get_kv_buffer(layer_id)
        expected_shape = (int(k_pool.shape[0]), *expected_tail)
        for name, tensor in (("K", k_pool), ("V", v_pool)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "mps"
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != expected_shape
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    "Qwen3 MPS operators require per-layer contiguous MPS bf16 "
                    f"NHD buffers shaped [slots, {expected_tail[0]}, "
                    f"{expected_tail[1]}]; layer={layer_id} {name} has "
                    f"device={getattr(tensor, 'device', None)}, "
                    f"dtype={getattr(tensor, 'dtype', None)}, "
                    f"shape={tuple(getattr(tensor, 'shape', ()))}"
                )
        if tuple(v_pool.shape) != tuple(k_pool.shape):
            raise RuntimeError(
                f"Qwen3 MPS K/V pool shapes differ at layer {layer_id}: "
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


def install_qwen3_operators(
    model: torch.nn.Module,
    model_config: Any,
    server_args: Any,
    *,
    req_to_token_pool: Any = None,
    token_to_kv_pool: Any = None,
) -> MpsOperatorPlan:
    """Build and atomically bind the per-operation MPS provider plan."""

    # Parse all gates before touching model objects.  Invalid configuration is
    # a startup error, while a valid but ineligible model simply receives the
    # ordinary Torch path.
    selection = MpsOperatorSelection.from_env()
    forced_backend = get_fused_op_backend()
    generic_backends = _configure_generic_mps_ops(selection, forced_backend)

    if not model_is_qwen3_06b(model_config):
        reason = "model is outside the dense Qwen3-0.6B operator contract"
        architectures = {
            str(architecture)
            for config in _hf_configs(model_config)
            for architecture in (_config_value(config, "architectures", None) or [])
        }
        model_name = (
            next(iter(architectures))
            if len(architectures) == 1
            else type(model).__name__
        )
        return _fallback(
            reason,
            model=model_name,
            provider_spec=_QWEN3_PROVIDER_SPEC,
            selection=selection,
            generic_backends=generic_backends,
        )

    try:
        prefill_backend, decode_backend = _server_args_support_mps(server_args)
    except RuntimeError as exc:
        return _fallback(
            str(exc),
            selection=selection,
            generic_backends=generic_backends,
        )

    attention_backend = (
        prefill_backend
        if prefill_backend == decode_backend
        else f"prefill={prefill_backend},decode={decode_backend}"
    )
    from sglang.kernels.ops.attention.qwen3_mps import (
        is_qwen3_metal_aot_available,
        set_qwen3_mps_priority,
    )
    from sglang.kernels.ops.kvcache import set_qwen3_kv_commit_priority

    set_qwen3_mps_priority(
        selection.qknorm_rope_store,
        selection.radix_decode,
    )
    set_qwen3_kv_commit_priority(selection.deferred_kv_commit)

    # Resolve static provider availability independently.  AOT absence may
    # skip to a later listed provider; once a provider is selected, warmup
    # failures remain fatal and cannot silently change the serving path.
    aot_available = None
    if (
        KernelBackend.METAL_AOT in selection.qknorm_rope_store
        or KernelBackend.METAL_AOT in selection.radix_decode
        or forced_backend is KernelBackend.METAL_AOT
    ):
        aot_available = is_qwen3_metal_aot_available()

    qkv_fallback_reason: Optional[str] = None
    decode_fallback_reason: Optional[str] = None
    supported_forced = {
        None,
        KernelBackend.METAL_AOT,
        KernelBackend.METAL_JIT,
        KernelBackend.TORCH,
    }
    if forced_backend not in supported_forced:
        qkv_fallback_reason = decode_fallback_reason = (
            "the Qwen3 attention provider does not support forced fused-op "
            f"backend {forced_backend.value!r}"
        )
        selected_qkv = selected_decode = KernelBackend.TORCH
    elif forced_backend is not None:
        selected_qkv = selected_decode = choose_kernel_backend(
            (forced_backend,),
            op_name="SGLANG_FORCE_FUSED_OP_BACKEND",
            aot_available=aot_available,
        )
    else:
        selected_qkv = choose_kernel_backend(
            selection.qknorm_rope_store,
            op_name="SGLANG_MPS_QWEN3_QKNORM_ROPE_STORE",
            aot_available=aot_available,
        )
        selected_decode = choose_kernel_backend(
            selection.radix_decode,
            op_name="SGLANG_MPS_QWEN3_RADIX_DECODE",
            aot_available=aot_available,
        )

    # The model-level QKV provider runs before either attention backend, but
    # only MpsAttnBackend consumes ``RadixAttention.decode_provider``.  Never
    # advertise or warm a decode provider that an explicitly selected
    # TorchNativeAttnBackend would silently bypass.
    if decode_backend == "torch_native" and selected_decode is not KernelBackend.TORCH:
        reason = (
            "decode attention backend 'torch_native' does not consume the "
            "Qwen3 MPS Radix decode provider"
        )
        if forced_backend is not None:
            raise RuntimeError(
                "SGLANG_FORCE_FUSED_OP_BACKEND cannot be honored: " + reason
            )
        decode_fallback_reason = reason
        selected_decode = KernelBackend.TORCH

    qkv_custom = selected_qkv is not KernelBackend.TORCH
    decode_custom = selected_decode is not KernelBackend.TORCH
    attentions: list[Any] = []
    attention_providers: list[Qwen3MpsAttentionProvider] = []
    if qkv_custom or decode_custom:
        # Discover the typed layers once, then validate each semantic operation
        # independently.  A QKV-specific contract miss must not disable an
        # otherwise valid decode provider (or vice versa).
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
            if forced_backend is not None:
                raise RuntimeError(
                    "SGLANG_FORCE_FUSED_OP_BACKEND requires the Qwen3 Metal "
                    f"attention contract, but discovery failed: {exc}"
                ) from exc
            if qkv_custom:
                qkv_fallback_reason = str(exc)
            if decode_custom:
                decode_fallback_reason = str(exc)
            selected_qkv = selected_decode = KernelBackend.TORCH
            qkv_custom = decode_custom = False
            attentions = []
        else:
            if qkv_custom:
                try:
                    for module in attentions:
                        validate_qwen3_qkv_module(module)
                except RuntimeError as exc:
                    if forced_backend is not None:
                        raise RuntimeError(
                            "SGLANG_FORCE_FUSED_OP_BACKEND requires the Qwen3 "
                            f"QKV Metal contract, but validation failed: {exc}"
                        ) from exc
                    qkv_fallback_reason = str(exc)
                    selected_qkv = KernelBackend.TORCH
                    qkv_custom = False
            if decode_custom:
                try:
                    for module in attentions:
                        validate_qwen3_decode_module(module)
                except RuntimeError as exc:
                    if forced_backend is not None:
                        raise RuntimeError(
                            "SGLANG_FORCE_FUSED_OP_BACKEND requires the Qwen3 "
                            f"decode Metal contract, but validation failed: {exc}"
                        ) from exc
                    decode_fallback_reason = str(exc)
                    selected_decode = KernelBackend.TORCH
                    decode_custom = False

            if qkv_custom or decode_custom:
                try:
                    contracts = _validate_kv_pool_contract(token_to_kv_pool, attentions)
                except RuntimeError as exc:
                    if forced_backend is not None:
                        raise RuntimeError(
                            "SGLANG_FORCE_FUSED_OP_BACKEND requires the Qwen3 "
                            f"Metal KV-pool contract, but validation failed: {exc}"
                        ) from exc
                    if qkv_custom:
                        qkv_fallback_reason = str(exc)
                    if decode_custom:
                        decode_fallback_reason = str(exc)
                    selected_qkv = selected_decode = KernelBackend.TORCH
                    qkv_custom = decode_custom = False
                    attentions = []
                else:
                    # The warmup API accepts independent providers and shares
                    # one library when both sides use the same backend.
                    warmup_qwen3_mps_provider(
                        selected_qkv if qkv_custom else KernelBackend.TORCH,
                        selected_decode if decode_custom else KernelBackend.TORCH,
                    )
                    attention_providers = [
                        Qwen3MpsAttentionProvider(
                            pool_contract=contract,
                            qkv_kernel_backend=(selected_qkv if qkv_custom else None),
                            decode_kernel_backend=(
                                selected_decode if decode_custom else None
                            ),
                        )
                        for contract in contracts
                    ]
            else:
                attentions = []

    # Deferred commit is only exercised by the coarse MLX provider.  Resolve it
    # independently so a model-forward experiment can use Torch commit while
    # another experiment enables only the Metal commit kernel.
    deferred_backend = choose_kernel_backend(
        selection.deferred_kv_commit,
        op_name="SGLANG_MPS_QWEN3_DEFERRED_KV_COMMIT",
    )

    model_forward_provider = None
    whole_model_backend = "torch"
    whole_model_fallback_reason: Optional[str] = None
    if forced_backend is not None:
        whole_model_fallback_reason = (
            f"explicit fused-op backend {forced_backend.value!r} disables "
            "whole-model MLX selection"
        )
    else:
        mlx_requested = "mlx" in selection.model_forward
        mlx_available = False
        if mlx_requested:
            try:
                import mlx.core  # noqa: F401

                mlx_available = True
            except ImportError:
                mlx_available = False
        selected_model_backend = choose_model_backend(
            selection.model_forward,
            mlx_available=mlx_available,
        )
        if selected_model_backend == "mlx":
            whole_model_fallback_reason = _whole_model_mlx_fallback_reason(
                model_config,
                server_args,
                forced_backend,
            )
            if whole_model_fallback_reason is not None:
                if "torch" not in selection.model_forward:
                    raise RuntimeError(
                        "SGLANG_MPS_QWEN3_MODEL_FORWARD selected MLX but its "
                        f"static contract is unavailable: {whole_model_fallback_reason}"
                    )
                whole_model_backend = "torch"
            else:
                from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (
                    create_qwen3_mlx_model_provider,
                    validate_qwen3_mlx_static_contract,
                )

                try:
                    validate_qwen3_mlx_static_contract(
                        model,
                        token_to_kv_pool,
                        req_to_token_pool,
                    )
                except RuntimeError as exc:
                    whole_model_fallback_reason = str(exc)
                    if "torch" not in selection.model_forward:
                        raise
                else:
                    selected_greedy_tail_backend = choose_model_backend(
                        selection.greedy_tail,
                        mlx_available=True,
                        op_name="SGLANG_MPS_QWEN3_GREEDY_TAIL",
                    )
                    # View construction, DLPack, compilation, warmup, and OOM
                    # failures are startup errors rather than hidden downgrades.
                    model_forward_provider = create_qwen3_mlx_model_provider(
                        model,
                        token_to_kv_pool,
                        req_to_token_pool,
                        kv_commit_backend=deferred_backend,
                        server_args=server_args,
                        greedy_tail_backend=selected_greedy_tail_backend,
                    )
                    whole_model_backend = "mlx"
        elif selected_model_backend == "torch":
            whole_model_fallback_reason = (
                "disabled by SGLANG_MPS_QWEN3_MODEL_FORWARD priority"
            )

    actual_generic = {key: value.value for key, value in generic_backends.items()}
    plan = MpsOperatorPlan(
        provider_spec=_QWEN3_PROVIDER_SPEC,
        patched_attention_modules=len(attention_providers),
        patched_qkv_modules=sum(
            1 for provider in attention_providers if provider.qkv_kernel_backend
        ),
        patched_decode_modules=sum(
            1 for provider in attention_providers if provider.decode_kernel_backend
        ),
        enabled=bool(
            attention_providers
            or model_forward_provider is not None
            or any(
                value is not KernelBackend.TORCH for value in generic_backends.values()
            )
        ),
        attention_backend=attention_backend,
        qkv_kernel_backend=selected_qkv.value,
        decode_kernel_backend=selected_decode.value,
        deferred_kv_commit_backend=(
            deferred_backend.value if model_forward_provider is not None else "off"
        ),
        qkv_fallback_reason=qkv_fallback_reason,
        decode_fallback_reason=decode_fallback_reason,
        whole_model_backend=whole_model_backend,
        whole_model_fallback_reason=whole_model_fallback_reason,
        provider_priorities=selection.as_state(),
        generic_kernel_backends=actual_generic,
        forward_lock=_MPS_OPERATOR_FORWARD_LOCK,
        _model=model,
        _attention_bindings=tuple(zip(attentions, attention_providers)),
        _model_forward_provider=model_forward_provider,
    )

    # Validate the complete lifecycle before publishing any model binding.
    # The router repeats this defensively after the installer returns, but by
    # then a contributor installer is already capable of mutating its model.
    validate_mps_operator_plan(
        plan,
        spec_name=_QWEN3_PROVIDER_SPEC,
        installer_path=_QWEN3_INSTALLER_PATH,
    )

    # Publish only after every selected provider has validated and warmed. A
    # contributor-owned setter may still raise, so make the binding sequence a
    # rollback-capable transaction and close unpublished provider state.
    try:
        with MpsBindingPublication() as publication:
            if model_forward_provider is not None:
                publication.bind(
                    model.model,
                    "model_forward_provider",
                    model_forward_provider,
                )
            for module, provider in zip(attentions, attention_providers):
                if provider.qkv_kernel_backend is not None:
                    publication.bind(module, "op_provider", provider)
                if provider.decode_kernel_backend is not None:
                    publication.bind(module.attn, "decode_provider", provider)
            publication.commit()
    except Exception:
        try:
            plan.close()
        except Exception:
            logger.exception("Failed to close an unpublished Qwen3 MPS operator plan")
        raise

    logger.info(
        "Installed Qwen3 MPS operator plan: whole_model=%s qkv=%s decode=%s "
        "attention_modules=%d; Torch owns weights, KV pools, and Radix cache",
        whole_model_backend,
        selected_qkv.value,
        selected_decode.value,
        len(attention_providers),
    )
    return plan


__all__ = [
    "MpsOperatorPlan",
    "QWEN3_MPS_MODEL",
    "install_qwen3_operators",
    "model_is_qwen3_06b",
]
