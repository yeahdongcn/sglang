"""Model introspection and attention patching."""

from typing import Any

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
    MLXBatchedAttentionWrapper,
)


def find_attention_layers(model: Any) -> tuple[list[Any], str]:
    """Find transformer layers and the attention attribute name."""
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layer_list = getattr(container, "layers", None) or getattr(root, "layers", [])

    if layer_list:
        sample = layer_list[0]
        if hasattr(sample, "self_attn"):
            return layer_list, "self_attn"
        if hasattr(sample, "attention"):
            return layer_list, "attention"
        raise ValueError(f"No attention attribute in layer type {type(sample)}")
    return layer_list, "self_attn"


def patch_model_attention(model: Any, *, enable_paged: bool = True) -> int:
    """Install MLXAttentionWrapper on all attention layers (idempotent).

    The wrapper delegates to the inner module when no active context is set,
    so it is always installed and never removed.
    """
    layer_list, attn_attr = find_attention_layers(model)
    patched = 0
    for idx, layer in enumerate(layer_list):
        attn = getattr(layer, attn_attr)
        wrapper_cls = (
            MLXAttentionWrapper if enable_paged else MLXBatchedAttentionWrapper
        )
        if type(attn) is wrapper_cls:
            object.__setattr__(attn, "_enable_paged", enable_paged)
            continue
        if isinstance(attn, MLXAttentionWrapper):
            attn = attn._inner
        setattr(
            layer,
            attn_attr,
            wrapper_cls(attn, idx),
        )
        patched += 1
    return patched


def unpatch_model_attention(model: Any) -> int:
    """Restore original attention modules from MLXAttentionWrapper instances."""
    layer_list, attn_attr = find_attention_layers(model)
    restored = 0
    for layer in layer_list:
        attn = getattr(layer, attn_attr)
        if isinstance(attn, MLXAttentionWrapper):
            setattr(layer, attn_attr, attn._inner)
            restored += 1
    return restored


def get_num_layers(model: Any) -> int:
    """Return the number of transformer layers."""
    layer_list, _ = find_attention_layers(model)
    return len(layer_list)
