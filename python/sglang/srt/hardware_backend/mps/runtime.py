"""Runtime validation for SGLang's Apple MPS backend."""

from functools import lru_cache
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

_MIN_MLX_VERSION = Version("0.32.0")
_SUPPORTED_TORCH_SERIES = (2, 13)


def _is_stable_series(raw_version: object, series: tuple[int, int]) -> bool:
    try:
        version = Version(str(raw_version))
    except InvalidVersion:
        return False
    return not version.is_prerelease and (version.major, version.minor) == series


def _is_stable_at_least(raw_version: object, minimum: Version) -> bool:
    try:
        version = Version(str(raw_version))
    except InvalidVersion:
        return False
    return not version.is_prerelease and version >= minimum


@lru_cache(maxsize=1)
def validate_mps_runtime() -> None:
    """Validate the supported Torch/MLX runtime pair for the MPS backend."""
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError(
            "The SGLang MPS backend requires stable Torch 2.13.x and stable "
            "MLX >= 0.32.0, but MLX is not installed; reinstall with the "
            "srt_mps extra"
        ) from None

    torch_version = getattr(torch, "__version__", None)
    mlx_version = getattr(mx, "__version__", None)
    if not _is_stable_series(
        torch_version, _SUPPORTED_TORCH_SERIES
    ) or not _is_stable_at_least(mlx_version, _MIN_MLX_VERSION):
        raise RuntimeError(
            "The SGLang MPS backend requires stable Torch 2.13.x and stable "
            "MLX >= 0.32.0; found "
            f"Torch {torch_version or 'unknown'} + MLX {mlx_version or 'unknown'}; "
            "reinstall with the srt_mps extra"
        )

    mps_backend = getattr(torch.backends, "mps", None)
    is_mps_available = getattr(mps_backend, "is_available", None)
    if not callable(is_mps_available) or not is_mps_available():
        raise RuntimeError(
            "The SGLang MPS backend requires an available PyTorch MPS device"
        )
    for memory_api in ("recommended_max_memory", "driver_allocated_memory"):
        if not callable(getattr(torch.mps, memory_api, None)):
            raise RuntimeError(
                f"The SGLang MPS backend requires torch.mps.{memory_api} from "
                "the tested Torch 2.13.x runtime"
            )
    metal = getattr(mx, "metal", None)
    is_metal_available = getattr(metal, "is_available", None)
    if not callable(is_metal_available) or not is_metal_available():
        raise RuntimeError(
            "The SGLang MPS backend requires an available MLX Metal device"
        )


def validate_mps_model_config(
    model_config: Any,
    *,
    lora_enabled: bool = False,
) -> None:
    """Reject checkpoint-derived execution modes outside the MPS contract.

    ``ServerArgs.quantization`` only records an explicit CLI choice.  A
    checkpoint can declare quantization in its Hugging Face config, which is
    resolved later into ``ModelConfig.quantization``.  Validate that effective
    value before model loading, and again at the platform binding boundary for
    programmatic ``ModelRunner`` construction.
    """
    quantization = getattr(model_config, "quantization", None)
    if quantization not in (None, "unquant"):
        raise ValueError(
            "Torch MPS currently supports only unquantized model weights; "
            "the resolved model configuration detected "
            f"quantization={quantization!r}"
        )
    if bool(getattr(model_config, "is_multimodal", False)):
        raise ValueError(
            "Torch MPS multimodal serving does not yet have a model-specific "
            "end-to-end contract; use a text-only model until its encoder, "
            "processor, and decoder paths are validated on MPS"
        )
    if lora_enabled:
        for config in (
            getattr(model_config, "hf_text_config", None),
            getattr(model_config, "hf_config", None),
        ):
            if config is None:
                continue
            for field_name in (
                "num_experts",
                "num_local_experts",
                "n_routed_experts",
            ):
                value = getattr(config, field_name, None)
                if value is not None and int(value) > 0:
                    raise ValueError(
                        "Torch MPS LoRA currently supports dense models only; "
                        f"the model config declares {field_name}={value!r}"
                    )


__all__ = ["validate_mps_model_config", "validate_mps_runtime"]
