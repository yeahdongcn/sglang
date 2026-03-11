# Copied and adapted from: https://github.com/vllm-project/vllm-metal
from __future__ import annotations

import logging
from typing import Any

import torch

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX setups
    mx = None

logger = logging.getLogger(__name__)

# MPS has a 4GB (2^32 bytes) limit for MPSTemporaryNDArray allocations.
# Metal may allocate multiple temporary buffers internally, so we use a
# conservative threshold of 1GB to avoid hitting the limit.
# See: https://github.com/anthropics/vllm-metal/issues/43
_MPS_SAFE_SIZE_BYTES = 1 << 30  # 1GB

MLX_AVAILABLE = mx is not None

MLX_TO_TORCH_DTYPE = (
    {
        mx.float32: torch.float32,
        mx.float16: torch.float16,
        mx.bfloat16: torch.bfloat16,
        mx.int32: torch.int32,
        mx.int64: torch.int64,
        mx.int16: torch.int16,
        mx.int8: torch.int8,
        mx.uint8: torch.uint8,
        mx.bool_: torch.bool,
    }
    if MLX_AVAILABLE
    else {}
)

TORCH_TO_MLX_DTYPE = {v: k for k, v in MLX_TO_TORCH_DTYPE.items()}


def _require_mlx() -> Any:
    if mx is None:
        raise ImportError("mlx is required for sglang.srt.utils.tensor_bridge")
    return mx


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_tensor_size_bytes(array) -> int:
    return array.size * array.dtype.size


def _is_safe_for_mps(array) -> bool:
    return _get_tensor_size_bytes(array) < _MPS_SAFE_SIZE_BYTES


def torch_to_mlx(tensor: torch.Tensor):
    mx_mod = _require_mlx()

    if tensor.device.type == "mps":
        tensor = tensor.cpu()

    tensor = tensor.detach()

    # Note: numpy does not support bfloat16.
    if tensor.dtype == torch.bfloat16:
        return mx_mod.array(tensor.float().numpy(), dtype=mx_mod.bfloat16)

    return mx_mod.array(tensor.numpy())


def mlx_to_torch(
    array,
    device: torch.device | str | None = None,
    already_contiguous: bool = False,
) -> torch.Tensor:
    mx_mod = _require_mlx()

    if device is None:
        device = get_torch_device()
    elif isinstance(device, str):
        device = torch.device(device)

    torch_dtype = MLX_TO_TORCH_DTYPE.get(array.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported MLX dtype: {array.dtype}")

    if not already_contiguous:
        array = mx_mod.contiguous(array)

    mx_mod.eval(array)
    tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(array.shape)

    if device.type == "mps":
        if _is_safe_for_mps(array):
            tensor = tensor.to(device)
        else:
            logger.debug(
                "Tensor too large for MPS (%d bytes > %d limit), keeping on CPU",
                _get_tensor_size_bytes(array),
                _MPS_SAFE_SIZE_BYTES,
            )
    elif device.type != "cpu":
        tensor = tensor.to(device)

    return tensor


def sync_mlx() -> None:
    mx_mod = _require_mlx()

    try:
        mx_mod.synchronize()
    except (AttributeError, TypeError):
        mx_mod.eval(mx_mod.array(0, dtype=mx_mod.int32))


def sync_torch() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


__all__ = [
    "MLX_AVAILABLE",
    "MLX_TO_TORCH_DTYPE",
    "TORCH_TO_MLX_DTYPE",
    "_MPS_SAFE_SIZE_BYTES",
    "_get_tensor_size_bytes",
    "_is_safe_for_mps",
    "get_torch_device",
    "mlx_to_torch",
    "sync_mlx",
    "sync_torch",
    "torch_to_mlx",
]
