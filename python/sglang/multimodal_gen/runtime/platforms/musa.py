"""
This file is a platform abstraction for MUSA GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MusaPlatform(Platform):
    _enum = PlatformEnum.MUSA
    device_name: str = "musa"
    device_type: str = "musa"  # torch uses 'musa' backend string
    dispatch_key: str = "MUSA"
    device_control_env_var: str = "MUSA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA graph. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass  # MUSA-specific warnings can be added here

    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.TORCH_SDPA:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}: {selected_backend}"
            )

        logger.info("Using Torch SDPA backend.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # works for MUSA too
