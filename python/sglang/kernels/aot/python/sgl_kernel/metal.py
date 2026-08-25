"""Torch-native entry points for the precompiled SGLang Metal library."""

from __future__ import annotations

import platform
import sys
from functools import lru_cache
from pathlib import Path

_METALLIB_NAME = "sgl_metal_kernels.metallib"


def metal_library_path() -> Path:
    """Return the metallib shipped next to this module."""
    return Path(__file__).resolve().with_name(_METALLIB_NAME)


def is_metal_aot_available() -> bool:
    """Whether this interpreter can load the packaged Torch Metal library."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    if not metal_library_path().is_file():
        return False
    try:
        import torch
    except ImportError:
        return False
    return callable(getattr(torch.mps, "load_metallib", None))


@lru_cache(maxsize=1)
def load_metal_library():
    """Load the packaged metallib on Torch's current MPS command queue."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("SGLang Metal AOT kernels require Apple Silicon")

    import torch

    load_metallib = getattr(torch.mps, "load_metallib", None)
    if not callable(load_metallib):
        raise RuntimeError(
            "SGLang Metal AOT kernels require torch.mps.load_metallib from Torch 2.13"
        )
    library_path = metal_library_path()
    if not library_path.is_file():
        raise RuntimeError(
            f"packaged Metal library is missing: {library_path}. Install the "
            "Apple Silicon sglang-kernel wheel built by setup_metal.py."
        )
    return load_metallib(library_path)


def warmup_qwen3_06b_metal_aot() -> None:
    """Load the library and resolve both fixed Qwen3-0.6B pipelines."""
    library = load_metal_library()
    library.qwen3_qknorm_rope_store_bf16
    library.qwen3_radix_decode_bf16


__all__ = [
    "is_metal_aot_available",
    "load_metal_library",
    "metal_library_path",
    "warmup_qwen3_06b_metal_aot",
]
