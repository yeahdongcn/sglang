"""Torch-owned Qwen3 Metal AOT adapters.

The precompiled pipelines use the same canonical MSL source and launch
contract as :mod:`._qwen3_metal_jit`.  Only library acquisition differs: the
AOT path loads the packaged metallib through ``torch.mps.load_metallib``.
There is no MLX primitive, tensor bridge, or C++ host extension on this path.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def is_qwen3_metal_aot_available() -> bool:
    """Whether the platform-specific sgl-kernel metallib can be loaded."""
    try:
        from sgl_kernel.metal import is_metal_aot_available
    except (ImportError, OSError):
        return False
    return is_metal_aot_available()


@lru_cache(maxsize=1)
def _load_library():
    """Load the packaged library once for all Qwen3 entry points.

    ``sgl_kernel.metal.load_metal_library`` is cached as well, but keeping the
    cache at this adapter boundary is intentional: callers that select the
    two Qwen3 operators independently still share exactly one library object,
    and tests can replace this boundary without depending on the wheel's
    implementation details.
    """
    try:
        from sgl_kernel.metal import load_metal_library
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Qwen3 Metal AOT requires the Apple Silicon sglang-kernel wheel"
        ) from exc
    return load_metal_library()


def warmup_qwen3_metal_aot_qknorm_rope_store() -> None:
    """Resolve the precompiled QK-norm/RoPE/KV-store entry point."""
    _load_library().qwen3_qknorm_rope_store_bf16


def warmup_qwen3_metal_aot_radix_decode() -> None:
    """Resolve the precompiled Radix decode entry point."""
    _load_library().qwen3_radix_decode_bf16


def warmup_qwen3_metal_aot_kernels(
    *,
    qknorm_rope_store: bool = True,
    radix_decode: bool = True,
) -> None:
    """Resolve selected precompiled Qwen3 pipelines.

    The no-argument form preserves the original adapter contract and warms
    both pipelines.  The keyword gates let the MPS operator plan warm only the
    providers it selected; all selections still share the cached metallib.
    """
    if not qknorm_rope_store and not radix_decode:
        return
    library = _load_library()
    if qknorm_rope_store:
        library.qwen3_qknorm_rope_store_bf16
    if radix_decode:
        library.qwen3_radix_decode_bf16


def qwen3_fused_qknorm_rope_store(*args, **kwargs) -> None:
    """Run the fixed QK-norm/RoPE/KV-store pipeline from the metallib."""
    from sglang.kernels.ops.attention._qwen3_metal_jit import (
        qwen3_fused_qknorm_rope_store as dispatch,
    )

    kwargs["_library"] = _load_library()
    dispatch(*args, **kwargs)


def qwen3_radix_decode(*args, **kwargs) -> None:
    """Run the fixed Radix decode pipeline from the metallib."""
    from sglang.kernels.ops.attention._qwen3_metal_jit import (
        qwen3_radix_decode as dispatch,
    )

    kwargs["_library"] = _load_library()
    dispatch(*args, **kwargs)


__all__ = [
    "is_qwen3_metal_aot_available",
    "qwen3_fused_qknorm_rope_store",
    "qwen3_radix_decode",
    "warmup_qwen3_metal_aot_qknorm_rope_store",
    "warmup_qwen3_metal_aot_radix_decode",
    "warmup_qwen3_metal_aot_kernels",
]
