"""Torch-owned Metal JIT kernels for the Apple Silicon execution path.

These kernels run on PyTorch's current MPS command queue through
``torch.mps.compile_shader``.  Their inputs, outputs, and KV cache remain
ordinary Torch tensors; no DLPack bridge or cross-runtime synchronization is
involved.

The first supported shape is intentionally narrow: dense Qwen3-0.6B, TP=1,
bf16, and the standard NHD token KV pool.  The wrappers fail on layout drift
instead of materializing hidden copies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from sglang.kernels.metal import (
    compile_metal_library,
    resolve_metal_entry_points,
)
from sglang.kernels.ops.attention.qwen3_mps import (
    QWEN3_06B_METAL_SPEC,
    Qwen3MetalKernelSpec,
    is_qwen3_attention_scale_supported,
    is_qwen3_rms_epsilon_supported,
)

_DECODE_NUM_THREADS = 256
_QWEN3_06B_SOURCE_PATH = Path(__file__).with_name("_qwen3_06b_attention.metal")


@lru_cache(maxsize=1)
def _metal_source(spec: Qwen3MetalKernelSpec) -> str:
    """Read the canonical JIT source.

    A future Torch data-only AOT wheel should consume this same file rather
    than copying the shader into a host-runtime-specific build tree.
    """
    _require_qwen3_06b_spec(spec)
    return _QWEN3_06B_SOURCE_PATH.read_text(encoding="utf-8")


def _require_qwen3_06b_spec(spec: Qwen3MetalKernelSpec) -> None:
    if spec != QWEN3_06B_METAL_SPEC:
        raise RuntimeError(
            "the Qwen3 Metal kernels are specialized for Qwen3-0.6B; "
            f"found head_dim={spec.head_dim}, num_q_heads={spec.num_q_heads}, "
            f"num_kv_heads={spec.num_kv_heads}"
        )


def _compile_qwen3_library(spec: Qwen3MetalKernelSpec):
    """Compile through the framework-neutral Metal substrate.

    ``compile_metal_library`` owns the process cache, so explicit provider
    warmup and direct kernel calls resolve the same Torch MPS library object.
    """
    _require_qwen3_06b_spec(spec)
    return compile_metal_library(_metal_source(spec))


def warmup_qwen3_metal_qknorm_rope_store(
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
) -> None:
    """Compile and resolve only the QK-norm/RoPE/KV-store pipeline."""
    # Attribute lookup resolves the named Metal pipeline now, so an invalid
    # entry point fails during server initialization rather than first traffic.
    resolve_metal_entry_points(
        _compile_qwen3_library(spec), ("qwen3_qknorm_rope_store_bf16",)
    )


def warmup_qwen3_metal_radix_decode(
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
) -> None:
    """Compile and resolve only the Radix decode pipeline."""
    resolve_metal_entry_points(
        _compile_qwen3_library(spec), ("qwen3_radix_decode_bf16",)
    )


def warmup_qwen3_metal_kernels(
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
    *,
    qknorm_rope_store: bool = True,
    radix_decode: bool = True,
) -> None:
    """Compile and resolve selected specialized pipelines.

    With its historical no-argument form this warms both entry points.  The
    keyword gates support per-operator provider selection while
    ``_compile_qwen3_library`` keeps one compiled library per static spec.
    """
    if not qknorm_rope_store and not radix_decode:
        return
    entry_points = []
    if qknorm_rope_store:
        entry_points.append("qwen3_qknorm_rope_store_bf16")
    if radix_decode:
        entry_points.append("qwen3_radix_decode_bf16")
    resolve_metal_entry_points(_compile_qwen3_library(spec), entry_points)


def _require_mps_bf16_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"{name} must be a Torch tensor")
    if tensor.device.type != "mps":
        raise RuntimeError(f"{name} must be on MPS, found {tensor.device}")
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be bfloat16, found {tensor.dtype}")
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"{name} must be contiguous; implicit Metal-path copies are forbidden"
        )


def _require_mps_integer(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if tensor.device.type != "mps" or tensor.dtype != dtype:
        raise RuntimeError(
            f"{name} must be a {dtype} MPS tensor, found "
            f"device={tensor.device}, dtype={tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def qwen3_fused_qknorm_rope_store(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    q_out: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    epsilon: float,
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
    _library=None,
) -> None:
    """Fuse Q/K RMSNorm, NeoX RoPE, and standard KV-pool writes."""
    _require_qwen3_06b_spec(spec)
    for name, tensor in (
        ("qkv", qkv),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin_cache", cos_sin_cache),
        ("q_out", q_out),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
    ):
        _require_mps_bf16_contiguous(name, tensor)
    _require_mps_integer("positions", positions, torch.int64)
    _require_mps_integer("slots", slots, torch.int64)

    num_tokens = qkv.shape[0]
    expected_shapes = {
        "qkv": (num_tokens, spec.qkv_width),
        "q_weight": (spec.head_dim,),
        "k_weight": (spec.head_dim,),
        "q_out": (num_tokens, spec.num_q_heads, spec.head_dim),
        "positions": (num_tokens,),
        "slots": (num_tokens,),
    }
    actual = {
        "qkv": tuple(qkv.shape),
        "q_weight": tuple(q_weight.shape),
        "k_weight": tuple(k_weight.shape),
        "q_out": tuple(q_out.shape),
        "positions": tuple(positions.shape),
        "slots": tuple(slots.shape),
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise RuntimeError(
                f"{name} shape mismatch: expected {expected}, found {actual[name]}"
            )
    pool_tail = (spec.num_kv_heads, spec.head_dim)
    if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != pool_tail:
        raise RuntimeError(
            "k_pool must use contiguous NHD layout [slots, num_kv_heads, "
            f"head_dim], found {tuple(k_pool.shape)}"
        )
    if tuple(v_pool.shape) != tuple(k_pool.shape):
        raise RuntimeError(
            f"v_pool shape must match k_pool, found {tuple(v_pool.shape)} vs "
            f"{tuple(k_pool.shape)}"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != spec.head_dim:
        raise RuntimeError(
            "cos_sin_cache must have shape [max_position, head_dim], found "
            f"{tuple(cos_sin_cache.shape)}"
        )
    if num_tokens == 0:
        return
    if not is_qwen3_rms_epsilon_supported(epsilon):
        raise RuntimeError(f"epsilon must be finite and positive, found {epsilon}")

    library = _compile_qwen3_library(spec) if _library is None else _library
    kernel = library.qwen3_qknorm_rope_store_bf16
    kernel(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        slots,
        q_out,
        k_pool,
        v_pool,
        float(epsilon),
        int(cos_sin_cache.shape[0]),
        int(k_pool.shape[0]),
        threads=(
            spec.head_dim,
            spec.num_q_heads + spec.num_kv_heads,
            num_tokens,
        ),
        group_size=(spec.head_dim, 1, 1),
    )


def qwen3_radix_decode(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
    _library=None,
) -> None:
    """Run one-token decode attention directly over the Radix slot table."""
    _require_qwen3_06b_spec(spec)
    for name, tensor in (
        ("q", q),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
        ("out", out),
    ):
        _require_mps_bf16_contiguous(name, tensor)
    _require_mps_integer("req_to_token", req_to_token, torch.int32)
    _require_mps_integer("req_pool_indices", req_pool_indices, torch.int64)
    _require_mps_integer("seq_lens", seq_lens, torch.int64)

    batch_size = q.shape[0]
    expected_q = (batch_size, spec.num_q_heads, spec.head_dim)
    if tuple(q.shape) != expected_q or tuple(out.shape) != expected_q:
        raise RuntimeError(
            f"q/out must have shape {expected_q}, found "
            f"q={tuple(q.shape)}, out={tuple(out.shape)}"
        )
    pool_tail = (spec.num_kv_heads, spec.head_dim)
    if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != pool_tail:
        raise RuntimeError(
            "k_pool must use contiguous NHD layout [slots, num_kv_heads, "
            f"head_dim], found {tuple(k_pool.shape)}"
        )
    if tuple(v_pool.shape) != tuple(k_pool.shape):
        raise RuntimeError(
            f"v_pool shape must match k_pool, found {tuple(v_pool.shape)} vs "
            f"{tuple(k_pool.shape)}"
        )
    if req_to_token.ndim != 2:
        raise RuntimeError(
            f"req_to_token must be 2-D, found {tuple(req_to_token.shape)}"
        )
    if tuple(req_pool_indices.shape) != (batch_size,) or tuple(seq_lens.shape) != (
        batch_size,
    ):
        raise RuntimeError(
            "decode metadata must have one entry per request: "
            f"batch={batch_size}, req_pool_indices={tuple(req_pool_indices.shape)}, "
            f"seq_lens={tuple(seq_lens.shape)}"
        )
    if batch_size == 0:
        return
    if not is_qwen3_attention_scale_supported(scale, spec):
        raise RuntimeError(
            "attention scale does not match the specialized Metal kernel: "
            f"expected {spec.attention_scale}, found {scale}"
        )

    library = _compile_qwen3_library(spec) if _library is None else _library
    kernel = library.qwen3_radix_decode_bf16
    kernel(
        q,
        k_pool,
        v_pool,
        req_to_token,
        req_pool_indices,
        seq_lens,
        out,
        int(req_to_token.stride(0)),
        int(req_to_token.shape[0]),
        int(k_pool.shape[0]),
        threads=(batch_size, spec.num_q_heads, _DECODE_NUM_THREADS),
        group_size=(1, 1, _DECODE_NUM_THREADS),
    )


__all__ = [
    "QWEN3_06B_METAL_SPEC",
    "Qwen3MetalKernelSpec",
    "qwen3_fused_qknorm_rope_store",
    "qwen3_radix_decode",
    "warmup_qwen3_metal_qknorm_rope_store",
    "warmup_qwen3_metal_radix_decode",
    "warmup_qwen3_metal_kernels",
]
