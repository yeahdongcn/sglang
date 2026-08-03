"""Packed QKV preparation for the Qwen3-0.6B whole-MLX island.

The kernel consumes the packed projection produced by the Torch-owned Qwen3
model, applies SGLang's fp32 Q/K RMSNorm contract, rotates Q/K with the
borrowed bf16 NeoX cache, and emits dense Q/K/V arrays.  K/V remain deferred:
the enclosing MLX island evaluates all layers before the selected Torch-stream
commit provider writes the standard SRT KV pool.

This module has an MLX-array contract.  It never synchronizes or evaluates a
lazy result, and it never mutates a borrowed Torch allocation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC

if TYPE_CHECKING:
    import mlx.core as mx

_THREADS = 32
_HALF_DIM = QWEN3_06B_METAL_SPEC.head_dim // 2
_VALUES_PER_LANE = QWEN3_06B_METAL_SPEC.head_dim // _THREADS


def _epsilon_literal(epsilon: float) -> str:
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError("Qwen3 MLX QKV preparation requires a positive epsilon")
    literal = f"{epsilon:.17g}"
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal + "f"


def _epsilon_tag(epsilon: float) -> str:
    """Encode a finite epsilon as a Metal-identifier-safe cache suffix."""
    _epsilon_literal(epsilon)
    return float(epsilon).hex().replace("-", "n").replace("+", "p").replace(".", "_")


def _source(epsilon: float) -> str:
    return f"""
const uint lane = thread_index_in_simdgroup;
const uint head = threadgroup_position_in_grid.y;
const uint token = threadgroup_position_in_grid.z;
const bool is_q = head < HQ;
const uint local_head = is_q ? head : head - HQ;
const uint source_head = is_q ? local_head : HQ + local_head;
const ulong source_base =
    ulong(token) * ulong(QKV_WIDTH) + ulong(source_head) * ulong(D);
const long position = positions[token];

threadgroup float inverse_rms;
threadgroup T normalized[D];

float square_sum = 0.0f;
for (uint index = 0; index < VPL; ++index) {{
    const uint dim = lane * VPL + index;
    const float value = float(qkv[source_base + dim]);
    square_sum += value * value;
}}
square_sum = simd_sum(square_sum);
if (lane == 0) {{
    inverse_rms = metal::precise::rsqrt(
        square_sum / float(D) + {_epsilon_literal(epsilon)});
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

const device T* weight = is_q ? q_weight : k_weight;
for (uint index = 0; index < VPL; ++index) {{
    const uint dim = lane * VPL + index;
    const float value = float(qkv[source_base + dim]);
    // SGLang multiplies the fp32-normalized activation by the fp32-promoted
    // weight before the one bf16 output cast.
    normalized[dim] = T(value * inverse_rms * float(weight[dim]));
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint index = 0; index < VPL; ++index) {{
    const uint dim = lane * VPL + index;
    T rotated = T(0.0f);
    if (position >= 0 && ulong(position) < ulong(cos_sin_shape[0])) {{
        const uint rope_dim = dim < HD ? dim : dim - HD;
        const uint peer_dim = dim < HD ? dim + HD : dim - HD;
        const ulong rope_base = ulong(position) * ulong(D);
        const T cosine = cos_sin[rope_base + rope_dim];
        const T sine = cos_sin[rope_base + HD + rope_dim];
        // Match the staged MLX bf16 graph: each multiply narrows before the
        // final add/subtract narrows once more.
        const T own_cos = T(float(normalized[dim]) * float(cosine));
        const T peer_sin = T(float(normalized[peer_dim]) * float(sine));
        rotated = dim < HD
            ? T(float(own_cos) - float(peer_sin))
            : T(float(own_cos) + float(peer_sin));
    }}

    if (is_q) {{
        const ulong output_index =
            (ulong(token) * ulong(HQ) + ulong(local_head)) * ulong(D) + dim;
        q_out[output_index] = rotated;
    }} else {{
        const ulong output_index =
            (ulong(token) * ulong(HK) + ulong(local_head)) * ulong(D) + dim;
        k_out[output_index] = rotated;
        const ulong v_source =
            ulong(token) * ulong(QKV_WIDTH) +
            ulong(HQ + HK + local_head) * ulong(D) + dim;
        v_out[output_index] = qkv[v_source];
    }}
}}
"""


@lru_cache(maxsize=4)
def _qkv_kernel(epsilon: float):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 MLX QKV preparation requires Metal")
    return mx.fast.metal_kernel(
        name=("qwen3_qkv_prepare_deferred_" + _epsilon_tag(epsilon)),
        input_names=["qkv", "q_weight", "k_weight", "cos_sin", "positions"],
        output_names=["q_out", "k_out", "v_out"],
        source=_source(epsilon),
        # The ordinary matmul/cache path is already dense.  Keep the wrapper's
        # safety materialization enabled for a future strided producer rather
        # than allowing raw pointer arithmetic to corrupt outputs.
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _require_shape(name: str, array: mx.array, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {shape}, found {tuple(array.shape)}"
        )


def qwen3_qkv_prepare_deferred(
    qkv: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    cos_sin: mx.array,
    positions: mx.array,
    *,
    epsilon: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return dense rotated Q/K and V without writing the Torch KV pool.

    ``qkv`` is normally the row-contiguous output of the packed projection.
    The custom-kernel wrapper enforces row contiguity for an exceptional
    strided input, so a future producer change cannot silently corrupt values;
    the dense matmul/cache path does not incur that materialization.  Positions
    are required to be valid indices into ``cos_sin``.
    """
    import mlx.core as mx

    arrays = (qkv, q_weight, k_weight, cos_sin, positions)
    if not all(isinstance(array, mx.array) for array in arrays):
        raise RuntimeError("Qwen3 MLX QKV preparation inputs must be MLX arrays")

    spec = QWEN3_06B_METAL_SPEC
    batch = int(qkv.shape[0]) if qkv.ndim == 2 else -1
    _require_shape("qkv", qkv, (batch, spec.qkv_width))
    _require_shape("q_weight", q_weight, (spec.head_dim,))
    _require_shape("k_weight", k_weight, (spec.head_dim,))
    if cos_sin.ndim != 2 or tuple(cos_sin.shape[1:]) != (spec.head_dim,):
        raise RuntimeError(
            f"cos_sin must have shape [max_position, 128], found {tuple(cos_sin.shape)}"
        )
    if int(cos_sin.shape[0]) <= 0:
        raise RuntimeError("cos_sin must contain at least one position")
    _require_shape("positions", positions, (batch,))

    for name, array in (
        ("qkv", qkv),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin", cos_sin),
    ):
        if array.dtype != mx.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16, found {array.dtype}")
    if positions.dtype != mx.int64:
        raise RuntimeError(f"positions must be int64, found {positions.dtype}")
    epsilon = float(epsilon)
    _epsilon_literal(epsilon)

    q_shape = (batch, spec.num_q_heads, spec.head_dim)
    kv_shape = (batch, spec.num_kv_heads, spec.head_dim)
    if batch == 0:
        return (
            mx.empty(q_shape, dtype=mx.bfloat16),
            mx.empty(kv_shape, dtype=mx.bfloat16),
            mx.empty(kv_shape, dtype=mx.bfloat16),
        )

    return tuple(
        _qkv_kernel(epsilon)(
            inputs=[qkv, q_weight, k_weight, cos_sin, positions],
            template=[
                ("T", qkv.dtype),
                ("D", spec.head_dim),
                ("HD", _HALF_DIM),
                ("HQ", spec.num_q_heads),
                ("HK", spec.num_kv_heads),
                ("QKV_WIDTH", spec.qkv_width),
                ("VPL", _VALUES_PER_LANE),
            ],
            grid=(_THREADS, spec.num_q_heads + spec.num_kv_heads, batch),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[q_shape, kv_shape, kv_shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
        )
    )


@lru_cache(maxsize=4)
def warmup_qwen3_qkv_prepare_deferred(epsilon: float = 1e-6) -> None:
    """Compile and execute the fixed Qwen3 QKV Metal pipeline at startup."""
    import mlx.core as mx

    spec = QWEN3_06B_METAL_SPEC
    qkv = mx.zeros((1, spec.qkv_width), dtype=mx.bfloat16)
    q_weight = mx.ones((spec.head_dim,), dtype=mx.bfloat16)
    k_weight = mx.ones_like(q_weight)
    cos_sin = mx.ones((1, spec.head_dim), dtype=mx.bfloat16)
    positions = mx.zeros((1,), dtype=mx.int64)
    outputs = qwen3_qkv_prepare_deferred(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        epsilon=epsilon,
    )
    mx.eval(*outputs)


__all__ = ["qwen3_qkv_prepare_deferred", "warmup_qwen3_qkv_prepare_deferred"]
