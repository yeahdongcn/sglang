"""MLX Metal RMSNorm kernels for the narrow Qwen3-0.6B provider.

The ordinary whole-model MLX graph widens Torch bf16 activations to fp32.  A
pair of explicit casts around every layer norm is measurable for long
prefills.  These kernels preserve that staged MLX graph's fp32 accumulation
and output contract in one MLX-owned Metal launch and return fresh MLX arrays;
they never mutate a Torch-owned weight, activation, or KV buffer.  This local
contract is not a claim of bitwise parity with Torch/MPS RMSNorm.

This is intentionally model-specific (hidden size 1024, bf16 inference).  A
caller that does not satisfy the contract should stay on the staged
fp32-accumulation path instead of silently materializing a different layout.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

_HIDDEN_SIZE = 1024
# Keep eight SIMD groups and the reduction order matched by the exactness
# audit against MLX 0.32.  A 128-thread variant is a little faster in some
# shapes but changes the fp32 accumulation tree for other row counts.
_THREADS = 256
_VALUES_PER_THREAD = _HIDDEN_SIZE // _THREADS


def _epsilon_literal(epsilon: float) -> str:
    # Metal accepts a decimal/scientific literal with an ``f`` suffix.  Keep
    # this separate from the kernel body so the source remains deterministic
    # for the small cache of supported Qwen3 epsilon values.
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError("Qwen3 MLX RMSNorm requires a positive epsilon")
    literal = f"{epsilon:.17g}"
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal + "f"


def _epsilon_tag(epsilon: float) -> str:
    """Encode a finite epsilon as a Metal-identifier-safe cache suffix."""
    _epsilon_literal(epsilon)
    return float(epsilon).hex().replace("-", "n").replace("+", "p").replace(".", "_")


def _rms_source(epsilon: float) -> str:
    eps = _epsilon_literal(epsilon)
    return f"""
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
const uint row = thread_position_in_grid.y;
threadgroup float simdgroup_sums[32];
threadgroup float inverse_rms;

const ulong row_base = ulong(row) * {_HIDDEN_SIZE};
const uint dim = tid * {_VALUES_PER_THREAD};
float values[{_VALUES_PER_THREAD}];
float square_sum = 0.0f;
for (uint index = 0; index < {_VALUES_PER_THREAD}; ++index) {{
    const float value = float(input[row_base + dim + index]);
    values[index] = value;
    square_sum += value * value;
}}

square_sum = simd_sum(square_sum);
if (simdgroup == 0) {{
    simdgroup_sums[lane] = 0.0f;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {{
    simdgroup_sums[simdgroup] = square_sum;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simdgroup == 0) {{
    const float total = simd_sum(simdgroup_sums[lane]);
    if (lane == 0) {{
        inverse_rms = metal::precise::rsqrt(
            total / float({_HIDDEN_SIZE}) + {eps});
    }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint index = 0; index < {_VALUES_PER_THREAD}; ++index) {{
    const ulong offset = row_base + dim + index;
    output[offset] = bfloat(
        values[index] * inverse_rms * float(weight[dim + index]));
}}
"""


def _add_source(epsilon: float) -> str:
    eps = _epsilon_literal(epsilon)
    return f"""
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
const uint row = thread_position_in_grid.y;
threadgroup float simdgroup_sums[32];
threadgroup float inverse_rms;

const ulong row_base = ulong(row) * {_HIDDEN_SIZE};
const uint dim = tid * {_VALUES_PER_THREAD};
float values[{_VALUES_PER_THREAD}];
float square_sum = 0.0f;
for (uint index = 0; index < {_VALUES_PER_THREAD}; ++index) {{
    const ulong offset = row_base + dim + index;
    // Keep the reduction in fp32, while narrowing the residual output just
    // as RMSNorm.forward_native does after the residual addition.
    const float value = float(input[offset]) + float(residual[offset]);
    values[index] = value;
    square_sum += value * value;
    residual_output[offset] = bfloat(value);
}}

square_sum = simd_sum(square_sum);
if (simdgroup == 0) {{
    simdgroup_sums[lane] = 0.0f;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {{
    simdgroup_sums[simdgroup] = square_sum;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simdgroup == 0) {{
    const float total = simd_sum(simdgroup_sums[lane]);
    if (lane == 0) {{
        inverse_rms = metal::precise::rsqrt(
            total / float({_HIDDEN_SIZE}) + {eps});
    }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint index = 0; index < {_VALUES_PER_THREAD}; ++index) {{
    const ulong offset = row_base + dim + index;
    output[offset] = bfloat(
        values[index] * inverse_rms * float(weight[dim + index]));
}}
"""


@lru_cache(maxsize=4)
def _rms_kernel(epsilon: float):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 MLX RMSNorm requires Metal")
    return mx.fast.metal_kernel(
        name=f"qwen3_rmsnorm_{_epsilon_tag(epsilon)}",
        input_names=["input", "weight"],
        output_names=["output"],
        source=_rms_source(epsilon),
        # Keep the candidate correct if a future graph change produces a
        # strided residual.  The current Qwen3 matmul/cast path is already
        # row-contiguous, so this does not add a copy on the measured path.
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=4)
def _add_kernel(epsilon: float):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 MLX RMSNorm requires Metal")
    return mx.fast.metal_kernel(
        name=f"qwen3_add_rmsnorm_{_epsilon_tag(epsilon)}",
        input_names=["input", "residual", "weight"],
        output_names=["output", "residual_output"],
        source=_add_source(epsilon),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _flatten(value: Any):
    import mlx.core as mx

    shape = tuple(value.shape)
    if not shape:
        raise ValueError("Qwen3 MLX RMSNorm expects at least one dimension")
    if value.dtype != mx.bfloat16 or shape[-1] != _HIDDEN_SIZE:
        raise ValueError("Qwen3 MLX RMSNorm expects bf16 values with hidden size 1024")
    rows = math.prod(shape[:-1])
    return value.reshape((rows, _HIDDEN_SIZE)), shape


def rms_norm(value: Any, weight: Any, epsilon: float):
    """Run fused fp32-accumulation RMSNorm and return a bf16 MLX array."""
    import mlx.core as mx

    flattened, shape = _flatten(value)
    if weight.dtype != mx.bfloat16 or tuple(weight.shape) != (_HIDDEN_SIZE,):
        raise ValueError("Qwen3 MLX RMSNorm expects a bf16 [1024] weight")
    epsilon = float(epsilon)
    _epsilon_literal(epsilon)
    if flattened.shape[0] == 0:
        return mx.fast.rms_norm(
            value.astype(mx.float32), weight.astype(mx.float32), epsilon
        ).astype(mx.bfloat16)
    output = _rms_kernel(float(epsilon))(
        inputs=[flattened, weight],
        template=[],
        grid=(_THREADS, int(flattened.shape[0]), 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[flattened.shape],
        output_dtypes=[mx.bfloat16],
    )[0]
    return output.reshape(shape)


def add_rms_norm(value: Any, residual: Any, weight: Any, epsilon: float):
    """Fuse residual addition, fp32 RMSNorm, and bf16 residual output."""
    import mlx.core as mx

    flattened, shape = _flatten(value)
    residual_flat, residual_shape = _flatten(residual)
    if residual_shape != shape:
        raise ValueError("Qwen3 MLX fused add RMSNorm requires matching shapes")
    if weight.dtype != mx.bfloat16 or tuple(weight.shape) != (_HIDDEN_SIZE,):
        raise ValueError("Qwen3 MLX RMSNorm expects a bf16 [1024] weight")
    epsilon = float(epsilon)
    _epsilon_literal(epsilon)
    if flattened.shape[0] == 0:
        summed = value.astype(mx.float32) + residual.astype(mx.float32)
        residual_output = summed.astype(mx.bfloat16)
        output = mx.fast.rms_norm(summed, weight.astype(mx.float32), epsilon).astype(
            mx.bfloat16
        )
        return output, residual_output
    output, residual_output = _add_kernel(float(epsilon))(
        inputs=[flattened, residual_flat, weight],
        template=[],
        grid=(_THREADS, int(flattened.shape[0]), 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[flattened.shape, flattened.shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return output.reshape(shape), residual_output.reshape(shape)


def warmup_rms_norm(epsilon: float = 1e-6) -> None:
    """Compile the plain RMSNorm kernel without retaining a model graph."""
    import mlx.core as mx

    value = mx.zeros((1, _HIDDEN_SIZE), dtype=mx.bfloat16)
    weight = mx.ones((_HIDDEN_SIZE,), dtype=mx.bfloat16)
    output = rms_norm(value, weight, epsilon)
    mx.eval(output)


def warmup_add_rms_norm(epsilon: float = 1e-6) -> None:
    """Compile the fused residual-add RMSNorm kernel in isolation."""
    import mlx.core as mx

    value = mx.zeros((1, _HIDDEN_SIZE), dtype=mx.bfloat16)
    residual = mx.zeros_like(value)
    weight = mx.ones((_HIDDEN_SIZE,), dtype=mx.bfloat16)
    normalized, residual_output = add_rms_norm(value, residual, weight, epsilon)
    mx.eval(normalized, residual_output)


def warmup(epsilon: float = 1e-6) -> None:
    """Compile both kernels without retaining a model-sized graph."""
    warmup_rms_norm(epsilon)
    warmup_add_rms_norm(epsilon)


__all__ = [
    "add_rms_norm",
    "rms_norm",
    "warmup",
    "warmup_add_rms_norm",
    "warmup_rms_norm",
]
