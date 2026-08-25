"""Deferred-commit Qwen3-0.6B decode attention for an MLX island.

This module deliberately has an MLX-array contract.  A caller may borrow the
Torch-owned Radix KV pool through :class:`MlxTensorView`, run the complete MLX
island, and commit the newly produced K/V tensors only after the island has
finished.  The kernel therefore treats both pool tensors as read-only and
uses ``current_k``/``current_v`` for the final logical token.

The returned array is lazy and newly allocated.  This module does not call
``mx.eval`` and does not synchronize either the Torch or MLX runtime.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC

_DECODE_NUM_THREADS = 1024
_SIMD_SIZE = 32
_NUM_SIMDGROUPS = _DECODE_NUM_THREADS // _SIMD_SIZE

_HEADER = f"""
#define HEAD_DIM {QWEN3_06B_METAL_SPEC.head_dim}
#define NUM_Q_HEADS {QWEN3_06B_METAL_SPEC.num_q_heads}
#define NUM_KV_HEADS {QWEN3_06B_METAL_SPEC.num_kv_heads}
#define Q_HEADS_PER_KV_HEAD \
  (NUM_Q_HEADS / NUM_KV_HEADS)
#define DECODE_NUM_THREADS {_DECODE_NUM_THREADS}
#define SIMD_SIZE {_SIMD_SIZE}
#define NUM_SIMDGROUPS {_NUM_SIMDGROUPS}
#define VALUES_PER_LANE (HEAD_DIM / SIMD_SIZE)
#define ATTENTION_SCALE {QWEN3_06B_METAL_SPEC.attention_scale:.12g}f

inline float qwen3_simd_max_32(float value) {{
  value = max(value, simd_shuffle_xor(value, ushort(16)));
  value = max(value, simd_shuffle_xor(value, ushort(8)));
  value = max(value, simd_shuffle_xor(value, ushort(4)));
  value = max(value, simd_shuffle_xor(value, ushort(2)));
  return max(value, simd_shuffle_xor(value, ushort(1)));
}}

inline float qwen3_simd_sum_32(float value) {{
  value += simd_shuffle_xor(value, ushort(16));
  value += simd_shuffle_xor(value, ushort(8));
  value += simd_shuffle_xor(value, ushort(4));
  value += simd_shuffle_xor(value, ushort(2));
  return value + simd_shuffle_xor(value, ushort(1));
}}

"""

_SOURCE = r"""
// One simdgroup owns one token at a time.  Its 32 lanes split HEAD_DIM into
// four adjacent values each, keeping Q and the online-softmax accumulator in
// registers while making every K/V access coalesced inside one Radix slot.
threadgroup float partial_maxes[NUM_SIMDGROUPS];
threadgroup float partial_sums[NUM_SIMDGROUPS];
// Reused as a 32x32 transpose tile for each of VALUES_PER_LANE components.
threadgroup float partial_outputs[NUM_SIMDGROUPS * SIMD_SIZE];

const uint batch = threadgroup_position_in_grid.z;
const uint q_head = threadgroup_position_in_grid.y;
const uint kv_head = q_head / Q_HEADS_PER_KV_HEAD;
const uint warp = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const long sequence_length = seq_lens[batch];
const long raw_request = req_pool_indices[batch];
const uint q_base = (batch * NUM_Q_HEADS + q_head) * HEAD_DIM;

if (raw_request < 0 || ulong(raw_request) >= ulong(REQUEST_ROWS) ||
    sequence_length <= 0 || sequence_length > long(TABLE_STRIDE)) {
  if (lane == 0) {
    const uint output_base = warp * VALUES_PER_LANE;
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      out[q_base + output_base + index] = static_cast<T>(0.0f);
    }
  }
  return;
}
const ulong request = ulong(raw_request);

const uint lane_dimension = lane * VALUES_PER_LANE;
float query_values[VALUES_PER_LANE];
float local_accumulators[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  query_values[index] =
      float(q[q_base + lane_dimension + index]) * ATTENTION_SCALE;
  local_accumulators[index] = 0.0f;
}

float local_max = -INFINITY;
float local_sum = 0.0f;
for (long token = long(warp);
     token < sequence_length;
     token += NUM_SIMDGROUPS) {
  const bool is_current = token == sequence_length - 1;
  ulong kv_base = (ulong(batch) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  if (!is_current) {
    int slot = -1;
    if (lane == 0) {
      slot = req_to_token[
          request * ulong(TABLE_STRIDE) + ulong(token)];
    }
    // All lanes consume the same token.  Broadcast one table lookup instead
    // of issuing 32 identical indirect reads.
    slot = simd_shuffle(slot, ushort(0));
    if (slot < 0 || uint(slot) >= POOL_SLOTS) {
      continue;
    }
    kv_base = (ulong(slot) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  }

  float logit = 0.0f;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float key_value = is_current
        ? float(current_k[kv_base + dimension])
        : float(k_pool[kv_base + dimension]);
    logit += query_values[index] * key_value;
  }
  logit = qwen3_simd_sum_32(logit);

  const float new_max = max(local_max, logit);
  const float old_scale = metal::fast::exp(local_max - new_max);
  const float weight = metal::fast::exp(logit - new_max);
  local_sum = local_sum * old_scale + weight;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float value = is_current
        ? float(current_v[kv_base + dimension])
        : float(v_pool[kv_base + dimension]);
    local_accumulators[index] =
        local_accumulators[index] * old_scale + weight * value;
  }
  local_max = new_max;
}

if (lane == 0) {
  partial_maxes[warp] = local_max;
  partial_sums[warp] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Every simdgroup performs the same 32-way state merge.  Lane N represents
// partial state N, so its scale can be reused for all four output components.
const float partial_max = partial_maxes[lane];
const float row_max = qwen3_simd_max_32(partial_max);
const bool row_has_tokens = row_max != -INFINITY;
const float partial_scale =
    row_has_tokens && partial_max != -INFINITY
    ? metal::fast::exp(partial_max - row_max)
    : 0.0f;
const float row_sum = row_has_tokens
    ? qwen3_simd_sum_32(partial_sums[lane] * partial_scale)
    : 0.0f;

float merged_values[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  // Transpose [source simdgroup, dimension lane] through threadgroup memory.
  // The destination simdgroup then owns one four-value output segment while
  // its lanes enumerate all 32 source partials for the final reduction.
  partial_outputs[lane * NUM_SIMDGROUPS + warp] =
      local_accumulators[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float numerator = qwen3_simd_sum_32(
      partial_outputs[warp * SIMD_SIZE + lane] * partial_scale);
  merged_values[index] = row_sum == 0.0f ? 0.0f : numerator / row_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lane == 0) {
  const uint output_base = warp * VALUES_PER_LANE;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    out[q_base + output_base + index] =
        static_cast<T>(merged_values[index]);
  }
}
"""


@lru_cache(maxsize=1)
def _deferred_decode_kernel():
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 deferred decode requires the MLX Metal backend")
    return mx.fast.metal_kernel(
        name="qwen3_radix_decode_deferred_bf16",
        input_names=[
            "q",
            "current_k",
            "current_v",
            "k_pool",
            "v_pool",
            "req_to_token",
            "req_pool_indices",
            "seq_lens",
        ],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
        # Torch has already guaranteed contiguous NHD inputs before exposing
        # the borrowed arrays.  Do not let the MLX wrapper hide a materializing
        # copy if that producer contract regresses.
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


def _require_shape(name: str, array: mx.array, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {shape}, found {tuple(array.shape)}"
        )


def qwen3_radix_decode_deferred(
    q: mx.array,
    current_k: mx.array,
    current_v: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    req_to_token: mx.array,
    req_pool_indices: mx.array,
    seq_lens: mx.array,
    *,
    scale: float = QWEN3_06B_METAL_SPEC.attention_scale,
) -> mx.array:
    """Return one-token GQA output without committing the current K/V.

    All arrays must be row-contiguous MLX Metal arrays.  ``k_pool`` and
    ``v_pool`` may be zero-copy, read-only views of Torch-owned MPS tensors.
    The function is intentionally asynchronous: it returns MLX's fresh lazy
    output and leaves the evaluation/export boundary to the enclosing island.
    """
    import mlx.core as mx

    spec = QWEN3_06B_METAL_SPEC
    if not all(
        isinstance(array, mx.array)
        for array in (
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
        )
    ):
        raise RuntimeError("Qwen3 deferred decode inputs must be MLX arrays")
    if not math.isclose(scale, spec.attention_scale, rel_tol=1e-6, abs_tol=0.0):
        raise RuntimeError(
            "attention scale does not match the Qwen3-0.6B kernel: "
            f"expected {spec.attention_scale}, found {scale}"
        )

    batch_size = q.shape[0] if q.ndim == 3 else -1
    _require_shape("q", q, (batch_size, spec.num_q_heads, spec.head_dim))
    current_shape = (batch_size, spec.num_kv_heads, spec.head_dim)
    _require_shape("current_k", current_k, current_shape)
    _require_shape("current_v", current_v, current_shape)
    if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != (
        spec.num_kv_heads,
        spec.head_dim,
    ):
        raise RuntimeError(
            "k_pool must have contiguous NHD shape [slots, 8, 128], found "
            f"{tuple(k_pool.shape)}"
        )
    _require_shape("v_pool", v_pool, tuple(k_pool.shape))
    if req_to_token.ndim != 2:
        raise RuntimeError(
            f"req_to_token must be 2-D, found {tuple(req_to_token.shape)}"
        )
    _require_shape("req_pool_indices", req_pool_indices, (batch_size,))
    _require_shape("seq_lens", seq_lens, (batch_size,))

    for name, array in (
        ("q", q),
        ("current_k", current_k),
        ("current_v", current_v),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
    ):
        if array.dtype != mx.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16, found {array.dtype}")
    for name, array, expected_dtype in (
        ("req_to_token", req_to_token, mx.int32),
        ("req_pool_indices", req_pool_indices, mx.int64),
        ("seq_lens", seq_lens, mx.int64),
    ):
        if array.dtype != expected_dtype:
            raise RuntimeError(f"{name} must be {expected_dtype}, found {array.dtype}")

    if batch_size == 0:
        return mx.empty(q.shape, dtype=q.dtype)

    return _launch_deferred_decode(
        q,
        current_k,
        current_v,
        k_pool,
        v_pool,
        req_to_token,
        req_pool_indices,
        seq_lens,
        request_rows=int(req_to_token.shape[0]),
        table_stride=int(req_to_token.shape[1]),
        pool_slots=int(k_pool.shape[0]),
    )


def _launch_deferred_decode(
    q,
    current_k,
    current_v,
    k_pool,
    v_pool,
    req_to_token,
    req_pool_indices,
    seq_lens,
    *,
    request_rows: int,
    table_stride: int,
    pool_slots: int,
):
    """Launch one already-validated shape specialization."""
    spec = QWEN3_06B_METAL_SPEC
    batch_size = int(q.shape[0])
    return _deferred_decode_kernel()(
        inputs=[
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
        ],
        template=[
            ("T", q.dtype),
            ("TABLE_STRIDE", int(table_stride)),
            ("REQUEST_ROWS", int(request_rows)),
            ("POOL_SLOTS", int(pool_slots)),
        ],
        grid=(_DECODE_NUM_THREADS, spec.num_q_heads, batch_size),
        threadgroup=(_DECODE_NUM_THREADS, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


@lru_cache(maxsize=4)
def warmup_qwen3_radix_decode_deferred(
    *,
    request_rows: int = 1,
    table_stride: int = 1,
    pool_slots: int = 1,
) -> None:
    """Compile the production deferred-attention specialization at startup.

    ``mx.fast.metal_kernel`` resolves its Metal pipeline lazily on the first
    invocation.  Deferring that compilation until a live request would turn a
    shader/toolchain error into a mid-serving failure.  The production request
    table and pool dimensions are Metal template constants, so warm exactly
    those constants while using one-token dummy buffers.  The current-token
    path does not read the dummy table or pool and therefore avoids allocating
    production-sized storage.
    """
    import mlx.core as mx

    request_rows = int(request_rows)
    table_stride = int(table_stride)
    pool_slots = int(pool_slots)
    if request_rows <= 0 or table_stride <= 0 or pool_slots <= 0:
        raise RuntimeError("Qwen3 deferred-decode warmup dimensions must be positive")

    spec = QWEN3_06B_METAL_SPEC
    q = mx.zeros((1, spec.num_q_heads, spec.head_dim), dtype=mx.bfloat16)
    current_k = mx.zeros((1, spec.num_kv_heads, spec.head_dim), dtype=mx.bfloat16)
    current_v = mx.zeros_like(current_k)
    k_pool = mx.zeros((1, spec.num_kv_heads, spec.head_dim), dtype=mx.bfloat16)
    v_pool = mx.zeros_like(k_pool)
    req_to_token = mx.zeros((1, 1), dtype=mx.int32)
    req_pool_indices = mx.zeros((1,), dtype=mx.int64)
    seq_lens = mx.ones((1,), dtype=mx.int64)
    output = _launch_deferred_decode(
        q,
        current_k,
        current_v,
        k_pool,
        v_pool,
        req_to_token,
        req_pool_indices,
        seq_lens,
        request_rows=request_rows,
        table_stride=table_stride,
        pool_slots=pool_slots,
    )
    mx.eval(output)


__all__ = [
    "qwen3_radix_decode_deferred",
    "warmup_qwen3_radix_decode_deferred",
]
