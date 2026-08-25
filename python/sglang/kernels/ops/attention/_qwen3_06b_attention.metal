#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM 128
#define HALF_DIM 64
#define NUM_Q_HEADS 16
#define NUM_KV_HEADS 8
#define QKV_WIDTH 4096
#define DECODE_NUM_THREADS 256
#define SIMD_SIZE 32
#define NUM_WARPS 8
#define ATTENTION_SCALE 0.0883883476483f

kernel void qwen3_qknorm_rope_store_bf16(
    const device bfloat* qkv [[buffer(0)]],
    const device bfloat* q_weight [[buffer(1)]],
    const device bfloat* k_weight [[buffer(2)]],
    const device bfloat* cos_sin [[buffer(3)]],
    const device long* positions [[buffer(4)]],
    const device long* slots [[buffer(5)]],
    device bfloat* q_out [[buffer(6)]],
    device bfloat* k_pool [[buffer(7)]],
    device bfloat* v_pool [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant uint& max_position [[buffer(10)]],
    constant uint& pool_slots [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float squares[HEAD_DIM];

  const uint dim = tid;
  const uint head = group.y;
  const uint token = group.z;
  const bool is_q = head < NUM_Q_HEADS;
  const uint local_head = is_q ? head : head - NUM_Q_HEADS;
  const uint source_head = is_q ? local_head : NUM_Q_HEADS + local_head;
  const uint source_base = token * QKV_WIDTH + source_head * HEAD_DIM;

  const long position = positions[token];
  if (position < 0 || ulong(position) >= ulong(max_position)) {
    if (is_q) {
      const uint output_index =
          (token * NUM_Q_HEADS + local_head) * HEAD_DIM + tid;
      q_out[output_index] = bfloat(0.0f);
    }
    return;
  }

  const float x = float(qkv[source_base + dim]);
  squares[dim] = x * x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
    if (dim < stride) {
      squares[dim] += squares[dim + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inverse_rms = metal::precise::rsqrt(
      squares[0] / float(HEAD_DIM) + epsilon);
  const device bfloat* weight = is_q ? q_weight : k_weight;
  const uint peer_dim = dim < HALF_DIM ? dim + HALF_DIM : dim - HALF_DIM;

  // Match the model's staged bf16 contract: RMSNorm narrows before RoPE.
  const bfloat normalized = bfloat(
      x * inverse_rms * float(weight[dim]));
  const bfloat normalized_peer = bfloat(
      float(qkv[source_base + peer_dim]) * inverse_rms *
      float(weight[peer_dim]));

  const uint rope_dim = dim < HALF_DIM ? dim : dim - HALF_DIM;
  const ulong rope_base = ulong(position) * HEAD_DIM;
  const float cosine = float(cos_sin[rope_base + rope_dim]);
  const float sine = float(cos_sin[rope_base + HALF_DIM + rope_dim]);
  const float value = float(normalized);
  const float peer_value = float(normalized_peer);
  const bfloat rotated = bfloat(
      dim < HALF_DIM ? value * cosine - peer_value * sine
                     : value * cosine + peer_value * sine);

  if (is_q) {
    const uint output_index =
        (token * NUM_Q_HEADS + local_head) * HEAD_DIM + dim;
    q_out[output_index] = rotated;
    return;
  }

  const long raw_slot = slots[token];
  if (raw_slot < 0 || ulong(raw_slot) >= ulong(pool_slots)) {
    return;
  }
  const ulong slot = ulong(raw_slot);
  const ulong pool_index =
      (slot * NUM_KV_HEADS + ulong(local_head)) * HEAD_DIM + dim;
  k_pool[pool_index] = rotated;
  const uint v_source = token * QKV_WIDTH +
      (NUM_Q_HEADS + NUM_KV_HEADS + local_head) * HEAD_DIM + dim;
  v_pool[pool_index] = qkv[v_source];
}

inline float simd_max_32(float value) {
  value = max(value, simd_shuffle_xor(value, ushort(16)));
  value = max(value, simd_shuffle_xor(value, ushort(8)));
  value = max(value, simd_shuffle_xor(value, ushort(4)));
  value = max(value, simd_shuffle_xor(value, ushort(2)));
  return max(value, simd_shuffle_xor(value, ushort(1)));
}

inline float simd_sum_32(float value) {
  value += simd_shuffle_xor(value, ushort(16));
  value += simd_shuffle_xor(value, ushort(8));
  value += simd_shuffle_xor(value, ushort(4));
  value += simd_shuffle_xor(value, ushort(2));
  return value + simd_shuffle_xor(value, ushort(1));
}

inline float state_scale(float state_max, float row_max) {
  return state_max == -INFINITY
      ? 0.0f
      : metal::fast::exp(state_max - row_max);
}

kernel void qwen3_radix_decode_bf16(
    const device bfloat* q [[buffer(0)]],
    const device bfloat* k_pool [[buffer(1)]],
    const device bfloat* v_pool [[buffer(2)]],
    const device int* req_to_token [[buffer(3)]],
    const device long* req_pool_indices [[buffer(4)]],
    const device long* seq_lens [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& table_stride [[buffer(7)]],
    constant uint& request_rows [[buffer(8)]],
    constant uint& pool_slots [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float warp_maxes[NUM_WARPS];
  threadgroup float warp_sums[NUM_WARPS];
  threadgroup float warp_accumulators[NUM_WARPS * HEAD_DIM];

  const uint batch = group.x;
  const uint q_head = group.y;
  const uint kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
  const long sequence_length = seq_lens[batch];
  const long raw_request = req_pool_indices[batch];
  const uint q_base = (batch * NUM_Q_HEADS + q_head) * HEAD_DIM;

  if (raw_request < 0 || ulong(raw_request) >= ulong(request_rows) ||
      sequence_length <= 0 || sequence_length > long(table_stride)) {
    for (uint dim = warp * SIMD_SIZE + lane;
         dim < HEAD_DIM;
         dim += DECODE_NUM_THREADS) {
      out[q_base + dim] = bfloat(0.0f);
    }
    return;
  }
  const ulong request = ulong(raw_request);

  float local_max = -INFINITY;
  float local_sum = 0.0f;
  float local_accumulator[HEAD_DIM];
  for (uint dim = 0; dim < HEAD_DIM; ++dim) {
    local_accumulator[dim] = 0.0f;
  }

  for (long token = long(warp * SIMD_SIZE + lane);
       token < sequence_length;
       token += DECODE_NUM_THREADS) {
    const int slot = req_to_token[
        request * ulong(table_stride) + ulong(token)];
    if (slot < 0 || uint(slot) >= pool_slots) {
      continue;
    }
    const ulong pool_base =
        (ulong(slot) * NUM_KV_HEADS + kv_head) * HEAD_DIM;

    float logit = 0.0f;
    for (uint dim = 0; dim < HEAD_DIM; dim += 4) {
      const float4 q_value = float4(
          float(q[q_base + dim]),
          float(q[q_base + dim + 1]),
          float(q[q_base + dim + 2]),
          float(q[q_base + dim + 3]));
      const float4 k_value = float4(
          float(k_pool[pool_base + dim]),
          float(k_pool[pool_base + dim + 1]),
          float(k_pool[pool_base + dim + 2]),
          float(k_pool[pool_base + dim + 3]));
      logit += dot(q_value, k_value);
    }
    logit *= ATTENTION_SCALE;

    const float new_max = max(local_max, logit);
    const float old_scale = metal::fast::exp(local_max - new_max);
    const float weight = metal::fast::exp(logit - new_max);
    local_sum = local_sum * old_scale + weight;
    for (uint dim = 0; dim < HEAD_DIM; ++dim) {
      local_accumulator[dim] = local_accumulator[dim] * old_scale +
          weight * float(v_pool[pool_base + dim]);
    }
    local_max = new_max;
  }

  const float warp_max = simd_max_32(local_max);
  const bool warp_has_tokens = warp_max != -INFINITY;
  const float lane_scale = warp_has_tokens
      ? metal::fast::exp(local_max - warp_max)
      : 0.0f;
  const float warp_sum = warp_has_tokens
      ? simd_sum_32(local_sum * lane_scale)
      : 0.0f;

  if (lane == 0) {
    warp_maxes[warp] = warp_max;
    warp_sums[warp] = warp_sum;
  }
  for (uint dim = 0; dim < HEAD_DIM; ++dim) {
    const float warp_accumulator = simd_sum_32(
        local_accumulator[dim] * lane_scale);
    if (lane == 0) {
      warp_accumulators[warp * HEAD_DIM + dim] =
          warp_has_tokens ? warp_accumulator : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp != 0) {
    return;
  }

  float row_max = -INFINITY;
  for (uint index = 0; index < NUM_WARPS; ++index) {
    row_max = max(row_max, warp_maxes[index]);
  }
  if (row_max == -INFINITY) {
    row_max = 0.0f;
  }

  float row_sum = 0.0f;
  for (uint index = 0; index < NUM_WARPS; ++index) {
    row_sum += warp_sums[index] * state_scale(warp_maxes[index], row_max);
  }
  const float inverse_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
  const uint output_base =
      (batch * NUM_Q_HEADS + q_head) * HEAD_DIM;

  for (uint dim = lane; dim < HEAD_DIM; dim += SIMD_SIZE) {
    float value = 0.0f;
    for (uint index = 0; index < NUM_WARPS; ++index) {
      value += warp_accumulators[index * HEAD_DIM + dim] *
          state_scale(warp_maxes[index], row_max);
    }
    out[output_base + dim] = bfloat(value * inverse_sum);
  }
}
