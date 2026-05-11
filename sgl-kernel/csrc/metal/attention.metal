#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define SGL_METAL_DECODE_THREADS_PER_GROUP 128
#define SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP 256
#define SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN 4096
#define SGL_METAL_DECODE_SMALL_CACHED_SEQ_LEN 512
#define SGL_METAL_PREFILL_MAX_CACHED_SEQ_LEN 1024

inline float sgl_metal_fast_exp(float x) {
  return fast::exp2(x * 1.4426950408889634f);
}

struct PrefillAttentionPagedParams {
  float scale;
  int total_q;
  int batch;
  int num_heads;
  int num_kv_heads;
  int max_blocks;
  int block_size;
  int head_dim;
  int causal;
  int q_token_stride;
  int q_head_stride;
  int q_dim_stride;
  int kv_token_stride;
  int kv_head_stride;
  int kv_dim_stride;
  int cache_block_stride;
  int cache_offset_stride;
  int cache_head_stride;
  int cache_dim_stride;
  int out_token_stride;
  int out_head_stride;
  int out_dim_stride;
  int block_tables_batch_stride;
  int block_tables_block_stride;
  int prefix_lens_stride;
  int cu_q_stride;
};

template <typename T>
[[kernel]] void sgl_metal_decode_attention(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch) {
    return;
  }

  const int q_batch = query_batch_offset + batch_idx;
  const int kv_batch = kv_batch_offset + batch_idx;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = q_batch * q_batch_stride + head * q_head_stride;
  const int kv_base = kv_batch * kv_batch_stride + kv_head * kv_head_stride;
  const int out_base = q_batch * out_batch_stride + head * out_head_stride;

  threadgroup float reductions[SGL_METAL_DECODE_THREADS_PER_GROUP];
  threadgroup float score_scratch[SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = seq_len <= SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN;

  float local_max = -INFINITY;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    const int k_base = kv_base + seq * kv_seq_stride;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
          static_cast<float>(k[k_base + dim * kv_dim_stride]);
    }
    const float scaled_score = score * scale;
    if (cache_scores) {
      score_scratch[seq] = scaled_score;
    }
    local_max = max(local_max, scaled_score);
  }

  reductions[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = max(reductions[tid], reductions[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_score = reductions[0];

  float local_sum = 0.0f;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float scaled_score;
    if (cache_scores) {
      scaled_score = score_scratch[seq];
    } else {
      const int k_base = kv_base + seq * kv_seq_stride;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k[k_base + dim * kv_dim_stride]);
      }
      scaled_score = score * scale;
    }
    const float weight = exp(scaled_score - max_score);
    if (cache_scores) {
      score_scratch[seq] = weight;
    }
    local_sum += weight;
  }

  reductions[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float sum_exp = reductions[0];

  for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float acc = 0.0f;
    for (int seq = 0; seq < seq_len; ++seq) {
      const int v_base = kv_base + seq * kv_seq_stride;
      float weight;
      if (cache_scores) {
        weight = score_scratch[seq] / sum_exp;
      } else {
        const int k_base = kv_base + seq * kv_seq_stride;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k[k_base + dim * kv_dim_stride]);
        }
        weight = exp(score * scale - max_score) / sum_exp;
      }
      acc += weight * static_cast<float>(v[v_base + out_dim * kv_dim_stride]);
    }
    out[out_base + out_dim * out_dim_stride] = static_cast<T>(acc);
  }
}

template <typename T>
[[kernel]] void sgl_metal_decode_attention_small(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch) {
    return;
  }

  const int q_batch = query_batch_offset + batch_idx;
  const int kv_batch = kv_batch_offset + batch_idx;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = q_batch * q_batch_stride + head * q_head_stride;
  const int kv_base = kv_batch * kv_batch_stride + kv_head * kv_head_stride;
  const int out_base = q_batch * out_batch_stride + head * out_head_stride;

  threadgroup float reductions[SGL_METAL_DECODE_THREADS_PER_GROUP];
  threadgroup float score_scratch[SGL_METAL_DECODE_SMALL_CACHED_SEQ_LEN];
  const bool cache_scores = seq_len <= SGL_METAL_DECODE_SMALL_CACHED_SEQ_LEN;

  float local_max = -INFINITY;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    const int k_base = kv_base + seq * kv_seq_stride;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
          static_cast<float>(k[k_base + dim * kv_dim_stride]);
    }
    const float scaled_score = score * scale;
    if (cache_scores) {
      score_scratch[seq] = scaled_score;
    }
    local_max = max(local_max, scaled_score);
  }

  reductions[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = max(reductions[tid], reductions[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_score = reductions[0];

  float local_sum = 0.0f;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float scaled_score;
    if (cache_scores) {
      scaled_score = score_scratch[seq];
    } else {
      const int k_base = kv_base + seq * kv_seq_stride;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k[k_base + dim * kv_dim_stride]);
      }
      scaled_score = score * scale;
    }
    const float weight = exp(scaled_score - max_score);
    if (cache_scores) {
      score_scratch[seq] = weight;
    }
    local_sum += weight;
  }

  reductions[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float sum_exp = reductions[0];

  for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float acc = 0.0f;
    for (int seq = 0; seq < seq_len; ++seq) {
      const int v_base = kv_base + seq * kv_seq_stride;
      float weight;
      if (cache_scores) {
        weight = score_scratch[seq] / sum_exp;
      } else {
        const int k_base = kv_base + seq * kv_seq_stride;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k[k_base + dim * kv_dim_stride]);
        }
        weight = exp(score * scale - max_score) / sum_exp;
      }
      acc += weight * static_cast<float>(v[v_base + out_dim * kv_dim_stride]);
    }
    out[out_base + out_dim * out_dim_stride] = static_cast<T>(acc);
  }
}


template <typename T>
[[kernel]] void sgl_metal_decode_attention_paged(
    const device T* q [[buffer(0)]],
    const device T* k_cache [[buffer(1)]],
    const device T* v_cache [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device int* block_tables [[buffer(4)]],
    const device int* context_lens [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& batch [[buffer(7)]],
    constant int& num_heads [[buffer(8)]],
    constant int& num_kv_heads [[buffer(9)]],
    constant int& max_blocks [[buffer(10)]],
    constant int& block_size [[buffer(11)]],
    constant int& head_dim [[buffer(12)]],
    constant int& q_batch_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& cache_block_stride [[buffer(16)]],
    constant int& cache_offset_stride [[buffer(17)]],
    constant int& cache_head_stride [[buffer(18)]],
    constant int& cache_dim_stride [[buffer(19)]],
    constant int& out_batch_stride [[buffer(20)]],
    constant int& out_head_stride [[buffer(21)]],
    constant int& out_dim_stride [[buffer(22)]],
    constant int& block_tables_batch_stride [[buffer(23)]],
    constant int& block_tables_block_stride [[buffer(24)]],
    constant int& context_lens_stride [[buffer(25)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch) {
    return;
  }

  const int max_seq_len = max_blocks * block_size;
  const int seq_len = min(context_lens[batch_idx * context_lens_stride], max_seq_len);
  if (seq_len <= 0) {
    return;
  }

  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = batch_idx * q_batch_stride + head * q_head_stride;
  const int out_base = batch_idx * out_batch_stride + head * out_head_stride;

  threadgroup float reductions[SGL_METAL_DECODE_THREADS_PER_GROUP];
  threadgroup float score_scratch[SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = seq_len <= SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN;

  float local_max = -INFINITY;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    const int block_index = seq / block_size;
    const int block_offset = seq - block_index * block_size;
    const int block = block_tables[batch_idx * block_tables_batch_stride + block_index * block_tables_block_stride];
    const int k_base =
        block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
          static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
    }
    const float scaled_score = score * scale;
    if (cache_scores) {
      score_scratch[seq] = scaled_score;
    }
    local_max = max(local_max, scaled_score);
  }

  reductions[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = max(reductions[tid], reductions[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_score = reductions[0];

  float local_sum = 0.0f;
  for (int seq = static_cast<int>(tid); seq < seq_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float scaled_score;
    if (cache_scores) {
      scaled_score = score_scratch[seq];
    } else {
      const int block_index = seq / block_size;
      const int block_offset = seq - block_index * block_size;
      const int block = block_tables[batch_idx * block_tables_batch_stride + block_index * block_tables_block_stride];
      const int k_base =
          block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
      }
      scaled_score = score * scale;
    }
    const float weight = exp(scaled_score - max_score);
    if (cache_scores) {
      score_scratch[seq] = weight;
    }
    local_sum += weight;
  }

  reductions[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float sum_exp = reductions[0];

  for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float acc = 0.0f;
    for (int seq = 0; seq < seq_len; ++seq) {
      const int block_index = seq / block_size;
      const int block_offset = seq - block_index * block_size;
      const int block = block_tables[batch_idx * block_tables_batch_stride + block_index * block_tables_block_stride];
      const int v_base =
          block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
      float weight;
      if (cache_scores) {
        weight = score_scratch[seq] / sum_exp;
      } else {
        const int k_base =
            block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
        }
        weight = exp(score * scale - max_score) / sum_exp;
      }
      acc += weight * static_cast<float>(v_cache[v_base + out_dim * cache_dim_stride]);
    }
    out[out_base + out_dim * out_dim_stride] = static_cast<T>(acc);
  }
}

template <typename T>
[[kernel]] void sgl_metal_decode_attention_paged_h128_b16_online(
    const device T* q [[buffer(0)]],
    const device T* k_cache [[buffer(1)]],
    const device T* v_cache [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device int* block_tables [[buffer(4)]],
    const device int* context_lens [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& batch [[buffer(7)]],
    constant int& num_heads [[buffer(8)]],
    constant int& num_kv_heads [[buffer(9)]],
    constant int& max_blocks [[buffer(10)]],
    constant int& block_size [[buffer(11)]],
    constant int& head_dim [[buffer(12)]],
    constant int& q_batch_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& cache_block_stride [[buffer(16)]],
    constant int& cache_offset_stride [[buffer(17)]],
    constant int& cache_head_stride [[buffer(18)]],
    constant int& cache_dim_stride [[buffer(19)]],
    constant int& out_batch_stride [[buffer(20)]],
    constant int& out_head_stride [[buffer(21)]],
    constant int& out_dim_stride [[buffer(22)]],
    constant int& block_tables_batch_stride [[buffer(23)]],
    constant int& block_tables_block_stride [[buffer(24)]],
    constant int& context_lens_stride [[buffer(25)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch || head_dim != kHeadDim || block_size != kBlockSize) {
    return;
  }

  const int max_seq_len = max_blocks * kBlockSize;
  const int seq_len = min(context_lens[batch_idx * context_lens_stride], max_seq_len);
  if (seq_len <= 0) {
    return;
  }

  const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = batch_idx * q_batch_stride + head * q_head_stride;
  const int out_base = batch_idx * out_batch_stride + head * out_head_stride;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;

  threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
  threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float v_acc[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
  }

  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);
  const device int* block_table =
      block_tables + batch_idx * block_tables_batch_stride;

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_context_blocks; block_idx += kNumWarps) {
    const int block = block_table[block_idx * block_tables_block_stride];
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
      const int k_base =
          block * cache_block_stride + token_lane * cache_offset_stride + kv_head * cache_head_stride;
      for (int dim = dim_lane; dim < kHeadDim; dim += 2) {
        partial_score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
      }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
      warp_scores[token_lane] = token_idx < seq_len ? paired_score * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, seq_len - block_start_token);
    float block_max = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
      new_max = 0.0f;
    }
    float old_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const float weight = sgl_metal_fast_exp(warp_scores[token] - running_max);
      running_sum += weight;
      const int v_base =
          block * cache_block_stride + token * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] += weight * static_cast<float>(v_cache[v_base + dim * cache_dim_stride]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max = scratch;
  threadgroup float* merge_sum = merge_max + kNumWarps;
  threadgroup float* merge_out = merge_sum + kNumWarps;
  if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
  }
  threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out[dim] = v_acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max = merge_max[warp];
      const float other_sum = merge_sum[warp];
      if (other_max == -INFINITY && other_sum == 0.0f) {
        continue;
      }
      float new_max = max(running_max, other_max);
      if (new_max == -INFINITY) {
        new_max = 0.0f;
      }
      const float this_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
      const float other_correction = other_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max - new_max);
      const threadgroup float* other_out = merge_out + warp * kHeadDim;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] = v_acc[i] * this_correction + other_out[dim] * other_correction;
      }
      running_sum = running_sum * this_correction + other_sum * other_correction;
      running_max = new_max;
    }

    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base + dim * out_dim_stride] = static_cast<T>(v_acc[i] * inv_sum);
    }
  }
}

[[kernel]] void sgl_metal_decode_attention_paged_h128_b16_online_vec_half(
    const device half* q [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device half* out [[buffer(3)]],
    const device int* block_tables [[buffer(4)]],
    const device int* context_lens [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& batch [[buffer(7)]],
    constant int& num_heads [[buffer(8)]],
    constant int& num_kv_heads [[buffer(9)]],
    constant int& max_blocks [[buffer(10)]],
    constant int& block_size [[buffer(11)]],
    constant int& head_dim [[buffer(12)]],
    constant int& q_batch_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& cache_block_stride [[buffer(16)]],
    constant int& cache_offset_stride [[buffer(17)]],
    constant int& cache_head_stride [[buffer(18)]],
    constant int& cache_dim_stride [[buffer(19)]],
    constant int& out_batch_stride [[buffer(20)]],
    constant int& out_head_stride [[buffer(21)]],
    constant int& out_dim_stride [[buffer(22)]],
    constant int& block_tables_batch_stride [[buffer(23)]],
    constant int& block_tables_block_stride [[buffer(24)]],
    constant int& context_lens_stride [[buffer(25)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch || head_dim != kHeadDim || block_size != kBlockSize ||
      q_dim_stride != 1 || cache_dim_stride != 1) {
    return;
  }

  const int max_seq_len = max_blocks * kBlockSize;
  const int seq_len = min(context_lens[batch_idx * context_lens_stride], max_seq_len);
  if (seq_len <= 0) {
    return;
  }

  const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = batch_idx * q_batch_stride + head * q_head_stride;
  const int out_base = batch_idx * out_batch_stride + head * out_head_stride;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;

  threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
  threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float v_acc[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
  }

  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);
  const device int* block_table =
      block_tables + batch_idx * block_tables_batch_stride;
  half4 q_vecs[kHeadDim / 8];
  for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
  }

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_context_blocks; block_idx += kNumWarps) {
    const int block = block_table[block_idx * block_tables_block_stride];
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
      const int k_base =
          block * cache_block_stride + token_lane * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kHeadDim / 8; ++i) {
        const int dim = dim_lane * 4 + i * 8;
        const half4 k_vec = *reinterpret_cast<const device half4*>(k_cache + k_base + dim);
        partial_score += dot(static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec));
      }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
      warp_scores[token_lane] = token_idx < seq_len ? paired_score * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, seq_len - block_start_token);
    float block_max = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
      new_max = 0.0f;
    }
    float old_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const float weight = sgl_metal_fast_exp(warp_scores[token] - running_max);
      running_sum += weight;
      const int v_base =
          block * cache_block_stride + token * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] += weight * static_cast<float>(v_cache[v_base + dim]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max = scratch;
  threadgroup float* merge_sum = merge_max + kNumWarps;
  threadgroup float* merge_out = merge_sum + kNumWarps;
  if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
  }
  threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out[dim] = v_acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max = merge_max[warp];
      const float other_sum = merge_sum[warp];
      if (other_max == -INFINITY && other_sum == 0.0f) {
        continue;
      }
      float new_max = max(running_max, other_max);
      if (new_max == -INFINITY) {
        new_max = 0.0f;
      }
      const float this_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
      const float other_correction = other_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max - new_max);
      const threadgroup float* other_out = merge_out + warp * kHeadDim;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] = v_acc[i] * this_correction + other_out[dim] * other_correction;
      }
      running_sum = running_sum * this_correction + other_sum * other_correction;
      running_max = new_max;
    }

    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
  }
}

[[kernel]] void sgl_metal_decode_attention_paged_h128_b16_fused_kv_vec_half(
    const device half* q [[buffer(0)]],
    const device half* k_new [[buffer(1)]],
    const device half* v_new [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    device half* out [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* context_lens [[buffer(7)]],
    const device int* slot_mapping [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant int& batch [[buffer(10)]],
    constant int& num_heads [[buffer(11)]],
    constant int& num_kv_heads [[buffer(12)]],
    constant int& max_blocks [[buffer(13)]],
    constant int& q_batch_stride [[buffer(14)]],
    constant int& q_head_stride [[buffer(15)]],
    constant int& new_token_stride [[buffer(16)]],
    constant int& new_head_stride [[buffer(17)]],
    constant int& cache_block_stride [[buffer(18)]],
    constant int& cache_offset_stride [[buffer(19)]],
    constant int& cache_head_stride [[buffer(20)]],
    constant int& out_batch_stride [[buffer(21)]],
    constant int& out_head_stride [[buffer(22)]],
    constant int& block_tables_batch_stride [[buffer(23)]],
    constant int& block_tables_block_stride [[buffer(24)]],
    constant int& context_lens_stride [[buffer(25)]],
    constant int& slot_mapping_stride [[buffer(26)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (head >= num_heads || batch_idx >= batch) {
    return;
  }

  const uint local = thread_position.x;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;
  const int slot = slot_mapping[batch_idx * slot_mapping_stride];

  if (head < num_kv_heads && slot >= 0) {
    const int write_block = slot / kBlockSize;
    const int write_offset = slot - write_block * kBlockSize;
    const int cache_write_base =
        write_block * cache_block_stride + write_offset * cache_offset_stride + head * cache_head_stride;
    const int new_write_base = batch_idx * new_token_stride + head * new_head_stride;
    for (int dim = static_cast<int>(local); dim < kHeadDim; dim += SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP) {
      k_cache[cache_write_base + dim] = k_new[new_write_base + dim];
      v_cache[cache_write_base + dim] = v_new[new_write_base + dim];
    }
  }

  const int max_seq_len = max_blocks * kBlockSize;
  const int seq_len = min(context_lens[batch_idx * context_lens_stride], max_seq_len);
  if (seq_len <= 0) {
    return;
  }

  const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  const int current_token_idx = seq_len - 1;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = batch_idx * q_batch_stride + head * q_head_stride;
  const int new_base = batch_idx * new_token_stride + kv_head * new_head_stride;
  const int out_base = batch_idx * out_batch_stride + head * out_head_stride;

  threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
  threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float v_acc[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
  }

  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);
  const device int* block_table = block_tables + batch_idx * block_tables_batch_stride;
  half4 q_vecs[kHeadDim / 8];
  for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
  }

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_context_blocks; block_idx += kNumWarps) {
    const int block = block_table[block_idx * block_tables_block_stride];
    const int token_idx = block_idx * kBlockSize + token_lane;
    const bool is_current_token = token_idx == current_token_idx;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
      const int k_base =
          block * cache_block_stride + token_lane * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kHeadDim / 8; ++i) {
        const int dim = dim_lane * 4 + i * 8;
        const half4 k_vec = is_current_token
            ? *reinterpret_cast<const device half4*>(k_new + new_base + dim)
            : *reinterpret_cast<device half4*>(k_cache + k_base + dim);
        partial_score += dot(static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec));
      }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
      warp_scores[token_lane] = token_idx < seq_len ? paired_score * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, seq_len - block_start_token);
    float block_max = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
      new_max = 0.0f;
    }
    float old_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const int absolute_token = block_start_token + token;
      const bool is_current_value = absolute_token == current_token_idx;
      const float weight = sgl_metal_fast_exp(warp_scores[token] - running_max);
      running_sum += weight;
      const int v_base =
          block * cache_block_stride + token * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        const half v_value = is_current_value ? v_new[new_base + dim] : v_cache[v_base + dim];
        v_acc[i] += weight * static_cast<float>(v_value);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max = scratch;
  threadgroup float* merge_sum = merge_max + kNumWarps;
  threadgroup float* merge_out = merge_sum + kNumWarps;
  if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
  }
  threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out[dim] = v_acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max = merge_max[warp];
      const float other_sum = merge_sum[warp];
      if (other_max == -INFINITY && other_sum == 0.0f) {
        continue;
      }
      float new_max = max(running_max, other_max);
      if (new_max == -INFINITY) {
        new_max = 0.0f;
      }
      const float this_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
      const float other_correction = other_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max - new_max);
      const threadgroup float* other_out = merge_out + warp * kHeadDim;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] = v_acc[i] * this_correction + other_out[dim] * other_correction;
      }
      running_sum = running_sum * this_correction + other_sum * other_correction;
      running_max = new_max;
    }

    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
  }
}

[[kernel]] void sgl_metal_decode_attention_paged_h128_b16_gqa2_online_vec_half(
    const device half* q [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device half* out [[buffer(3)]],
    const device int* block_tables [[buffer(4)]],
    const device int* context_lens [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& batch [[buffer(7)]],
    constant int& num_heads [[buffer(8)]],
    constant int& num_kv_heads [[buffer(9)]],
    constant int& max_blocks [[buffer(10)]],
    constant int& block_size [[buffer(11)]],
    constant int& head_dim [[buffer(12)]],
    constant int& q_batch_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& cache_block_stride [[buffer(16)]],
    constant int& cache_offset_stride [[buffer(17)]],
    constant int& cache_head_stride [[buffer(18)]],
    constant int& cache_dim_stride [[buffer(19)]],
    constant int& out_batch_stride [[buffer(20)]],
    constant int& out_head_stride [[buffer(21)]],
    constant int& out_dim_stride [[buffer(22)]],
    constant int& block_tables_batch_stride [[buffer(23)]],
    constant int& block_tables_block_stride [[buffer(24)]],
    constant int& context_lens_stride [[buffer(25)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int kv_head = static_cast<int>(group_id.x);
  const int batch_idx = static_cast<int>(group_id.y);
  if (kv_head >= num_kv_heads || batch_idx >= batch || num_heads != num_kv_heads * 2 ||
      head_dim != kHeadDim || block_size != kBlockSize || q_dim_stride != 1 || cache_dim_stride != 1 ||
      out_dim_stride != 1) {
    return;
  }

  const int max_seq_len = max_blocks * kBlockSize;
  const int seq_len = min(context_lens[batch_idx * context_lens_stride], max_seq_len);
  if (seq_len <= 0) {
    return;
  }

  const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  const int head0 = kv_head * 2;
  const int head1 = head0 + 1;
  const int q_base0 = batch_idx * q_batch_stride + head0 * q_head_stride;
  const int q_base1 = batch_idx * q_batch_stride + head1 * q_head_stride;
  const int out_base0 = batch_idx * out_batch_stride + head0 * out_head_stride;
  const int out_base1 = batch_idx * out_batch_stride + head1 * out_head_stride;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;

  threadgroup float scratch[2 * kNumWarps * kHeadDim + 4 * kNumWarps];
  threadgroup float* warp_scores0 = scratch + warp_idx * kBlockSize;
  threadgroup float* warp_scores1 = scratch + kNumWarps * kBlockSize + warp_idx * kBlockSize;

  float running_max0 = -INFINITY;
  float running_max1 = -INFINITY;
  float running_sum0 = 0.0f;
  float running_sum1 = 0.0f;
  float v_acc0[kVElemsPerLane];
  float v_acc1[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc0[i] = 0.0f;
    v_acc1[i] = 0.0f;
  }

  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);
  const device int* block_table = block_tables + batch_idx * block_tables_batch_stride;
  half4 q_vecs0[kHeadDim / 8];
  half4 q_vecs1[kHeadDim / 8];
  for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs0[i] = *reinterpret_cast<const device half4*>(q + q_base0 + dim);
    q_vecs1[i] = *reinterpret_cast<const device half4*>(q + q_base1 + dim);
  }

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_context_blocks; block_idx += kNumWarps) {
    const int block = block_table[block_idx * block_tables_block_stride];
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score0 = 0.0f;
    float partial_score1 = 0.0f;
    if (token_idx < seq_len) {
      const int k_base =
          block * cache_block_stride + token_lane * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kHeadDim / 8; ++i) {
        const int dim = dim_lane * 4 + i * 8;
        const half4 k_vec = *reinterpret_cast<const device half4*>(k_cache + k_base + dim);
        const float4 k_float = static_cast<float4>(k_vec);
        partial_score0 += dot(static_cast<float4>(q_vecs0[i]), k_float);
        partial_score1 += dot(static_cast<float4>(q_vecs1[i]), k_float);
      }
    }
    const float paired_score0 = partial_score0 + simd_shuffle_xor(partial_score0, 1);
    const float paired_score1 = partial_score1 + simd_shuffle_xor(partial_score1, 1);
    if (dim_lane == 0) {
      const bool valid_token = token_idx < seq_len;
      warp_scores0[token_lane] = valid_token ? paired_score0 * scale : -INFINITY;
      warp_scores1[token_lane] = valid_token ? paired_score1 * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, seq_len - block_start_token);
    float block_max0 = -INFINITY;
    float block_max1 = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max0 = max(block_max0, warp_scores0[token]);
      block_max1 = max(block_max1, warp_scores1[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max0 = max(block_max0, simd_shuffle_xor(block_max0, mask));
      block_max1 = max(block_max1, simd_shuffle_xor(block_max1, mask));
    }

    float new_max0 = max(running_max0, block_max0);
    float new_max1 = max(running_max1, block_max1);
    if (new_max0 == -INFINITY) {
      new_max0 = 0.0f;
    }
    if (new_max1 == -INFINITY) {
      new_max1 = 0.0f;
    }
    const float old_correction0 = running_max0 == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max0 - new_max0);
    const float old_correction1 = running_max1 == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max1 - new_max1);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc0[i] *= old_correction0;
      v_acc1[i] *= old_correction1;
    }
    running_sum0 *= old_correction0;
    running_sum1 *= old_correction1;
    running_max0 = new_max0;
    running_max1 = new_max1;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const float weight0 = sgl_metal_fast_exp(warp_scores0[token] - running_max0);
      const float weight1 = sgl_metal_fast_exp(warp_scores1[token] - running_max1);
      running_sum0 += weight0;
      running_sum1 += weight1;
      const int v_base =
          block * cache_block_stride + token * cache_offset_stride + kv_head * cache_head_stride;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        const float v_value = static_cast<float>(v_cache[v_base + dim]);
        v_acc0[i] += weight0 * v_value;
        v_acc1[i] += weight1 * v_value;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max0 = scratch;
  threadgroup float* merge_sum0 = merge_max0 + kNumWarps;
  threadgroup float* merge_max1 = merge_sum0 + kNumWarps;
  threadgroup float* merge_sum1 = merge_max1 + kNumWarps;
  threadgroup float* merge_out0 = merge_sum1 + kNumWarps;
  threadgroup float* merge_out1 = merge_out0 + kNumWarps * kHeadDim;
  if (lane == 0) {
    merge_max0[warp_idx] = running_max0;
    merge_sum0[warp_idx] = running_sum0;
    merge_max1[warp_idx] = running_max1;
    merge_sum1[warp_idx] = running_sum1;
  }
  threadgroup float* this_out0 = merge_out0 + warp_idx * kHeadDim;
  threadgroup float* this_out1 = merge_out1 + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out0[dim] = v_acc0[i];
    this_out1[dim] = v_acc1[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max0 = merge_max0[warp];
      const float other_sum0 = merge_sum0[warp];
      if (!(other_max0 == -INFINITY && other_sum0 == 0.0f)) {
        float new_max = max(running_max0, other_max0);
        if (new_max == -INFINITY) {
          new_max = 0.0f;
        }
        const float this_correction = running_max0 == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max0 - new_max);
        const float other_correction = other_max0 == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max0 - new_max);
        const threadgroup float* other_out = merge_out0 + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
          const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
          v_acc0[i] = v_acc0[i] * this_correction + other_out[dim] * other_correction;
        }
        running_sum0 = running_sum0 * this_correction + other_sum0 * other_correction;
        running_max0 = new_max;
      }

      const float other_max1 = merge_max1[warp];
      const float other_sum1 = merge_sum1[warp];
      if (!(other_max1 == -INFINITY && other_sum1 == 0.0f)) {
        float new_max = max(running_max1, other_max1);
        if (new_max == -INFINITY) {
          new_max = 0.0f;
        }
        const float this_correction = running_max1 == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max1 - new_max);
        const float other_correction = other_max1 == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max1 - new_max);
        const threadgroup float* other_out = merge_out1 + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
          const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
          v_acc1[i] = v_acc1[i] * this_correction + other_out[dim] * other_correction;
        }
        running_sum1 = running_sum1 * this_correction + other_sum1 * other_correction;
        running_max1 = new_max;
      }
    }

    const float inv_sum0 = 1.0f / (running_sum0 + 1e-6f);
    const float inv_sum1 = 1.0f / (running_sum1 + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base0 + dim] = static_cast<half>(v_acc0[i] * inv_sum0);
      out[out_base1 + dim] = static_cast<half>(v_acc1[i] * inv_sum1);
    }
  }
}


template <typename T>
[[kernel]] void sgl_metal_flash_attn_varlen(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device int* cu_seqlens_q [[buffer(4)]],
    const device int* cu_seqlens_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& total_q [[buffer(7)]],
    constant int& num_seqs [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& causal [[buffer(12)]],
    constant int& q_token_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& kv_token_stride [[buffer(16)]],
    constant int& kv_head_stride [[buffer(17)]],
    constant int& kv_dim_stride [[buffer(18)]],
    constant int& out_token_stride [[buffer(19)]],
    constant int& out_head_stride [[buffer(20)]],
    constant int& out_dim_stride [[buffer(21)]],
    constant int& cu_q_stride [[buffer(22)]],
    constant int& cu_k_stride [[buffer(23)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const int head = static_cast<int>(group_id.x);
  const int q_token = static_cast<int>(group_id.y);
  if (head >= num_heads || q_token >= total_q) {
    return;
  }

  int seq_id = -1;
  int q_start = 0;
  int q_end = 0;
  int k_start = 0;
  int k_end = 0;
  for (int seq = 0; seq < num_seqs; ++seq) {
    const int candidate_q_start = cu_seqlens_q[seq * cu_q_stride];
    const int candidate_q_end = cu_seqlens_q[(seq + 1) * cu_q_stride];
    if (q_token >= candidate_q_start && q_token < candidate_q_end) {
      seq_id = seq;
      q_start = candidate_q_start;
      q_end = candidate_q_end;
      k_start = cu_seqlens_k[seq * cu_k_stride];
      k_end = cu_seqlens_k[(seq + 1) * cu_k_stride];
      break;
    }
  }

  const int out_base = q_token * out_token_stride + head * out_head_stride;
  if (seq_id < 0) {
    for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
      out[out_base + out_dim * out_dim_stride] = static_cast<T>(0.0f);
    }
    return;
  }

  const int seq_q_len = q_end - q_start;
  const int seq_k_len = k_end - k_start;
  const int q_offset = q_token - q_start;
  int visible_len = seq_k_len;
  if (causal != 0) {
    visible_len = min(seq_k_len, max(0, q_offset + seq_k_len - seq_q_len + 1));
  }
  if (visible_len <= 0) {
    for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
      out[out_base + out_dim * out_dim_stride] = static_cast<T>(0.0f);
    }
    return;
  }

  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = q_token * q_token_stride + head * q_head_stride;

  threadgroup float reductions[SGL_METAL_DECODE_THREADS_PER_GROUP];
  threadgroup float score_scratch[SGL_METAL_PREFILL_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = visible_len <= SGL_METAL_PREFILL_MAX_CACHED_SEQ_LEN;

  float local_max = -INFINITY;
  for (int seq = static_cast<int>(tid); seq < visible_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    const int k_base = (k_start + seq) * kv_token_stride + kv_head * kv_head_stride;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
          static_cast<float>(k[k_base + dim * kv_dim_stride]);
    }
    const float scaled_score = score * scale;
    if (cache_scores) {
      score_scratch[seq] = scaled_score;
    }
    local_max = max(local_max, scaled_score);
  }

  reductions[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = max(reductions[tid], reductions[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_score = reductions[0];

  float local_sum = 0.0f;
  for (int seq = static_cast<int>(tid); seq < visible_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float scaled_score;
    if (cache_scores) {
      scaled_score = score_scratch[seq];
    } else {
      const int k_base = (k_start + seq) * kv_token_stride + kv_head * kv_head_stride;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k[k_base + dim * kv_dim_stride]);
      }
      scaled_score = score * scale;
    }
    const float weight = exp(scaled_score - max_score);
    if (cache_scores) {
      score_scratch[seq] = weight;
    }
    local_sum += weight;
  }

  reductions[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float sum_exp = reductions[0];

  for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float acc = 0.0f;
    for (int seq = 0; seq < visible_len; ++seq) {
      const int v_base = (k_start + seq) * kv_token_stride + kv_head * kv_head_stride;
      float weight;
      if (cache_scores) {
        weight = score_scratch[seq] / sum_exp;
      } else {
        const int k_base = (k_start + seq) * kv_token_stride + kv_head * kv_head_stride;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k[k_base + dim * kv_dim_stride]);
        }
        weight = exp(score * scale - max_score) / sum_exp;
      }
      acc += weight * static_cast<float>(v[v_base + out_dim * kv_dim_stride]);
    }
    out[out_base + out_dim * out_dim_stride] = static_cast<T>(acc);
  }
}

[[kernel]] void sgl_metal_flash_attn_varlen_h128_online_vec_half(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    device half* out [[buffer(3)]],
    const device int* cu_seqlens_q [[buffer(4)]],
    const device int* cu_seqlens_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& total_q [[buffer(7)]],
    constant int& num_seqs [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& causal [[buffer(12)]],
    constant int& q_token_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& kv_token_stride [[buffer(16)]],
    constant int& kv_head_stride [[buffer(17)]],
    constant int& kv_dim_stride [[buffer(18)]],
    constant int& out_token_stride [[buffer(19)]],
    constant int& out_head_stride [[buffer(20)]],
    constant int& out_dim_stride [[buffer(21)]],
    constant int& cu_q_stride [[buffer(22)]],
    constant int& cu_k_stride [[buffer(23)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int head = static_cast<int>(group_id.x);
  const int q_token = static_cast<int>(group_id.y);
  if (head >= num_heads || q_token >= total_q || head_dim != kHeadDim ||
      q_dim_stride != 1 || kv_dim_stride != 1 || out_dim_stride != 1) {
    return;
  }

  int q_start = 0;
  int q_end = 0;
  int k_start = 0;
  int k_end = 0;
  bool found = false;
  for (int seq = 0; seq < num_seqs; ++seq) {
    const int candidate_q_start = cu_seqlens_q[seq * cu_q_stride];
    const int candidate_q_end = cu_seqlens_q[(seq + 1) * cu_q_stride];
    if (q_token >= candidate_q_start && q_token < candidate_q_end) {
      q_start = candidate_q_start;
      q_end = candidate_q_end;
      k_start = cu_seqlens_k[seq * cu_k_stride];
      k_end = cu_seqlens_k[(seq + 1) * cu_k_stride];
      found = true;
      break;
    }
  }

  const int out_base = q_token * out_token_stride + head * out_head_stride;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;
  if (!found) {
    if (warp_idx == 0) {
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        out[out_base + dim] = half(0.0f);
      }
    }
    return;
  }

  const int seq_q_len = q_end - q_start;
  const int seq_k_len = k_end - k_start;
  const int q_offset = q_token - q_start;
  int visible_len = seq_k_len;
  if (causal != 0) {
    visible_len = min(seq_k_len, max(0, q_offset + seq_k_len - seq_q_len + 1));
  }
  if (visible_len <= 0) {
    if (warp_idx == 0) {
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        out[out_base + dim] = half(0.0f);
      }
    }
    return;
  }

  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = q_token * q_token_stride + head * q_head_stride;
  const int num_blocks = (visible_len + kBlockSize - 1) / kBlockSize;
  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);

  half4 q_vecs[kHeadDim / 8];
  for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
  }

  threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
  threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float v_acc[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
  }

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_blocks; block_idx += kNumWarps) {
    const int token_offset = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_offset < visible_len) {
      const int k_base = (k_start + token_offset) * kv_token_stride + kv_head * kv_head_stride;
      for (int i = 0; i < kHeadDim / 8; ++i) {
        const int dim = dim_lane * 4 + i * 8;
        const half4 k_vec = *reinterpret_cast<const device half4*>(k + k_base + dim);
        partial_score += dot(static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec));
      }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
      warp_scores[token_lane] = token_offset < visible_len ? paired_score * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, visible_len - block_start_token);
    float block_max = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
      new_max = 0.0f;
    }
    const float old_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const float weight = sgl_metal_fast_exp(warp_scores[token] - running_max);
      running_sum += weight;
      const int v_base = (k_start + block_start_token + token) * kv_token_stride + kv_head * kv_head_stride;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] += weight * static_cast<float>(v[v_base + dim]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max = scratch;
  threadgroup float* merge_sum = merge_max + kNumWarps;
  threadgroup float* merge_out = merge_sum + kNumWarps;
  if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
  }
  threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out[dim] = v_acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max = merge_max[warp];
      const float other_sum = merge_sum[warp];
      if (other_max == -INFINITY && other_sum == 0.0f) {
        continue;
      }
      float new_max = max(running_max, other_max);
      if (new_max == -INFINITY) {
        new_max = 0.0f;
      }
      const float this_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
      const float other_correction = other_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max - new_max);
      const threadgroup float* other_out = merge_out + warp * kHeadDim;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] = v_acc[i] * this_correction + other_out[dim] * other_correction;
      }
      running_sum = running_sum * this_correction + other_sum * other_correction;
      running_max = new_max;
    }

    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
  }
}

[[kernel]] void sgl_metal_prefill_attention_paged_h128_b16_online_vec_half(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    const device half* k_cache [[buffer(3)]],
    const device half* v_cache [[buffer(4)]],
    device half* out [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* prefix_lens [[buffer(7)]],
    const device int* cu_seqlens_q [[buffer(8)]],
    constant PrefillAttentionPagedParams& params [[buffer(9)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int kHeadDim = 128;
  constexpr int kBlockSize = 16;
  constexpr int kNumSimdLanes = 32;
  constexpr int kNumWarps = SGL_METAL_ONLINE_DECODE_THREADS_PER_GROUP / kNumSimdLanes;
  constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

  const int head = static_cast<int>(group_id.x);
  const int q_token = static_cast<int>(group_id.y);
  if (head >= params.num_heads || q_token >= params.total_q || params.head_dim != kHeadDim ||
      params.block_size != kBlockSize || params.q_dim_stride != 1 || params.kv_dim_stride != 1 ||
      params.cache_dim_stride != 1 || params.out_dim_stride != 1) {
    return;
  }

  int seq_id = -1;
  int q_start = 0;
  int q_end = 0;
  for (int seq = 0; seq < params.batch; ++seq) {
    const int candidate_q_start = cu_seqlens_q[seq * params.cu_q_stride];
    const int candidate_q_end = cu_seqlens_q[(seq + 1) * params.cu_q_stride];
    if (q_token >= candidate_q_start && q_token < candidate_q_end) {
      seq_id = seq;
      q_start = candidate_q_start;
      q_end = candidate_q_end;
      break;
    }
  }

  const int out_base = q_token * params.out_token_stride + head * params.out_head_stride;
  const uint lane = simd_lid;
  const uint warp_idx = simd_tid;
  if (seq_id < 0) {
    if (warp_idx == 0) {
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        out[out_base + dim] = half(0.0f);
      }
    }
    return;
  }

  const int seq_q_len = q_end - q_start;
  const int q_offset = q_token - q_start;
  const int max_prefix_len = params.max_blocks * kBlockSize;
  const int prefix_len = max(0, min(prefix_lens[seq_id * params.prefix_lens_stride], max_prefix_len));
  const int visible_suffix_len = params.causal != 0 ? min(seq_q_len, q_offset + 1) : seq_q_len;
  const int visible_len = prefix_len + visible_suffix_len;
  if (visible_len <= 0) {
    if (warp_idx == 0) {
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        out[out_base + dim] = half(0.0f);
      }
    }
    return;
  }

  const int kv_head = head / (params.num_heads / params.num_kv_heads);
  const int q_base = q_token * params.q_token_stride + head * params.q_head_stride;
  const int num_blocks = (visible_len + kBlockSize - 1) / kBlockSize;
  const int token_lane = static_cast<int>(lane >> 1);
  const int dim_lane = static_cast<int>(lane & 1);

  half4 q_vecs[kHeadDim / 8];
  for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
  }

  threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
  threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;
  const device int* block_table = block_tables + seq_id * params.block_tables_batch_stride;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float v_acc[kVElemsPerLane];
  for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
  }

  for (int block_idx = static_cast<int>(warp_idx); block_idx < num_blocks; block_idx += kNumWarps) {
    const int token_offset = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_offset < visible_len) {
      if (token_offset < prefix_len) {
        const int cache_block_idx = token_offset / kBlockSize;
        const int block_offset = token_offset - cache_block_idx * kBlockSize;
        const int block = block_table[cache_block_idx * params.block_tables_block_stride];
        const int k_base =
            block * params.cache_block_stride + block_offset * params.cache_offset_stride +
            kv_head * params.cache_head_stride;
        for (int i = 0; i < kHeadDim / 8; ++i) {
          const int dim = dim_lane * 4 + i * 8;
          const half4 k_vec = *reinterpret_cast<const device half4*>(k_cache + k_base + dim);
          partial_score += dot(static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec));
        }
      } else {
        const int suffix_offset = token_offset - prefix_len;
        const int k_base = (q_start + suffix_offset) * params.kv_token_stride + kv_head * params.kv_head_stride;
        for (int i = 0; i < kHeadDim / 8; ++i) {
          const int dim = dim_lane * 4 + i * 8;
          const half4 k_vec = *reinterpret_cast<const device half4*>(k + k_base + dim);
          partial_score += dot(static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec));
        }
      }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
      warp_scores[token_lane] = token_offset < visible_len ? paired_score * params.scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_start_token = block_idx * kBlockSize;
    const int block_valid_tokens = min(kBlockSize, visible_len - block_start_token);
    float block_max = -INFINITY;
    for (int token = static_cast<int>(lane); token < block_valid_tokens; token += kNumSimdLanes) {
      block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
      new_max = 0.0f;
    }
    const float old_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
      const int token_index = block_start_token + token;
      const float weight = sgl_metal_fast_exp(warp_scores[token] - running_max);
      running_sum += weight;
      if (token_index < prefix_len) {
        const int cache_block_idx = token_index / kBlockSize;
        const int block_offset = token_index - cache_block_idx * kBlockSize;
        const int block = block_table[cache_block_idx * params.block_tables_block_stride];
        const int v_base =
            block * params.cache_block_stride + block_offset * params.cache_offset_stride +
            kv_head * params.cache_head_stride;
        for (int i = 0; i < kVElemsPerLane; ++i) {
          const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
          v_acc[i] += weight * static_cast<float>(v_cache[v_base + dim]);
        }
      } else {
        const int suffix_offset = token_index - prefix_len;
        const int v_base = (q_start + suffix_offset) * params.kv_token_stride + kv_head * params.kv_head_stride;
        for (int i = 0; i < kVElemsPerLane; ++i) {
          const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
          v_acc[i] += weight * static_cast<float>(v[v_base + dim]);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* merge_max = scratch;
  threadgroup float* merge_sum = merge_max + kNumWarps;
  threadgroup float* merge_out = merge_sum + kNumWarps;
  if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
  }
  threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
  for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
    this_out[dim] = v_acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
      const float other_max = merge_max[warp];
      const float other_sum = merge_sum[warp];
      if (other_max == -INFINITY && other_sum == 0.0f) {
        continue;
      }
      float new_max = max(running_max, other_max);
      if (new_max == -INFINITY) {
        new_max = 0.0f;
      }
      const float this_correction = running_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(running_max - new_max);
      const float other_correction = other_max == -INFINITY ? 0.0f : sgl_metal_fast_exp(other_max - new_max);
      const threadgroup float* other_out = merge_out + warp * kHeadDim;
      for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
        v_acc[i] = v_acc[i] * this_correction + other_out[dim] * other_correction;
      }
      running_sum = running_sum * this_correction + other_sum * other_correction;
      running_max = new_max;
    }

    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
      const int dim = static_cast<int>(lane) + i * kNumSimdLanes;
      out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
  }
}

template [[host_name("sgl_metal_flash_attn_varlen_float")]] [[kernel]] void
sgl_metal_flash_attn_varlen<float>(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    const device int* cu_seqlens_q [[buffer(4)]],
    const device int* cu_seqlens_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& total_q [[buffer(7)]],
    constant int& num_seqs [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& causal [[buffer(12)]],
    constant int& q_token_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& kv_token_stride [[buffer(16)]],
    constant int& kv_head_stride [[buffer(17)]],
    constant int& kv_dim_stride [[buffer(18)]],
    constant int& out_token_stride [[buffer(19)]],
    constant int& out_head_stride [[buffer(20)]],
    constant int& out_dim_stride [[buffer(21)]],
    constant int& cu_q_stride [[buffer(22)]],
    constant int& cu_k_stride [[buffer(23)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_flash_attn_varlen_half")]] [[kernel]] void
sgl_metal_flash_attn_varlen<half>(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    device half* out [[buffer(3)]],
    const device int* cu_seqlens_q [[buffer(4)]],
    const device int* cu_seqlens_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& total_q [[buffer(7)]],
    constant int& num_seqs [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& causal [[buffer(12)]],
    constant int& q_token_stride [[buffer(13)]],
    constant int& q_head_stride [[buffer(14)]],
    constant int& q_dim_stride [[buffer(15)]],
    constant int& kv_token_stride [[buffer(16)]],
    constant int& kv_head_stride [[buffer(17)]],
    constant int& kv_dim_stride [[buffer(18)]],
    constant int& out_token_stride [[buffer(19)]],
    constant int& out_head_stride [[buffer(20)]],
    constant int& out_dim_stride [[buffer(21)]],
    constant int& cu_q_stride [[buffer(22)]],
    constant int& cu_k_stride [[buffer(23)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);


template <typename T>
[[kernel]] void sgl_metal_prefill_attention_paged(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* k_cache [[buffer(3)]],
    const device T* v_cache [[buffer(4)]],
    device T* out [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* prefix_lens [[buffer(7)]],
    const device int* cu_seqlens_q [[buffer(8)]],
    constant PrefillAttentionPagedParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const float scale = params.scale;
  const int total_q = params.total_q;
  const int batch = params.batch;
  const int num_heads = params.num_heads;
  const int num_kv_heads = params.num_kv_heads;
  const int max_blocks = params.max_blocks;
  const int block_size = params.block_size;
  const int head_dim = params.head_dim;
  const int causal = params.causal;
  const int q_token_stride = params.q_token_stride;
  const int q_head_stride = params.q_head_stride;
  const int q_dim_stride = params.q_dim_stride;
  const int kv_token_stride = params.kv_token_stride;
  const int kv_head_stride = params.kv_head_stride;
  const int kv_dim_stride = params.kv_dim_stride;
  const int cache_block_stride = params.cache_block_stride;
  const int cache_offset_stride = params.cache_offset_stride;
  const int cache_head_stride = params.cache_head_stride;
  const int cache_dim_stride = params.cache_dim_stride;
  const int out_token_stride = params.out_token_stride;
  const int out_head_stride = params.out_head_stride;
  const int out_dim_stride = params.out_dim_stride;
  const int block_tables_batch_stride = params.block_tables_batch_stride;
  const int block_tables_block_stride = params.block_tables_block_stride;
  const int prefix_lens_stride = params.prefix_lens_stride;
  const int cu_q_stride = params.cu_q_stride;
  const int head = static_cast<int>(group_id.x);
  const int q_token = static_cast<int>(group_id.y);
  if (head >= num_heads || q_token >= total_q) {
    return;
  }

  int seq_id = -1;
  int q_start = 0;
  int q_end = 0;
  for (int seq = 0; seq < batch; ++seq) {
    const int candidate_q_start = cu_seqlens_q[seq * cu_q_stride];
    const int candidate_q_end = cu_seqlens_q[(seq + 1) * cu_q_stride];
    if (q_token >= candidate_q_start && q_token < candidate_q_end) {
      seq_id = seq;
      q_start = candidate_q_start;
      q_end = candidate_q_end;
      break;
    }
  }

  const int out_base = q_token * out_token_stride + head * out_head_stride;
  if (seq_id < 0) {
    for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
      out[out_base + out_dim * out_dim_stride] = static_cast<T>(0.0f);
    }
    return;
  }

  const int seq_q_len = q_end - q_start;
  const int q_offset = q_token - q_start;
  const int max_prefix_len = max_blocks * block_size;
  const int prefix_len = max(0, min(prefix_lens[seq_id * prefix_lens_stride], max_prefix_len));
  const int visible_suffix_len = causal != 0 ? min(seq_q_len, q_offset + 1) : seq_q_len;
  const int visible_len = prefix_len + visible_suffix_len;
  if (visible_len <= 0) {
    for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
      out[out_base + out_dim * out_dim_stride] = static_cast<T>(0.0f);
    }
    return;
  }

  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = q_token * q_token_stride + head * q_head_stride;

  threadgroup float reductions[SGL_METAL_DECODE_THREADS_PER_GROUP];
  threadgroup float score_scratch[SGL_METAL_PREFILL_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = visible_len <= SGL_METAL_PREFILL_MAX_CACHED_SEQ_LEN;

  float local_max = -INFINITY;
  for (int seq = static_cast<int>(tid); seq < visible_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float score = 0.0f;
    if (seq < prefix_len) {
      const int block_index = seq / block_size;
      const int block_offset = seq - block_index * block_size;
      const int block = block_tables[seq_id * block_tables_batch_stride + block_index * block_tables_block_stride];
      const int k_base =
          block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
      }
    } else {
      const int suffix_offset = seq - prefix_len;
      const int k_base = (q_start + suffix_offset) * kv_token_stride + kv_head * kv_head_stride;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
            static_cast<float>(k[k_base + dim * kv_dim_stride]);
      }
    }
    const float scaled_score = score * scale;
    if (cache_scores) {
      score_scratch[seq] = scaled_score;
    }
    local_max = max(local_max, scaled_score);
  }

  reductions[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = max(reductions[tid], reductions[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_score = reductions[0];

  float local_sum = 0.0f;
  for (int seq = static_cast<int>(tid); seq < visible_len; seq += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float scaled_score;
    if (cache_scores) {
      scaled_score = score_scratch[seq];
    } else {
      float score = 0.0f;
      if (seq < prefix_len) {
        const int block_index = seq / block_size;
        const int block_offset = seq - block_index * block_size;
        const int block = block_tables[seq_id * block_tables_batch_stride + block_index * block_tables_block_stride];
        const int k_base =
            block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
        }
      } else {
        const int suffix_offset = seq - prefix_len;
        const int k_base = (q_start + suffix_offset) * kv_token_stride + kv_head * kv_head_stride;
        for (int dim = 0; dim < head_dim; ++dim) {
          score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
              static_cast<float>(k[k_base + dim * kv_dim_stride]);
        }
      }
      scaled_score = score * scale;
    }
    const float weight = exp(scaled_score - max_score);
    if (cache_scores) {
      score_scratch[seq] = weight;
    }
    local_sum += weight;
  }

  reductions[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = SGL_METAL_DECODE_THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float sum_exp = reductions[0];

  for (int out_dim = static_cast<int>(tid); out_dim < head_dim; out_dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    float acc = 0.0f;
    for (int seq = 0; seq < visible_len; ++seq) {
      float weight;
      if (cache_scores) {
        weight = score_scratch[seq] / sum_exp;
      } else {
        float score = 0.0f;
        if (seq < prefix_len) {
          const int block_index = seq / block_size;
          const int block_offset = seq - block_index * block_size;
          const int block = block_tables[seq_id * block_tables_batch_stride + block_index * block_tables_block_stride];
          const int k_base =
              block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
          for (int dim = 0; dim < head_dim; ++dim) {
            score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
                static_cast<float>(k_cache[k_base + dim * cache_dim_stride]);
          }
        } else {
          const int suffix_offset = seq - prefix_len;
          const int k_base = (q_start + suffix_offset) * kv_token_stride + kv_head * kv_head_stride;
          for (int dim = 0; dim < head_dim; ++dim) {
            score += static_cast<float>(q[q_base + dim * q_dim_stride]) *
                static_cast<float>(k[k_base + dim * kv_dim_stride]);
          }
        }
        weight = exp(score * scale - max_score) / sum_exp;
      }

      if (seq < prefix_len) {
        const int block_index = seq / block_size;
        const int block_offset = seq - block_index * block_size;
        const int block = block_tables[seq_id * block_tables_batch_stride + block_index * block_tables_block_stride];
        const int v_base =
            block * cache_block_stride + block_offset * cache_offset_stride + kv_head * cache_head_stride;
        acc += weight * static_cast<float>(v_cache[v_base + out_dim * cache_dim_stride]);
      } else {
        const int suffix_offset = seq - prefix_len;
        const int v_base = (q_start + suffix_offset) * kv_token_stride + kv_head * kv_head_stride;
        acc += weight * static_cast<float>(v[v_base + out_dim * kv_dim_stride]);
      }
    }
    out[out_base + out_dim * out_dim_stride] = static_cast<T>(acc);
  }
}

template [[host_name("sgl_metal_prefill_attention_paged_float")]] [[kernel]] void
sgl_metal_prefill_attention_paged<float>(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* k_cache [[buffer(3)]],
    const device float* v_cache [[buffer(4)]],
    device float* out [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* prefix_lens [[buffer(7)]],
    const device int* cu_seqlens_q [[buffer(8)]],
    constant PrefillAttentionPagedParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_prefill_attention_paged_half")]] [[kernel]] void
sgl_metal_prefill_attention_paged<half>(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    const device half* k_cache [[buffer(3)]],
    const device half* v_cache [[buffer(4)]],
    device half* out [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* prefix_lens [[buffer(7)]],
    const device int* cu_seqlens_q [[buffer(8)]],
    constant PrefillAttentionPagedParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template <typename T>
[[kernel]] void sgl_metal_paged_kv_scatter(
    const device T* k [[buffer(0)]],
    const device T* v [[buffer(1)]],
    device T* k_cache [[buffer(2)]],
    device T* v_cache [[buffer(3)]],
    const device int* slot_mapping [[buffer(4)]],
    constant int& num_tokens [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& cache_slot_count [[buffer(9)]],
    constant int& src_token_stride [[buffer(10)]],
    constant int& src_head_stride [[buffer(11)]],
    constant int& src_dim_stride [[buffer(12)]],
    constant int& cache_block_stride [[buffer(13)]],
    constant int& cache_offset_stride [[buffer(14)]],
    constant int& cache_head_stride [[buffer(15)]],
    constant int& cache_dim_stride [[buffer(16)]],
    constant int& slot_mapping_stride [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const int head = static_cast<int>(group_id.x);
  const int token = static_cast<int>(group_id.y);
  if (head >= num_kv_heads || token >= num_tokens) {
    return;
  }

  const int slot = slot_mapping[token * slot_mapping_stride];
  if (slot < 0 || slot >= cache_slot_count) {
    return;
  }

  const int block = slot / block_size;
  const int block_offset = slot - block * block_size;
  const int src_base = token * src_token_stride + head * src_head_stride;
  const int cache_base =
      block * cache_block_stride + block_offset * cache_offset_stride + head * cache_head_stride;

  for (int dim = static_cast<int>(tid); dim < head_dim; dim += SGL_METAL_DECODE_THREADS_PER_GROUP) {
    k_cache[cache_base + dim * cache_dim_stride] = k[src_base + dim * src_dim_stride];
    v_cache[cache_base + dim * cache_dim_stride] = v[src_base + dim * src_dim_stride];
  }
}

template [[host_name("sgl_metal_decode_attention_float")]] [[kernel]] void
sgl_metal_decode_attention<float>(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_decode_attention_half")]] [[kernel]] void
sgl_metal_decode_attention<half>(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_decode_attention_small_float")]] [[kernel]] void
sgl_metal_decode_attention_small<float>(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_decode_attention_small_half")]] [[kernel]] void
sgl_metal_decode_attention_small<half>(
    const device half* q [[buffer(0)]],
    const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& q_batch_stride [[buffer(10)]],
    constant int& q_head_stride [[buffer(11)]],
    constant int& q_dim_stride [[buffer(12)]],
    constant int& kv_batch_stride [[buffer(13)]],
    constant int& kv_head_stride [[buffer(14)]],
    constant int& kv_seq_stride [[buffer(15)]],
    constant int& kv_dim_stride [[buffer(16)]],
    constant int& out_batch_stride [[buffer(17)]],
    constant int& out_head_stride [[buffer(18)]],
    constant int& out_dim_stride [[buffer(19)]],
    constant int& query_batch_offset [[buffer(20)]],
    constant int& kv_batch_offset [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);



template [[host_name("sgl_metal_decode_attention_paged_float")]] [[kernel]] void
sgl_metal_decode_attention_paged<float>(
    const device float*,
    const device float*,
    const device float*,
    device float*,
    const device int*,
    const device int*,
    constant float&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    uint,
    uint3);

template [[host_name("sgl_metal_decode_attention_paged_half")]] [[kernel]] void
sgl_metal_decode_attention_paged<half>(
    const device half*,
    const device half*,
    const device half*,
    device half*,
    const device int*,
    const device int*,
    constant float&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    uint,
    uint3);

template [[host_name("sgl_metal_decode_attention_paged_h128_b16_online_float")]] [[kernel]] void
sgl_metal_decode_attention_paged_h128_b16_online<float>(
    const device float*,
    const device float*,
    const device float*,
    device float*,
    const device int*,
    const device int*,
    constant float&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    uint3,
    uint3,
    uint,
    uint);

template [[host_name("sgl_metal_decode_attention_paged_h128_b16_online_half")]] [[kernel]] void
sgl_metal_decode_attention_paged_h128_b16_online<half>(
    const device half*,
    const device half*,
    const device half*,
    device half*,
    const device int*,
    const device int*,
    constant float&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    constant int&,
    uint3,
    uint3,
    uint,
    uint);

template [[host_name("sgl_metal_paged_kv_scatter_float")]] [[kernel]] void
sgl_metal_paged_kv_scatter<float>(
    const device float* k [[buffer(0)]],
    const device float* v [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    const device int* slot_mapping [[buffer(4)]],
    constant int& num_tokens [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& cache_slot_count [[buffer(9)]],
    constant int& src_token_stride [[buffer(10)]],
    constant int& src_head_stride [[buffer(11)]],
    constant int& src_dim_stride [[buffer(12)]],
    constant int& cache_block_stride [[buffer(13)]],
    constant int& cache_offset_stride [[buffer(14)]],
    constant int& cache_head_stride [[buffer(15)]],
    constant int& cache_dim_stride [[buffer(16)]],
    constant int& slot_mapping_stride [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_paged_kv_scatter_half")]] [[kernel]] void
sgl_metal_paged_kv_scatter<half>(
    const device half* k [[buffer(0)]],
    const device half* v [[buffer(1)]],
    device half* k_cache [[buffer(2)]],
    device half* v_cache [[buffer(3)]],
    const device int* slot_mapping [[buffer(4)]],
    constant int& num_tokens [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& cache_slot_count [[buffer(9)]],
    constant int& src_token_stride [[buffer(10)]],
    constant int& src_head_stride [[buffer(11)]],
    constant int& src_dim_stride [[buffer(12)]],
    constant int& cache_block_stride [[buffer(13)]],
    constant int& cache_offset_stride [[buffer(14)]],
    constant int& cache_head_stride [[buffer(15)]],
    constant int& cache_dim_stride [[buffer(16)]],
    constant int& slot_mapping_stride [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);

template [[host_name("sgl_metal_paged_kv_scatter_bfloat")]] [[kernel]] void
sgl_metal_paged_kv_scatter<bfloat>(
    const device bfloat* k [[buffer(0)]],
    const device bfloat* v [[buffer(1)]],
    device bfloat* k_cache [[buffer(2)]],
    device bfloat* v_cache [[buffer(3)]],
    const device int* slot_mapping [[buffer(4)]],
    constant int& num_tokens [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& cache_slot_count [[buffer(9)]],
    constant int& src_token_stride [[buffer(10)]],
    constant int& src_head_stride [[buffer(11)]],
    constant int& src_dim_stride [[buffer(12)]],
    constant int& cache_block_stride [[buffer(13)]],
    constant int& cache_offset_stride [[buffer(14)]],
    constant int& cache_head_stride [[buffer(15)]],
    constant int& cache_dim_stride [[buffer(16)]],
    constant int& slot_mapping_stride [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]);
