#include <metal_stdlib>

using namespace metal;

#define SGL_METAL_DECODE_THREADS_PER_GROUP 256
#define SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN 1024
#define SGL_METAL_DECODE_SMALL_CACHED_SEQ_LEN 512
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
  threadgroup float score_scratch[SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = visible_len <= SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN;

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
  threadgroup float score_scratch[SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN];
  const bool cache_scores = visible_len <= SGL_METAL_DECODE_MAX_CACHED_SEQ_LEN;

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
