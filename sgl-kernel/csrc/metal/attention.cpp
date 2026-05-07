#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

namespace nb = nanobind;
using namespace mlx::core;

namespace {

constexpr const char* kMetalLibraryName = "sgl_metal_kernels";
constexpr uint32_t kDecodeThreadsPerGroup = 256;
constexpr int32_t kSmallDecodeCachedSeqLen = 512;
constexpr size_t kMaxDecodeHeadDim = 256;

struct PrefillAttentionPagedParams {
  float scale;
  int32_t total_q;
  int32_t batch;
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t max_blocks;
  int32_t block_size;
  int32_t head_dim;
  int32_t causal;
  int32_t q_token_stride;
  int32_t q_head_stride;
  int32_t q_dim_stride;
  int32_t kv_token_stride;
  int32_t kv_head_stride;
  int32_t kv_dim_stride;
  int32_t cache_block_stride;
  int32_t cache_offset_stride;
  int32_t cache_head_stride;
  int32_t cache_dim_stride;
  int32_t out_token_stride;
  int32_t out_head_stride;
  int32_t out_dim_stride;
  int32_t block_tables_batch_stride;
  int32_t block_tables_block_stride;
  int32_t prefix_lens_stride;
  int32_t cu_q_stride;
};

std::string registered_library_path;

array& checked_array(nb::handle h, const char* name) {
  auto mlx_core = nb::module_::import_("mlx.core");
  auto array_type = mlx_core.attr("array");
  auto is_array = PyObject_IsInstance(h.ptr(), array_type.ptr());
  if (is_array < 0) {
    PyErr_Clear();
  }
  if (is_array != 1) {
    throw nb::type_error((std::string(name) + " must be an MLX array").c_str());
  }
  auto* ptr = nb::inst_ptr<array>(h);
  if (ptr == nullptr) {
    throw nb::type_error((std::string(name) + " must be an MLX array").c_str());
  }
  return *ptr;
}

std::string dtype_suffix(Dtype dtype, const char* op_name) {
  switch (dtype) {
    case float16:
      return "half";
    case float32:
      return "float";
    default:
      throw std::runtime_error(std::string(op_name) + " supports only float16 and float32 arrays");
  }
}

void require_registered_library(const char* op_name) {
  if (registered_library_path.empty()) {
    throw std::runtime_error(std::string("register_library must be called before ") + op_name);
  }
}

void validate_dense_row_contiguous(const array& x, const char* name) {
  if (!x.flags().row_contiguous || x.data_size() < x.size()) {
    throw std::runtime_error(std::string(name) + " must be row-contiguous");
  }
}

int32_t checked_dim(const array& x, int dim, const char* name) {
  auto value = x.shape(dim);
  if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(std::string(name) + " dimension is too large for Metal attention");
  }
  return static_cast<int32_t>(value);
}

int32_t checked_stride(const array& x, int dim, const char* name) {
  auto value = static_cast<int64_t>(x.strides(dim));
  if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(std::string(name) + " stride is too large for Metal attention");
  }
  return static_cast<int32_t>(value);
}

void validate_batch_window(const array& x, int32_t batch_offset, int32_t dispatch_batch, const char* name) {
  if (batch_offset < 0 || dispatch_batch < 0) {
    throw std::runtime_error(std::string(name) + " batch window must be non-negative");
  }
  auto end = static_cast<size_t>(batch_offset) + static_cast<size_t>(dispatch_batch);
  if (end > x.shape(0)) {
    throw std::runtime_error(std::string(name) + " batch window is out of bounds");
  }
}

void validate_decode_inputs(
    const array& out,
    const array& q,
    const array& k,
    const array& v,
    float scale,
    int32_t query_batch_offset,
    int32_t kv_batch_offset,
    int32_t dispatch_batch) {
  if (q.ndim() != 4) {
    throw std::runtime_error("decode_attention query must have shape (B, H, 1, D)");
  }
  if (k.ndim() != 4 || v.ndim() != 4) {
    throw std::runtime_error("decode_attention K/V caches must have shape (B, KVH, S, D)");
  }
  if (out.ndim() != 4) {
    throw std::runtime_error("decode_attention output must have shape (B, H, 1, D)");
  }
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != out.dtype()) {
    throw std::runtime_error("decode_attention query, K/V, and output dtypes must match");
  }
  dtype_suffix(q.dtype(), "decode_attention");
  if (q.shape(2) != 1 || out.shape(2) != 1) {
    throw std::runtime_error("decode_attention supports decode-only query length 1");
  }
  if (q.shape(1) != out.shape(1)) {
    throw std::runtime_error("decode_attention output head count must match query head count");
  }
  if (q.shape(3) != k.shape(3) || q.shape(3) != v.shape(3) || q.shape(3) != out.shape(3)) {
    throw std::runtime_error("decode_attention head dimensions must match");
  }
  if (k.shape(1) != v.shape(1)) {
    throw std::runtime_error("decode_attention K/V head counts must match");
  }
  if (k.shape(2) != v.shape(2)) {
    throw std::runtime_error("decode_attention K/V sequence lengths must match");
  }
  if (k.shape(0) != v.shape(0)) {
    throw std::runtime_error("decode_attention K/V batch dimensions must match");
  }
  if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0) {
    throw std::runtime_error("decode_attention query heads must be divisible by KV heads");
  }
  if (k.shape(2) == 0) {
    throw std::runtime_error("decode_attention K/V sequence length must be positive");
  }
  if (q.shape(3) == 0 || q.shape(3) > kMaxDecodeHeadDim) {
    throw std::runtime_error("decode_attention head dimension must be in the range [1, 256]");
  }
  if (!(scale > 0.0f)) {
    throw std::runtime_error("decode_attention scale must be positive");
  }
  validate_batch_window(q, query_batch_offset, dispatch_batch, "decode_attention query");
  validate_batch_window(out, query_batch_offset, dispatch_batch, "decode_attention output");
  validate_batch_window(k, kv_batch_offset, dispatch_batch, "decode_attention key cache");
  validate_batch_window(v, kv_batch_offset, dispatch_batch, "decode_attention value cache");
  validate_dense_row_contiguous(q, "decode_attention query");
  validate_dense_row_contiguous(k, "decode_attention key cache");
  validate_dense_row_contiguous(v, "decode_attention value cache");
  validate_dense_row_contiguous(out, "decode_attention output");
}

void dispatch_decode_attention(
    array& out,
    const array& q,
    const array& k,
    const array& v,
    float scale,
    int32_t query_batch_offset,
    int32_t kv_batch_offset,
    int32_t dispatch_batch) {
  validate_decode_inputs(out, q, k, v, scale, query_batch_offset, kv_batch_offset, dispatch_batch);

  if (dispatch_batch == 0 || q.size() == 0) {
    return;
  }
  require_registered_library("decode_attention");

  auto num_heads = checked_dim(q, 1, "decode_attention query");
  auto num_kv_heads = checked_dim(k, 1, "decode_attention key cache");
  auto seq_len = checked_dim(k, 2, "decode_attention key cache");
  auto head_dim = checked_dim(q, 3, "decode_attention query");
  auto q_batch_stride = checked_stride(q, 0, "decode_attention query");
  auto q_head_stride = checked_stride(q, 1, "decode_attention query");
  auto q_dim_stride = checked_stride(q, 3, "decode_attention query");
  auto kv_batch_stride = checked_stride(k, 0, "decode_attention key cache");
  auto kv_head_stride = checked_stride(k, 1, "decode_attention key cache");
  auto kv_seq_stride = checked_stride(k, 2, "decode_attention key cache");
  auto kv_dim_stride = checked_stride(k, 3, "decode_attention key cache");
  auto out_batch_stride = checked_stride(out, 0, "decode_attention output");
  auto out_head_stride = checked_stride(out, 1, "decode_attention output");
  auto out_dim_stride = checked_stride(out, 3, "decode_attention output");

  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(Device::gpu);
  auto* library = device.get_library(kMetalLibraryName);
  auto use_small_kernel = seq_len <= kSmallDecodeCachedSeqLen;
  auto kernel_prefix = use_small_kernel ? "sgl_metal_decode_attention_small_" : "sgl_metal_decode_attention_";
  auto kernel_name = std::string(kernel_prefix) + dtype_suffix(q.dtype(), "decode_attention");
  auto* kernel = device.get_kernel(kernel_name, library, kernel_name);

  auto& encoder = device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k, 1);
  encoder.set_input_array(v, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(scale, 4);
  encoder.set_bytes(dispatch_batch, 5);
  encoder.set_bytes(num_heads, 6);
  encoder.set_bytes(num_kv_heads, 7);
  encoder.set_bytes(seq_len, 8);
  encoder.set_bytes(head_dim, 9);
  encoder.set_bytes(q_batch_stride, 10);
  encoder.set_bytes(q_head_stride, 11);
  encoder.set_bytes(q_dim_stride, 12);
  encoder.set_bytes(kv_batch_stride, 13);
  encoder.set_bytes(kv_head_stride, 14);
  encoder.set_bytes(kv_seq_stride, 15);
  encoder.set_bytes(kv_dim_stride, 16);
  encoder.set_bytes(out_batch_stride, 17);
  encoder.set_bytes(out_head_stride, 18);
  encoder.set_bytes(out_dim_stride, 19);
  encoder.set_bytes(query_batch_offset, 20);
  encoder.set_bytes(kv_batch_offset, 21);

  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_heads, dispatch_batch, 1), MTL::Size::Make(kDecodeThreadsPerGroup, 1, 1));

  device.add_temporary(q, stream.index);
  device.add_temporary(k, stream.index);
  device.add_temporary(v, stream.index);
  device.add_temporary(out, stream.index);
}

void validate_decode_paged_inputs(
    const array& out,
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    float scale) {
  if (q.ndim() != 4 || q.shape(2) != 1) {
    throw std::runtime_error("decode_attention_paged query must have shape (B, H, 1, D)");
  }
  if (k_cache.ndim() != 4 || v_cache.ndim() != 4) {
    throw std::runtime_error("decode_attention_paged K/V caches must have shape (num_blocks, block_size, KVH, D)");
  }
  if (out.ndim() != 4 || out.shape() != q.shape()) {
    throw std::runtime_error("decode_attention_paged output shape must match query shape");
  }
  if (block_tables.ndim() != 2 || block_tables.shape(0) != q.shape(0)) {
    throw std::runtime_error("decode_attention_paged block_tables must have shape (B, max_blocks)");
  }
  if (context_lens.ndim() != 1 || context_lens.shape(0) != q.shape(0)) {
    throw std::runtime_error("decode_attention_paged context_lens must have shape (B,)");
  }
  if (q.dtype() != k_cache.dtype() || q.dtype() != v_cache.dtype() || q.dtype() != out.dtype()) {
    throw std::runtime_error("decode_attention_paged query, K/V caches, and output dtypes must match");
  }
  dtype_suffix(q.dtype(), "decode_attention_paged");
  if (block_tables.dtype() != int32 || context_lens.dtype() != int32) {
    throw std::runtime_error("decode_attention_paged block_tables and context_lens must be int32 arrays");
  }
  if (k_cache.shape() != v_cache.shape()) {
    throw std::runtime_error("decode_attention_paged K/V cache shapes must match");
  }
  if (q.shape(3) != k_cache.shape(3)) {
    throw std::runtime_error("decode_attention_paged head dimensions must match");
  }
  if (k_cache.shape(2) == 0 || q.shape(1) % k_cache.shape(2) != 0) {
    throw std::runtime_error("decode_attention_paged query heads must be divisible by KV heads");
  }
  if (block_tables.shape(1) == 0) {
    throw std::runtime_error("decode_attention_paged max block count must be positive");
  }
  if (k_cache.shape(1) == 0) {
    throw std::runtime_error("decode_attention_paged block size must be positive");
  }
  if (q.shape(0) > 0 && k_cache.shape(0) == 0) {
    throw std::runtime_error("decode_attention_paged KV cache must contain at least one block");
  }
  if (q.shape(3) == 0 || q.shape(3) > kMaxDecodeHeadDim) {
    throw std::runtime_error("decode_attention_paged head dimension must be in the range [1, 256]");
  }
  auto max_seq_len = block_tables.shape(1) * k_cache.shape(1);
  if (max_seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("decode_attention_paged maximum sequence length is too large for Metal attention");
  }
  if (!(scale > 0.0f)) {
    throw std::runtime_error("decode_attention_paged scale must be positive");
  }
  validate_dense_row_contiguous(q, "decode_attention_paged query");
  validate_dense_row_contiguous(k_cache, "decode_attention_paged key cache");
  validate_dense_row_contiguous(v_cache, "decode_attention_paged value cache");
  validate_dense_row_contiguous(out, "decode_attention_paged output");
  validate_dense_row_contiguous(block_tables, "decode_attention_paged block_tables");
  validate_dense_row_contiguous(context_lens, "decode_attention_paged context_lens");
}

void dispatch_decode_attention_paged(
    array& out,
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    float scale) {
  validate_decode_paged_inputs(out, q, k_cache, v_cache, block_tables, context_lens, scale);

  if (q.size() == 0) {
    return;
  }
  require_registered_library("decode_attention_paged");

  auto batch = checked_dim(q, 0, "decode_attention_paged query");
  auto num_heads = checked_dim(q, 1, "decode_attention_paged query");
  auto num_kv_heads = checked_dim(k_cache, 2, "decode_attention_paged key cache");
  auto max_blocks = checked_dim(block_tables, 1, "decode_attention_paged block_tables");
  auto block_size = checked_dim(k_cache, 1, "decode_attention_paged key cache");
  auto head_dim = checked_dim(q, 3, "decode_attention_paged query");
  auto q_batch_stride = checked_stride(q, 0, "decode_attention_paged query");
  auto q_head_stride = checked_stride(q, 1, "decode_attention_paged query");
  auto q_dim_stride = checked_stride(q, 3, "decode_attention_paged query");
  auto cache_block_stride = checked_stride(k_cache, 0, "decode_attention_paged key cache");
  auto cache_offset_stride = checked_stride(k_cache, 1, "decode_attention_paged key cache");
  auto cache_head_stride = checked_stride(k_cache, 2, "decode_attention_paged key cache");
  auto cache_dim_stride = checked_stride(k_cache, 3, "decode_attention_paged key cache");
  auto out_batch_stride = checked_stride(out, 0, "decode_attention_paged output");
  auto out_head_stride = checked_stride(out, 1, "decode_attention_paged output");
  auto out_dim_stride = checked_stride(out, 3, "decode_attention_paged output");
  auto block_tables_batch_stride = checked_stride(block_tables, 0, "decode_attention_paged block_tables");
  auto block_tables_block_stride = checked_stride(block_tables, 1, "decode_attention_paged block_tables");
  auto context_lens_stride = checked_stride(context_lens, 0, "decode_attention_paged context_lens");

  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(Device::gpu);
  auto* library = device.get_library(kMetalLibraryName);
  auto kernel_name = "sgl_metal_decode_attention_paged_" + dtype_suffix(q.dtype(), "decode_attention_paged");
  auto* kernel = device.get_kernel(kernel_name, library, kernel_name);

  auto& encoder = device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k_cache, 1);
  encoder.set_input_array(v_cache, 2);
  encoder.set_output_array(out, 3);
  encoder.set_input_array(block_tables, 4);
  encoder.set_input_array(context_lens, 5);
  encoder.set_bytes(scale, 6);
  encoder.set_bytes(batch, 7);
  encoder.set_bytes(num_heads, 8);
  encoder.set_bytes(num_kv_heads, 9);
  encoder.set_bytes(max_blocks, 10);
  encoder.set_bytes(block_size, 11);
  encoder.set_bytes(head_dim, 12);
  encoder.set_bytes(q_batch_stride, 13);
  encoder.set_bytes(q_head_stride, 14);
  encoder.set_bytes(q_dim_stride, 15);
  encoder.set_bytes(cache_block_stride, 16);
  encoder.set_bytes(cache_offset_stride, 17);
  encoder.set_bytes(cache_head_stride, 18);
  encoder.set_bytes(cache_dim_stride, 19);
  encoder.set_bytes(out_batch_stride, 20);
  encoder.set_bytes(out_head_stride, 21);
  encoder.set_bytes(out_dim_stride, 22);
  encoder.set_bytes(block_tables_batch_stride, 23);
  encoder.set_bytes(block_tables_block_stride, 24);
  encoder.set_bytes(context_lens_stride, 25);

  encoder.dispatch_threadgroups(MTL::Size::Make(num_heads, batch, 1), MTL::Size::Make(kDecodeThreadsPerGroup, 1, 1));

  device.add_temporary(q, stream.index);
  device.add_temporary(k_cache, stream.index);
  device.add_temporary(v_cache, stream.index);
  device.add_temporary(out, stream.index);
  device.add_temporary(block_tables, stream.index);
  device.add_temporary(context_lens, stream.index);
}

void validate_flash_attn_varlen_inputs(
    const array& out,
    const array& q,
    const array& k,
    const array& v,
    const array& cu_seqlens_q,
    const array& cu_seqlens_k,
    float scale,
    bool causal) {
  if (q.ndim() != 3) {
    throw std::runtime_error("flash_attn_varlen query must have shape (total_q, H, D)");
  }
  if (k.ndim() != 3 || v.ndim() != 3) {
    throw std::runtime_error("flash_attn_varlen K/V must have shape (total_k, KVH, D)");
  }
  if (out.ndim() != 3 || out.shape() != q.shape()) {
    throw std::runtime_error("flash_attn_varlen output shape must match query shape");
  }
  if (cu_seqlens_q.ndim() != 1 || cu_seqlens_k.ndim() != 1) {
    throw std::runtime_error("flash_attn_varlen cu_seqlens_q and cu_seqlens_k must be 1-D arrays");
  }
  if (cu_seqlens_q.shape() != cu_seqlens_k.shape()) {
    throw std::runtime_error("flash_attn_varlen cu_seqlens_q and cu_seqlens_k must have the same shape");
  }
  if (cu_seqlens_q.shape(0) < 2) {
    throw std::runtime_error("flash_attn_varlen cu_seqlens arrays must contain at least two entries");
  }
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != out.dtype()) {
    throw std::runtime_error("flash_attn_varlen query, K/V, and output dtypes must match");
  }
  dtype_suffix(q.dtype(), "flash_attn_varlen");
  if (cu_seqlens_q.dtype() != int32 || cu_seqlens_k.dtype() != int32) {
    throw std::runtime_error("flash_attn_varlen cu_seqlens_q and cu_seqlens_k must be int32 arrays");
  }
  if (k.shape() != v.shape()) {
    throw std::runtime_error("flash_attn_varlen K/V shapes must match");
  }
  if (q.shape(2) != k.shape(2)) {
    throw std::runtime_error("flash_attn_varlen head dimensions must match");
  }
  if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0) {
    throw std::runtime_error("flash_attn_varlen query heads must be divisible by KV heads");
  }
  if (q.shape(2) == 0 || q.shape(2) > kMaxDecodeHeadDim) {
    throw std::runtime_error("flash_attn_varlen head dimension must be in the range [1, 256]");
  }
  if (!(scale > 0.0f)) {
    throw std::runtime_error("flash_attn_varlen scale must be positive");
  }
  (void)causal;
  validate_dense_row_contiguous(q, "flash_attn_varlen query");
  validate_dense_row_contiguous(k, "flash_attn_varlen key");
  validate_dense_row_contiguous(v, "flash_attn_varlen value");
  validate_dense_row_contiguous(out, "flash_attn_varlen output");
  validate_dense_row_contiguous(cu_seqlens_q, "flash_attn_varlen cu_seqlens_q");
  validate_dense_row_contiguous(cu_seqlens_k, "flash_attn_varlen cu_seqlens_k");
}

void dispatch_flash_attn_varlen(
    array& out,
    const array& q,
    const array& k,
    const array& v,
    const array& cu_seqlens_q,
    const array& cu_seqlens_k,
    float scale,
    bool causal) {
  validate_flash_attn_varlen_inputs(out, q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal);

  if (q.size() == 0) {
    return;
  }
  require_registered_library("flash_attn_varlen");

  auto total_q = checked_dim(q, 0, "flash_attn_varlen query");
  auto num_seqs = checked_dim(cu_seqlens_q, 0, "flash_attn_varlen cu_seqlens_q") - 1;
  auto num_heads = checked_dim(q, 1, "flash_attn_varlen query");
  auto num_kv_heads = checked_dim(k, 1, "flash_attn_varlen key");
  auto head_dim = checked_dim(q, 2, "flash_attn_varlen query");
  auto causal_int = causal ? 1 : 0;
  auto q_token_stride = checked_stride(q, 0, "flash_attn_varlen query");
  auto q_head_stride = checked_stride(q, 1, "flash_attn_varlen query");
  auto q_dim_stride = checked_stride(q, 2, "flash_attn_varlen query");
  auto kv_token_stride = checked_stride(k, 0, "flash_attn_varlen key");
  auto kv_head_stride = checked_stride(k, 1, "flash_attn_varlen key");
  auto kv_dim_stride = checked_stride(k, 2, "flash_attn_varlen key");
  auto out_token_stride = checked_stride(out, 0, "flash_attn_varlen output");
  auto out_head_stride = checked_stride(out, 1, "flash_attn_varlen output");
  auto out_dim_stride = checked_stride(out, 2, "flash_attn_varlen output");
  auto cu_q_stride = checked_stride(cu_seqlens_q, 0, "flash_attn_varlen cu_seqlens_q");
  auto cu_k_stride = checked_stride(cu_seqlens_k, 0, "flash_attn_varlen cu_seqlens_k");

  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(Device::gpu);
  auto* library = device.get_library(kMetalLibraryName);
  auto kernel_name = "sgl_metal_flash_attn_varlen_" + dtype_suffix(q.dtype(), "flash_attn_varlen");
  auto* kernel = device.get_kernel(kernel_name, library, kernel_name);

  auto& encoder = device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k, 1);
  encoder.set_input_array(v, 2);
  encoder.set_output_array(out, 3);
  encoder.set_input_array(cu_seqlens_q, 4);
  encoder.set_input_array(cu_seqlens_k, 5);
  encoder.set_bytes(scale, 6);
  encoder.set_bytes(total_q, 7);
  encoder.set_bytes(num_seqs, 8);
  encoder.set_bytes(num_heads, 9);
  encoder.set_bytes(num_kv_heads, 10);
  encoder.set_bytes(head_dim, 11);
  encoder.set_bytes(causal_int, 12);
  encoder.set_bytes(q_token_stride, 13);
  encoder.set_bytes(q_head_stride, 14);
  encoder.set_bytes(q_dim_stride, 15);
  encoder.set_bytes(kv_token_stride, 16);
  encoder.set_bytes(kv_head_stride, 17);
  encoder.set_bytes(kv_dim_stride, 18);
  encoder.set_bytes(out_token_stride, 19);
  encoder.set_bytes(out_head_stride, 20);
  encoder.set_bytes(out_dim_stride, 21);
  encoder.set_bytes(cu_q_stride, 22);
  encoder.set_bytes(cu_k_stride, 23);

  encoder.dispatch_threadgroups(MTL::Size::Make(num_heads, total_q, 1), MTL::Size::Make(kDecodeThreadsPerGroup, 1, 1));

  device.add_temporary(q, stream.index);
  device.add_temporary(k, stream.index);
  device.add_temporary(v, stream.index);
  device.add_temporary(out, stream.index);
  device.add_temporary(cu_seqlens_q, stream.index);
  device.add_temporary(cu_seqlens_k, stream.index);
}

void validate_prefill_attention_paged_inputs(
    const array& out,
    const array& q,
    const array& k,
    const array& v,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& prefix_lens,
    const array& cu_seqlens_q,
    float scale,
    bool causal) {
  if (q.ndim() != 3) {
    throw std::runtime_error("prefill_attention_paged query must have shape (total_q, H, D)");
  }
  if (k.ndim() != 3 || v.ndim() != 3) {
    throw std::runtime_error("prefill_attention_paged K/V must have shape (total_q, KVH, D)");
  }
  if (k_cache.ndim() != 4 || v_cache.ndim() != 4) {
    throw std::runtime_error("prefill_attention_paged K/V caches must have shape (num_blocks, block_size, KVH, D)");
  }
  if (out.ndim() != 3 || out.shape() != q.shape()) {
    throw std::runtime_error("prefill_attention_paged output shape must match query shape");
  }
  if (block_tables.ndim() != 2) {
    throw std::runtime_error("prefill_attention_paged block_tables must have shape (B, max_blocks)");
  }
  if (prefix_lens.ndim() != 1 || prefix_lens.shape(0) != block_tables.shape(0)) {
    throw std::runtime_error("prefill_attention_paged prefix_lens must have shape (B,)");
  }
  if (cu_seqlens_q.ndim() != 1 || cu_seqlens_q.shape(0) != block_tables.shape(0) + 1) {
    throw std::runtime_error("prefill_attention_paged cu_seqlens_q must have shape (B + 1,)");
  }
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != k_cache.dtype() ||
      q.dtype() != v_cache.dtype() || q.dtype() != out.dtype()) {
    throw std::runtime_error("prefill_attention_paged query, K/V, caches, and output dtypes must match");
  }
  dtype_suffix(q.dtype(), "prefill_attention_paged");
  if (block_tables.dtype() != int32 || prefix_lens.dtype() != int32 || cu_seqlens_q.dtype() != int32) {
    throw std::runtime_error(
        "prefill_attention_paged block_tables, prefix_lens, and cu_seqlens_q must be int32 arrays");
  }
  if (k.shape() != v.shape()) {
    throw std::runtime_error("prefill_attention_paged K/V shapes must match");
  }
  if (k_cache.shape() != v_cache.shape()) {
    throw std::runtime_error("prefill_attention_paged K/V cache shapes must match");
  }
  if (k.shape(0) != q.shape(0)) {
    throw std::runtime_error("prefill_attention_paged K/V token count must match query token count");
  }
  if (q.shape(2) != k.shape(2) || q.shape(2) != k_cache.shape(3)) {
    throw std::runtime_error("prefill_attention_paged head dimensions must match");
  }
  if (k.shape(1) != k_cache.shape(2)) {
    throw std::runtime_error("prefill_attention_paged KV head counts must match");
  }
  if (k.shape(1) == 0 || q.shape(1) % k.shape(1) != 0) {
    throw std::runtime_error("prefill_attention_paged query heads must be divisible by KV heads");
  }
  if (q.shape(2) == 0 || q.shape(2) > kMaxDecodeHeadDim) {
    throw std::runtime_error("prefill_attention_paged head dimension must be in the range [1, 256]");
  }
  if (k_cache.shape(1) == 0) {
    throw std::runtime_error("prefill_attention_paged block size must be positive");
  }
  if (block_tables.shape(1) == 0) {
    throw std::runtime_error("prefill_attention_paged max block count must be positive");
  }
  if (q.shape(0) > 0 && k_cache.shape(0) == 0) {
    throw std::runtime_error("prefill_attention_paged KV cache must contain at least one block");
  }
  if (!(scale > 0.0f)) {
    throw std::runtime_error("prefill_attention_paged scale must be positive");
  }
  (void)causal;
  validate_dense_row_contiguous(q, "prefill_attention_paged query");
  validate_dense_row_contiguous(k, "prefill_attention_paged key");
  validate_dense_row_contiguous(v, "prefill_attention_paged value");
  validate_dense_row_contiguous(k_cache, "prefill_attention_paged key cache");
  validate_dense_row_contiguous(v_cache, "prefill_attention_paged value cache");
  validate_dense_row_contiguous(out, "prefill_attention_paged output");
  validate_dense_row_contiguous(block_tables, "prefill_attention_paged block_tables");
  validate_dense_row_contiguous(prefix_lens, "prefill_attention_paged prefix_lens");
  validate_dense_row_contiguous(cu_seqlens_q, "prefill_attention_paged cu_seqlens_q");

  auto max_seq_len = block_tables.shape(1) * k_cache.shape(1);
  if (max_seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("prefill_attention_paged maximum sequence length is too large for Metal attention");
  }
  if (q.shape(0) > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("prefill_attention_paged query token count is too large for Metal attention");
  }

  auto cu_seqlens_q_host = cu_seqlens_q;
  auto prefix_lens_host = prefix_lens;
  auto block_tables_host = block_tables;
  cu_seqlens_q_host.eval();
  prefix_lens_host.eval();
  block_tables_host.eval();

  auto batch = block_tables.shape(0);
  auto max_blocks = block_tables.shape(1);
  auto num_cache_blocks = k_cache.shape(0);
  auto block_size = k_cache.shape(1);
  const auto* cu = cu_seqlens_q_host.data<int32_t>();
  const auto* prefixes = prefix_lens_host.data<int32_t>();
  const auto* tables = block_tables_host.data<int32_t>();
  auto cu_stride = cu_seqlens_q_host.strides()[0];
  auto prefix_stride = prefix_lens_host.strides()[0];
  auto table_batch_stride = block_tables_host.strides()[0];
  auto table_block_stride = block_tables_host.strides()[1];

  if (cu[0] != 0) {
    throw std::runtime_error("prefill_attention_paged cu_seqlens_q must start at 0");
  }
  if (cu[batch * cu_stride] != static_cast<int32_t>(q.shape(0))) {
    throw std::runtime_error("prefill_attention_paged cu_seqlens_q must end at total_q");
  }

  for (size_t b = 0; b < batch; ++b) {
    auto q_start = cu[b * cu_stride];
    auto q_end = cu[(b + 1) * cu_stride];
    if (q_start < 0 || q_end < q_start) {
      throw std::runtime_error("prefill_attention_paged cu_seqlens_q must be nondecreasing");
    }

    auto prefix_len = prefixes[b * prefix_stride];
    if (prefix_len < 0 || static_cast<size_t>(prefix_len) > max_seq_len) {
      throw std::runtime_error("prefill_attention_paged prefix_lens entries must be in range");
    }

    auto used_blocks = (static_cast<size_t>(prefix_len) + block_size - 1) / block_size;
    if (used_blocks > max_blocks) {
      throw std::runtime_error("prefill_attention_paged prefix_lens entries exceed block_tables capacity");
    }
    for (size_t block_idx = 0; block_idx < used_blocks; ++block_idx) {
      auto block_id = tables[b * table_batch_stride + block_idx * table_block_stride];
      if (block_id < 0 || static_cast<size_t>(block_id) >= num_cache_blocks) {
        throw std::runtime_error("prefill_attention_paged block_tables entries must reference valid cache blocks");
      }
    }
  }
}

void dispatch_prefill_attention_paged(
    array& out,
    const array& q,
    const array& k,
    const array& v,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& prefix_lens,
    const array& cu_seqlens_q,
    float scale,
    bool causal) {
  validate_prefill_attention_paged_inputs(
      out, q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_seqlens_q, scale, causal);

  if (q.size() == 0) {
    return;
  }
  require_registered_library("prefill_attention_paged");

  auto total_q = checked_dim(q, 0, "prefill_attention_paged query");
  auto batch = checked_dim(block_tables, 0, "prefill_attention_paged block_tables");
  auto num_heads = checked_dim(q, 1, "prefill_attention_paged query");
  auto num_kv_heads = checked_dim(k, 1, "prefill_attention_paged key");
  auto max_blocks = checked_dim(block_tables, 1, "prefill_attention_paged block_tables");
  auto block_size = checked_dim(k_cache, 1, "prefill_attention_paged key cache");
  auto head_dim = checked_dim(q, 2, "prefill_attention_paged query");
  auto causal_int = causal ? 1 : 0;
  auto q_token_stride = checked_stride(q, 0, "prefill_attention_paged query");
  auto q_head_stride = checked_stride(q, 1, "prefill_attention_paged query");
  auto q_dim_stride = checked_stride(q, 2, "prefill_attention_paged query");
  auto kv_token_stride = checked_stride(k, 0, "prefill_attention_paged key");
  auto kv_head_stride = checked_stride(k, 1, "prefill_attention_paged key");
  auto kv_dim_stride = checked_stride(k, 2, "prefill_attention_paged key");
  auto cache_block_stride = checked_stride(k_cache, 0, "prefill_attention_paged key cache");
  auto cache_offset_stride = checked_stride(k_cache, 1, "prefill_attention_paged key cache");
  auto cache_head_stride = checked_stride(k_cache, 2, "prefill_attention_paged key cache");
  auto cache_dim_stride = checked_stride(k_cache, 3, "prefill_attention_paged key cache");
  auto out_token_stride = checked_stride(out, 0, "prefill_attention_paged output");
  auto out_head_stride = checked_stride(out, 1, "prefill_attention_paged output");
  auto out_dim_stride = checked_stride(out, 2, "prefill_attention_paged output");
  auto block_tables_batch_stride = checked_stride(block_tables, 0, "prefill_attention_paged block_tables");
  auto block_tables_block_stride = checked_stride(block_tables, 1, "prefill_attention_paged block_tables");
  auto prefix_lens_stride = checked_stride(prefix_lens, 0, "prefill_attention_paged prefix_lens");
  auto cu_q_stride = checked_stride(cu_seqlens_q, 0, "prefill_attention_paged cu_seqlens_q");

  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(Device::gpu);
  auto* library = device.get_library(kMetalLibraryName);
  auto kernel_name = "sgl_metal_prefill_attention_paged_" + dtype_suffix(q.dtype(), "prefill_attention_paged");
  auto* kernel = device.get_kernel(kernel_name, library, kernel_name);

  PrefillAttentionPagedParams params{
      scale,
      total_q,
      batch,
      num_heads,
      num_kv_heads,
      max_blocks,
      block_size,
      head_dim,
      causal_int,
      q_token_stride,
      q_head_stride,
      q_dim_stride,
      kv_token_stride,
      kv_head_stride,
      kv_dim_stride,
      cache_block_stride,
      cache_offset_stride,
      cache_head_stride,
      cache_dim_stride,
      out_token_stride,
      out_head_stride,
      out_dim_stride,
      block_tables_batch_stride,
      block_tables_block_stride,
      prefix_lens_stride,
      cu_q_stride,
  };

  auto& encoder = device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k, 1);
  encoder.set_input_array(v, 2);
  encoder.set_input_array(k_cache, 3);
  encoder.set_input_array(v_cache, 4);
  encoder.set_output_array(out, 5);
  encoder.set_input_array(block_tables, 6);
  encoder.set_input_array(prefix_lens, 7);
  encoder.set_input_array(cu_seqlens_q, 8);
  encoder.set_bytes(params, 9);

  encoder.dispatch_threadgroups(MTL::Size::Make(num_heads, total_q, 1), MTL::Size::Make(kDecodeThreadsPerGroup, 1, 1));

  device.add_temporary(q, stream.index);
  device.add_temporary(k, stream.index);
  device.add_temporary(v, stream.index);
  device.add_temporary(k_cache, stream.index);
  device.add_temporary(v_cache, stream.index);
  device.add_temporary(out, stream.index);
  device.add_temporary(block_tables, stream.index);
  device.add_temporary(prefix_lens, stream.index);
  device.add_temporary(cu_seqlens_q, stream.index);
}

void validate_paged_kv_scatter_inputs(
    const array& k, const array& v, const array& k_cache, const array& v_cache, const array& slot_mapping) {
  if (k.ndim() != 3 || v.ndim() != 3) {
    throw std::runtime_error("paged_kv_scatter K/V tensors must have shape (num_tokens, KVH, D)");
  }
  if (k_cache.ndim() != 4 || v_cache.ndim() != 4) {
    throw std::runtime_error("paged_kv_scatter K/V caches must have shape (num_blocks, block_size, KVH, D)");
  }
  if (k.shape() != v.shape()) {
    throw std::runtime_error("paged_kv_scatter K/V tensor shapes must match");
  }
  if (k_cache.shape() != v_cache.shape()) {
    throw std::runtime_error("paged_kv_scatter K/V cache shapes must match");
  }
  if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != k.shape(0)) {
    throw std::runtime_error("paged_kv_scatter slot_mapping must have shape (num_tokens,)");
  }
  if (k.dtype() != v.dtype() || k.dtype() != k_cache.dtype() || k.dtype() != v_cache.dtype()) {
    throw std::runtime_error("paged_kv_scatter K/V tensors and caches must have matching dtypes");
  }
  dtype_suffix(k.dtype(), "paged_kv_scatter");
  if (slot_mapping.dtype() != int32) {
    throw std::runtime_error("paged_kv_scatter slot_mapping must be an int32 array");
  }
  if (k.shape(1) != k_cache.shape(2)) {
    throw std::runtime_error("paged_kv_scatter KV head counts must match");
  }
  if (k.shape(2) != k_cache.shape(3)) {
    throw std::runtime_error("paged_kv_scatter head dimensions must match");
  }
  if (k.shape(1) == 0) {
    throw std::runtime_error("paged_kv_scatter KV head count must be positive");
  }
  if (k.shape(2) == 0 || k.shape(2) > kMaxDecodeHeadDim) {
    throw std::runtime_error("paged_kv_scatter head dimension must be in the range [1, 256]");
  }
  if (k_cache.shape(1) == 0) {
    throw std::runtime_error("paged_kv_scatter block size must be positive");
  }
  if (k.shape(0) > 0 && k_cache.shape(0) == 0) {
    throw std::runtime_error("paged_kv_scatter KV cache must contain at least one block");
  }
  auto cache_slot_count = k_cache.shape(0) * k_cache.shape(1);
  if (cache_slot_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("paged_kv_scatter KV cache is too large for Metal attention");
  }
  validate_dense_row_contiguous(k, "paged_kv_scatter key tensor");
  validate_dense_row_contiguous(v, "paged_kv_scatter value tensor");
  validate_dense_row_contiguous(k_cache, "paged_kv_scatter key cache");
  validate_dense_row_contiguous(v_cache, "paged_kv_scatter value cache");
  validate_dense_row_contiguous(slot_mapping, "paged_kv_scatter slot_mapping");
}

void dispatch_paged_kv_scatter(
    const array& k, const array& v, array& k_cache, array& v_cache, const array& slot_mapping) {
  validate_paged_kv_scatter_inputs(k, v, k_cache, v_cache, slot_mapping);

  if (k.size() == 0) {
    return;
  }
  require_registered_library("paged_kv_scatter");

  auto num_tokens = checked_dim(k, 0, "paged_kv_scatter key tensor");
  auto num_kv_heads = checked_dim(k, 1, "paged_kv_scatter key tensor");
  auto head_dim = checked_dim(k, 2, "paged_kv_scatter key tensor");
  auto block_size = checked_dim(k_cache, 1, "paged_kv_scatter key cache");
  auto cache_slot_count_size = k_cache.shape(0) * k_cache.shape(1);
  auto cache_slot_count = static_cast<int32_t>(cache_slot_count_size);
  auto src_token_stride = checked_stride(k, 0, "paged_kv_scatter key tensor");
  auto src_head_stride = checked_stride(k, 1, "paged_kv_scatter key tensor");
  auto src_dim_stride = checked_stride(k, 2, "paged_kv_scatter key tensor");
  auto cache_block_stride = checked_stride(k_cache, 0, "paged_kv_scatter key cache");
  auto cache_offset_stride = checked_stride(k_cache, 1, "paged_kv_scatter key cache");
  auto cache_head_stride = checked_stride(k_cache, 2, "paged_kv_scatter key cache");
  auto cache_dim_stride = checked_stride(k_cache, 3, "paged_kv_scatter key cache");
  auto slot_mapping_stride = checked_stride(slot_mapping, 0, "paged_kv_scatter slot_mapping");

  auto stream = default_stream(Device::gpu);
  auto& device = metal::device(Device::gpu);
  auto* library = device.get_library(kMetalLibraryName);
  auto kernel_name = "sgl_metal_paged_kv_scatter_" + dtype_suffix(k.dtype(), "paged_kv_scatter");
  auto* kernel = device.get_kernel(kernel_name, library, kernel_name);

  auto& encoder = device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(k, 0);
  encoder.set_input_array(v, 1);
  encoder.set_output_array(k_cache, 2);
  encoder.set_output_array(v_cache, 3);
  encoder.set_input_array(slot_mapping, 4);
  encoder.set_bytes(num_tokens, 5);
  encoder.set_bytes(num_kv_heads, 6);
  encoder.set_bytes(head_dim, 7);
  encoder.set_bytes(block_size, 8);
  encoder.set_bytes(cache_slot_count, 9);
  encoder.set_bytes(src_token_stride, 10);
  encoder.set_bytes(src_head_stride, 11);
  encoder.set_bytes(src_dim_stride, 12);
  encoder.set_bytes(cache_block_stride, 13);
  encoder.set_bytes(cache_offset_stride, 14);
  encoder.set_bytes(cache_head_stride, 15);
  encoder.set_bytes(cache_dim_stride, 16);
  encoder.set_bytes(slot_mapping_stride, 17);

  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_kv_heads, num_tokens, 1), MTL::Size::Make(kDecodeThreadsPerGroup, 1, 1));

  device.add_temporary(k, stream.index);
  device.add_temporary(v, stream.index);
  device.add_temporary(k_cache, stream.index);
  device.add_temporary(v_cache, stream.index);
  device.add_temporary(slot_mapping, stream.index);
}

}  // namespace

void register_library(const std::string& metallib_path) {
  registered_library_path = metallib_path;
  auto& device = metal::device(Device::gpu);
  device.get_library(kMetalLibraryName, registered_library_path);
}

void decode_attention(nb::handle out_h, nb::handle q_h, nb::handle k_h, nb::handle v_h, float scale) {
  auto& out = checked_array(out_h, "out");
  auto& q = checked_array(q_h, "q");
  auto& k = checked_array(k_h, "k");
  auto& v = checked_array(v_h, "v");
  auto batch = checked_dim(q, 0, "decode_attention query");
  dispatch_decode_attention(out, q, k, v, scale, 0, 0, batch);
}

void decode_attention_ragged(nb::handle out_h, nb::handle q_h, nb::handle k_list_h, nb::handle v_list_h, float scale) {
  auto& out = checked_array(out_h, "out");
  auto& q = checked_array(q_h, "q");
  if (!nb::isinstance<nb::sequence>(k_list_h) || !nb::isinstance<nb::sequence>(v_list_h)) {
    throw nb::type_error("decode_attention_ragged K/V caches must be sequences");
  }

  auto batch = checked_dim(q, 0, "decode_attention query");
  auto k_list_size = PySequence_Size(k_list_h.ptr());
  auto v_list_size = PySequence_Size(v_list_h.ptr());
  if (k_list_size < 0 || v_list_size < 0) {
    throw nb::type_error("decode_attention_ragged K/V caches must be sequences");
  }
  if (k_list_size != batch || v_list_size != batch) {
    throw std::runtime_error("decode_attention_ragged K/V list lengths must match query batch");
  }
  if (out.ndim() != 4 || out.shape(0) != static_cast<size_t>(batch)) {
    throw std::runtime_error("decode_attention_ragged output batch must match query batch");
  }

  for (int32_t i = 0; i < batch; ++i) {
    nb::object k_obj = nb::steal(PySequence_GetItem(k_list_h.ptr(), i));
    nb::object v_obj = nb::steal(PySequence_GetItem(v_list_h.ptr(), i));
    if (!k_obj.is_valid() || !v_obj.is_valid()) {
      throw std::runtime_error("decode_attention_ragged failed to read K/V cache entry");
    }
    auto& k = checked_array(k_obj, "k cache");
    auto& v = checked_array(v_obj, "v cache");
    if (k.ndim() != 4 || k.shape(0) != 1 || v.ndim() != 4 || v.shape(0) != 1) {
      throw std::runtime_error("decode_attention_ragged K/V entries must have shape (1, KVH, S, D)");
    }
    dispatch_decode_attention(out, q, k, v, scale, i, 0, 1);
  }
}

void decode_attention_paged(
    nb::handle out_h,
    nb::handle q_h,
    nb::handle k_cache_h,
    nb::handle v_cache_h,
    nb::handle block_tables_h,
    nb::handle context_lens_h,
    float scale) {
  auto& out = checked_array(out_h, "out");
  auto& q = checked_array(q_h, "q");
  auto& k_cache = checked_array(k_cache_h, "k_cache");
  auto& v_cache = checked_array(v_cache_h, "v_cache");
  auto& block_tables = checked_array(block_tables_h, "block_tables");
  auto& context_lens = checked_array(context_lens_h, "context_lens");
  dispatch_decode_attention_paged(out, q, k_cache, v_cache, block_tables, context_lens, scale);
}

void flash_attn_varlen(
    nb::handle out_h,
    nb::handle q_h,
    nb::handle k_h,
    nb::handle v_h,
    nb::handle cu_seqlens_q_h,
    nb::handle cu_seqlens_k_h,
    float scale,
    bool causal) {
  auto& out = checked_array(out_h, "out");
  auto& q = checked_array(q_h, "q");
  auto& k = checked_array(k_h, "k");
  auto& v = checked_array(v_h, "v");
  auto& cu_seqlens_q = checked_array(cu_seqlens_q_h, "cu_seqlens_q");
  auto& cu_seqlens_k = checked_array(cu_seqlens_k_h, "cu_seqlens_k");
  dispatch_flash_attn_varlen(out, q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal);
}

void prefill_attention_paged(
    nb::handle out_h,
    nb::handle q_h,
    nb::handle k_h,
    nb::handle v_h,
    nb::handle k_cache_h,
    nb::handle v_cache_h,
    nb::handle block_tables_h,
    nb::handle prefix_lens_h,
    nb::handle cu_seqlens_q_h,
    float scale,
    bool causal) {
  auto& out = checked_array(out_h, "out");
  auto& q = checked_array(q_h, "q");
  auto& k = checked_array(k_h, "k");
  auto& v = checked_array(v_h, "v");
  auto& k_cache = checked_array(k_cache_h, "k_cache");
  auto& v_cache = checked_array(v_cache_h, "v_cache");
  auto& block_tables = checked_array(block_tables_h, "block_tables");
  auto& prefix_lens = checked_array(prefix_lens_h, "prefix_lens");
  auto& cu_seqlens_q = checked_array(cu_seqlens_q_h, "cu_seqlens_q");
  dispatch_prefill_attention_paged(
      out, q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_seqlens_q, scale, causal);
}

void paged_kv_scatter(
    nb::handle k_h, nb::handle v_h, nb::handle k_cache_h, nb::handle v_cache_h, nb::handle slot_mapping_h) {
  auto& k = checked_array(k_h, "k");
  auto& v = checked_array(v_h, "v");
  auto& k_cache = checked_array(k_cache_h, "k_cache");
  auto& v_cache = checked_array(v_cache_h, "v_cache");
  auto& slot_mapping = checked_array(slot_mapping_h, "slot_mapping");
  dispatch_paged_kv_scatter(k, v, k_cache, v_cache, slot_mapping);
}

NB_MODULE(_metal, m) {
  m.def(
      "register_library",
      &register_library,
      nb::arg("metallib_path"),
      "Register the precompiled sgl-kernel Metal library.");
  m.def(
      "decode_attention",
      &decode_attention,
      nb::arg("out"),
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("scale"),
      "Run decode-only Metal attention for dense MLX K/V caches.");
  m.def(
      "decode_attention_ragged",
      &decode_attention_ragged,
      nb::arg("out"),
      nb::arg("q"),
      nb::arg("k_list"),
      nb::arg("v_list"),
      nb::arg("scale"),
      "Run decode-only Metal attention for a ragged list of MLX K/V caches.");
  m.def(
      "decode_attention_paged",
      &decode_attention_paged,
      nb::arg("out"),
      nb::arg("q"),
      nb::arg("k_cache"),
      nb::arg("v_cache"),
      nb::arg("block_tables"),
      nb::arg("context_lens"),
      nb::arg("scale"),
      "Run decode-only Metal attention directly from block-paged KV caches.");
  m.def(
      "flash_attn_varlen",
      &flash_attn_varlen,
      nb::arg("out"),
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("cu_seqlens_q"),
      nb::arg("cu_seqlens_k"),
      nb::arg("scale"),
      nb::arg("causal"),
      "Run packed varlen Metal attention.");
  m.def(
      "prefill_attention_paged",
      &prefill_attention_paged,
      nb::arg("out"),
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("k_cache"),
      nb::arg("v_cache"),
      nb::arg("block_tables"),
      nb::arg("prefix_lens"),
      nb::arg("cu_seqlens_q"),
      nb::arg("scale"),
      nb::arg("causal"),
      "Run native paged-prefix Metal prefill attention.");
  m.def(
      "paged_kv_scatter",
      &paged_kv_scatter,
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("k_cache"),
      nb::arg("v_cache"),
      nb::arg("slot_mapping"),
      "Scatter token K/V tensors into block-paged MLX KV caches.");
}
