#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/stream.h"

namespace nb = nanobind;
using namespace mlx::core;

namespace {

constexpr const char* kLibraryName = "sgl_metal_kernels";

enum RopeDtypeIdx : int {
  kF16 = 0,
  kBF16 = 1,
  kF32 = 2,
  kNumDtypes = 3,
};

constexpr const char* kKernelNames[kNumDtypes] = {
    "rope_neox_f16",
    "rope_neox_bf16",
    "rope_neox_f32",
};

MTL::Library* g_library = nullptr;
MTL::ComputePipelineState* g_pipelines[kNumDtypes] = {nullptr, nullptr, nullptr};

int dtype_idx(Dtype dt) {
  switch (dt) {
    case float16:
      return kF16;
    case bfloat16:
      return kBF16;
    case float32:
      return kF32;
    default:
      throw std::runtime_error("Unsupported dtype for rope_neox Metal kernel");
  }
}

struct RopeParams {
  uint32_t head_dim;
  uint32_t rope_dim;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
};

void ensure_supported(
    const array& q,
    const array& k,
    const array& cos_sin_cache,
    const array& positions,
    const array& q_out,
    const array& k_out,
    int head_dim,
    int rope_dim,
    int num_qo_heads,
    int num_kv_heads) {
  if (q.ndim() != 3 || k.ndim() != 3 || q_out.ndim() != 3 || k_out.ndim() != 3) {
    throw std::runtime_error("rope_neox expects q, k, q_out, and k_out to have shape [tokens, heads, head_dim]");
  }
  if (cos_sin_cache.ndim() != 2) {
    throw std::runtime_error("rope_neox expects cos_sin_cache to have shape [max_pos, rope_dim]");
  }
  if (positions.ndim() != 1) {
    throw std::runtime_error("rope_neox expects positions to have shape [tokens]");
  }
  if (q.shape() != q_out.shape() || k.shape() != k_out.shape()) {
    throw std::runtime_error("rope_neox requires output arrays to match input shapes");
  }
  if (q.dtype() != q_out.dtype() || k.dtype() != k_out.dtype()) {
    throw std::runtime_error("rope_neox requires output arrays to match input dtypes");
  }
  if (q.shape(0) != k.shape(0) || q.shape(0) != positions.shape(0)) {
    throw std::runtime_error("rope_neox expects matching token dimension for q, k, and positions");
  }
  if (q.shape(1) != num_qo_heads || k.shape(1) != num_kv_heads) {
    throw std::runtime_error("rope_neox head-count arguments do not match q/k shapes");
  }
  if (q.shape(2) != head_dim || k.shape(2) != head_dim) {
    throw std::runtime_error("rope_neox head_dim argument does not match q/k shapes");
  }
  if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
    throw std::runtime_error("rope_neox requires even rope_dim with 0 < rope_dim <= head_dim");
  }
  if (rope_dim != head_dim) {
    throw std::runtime_error("rope_neox Metal kernel currently requires rope_dim == head_dim");
  }
  if (cos_sin_cache.shape(1) != rope_dim) {
    throw std::runtime_error("rope_neox rope_dim does not match cos_sin_cache.shape[1]");
  }
  if (positions.dtype() != int32) {
    throw std::runtime_error("rope_neox requires positions dtype=int32");
  }
  if (cos_sin_cache.dtype() != float32) {
    throw std::runtime_error("rope_neox requires cos_sin_cache dtype=float32");
  }
  if (q.dtype() != k.dtype()) {
    throw std::runtime_error("rope_neox requires q and k to have the same dtype");
  }
  if (!q.flags().row_contiguous || !k.flags().row_contiguous || !q_out.flags().row_contiguous ||
      !k_out.flags().row_contiguous || !cos_sin_cache.flags().row_contiguous || !positions.flags().row_contiguous) {
    throw std::runtime_error("rope_neox requires row-contiguous inputs and outputs");
  }
}

void register_library_impl(const std::string& path) {
  if (path.empty()) {
    throw std::runtime_error("register_library requires a non-empty path");
  }
  auto& d = metal::device(Device::gpu);
  g_library = d.get_library(kLibraryName, path);
  for (int i = 0; i < kNumDtypes; ++i) {
    g_pipelines[i] = d.get_kernel(kKernelNames[i], g_library);
    if (g_pipelines[i] == nullptr) {
      throw std::runtime_error(std::string("failed to resolve Metal kernel: ") + kKernelNames[i]);
    }
  }
}

void dispatch_rope_kernel(
    metal::Device& d,
    const Stream& s,
    const array& q,
    const array& k,
    const array& cos_sin_cache,
    const array& positions,
    array& q_out,
    array& k_out,
    int head_dim,
    int rope_dim,
    int num_qo_heads,
    int num_kv_heads) {
  if (g_library == nullptr) {
    throw std::runtime_error("sgl_metal_kernels.metallib not registered; call _metal.register_library(path) first");
  }

  auto* kernel = g_pipelines[dtype_idx(q.dtype())];

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_input_array(cos_sin_cache, 2);
  enc.set_input_array(positions, 3);
  enc.set_output_array(q_out, 4);
  enc.set_output_array(k_out, 5);

  RopeParams params{
      static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(rope_dim),
      static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(num_kv_heads)};
  enc.set_bytes(params, 6);

  const int tokens = static_cast<int>(q.shape(0));
  const int total_work = tokens * (num_qo_heads + num_kv_heads) * (rope_dim / 2);
  const int tpg = std::min(256, std::max(total_work, 1));
  enc.dispatch_threads(MTL::Size::Make(total_work, 1, 1), MTL::Size::Make(tpg, 1, 1));
}

// AOT eager dispatch into caller-provided output buffers. The caller is
// responsible for materialising the inputs (mx.eval) and providing
// shape/dtype-matching outputs. Output buffers may alias the input buffers
// (q_out == q, k_out == k) for in-place operation; the kernel reads each
// element pair into registers before writing, so in-place is race-free.
// We register every referenced array as a stream temporary so the command
// buffer keeps them alive until completion.
void rope_neox_impl(
    nb::handle q_h,
    nb::handle k_h,
    nb::handle cos_sin_cache_h,
    nb::handle positions_h,
    nb::handle q_out_h,
    nb::handle k_out_h,
    int head_dim,
    int rope_dim,
    int num_qo_heads,
    int num_kv_heads) {
  auto& q = *nb::inst_ptr<array>(q_h);
  auto& k = *nb::inst_ptr<array>(k_h);
  auto& cos_sin_cache = *nb::inst_ptr<array>(cos_sin_cache_h);
  auto& positions = *nb::inst_ptr<array>(positions_h);
  auto& q_out = *nb::inst_ptr<array>(q_out_h);
  auto& k_out = *nb::inst_ptr<array>(k_out_h);

  ensure_supported(q, k, cos_sin_cache, positions, q_out, k_out, head_dim, rope_dim, num_qo_heads, num_kv_heads);

  auto s = default_stream(Device::gpu);
  auto& d = metal::device(Device::gpu);

  dispatch_rope_kernel(
      d, s, q, k, cos_sin_cache, positions, q_out, k_out, head_dim, rope_dim, num_qo_heads, num_kv_heads);

  d.add_temporary(q, s.index);
  d.add_temporary(k, s.index);
  d.add_temporary(cos_sin_cache, s.index);
  d.add_temporary(positions, s.index);
  d.add_temporary(q_out, s.index);
  d.add_temporary(k_out, s.index);
}

}  // namespace

NB_MODULE(_metal, m) {
  m.def("register_library", &register_library_impl, nb::arg("path"));
  m.def(
      "rope_neox",
      &rope_neox_impl,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("cos_sin_cache"),
      nb::arg("positions"),
      nb::arg("q_out"),
      nb::arg("k_out"),
      nb::arg("head_dim"),
      nb::arg("rope_dim"),
      nb::arg("num_qo_heads"),
      nb::arg("num_kv_heads"));
}
