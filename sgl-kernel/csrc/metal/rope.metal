#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint head_dim;
    uint rope_dim;
    uint num_qo_heads;
    uint num_kv_heads;
};

template <typename T>
[[kernel]] void rope_neox_kernel(
    const device T* q_in [[buffer(0)]],
    const device T* k_in [[buffer(1)]],
    const device float* cos_sin_cache [[buffer(2)]],
    const device int32_t* positions [[buffer(3)]],
    device T* q_out [[buffer(4)]],
    device T* k_out [[buffer(5)]],
    constant RopeParams& params [[buffer(6)]],
    uint elem [[thread_position_in_grid]]
) {
    const uint half_dim = params.rope_dim / 2;
    const uint total_heads = params.num_qo_heads + params.num_kv_heads;
    const uint work_per_token = total_heads * half_dim;

    const uint token_id = elem / work_per_token;
    const uint rem = elem % work_per_token;
    const uint head_id = rem / half_dim;
    const uint dim_idx = rem % half_dim;

    const bool is_q = head_id < params.num_qo_heads;
    const uint actual_head = is_q ? head_id : (head_id - params.num_qo_heads);
    const uint heads_in_tensor = is_q ? params.num_qo_heads : params.num_kv_heads;

    const int32_t pos = positions[token_id];
    const float cos_val = cos_sin_cache[pos * params.rope_dim + dim_idx];
    const float sin_val = cos_sin_cache[pos * params.rope_dim + half_dim + dim_idx];

    const uint base = token_id * heads_in_tensor * params.head_dim + actual_head * params.head_dim;
    const uint idx1 = base + dim_idx;
    const uint idx2 = base + half_dim + dim_idx;

    if (is_q) {
        const float x1 = static_cast<float>(q_in[idx1]);
        const float x2 = static_cast<float>(q_in[idx2]);
        q_out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
        q_out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
    } else {
        const float x1 = static_cast<float>(k_in[idx1]);
        const float x2 = static_cast<float>(k_in[idx2]);
        k_out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
        k_out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
    }
}

template [[host_name("rope_neox_f16")]] [[kernel]] void
rope_neox_kernel<half>(
    const device half*, const device half*, const device float*,
    const device int32_t*, device half*, device half*,
    constant RopeParams&, uint);

template [[host_name("rope_neox_bf16")]] [[kernel]] void
rope_neox_kernel<bfloat>(
    const device bfloat*, const device bfloat*, const device float*,
    const device int32_t*, device bfloat*, device bfloat*,
    constant RopeParams&, uint);

template [[host_name("rope_neox_f32")]] [[kernel]] void
rope_neox_kernel<float>(
    const device float*, const device float*, const device float*,
    const device int32_t*, device float*, device float*,
    constant RopeParams&, uint);
