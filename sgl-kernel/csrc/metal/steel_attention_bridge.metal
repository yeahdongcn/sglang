#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

instantiate_kernel(
    "sgl_steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
    attention,
    half,
    32,
    16,
    128,
    4,
    1,
    half,
    float)
