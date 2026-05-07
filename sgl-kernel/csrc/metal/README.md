# sgl-kernel Metal kernels

Custom Apple Metal kernels for the MLX backend on Apple Silicon. Shader sources (`*.metal`) and C++ host / nanobind sources (`*.cpp`) in this directory are compiled by [`sgl-kernel/setup_metal.py`](../../setup_metal.py) into the `sgl_kernel._metal` extension and the `sgl_metal_kernels.metallib` archive, and exposed through Python wrappers in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py).

## Public Python API

The supported public entry points are exported from `sgl_kernel.metal`:

| API | Input layout | Description |
| --- | --- | --- |
| `decode_attention` | `q: (B, H, 1, D)`, `k/v: (B, KVH, S, D)` | Decode-only dense-cache attention with GQA/MQA and fp32 softmax accumulation. |
| `decode_attention_ragged` | `q: (B, H, 1, D)`, per-request `k/v: (1, KVH, S_i, D)` | Decode-only attention for ragged per-request K/V caches. |
| `paged_kv_scatter` | `k/v: (num_tokens, KVH, D)`, cache `(num_blocks, block_size, KVH, D)` | Scatter projected token K/V into block-paged caches by slot mapping. |
| `decode_attention_paged` | `q: (B, H, 1, D)`, cache `(num_blocks, block_size, KVH, D)` | Decode-only attention directly over block tables and context lengths. |
| `prefill_attention_paged` | packed `q/k/v: (total_q, H/KVH, D)`, paged prefix cache | Native paged-prefix prefill attention using `block_tables`, `prefix_lens`, and `cu_seqlens_q`. |
| `flash_attn_varlen_func` | packed `q/k/v: (total, H/KVH, D)` | Compatibility wrapper for packed varlen attention. |
| `flash_attn_with_kvcache` | `q: (B, 1, H, D)`, dense or paged K/V cache | Compatibility wrapper for decode with optional dense-cache append and paged-cache decode. |

## Kernel entry points

| Metal kernel | Python API | Tested coverage |
| --- | --- | --- |
| `sgl_metal_decode_attention_*` | `decode_attention`, `decode_attention_ragged`, `flash_attn_with_kvcache` dense/ragged paths | Dense/ragged decode, GQA/MQA, fp16/fp32, invalid metadata |
| `sgl_metal_paged_kv_scatter_*` | `paged_kv_scatter` | Block-boundary slot scatter and invalid slot metadata |
| `sgl_metal_decode_attention_paged_*` | `decode_attention_paged`, `flash_attn_with_kvcache` paged path | Mixed lengths, block-table traversal, GQA/MQA, invalid metadata |
| `sgl_metal_flash_attn_varlen_*` | `flash_attn_varlen_func` | Packed causal/noncausal varlen attention and unsupported-option rejection |
| `sgl_metal_prefill_attention_paged_*` | `prefill_attention_paged` | No-prefix, partial-prefix, full-prefix, fp16/fp32, invalid metadata |

## Compatibility wrapper limits

The Metal wrappers intentionally expose a serving-oriented subset rather than full CUDA FlashAttention parity.

`flash_attn_varlen_func` does not support `seqused_q`, `seqused_k`, `page_table`, `qv`, descale tensors, sliding windows, attention chunks, softcap, split reductions, `pack_gqa`, sinks, score modifiers, auxiliary tensors, output reuse, softmax-LSE return, or `ver != 3`.

`flash_attn_with_kvcache` does not support `qv`, rotary tensors, cache left padding, packed new-K/V metadata, descale tensors, sliding windows, attention chunks, softcap, scheduler metadata, split reductions, `pack_gqa`, sinks, score modifiers, auxiliary tensors, output reuse, softmax-LSE return, or `ver != 3`. The paged path supports decode from `page_table`; it does not support `cache_batch_idx` or K/V append with `page_table`.

For first-class SGLang serving integration, prefer the native paged APIs (`paged_kv_scatter`, `decode_attention_paged`, and `prefill_attention_paged`) over the compatibility wrappers.

## Adding a new Metal kernel

1. Add the shader under `csrc/metal/<kernel>.metal`.
2. Add the C++ host / nanobind binding under `csrc/metal/<kernel>.cpp`, exporting the entry point on the `sgl_kernel._metal` module.
3. Append both files to `metal_shader_sources` and `cxx_sources` in [`sgl-kernel/setup_metal.py`](../../setup_metal.py).
4. Add a Python wrapper in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py) that validates input shapes/dtypes and calls `mx.eval` on its operands before invoking the AOT C++ entry point.
5. Add a test under [`sgl-kernel/tests/`](../../tests) and update the tables above with a short description and tested coverage.
