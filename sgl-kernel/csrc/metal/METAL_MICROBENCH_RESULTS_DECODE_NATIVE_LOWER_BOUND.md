# Metal Paged Decode Native Dispatch Lower Bound

Generated: `2026-05-09`

Purpose: separate public Python wrapper/allocation/materialization overhead from
the native h128/block16 paged decode dispatch. This benchmark reuses a
preallocated output array and directly calls `sgl_kernel._metal.decode_attention_paged`
after materializing query, K/V cache, block tables, context lengths, and output.

Command shape:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
seq_len=2162, warmups=8, iters=40
```

| Case | Variant | Batch | Median ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|
| decode | wrapper `decode_attention_paged_unchecked` | 1 | 1.4211 | 0.5897 | 10.6181 |
| decode | native preallocated output | 1 | 0.5375 | 0.3961 | 7.1780 |
| decode | dense `mx.fast.scaled_dot_product_attention` | 1 | 0.4344 | 0.3991 | 1.0819 |
| decode | wrapper `decode_attention_paged_unchecked` | 4 | 1.4892 | 1.0557 | 4.8141 |
| decode | native preallocated output | 4 | 1.0442 | 0.8767 | 5.5006 |
| decode | dense `mx.fast.scaled_dot_product_attention` | 4 | 1.0072 | 0.8703 | 5.9257 |
| decode | wrapper `decode_attention_paged_unchecked` | 8 | 2.3217 | 1.6271 | 6.4426 |
| decode | native preallocated output | 8 | 2.0654 | 1.5203 | 11.1007 |
| decode | dense `mx.fast.scaled_dot_product_attention` | 8 | 2.0216 | 1.4922 | 6.0320 |

Interpretation:

- The shader/native dispatch is close to dense MLX for B4/B8 long decode when
  output allocation and per-call wrapper materialization are removed.
- The public wrapper still needs to materialize `q`, `block_tables`,
  `context_lens`, and `out` before native validation; reducing that eval set
  was tested and failed correctness because lazy metadata is not
  row-contiguous to the native extension.
- The next raw-decode direction should be an output-reuse/prepared-metadata API
  or MLX custom-primitive integration, not another shader-only rewrite.
