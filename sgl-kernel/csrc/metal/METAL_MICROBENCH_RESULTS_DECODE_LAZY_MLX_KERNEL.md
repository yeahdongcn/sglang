# Metal Paged Decode Lazy MLX Kernel Microbenchmark Results

Generated: `2026-05-09 20:45:44`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, iters=50
lazy_mlx_metal_kernel uses mx.fast.metal_kernel with block_tables indexed directly so tiny constant-address block tables compile
```

| Seq Len | Batch | Variant | Median ms | Mean ms | Min ms | Max ms |
|---:|---:|---|---:|---:|---:|---:|
| 288 | 1 | current_wrapper | 0.4904 | 0.8920 | 0.4011 | 5.7839 |
| 288 | 1 | lazy_mlx_metal_kernel | 0.2817 | 0.5855 | 0.2361 | 6.2563 |
| 288 | 1 | dense_mx_fast | 0.2733 | 0.5745 | 0.2118 | 3.8520 |
| 288 | 1 | current_wrapper_radix_shuffled | 0.4956 | 1.0410 | 0.3815 | 4.9975 |
| 288 | 1 | lazy_mlx_metal_kernel_radix_shuffled | 0.2941 | 0.7077 | 0.2446 | 4.9982 |
| 288 | 1 | mlx_fast_radix_gathered_paged | 0.3764 | 0.8316 | 0.3076 | 4.3958 |
| 288 | 4 | current_wrapper | 0.5403 | 1.0100 | 0.4819 | 4.0013 |
| 288 | 4 | lazy_mlx_metal_kernel | 0.4474 | 0.7747 | 0.3466 | 7.5163 |
| 288 | 4 | dense_mx_fast | 0.3545 | 0.6691 | 0.3061 | 5.1924 |
| 288 | 4 | current_wrapper_radix_shuffled | 0.6592 | 1.2722 | 0.5232 | 5.2411 |
| 288 | 4 | lazy_mlx_metal_kernel_radix_shuffled | 0.4420 | 0.8989 | 0.3534 | 4.3145 |
| 288 | 4 | mlx_fast_radix_gathered_paged | 1.0277 | 1.6696 | 0.7806 | 8.6457 |
| 288 | 8 | current_wrapper | 0.9468 | 1.5777 | 0.6295 | 5.2444 |
| 288 | 8 | lazy_mlx_metal_kernel | 0.4835 | 0.9045 | 0.4357 | 5.4435 |
| 288 | 8 | dense_mx_fast | 0.4580 | 0.8806 | 0.4075 | 6.0630 |
| 288 | 8 | current_wrapper_radix_shuffled | 0.7223 | 1.4580 | 0.5754 | 8.5994 |
| 288 | 8 | lazy_mlx_metal_kernel_radix_shuffled | 0.6247 | 1.2399 | 0.4608 | 7.8950 |
| 288 | 8 | mlx_fast_radix_gathered_paged | 1.7494 | 2.4866 | 1.1829 | 8.0757 |
| 2162 | 1 | current_wrapper | 1.1545 | 1.4106 | 0.6169 | 4.5817 |
| 2162 | 1 | lazy_mlx_metal_kernel | 0.5754 | 1.1414 | 0.4457 | 4.8758 |
| 2162 | 1 | dense_mx_fast | 0.4995 | 1.0011 | 0.4180 | 4.9947 |
| 2162 | 1 | current_wrapper_radix_shuffled | 0.6801 | 1.2438 | 0.6130 | 5.2614 |
| 2162 | 1 | lazy_mlx_metal_kernel_radix_shuffled | 0.4773 | 0.8525 | 0.4161 | 4.2529 |
| 2162 | 1 | mlx_fast_radix_gathered_paged | 1.3844 | 2.1047 | 1.0562 | 7.8570 |
| 2162 | 4 | current_wrapper | 1.4120 | 2.4998 | 1.0063 | 10.9927 |
| 2162 | 4 | lazy_mlx_metal_kernel | 1.1499 | 1.9690 | 0.9315 | 9.1875 |
| 2162 | 4 | dense_mx_fast | 1.1694 | 1.9154 | 0.8958 | 7.8858 |
| 2162 | 4 | current_wrapper_radix_shuffled | 2.0629 | 2.6687 | 1.1538 | 5.7332 |
| 2162 | 4 | lazy_mlx_metal_kernel_radix_shuffled | 1.2585 | 1.9137 | 0.9618 | 7.3825 |
| 2162 | 4 | mlx_fast_radix_gathered_paged | 7.1466 | 7.5142 | 3.4914 | 16.9203 |
| 2162 | 8 | current_wrapper | 2.8090 | 3.8437 | 1.7314 | 10.1009 |
| 2162 | 8 | lazy_mlx_metal_kernel | 2.7463 | 3.8244 | 1.6271 | 10.7917 |
| 2162 | 8 | dense_mx_fast | 2.7690 | 3.5380 | 1.5958 | 9.9489 |
| 2162 | 8 | current_wrapper_radix_shuffled | 2.9883 | 3.8355 | 1.6768 | 7.8019 |
| 2162 | 8 | lazy_mlx_metal_kernel_radix_shuffled | 2.7048 | 3.4452 | 1.6335 | 10.3601 |
| 2162 | 8 | mlx_fast_radix_gathered_paged | 14.0863 | 14.0656 | 7.4490 | 21.4587 |

## Batch-1 Radix Length Sweep

| Seq Len | lazy radix median ms | gather only median ms | gather+mx.fast median ms | dense mx.fast median ms |
|---:|---:|---:|---:|---:|
| 16 | 0.3303 | 0.2170 | 0.2723 | 0.2250 |
| 32 | 0.2433 | 0.2378 | 0.2672 | 0.2447 |
| 64 | 0.2543 | 0.2495 | 0.2739 | 0.2329 |
| 128 | 0.2561 | 0.2613 | 0.2895 | 0.2153 |
| 256 | 0.2514 | 0.2936 | 0.3819 | 0.2504 |
| 288 | 0.2813 | 0.3336 | 0.4474 | 0.2817 |
| 512 | 0.3166 | 0.4789 | 0.5983 | 0.2728 |
| 768 | 0.3190 | 0.4879 | 0.5692 | 0.3141 |
| 1024 | 0.3699 | 0.7316 | 1.7118 | 0.3273 |
| 1536 | 0.4829 | 1.1047 | 1.1193 | 0.3791 |
| 2162 | 0.5409 | 1.2258 | 1.3592 | 0.4776 |

Interpretation:

- The direct-indexed lazy source fixes tiny `block_tables` compilation for MLX
  `mx.fast.metal_kernel`; the previous `const device int*` block-table pointer
  failed when MLX placed a small block table in constant-address memory.
- The production radix decode route can now use lazy direct paged decode for
  supported fp16/head_dim=128/block_size=16 decode shapes at batch > 1 or
  single-request context lengths of at least 32. In the batch-1 shuffled radix
  sweep, lazy direct paged decode beats gather+MLX attention from S32 through
  S2162; the tiny S16 row remains on gather+MLX because that row was faster in
  the latest sweep.
- Contiguous no-radix raw decode remains mixed versus dense
  `mx.fast.scaled_dot_product_attention`: lazy direct paged decode is close and
  wins some long rows, but it is not a decisive replacement for the guarded
  no-radix contiguous-cache path.
