# Metal Attention Microbenchmark Results

Generated: `2026-05-09 16:53:00`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 1.1354 | 1.5313 | 0.5039 | 3.3168 |
| decode | mlx_fast_dense | 1 | 288 | 0.2427 | 0.3684 | 0.1798 | 0.8773 |
| decode | mlx_gather_paged_kv | 1 | 288 | 1.2545 | 1.4451 | 0.3490 | 4.1273 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3509 | 0.4355 | 0.3020 | 1.0989 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6090 | 0.8255 | 0.5842 | 1.8274 |
| decode | metal_paged | 1 | 2162 | 0.8156 | 0.8895 | 0.7864 | 1.2140 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4792 | 0.6021 | 0.3907 | 1.2855 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.0576 | 1.7348 | 0.8905 | 4.6148 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 3.0548 | 3.8273 | 1.1808 | 10.5809 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.9430 | 3.2623 | 1.7385 | 7.6200 |
| decode | metal_paged | 4 | 288 | 0.6957 | 1.4136 | 0.5332 | 3.4850 |
| decode | mlx_fast_dense | 4 | 288 | 0.3440 | 0.3662 | 0.3163 | 0.4627 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7135 | 0.9335 | 0.6523 | 2.5249 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 1.3857 | 1.5142 | 0.8121 | 3.6461 |
| decode | metal_dense_wrapper | 4 | 288 | 1.8655 | 2.2673 | 1.0972 | 5.7716 |
| decode | metal_paged | 4 | 2162 | 2.8465 | 3.0819 | 2.0774 | 5.1882 |
| decode | mlx_fast_dense | 4 | 2162 | 1.7122 | 2.3011 | 1.0082 | 6.4886 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.8059 | 6.6020 | 3.8728 | 11.6398 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 7.3535 | 7.3419 | 3.6184 | 12.9232 |
| decode | metal_dense_wrapper | 4 | 2162 | 7.2016 | 8.0932 | 5.3107 | 11.5439 |
| decode | metal_paged | 8 | 288 | 1.2422 | 1.7057 | 0.7803 | 3.8597 |
| decode | mlx_fast_dense | 8 | 288 | 0.7127 | 1.0169 | 0.5235 | 2.4235 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.9594 | 2.2910 | 1.1527 | 4.5372 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.7091 | 2.0903 | 1.2659 | 4.3661 |
| decode | metal_dense_wrapper | 8 | 288 | 2.0089 | 2.9545 | 1.3743 | 9.4492 |
| decode | metal_paged | 8 | 2162 | 5.3695 | 5.6506 | 4.4983 | 7.9421 |
| decode | mlx_fast_dense | 8 | 2162 | 3.2659 | 3.6925 | 1.9440 | 7.2744 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 10.0754 | 10.1730 | 6.1224 | 14.2802 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.6062 | 8.1751 | 7.4743 | 11.8151 |
| decode | metal_dense_wrapper | 8 | 2162 | 11.5231 | 11.8008 | 9.7892 | 14.5944 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 9.9961 | 10.0351 | 9.9435 | 10.1657 |
| prefill | metal_flash_varlen | 1 | 256 | 9.9246 | 9.9265 | 9.8820 | 9.9727 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5957 | 0.6296 | 0.5773 | 0.7158 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 149.9202 | 154.8475 | 149.6239 | 164.9985 |
| prefill | metal_flash_varlen | 1 | 1024 | 149.6707 | 149.7132 | 149.6613 | 149.8075 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.3524 | 5.3054 | 5.1512 | 5.4127 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 38.8888 | 38.9135 | 38.8570 | 38.9948 |
| prefill | metal_flash_varlen | 4 | 256 | 38.5884 | 38.5880 | 38.5169 | 38.6586 |
| prefill | mlx_fast_dense | 4 | 256 | 1.6066 | 1.6205 | 1.6009 | 1.6540 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 599.7439 | 599.8367 | 595.7088 | 604.0575 |
| prefill | metal_flash_varlen | 4 | 1024 | 613.8462 | 622.0864 | 598.4217 | 653.9913 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.6275 | 18.6405 | 18.5755 | 18.7184 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3260 | 0.3367 | 0.2870 | 0.4629 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2982 | 0.3603 | 0.2236 | 0.6648 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.2518 | 0.2562 | 0.2359 | 0.3245 |
