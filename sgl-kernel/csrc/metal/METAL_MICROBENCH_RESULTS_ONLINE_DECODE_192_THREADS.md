# Metal Attention Microbenchmark Results

Generated: `2026-05-09 18:10:59`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=1
prefill_prefix_lens=[]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 2162 | 0.6239 | 0.6649 | 0.5866 | 0.9385 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5407 | 0.5619 | 0.4170 | 0.7521 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.2180 | 1.4345 | 0.8978 | 3.1149 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2575 | 1.4406 | 1.1613 | 2.0603 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6459 | 0.6437 | 0.6264 | 0.6637 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 0.9432 | 1.0431 | 0.8825 | 1.5037 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.2547 | 1.4528 | 1.1965 | 2.7534 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.1553 | 2.1831 | 1.8944 | 2.9451 |
| decode | metal_paged | 4 | 2162 | 1.1775 | 1.1940 | 1.0441 | 1.4298 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9077 | 1.0490 | 0.8501 | 2.1100 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.4530 | 3.6362 | 3.0587 | 4.8630 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.9348 | 3.9595 | 3.5259 | 4.9028 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.1850 | 1.1704 | 1.0809 | 1.2521 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.1337 | 3.2174 | 2.9098 | 3.7093 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 3.9887 | 4.0727 | 3.6430 | 4.4957 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.0560 | 5.1601 | 4.9595 | 5.5192 |
| decode | metal_paged | 8 | 2162 | 1.9805 | 2.0923 | 1.8559 | 2.7274 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5379 | 1.6644 | 1.4058 | 2.6411 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.1055 | 6.2822 | 5.6671 | 7.2133 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.5995 | 7.5662 | 7.2911 | 7.8577 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 1.7994 | 1.9037 | 1.7509 | 2.3278 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 6.1195 | 6.1753 | 5.5287 | 6.9239 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 10.4050 | 13.3510 | 7.4958 | 28.5539 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.3251 | 11.0485 | 9.7435 | 15.6705 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.4795 | 0.4795 | 0.4795 | 0.4795 |
| prefill | metal_flash_varlen | 1 | 16 | 0.4628 | 0.4628 | 0.4628 | 0.4628 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2410 | 0.2410 | 0.2410 | 0.2410 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.1818 | 0.1901 | 0.1785 | 0.2302 |
