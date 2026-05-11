# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:20:24`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=12, prefill iters=12
prefill_prefix_lens=[432], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 434 | 0.4804 | 0.9767 | 0.3801 | 2.8095 |
| decode | mlx_fast_dense | 1 | 434 | 0.2693 | 0.3500 | 0.2298 | 1.1030 |
| decode | mlx_gather_paged_kv | 1 | 434 | 0.4236 | 0.8713 | 0.3320 | 5.6325 |
| decode | mlx_fast_gathered_paged | 1 | 434 | 0.5193 | 0.8531 | 0.3670 | 3.4528 |
| decode | metal_paged_radix_shuffled | 1 | 434 | 0.4030 | 0.8660 | 0.3168 | 3.9630 |
| decode | mlx_gather_radix_paged_kv | 1 | 434 | 0.3710 | 0.7456 | 0.3131 | 3.6141 |
| decode | mlx_fast_radix_gathered_paged | 1 | 434 | 0.4808 | 0.6918 | 0.3682 | 2.3586 |
| decode | metal_dense_wrapper | 1 | 434 | 0.7121 | 1.2159 | 0.6686 | 4.3329 |
| prefill | metal_prefill_paged_no_prefix | 1 | 2 | 0.5753 | 0.7510 | 0.5043 | 2.1133 |
| prefill | metal_flash_varlen | 1 | 2 | 0.6445 | 1.0915 | 0.4910 | 3.4925 |
| prefill | mlx_fast_dense | 1 | 2 | 0.2080 | 0.2090 | 0.1999 | 0.2256 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 2 | 0.9885 | 1.8913 | 0.7001 | 6.9873 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 2 | 0.2701 | 0.2729 | 0.2586 | 0.2903 |
| prefill | metal_prefill_paged_no_prefix | 1 | 434 | 1.5446 | 1.6741 | 1.1274 | 2.6440 |
| prefill | metal_flash_varlen | 1 | 434 | 1.3286 | 1.6313 | 1.0923 | 2.6261 |
| prefill | mlx_fast_dense | 1 | 434 | 2.0739 | 2.7591 | 1.4718 | 5.2222 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 434 | 2.7881 | 3.3491 | 2.3271 | 7.1217 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 434 | 3.3111 | 3.2218 | 2.2990 | 4.5535 |
| scatter | metal_paged_kv_scatter | 1 | 2 | 0.2427 | 0.2503 | 0.2263 | 0.3170 |
| radix_sync | mlx_set_kv_all_layers_stacked | 2 | 28 | 2.0375 | 2.4772 | 1.7438 | 4.1082 |
| radix_sync | mlx_set_kv_all_layers_list | 2 | 28 | 1.3627 | 1.6097 | 1.3118 | 3.4448 |
