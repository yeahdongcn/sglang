# Metal Attention Microbenchmark Results

Generated: `2026-05-09 16:57:30`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3584 | 0.3595 | 0.3475 | 0.3856 |
| decode | mlx_fast_dense | 1 | 288 | 0.2159 | 0.2169 | 0.2110 | 0.2232 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2412 | 0.2426 | 0.2326 | 0.2543 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.2796 | 0.2854 | 0.2696 | 0.3277 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6470 | 0.6408 | 0.6168 | 0.6621 |
| decode | metal_paged | 1 | 2162 | 0.6032 | 0.6004 | 0.5614 | 0.6400 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4363 | 0.4343 | 0.3764 | 0.4778 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9665 | 0.9942 | 0.8838 | 1.1304 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.0975 | 1.0972 | 1.0807 | 1.1129 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.3864 | 2.4796 | 2.2030 | 2.8100 |
| decode | metal_paged | 4 | 288 | 0.5107 | 0.5254 | 0.4845 | 0.5767 |
| decode | mlx_fast_dense | 4 | 288 | 0.3521 | 0.3530 | 0.3210 | 0.3739 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7901 | 0.7775 | 0.6895 | 0.8345 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9076 | 0.8979 | 0.8130 | 0.9897 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2696 | 1.2549 | 1.0740 | 1.4280 |
| decode | metal_paged | 4 | 2162 | 1.1472 | 1.2240 | 1.0472 | 1.5932 |
| decode | mlx_fast_dense | 4 | 2162 | 0.8923 | 0.9168 | 0.8667 | 0.9941 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.2103 | 3.2301 | 3.0324 | 3.4884 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.5913 | 3.6162 | 3.4948 | 3.9256 |
| decode | metal_dense_wrapper | 4 | 2162 | 4.9290 | 4.9544 | 4.8870 | 5.2170 |
| decode | metal_paged | 8 | 288 | 0.6848 | 0.6708 | 0.5829 | 0.7311 |
| decode | mlx_fast_dense | 8 | 288 | 0.4361 | 0.4385 | 0.4180 | 0.4637 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0176 | 1.0269 | 0.9416 | 1.1282 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.1621 | 1.1707 | 1.1373 | 1.2505 |
| decode | metal_dense_wrapper | 8 | 288 | 1.2508 | 1.3964 | 1.1942 | 2.3472 |
| decode | metal_paged | 8 | 2162 | 1.7833 | 1.8247 | 1.6430 | 2.0756 |
| decode | mlx_fast_dense | 8 | 2162 | 1.4936 | 1.4964 | 1.4240 | 1.5985 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.7124 | 5.9491 | 5.4988 | 6.9153 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 6.9457 | 7.0016 | 6.6182 | 7.4095 |
| decode | metal_dense_wrapper | 8 | 2162 | 9.7410 | 9.7645 | 9.4976 | 10.0045 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 10.2133 | 10.1584 | 10.0303 | 10.2317 |
| prefill | metal_flash_varlen | 1 | 256 | 9.9905 | 9.9620 | 9.8787 | 10.0169 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6325 | 0.6434 | 0.5942 | 0.7037 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 150.5815 | 151.1202 | 149.9757 | 152.8035 |
| prefill | metal_flash_varlen | 1 | 1024 | 152.1956 | 152.1696 | 151.8406 | 152.4725 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.1362 | 5.1439 | 5.0978 | 5.1976 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 39.0290 | 39.2185 | 38.5514 | 40.0752 |
| prefill | metal_flash_varlen | 4 | 256 | 38.8352 | 38.7630 | 38.4891 | 38.9647 |
| prefill | mlx_fast_dense | 4 | 256 | 1.6338 | 1.6840 | 1.5881 | 1.8300 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 623.6524 | 623.8032 | 601.3296 | 646.4277 |
| prefill | metal_flash_varlen | 4 | 1024 | 601.3310 | 605.1188 | 599.4470 | 614.5785 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.8582 | 19.0096 | 18.6815 | 19.4891 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3695 | 0.3750 | 0.3414 | 0.4345 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3431 | 0.3449 | 0.3046 | 0.3854 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3897 | 0.3874 | 0.3548 | 0.4161 |
