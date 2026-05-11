# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:14:51`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3579 | 0.3585 | 0.3431 | 0.3842 |
| decode | mlx_fast_dense | 1 | 288 | 0.2109 | 0.2158 | 0.2066 | 0.2400 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2436 | 0.2448 | 0.2343 | 0.2580 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.2677 | 0.2736 | 0.2652 | 0.3109 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7607 | 0.7520 | 0.7019 | 0.8163 |
| decode | metal_paged | 1 | 2162 | 0.5552 | 0.5571 | 0.5276 | 0.6127 |
| decode | mlx_fast_dense | 1 | 2162 | 0.3507 | 0.3541 | 0.3448 | 0.3663 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9029 | 0.9007 | 0.8635 | 0.9250 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1010 | 1.1105 | 1.0766 | 1.1707 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.4579 | 2.4208 | 2.1678 | 2.5808 |
| decode | metal_paged | 4 | 288 | 0.4753 | 0.4829 | 0.4482 | 0.5635 |
| decode | mlx_fast_dense | 4 | 288 | 0.2895 | 0.2928 | 0.2824 | 0.3151 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7217 | 0.7141 | 0.6577 | 0.7682 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8875 | 0.8828 | 0.8255 | 0.9168 |
| decode | metal_dense_wrapper | 4 | 288 | 1.1233 | 1.1027 | 0.9731 | 1.1418 |
| decode | metal_paged | 4 | 2162 | 1.1868 | 1.1760 | 1.1073 | 1.2245 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9140 | 0.9262 | 0.8885 | 1.0051 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.4000 | 3.4581 | 2.9818 | 4.0960 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.6999 | 3.6834 | 3.5539 | 3.7390 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.4568 | 5.7186 | 5.2200 | 7.6494 |
| decode | metal_paged | 8 | 288 | 0.7256 | 0.7377 | 0.6800 | 0.9368 |
| decode | mlx_fast_dense | 8 | 288 | 0.5366 | 0.5470 | 0.5207 | 0.5866 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.1115 | 1.1040 | 1.0541 | 1.1634 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3518 | 1.3750 | 1.3087 | 1.5403 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4431 | 1.4466 | 1.3190 | 1.5961 |
| decode | metal_paged | 8 | 2162 | 1.8210 | 1.8785 | 1.7092 | 2.2385 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5070 | 1.5202 | 1.4369 | 1.6112 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.7378 | 5.7335 | 5.5802 | 5.9237 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.0736 | 7.1570 | 6.9036 | 7.6187 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.1546 | 10.3504 | 9.8395 | 12.1836 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 2.8859 | 2.8889 | 2.8747 | 2.9062 |
| prefill | metal_flash_varlen | 1 | 256 | 2.2495 | 2.2510 | 2.2390 | 2.2645 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6701 | 0.6645 | 0.6340 | 0.6892 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 30.6368 | 30.6327 | 30.3783 | 30.8828 |
| prefill | metal_flash_varlen | 1 | 1024 | 24.7330 | 24.6552 | 24.2783 | 24.9544 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.3609 | 5.3603 | 5.3132 | 5.4068 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 10.6159 | 10.6166 | 10.6074 | 10.6265 |
| prefill | metal_flash_varlen | 4 | 256 | 8.3698 | 8.3589 | 8.1724 | 8.5346 |
| prefill | mlx_fast_dense | 4 | 256 | 1.7760 | 1.7555 | 1.6731 | 1.8175 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 121.5307 | 121.7893 | 120.4321 | 123.4052 |
| prefill | metal_flash_varlen | 4 | 1024 | 92.2452 | 92.1750 | 91.9178 | 92.3620 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.7207 | 18.7319 | 18.7118 | 18.7633 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4091 | 0.4361 | 0.3410 | 0.6336 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3368 | 0.3449 | 0.3124 | 0.3920 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.4462 | 0.4463 | 0.4052 | 0.4770 |
