# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:10:37`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3770 | 0.3755 | 0.3555 | 0.3882 |
| decode | mlx_fast_dense | 1 | 288 | 0.2140 | 0.2287 | 0.2084 | 0.3332 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2320 | 0.2360 | 0.2300 | 0.2501 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.2840 | 0.2860 | 0.2781 | 0.2947 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6466 | 0.6773 | 0.6372 | 0.7556 |
| decode | metal_paged | 1 | 2162 | 0.5553 | 0.5613 | 0.5479 | 0.6023 |
| decode | mlx_fast_dense | 1 | 2162 | 0.3655 | 0.3850 | 0.3527 | 0.4515 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.8878 | 0.8911 | 0.8685 | 0.9420 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.0845 | 1.0887 | 1.0679 | 1.1107 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.4165 | 2.4136 | 2.1298 | 2.7103 |
| decode | metal_paged | 4 | 288 | 0.4859 | 0.5038 | 0.4533 | 0.5918 |
| decode | mlx_fast_dense | 4 | 288 | 0.3564 | 0.3616 | 0.3456 | 0.3904 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7611 | 0.7624 | 0.6929 | 0.8365 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9236 | 0.9164 | 0.8301 | 0.9915 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2729 | 1.2690 | 1.1705 | 1.3458 |
| decode | metal_paged | 4 | 2162 | 1.5199 | 1.5373 | 1.4303 | 1.6750 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1667 | 1.1112 | 0.9317 | 1.1971 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 2.9332 | 3.0714 | 2.8403 | 3.6630 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.6289 | 3.6317 | 3.5695 | 3.7395 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.2766 | 5.3002 | 5.1607 | 5.4644 |
| decode | metal_paged | 8 | 288 | 0.7505 | 0.7607 | 0.7383 | 0.8189 |
| decode | mlx_fast_dense | 8 | 288 | 0.5305 | 0.5313 | 0.4887 | 0.5791 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.1429 | 1.1430 | 1.0964 | 1.2100 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3245 | 1.3516 | 1.3063 | 1.5272 |
| decode | metal_dense_wrapper | 8 | 288 | 1.5187 | 1.6845 | 1.4384 | 2.3475 |
| decode | metal_paged | 8 | 2162 | 1.7983 | 1.7925 | 1.7075 | 1.8442 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5161 | 1.5242 | 1.4748 | 1.5720 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.6448 | 5.7100 | 5.5990 | 5.9673 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.0699 | 7.0666 | 6.8789 | 7.2430 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.5930 | 10.5166 | 9.9796 | 10.7015 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 10.3997 | 10.4245 | 10.3214 | 10.5525 |
| prefill | metal_flash_varlen | 1 | 256 | 2.2998 | 2.2805 | 2.2382 | 2.3036 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6463 | 0.6540 | 0.6453 | 0.6706 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 150.5054 | 150.4541 | 149.7972 | 151.0597 |
| prefill | metal_flash_varlen | 1 | 1024 | 23.8415 | 23.9465 | 23.7051 | 24.2930 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.0416 | 5.0585 | 5.0380 | 5.0958 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 38.9185 | 38.8114 | 38.5851 | 38.9307 |
| prefill | metal_flash_varlen | 4 | 256 | 8.0980 | 8.1625 | 8.0815 | 8.3080 |
| prefill | mlx_fast_dense | 4 | 256 | 1.8398 | 2.6368 | 1.8373 | 4.2334 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 596.0683 | 596.0941 | 595.8079 | 596.4060 |
| prefill | metal_flash_varlen | 4 | 1024 | 93.1022 | 94.0717 | 92.4665 | 96.6465 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.6120 | 18.5979 | 18.5473 | 18.6344 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4259 | 0.3954 | 0.3145 | 0.4672 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2416 | 0.2453 | 0.2328 | 0.2730 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3081 | 0.3099 | 0.2842 | 0.3387 |
