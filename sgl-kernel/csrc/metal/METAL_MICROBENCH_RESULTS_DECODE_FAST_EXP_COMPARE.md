# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:21:55`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=40, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4652 | 0.9329 | 0.3448 | 5.9582 |
| decode | mlx_fast_dense | 1 | 288 | 0.2542 | 0.3353 | 0.2218 | 1.1530 |
| decode | metal_paged_lazy_h128_b16 | 1 | 288 | 0.3003 | 0.4961 | 0.2545 | 3.1664 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 288 | 0.2780 | 0.5918 | 0.2556 | 6.1764 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2819 | 0.4401 | 0.2552 | 3.9283 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3451 | 0.6823 | 0.3077 | 4.2101 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4552 | 0.7359 | 0.4044 | 4.3076 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 288 | 0.3378 | 0.7463 | 0.2570 | 4.4420 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 288 | 0.3260 | 0.5151 | 0.2591 | 4.2728 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3757 | 0.8629 | 0.2983 | 4.6401 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.4104 | 0.6327 | 0.3486 | 4.2486 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7070 | 1.1942 | 0.6415 | 4.9360 |
| decode | metal_paged | 1 | 2162 | 0.6824 | 0.8035 | 0.6202 | 2.1397 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4653 | 0.6692 | 0.4305 | 4.1482 |
| decode | metal_paged_lazy_h128_b16 | 1 | 2162 | 0.5095 | 0.7069 | 0.4675 | 4.0808 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 2162 | 0.4949 | 0.5277 | 0.4664 | 1.0306 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.0703 | 1.7697 | 0.9400 | 6.1481 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.7041 | 2.4250 | 1.2425 | 6.4266 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.8416 | 1.4039 | 0.6860 | 6.7347 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 2162 | 0.5936 | 0.8575 | 0.5140 | 4.3969 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 2162 | 0.7338 | 1.2777 | 0.5130 | 9.4786 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.4599 | 2.1059 | 1.0171 | 6.2561 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.6081 | 2.4229 | 1.2818 | 7.9900 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.7207 | 3.4518 | 2.0470 | 9.0020 |
| decode | metal_paged | 4 | 288 | 0.8586 | 1.4956 | 0.6115 | 5.6474 |
| decode | mlx_fast_dense | 4 | 288 | 0.4765 | 0.8602 | 0.3698 | 5.3110 |
| decode | metal_paged_lazy_h128_b16 | 4 | 288 | 0.5366 | 0.9173 | 0.4312 | 4.4282 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 288 | 0.5113 | 0.7035 | 0.4299 | 3.5031 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7883 | 0.9568 | 0.6763 | 3.8991 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7512 | 1.0560 | 0.7095 | 4.3636 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5665 | 1.0566 | 0.5005 | 7.1851 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 288 | 0.3937 | 0.5335 | 0.3407 | 4.1928 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 288 | 0.4345 | 0.8219 | 0.3794 | 4.1556 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7362 | 1.3070 | 0.6702 | 4.7630 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.9837 | 1.6441 | 0.8375 | 6.0996 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2568 | 1.8722 | 0.9853 | 6.5854 |
| decode | metal_paged | 4 | 2162 | 1.5041 | 2.2003 | 1.1978 | 5.9287 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1498 | 1.6378 | 0.9155 | 5.6596 |
| decode | metal_paged_lazy_h128_b16 | 4 | 2162 | 1.5281 | 2.3898 | 1.0550 | 8.0952 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 2162 | 1.5231 | 2.0153 | 1.0785 | 5.4364 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.7307 | 4.6528 | 3.0525 | 10.1462 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 6.8544 | 7.1449 | 3.8118 | 12.6638 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 2.1854 | 2.9142 | 1.4725 | 9.9349 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 2162 | 1.6583 | 2.2845 | 1.0580 | 5.8406 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 2162 | 1.6562 | 2.4629 | 1.1760 | 7.0042 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.5300 | 4.5980 | 2.9823 | 10.1256 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 5.7059 | 7.1752 | 3.8722 | 13.6110 |
| decode | metal_dense_wrapper | 4 | 2162 | 9.1610 | 9.1314 | 5.6218 | 15.9169 |
| decode | metal_paged | 8 | 288 | 0.8138 | 1.1106 | 0.7114 | 4.9515 |
| decode | mlx_fast_dense | 8 | 288 | 0.4549 | 0.6147 | 0.4178 | 5.5446 |
| decode | metal_paged_lazy_h128_b16 | 8 | 288 | 0.5200 | 0.8871 | 0.4495 | 5.5244 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 288 | 0.5289 | 0.7910 | 0.4720 | 4.1906 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.2197 | 2.0287 | 0.9841 | 7.4664 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3421 | 2.1894 | 1.1798 | 6.5152 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.7263 | 1.3041 | 0.6710 | 6.9688 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 288 | 0.5660 | 1.0042 | 0.5103 | 9.4973 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 288 | 0.6500 | 1.0167 | 0.5675 | 4.8072 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.7133 | 2.2500 | 1.0603 | 6.3885 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.8378 | 2.9373 | 1.3327 | 8.9447 |
| decode | metal_dense_wrapper | 8 | 288 | 1.7592 | 2.3583 | 1.4508 | 5.9099 |
| decode | metal_paged | 8 | 2162 | 2.5238 | 3.4958 | 1.7622 | 10.0157 |
| decode | mlx_fast_dense | 8 | 2162 | 2.8094 | 3.6150 | 1.8641 | 10.3219 |
| decode | metal_paged_lazy_h128_b16 | 8 | 2162 | 2.9411 | 3.5761 | 1.6985 | 9.9395 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 2162 | 2.4100 | 3.5635 | 1.7746 | 9.2497 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 9.1230 | 9.2612 | 5.5692 | 15.8321 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 13.2919 | 13.2492 | 7.0191 | 20.9926 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.2683 | 3.0567 | 1.7797 | 9.8893 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 2162 | 2.2272 | 3.0159 | 1.6813 | 10.0503 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 2162 | 2.6573 | 3.4459 | 1.7815 | 9.5683 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 11.3533 | 11.0608 | 6.0176 | 15.6913 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 12.0245 | 12.1096 | 7.1348 | 19.4048 |
| decode | metal_dense_wrapper | 8 | 2162 | 14.9751 | 15.8315 | 10.3253 | 27.5986 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 1.0618 | 1.0618 | 1.0618 | 1.0618 |
| prefill | metal_flash_varlen | 1 | 16 | 0.7255 | 0.7255 | 0.7255 | 0.7255 |
| prefill | mlx_fast_dense | 1 | 16 | 0.3790 | 0.3790 | 0.3790 | 0.3790 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4015 | 0.7090 | 0.3427 | 3.4993 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.6252 | 0.9658 | 0.5875 | 5.1452 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.5497 | 0.6964 | 0.4436 | 2.0980 |
