# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:18:50`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=40, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4674 | 0.8824 | 0.3966 | 5.8635 |
| decode | mlx_fast_dense | 1 | 288 | 0.2869 | 0.4295 | 0.2328 | 3.8244 |
| decode | metal_paged_lazy_h128_b16 | 1 | 288 | 0.3077 | 0.5782 | 0.2520 | 4.6545 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 288 | 0.2817 | 0.6795 | 0.2157 | 4.6745 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3060 | 0.5413 | 0.2890 | 7.1762 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3512 | 0.4694 | 0.3219 | 2.4425 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.5176 | 0.9526 | 0.4278 | 9.2697 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 288 | 0.3124 | 0.4609 | 0.2817 | 4.6492 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 288 | 0.3487 | 0.7224 | 0.2910 | 4.7442 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3557 | 0.7684 | 0.3132 | 6.0681 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.4336 | 0.8166 | 0.3377 | 4.3413 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7874 | 1.2650 | 0.6764 | 6.0876 |
| decode | metal_paged | 1 | 2162 | 0.7665 | 1.2364 | 0.6622 | 7.7967 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5085 | 0.5768 | 0.4565 | 1.8667 |
| decode | metal_paged_lazy_h128_b16 | 1 | 2162 | 0.5941 | 0.6240 | 0.5087 | 1.5200 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 2162 | 0.5665 | 0.9821 | 0.5011 | 7.3600 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.2929 | 2.2254 | 0.9933 | 8.2025 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.9982 | 2.7736 | 1.2515 | 7.9842 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.8023 | 1.2343 | 0.7261 | 9.1604 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 2162 | 0.6884 | 1.2795 | 0.5482 | 8.2334 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 2162 | 0.6099 | 0.9067 | 0.5082 | 4.9573 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.2934 | 2.1603 | 0.9308 | 7.9424 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.5840 | 2.4222 | 1.2166 | 7.2862 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.3713 | 3.5348 | 1.9771 | 9.7716 |
| decode | metal_paged | 4 | 288 | 0.6764 | 1.0089 | 0.5720 | 4.5424 |
| decode | mlx_fast_dense | 4 | 288 | 0.3579 | 0.5353 | 0.3126 | 4.3245 |
| decode | metal_paged_lazy_h128_b16 | 4 | 288 | 0.5310 | 0.8192 | 0.4053 | 6.4023 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 288 | 0.3890 | 0.4065 | 0.3569 | 0.5398 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6545 | 0.9084 | 0.6106 | 4.7272 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7825 | 1.2281 | 0.7427 | 8.4780 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.6332 | 1.2626 | 0.5519 | 5.3930 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 288 | 0.5055 | 0.9819 | 0.3690 | 11.0745 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 288 | 0.4450 | 0.8197 | 0.3867 | 4.0667 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7535 | 1.3187 | 0.6760 | 6.3682 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.9896 | 1.7841 | 0.8118 | 6.6800 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2300 | 2.0013 | 0.9572 | 6.9990 |
| decode | metal_paged | 4 | 2162 | 1.3597 | 2.1449 | 1.0595 | 5.5455 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1643 | 1.5262 | 0.9823 | 6.8493 |
| decode | metal_paged_lazy_h128_b16 | 4 | 2162 | 1.3119 | 2.1674 | 1.0020 | 10.1660 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 2162 | 1.6468 | 2.3143 | 1.0871 | 6.4318 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.6349 | 5.5545 | 3.1243 | 12.8893 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 7.3923 | 7.3166 | 3.7258 | 17.2156 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.6618 | 2.4993 | 1.0784 | 9.0246 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 2162 | 1.2670 | 1.9715 | 1.0300 | 8.5572 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 2162 | 1.4718 | 2.2300 | 1.0174 | 5.8642 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.4741 | 4.7422 | 3.0517 | 13.5833 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 7.2066 | 7.4009 | 3.9366 | 14.5900 |
| decode | metal_dense_wrapper | 4 | 2162 | 9.2764 | 9.0653 | 5.5285 | 14.7548 |
| decode | metal_paged | 8 | 288 | 0.8239 | 1.2736 | 0.7258 | 7.8572 |
| decode | mlx_fast_dense | 8 | 288 | 0.4937 | 0.9226 | 0.4364 | 4.3623 |
| decode | metal_paged_lazy_h128_b16 | 8 | 288 | 0.5951 | 0.8572 | 0.5345 | 7.5668 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 288 | 0.5096 | 0.8960 | 0.4731 | 4.1617 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0771 | 1.7769 | 0.9576 | 8.8953 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.2397 | 2.1785 | 1.1502 | 7.0543 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.7646 | 1.3526 | 0.6627 | 5.2587 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 288 | 0.5904 | 0.9544 | 0.5418 | 4.2804 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 288 | 0.5498 | 0.8475 | 0.4955 | 5.3102 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.5656 | 2.3347 | 1.0589 | 5.7251 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.7714 | 2.5729 | 1.2756 | 7.7492 |
| decode | metal_dense_wrapper | 8 | 288 | 1.8318 | 2.5188 | 1.3423 | 9.8811 |
| decode | metal_paged | 8 | 2162 | 2.7288 | 3.7635 | 1.7021 | 10.8969 |
| decode | mlx_fast_dense | 8 | 2162 | 2.4534 | 3.8137 | 1.7041 | 10.8992 |
| decode | metal_paged_lazy_h128_b16 | 8 | 2162 | 2.6199 | 3.6328 | 1.8176 | 9.1980 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 2162 | 3.0319 | 3.6689 | 2.0716 | 6.6867 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 9.0873 | 9.3798 | 5.4908 | 17.5440 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 13.1681 | 13.8264 | 7.0756 | 22.1852 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.2261 | 3.1880 | 1.9411 | 8.5693 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 2162 | 2.2400 | 3.4184 | 1.8319 | 10.2404 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 2162 | 2.6004 | 3.7782 | 1.7782 | 12.4936 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 10.6170 | 10.8125 | 6.0800 | 18.9952 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 11.9911 | 12.4120 | 7.1081 | 19.3115 |
| decode | metal_dense_wrapper | 8 | 2162 | 15.2665 | 14.7887 | 9.6123 | 19.5950 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.8460 | 0.8460 | 0.8460 | 0.8460 |
| prefill | metal_flash_varlen | 1 | 16 | 4.2867 | 4.2867 | 4.2867 | 4.2867 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2750 | 0.2750 | 0.2750 | 0.2750 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3035 | 0.5056 | 0.2382 | 4.0278 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.5417 | 0.9434 | 0.4263 | 4.2045 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.4092 | 0.6502 | 0.3629 | 5.0326 |
