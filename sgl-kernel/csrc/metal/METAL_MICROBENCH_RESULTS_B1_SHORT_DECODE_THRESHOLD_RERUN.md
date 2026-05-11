# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:22:39`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=12, decode/scatter iters=120, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 16 | 0.4142 | 0.7949 | 0.3536 | 8.3514 |
| decode | mlx_fast_dense | 1 | 16 | 0.2399 | 0.4464 | 0.2078 | 4.1385 |
| decode | metal_paged_lazy_h128_b16 | 1 | 16 | 0.2537 | 0.3758 | 0.2115 | 4.6722 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 16 | 0.2570 | 0.4728 | 0.2000 | 4.8615 |
| decode | mlx_gather_paged_kv | 1 | 16 | 0.2452 | 0.4294 | 0.2061 | 4.0055 |
| decode | mlx_fast_gathered_paged | 1 | 16 | 0.3684 | 0.6734 | 0.2603 | 4.2499 |
| decode | metal_paged_radix_shuffled | 1 | 16 | 0.4850 | 0.8419 | 0.3888 | 9.9588 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 16 | 0.3097 | 0.4920 | 0.2680 | 6.5354 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 16 | 0.3107 | 0.5447 | 0.2557 | 7.0020 |
| decode | mlx_gather_radix_paged_kv | 1 | 16 | 0.2394 | 0.4081 | 0.2086 | 7.8906 |
| decode | mlx_fast_radix_gathered_paged | 1 | 16 | 0.2873 | 0.5189 | 0.2532 | 6.2184 |
| decode | metal_dense_wrapper | 1 | 16 | 0.4851 | 0.8391 | 0.4027 | 6.3222 |
| decode | metal_paged | 1 | 24 | 0.4462 | 0.7471 | 0.3792 | 4.4984 |
| decode | mlx_fast_dense | 1 | 24 | 0.2767 | 0.4084 | 0.2220 | 5.8768 |
| decode | metal_paged_lazy_h128_b16 | 1 | 24 | 0.2618 | 0.4155 | 0.2265 | 5.2207 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 24 | 0.2828 | 0.5010 | 0.2321 | 4.2010 |
| decode | mlx_gather_paged_kv | 1 | 24 | 0.2769 | 0.4845 | 0.2234 | 6.0971 |
| decode | mlx_fast_gathered_paged | 1 | 24 | 0.3389 | 0.5233 | 0.2752 | 3.8957 |
| decode | metal_paged_radix_shuffled | 1 | 24 | 0.4874 | 0.9157 | 0.4005 | 9.7326 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 24 | 0.2754 | 0.4549 | 0.2347 | 4.4048 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 24 | 0.2879 | 0.5171 | 0.2338 | 4.8656 |
| decode | mlx_gather_radix_paged_kv | 1 | 24 | 0.2929 | 0.4745 | 0.2432 | 6.5633 |
| decode | mlx_fast_radix_gathered_paged | 1 | 24 | 0.3482 | 0.5435 | 0.2887 | 6.6657 |
| decode | metal_dense_wrapper | 1 | 24 | 0.6226 | 0.9988 | 0.4825 | 7.5384 |
| decode | metal_paged | 1 | 32 | 0.6336 | 1.0799 | 0.4923 | 5.2891 |
| decode | mlx_fast_dense | 1 | 32 | 0.2704 | 0.4744 | 0.2126 | 4.1227 |
| decode | metal_paged_lazy_h128_b16 | 1 | 32 | 0.4024 | 0.6628 | 0.2572 | 12.3149 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 32 | 0.2883 | 0.4803 | 0.2230 | 4.0675 |
| decode | mlx_gather_paged_kv | 1 | 32 | 0.2714 | 0.5352 | 0.2255 | 8.9607 |
| decode | mlx_fast_gathered_paged | 1 | 32 | 0.3342 | 0.5983 | 0.2593 | 8.3624 |
| decode | metal_paged_radix_shuffled | 1 | 32 | 0.4709 | 0.7584 | 0.3795 | 4.5700 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 32 | 0.2786 | 0.5031 | 0.2407 | 3.8974 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 32 | 0.3030 | 0.4556 | 0.2143 | 4.4822 |
| decode | mlx_gather_radix_paged_kv | 1 | 32 | 0.2605 | 0.3666 | 0.2201 | 7.4833 |
| decode | mlx_fast_radix_gathered_paged | 1 | 32 | 0.3201 | 0.4487 | 0.2581 | 4.3547 |
| decode | metal_dense_wrapper | 1 | 32 | 0.6587 | 1.0747 | 0.4761 | 8.1919 |
| decode | metal_paged | 1 | 48 | 0.5005 | 0.8691 | 0.4171 | 5.2828 |
| decode | mlx_fast_dense | 1 | 48 | 0.3643 | 0.6075 | 0.2744 | 4.1778 |
| decode | metal_paged_lazy_h128_b16 | 1 | 48 | 0.3487 | 0.5883 | 0.2707 | 4.5648 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 48 | 0.3491 | 0.6355 | 0.2833 | 8.8905 |
| decode | mlx_gather_paged_kv | 1 | 48 | 0.3471 | 0.5701 | 0.2373 | 4.3160 |
| decode | mlx_fast_gathered_paged | 1 | 48 | 0.3699 | 0.7041 | 0.2696 | 4.5135 |
| decode | metal_paged_radix_shuffled | 1 | 48 | 0.5708 | 0.9646 | 0.4490 | 5.6597 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 48 | 0.3744 | 0.6725 | 0.3017 | 8.6283 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 48 | 0.3814 | 0.6162 | 0.3127 | 5.1785 |
| decode | mlx_gather_radix_paged_kv | 1 | 48 | 0.4239 | 0.6871 | 0.3264 | 6.5150 |
| decode | mlx_fast_radix_gathered_paged | 1 | 48 | 0.3758 | 0.5212 | 0.2674 | 3.9674 |
| decode | metal_dense_wrapper | 1 | 48 | 0.4674 | 0.7594 | 0.4180 | 4.7402 |
| decode | metal_paged | 1 | 64 | 0.4937 | 0.8305 | 0.3717 | 4.3084 |
| decode | mlx_fast_dense | 1 | 64 | 0.2904 | 0.4757 | 0.2235 | 4.6753 |
| decode | metal_paged_lazy_h128_b16 | 1 | 64 | 0.2847 | 0.4361 | 0.2280 | 3.6945 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 64 | 0.2814 | 0.4387 | 0.2416 | 4.7084 |
| decode | mlx_gather_paged_kv | 1 | 64 | 0.2980 | 0.4323 | 0.2439 | 4.0441 |
| decode | mlx_fast_gathered_paged | 1 | 64 | 0.3001 | 0.5631 | 0.2662 | 6.4143 |
| decode | metal_paged_radix_shuffled | 1 | 64 | 0.5396 | 0.9082 | 0.3583 | 5.7410 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 64 | 0.3307 | 0.5497 | 0.2660 | 6.3317 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 64 | 0.3429 | 0.5770 | 0.2605 | 6.0795 |
| decode | mlx_gather_radix_paged_kv | 1 | 64 | 0.3523 | 0.5487 | 0.2899 | 4.8132 |
| decode | mlx_fast_radix_gathered_paged | 1 | 64 | 0.4648 | 0.7347 | 0.3392 | 7.9605 |
| decode | metal_dense_wrapper | 1 | 64 | 0.7808 | 1.2555 | 0.5786 | 5.6392 |
| decode | metal_paged | 1 | 128 | 0.6480 | 1.1079 | 0.5057 | 6.9867 |
| decode | mlx_fast_dense | 1 | 128 | 0.3401 | 0.6422 | 0.2730 | 6.7594 |
| decode | metal_paged_lazy_h128_b16 | 1 | 128 | 0.4075 | 0.6096 | 0.3227 | 4.0721 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 128 | 0.4117 | 0.6391 | 0.3092 | 4.1983 |
| decode | mlx_gather_paged_kv | 1 | 128 | 0.3069 | 0.5696 | 0.2605 | 4.6740 |
| decode | mlx_fast_gathered_paged | 1 | 128 | 0.3553 | 0.6610 | 0.2907 | 10.2680 |
| decode | metal_paged_radix_shuffled | 1 | 128 | 0.5510 | 0.8562 | 0.4330 | 5.8890 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 128 | 0.3315 | 0.4878 | 0.2599 | 4.0245 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 128 | 0.3900 | 0.6163 | 0.2820 | 6.6808 |
| decode | mlx_gather_radix_paged_kv | 1 | 128 | 0.3081 | 0.5198 | 0.2495 | 5.9571 |
| decode | mlx_fast_radix_gathered_paged | 1 | 128 | 0.3666 | 0.6094 | 0.2941 | 4.2223 |
| decode | metal_dense_wrapper | 1 | 128 | 0.7155 | 1.1298 | 0.5677 | 5.1664 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 1.2646 | 1.2646 | 1.2646 | 1.2646 |
| prefill | metal_flash_varlen | 1 | 16 | 1.1415 | 1.1415 | 1.1415 | 1.1415 |
| prefill | mlx_fast_dense | 1 | 16 | 0.3785 | 0.3785 | 0.3785 | 0.3785 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3208 | 0.5419 | 0.2565 | 5.9588 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.6621 | 1.0747 | 0.5379 | 5.1002 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.6190 | 0.9945 | 0.4220 | 14.7528 |
