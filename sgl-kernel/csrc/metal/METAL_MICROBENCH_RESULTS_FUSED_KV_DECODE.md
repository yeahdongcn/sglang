# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:34:13`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=40, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 24 | 0.4433 | 0.6268 | 0.3098 | 4.4010 |
| decode | mlx_fast_dense | 1 | 24 | 0.2432 | 0.3800 | 0.2030 | 3.9456 |
| decode | metal_paged_lazy_h128_b16 | 1 | 24 | 0.2735 | 0.4945 | 0.2295 | 3.2733 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 24 | 0.2397 | 0.4329 | 0.2055 | 3.9992 |
| decode | metal_paged_fused_kv_h128_b16 | 1 | 24 | 0.3553 | 0.7229 | 0.3210 | 10.5861 |
| decode | mlx_gather_paged_kv | 1 | 24 | 0.2486 | 0.4000 | 0.2117 | 3.9764 |
| decode | mlx_fast_gathered_paged | 1 | 24 | 0.2972 | 0.5817 | 0.2502 | 3.5071 |
| decode | metal_paged_radix_shuffled | 1 | 24 | 0.3937 | 0.8039 | 0.3408 | 7.7642 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 24 | 0.2476 | 0.4228 | 0.2258 | 3.7176 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 24 | 0.2551 | 0.2742 | 0.2210 | 0.6482 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 1 | 24 | 0.4304 | 0.7535 | 0.3642 | 3.6290 |
| decode | mlx_gather_radix_paged_kv | 1 | 24 | 0.2599 | 0.4752 | 0.2205 | 3.8672 |
| decode | mlx_fast_radix_gathered_paged | 1 | 24 | 0.2708 | 0.5838 | 0.2506 | 3.9319 |
| decode | metal_dense_wrapper | 1 | 24 | 0.4886 | 1.0134 | 0.4153 | 7.6486 |
| decode | metal_paged | 1 | 288 | 0.4879 | 0.9664 | 0.3999 | 4.3368 |
| decode | mlx_fast_dense | 1 | 288 | 0.2992 | 0.5200 | 0.2585 | 3.6422 |
| decode | metal_paged_lazy_h128_b16 | 1 | 288 | 0.3245 | 0.5039 | 0.2617 | 4.0539 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 288 | 0.3100 | 0.4961 | 0.2832 | 3.4478 |
| decode | metal_paged_fused_kv_h128_b16 | 1 | 288 | 0.5252 | 1.1069 | 0.4384 | 5.9616 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3671 | 0.5530 | 0.3297 | 4.1727 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.4181 | 0.7657 | 0.3610 | 4.4674 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.5653 | 0.9929 | 0.4858 | 6.9104 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 288 | 0.3685 | 0.6168 | 0.3126 | 4.6075 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 288 | 0.3085 | 0.5387 | 0.2636 | 7.9807 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 1 | 288 | 0.6174 | 1.0638 | 0.5203 | 5.5122 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3791 | 0.6993 | 0.2918 | 3.9980 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.4211 | 0.7888 | 0.3291 | 4.0054 |
| decode | metal_dense_wrapper | 1 | 288 | 0.8293 | 1.4523 | 0.6429 | 11.5786 |
| decode | metal_paged | 1 | 2162 | 0.6540 | 0.9227 | 0.6030 | 7.7137 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4105 | 0.6087 | 0.3696 | 6.9642 |
| decode | metal_paged_lazy_h128_b16 | 1 | 2162 | 0.4696 | 0.7112 | 0.4113 | 4.7556 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 2162 | 0.4578 | 0.4722 | 0.4010 | 0.7385 |
| decode | metal_paged_fused_kv_h128_b16 | 1 | 2162 | 0.6098 | 0.8654 | 0.5453 | 7.7512 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.1271 | 1.8553 | 0.8924 | 6.6616 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2193 | 1.8674 | 1.1028 | 9.3528 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6342 | 1.0457 | 0.5840 | 4.1140 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 2162 | 0.4758 | 0.7963 | 0.4345 | 5.1725 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 2162 | 0.5127 | 0.9171 | 0.4618 | 8.7647 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 1 | 2162 | 0.6824 | 1.0504 | 0.6372 | 4.3434 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.1257 | 1.8199 | 0.9329 | 8.0645 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.7631 | 2.8094 | 1.1285 | 12.1421 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.6913 | 3.5951 | 1.8516 | 9.8474 |
| decode | metal_paged | 4 | 24 | 0.5040 | 0.7442 | 0.4170 | 4.4878 |
| decode | mlx_fast_dense | 4 | 24 | 0.3085 | 0.4447 | 0.2781 | 3.6923 |
| decode | metal_paged_lazy_h128_b16 | 4 | 24 | 0.3205 | 0.3348 | 0.2843 | 0.6490 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 24 | 0.3422 | 0.4343 | 0.3141 | 3.7670 |
| decode | metal_paged_fused_kv_h128_b16 | 4 | 24 | 0.5540 | 0.8015 | 0.4592 | 4.4324 |
| decode | mlx_gather_paged_kv | 4 | 24 | 0.3721 | 0.5603 | 0.2815 | 3.4969 |
| decode | mlx_fast_gathered_paged | 4 | 24 | 0.4482 | 0.6260 | 0.3958 | 3.6737 |
| decode | metal_paged_radix_shuffled | 4 | 24 | 0.5685 | 1.1330 | 0.5001 | 6.9061 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 24 | 0.4016 | 0.8373 | 0.3215 | 10.0003 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 24 | 0.3899 | 0.6442 | 0.3128 | 3.6133 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 4 | 24 | 0.6698 | 1.0799 | 0.4874 | 5.6922 |
| decode | mlx_gather_radix_paged_kv | 4 | 24 | 0.3985 | 0.6868 | 0.3405 | 4.3815 |
| decode | mlx_fast_radix_gathered_paged | 4 | 24 | 0.4380 | 0.5713 | 0.3614 | 4.5764 |
| decode | metal_dense_wrapper | 4 | 24 | 1.0459 | 1.6650 | 0.5905 | 7.6502 |
| decode | metal_paged | 4 | 288 | 0.7732 | 1.1938 | 0.5696 | 4.6035 |
| decode | mlx_fast_dense | 4 | 288 | 0.4499 | 0.7223 | 0.3736 | 3.3041 |
| decode | metal_paged_lazy_h128_b16 | 4 | 288 | 0.4625 | 0.8307 | 0.4048 | 4.2760 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 288 | 0.6976 | 1.0716 | 0.4477 | 5.0060 |
| decode | metal_paged_fused_kv_h128_b16 | 4 | 288 | 0.9098 | 1.5056 | 0.6606 | 5.3011 |
| decode | mlx_gather_paged_kv | 4 | 288 | 1.0009 | 1.6247 | 0.7288 | 4.7787 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 1.2421 | 1.9264 | 0.8779 | 7.3633 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.8620 | 1.5933 | 0.5843 | 6.0383 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 288 | 0.4946 | 0.8346 | 0.4007 | 4.0773 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 288 | 0.5196 | 0.9794 | 0.4075 | 7.4654 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 4 | 288 | 0.7217 | 1.0918 | 0.6048 | 8.3841 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.9204 | 1.4713 | 0.7782 | 9.1519 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 1.3696 | 2.1291 | 1.0060 | 6.6330 |
| decode | metal_dense_wrapper | 4 | 288 | 1.6089 | 2.3839 | 1.1926 | 9.2122 |
| decode | metal_paged | 4 | 2162 | 1.4753 | 2.3179 | 1.1110 | 9.6461 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1629 | 1.8996 | 0.9343 | 9.8360 |
| decode | metal_paged_lazy_h128_b16 | 4 | 2162 | 1.7285 | 2.3490 | 1.0005 | 7.6359 |
| decode | metal_paged_lazy_regscore_h128_b16 | 4 | 2162 | 1.6548 | 2.1456 | 1.1300 | 7.3527 |
| decode | metal_paged_fused_kv_h128_b16 | 4 | 2162 | 1.9870 | 2.6909 | 1.3020 | 10.0548 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.4803 | 5.0280 | 2.9553 | 10.6993 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 6.3638 | 7.2422 | 3.8363 | 13.9599 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.8991 | 2.5980 | 1.1421 | 9.0737 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 2162 | 1.3287 | 1.8756 | 0.9167 | 5.1188 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 4 | 2162 | 1.5829 | 2.0653 | 1.0882 | 8.3990 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 4 | 2162 | 1.4540 | 2.2279 | 1.2075 | 7.7384 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 4.6568 | 5.1783 | 3.0871 | 9.4394 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 6.9396 | 7.0336 | 3.9736 | 12.2316 |
| decode | metal_dense_wrapper | 4 | 2162 | 8.7661 | 8.9094 | 5.5131 | 15.0600 |
| decode | metal_paged | 8 | 24 | 0.5174 | 0.8005 | 0.4430 | 3.8356 |
| decode | mlx_fast_dense | 8 | 24 | 0.3096 | 0.3362 | 0.2925 | 1.3693 |
| decode | metal_paged_lazy_h128_b16 | 8 | 24 | 0.3298 | 0.5095 | 0.2937 | 3.8040 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 24 | 0.3414 | 0.6099 | 0.3176 | 4.1024 |
| decode | metal_paged_fused_kv_h128_b16 | 8 | 24 | 0.4997 | 0.7525 | 0.4300 | 4.7840 |
| decode | mlx_gather_paged_kv | 8 | 24 | 0.3766 | 0.6349 | 0.3469 | 4.8664 |
| decode | mlx_fast_gathered_paged | 8 | 24 | 0.5356 | 0.8432 | 0.4786 | 4.2701 |
| decode | metal_paged_radix_shuffled | 8 | 24 | 0.5476 | 0.9054 | 0.4098 | 6.7088 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 24 | 0.3280 | 0.6027 | 0.2750 | 4.0342 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 24 | 0.3568 | 0.6572 | 0.3014 | 4.8873 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 8 | 24 | 0.6467 | 1.0700 | 0.4518 | 4.5616 |
| decode | mlx_gather_radix_paged_kv | 8 | 24 | 0.3246 | 0.5903 | 0.2912 | 3.7683 |
| decode | mlx_fast_radix_gathered_paged | 8 | 24 | 0.4643 | 0.9571 | 0.4038 | 5.9206 |
| decode | metal_dense_wrapper | 8 | 24 | 0.5782 | 0.9411 | 0.5153 | 4.8023 |
| decode | metal_paged | 8 | 288 | 0.7263 | 0.9880 | 0.6451 | 4.4263 |
| decode | mlx_fast_dense | 8 | 288 | 0.7497 | 1.1592 | 0.4920 | 5.0374 |
| decode | metal_paged_lazy_h128_b16 | 8 | 288 | 0.6099 | 1.0257 | 0.5025 | 5.2994 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 288 | 0.8339 | 1.2020 | 0.6075 | 4.7186 |
| decode | metal_paged_fused_kv_h128_b16 | 8 | 288 | 0.8664 | 1.3177 | 0.7236 | 4.4320 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.3102 | 2.1188 | 1.0679 | 6.6612 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.5336 | 2.2282 | 1.2872 | 8.8967 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.8544 | 1.0796 | 0.7225 | 6.7503 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 288 | 0.7351 | 1.0304 | 0.5385 | 5.6526 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 288 | 0.7519 | 1.2044 | 0.5656 | 5.4583 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 8 | 288 | 1.0001 | 1.6001 | 0.7413 | 8.7622 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.4745 | 2.3029 | 0.9871 | 7.9280 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.7182 | 2.4920 | 1.1847 | 7.0733 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4894 | 2.1955 | 1.2855 | 7.5544 |
| decode | metal_paged | 8 | 2162 | 2.8960 | 3.6236 | 1.7613 | 8.8978 |
| decode | mlx_fast_dense | 8 | 2162 | 2.3780 | 3.3672 | 1.6539 | 11.4135 |
| decode | metal_paged_lazy_h128_b16 | 8 | 2162 | 2.0927 | 3.1615 | 1.7230 | 10.0072 |
| decode | metal_paged_lazy_regscore_h128_b16 | 8 | 2162 | 2.6428 | 3.5251 | 1.8067 | 7.1678 |
| decode | metal_paged_fused_kv_h128_b16 | 8 | 2162 | 2.7102 | 3.6645 | 1.9485 | 11.3829 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 9.6065 | 9.9802 | 5.5505 | 17.8908 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 13.9204 | 13.0920 | 7.1904 | 18.8066 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.4461 | 3.2858 | 1.6783 | 9.9755 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 2162 | 2.8258 | 3.3795 | 1.7589 | 8.4200 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 8 | 2162 | 2.4801 | 3.5967 | 1.8739 | 11.2611 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 8 | 2162 | 2.7842 | 3.3539 | 2.0570 | 8.7395 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 10.7664 | 10.5945 | 5.6451 | 16.5141 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 12.0419 | 12.2495 | 7.0764 | 20.6783 |
| decode | metal_dense_wrapper | 8 | 2162 | 15.1701 | 15.6495 | 9.2812 | 22.1337 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.7113 | 0.7113 | 0.7113 | 0.7113 |
| prefill | metal_flash_varlen | 1 | 16 | 0.6232 | 0.6232 | 0.6232 | 0.6232 |
| prefill | mlx_fast_dense | 1 | 16 | 0.3110 | 0.3110 | 0.3110 | 0.3110 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3691 | 0.5516 | 0.3109 | 2.8890 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.7035 | 1.0158 | 0.6153 | 4.1038 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.6748 | 1.0591 | 0.5785 | 7.3565 |
