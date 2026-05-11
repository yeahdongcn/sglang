# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:54:00`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4248 | 0.5041 | 0.3342 | 1.3462 |
| decode | mlx_fast_dense | 1 | 288 | 0.2226 | 0.2318 | 0.2068 | 0.4332 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2870 | 0.3564 | 0.2420 | 1.2868 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.4373 | 0.7038 | 0.3453 | 2.4099 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.3869 | 0.5171 | 0.3399 | 1.4964 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3711 | 0.4439 | 0.2749 | 1.3760 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.4292 | 0.6312 | 0.3600 | 1.7043 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7397 | 0.8905 | 0.6523 | 1.7008 |
| decode | metal_paged | 1 | 2162 | 0.7473 | 0.8997 | 0.6560 | 1.8861 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4512 | 0.5608 | 0.4228 | 1.1800 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.0263 | 1.2971 | 0.8835 | 2.7493 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2957 | 1.4626 | 1.0484 | 2.5523 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.7220 | 0.8766 | 0.6497 | 2.0908 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.0892 | 1.1472 | 0.8985 | 1.8738 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.9049 | 1.9391 | 1.2966 | 3.2231 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.1623 | 2.2659 | 1.8235 | 3.1923 |
| decode | metal_paged | 4 | 288 | 0.5074 | 0.6147 | 0.4583 | 1.3795 |
| decode | mlx_fast_dense | 4 | 288 | 0.3525 | 0.4157 | 0.3013 | 0.8738 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7100 | 0.7580 | 0.6010 | 1.1467 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7890 | 0.9232 | 0.7137 | 1.6954 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5265 | 0.6470 | 0.4724 | 1.2486 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.6664 | 0.7719 | 0.6237 | 1.2485 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.8125 | 0.9116 | 0.7476 | 1.4336 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9580 | 1.0626 | 0.9073 | 1.7095 |
| decode | metal_paged | 4 | 2162 | 1.2245 | 1.3303 | 1.0705 | 1.9564 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0255 | 1.0770 | 0.8800 | 1.4681 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.5081 | 3.6389 | 2.9181 | 5.2018 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.2287 | 4.3639 | 3.7603 | 5.5270 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.3235 | 1.4133 | 1.0849 | 2.0758 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.3538 | 3.4181 | 2.9447 | 4.1105 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 4.0561 | 4.2626 | 3.7006 | 5.2685 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.7629 | 5.7648 | 5.4097 | 6.3818 |
| decode | metal_paged | 8 | 288 | 0.6596 | 0.7510 | 0.6023 | 1.6339 |
| decode | mlx_fast_dense | 8 | 288 | 0.4881 | 0.5803 | 0.4440 | 1.4340 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0258 | 1.1133 | 0.9675 | 1.7893 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.4052 | 1.7127 | 1.1941 | 3.8935 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.6888 | 0.7825 | 0.6239 | 1.8928 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.0520 | 1.1610 | 0.9920 | 2.1595 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.3252 | 1.4494 | 1.1777 | 2.0760 |
| decode | metal_dense_wrapper | 8 | 288 | 1.3292 | 1.4195 | 1.2468 | 1.8247 |
| decode | metal_paged | 8 | 2162 | 2.2871 | 2.3383 | 1.8263 | 3.3276 |
| decode | mlx_fast_dense | 8 | 2162 | 1.8633 | 1.9576 | 1.4825 | 2.6151 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 7.7273 | 7.9931 | 5.9810 | 10.5331 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 8.9272 | 9.8127 | 7.7179 | 17.7913 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.1686 | 2.2193 | 1.7656 | 3.4776 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 8.1241 | 8.2814 | 6.2571 | 10.9794 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 8.3085 | 8.5158 | 7.2880 | 12.2560 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.6740 | 10.9348 | 9.9768 | 13.4355 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 1.1844 | 1.1844 | 1.1844 | 1.1844 |
| prefill | metal_flash_varlen | 1 | 16 | 0.5182 | 0.5182 | 0.5182 | 0.5182 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2748 | 0.2748 | 0.2748 | 0.2748 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2848 | 0.2793 | 0.2220 | 0.3961 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.6958 | 1.7701 | 1.4244 | 2.3388 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.0396 | 1.1625 | 0.9985 | 1.8780 |
