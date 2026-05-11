# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:16:21`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=40, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4381 | 0.6214 | 0.3704 | 2.1513 |
| decode | mlx_fast_dense | 1 | 288 | 0.2688 | 0.4139 | 0.2330 | 3.7063 |
| decode | metal_paged_lazy_h128_b16 | 1 | 288 | 0.2597 | 0.4238 | 0.2157 | 3.8717 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3268 | 0.6716 | 0.2709 | 4.4062 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3239 | 0.3624 | 0.2996 | 1.6414 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4508 | 0.6910 | 0.4060 | 9.0058 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 288 | 0.2742 | 0.4212 | 0.2493 | 4.7637 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3055 | 0.3600 | 0.2750 | 1.7060 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3343 | 0.5671 | 0.3008 | 8.4104 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6321 | 0.7045 | 0.6069 | 1.7958 |
| decode | metal_paged | 1 | 2162 | 0.6415 | 1.1036 | 0.5947 | 6.7599 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5002 | 0.8053 | 0.4424 | 3.7183 |
| decode | metal_paged_lazy_h128_b16 | 1 | 2162 | 0.5474 | 0.9576 | 0.4468 | 3.9457 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.1612 | 1.9015 | 0.9937 | 6.5975 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.6634 | 2.3162 | 1.1760 | 7.5635 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.9635 | 1.4722 | 0.6875 | 5.9031 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 2162 | 0.6979 | 1.1194 | 0.5498 | 4.9512 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.5242 | 2.2555 | 1.0880 | 5.4685 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.6758 | 2.5634 | 1.2110 | 10.2957 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.5969 | 3.4480 | 2.0666 | 8.8593 |
| decode | metal_paged | 4 | 288 | 0.6889 | 1.1680 | 0.5980 | 8.6635 |
| decode | mlx_fast_dense | 4 | 288 | 0.4657 | 0.7429 | 0.3574 | 4.4940 |
| decode | metal_paged_lazy_h128_b16 | 4 | 288 | 0.4774 | 0.7264 | 0.4195 | 5.8157 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.8239 | 1.3401 | 0.6914 | 8.8794 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 1.0296 | 1.6756 | 0.8248 | 4.6983 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.7174 | 1.3370 | 0.5644 | 6.8538 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 288 | 0.5299 | 0.9585 | 0.4258 | 9.2905 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 1.0247 | 1.6004 | 0.7326 | 5.2570 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 1.1401 | 1.7984 | 0.8237 | 8.2070 |
| decode | metal_dense_wrapper | 4 | 288 | 1.4869 | 2.2637 | 1.1102 | 9.8205 |
| decode | metal_paged | 4 | 2162 | 1.2717 | 1.9837 | 1.0662 | 7.9179 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1882 | 1.6552 | 0.9437 | 5.5573 |
| decode | metal_paged_lazy_h128_b16 | 4 | 2162 | 1.5450 | 2.5075 | 1.0380 | 8.9703 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.0837 | 4.8545 | 3.0098 | 11.3913 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 6.6339 | 6.8110 | 3.6287 | 13.6280 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.9072 | 2.6558 | 1.1814 | 10.5015 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 2162 | 1.4775 | 2.2816 | 1.0410 | 8.3293 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 4.2546 | 5.4292 | 3.1115 | 14.0652 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 5.9360 | 6.6430 | 3.8192 | 13.6603 |
| decode | metal_dense_wrapper | 4 | 2162 | 8.9468 | 9.0383 | 5.4232 | 15.9511 |
| decode | metal_paged | 8 | 288 | 0.6984 | 0.9421 | 0.6383 | 4.0891 |
| decode | mlx_fast_dense | 8 | 288 | 0.4500 | 0.8968 | 0.4262 | 9.1338 |
| decode | metal_paged_lazy_h128_b16 | 8 | 288 | 0.5058 | 0.5815 | 0.4659 | 1.6482 |
| decode | mlx_gather_paged_kv | 8 | 288 | 0.9804 | 1.2751 | 0.9452 | 9.2608 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.2664 | 1.8630 | 1.1738 | 8.8105 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.8389 | 1.4637 | 0.6867 | 5.3493 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 288 | 0.6535 | 1.0870 | 0.5232 | 4.8595 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.4840 | 2.2170 | 1.0517 | 6.5873 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.8873 | 2.5490 | 1.2853 | 8.3246 |
| decode | metal_dense_wrapper | 8 | 288 | 1.7830 | 2.5835 | 1.3458 | 10.9958 |
| decode | metal_paged | 8 | 2162 | 2.0700 | 3.0557 | 1.6864 | 9.7864 |
| decode | mlx_fast_dense | 8 | 2162 | 2.0167 | 2.8674 | 1.6334 | 9.4395 |
| decode | metal_paged_lazy_h128_b16 | 8 | 2162 | 2.0719 | 2.7871 | 1.6360 | 6.8850 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 10.4870 | 10.7482 | 5.8926 | 16.6490 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 11.8418 | 11.9202 | 6.9101 | 19.0927 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.9542 | 3.6041 | 1.9173 | 8.5884 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 2162 | 2.3409 | 3.6201 | 1.7551 | 11.7970 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 9.1514 | 9.3705 | 5.8071 | 16.0944 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 12.2770 | 12.8223 | 6.8927 | 19.2247 |
| decode | metal_dense_wrapper | 8 | 2162 | 15.5607 | 15.9078 | 9.9385 | 23.4059 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.9502 | 0.9502 | 0.9502 | 0.9502 |
| prefill | metal_flash_varlen | 1 | 16 | 0.6043 | 0.6043 | 0.6043 | 0.6043 |
| prefill | mlx_fast_dense | 1 | 16 | 0.3541 | 0.3541 | 0.3541 | 0.3541 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3562 | 0.7950 | 0.2840 | 3.9385 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.6156 | 0.8767 | 0.5400 | 4.5930 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.6452 | 0.9542 | 0.5379 | 4.5570 |
