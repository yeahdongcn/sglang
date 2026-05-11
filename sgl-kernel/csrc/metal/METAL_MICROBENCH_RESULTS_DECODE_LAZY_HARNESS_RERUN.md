# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:13:02`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=1
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4825 | 0.6558 | 0.3923 | 2.8972 |
| decode | mlx_fast_dense | 1 | 288 | 0.2705 | 0.4685 | 0.2293 | 3.8433 |
| decode | metal_paged_lazy_h128_b16 | 1 | 288 | 0.3083 | 0.5827 | 0.2382 | 3.4741 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3056 | 0.3821 | 0.2798 | 1.7555 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3376 | 0.3645 | 0.2927 | 0.8548 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4590 | 0.9549 | 0.3957 | 8.3735 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 288 | 0.2789 | 0.2930 | 0.2590 | 0.5892 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3337 | 0.4598 | 0.2865 | 1.2984 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3732 | 0.7751 | 0.3103 | 3.4560 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6709 | 1.0338 | 0.5778 | 3.8856 |
| decode | metal_paged | 1 | 2162 | 0.6439 | 0.9998 | 0.6086 | 3.9145 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5135 | 0.8120 | 0.4323 | 4.2125 |
| decode | metal_paged_lazy_h128_b16 | 1 | 2162 | 0.5061 | 0.8858 | 0.4446 | 4.6313 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.0922 | 1.7202 | 0.9367 | 5.2587 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.3677 | 1.8716 | 1.1765 | 4.7553 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 1.1115 | 1.6225 | 0.7308 | 4.2778 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 2162 | 0.5869 | 0.8320 | 0.5142 | 4.0740 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.2945 | 2.1081 | 1.0096 | 9.9740 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.4087 | 1.8897 | 1.2923 | 7.6593 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.2594 | 2.7894 | 2.0215 | 8.9258 |
| decode | metal_paged | 4 | 288 | 0.7163 | 0.8165 | 0.6694 | 2.2411 |
| decode | mlx_fast_dense | 4 | 288 | 0.4826 | 0.9490 | 0.3584 | 3.6920 |
| decode | metal_paged_lazy_h128_b16 | 4 | 288 | 0.4546 | 0.8734 | 0.4047 | 7.5338 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7805 | 1.0706 | 0.6857 | 3.7015 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 1.1270 | 1.5615 | 0.8211 | 8.2073 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.7725 | 1.0052 | 0.5842 | 3.8735 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 288 | 0.6723 | 0.9448 | 0.4333 | 3.8145 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7823 | 1.1075 | 0.6897 | 3.4983 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 1.5341 | 2.1517 | 0.9228 | 5.1879 |
| decode | metal_dense_wrapper | 4 | 288 | 1.3443 | 2.1009 | 0.9937 | 10.2120 |
| decode | metal_paged | 4 | 2162 | 1.6701 | 2.3040 | 1.1801 | 10.2994 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1973 | 2.0465 | 0.9374 | 5.6695 |
| decode | metal_paged_lazy_h128_b16 | 4 | 2162 | 1.5704 | 2.3133 | 1.0640 | 5.3811 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 5.8479 | 6.1846 | 3.3532 | 12.9191 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 6.3059 | 6.7008 | 3.9268 | 13.1868 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.4907 | 2.3368 | 1.2020 | 8.0520 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 4 | 2162 | 1.1326 | 1.5845 | 0.9742 | 8.1261 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 4.2745 | 5.0802 | 3.1368 | 10.8399 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 6.8931 | 6.9391 | 4.0968 | 11.1909 |
| decode | metal_dense_wrapper | 4 | 2162 | 8.2846 | 8.8087 | 5.7113 | 12.7516 |
| decode | metal_paged | 8 | 288 | 0.7919 | 1.3378 | 0.7062 | 5.2517 |
| decode | mlx_fast_dense | 8 | 288 | 0.7075 | 0.9458 | 0.5727 | 2.3074 |
| decode | metal_paged_lazy_h128_b16 | 8 | 288 | 0.6434 | 1.1276 | 0.5404 | 6.6994 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.5202 | 2.5736 | 1.1740 | 9.4325 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 2.0654 | 2.8763 | 1.2943 | 8.2355 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.8392 | 1.1349 | 0.7418 | 5.6011 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 288 | 0.9735 | 1.4569 | 0.6534 | 8.4125 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 2.0394 | 2.8331 | 1.1386 | 7.9756 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 2.8297 | 3.3746 | 1.5255 | 7.5549 |
| decode | metal_dense_wrapper | 8 | 288 | 2.5904 | 3.8382 | 1.4752 | 11.9645 |
| decode | metal_paged | 8 | 2162 | 2.9060 | 3.7284 | 1.8181 | 11.0718 |
| decode | mlx_fast_dense | 8 | 2162 | 3.0209 | 3.6936 | 1.6641 | 8.9791 |
| decode | metal_paged_lazy_h128_b16 | 8 | 2162 | 2.2236 | 3.2229 | 1.6427 | 7.5568 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 10.7953 | 10.8961 | 6.7088 | 15.6860 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 12.4818 | 13.5470 | 8.7214 | 22.0882 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 3.5881 | 4.3278 | 2.0534 | 8.7209 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 8 | 2162 | 2.7340 | 3.5613 | 1.6756 | 6.8543 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 11.2789 | 11.5783 | 6.1548 | 18.0790 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 11.8835 | 12.6815 | 6.9414 | 21.3684 |
| decode | metal_dense_wrapper | 8 | 2162 | 16.3579 | 15.7359 | 8.6787 | 21.1306 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.7952 | 0.7952 | 0.7952 | 0.7952 |
| prefill | metal_flash_varlen | 1 | 16 | 0.8810 | 0.8810 | 0.8810 | 0.8810 |
| prefill | mlx_fast_dense | 1 | 16 | 0.8223 | 0.8223 | 0.8223 | 0.8223 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4383 | 0.5984 | 0.3309 | 1.9123 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 1 | 0.7280 | 0.8819 | 0.6310 | 1.7362 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 1 | 0.6924 | 1.0445 | 0.6186 | 2.8972 |
