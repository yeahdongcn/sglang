# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:48:23`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=5
prefill_prefix_lens=[256]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3828 | 0.9549 | 0.3610 | 5.7722 |
| decode | mlx_fast_dense | 1 | 288 | 0.2284 | 0.2296 | 0.2182 | 0.2427 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3018 | 0.8823 | 0.2616 | 5.9510 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3746 | 0.4813 | 0.3169 | 0.9397 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7713 | 0.8827 | 0.6797 | 2.0607 |
| decode | metal_paged | 1 | 2162 | 0.6304 | 0.6451 | 0.5703 | 0.8637 |
| decode | mlx_fast_dense | 1 | 2162 | 0.6621 | 0.9522 | 0.4705 | 3.5385 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.3120 | 1.5363 | 0.9683 | 3.8174 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2718 | 1.7051 | 1.1877 | 4.4996 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.9002 | 2.3397 | 1.7395 | 4.1576 |
| decode | metal_paged | 4 | 288 | 0.5254 | 0.8727 | 0.4760 | 3.6069 |
| decode | mlx_fast_dense | 4 | 288 | 0.3408 | 0.3467 | 0.3152 | 0.3965 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6664 | 0.9801 | 0.6341 | 3.4533 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7607 | 0.8737 | 0.7376 | 1.6384 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9082 | 0.9970 | 0.8625 | 1.6265 |
| decode | metal_paged | 4 | 2162 | 1.4734 | 1.7385 | 1.2066 | 4.0212 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9425 | 1.0585 | 0.8596 | 1.6337 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.8502 | 4.2084 | 2.9900 | 6.3960 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.2455 | 4.7725 | 3.8311 | 6.5890 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.5087 | 5.9779 | 5.1044 | 8.6769 |
| decode | metal_paged | 8 | 288 | 0.7516 | 0.7657 | 0.7087 | 0.8419 |
| decode | mlx_fast_dense | 8 | 288 | 0.6001 | 0.7893 | 0.5099 | 1.4797 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0844 | 1.4625 | 1.0139 | 3.7405 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3428 | 1.7390 | 1.2577 | 4.1451 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4401 | 2.0431 | 1.3548 | 4.6060 |
| decode | metal_paged | 8 | 2162 | 1.8807 | 2.3405 | 1.6460 | 5.2881 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5173 | 1.9587 | 1.4333 | 5.2265 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.9244 | 6.9196 | 5.4533 | 8.6408 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 9.0116 | 9.1172 | 7.3107 | 11.3816 |
| decode | metal_dense_wrapper | 8 | 2162 | 11.9295 | 11.8357 | 9.6477 | 14.2460 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 0.7227 | 0.7232 | 0.6966 | 0.7558 |
| prefill | metal_flash_varlen | 1 | 256 | 0.7666 | 0.7971 | 0.7488 | 0.8744 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5830 | 0.5913 | 0.5711 | 0.6153 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 2.0835 | 2.1620 | 1.4603 | 3.4556 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 0.8921 | 0.8921 | 0.8745 | 0.9115 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 3.4594 | 3.9982 | 3.3005 | 5.8975 |
| prefill | metal_flash_varlen | 1 | 1024 | 3.5470 | 3.8856 | 3.2646 | 5.3681 |
| prefill | mlx_fast_dense | 1 | 1024 | 6.2257 | 6.5714 | 5.1816 | 8.4972 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 5.5780 | 6.3211 | 4.9003 | 8.2500 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 7.0027 | 7.8492 | 6.2797 | 10.3269 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 1.3835 | 2.0941 | 1.3349 | 4.2380 |
| prefill | metal_flash_varlen | 4 | 256 | 1.3029 | 1.4739 | 1.2775 | 1.8947 |
| prefill | mlx_fast_dense | 4 | 256 | 1.4957 | 1.6469 | 1.4934 | 2.2225 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 3.5777 | 3.8249 | 3.1067 | 5.5295 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 3.3786 | 3.7116 | 2.6593 | 5.5218 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 11.3395 | 12.3182 | 11.1961 | 13.9748 |
| prefill | metal_flash_varlen | 4 | 1024 | 12.0345 | 12.7762 | 10.9587 | 15.5294 |
| prefill | mlx_fast_dense | 4 | 1024 | 29.8142 | 28.6248 | 24.7270 | 33.2060 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 22.7430 | 23.1897 | 21.6288 | 25.4108 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 34.2660 | 34.1581 | 33.5491 | 34.4834 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2414 | 0.3430 | 0.2132 | 1.2480 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2331 | 0.2349 | 0.2126 | 0.2695 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3082 | 0.6766 | 0.2734 | 3.3396 |
