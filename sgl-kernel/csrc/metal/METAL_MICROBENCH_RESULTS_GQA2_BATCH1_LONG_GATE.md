# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:04:19`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3517 | 0.3498 | 0.3282 | 0.3771 |
| decode | mlx_fast_dense | 1 | 288 | 0.2089 | 0.2109 | 0.2038 | 0.2288 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2372 | 0.2387 | 0.2328 | 0.2545 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.2836 | 0.2833 | 0.2608 | 0.3115 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4195 | 0.4186 | 0.3956 | 0.4488 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3010 | 0.3047 | 0.2774 | 0.3868 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3321 | 0.3344 | 0.3103 | 0.4075 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7061 | 0.6987 | 0.6288 | 0.7522 |
| decode | metal_paged | 1 | 2162 | 0.5841 | 0.5909 | 0.5438 | 0.6826 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4027 | 0.4115 | 0.3794 | 0.4797 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.8983 | 0.8972 | 0.8642 | 0.9205 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1287 | 1.1325 | 1.0987 | 1.2010 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6132 | 0.6134 | 0.5465 | 0.6616 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 0.9315 | 0.9377 | 0.9057 | 1.0309 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.1469 | 1.1567 | 1.1278 | 1.2227 |
| decode | metal_dense_wrapper | 1 | 2162 | 3.0198 | 3.0228 | 2.8440 | 3.3366 |
| decode | metal_paged | 4 | 288 | 0.5607 | 0.6356 | 0.4721 | 1.4868 |
| decode | mlx_fast_dense | 4 | 288 | 0.3321 | 0.3400 | 0.3175 | 0.3961 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7801 | 0.7829 | 0.7492 | 0.8673 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9390 | 1.0210 | 0.8915 | 2.3937 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5146 | 0.6556 | 0.5007 | 3.2192 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7923 | 0.7946 | 0.7657 | 0.8697 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.9381 | 0.9379 | 0.9180 | 0.9672 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2295 | 1.2307 | 1.1974 | 1.2867 |
| decode | metal_paged | 4 | 2162 | 1.1476 | 1.1682 | 1.0922 | 1.2923 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9402 | 0.9438 | 0.9140 | 1.0003 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 2.9517 | 2.9796 | 2.8849 | 3.1756 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.7451 | 3.7490 | 3.5635 | 3.9999 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.2920 | 1.3422 | 1.1830 | 1.6476 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.1613 | 3.1762 | 3.0153 | 3.5941 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 3.9777 | 3.9951 | 3.8506 | 4.2771 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.6063 | 5.5952 | 5.3132 | 5.9312 |
| decode | metal_paged | 8 | 288 | 0.7858 | 0.8053 | 0.7462 | 0.9723 |
| decode | mlx_fast_dense | 8 | 288 | 0.5583 | 0.5662 | 0.5069 | 0.6920 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.2150 | 1.2411 | 1.1213 | 1.3623 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.5690 | 1.5450 | 1.3327 | 1.6527 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.9245 | 0.9730 | 0.8253 | 1.4804 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.6209 | 1.6377 | 1.3713 | 2.0898 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.6989 | 1.7568 | 1.6201 | 2.6504 |
| decode | metal_dense_wrapper | 8 | 288 | 2.4779 | 2.5173 | 2.2153 | 3.1075 |
| decode | metal_paged | 8 | 2162 | 1.6965 | 1.7052 | 1.6029 | 1.8236 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5155 | 1.5156 | 1.4041 | 1.6304 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.5498 | 5.5777 | 5.4815 | 5.7192 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.0510 | 7.0546 | 6.7217 | 7.3240 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 1.9550 | 2.0343 | 1.8365 | 2.3547 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 5.6579 | 5.6988 | 5.4495 | 6.4649 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 6.8490 | 6.8196 | 6.6557 | 6.9710 |
| decode | metal_dense_wrapper | 8 | 2162 | 9.9040 | 9.9217 | 9.6292 | 10.3155 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.6357 | 0.6357 | 0.6357 | 0.6357 |
| prefill | metal_flash_varlen | 1 | 16 | 0.4966 | 0.4966 | 0.4966 | 0.4966 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2662 | 0.2662 | 0.2662 | 0.2662 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2714 | 0.2748 | 0.2377 | 0.3344 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 2.0190 | 2.0243 | 1.9418 | 2.1353 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.3046 | 1.3264 | 1.2058 | 1.5050 |
