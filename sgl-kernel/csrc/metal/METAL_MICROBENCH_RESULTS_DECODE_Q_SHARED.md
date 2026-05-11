# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:46:00`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4087 | 0.9686 | 0.3613 | 5.5950 |
| decode | mlx_fast_dense | 1 | 288 | 0.2417 | 0.3428 | 0.1999 | 1.6214 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3494 | 0.3826 | 0.2783 | 0.8699 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3820 | 0.3908 | 0.3272 | 0.5058 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4334 | 0.7190 | 0.3848 | 5.7819 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3213 | 0.5641 | 0.2761 | 4.9943 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3336 | 0.6385 | 0.2974 | 4.9379 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7050 | 1.0488 | 0.5766 | 4.8711 |
| decode | metal_paged | 1 | 2162 | 0.6473 | 0.9223 | 0.5948 | 4.0628 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4382 | 0.6777 | 0.4135 | 4.0146 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.1594 | 1.5672 | 0.9045 | 4.3027 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2385 | 1.6880 | 1.1196 | 4.4081 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6757 | 0.9668 | 0.6148 | 4.2618 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.1728 | 1.5870 | 0.9533 | 5.1988 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.3827 | 1.7889 | 1.1553 | 5.6434 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.0931 | 2.6660 | 1.8620 | 6.3151 |
| decode | metal_paged | 4 | 288 | 0.5864 | 0.6663 | 0.5029 | 1.4579 |
| decode | mlx_fast_dense | 4 | 288 | 0.3605 | 0.6904 | 0.3372 | 4.4367 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7017 | 1.0062 | 0.6333 | 5.8374 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8483 | 1.1119 | 0.7858 | 4.5527 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5825 | 0.8258 | 0.5280 | 4.5453 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7234 | 0.9874 | 0.6585 | 4.1011 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.8577 | 1.1502 | 0.7859 | 3.9734 |
| decode | metal_dense_wrapper | 4 | 288 | 1.0362 | 1.4119 | 0.9514 | 4.4146 |
| decode | metal_paged | 4 | 2162 | 1.3122 | 1.9478 | 1.1136 | 5.6797 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0204 | 1.5430 | 0.8925 | 6.4109 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.8414 | 4.5148 | 2.9238 | 8.7116 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 5.1687 | 5.7126 | 3.6937 | 9.4556 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.4474 | 1.8447 | 1.1458 | 6.4115 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.7572 | 4.8379 | 3.0535 | 9.0639 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 5.0036 | 6.3028 | 3.9417 | 11.4336 |
| decode | metal_dense_wrapper | 4 | 2162 | 6.3367 | 6.9403 | 4.9478 | 10.4184 |
| decode | metal_paged | 8 | 288 | 0.7712 | 1.2327 | 0.6384 | 4.3678 |
| decode | mlx_fast_dense | 8 | 288 | 0.4922 | 0.7625 | 0.4565 | 3.7266 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.3153 | 1.6056 | 1.0128 | 4.4869 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.5418 | 2.0686 | 1.2163 | 5.6708 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.7584 | 1.0969 | 0.6724 | 5.0125 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.3044 | 1.7518 | 1.0134 | 5.2013 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.3482 | 1.8533 | 1.2350 | 5.4717 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4579 | 1.8722 | 1.3286 | 5.2238 |
| decode | metal_paged | 8 | 2162 | 2.1160 | 2.5490 | 1.6748 | 5.5017 |
| decode | mlx_fast_dense | 8 | 2162 | 1.6015 | 2.5207 | 1.4113 | 8.2185 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 8.8589 | 8.6213 | 5.8916 | 12.7136 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 11.6920 | 10.7998 | 7.0973 | 13.7765 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.1933 | 2.6849 | 1.8481 | 5.9779 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 9.2869 | 8.8914 | 6.4815 | 11.3963 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 10.8411 | 10.6858 | 6.9380 | 13.6954 |
| decode | metal_dense_wrapper | 8 | 2162 | 13.2273 | 12.6396 | 9.9441 | 14.8955 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.5224 | 0.5224 | 0.5224 | 0.5224 |
| prefill | metal_flash_varlen | 1 | 16 | 0.5024 | 0.5024 | 0.5024 | 0.5024 |
| prefill | mlx_fast_dense | 1 | 16 | 0.4888 | 0.4888 | 0.4888 | 0.4888 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2474 | 0.2553 | 0.2317 | 0.3831 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.6606 | 2.1768 | 1.3835 | 5.0280 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.1670 | 1.5938 | 1.0416 | 4.8136 |
