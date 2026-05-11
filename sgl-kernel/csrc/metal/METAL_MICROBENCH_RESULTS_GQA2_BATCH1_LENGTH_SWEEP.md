# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:08:22`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 768 | 0.4156 | 0.4538 | 0.3700 | 0.9150 |
| decode | mlx_fast_dense | 1 | 768 | 0.2514 | 0.2509 | 0.2377 | 0.2686 |
| decode | mlx_gather_paged_kv | 1 | 768 | 0.4312 | 0.4612 | 0.3994 | 0.9154 |
| decode | mlx_fast_gathered_paged | 1 | 768 | 0.5537 | 0.5858 | 0.5267 | 1.0850 |
| decode | metal_paged_radix_shuffled | 1 | 768 | 0.4552 | 0.5921 | 0.4127 | 2.5065 |
| decode | mlx_gather_radix_paged_kv | 1 | 768 | 0.4697 | 0.5561 | 0.4518 | 1.5749 |
| decode | mlx_fast_radix_gathered_paged | 1 | 768 | 0.5476 | 0.5561 | 0.5253 | 0.6509 |
| decode | metal_dense_wrapper | 1 | 768 | 1.1029 | 1.1323 | 0.9569 | 1.8521 |
| decode | metal_paged | 1 | 1024 | 0.4852 | 0.5003 | 0.4698 | 0.6411 |
| decode | mlx_fast_dense | 1 | 1024 | 0.2965 | 0.3694 | 0.2737 | 1.1821 |
| decode | mlx_gather_paged_kv | 1 | 1024 | 0.5559 | 0.5919 | 0.5300 | 1.0190 |
| decode | mlx_fast_gathered_paged | 1 | 1024 | 0.6568 | 0.6816 | 0.6324 | 0.9874 |
| decode | metal_paged_radix_shuffled | 1 | 1024 | 0.5195 | 0.5505 | 0.4865 | 0.9210 |
| decode | mlx_gather_radix_paged_kv | 1 | 1024 | 0.5909 | 0.6309 | 0.5497 | 1.0903 |
| decode | mlx_fast_radix_gathered_paged | 1 | 1024 | 0.6951 | 0.7995 | 0.6591 | 1.9240 |
| decode | metal_dense_wrapper | 1 | 1024 | 1.6573 | 1.7913 | 1.5596 | 2.9005 |
| decode | metal_paged | 1 | 1536 | 0.6425 | 0.6878 | 0.5808 | 1.1640 |
| decode | mlx_fast_dense | 1 | 1536 | 0.4019 | 0.4294 | 0.3599 | 0.6955 |
| decode | mlx_gather_paged_kv | 1 | 1536 | 0.8790 | 0.9746 | 0.7252 | 2.4689 |
| decode | mlx_fast_gathered_paged | 1 | 1536 | 0.8700 | 0.9935 | 0.8475 | 2.4004 |
| decode | metal_paged_radix_shuffled | 1 | 1536 | 0.6550 | 0.6821 | 0.5925 | 1.2167 |
| decode | mlx_gather_radix_paged_kv | 1 | 1536 | 0.7790 | 0.8680 | 0.7287 | 2.0145 |
| decode | mlx_fast_radix_gathered_paged | 1 | 1536 | 0.9258 | 0.9719 | 0.8988 | 1.4545 |
| decode | metal_dense_wrapper | 1 | 1536 | 2.3930 | 2.4794 | 2.2217 | 3.9087 |
| decode | metal_paged | 1 | 2162 | 0.7819 | 0.8248 | 0.6913 | 1.3063 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5057 | 0.5460 | 0.4647 | 1.0482 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9719 | 1.0688 | 0.9063 | 2.1481 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1707 | 1.2682 | 1.0945 | 2.4599 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6462 | 0.6651 | 0.5507 | 1.0472 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 0.9235 | 1.0007 | 0.8545 | 1.8729 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.0955 | 1.1918 | 1.0604 | 2.2608 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.8828 | 1.9219 | 1.7301 | 2.3720 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.4856 | 0.4856 | 0.4856 | 0.4856 |
| prefill | metal_flash_varlen | 1 | 16 | 0.4797 | 0.4797 | 0.4797 | 0.4797 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2964 | 0.2964 | 0.2964 | 0.2964 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2150 | 0.2377 | 0.1918 | 0.7035 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.4167 | 1.4713 | 1.2352 | 1.8992 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.2709 | 1.5816 | 1.0647 | 6.7224 |
