# Metal Attention Microbenchmark Results

Generated: `2026-05-09 14:09:57`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.6826 | 1.3125 | 0.6094 | 4.4135 |
| decode | mlx_fast_dense | 1 | 288 | 0.2281 | 0.2283 | 0.2084 | 0.2743 |
| decode | metal_dense_wrapper | 1 | 288 | 0.5785 | 0.6257 | 0.5341 | 1.0194 |
| decode | metal_paged | 1 | 2162 | 1.9559 | 2.1269 | 1.8760 | 2.8024 |
| decode | mlx_fast_dense | 1 | 2162 | 0.5092 | 0.8235 | 0.3864 | 3.6269 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.0555 | 2.7536 | 1.7543 | 5.7969 |
| decode | metal_paged | 4 | 288 | 1.1735 | 1.6395 | 1.1137 | 4.6072 |
| decode | mlx_fast_dense | 4 | 288 | 0.3619 | 0.3967 | 0.3154 | 0.7387 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9849 | 1.3079 | 0.9206 | 3.5067 |
| decode | metal_paged | 4 | 2162 | 7.6104 | 8.2297 | 5.8759 | 11.0425 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0222 | 1.3858 | 0.8776 | 4.1043 |
| decode | metal_dense_wrapper | 4 | 2162 | 6.5155 | 6.9135 | 5.4935 | 9.3989 |
| decode | metal_paged | 8 | 288 | 2.1410 | 2.5626 | 1.8274 | 5.3960 |
| decode | mlx_fast_dense | 8 | 288 | 0.5050 | 0.8330 | 0.4404 | 3.2450 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4224 | 1.7589 | 1.1908 | 4.2881 |
| decode | metal_paged | 8 | 2162 | 13.8104 | 13.9199 | 11.4050 | 17.3457 |
| decode | mlx_fast_dense | 8 | 2162 | 1.8764 | 2.2934 | 1.5460 | 4.9557 |
| decode | metal_dense_wrapper | 8 | 2162 | 12.0368 | 11.9515 | 10.0996 | 14.4839 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 20.3216 | 19.8569 | 18.3887 | 20.8605 |
| prefill | metal_flash_varlen | 1 | 256 | 18.5018 | 18.1031 | 16.9441 | 18.8634 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6038 | 0.6463 | 0.5436 | 0.7915 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 317.0544 | 309.2501 | 280.8217 | 329.8743 |
| prefill | metal_flash_varlen | 1 | 1024 | 294.2801 | 290.0278 | 274.4665 | 301.3367 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.5238 | 6.8593 | 5.2537 | 9.8004 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 76.8143 | 72.1443 | 61.3913 | 78.2273 |
| prefill | metal_flash_varlen | 4 | 256 | 62.9315 | 66.9299 | 61.9011 | 75.9571 |
| prefill | mlx_fast_dense | 4 | 256 | 1.9928 | 2.9766 | 1.7301 | 5.2068 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 1058.8418 | 1059.7334 | 1052.0132 | 1068.3451 |
| prefill | metal_flash_varlen | 4 | 1024 | 1031.9004 | 1030.1472 | 1021.2922 | 1037.2490 |
| prefill | mlx_fast_dense | 4 | 1024 | 23.3863 | 22.0603 | 19.0767 | 23.7180 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2691 | 0.3510 | 0.2385 | 0.6419 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2513 | 0.5891 | 0.2318 | 3.2268 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3191 | 0.3669 | 0.2719 | 0.6353 |
