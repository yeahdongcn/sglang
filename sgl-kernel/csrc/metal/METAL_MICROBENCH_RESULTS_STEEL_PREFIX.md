# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:45:37`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=5
prefill_prefix_lens=[256]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3697 | 0.4261 | 0.3582 | 0.6981 |
| decode | mlx_fast_dense | 1 | 288 | 0.2593 | 0.3744 | 0.2298 | 1.4517 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3294 | 0.7935 | 0.2728 | 4.3616 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3219 | 0.4667 | 0.3088 | 1.7231 |
| decode | metal_dense_wrapper | 1 | 288 | 0.8072 | 1.3991 | 0.6622 | 6.1977 |
| decode | metal_paged | 1 | 2162 | 0.5958 | 0.6450 | 0.5379 | 0.9180 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4112 | 0.4934 | 0.3878 | 1.1597 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9180 | 1.3883 | 0.8790 | 4.4505 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1941 | 1.5364 | 1.0635 | 4.2738 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.8375 | 2.5979 | 1.7752 | 5.6002 |
| decode | metal_paged | 4 | 288 | 0.5610 | 0.9921 | 0.5290 | 3.9823 |
| decode | mlx_fast_dense | 4 | 288 | 0.3548 | 0.4055 | 0.3171 | 0.8409 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6660 | 0.7619 | 0.6286 | 1.1668 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7980 | 1.2029 | 0.7608 | 4.6405 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9823 | 1.3447 | 0.9221 | 4.1675 |
| decode | metal_paged | 4 | 2162 | 1.1148 | 1.6091 | 1.0743 | 4.7031 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0393 | 1.4212 | 0.8783 | 4.3725 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.5342 | 3.9528 | 2.9697 | 6.2965 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.4762 | 5.1174 | 3.6525 | 7.6518 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.7797 | 6.8203 | 5.2220 | 8.9594 |
| decode | metal_paged | 8 | 288 | 0.8208 | 0.8921 | 0.7383 | 1.1394 |
| decode | mlx_fast_dense | 8 | 288 | 0.5453 | 0.5676 | 0.4851 | 0.7738 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.1723 | 1.5710 | 1.0798 | 4.8052 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3695 | 2.1037 | 1.2664 | 5.2341 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4815 | 1.8775 | 1.3968 | 4.6266 |
| decode | metal_paged | 8 | 2162 | 1.8548 | 2.2428 | 1.7189 | 5.4339 |
| decode | mlx_fast_dense | 8 | 2162 | 1.7638 | 2.1099 | 1.5202 | 4.8275 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.5718 | 7.7893 | 5.7266 | 10.8727 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 9.5485 | 9.6500 | 6.9955 | 11.9477 |
| decode | metal_dense_wrapper | 8 | 2162 | 13.4797 | 13.4376 | 10.1177 | 21.6936 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 0.9210 | 1.7385 | 0.6887 | 4.3129 |
| prefill | metal_flash_varlen | 1 | 256 | 0.7497 | 0.7874 | 0.7324 | 0.9038 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5278 | 0.5368 | 0.5121 | 0.5645 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 1.1727 | 1.2947 | 1.0584 | 1.6989 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 1.1318 | 1.6966 | 0.8588 | 4.3410 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 4.0778 | 4.5700 | 3.0907 | 6.5720 |
| prefill | metal_flash_varlen | 1 | 1024 | 3.1073 | 3.9303 | 3.0527 | 6.2212 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.8621 | 6.7999 | 5.4801 | 8.8151 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 5.4014 | 5.8700 | 4.8381 | 8.0831 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 6.7539 | 7.9413 | 6.3970 | 9.9476 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 1.2958 | 2.1012 | 1.1882 | 4.5884 |
| prefill | metal_flash_varlen | 4 | 256 | 1.2975 | 1.3032 | 1.2720 | 1.3394 |
| prefill | mlx_fast_dense | 4 | 256 | 1.4787 | 1.5457 | 1.4667 | 1.8260 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 3.8752 | 4.5080 | 3.0412 | 6.4935 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 2.7093 | 3.3610 | 2.6042 | 5.9732 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 15.0380 | 14.3188 | 11.2292 | 15.6343 |
| prefill | metal_flash_varlen | 4 | 1024 | 14.5590 | 14.1717 | 11.8568 | 15.2529 |
| prefill | mlx_fast_dense | 4 | 1024 | 26.5875 | 25.7831 | 23.4606 | 27.5930 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 20.8702 | 20.8893 | 17.6983 | 23.7275 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 26.9836 | 26.7398 | 23.8956 | 28.0520 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3782 | 0.7293 | 0.3553 | 3.9212 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3480 | 0.3626 | 0.3337 | 0.4837 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.4470 | 0.8886 | 0.3781 | 4.0704 |
