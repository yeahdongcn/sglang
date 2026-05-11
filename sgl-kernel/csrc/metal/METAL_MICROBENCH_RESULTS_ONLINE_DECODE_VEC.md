# Metal Attention Microbenchmark Results

Generated: `2026-05-09 16:55:02`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3970 | 0.4735 | 0.3757 | 0.9080 |
| decode | mlx_fast_dense | 1 | 288 | 0.2638 | 0.3628 | 0.2454 | 1.2576 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2836 | 0.3123 | 0.2631 | 0.5116 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3913 | 0.7206 | 0.3300 | 2.5482 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6920 | 0.7094 | 0.6353 | 0.8453 |
| decode | metal_paged | 1 | 2162 | 0.8024 | 1.1123 | 0.6329 | 2.6625 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4811 | 0.4838 | 0.4075 | 0.6389 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.0074 | 1.0280 | 0.9313 | 1.2667 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1795 | 1.2352 | 1.0927 | 1.6026 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.2214 | 2.4071 | 1.7375 | 3.9594 |
| decode | metal_paged | 4 | 288 | 0.5135 | 0.5382 | 0.4703 | 0.6543 |
| decode | mlx_fast_dense | 4 | 288 | 0.3484 | 0.3759 | 0.3159 | 0.6688 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6495 | 0.6533 | 0.6187 | 0.7045 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8207 | 0.9730 | 0.7356 | 1.7636 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9094 | 0.9411 | 0.8505 | 1.1299 |
| decode | metal_paged | 4 | 2162 | 1.2704 | 1.3672 | 1.0693 | 2.1293 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0974 | 1.2731 | 0.8716 | 2.4107 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.5355 | 3.6389 | 3.0180 | 4.8941 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.9587 | 4.2511 | 3.6788 | 5.7948 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.5302 | 5.6090 | 4.9710 | 7.0388 |
| decode | metal_paged | 8 | 288 | 0.6834 | 0.7233 | 0.5932 | 1.0178 |
| decode | mlx_fast_dense | 8 | 288 | 0.5082 | 0.5988 | 0.4483 | 1.5289 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.2119 | 1.3419 | 1.0034 | 2.0637 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3643 | 1.4686 | 1.1773 | 2.0211 |
| decode | metal_dense_wrapper | 8 | 288 | 1.2380 | 1.4435 | 1.1784 | 2.5187 |
| decode | metal_paged | 8 | 2162 | 2.4693 | 2.4468 | 1.8508 | 3.6584 |
| decode | mlx_fast_dense | 8 | 2162 | 1.6859 | 1.9799 | 1.4552 | 4.0188 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.6153 | 6.4520 | 5.5500 | 7.1783 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 9.0240 | 9.0016 | 7.2975 | 11.1621 |
| decode | metal_dense_wrapper | 8 | 2162 | 11.5127 | 11.5301 | 10.0186 | 13.1952 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 12.5532 | 12.2947 | 10.5764 | 13.7545 |
| prefill | metal_flash_varlen | 1 | 256 | 11.5830 | 11.8194 | 10.4179 | 13.4574 |
| prefill | mlx_fast_dense | 1 | 256 | 0.7108 | 0.9247 | 0.5545 | 1.5088 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 194.3755 | 195.5061 | 190.1248 | 202.0182 |
| prefill | metal_flash_varlen | 1 | 1024 | 163.9278 | 166.3918 | 163.4093 | 171.8382 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.6765 | 5.8337 | 5.0245 | 6.8000 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 46.7835 | 47.0129 | 45.2859 | 48.9692 |
| prefill | metal_flash_varlen | 4 | 256 | 43.6479 | 43.7011 | 41.9040 | 45.5513 |
| prefill | mlx_fast_dense | 4 | 256 | 1.5039 | 1.5038 | 1.5035 | 1.5041 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 677.9171 | 679.6207 | 632.2460 | 728.6990 |
| prefill | metal_flash_varlen | 4 | 1024 | 680.0903 | 700.6310 | 669.6797 | 752.1230 |
| prefill | mlx_fast_dense | 4 | 1024 | 21.2081 | 21.1369 | 19.5871 | 22.6155 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2224 | 0.2341 | 0.2047 | 0.2792 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.1790 | 0.1825 | 0.1623 | 0.2316 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.2971 | 0.4952 | 0.2762 | 2.1750 |
