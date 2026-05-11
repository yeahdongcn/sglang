# Metal Attention Microbenchmark Results

Generated: `2026-05-09 14:16:48`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.7411 | 0.7919 | 0.6655 | 1.0808 |
| decode | mlx_fast_dense | 1 | 288 | 0.2169 | 0.2205 | 0.2115 | 0.2330 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2836 | 0.2782 | 0.2473 | 0.3078 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.4312 | 0.4605 | 0.3055 | 0.7222 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7687 | 0.8410 | 0.7156 | 1.2055 |
| decode | metal_paged | 1 | 2162 | 3.0007 | 3.1879 | 2.5572 | 4.3181 |
| decode | mlx_fast_dense | 1 | 2162 | 0.6701 | 0.8026 | 0.5328 | 1.7600 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.5663 | 1.7917 | 1.1752 | 3.8620 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 2.2685 | 2.4790 | 1.6478 | 3.6813 |
| decode | metal_dense_wrapper | 1 | 2162 | 3.0861 | 4.5380 | 2.5101 | 13.0835 |
| decode | metal_paged | 4 | 288 | 2.1855 | 2.5125 | 1.5282 | 4.0599 |
| decode | mlx_fast_dense | 4 | 288 | 0.3885 | 0.7976 | 0.3434 | 4.3541 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.8245 | 1.0002 | 0.6848 | 1.9360 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9301 | 1.1364 | 0.8832 | 1.9828 |
| decode | metal_dense_wrapper | 4 | 288 | 1.3591 | 1.5690 | 1.1506 | 3.1075 |
| decode | metal_paged | 4 | 2162 | 6.2634 | 6.4666 | 5.7483 | 8.3806 |
| decode | mlx_fast_dense | 4 | 2162 | 1.3139 | 1.5080 | 0.9176 | 3.2373 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.4851 | 3.7079 | 2.9463 | 5.2297 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.8165 | 4.7944 | 3.9690 | 5.6802 |
| decode | metal_dense_wrapper | 4 | 2162 | 6.1898 | 6.4539 | 5.5443 | 9.5249 |
| decode | metal_paged | 8 | 288 | 1.9045 | 2.1136 | 1.8350 | 3.2977 |
| decode | mlx_fast_dense | 8 | 288 | 0.5986 | 0.7721 | 0.4705 | 1.5315 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0608 | 1.2340 | 0.9414 | 2.0600 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.4402 | 1.4894 | 1.1452 | 2.0224 |
| decode | metal_dense_wrapper | 8 | 288 | 1.5184 | 1.5131 | 1.3921 | 1.7648 |
| decode | metal_paged | 8 | 2162 | 11.8051 | 11.8610 | 11.3091 | 12.3252 |
| decode | mlx_fast_dense | 8 | 2162 | 1.6704 | 1.6950 | 1.5340 | 2.0114 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.1879 | 6.1980 | 5.7693 | 6.9350 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.9172 | 7.9329 | 7.1208 | 9.0470 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.3953 | 10.4245 | 9.8902 | 11.1387 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 13.1113 | 12.4997 | 11.1283 | 13.2597 |
| prefill | metal_flash_varlen | 1 | 256 | 10.9805 | 11.2098 | 10.7314 | 11.9175 |
| prefill | mlx_fast_dense | 1 | 256 | 0.7704 | 0.8552 | 0.5995 | 1.1956 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 177.7674 | 178.7152 | 161.4573 | 196.9208 |
| prefill | metal_flash_varlen | 1 | 1024 | 161.4870 | 162.9250 | 159.6795 | 167.6086 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.2793 | 5.4444 | 5.0977 | 5.9562 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 42.6900 | 42.8964 | 41.6970 | 44.3023 |
| prefill | metal_flash_varlen | 4 | 256 | 41.9280 | 41.9375 | 41.4992 | 42.3852 |
| prefill | mlx_fast_dense | 4 | 256 | 2.3744 | 2.8232 | 2.1956 | 3.8996 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 667.4348 | 665.5676 | 660.7128 | 668.5551 |
| prefill | metal_flash_varlen | 4 | 1024 | 699.5207 | 692.9690 | 671.2840 | 708.1024 |
| prefill | mlx_fast_dense | 4 | 1024 | 21.3772 | 21.1651 | 19.9149 | 22.2032 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4769 | 0.5326 | 0.2326 | 0.8882 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2006 | 0.2817 | 0.1714 | 0.5642 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3133 | 0.4061 | 0.2378 | 0.8583 |
