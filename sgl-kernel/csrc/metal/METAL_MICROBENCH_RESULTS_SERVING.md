# Metal Attention Microbenchmark Results

Generated: `2026-05-09 14:03:09`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.7286 | 0.7622 | 0.6894 | 1.0860 |
| decode | mlx_fast_dense | 1 | 288 | 0.2225 | 0.2579 | 0.2085 | 0.4264 |
| decode | metal_dense_wrapper | 1 | 288 | 0.8247 | 1.1093 | 0.6774 | 2.6206 |
| decode | metal_paged | 1 | 2162 | 29.5721 | 29.6463 | 28.9960 | 30.4656 |
| decode | mlx_fast_dense | 1 | 2162 | 0.6409 | 0.7057 | 0.5480 | 1.1079 |
| decode | metal_dense_wrapper | 1 | 2162 | 29.1571 | 29.2091 | 28.6008 | 30.2704 |
| decode | metal_paged | 4 | 288 | 1.1683 | 1.2967 | 0.9487 | 2.2031 |
| decode | mlx_fast_dense | 4 | 288 | 0.4029 | 0.5183 | 0.3389 | 1.1687 |
| decode | metal_dense_wrapper | 4 | 288 | 1.0795 | 1.2292 | 1.0065 | 1.8422 |
| decode | metal_paged | 4 | 2162 | 34.1278 | 34.3718 | 33.8649 | 35.9531 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0635 | 1.1813 | 0.9547 | 2.3259 |
| decode | metal_dense_wrapper | 4 | 2162 | 34.5064 | 38.5202 | 34.1349 | 66.7184 |
| decode | metal_paged | 8 | 288 | 2.3498 | 2.3307 | 1.7075 | 2.7908 |
| decode | mlx_fast_dense | 8 | 288 | 2.1970 | 2.3041 | 0.8312 | 5.1742 |
| decode | metal_dense_wrapper | 8 | 288 | 3.1404 | 4.6753 | 2.0704 | 10.1002 |
| decode | metal_paged | 8 | 2162 | 74.4134 | 81.1956 | 72.1798 | 99.7486 |
| decode | mlx_fast_dense | 8 | 2162 | 2.0070 | 2.2376 | 1.4138 | 3.9262 |
| decode | metal_dense_wrapper | 8 | 2162 | 74.4911 | 77.5026 | 69.7300 | 100.3486 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 14.7883 | 15.0631 | 14.4050 | 15.9960 |
| prefill | metal_flash_varlen | 1 | 256 | 14.3609 | 14.5315 | 13.4585 | 15.7751 |
| prefill | mlx_fast_dense | 1 | 256 | 1.3340 | 1.2209 | 0.7338 | 1.5948 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 190.1761 | 191.2374 | 189.1033 | 194.4328 |
| prefill | metal_flash_varlen | 1 | 1024 | 214.1660 | 210.8895 | 201.9934 | 216.5091 |
| prefill | mlx_fast_dense | 1 | 1024 | 6.7911 | 6.5155 | 5.8191 | 6.9363 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 42.3934 | 43.4372 | 41.7488 | 46.1692 |
| prefill | metal_flash_varlen | 4 | 256 | 50.6497 | 50.6817 | 43.9635 | 57.4319 |
| prefill | mlx_fast_dense | 4 | 256 | 2.0329 | 1.9873 | 1.6544 | 2.2745 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 787.1811 | 773.4364 | 742.8589 | 790.2691 |
| prefill | metal_flash_varlen | 4 | 1024 | 676.6372 | 670.0709 | 652.6717 | 680.9040 |
| prefill | mlx_fast_dense | 4 | 1024 | 21.3557 | 21.3449 | 19.3463 | 23.3328 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2224 | 0.2221 | 0.1832 | 0.2488 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.1834 | 0.1934 | 0.1775 | 0.2347 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3064 | 0.3361 | 0.2754 | 0.5107 |
