# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:38:23`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3908 | 0.9087 | 0.3373 | 7.2068 |
| decode | mlx_fast_dense | 1 | 288 | 0.2667 | 0.5958 | 0.1980 | 6.1634 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3095 | 0.7650 | 0.2716 | 6.8287 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3306 | 0.6719 | 0.2965 | 3.2446 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4367 | 0.8582 | 0.3908 | 7.4175 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3240 | 0.6162 | 0.2591 | 3.4953 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3080 | 0.3768 | 0.2828 | 1.2880 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6845 | 1.0088 | 0.5901 | 5.2793 |
| decode | metal_paged | 1 | 2162 | 0.6183 | 0.9498 | 0.5564 | 6.0742 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4331 | 0.6816 | 0.3928 | 4.6648 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.2204 | 1.6413 | 0.8965 | 6.2243 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2852 | 1.7682 | 1.0765 | 5.2926 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6732 | 1.0038 | 0.6034 | 4.7348 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.1001 | 1.4448 | 0.8945 | 4.7988 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.4269 | 1.7825 | 1.0894 | 4.7957 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.1016 | 2.7538 | 1.7485 | 6.5081 |
| decode | metal_paged | 4 | 288 | 0.4969 | 0.7722 | 0.4622 | 4.4530 |
| decode | mlx_fast_dense | 4 | 288 | 0.3470 | 0.4076 | 0.3089 | 1.1567 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6791 | 0.9242 | 0.6114 | 4.6970 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8419 | 1.3454 | 0.7511 | 4.6559 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5624 | 0.8787 | 0.4957 | 5.0990 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.6966 | 1.0018 | 0.6492 | 4.5780 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.8472 | 1.1690 | 0.7384 | 4.3128 |
| decode | metal_dense_wrapper | 4 | 288 | 1.1226 | 1.5911 | 0.9510 | 4.6600 |
| decode | metal_paged | 4 | 2162 | 1.2338 | 1.8080 | 1.0510 | 6.3053 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0815 | 1.5803 | 0.8896 | 5.2300 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.8701 | 4.6815 | 2.9272 | 11.8081 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 5.2865 | 6.0602 | 3.6456 | 10.3620 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.3757 | 1.8835 | 1.1133 | 5.6502 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 4.1729 | 4.9015 | 3.0400 | 8.3527 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 4.5505 | 4.9833 | 3.6331 | 8.5415 |
| decode | metal_dense_wrapper | 4 | 2162 | 6.9676 | 7.5391 | 5.1120 | 12.8115 |
| decode | metal_paged | 8 | 288 | 0.6901 | 1.0778 | 0.6538 | 6.1585 |
| decode | mlx_fast_dense | 8 | 288 | 0.5588 | 0.8962 | 0.4747 | 4.2434 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.3155 | 1.8182 | 0.9280 | 5.2981 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.3674 | 1.4853 | 1.1633 | 2.2623 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.6842 | 0.9604 | 0.6143 | 4.7901 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.2878 | 1.8000 | 0.9731 | 5.4474 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.7854 | 2.0673 | 1.1879 | 4.9470 |
| decode | metal_dense_wrapper | 8 | 288 | 1.6200 | 2.3048 | 1.1962 | 5.6718 |
| decode | metal_paged | 8 | 2162 | 1.9925 | 2.7500 | 1.7365 | 5.8986 |
| decode | mlx_fast_dense | 8 | 2162 | 1.9349 | 2.7262 | 1.5396 | 5.4060 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 8.2886 | 10.0167 | 5.6644 | 28.5828 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 11.8305 | 11.1334 | 6.9173 | 14.6480 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 2.1438 | 2.8174 | 1.6688 | 7.5543 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 8.4603 | 9.2926 | 6.4914 | 12.6551 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 11.6977 | 11.6592 | 8.0998 | 17.3353 |
| decode | metal_dense_wrapper | 8 | 2162 | 14.7949 | 14.0035 | 9.3308 | 19.6408 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.8675 | 0.8675 | 0.8675 | 0.8675 |
| prefill | metal_flash_varlen | 1 | 16 | 0.4087 | 0.4087 | 0.4087 | 0.4087 |
| prefill | mlx_fast_dense | 1 | 16 | 0.5833 | 0.5833 | 0.5833 | 0.5833 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2149 | 0.2506 | 0.1819 | 0.5606 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.3313 | 1.9163 | 1.2190 | 6.2918 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.2237 | 2.8197 | 0.9999 | 12.8385 |
