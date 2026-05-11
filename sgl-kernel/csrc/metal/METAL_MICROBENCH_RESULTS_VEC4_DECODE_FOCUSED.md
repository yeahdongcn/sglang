# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:57:19`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4559 | 0.7359 | 0.3632 | 3.2665 |
| decode | mlx_fast_dense | 1 | 288 | 0.2433 | 0.3271 | 0.2072 | 1.1569 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2709 | 0.2789 | 0.2362 | 0.3994 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.4287 | 0.6270 | 0.3584 | 3.8273 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6854 | 0.9768 | 0.5918 | 5.0480 |
| decode | metal_paged | 1 | 2162 | 0.6576 | 0.9444 | 0.5600 | 4.1747 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4945 | 0.7023 | 0.4191 | 3.5734 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.2000 | 1.6041 | 0.9633 | 4.8565 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.4518 | 3.0610 | 1.0492 | 26.0147 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.8980 | 2.3238 | 1.7333 | 5.1482 |
| decode | metal_paged | 4 | 288 | 0.5343 | 0.8643 | 0.4598 | 3.4455 |
| decode | mlx_fast_dense | 4 | 288 | 0.4063 | 0.6355 | 0.2958 | 3.6645 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7103 | 0.9172 | 0.6103 | 1.9245 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8825 | 1.1142 | 0.7509 | 2.4589 |
| decode | metal_dense_wrapper | 4 | 288 | 1.0504 | 1.3605 | 0.8858 | 3.6650 |
| decode | metal_paged | 4 | 2162 | 1.3214 | 1.7645 | 0.9914 | 3.3265 |
| decode | mlx_fast_dense | 4 | 2162 | 1.3869 | 1.7185 | 0.9684 | 4.3950 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.3341 | 5.2418 | 3.1068 | 11.0905 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 5.1017 | 6.3244 | 3.8469 | 10.7605 |
| decode | metal_dense_wrapper | 4 | 2162 | 7.1906 | 7.6876 | 4.8930 | 9.8973 |
| decode | metal_paged | 8 | 288 | 0.6506 | 0.8020 | 0.5963 | 2.0198 |
| decode | mlx_fast_dense | 8 | 288 | 0.4590 | 0.5403 | 0.4125 | 1.3510 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.2161 | 1.7708 | 0.9862 | 6.3905 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 2.5901 | 4.6592 | 1.5781 | 28.1462 |
| decode | metal_dense_wrapper | 8 | 288 | 1.3179 | 1.9069 | 1.1863 | 5.6827 |
| decode | metal_paged | 8 | 2162 | 2.6250 | 2.9971 | 1.8792 | 5.6530 |
| decode | mlx_fast_dense | 8 | 2162 | 2.3218 | 2.6504 | 1.5353 | 5.8549 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 10.6065 | 10.4800 | 6.4483 | 14.5920 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 11.3747 | 11.0435 | 7.8571 | 15.2824 |
| decode | metal_dense_wrapper | 8 | 2162 | 14.4136 | 15.6540 | 9.8618 | 27.0424 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.4715 | 0.4715 | 0.4715 | 0.4715 |
| prefill | metal_flash_varlen | 1 | 16 | 1.1746 | 1.1746 | 1.1746 | 1.1746 |
| prefill | mlx_fast_dense | 1 | 16 | 1.2369 | 1.2369 | 1.2369 | 1.2369 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2277 | 0.3939 | 0.2149 | 2.5923 |
