# Metal Attention Microbenchmark Results

Generated: `2026-05-09 16:39:00`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.6856 | 0.7131 | 0.6408 | 0.8709 |
| decode | mlx_fast_dense | 1 | 288 | 0.2114 | 0.2107 | 0.2024 | 0.2225 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2517 | 0.2541 | 0.2403 | 0.2686 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.4099 | 0.7090 | 0.2983 | 3.3124 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7075 | 0.7346 | 0.6953 | 0.8919 |
| decode | metal_paged | 1 | 2162 | 2.2729 | 2.4751 | 1.8935 | 3.7071 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4245 | 0.5000 | 0.4116 | 1.1272 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9185 | 1.0068 | 0.8724 | 1.6584 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1029 | 1.1599 | 1.0386 | 1.6202 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.8343 | 1.8593 | 1.7152 | 2.1899 |
| decode | metal_paged | 4 | 288 | 1.1817 | 1.3939 | 1.0855 | 3.0798 |
| decode | mlx_fast_dense | 4 | 288 | 0.3294 | 0.3359 | 0.3132 | 0.4214 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.6590 | 0.7215 | 0.6392 | 1.0236 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7908 | 0.8365 | 0.7086 | 1.0920 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9585 | 1.1800 | 0.9002 | 2.1508 |
| decode | metal_paged | 4 | 2162 | 5.9141 | 5.9580 | 5.5540 | 6.6490 |
| decode | mlx_fast_dense | 4 | 2162 | 0.8849 | 1.0123 | 0.8510 | 1.6862 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.6469 | 3.6230 | 3.0974 | 4.4749 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.6205 | 4.6494 | 4.0639 | 5.6224 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.3545 | 5.5471 | 5.0479 | 6.4460 |
| decode | metal_paged | 8 | 288 | 1.9404 | 2.0260 | 1.8205 | 2.5372 |
| decode | mlx_fast_dense | 8 | 288 | 0.4453 | 0.4449 | 0.4277 | 0.4771 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.0515 | 1.2552 | 0.9537 | 2.1470 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.2047 | 1.4194 | 1.1494 | 2.1138 |
| decode | metal_dense_wrapper | 8 | 288 | 1.2497 | 1.4360 | 1.1923 | 2.3945 |
| decode | metal_paged | 8 | 2162 | 11.5096 | 11.7172 | 10.6464 | 13.0326 |
| decode | mlx_fast_dense | 8 | 2162 | 1.8197 | 1.9865 | 1.5464 | 3.1572 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.4553 | 6.6801 | 5.6034 | 8.4419 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 8.5905 | 8.1272 | 6.9192 | 8.8300 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.2453 | 10.3597 | 9.4162 | 12.1858 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 11.2340 | 11.1450 | 10.9119 | 11.2890 |
| prefill | metal_flash_varlen | 1 | 256 | 11.0094 | 10.8854 | 10.0204 | 11.6263 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5572 | 0.5676 | 0.5221 | 0.6236 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 182.8721 | 180.3238 | 171.5823 | 186.5170 |
| prefill | metal_flash_varlen | 1 | 1024 | 167.1255 | 164.5121 | 159.0719 | 167.3387 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.4948 | 5.5569 | 4.9235 | 6.2523 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 44.5869 | 44.5796 | 44.2691 | 44.8827 |
| prefill | metal_flash_varlen | 4 | 256 | 43.4552 | 43.1585 | 41.0236 | 44.9967 |
| prefill | mlx_fast_dense | 4 | 256 | 1.5983 | 1.5654 | 1.4708 | 1.6272 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 659.8294 | 683.8538 | 650.8426 | 740.8894 |
| prefill | metal_flash_varlen | 4 | 1024 | 653.0847 | 658.6130 | 650.8269 | 671.9273 |
| prefill | mlx_fast_dense | 4 | 1024 | 20.8479 | 20.8008 | 20.2975 | 21.2570 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2545 | 0.2491 | 0.1947 | 0.2877 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2327 | 0.2549 | 0.2039 | 0.3818 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3018 | 0.4563 | 0.2767 | 1.3935 |
