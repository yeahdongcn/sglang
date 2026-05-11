# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:53:45`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=1
prefill_prefix_lens=[]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.7075 | 3.1728 | 0.3453 | 33.4651 |
| decode | mlx_fast_dense | 1 | 288 | 0.6778 | 0.8246 | 0.2099 | 2.0498 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.3436 | 0.8275 | 0.2538 | 5.4634 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3595 | 0.7769 | 0.3155 | 7.4827 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7002 | 1.0812 | 0.6074 | 4.0460 |
| decode | metal_paged | 1 | 2162 | 0.5670 | 0.8366 | 0.5216 | 4.1995 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4389 | 0.6692 | 0.3960 | 4.0045 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 1.1762 | 1.5768 | 0.8875 | 4.4116 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.2814 | 1.7151 | 1.0498 | 5.6032 |
| decode | metal_dense_wrapper | 1 | 2162 | 1.8069 | 2.1668 | 1.7302 | 5.0565 |
| decode | metal_paged | 4 | 288 | 0.5354 | 0.8664 | 0.4660 | 4.8525 |
| decode | mlx_fast_dense | 4 | 288 | 0.3930 | 0.6116 | 0.2820 | 4.7658 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7124 | 0.9678 | 0.6121 | 3.8925 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.7949 | 1.0506 | 0.6896 | 4.0515 |
| decode | metal_dense_wrapper | 4 | 288 | 0.9929 | 1.2521 | 0.8085 | 3.3644 |
| decode | metal_paged | 4 | 2162 | 1.9714 | 2.6631 | 1.3159 | 5.9089 |
| decode | mlx_fast_dense | 4 | 2162 | 1.9551 | 2.4697 | 1.1022 | 5.3539 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.8517 | 5.3705 | 3.0209 | 11.1560 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 5.4972 | 6.3544 | 3.7252 | 9.8937 |
| decode | metal_dense_wrapper | 4 | 2162 | 7.9842 | 10.9389 | 5.2037 | 44.8318 |
| decode | metal_paged | 8 | 288 | 0.7980 | 1.5039 | 0.6248 | 7.1603 |
| decode | mlx_fast_dense | 8 | 288 | 0.8618 | 1.1331 | 0.4678 | 2.4165 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.7275 | 2.7341 | 0.9937 | 6.4023 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 2.6269 | 2.9040 | 1.3218 | 6.8393 |
| decode | metal_dense_wrapper | 8 | 288 | 1.6776 | 1.9466 | 1.1610 | 4.4755 |
| decode | metal_paged | 8 | 2162 | 2.6123 | 3.1088 | 1.7000 | 6.4960 |
| decode | mlx_fast_dense | 8 | 2162 | 2.0937 | 2.7958 | 1.4070 | 7.5386 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 9.9387 | 9.7008 | 6.4689 | 15.7047 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 12.1679 | 11.3049 | 7.4758 | 15.1107 |
| decode | metal_dense_wrapper | 8 | 2162 | 12.1797 | 12.6289 | 9.9315 | 23.3185 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.9912 | 0.9912 | 0.9912 | 0.9912 |
| prefill | metal_flash_varlen | 1 | 16 | 0.3909 | 0.3909 | 0.3909 | 0.3909 |
| prefill | mlx_fast_dense | 1 | 16 | 0.7059 | 0.7059 | 0.7059 | 0.7059 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2360 | 0.4006 | 0.2015 | 2.0281 |
