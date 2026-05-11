# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:21:37`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=2, decode/scatter iters=5, prefill iters=2
prefill_prefix_lens=[256]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 2162 | 0.6993 | 0.6847 | 0.5993 | 0.7187 |
| decode | mlx_fast_dense | 1 | 2162 | 0.3980 | 0.4005 | 0.3904 | 0.4128 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9370 | 0.9375 | 0.9252 | 0.9553 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1718 | 1.1958 | 1.1305 | 1.2742 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.5002 | 2.4242 | 2.1535 | 2.5122 |
| decode | metal_paged | 4 | 2162 | 1.1862 | 1.2095 | 1.1519 | 1.2680 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9824 | 0.9777 | 0.9452 | 0.9989 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.3382 | 3.3255 | 3.0083 | 3.5523 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.6781 | 3.6768 | 3.6147 | 3.7635 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.0432 | 5.0456 | 5.0264 | 5.0622 |
| decode | metal_paged | 8 | 2162 | 2.2597 | 2.2522 | 2.0030 | 2.5408 |
| decode | mlx_fast_dense | 8 | 2162 | 1.6525 | 1.6198 | 1.5449 | 1.6727 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.6612 | 5.6423 | 5.4250 | 5.8307 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 6.7778 | 6.7878 | 6.7684 | 6.8216 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.2585 | 10.3115 | 10.1507 | 10.5036 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 35.7540 | 35.7540 | 35.5000 | 36.0079 |
| prefill | metal_flash_varlen | 1 | 1024 | 27.4870 | 27.4870 | 27.4200 | 27.5539 |
| prefill | mlx_fast_dense | 1 | 1024 | 4.9867 | 4.9867 | 4.9496 | 5.0238 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 54.3413 | 54.3413 | 53.3404 | 55.3421 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 6.2538 | 6.2538 | 6.1554 | 6.3522 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 141.4250 | 141.4250 | 141.0045 | 141.8455 |
| prefill | metal_flash_varlen | 4 | 1024 | 109.1792 | 109.1792 | 108.4203 | 109.9380 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.7097 | 18.7097 | 18.6562 | 18.7633 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 209.5361 | 209.5361 | 208.6662 | 210.4060 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 22.7070 | 22.7070 | 22.7045 | 22.7095 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3227 | 0.3257 | 0.3103 | 0.3398 |
