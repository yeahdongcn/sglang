# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:08:43`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=30, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 2162 | 0.6601 | 0.6971 | 0.6188 | 1.2483 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4596 | 0.5220 | 0.4196 | 1.7988 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9056 | 0.9930 | 0.8865 | 2.1325 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1553 | 1.2328 | 1.0805 | 2.3786 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6981 | 0.7562 | 0.6213 | 1.7437 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 0.9733 | 1.0710 | 0.8988 | 2.0103 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.1849 | 1.3051 | 1.0843 | 3.3353 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.2891 | 2.3417 | 1.9589 | 3.0649 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.7744 | 0.7744 | 0.7744 | 0.7744 |
| prefill | metal_flash_varlen | 1 | 16 | 0.6089 | 0.6089 | 0.6089 | 0.6089 |
| prefill | mlx_fast_dense | 1 | 16 | 0.3623 | 0.3623 | 0.3623 | 0.3623 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3234 | 0.3750 | 0.2218 | 1.4139 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 2.0885 | 2.1018 | 1.8008 | 3.1032 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.3744 | 1.4758 | 1.2680 | 2.6427 |
