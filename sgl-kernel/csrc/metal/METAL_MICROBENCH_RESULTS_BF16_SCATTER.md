# Metal Attention Microbenchmark Results

Generated: `2026-05-10 22:47:19`

Command context:

```text
dtype=bfloat16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=10
prefill_prefix_lens=[], sync_layers=28, bf16_p1_decode=False
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2375 | 0.6789 | 0.1859 | 6.8650 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2573 | 0.2575 | 0.2177 | 0.2888 |
| scatter | metal_paged_kv_scatter | 1 | 16 | 0.2710 | 0.2772 | 0.2220 | 0.3617 |
| scatter | metal_paged_kv_scatter | 1 | 64 | 0.2894 | 0.3669 | 0.2252 | 2.1402 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3576 | 0.3476 | 0.2924 | 0.4020 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.6459 | 2.4422 | 1.4357 | 8.3663 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.3823 | 1.4480 | 1.3407 | 2.7287 |
| radix_sync | metal_scatter_all_layers_list | 1 | 28 | 1.1326 | 1.8389 | 1.0885 | 7.1303 |
| radix_sync | mlx_set_kv_all_layers_stacked | 4 | 28 | 2.5245 | 3.6802 | 1.8736 | 8.1073 |
| radix_sync | mlx_set_kv_all_layers_list | 4 | 28 | 1.8289 | 3.1335 | 1.4917 | 9.2234 |
| radix_sync | metal_scatter_all_layers_list | 4 | 28 | 1.4466 | 2.3454 | 1.3624 | 7.2171 |
| radix_sync | mlx_set_kv_all_layers_stacked | 16 | 28 | 3.2829 | 4.1545 | 2.3038 | 8.3349 |
| radix_sync | mlx_set_kv_all_layers_list | 16 | 28 | 2.2875 | 3.4675 | 1.9050 | 7.8375 |
| radix_sync | metal_scatter_all_layers_list | 16 | 28 | 2.1661 | 3.6756 | 1.8228 | 10.4909 |
| radix_sync | mlx_set_kv_all_layers_stacked | 64 | 28 | 4.6629 | 5.1354 | 3.7093 | 7.9710 |
| radix_sync | mlx_set_kv_all_layers_list | 64 | 28 | 3.3362 | 3.8336 | 3.0996 | 8.4065 |
| radix_sync | metal_scatter_all_layers_list | 64 | 28 | 3.8552 | 4.3793 | 3.1750 | 9.5363 |
| radix_sync | mlx_set_kv_all_layers_stacked | 256 | 28 | 13.7020 | 13.0203 | 9.4717 | 17.8907 |
| radix_sync | mlx_set_kv_all_layers_list | 256 | 28 | 12.0600 | 11.0120 | 8.0988 | 12.9623 |
| radix_sync | metal_scatter_all_layers_list | 256 | 28 | 7.6903 | 8.4603 | 6.6900 | 12.9180 |
