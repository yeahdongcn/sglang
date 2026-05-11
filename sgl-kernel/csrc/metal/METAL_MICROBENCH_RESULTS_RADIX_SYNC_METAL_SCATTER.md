# Metal Attention Microbenchmark Results

Generated: `2026-05-09 21:47:45`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=8, decode/scatter iters=40, prefill iters=1
prefill_prefix_lens=[], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 24 | 0.3854 | 0.8208 | 0.3556 | 8.2254 |
| decode | mlx_fast_dense | 1 | 24 | 0.2071 | 0.4205 | 0.1863 | 7.8694 |
| decode | metal_paged_lazy_h128_b16 | 1 | 24 | 0.2588 | 0.5042 | 0.2095 | 4.5320 |
| decode | metal_paged_lazy_regscore_h128_b16 | 1 | 24 | 0.2569 | 0.3156 | 0.2038 | 1.7684 |
| decode | metal_paged_fused_kv_h128_b16 | 1 | 24 | 0.4346 | 0.7116 | 0.3457 | 5.0971 |
| decode | mlx_gather_paged_kv | 1 | 24 | 0.2515 | 0.5193 | 0.1882 | 6.5951 |
| decode | mlx_fast_gathered_paged | 1 | 24 | 0.2816 | 0.3088 | 0.2681 | 0.8318 |
| decode | metal_paged_radix_shuffled | 1 | 24 | 0.4469 | 0.7326 | 0.3888 | 9.0875 |
| decode | metal_paged_lazy_h128_b16_radix_shuffled | 1 | 24 | 0.3305 | 0.6395 | 0.2678 | 5.5982 |
| decode | metal_paged_lazy_regscore_h128_b16_radix_shuffled | 1 | 24 | 0.2883 | 0.6350 | 0.2405 | 4.2928 |
| decode | metal_paged_fused_kv_h128_b16_radix_shuffled | 1 | 24 | 0.4440 | 0.8089 | 0.4090 | 10.6914 |
| decode | mlx_gather_radix_paged_kv | 1 | 24 | 0.2585 | 0.4473 | 0.2161 | 4.2779 |
| decode | mlx_fast_radix_gathered_paged | 1 | 24 | 0.3584 | 0.6879 | 0.2638 | 4.1070 |
| decode | metal_dense_wrapper | 1 | 24 | 0.5386 | 0.9573 | 0.4220 | 6.6285 |
| prefill | metal_prefill_paged_no_prefix | 1 | 16 | 0.8162 | 0.8162 | 0.8162 | 0.8162 |
| prefill | metal_flash_varlen | 1 | 16 | 0.6854 | 0.6854 | 0.6854 | 0.6854 |
| prefill | mlx_fast_dense | 1 | 16 | 0.2655 | 0.2655 | 0.2655 | 0.2655 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.2788 | 0.4337 | 0.2310 | 2.7807 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.2584 | 0.4743 | 0.2269 | 3.4245 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3509 | 0.6471 | 0.3013 | 3.8997 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 1.8847 | 2.7617 | 1.7375 | 9.4044 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.4097 | 2.0517 | 1.1998 | 10.2227 |
| radix_sync | metal_scatter_all_layers_list | 1 | 28 | 0.8832 | 1.6753 | 0.7763 | 6.5558 |
| radix_sync | mlx_set_kv_all_layers_stacked | 4 | 28 | 1.8387 | 3.0346 | 1.5868 | 12.1182 |
| radix_sync | mlx_set_kv_all_layers_list | 4 | 28 | 1.4123 | 2.1669 | 1.2969 | 7.4585 |
| radix_sync | metal_scatter_all_layers_list | 4 | 28 | 1.1346 | 1.7038 | 0.9651 | 11.2643 |
| radix_sync | mlx_set_kv_all_layers_stacked | 256 | 28 | 14.9593 | 15.1135 | 8.8478 | 21.9056 |
| radix_sync | mlx_set_kv_all_layers_list | 256 | 28 | 11.1655 | 11.5813 | 6.7080 | 17.8423 |
| radix_sync | metal_scatter_all_layers_list | 256 | 28 | 9.5193 | 9.6650 | 5.1605 | 14.9982 |

## Mixed-Dtype Serving-Shaped Sync Spot Check

The radix server hit a real dtype mismatch after the first native all-layer
scatter change: Qwen-side K/V can be `bfloat16`, while the Metal-compatible
paged side-store is `float16`. The helper now casts each layer to the cache
dtype before native dispatch. A focused 28-layer sync spot check measured the
actual mixed-dtype shape directly:

Command context: 28 layers, 8 KV heads, head_dim 128, block_size 16,
`bfloat16` K/V inputs, `float16` paged caches, 8 warmups, and 40 timed
iterations.

| Sync path | Input dtype | Cache dtype | Tokens | Layers | Median ms |
|---|---|---|---:|---:|---:|
| list assignment | bfloat16 | float16 | 1 | 28 | 1.3566 |
| native all-layer scatter | bfloat16 | float16 | 1 | 28 | 1.2174 |
| list assignment | bfloat16 | float16 | 4 | 28 | 1.6968 |
| native all-layer scatter | bfloat16 | float16 | 4 | 28 | 1.5727 |
| list assignment | bfloat16 | float16 | 256 | 28 | 14.6236 |
| native all-layer scatter | bfloat16 | float16 | 256 | 28 | 12.6477 |
