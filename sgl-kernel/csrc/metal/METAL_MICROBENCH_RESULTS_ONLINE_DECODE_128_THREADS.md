# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:02:07`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3470 | 0.3624 | 0.3323 | 0.4655 |
| decode | mlx_fast_dense | 1 | 288 | 0.2122 | 0.2709 | 0.2080 | 0.5456 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2637 | 0.2631 | 0.2490 | 0.2777 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3198 | 0.3529 | 0.2805 | 0.6170 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6738 | 0.7683 | 0.6293 | 1.1427 |
| decode | metal_paged | 1 | 2162 | 0.7102 | 0.7394 | 0.6885 | 0.8394 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4479 | 0.4974 | 0.4028 | 0.6643 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9659 | 0.9451 | 0.8590 | 1.0529 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1331 | 1.1502 | 1.0641 | 1.2825 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.1324 | 2.1212 | 1.7923 | 2.5023 |
| decode | metal_paged | 4 | 288 | 0.5233 | 0.5635 | 0.4791 | 0.8881 |
| decode | mlx_fast_dense | 4 | 288 | 0.3104 | 0.3106 | 0.2968 | 0.3195 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7011 | 0.7453 | 0.6108 | 1.0333 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8802 | 1.0027 | 0.7720 | 1.7889 |
| decode | metal_dense_wrapper | 4 | 288 | 1.1785 | 1.3809 | 1.0672 | 2.3791 |
| decode | metal_paged | 4 | 2162 | 1.1658 | 1.3560 | 1.0838 | 2.3947 |
| decode | mlx_fast_dense | 4 | 2162 | 1.1430 | 1.4050 | 0.9139 | 2.9647 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 4.0418 | 4.1820 | 3.1402 | 5.6188 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.3234 | 4.4284 | 3.6713 | 5.2134 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.2927 | 5.4662 | 5.1369 | 6.3010 |
| decode | metal_paged | 8 | 288 | 0.7088 | 0.7924 | 0.5865 | 1.4047 |
| decode | mlx_fast_dense | 8 | 288 | 0.5403 | 0.6493 | 0.4685 | 1.2258 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.2205 | 1.3473 | 1.0959 | 1.9100 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.5141 | 1.5749 | 1.1547 | 2.7398 |
| decode | metal_dense_wrapper | 8 | 288 | 1.2827 | 1.4835 | 1.2206 | 2.9840 |
| decode | metal_paged | 8 | 2162 | 2.9780 | 2.9801 | 2.1834 | 3.7493 |
| decode | mlx_fast_dense | 8 | 2162 | 1.8057 | 1.9813 | 1.5920 | 2.8620 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.4810 | 6.6698 | 5.7939 | 7.9480 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 8.4348 | 8.7592 | 7.5428 | 10.6002 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.3168 | 10.5013 | 8.6825 | 12.2797 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 11.6736 | 12.0749 | 11.1671 | 13.3841 |
| prefill | metal_flash_varlen | 1 | 256 | 11.0190 | 10.9484 | 10.6166 | 11.2096 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6602 | 0.6637 | 0.6523 | 0.6786 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 183.3497 | 177.7226 | 165.5375 | 184.2805 |
| prefill | metal_flash_varlen | 1 | 1024 | 167.8646 | 165.5818 | 159.3773 | 169.5034 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.6322 | 5.5701 | 5.0690 | 6.0090 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 42.8578 | 43.0269 | 42.5058 | 43.7170 |
| prefill | metal_flash_varlen | 4 | 256 | 42.2285 | 41.8201 | 40.9300 | 42.3019 |
| prefill | mlx_fast_dense | 4 | 256 | 1.6474 | 1.6573 | 1.5886 | 1.7360 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 683.4659 | 700.7953 | 681.6525 | 737.2675 |
| prefill | metal_flash_varlen | 4 | 1024 | 667.1382 | 660.8481 | 636.1823 | 679.2238 |
| prefill | mlx_fast_dense | 4 | 1024 | 19.6655 | 19.9806 | 19.4762 | 20.8000 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3979 | 0.4261 | 0.3454 | 0.5578 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.4279 | 0.4908 | 0.3369 | 0.8410 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.4121 | 0.4173 | 0.3956 | 0.4582 |
