# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:16:19`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=3
prefill_prefix_lens=[256]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4300 | 0.4357 | 0.4184 | 0.4655 |
| decode | mlx_fast_dense | 1 | 288 | 0.2689 | 0.2738 | 0.2617 | 0.2968 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2681 | 0.2711 | 0.2543 | 0.2951 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.2918 | 0.3168 | 0.2833 | 0.3932 |
| decode | metal_dense_wrapper | 1 | 288 | 0.7798 | 0.7628 | 0.7000 | 0.8170 |
| decode | metal_paged | 1 | 2162 | 0.6219 | 0.6264 | 0.5995 | 0.6572 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4180 | 0.4226 | 0.3899 | 0.4718 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9199 | 0.9215 | 0.8638 | 0.9941 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1016 | 1.1029 | 1.0845 | 1.1287 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.5772 | 2.6049 | 2.2630 | 2.9423 |
| decode | metal_paged | 4 | 288 | 0.5578 | 0.5608 | 0.5150 | 0.6496 |
| decode | mlx_fast_dense | 4 | 288 | 0.3817 | 0.3774 | 0.3473 | 0.3937 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7612 | 0.7663 | 0.7498 | 0.7997 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9399 | 0.9411 | 0.9082 | 0.9675 |
| decode | metal_dense_wrapper | 4 | 288 | 1.3034 | 1.3113 | 1.2719 | 1.4170 |
| decode | metal_paged | 4 | 2162 | 1.5168 | 1.5389 | 1.4811 | 1.7060 |
| decode | mlx_fast_dense | 4 | 2162 | 1.2405 | 1.2479 | 1.2131 | 1.2992 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.0833 | 3.1819 | 3.0084 | 3.6739 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 3.7412 | 3.7590 | 3.7007 | 3.8844 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.6564 | 5.6569 | 5.4255 | 5.8477 |
| decode | metal_paged | 8 | 288 | 0.5649 | 0.5593 | 0.5330 | 0.5846 |
| decode | mlx_fast_dense | 8 | 288 | 0.4195 | 0.4171 | 0.3910 | 0.4389 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.1197 | 1.1059 | 0.9694 | 1.3667 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.4240 | 1.4146 | 1.2608 | 1.5366 |
| decode | metal_dense_wrapper | 8 | 288 | 1.5452 | 1.5494 | 1.4339 | 1.6690 |
| decode | metal_paged | 8 | 2162 | 1.8296 | 1.8119 | 1.7062 | 1.9110 |
| decode | mlx_fast_dense | 8 | 2162 | 1.5304 | 1.5365 | 1.4745 | 1.6192 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 5.8292 | 5.8680 | 5.6142 | 6.2438 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.0707 | 7.0449 | 6.8583 | 7.1851 |
| decode | metal_dense_wrapper | 8 | 2162 | 9.9676 | 10.0390 | 9.7411 | 10.8350 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 2.9431 | 2.9305 | 2.8727 | 2.9756 |
| prefill | metal_flash_varlen | 1 | 256 | 2.2746 | 2.2711 | 2.2638 | 2.2748 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6930 | 0.7051 | 0.6764 | 0.7458 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 7.5172 | 7.4766 | 7.2468 | 7.6658 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 0.8766 | 0.8751 | 0.8668 | 0.8818 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 31.7120 | 31.4525 | 30.6575 | 31.9880 |
| prefill | metal_flash_varlen | 1 | 1024 | 23.8571 | 23.8921 | 23.7686 | 24.0507 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.2433 | 5.2206 | 5.1708 | 5.2478 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 47.3442 | 47.5386 | 46.9127 | 48.3587 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 6.5104 | 6.5240 | 6.4682 | 6.5934 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 9.7895 | 9.7891 | 9.7130 | 9.8647 |
| prefill | metal_flash_varlen | 4 | 256 | 7.7894 | 7.7072 | 7.5178 | 7.8144 |
| prefill | mlx_fast_dense | 4 | 256 | 1.7445 | 1.7400 | 1.7163 | 1.7590 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 28.2193 | 28.2276 | 27.3239 | 29.1396 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 2.8378 | 2.8789 | 2.8123 | 2.9865 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 121.2890 | 121.5847 | 120.3837 | 123.0813 |
| prefill | metal_flash_varlen | 4 | 1024 | 93.6472 | 93.8110 | 92.5503 | 95.2355 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.6010 | 18.5753 | 18.5154 | 18.6094 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 191.9537 | 192.4347 | 190.5923 | 194.7582 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 23.1551 | 23.1572 | 23.1539 | 23.1627 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.4146 | 0.4260 | 0.3493 | 0.6242 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3187 | 0.3241 | 0.3109 | 0.3472 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3789 | 0.3781 | 0.3543 | 0.4055 |
