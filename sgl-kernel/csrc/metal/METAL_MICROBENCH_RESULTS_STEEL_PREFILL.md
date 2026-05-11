# Metal Attention Microbenchmark Results

Generated: `2026-05-09 17:40:13`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=5
prefill_prefix_lens=[256]
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.4174 | 0.6127 | 0.3760 | 1.5503 |
| decode | mlx_fast_dense | 1 | 288 | 0.2166 | 0.2345 | 0.2116 | 0.3908 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2460 | 0.2469 | 0.2383 | 0.2573 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3829 | 0.5105 | 0.2822 | 1.2907 |
| decode | metal_dense_wrapper | 1 | 288 | 0.6358 | 0.6430 | 0.6161 | 0.7510 |
| decode | metal_paged | 1 | 2162 | 0.6890 | 0.7943 | 0.6045 | 1.4425 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4593 | 0.4610 | 0.4376 | 0.4838 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9233 | 1.1324 | 0.8707 | 2.4205 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1498 | 1.3394 | 1.0737 | 2.3530 |
| decode | metal_dense_wrapper | 1 | 2162 | 2.2625 | 2.3582 | 1.9294 | 3.0346 |
| decode | metal_paged | 4 | 288 | 0.5863 | 0.7738 | 0.5403 | 1.6818 |
| decode | mlx_fast_dense | 4 | 288 | 0.3905 | 0.3927 | 0.3438 | 0.4649 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.7424 | 0.8693 | 0.6943 | 1.3553 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.8690 | 0.8974 | 0.8397 | 1.0665 |
| decode | metal_dense_wrapper | 4 | 288 | 1.1704 | 1.2028 | 1.1114 | 1.3638 |
| decode | metal_paged | 4 | 2162 | 1.1515 | 1.1886 | 1.0556 | 1.4523 |
| decode | mlx_fast_dense | 4 | 2162 | 0.9165 | 0.9446 | 0.8720 | 1.0965 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.3338 | 3.4654 | 2.9184 | 4.1761 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.1153 | 4.0390 | 3.6178 | 4.3622 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.4536 | 5.5611 | 5.1824 | 6.2633 |
| decode | metal_paged | 8 | 288 | 0.7409 | 0.9761 | 0.6513 | 2.3691 |
| decode | mlx_fast_dense | 8 | 288 | 0.5418 | 0.7144 | 0.4766 | 2.4108 |
| decode | mlx_gather_paged_kv | 8 | 288 | 1.4685 | 1.4940 | 1.2869 | 1.8883 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.6284 | 1.7006 | 1.2975 | 2.7350 |
| decode | metal_dense_wrapper | 8 | 288 | 1.4928 | 1.5760 | 1.3315 | 1.9340 |
| decode | metal_paged | 8 | 2162 | 2.7001 | 2.9269 | 2.1157 | 3.9353 |
| decode | mlx_fast_dense | 8 | 2162 | 2.0893 | 2.1327 | 1.6940 | 2.4353 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 7.1016 | 7.1983 | 6.4342 | 8.4854 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 7.9795 | 8.1274 | 7.4688 | 9.7698 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.6255 | 10.7101 | 9.8743 | 11.7566 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 0.7025 | 0.7229 | 0.6910 | 0.8088 |
| prefill | metal_flash_varlen | 1 | 256 | 0.6690 | 0.7159 | 0.6600 | 0.8289 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5723 | 0.5749 | 0.5688 | 0.5829 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 7.6097 | 7.5849 | 7.1780 | 8.0148 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 0.8421 | 0.8402 | 0.8125 | 0.8817 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 3.3220 | 3.5575 | 3.1948 | 4.0800 |
| prefill | metal_flash_varlen | 1 | 1024 | 3.4110 | 3.6093 | 3.3051 | 4.2232 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.2470 | 5.5531 | 4.9672 | 6.9853 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 55.8932 | 56.0665 | 55.3806 | 57.1920 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 6.4615 | 6.4694 | 6.2815 | 6.6415 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 1.6037 | 1.6501 | 1.4993 | 1.8531 |
| prefill | metal_flash_varlen | 4 | 256 | 2.0225 | 1.8874 | 1.5523 | 2.1673 |
| prefill | mlx_fast_dense | 4 | 256 | 1.6671 | 1.7862 | 1.5778 | 2.2022 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 31.0640 | 31.7175 | 29.6502 | 35.7545 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 2.9422 | 2.9119 | 2.6455 | 3.2385 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 11.1182 | 11.5752 | 10.7514 | 12.5521 |
| prefill | metal_flash_varlen | 4 | 1024 | 11.5278 | 11.4699 | 10.7916 | 12.1208 |
| prefill | mlx_fast_dense | 4 | 1024 | 20.2313 | 20.2635 | 19.6385 | 20.7282 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 224.3417 | 223.4442 | 219.6397 | 227.3472 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 24.5898 | 24.3794 | 23.2856 | 25.0051 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3817 | 0.4091 | 0.3325 | 0.6403 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.4030 | 0.4450 | 0.3718 | 0.6789 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.4147 | 0.4152 | 0.3769 | 0.5097 |
