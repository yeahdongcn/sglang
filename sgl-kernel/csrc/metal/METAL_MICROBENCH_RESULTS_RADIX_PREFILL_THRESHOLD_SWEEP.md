# Metal Attention Microbenchmark Results

Generated: `2026-05-09 19:47:52`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=4, decode/scatter iters=8, prefill iters=5
prefill_prefix_lens=[256, 432], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 16 | 0.7246 | 1.5816 | 0.3432 | 4.7150 |
| decode | mlx_fast_dense | 1 | 16 | 0.1866 | 0.1949 | 0.1728 | 0.2441 |
| decode | mlx_gather_paged_kv | 1 | 16 | 0.2278 | 0.8403 | 0.1850 | 3.0297 |
| decode | mlx_fast_gathered_paged | 1 | 16 | 0.2652 | 0.3270 | 0.2465 | 0.6761 |
| decode | metal_paged_radix_shuffled | 1 | 16 | 0.8038 | 1.5717 | 0.3516 | 4.5280 |
| decode | mlx_gather_radix_paged_kv | 1 | 16 | 0.2720 | 0.3022 | 0.2285 | 0.4015 |
| decode | mlx_fast_radix_gathered_paged | 1 | 16 | 0.2592 | 0.8194 | 0.2359 | 4.4450 |
| decode | metal_dense_wrapper | 1 | 16 | 0.4636 | 0.4695 | 0.4164 | 0.5533 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 0.7167 | 0.7221 | 0.6737 | 0.7738 |
| prefill | metal_flash_varlen | 1 | 256 | 0.7708 | 1.4384 | 0.6742 | 4.1803 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6015 | 1.9940 | 0.5978 | 4.2055 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 2.1134 | 2.9807 | 1.4640 | 4.6513 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 1.4677 | 2.5164 | 0.9022 | 4.6611 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 256 | 2.5584 | 3.2176 | 2.2284 | 5.2134 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 256 | 2.1633 | 2.2559 | 1.0598 | 4.7792 |
| prefill | metal_prefill_paged_no_prefix | 1 | 384 | 1.6120 | 2.5265 | 0.9835 | 4.4552 |
| prefill | metal_flash_varlen | 1 | 384 | 2.2197 | 2.8601 | 0.9794 | 5.3093 |
| prefill | mlx_fast_dense | 1 | 384 | 1.6135 | 2.0192 | 0.9877 | 4.5688 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 384 | 2.1496 | 3.0095 | 1.7641 | 6.0680 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 384 | 1.4260 | 1.5565 | 1.3696 | 2.1119 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 384 | 2.4104 | 4.2642 | 2.0922 | 12.0175 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 384 | 2.8365 | 3.5086 | 1.8621 | 6.6107 |
| prefill | metal_prefill_paged_no_prefix | 1 | 512 | 1.4203 | 2.1194 | 1.3961 | 4.6777 |
| prefill | metal_flash_varlen | 1 | 512 | 1.9406 | 2.1940 | 1.2514 | 3.9460 |
| prefill | mlx_fast_dense | 1 | 512 | 2.4172 | 3.4457 | 1.5655 | 6.7037 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 512 | 3.1012 | 3.3675 | 2.4363 | 5.6480 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 512 | 2.6806 | 3.9151 | 2.1446 | 6.2307 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 512 | 4.1971 | 4.6559 | 3.3210 | 6.2807 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 512 | 4.0980 | 5.7400 | 2.5847 | 9.6217 |
| prefill | metal_prefill_paged_no_prefix | 1 | 768 | 4.9689 | 4.8876 | 2.4425 | 7.5719 |
| prefill | metal_flash_varlen | 1 | 768 | 4.2739 | 4.0634 | 2.6782 | 5.9694 |
| prefill | mlx_fast_dense | 1 | 768 | 7.1243 | 6.9356 | 3.6325 | 8.8707 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 768 | 7.5695 | 7.0604 | 4.2116 | 8.1067 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 768 | 8.0368 | 8.4239 | 6.1643 | 12.0777 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 768 | 7.3710 | 8.4834 | 4.3315 | 14.0612 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 768 | 13.1525 | 10.1714 | 5.3552 | 13.5512 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 6.5239 | 6.0603 | 3.4150 | 8.0078 |
| prefill | metal_flash_varlen | 1 | 1024 | 7.1436 | 6.6177 | 5.2315 | 7.7121 |
| prefill | mlx_fast_dense | 1 | 1024 | 12.4259 | 12.1275 | 5.1076 | 18.3546 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 8.1416 | 7.3076 | 5.5865 | 8.7217 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 11.6832 | 11.6261 | 6.7618 | 18.1440 |
| prefill | metal_prefill_paged_prefix_432 | 1 | 1024 | 10.4087 | 11.1641 | 9.4638 | 13.7376 |
| prefill | mlx_fast_dense_prefix_432 | 1 | 1024 | 15.0295 | 13.5874 | 7.3417 | 16.9598 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 1.4420 | 2.1366 | 1.3924 | 4.8243 |
| prefill | metal_flash_varlen | 4 | 256 | 2.2696 | 3.0813 | 1.3558 | 5.3688 |
| prefill | mlx_fast_dense | 4 | 256 | 2.7008 | 3.4788 | 1.7044 | 5.4842 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 4.5489 | 5.6784 | 3.7719 | 11.1752 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 3.4515 | 6.7151 | 2.7512 | 12.5546 |
| prefill | metal_prefill_paged_prefix_432 | 4 | 256 | 10.4402 | 9.2929 | 5.2761 | 13.9991 |
| prefill | mlx_fast_dense_prefix_432 | 4 | 256 | 4.1262 | 5.4230 | 3.8032 | 8.0010 |
| prefill | metal_prefill_paged_no_prefix | 4 | 384 | 3.7113 | 4.7115 | 2.4469 | 10.9864 |
| prefill | metal_flash_varlen | 4 | 384 | 4.2717 | 5.1090 | 2.3525 | 9.5168 |
| prefill | mlx_fast_dense | 4 | 384 | 6.5696 | 6.6320 | 4.3181 | 9.6520 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 384 | 6.1586 | 7.2202 | 5.2140 | 10.2981 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 384 | 10.2120 | 10.5582 | 5.6496 | 15.5506 |
| prefill | metal_prefill_paged_prefix_432 | 4 | 384 | 10.3104 | 10.3210 | 8.2747 | 12.3852 |
| prefill | mlx_fast_dense_prefix_432 | 4 | 384 | 11.8432 | 12.0441 | 7.0856 | 15.4923 |
| prefill | metal_prefill_paged_no_prefix | 4 | 512 | 4.8825 | 7.4091 | 3.7389 | 12.4812 |
| prefill | metal_flash_varlen | 4 | 512 | 8.0962 | 9.6756 | 5.4038 | 15.4162 |
| prefill | mlx_fast_dense | 4 | 512 | 10.8982 | 10.7286 | 7.6330 | 13.0145 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 512 | 14.6258 | 13.3635 | 8.1668 | 16.1863 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 512 | 16.5710 | 16.9170 | 16.4243 | 17.8379 |
| prefill | metal_prefill_paged_prefix_432 | 4 | 512 | 13.5078 | 15.9082 | 10.1249 | 28.3238 |
| prefill | mlx_fast_dense_prefix_432 | 4 | 512 | 19.7776 | 19.6980 | 14.2943 | 25.0690 |
| prefill | metal_prefill_paged_no_prefix | 4 | 768 | 8.2639 | 9.5047 | 6.9502 | 14.1635 |
| prefill | metal_flash_varlen | 4 | 768 | 16.5029 | 14.9192 | 10.8405 | 18.0972 |
| prefill | mlx_fast_dense | 4 | 768 | 22.2922 | 25.2092 | 17.8134 | 43.2285 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 768 | 18.1016 | 18.4629 | 12.4191 | 24.1910 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 768 | 37.2829 | 37.6467 | 30.3008 | 46.0104 |
| prefill | metal_prefill_paged_prefix_432 | 4 | 768 | 31.4610 | 32.7742 | 24.5531 | 41.1737 |
| prefill | mlx_fast_dense_prefix_432 | 4 | 768 | 38.2706 | 38.9785 | 28.6270 | 47.7131 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 15.0544 | 17.4528 | 11.2185 | 25.4797 |
| prefill | metal_flash_varlen | 4 | 1024 | 19.2319 | 18.6778 | 15.4365 | 22.0225 |
| prefill | mlx_fast_dense | 4 | 1024 | 31.8508 | 34.5882 | 29.9465 | 42.5571 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 30.7828 | 31.1359 | 22.9362 | 35.7555 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 48.8778 | 47.3091 | 39.6394 | 51.9859 |
| prefill | metal_prefill_paged_prefix_432 | 4 | 1024 | 45.9729 | 51.1035 | 38.8000 | 76.3759 |
| prefill | mlx_fast_dense_prefix_432 | 4 | 1024 | 48.8993 | 47.8897 | 38.2605 | 58.3575 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3356 | 0.3392 | 0.3045 | 0.3721 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 2.5937 | 4.1850 | 2.1856 | 10.1233 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.7208 | 2.0320 | 1.4698 | 3.5534 |
