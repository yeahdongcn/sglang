# Metal Attention Microbenchmark Results

Generated: `2026-05-09 14:02:08`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=5, decode/scatter iters=20, prefill iters=5
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 256 | 0.5675 | 0.9006 | 0.5246 | 5.7091 |
| decode | mlx_fast_dense | 1 | 256 | 0.2100 | 0.2670 | 0.2027 | 1.2156 |
| decode | metal_dense_wrapper | 1 | 256 | 0.6140 | 0.9330 | 0.5150 | 5.1295 |
| decode | metal_paged | 1 | 2048 | 36.4585 | 35.8827 | 28.1379 | 40.6199 |
| decode | mlx_fast_dense | 1 | 2048 | 0.4558 | 0.5286 | 0.3910 | 1.3274 |
| decode | metal_dense_wrapper | 1 | 2048 | 36.1965 | 36.4379 | 27.8800 | 42.8332 |
| decode | metal_paged | 4 | 256 | 0.8611 | 1.1426 | 0.7922 | 4.7921 |
| decode | mlx_fast_dense | 4 | 256 | 0.3022 | 0.3517 | 0.2904 | 1.2776 |
| decode | metal_dense_wrapper | 4 | 256 | 0.8407 | 1.3281 | 0.8017 | 4.6083 |
| decode | metal_paged | 4 | 2048 | 44.8076 | 48.1643 | 38.2488 | 87.7678 |
| decode | mlx_fast_dense | 4 | 2048 | 0.9673 | 1.1720 | 0.8710 | 4.5147 |
| decode | metal_dense_wrapper | 4 | 2048 | 46.0373 | 47.1122 | 41.4688 | 83.1753 |
| decode | metal_paged | 8 | 256 | 1.3611 | 1.8744 | 1.2687 | 5.4650 |
| decode | mlx_fast_dense | 8 | 256 | 0.5791 | 0.8427 | 0.4497 | 4.8725 |
| decode | metal_dense_wrapper | 8 | 256 | 1.4982 | 2.0079 | 1.2827 | 5.5618 |
| decode | metal_paged | 8 | 2048 | 84.0768 | 85.7628 | 74.3386 | 117.8690 |
| decode | mlx_fast_dense | 8 | 2048 | 1.7410 | 2.2415 | 1.5468 | 5.3691 |
| decode | metal_dense_wrapper | 8 | 2048 | 84.9444 | 85.4135 | 75.0516 | 104.0450 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 14.8978 | 14.6898 | 11.6322 | 16.3070 |
| prefill | metal_flash_varlen | 1 | 256 | 14.9689 | 14.2234 | 11.3285 | 15.7437 |
| prefill | mlx_fast_dense | 1 | 256 | 0.5657 | 0.7755 | 0.5289 | 1.6496 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 56.7070 | 57.8046 | 55.1115 | 60.8808 |
| prefill | metal_flash_varlen | 4 | 256 | 55.0771 | 56.1223 | 54.5786 | 60.2480 |
| prefill | mlx_fast_dense | 4 | 256 | 1.5715 | 1.5867 | 1.5420 | 1.6569 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3785 | 0.4016 | 0.3293 | 0.8377 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3037 | 0.5682 | 0.2811 | 4.1392 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.3801 | 0.3972 | 0.3577 | 0.6316 |
