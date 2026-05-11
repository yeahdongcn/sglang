# Metal Contiguous Paged Decode Specialization Microbenchmark Results

Generated: `2026-05-09 21:02`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
temporary probes only; no runtime code kept from this experiment
```

## Direct Contiguous Block Formula

This probe replaced the generic block-table read in the lazy h128/block16
decode shader with:

```metal
const int block = batch_idx * max_blocks + block_idx + 1;
```

The specialization is correct for synthetic contiguous page layouts, but the
speedup is shape-dependent and does not clear dense MLX fast attention broadly.

| S | B | dense | lazy | contiguous-special | maxdiff |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 0.2350 | 0.2516 | 0.2810 | 0.00000 |
| 128 | 4 | 0.2744 | 0.2818 | 0.2735 | 0.00000 |
| 128 | 8 | 0.3768 | 0.4035 | 0.3539 | 0.00000 |
| 288 | 1 | 0.2688 | 0.2919 | 0.2734 | 0.00000 |
| 288 | 4 | 0.3405 | 0.3852 | 0.3291 | 0.00000 |
| 288 | 8 | 0.4552 | 0.4688 | 0.4668 | 0.00000 |
| 512 | 1 | 0.2625 | 0.2790 | 0.2952 | 0.00000 |
| 512 | 4 | 0.4380 | 0.4723 | 0.4506 | 0.00000 |
| 512 | 8 | 0.6081 | 0.6171 | 0.5859 | 0.00000 |
| 1024 | 1 | 0.2926 | 0.3330 | 0.3227 | 0.00000 |
| 1024 | 4 | 0.6233 | 0.6746 | 0.7526 | 0.00000 |
| 1024 | 8 | 0.9060 | 0.9615 | 1.0242 | 0.00000 |
| 2162 | 1 | 0.4499 | 0.5648 | 0.4935 | 0.00000 |
| 2162 | 4 | 1.0287 | 1.4146 | 1.0750 | 0.00000 |
| 2162 | 8 | 2.0533 | 2.3563 | 2.3607 | 0.00000 |
| 4096 | 1 | 0.6714 | 0.6696 | 0.6589 | 0.00000 |
| 4096 | 4 | 2.1222 | 1.6478 | 2.2771 | 0.00000 |
| 4096 | 8 | 4.0421 | 3.4608 | 3.6515 | 0.00000 |

Threadgroup-size variants were also mixed:

| S | B | dense | t128 | t192 | t256 | t512 |
|---:|---:|---:|---:|---:|---:|---:|
| 288 | 1 | 0.2648 | 0.2684 | 0.2481 | 0.2562 | 0.2560 |
| 288 | 4 | 0.3373 | 0.3153 | 0.3399 | 0.3473 | 0.3697 |
| 288 | 8 | 0.4569 | 0.4381 | 0.5180 | 0.4576 | 0.4447 |
| 1024 | 1 | 0.2565 | 0.3039 | 0.2847 | 0.2779 | 0.2985 |
| 1024 | 4 | 0.6599 | 0.8700 | 0.6998 | 0.6775 | 0.7456 |
| 1024 | 8 | 0.9556 | 1.6815 | 1.0310 | 0.9293 | 1.0026 |
| 2162 | 1 | 0.4161 | 0.5694 | 0.5805 | 0.5534 | 0.5410 |
| 2162 | 4 | 2.2449 | 0.9880 | 1.0218 | 1.0709 | 1.1211 |
| 2162 | 8 | 2.0329 | 1.9970 | 2.1252 | 1.9607 | 1.9875 |
| 4096 | 1 | 0.6616 | 0.8128 | 0.8079 | 0.6541 | 0.6895 |
| 4096 | 4 | 2.1099 | 2.4266 | 2.2133 | 2.0468 | 2.4893 |
| 4096 | 8 | 4.6045 | 4.4338 | 4.2506 | 4.6438 | 3.3217 |

## Reshape-to-MLX Fast Probe

For contiguous page layouts, this probe reshaped `k_cache[1:]` and `v_cache[1:]`
into dense K/V tensors and called `mx.fast.scaled_dot_product_attention`
without block-table gather. It was consistently worse than the current lazy
decode kernel because the transpose/materialization and padding mask dominate.

| S | B | dense | lazy | reshape_fast_view | reshape_fast_contig |
|---:|---:|---:|---:|---:|---:|
| 288 | 1 | 0.2954 | 0.2797 | 0.4888 | 0.4763 |
| 288 | 4 | 0.3299 | 0.3628 | 0.7741 | 0.8015 |
| 288 | 8 | 0.4636 | 0.4950 | 1.2032 | 1.1189 |
| 512 | 1 | 0.2743 | 0.3088 | 0.5064 | 0.5855 |
| 512 | 4 | 0.4303 | 0.4536 | 1.3272 | 1.3096 |
| 512 | 8 | 0.6144 | 0.5991 | 2.1281 | 2.1781 |
| 1024 | 1 | 0.3899 | 0.3099 | 0.4870 | 0.6970 |
| 1024 | 4 | 0.6392 | 0.6647 | 1.7228 | 2.0624 |
| 1024 | 8 | 1.1260 | 1.2707 | 3.9512 | 3.4177 |
| 2162 | 1 | 0.4823 | 0.5159 | 0.7418 | 1.1176 |
| 2162 | 4 | 1.4577 | 1.3752 | 3.2493 | 3.3494 |
| 2162 | 8 | 2.2853 | 2.1501 | 6.9231 | 6.0509 |
| 4096 | 1 | 0.7095 | 0.7270 | 1.0368 | 2.0722 |
| 4096 | 4 | 2.2796 | 2.0871 | 6.3016 | 7.3539 |
| 4096 | 8 | 3.7844 | 6.7030 | 15.3338 | 14.1703 |

## Decision

Rejected. The contiguous-block formula is a useful diagnostic but not a robust
runtime direction, and reshape-to-MLX fast is not competitive. The open raw
contiguous decode gap needs a deeper algorithmic change rather than a
block-table-load shortcut.
