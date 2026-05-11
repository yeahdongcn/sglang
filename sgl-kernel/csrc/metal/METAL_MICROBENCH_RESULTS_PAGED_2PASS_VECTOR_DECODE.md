# Metal Paged Two-Pass Vector Decode Microbenchmark Results

Generated: `2026-05-09 21:23`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
temporary two-pass mx.fast.metal_kernel probe for paged K/V decode
no runtime code kept from this experiment
```

This probe split q_len=1 paged decode into:

1. One-simdgroup partial kernels over interleaved KV token ranges.
2. A second per-head reduction kernel over partial max/sum/output values.

The goal was to mimic MLX's two-pass vector strategy for long dense decode
while preserving direct paged K/V reads and increasing B1 long-context
parallelism.

The probe was correct against the accepted lazy paged decode kernel, but the
two kernel launches and partial writes did not produce a broad win.

| S | B | Layout | dense | lazy | two16 | two32 | maxdiff32 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 1 | contiguous | 0.2368 | 0.2922 | 0.2304 | 0.2776 | 0.00012 |
| 512 | 1 | shuffled | 0.2368 | 0.3586 | 0.3422 | 0.2900 | 0.00012 |
| 512 | 4 | contiguous | 0.4654 | 0.4642 | 0.4909 | 0.4973 | 0.00012 |
| 512 | 4 | shuffled | 0.4654 | 0.5350 | 0.5383 | 0.5235 | 0.00012 |
| 512 | 8 | contiguous | 0.6566 | 0.6320 | 0.7937 | 0.6225 | 0.00024 |
| 512 | 8 | shuffled | 0.6566 | 0.6781 | 0.6582 | 0.6898 | 0.00024 |
| 1024 | 1 | contiguous | 0.3450 | 0.3759 | 0.4086 | 0.4074 | 0.00012 |
| 1024 | 1 | shuffled | 0.3450 | 0.3842 | 0.3982 | 0.4247 | 0.00012 |
| 1024 | 4 | contiguous | 0.6427 | 0.7163 | 0.6380 | 0.6439 | 0.00012 |
| 1024 | 4 | shuffled | 0.6427 | 0.6411 | 0.6706 | 0.6840 | 0.00012 |
| 1024 | 8 | contiguous | 0.8924 | 1.1280 | 1.1176 | 0.9769 | 0.00012 |
| 1024 | 8 | shuffled | 0.8924 | 1.3144 | 1.3653 | 1.2773 | 0.00012 |
| 2162 | 1 | contiguous | 0.5211 | 0.5563 | 0.5775 | 0.6237 | 0.00006 |
| 2162 | 1 | shuffled | 0.5211 | 0.5618 | 0.5897 | 0.7097 | 0.00006 |
| 2162 | 4 | contiguous | 1.1917 | 1.2542 | 1.6494 | 1.5421 | 0.00006 |
| 2162 | 4 | shuffled | 1.1917 | 1.5959 | 1.3697 | 1.3988 | 0.00006 |
| 2162 | 8 | contiguous | 1.7086 | 2.5778 | 2.4569 | 2.4160 | 0.00006 |
| 2162 | 8 | shuffled | 1.7086 | 2.4866 | 2.9626 | 2.6576 | 0.00006 |
| 4096 | 1 | contiguous | 0.8025 | 0.6269 | 0.6941 | 0.8876 | 0.00006 |
| 4096 | 1 | shuffled | 0.8025 | 0.7626 | 0.7349 | 0.8273 | 0.00006 |
| 4096 | 4 | contiguous | 1.9518 | 1.6227 | 1.7081 | 1.7938 | 0.00006 |
| 4096 | 4 | shuffled | 1.9518 | 2.7343 | 3.1564 | 2.3210 | 0.00006 |
| 4096 | 8 | contiguous | 5.2183 | 2.9753 | 3.4349 | 4.7845 | 0.00006 |
| 4096 | 8 | shuffled | 5.2183 | 4.8202 | 5.3380 | 4.2989 | 0.00006 |

## Decision

Rejected. The two-pass vector split validates correctly, but it does not solve
the B1/B4/B8 long decode gap. For direct paged decode, the accepted lazy
single-pass online kernel remains the best retained option.
