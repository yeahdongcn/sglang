# Metal Paged SDPA-Vector Decode Microbenchmark Results

Generated: `2026-05-09 21:16`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
temporary mx.fast.metal_kernel probe adapting MLX sdpa_vector to paged K/V
no runtime code kept from this experiment
```

This probe adapted MLX's q_len=1 `sdpa_vector` algorithm to paged K/V. It uses
32 simdgroups per `(batch, head)` group, with each simdgroup scanning every
32nd token and then reducing partial max/sum/output across simdgroups. The goal
was to get closer to the dense MLX decode algorithm while keeping direct paged
K/V reads.

The kernel was correct against the accepted lazy paged decode kernel, but the
performance was mixed and did not clear dense MLX fast attention across the
target matrix.

| S | B | Layout | dense | lazy | sdpa_vector_paged | maxdiff |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 1 | contiguous | 0.2706 | 0.1998 | 0.2299 | 0.00000 |
| 128 | 1 | shuffled | 0.2706 | 0.2506 | 0.2592 | 0.00000 |
| 128 | 4 | contiguous | 0.3072 | 0.3146 | 0.2682 | 0.00012 |
| 128 | 4 | shuffled | 0.3072 | 0.2654 | 0.2884 | 0.00012 |
| 128 | 8 | contiguous | 0.3618 | 0.4593 | 0.6572 | 0.00012 |
| 128 | 8 | shuffled | 0.3618 | 0.4747 | 0.3804 | 0.00012 |
| 288 | 1 | contiguous | 0.3322 | 0.2686 | 0.3292 | 0.00006 |
| 288 | 1 | shuffled | 0.3322 | 0.2823 | 0.2547 | 0.00006 |
| 288 | 4 | contiguous | 0.3538 | 0.3440 | 0.3993 | 0.00012 |
| 288 | 4 | shuffled | 0.3538 | 0.4058 | 0.3791 | 0.00012 |
| 288 | 8 | contiguous | 0.4551 | 0.4660 | 0.5039 | 0.00006 |
| 288 | 8 | shuffled | 0.4551 | 0.5339 | 0.5435 | 0.00006 |
| 512 | 1 | contiguous | 0.2682 | 0.3140 | 0.3034 | 0.00001 |
| 512 | 1 | shuffled | 0.2682 | 0.2922 | 0.3086 | 0.00001 |
| 512 | 4 | contiguous | 0.4823 | 0.4985 | 0.4513 | 0.00003 |
| 512 | 4 | shuffled | 0.4823 | 0.5021 | 0.4915 | 0.00003 |
| 512 | 8 | contiguous | 0.7088 | 0.7232 | 0.8756 | 0.00003 |
| 512 | 8 | shuffled | 0.7088 | 0.6810 | 0.8297 | 0.00003 |
| 1024 | 1 | contiguous | 0.3604 | 0.3725 | 0.3765 | 0.00002 |
| 1024 | 1 | shuffled | 0.3604 | 0.3680 | 0.5211 | 0.00002 |
| 1024 | 4 | contiguous | 0.6668 | 0.6940 | 1.0340 | 0.00012 |
| 1024 | 4 | shuffled | 0.6668 | 0.8078 | 0.6809 | 0.00012 |
| 1024 | 8 | contiguous | 1.4434 | 1.8660 | 1.6091 | 0.00006 |
| 1024 | 8 | shuffled | 1.4434 | 1.6168 | 0.9443 | 0.00006 |
| 2162 | 1 | contiguous | 0.4771 | 0.5546 | 0.5487 | 0.00000 |
| 2162 | 1 | shuffled | 0.4771 | 0.6647 | 0.5551 | 0.00000 |
| 2162 | 4 | contiguous | 1.0735 | 1.0981 | 1.1540 | 0.00001 |
| 2162 | 4 | shuffled | 1.0735 | 1.0318 | 1.1474 | 0.00001 |
| 2162 | 8 | contiguous | 2.0761 | 2.5884 | 3.2541 | 0.00003 |
| 2162 | 8 | shuffled | 2.0761 | 2.7797 | 2.9754 | 0.00003 |
| 4096 | 1 | contiguous | 0.7345 | 0.6788 | 0.7403 | 0.00000 |
| 4096 | 1 | shuffled | 0.7345 | 0.6337 | 0.7952 | 0.00000 |
| 4096 | 4 | contiguous | 2.7523 | 1.6429 | 2.1200 | 0.00003 |
| 4096 | 4 | shuffled | 2.7523 | 2.8548 | 2.4526 | 0.00003 |
| 4096 | 8 | contiguous | 5.1783 | 6.5131 | 6.5277 | 0.00002 |
| 4096 | 8 | shuffled | 5.1783 | 5.2729 | 4.7788 | 0.00002 |

## Decision

Rejected. The paged `sdpa_vector` adaptation validates that the MLX q_len=1
algorithm can read paged K/V correctly, but its higher threadgroup footprint and
extra reduction do not produce a stable win. The accepted lazy paged decode
kernel remains the better direct-paged route for now.
