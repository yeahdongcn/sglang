# Metal Dense Decode Lazy Kernel Microbenchmark Results

Generated: `2026-05-09 21:07`

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128
temporary mx.fast.metal_kernel probe for dense K/V layout (B, KVH, S, D)
no runtime code kept from this experiment
```

This probe adapted the accepted lazy paged online-softmax decode kernel to read
dense contiguous K/V directly, with no block tables or page layout. The goal was
to see whether no-radix batched decode should route through a new Metal kernel
instead of dense `mx.fast.scaled_dot_product_attention`.

The kernel was correct against MLX fast attention, but the performance result
was mixed:

| S | B | mlx_fast | dense_lazy | maxdiff |
|---:|---:|---:|---:|---:|
| 128 | 1 | 0.2133 | 0.2390 | 0.00000 |
| 128 | 4 | 0.3129 | 0.2924 | 0.00002 |
| 128 | 8 | 0.3734 | 0.4015 | 0.00012 |
| 288 | 1 | 0.2550 | 0.2688 | 0.00000 |
| 288 | 4 | 0.3360 | 0.3487 | 0.00006 |
| 288 | 8 | 0.4525 | 0.4889 | 0.00006 |
| 512 | 1 | 0.2658 | 0.2707 | 0.00000 |
| 512 | 4 | 0.4412 | 0.5269 | 0.00006 |
| 512 | 8 | 0.6589 | 0.6839 | 0.00006 |
| 1024 | 1 | 0.3521 | 0.3648 | 0.00000 |
| 1024 | 4 | 0.6763 | 0.6225 | 0.00003 |
| 1024 | 8 | 0.9433 | 1.2804 | 0.00006 |
| 2162 | 1 | 0.5553 | 0.5180 | 0.00000 |
| 2162 | 4 | 1.3960 | 1.5421 | 0.00003 |
| 2162 | 8 | 1.8857 | 2.4490 | 0.00003 |
| 4096 | 1 | 0.7444 | 0.7063 | 0.00006 |
| 4096 | 4 | 2.1717 | 2.3695 | 0.00006 |
| 4096 | 8 | 5.4533 | 4.1316 | 0.00006 |

Threadgroup-size variants did not produce a stable replacement:

| S | B | mlx_fast | t128 | t192 | t256 | t512 |
|---:|---:|---:|---:|---:|---:|---:|
| 288 | 1 | 0.2716 | 0.2586 | 0.2721 | 0.2344 | 0.2645 |
| 288 | 4 | 0.3452 | 0.3442 | 0.3211 | 0.3396 | 0.3409 |
| 288 | 8 | 0.4820 | 0.4626 | 0.5038 | 0.5000 | 0.5494 |
| 1024 | 1 | 0.3363 | 0.3934 | 0.3632 | 0.3439 | 0.3707 |
| 1024 | 4 | 0.6522 | 0.7088 | 0.6978 | 0.6868 | 0.7713 |
| 1024 | 8 | 0.9599 | 1.0351 | 1.0574 | 1.8405 | 0.9214 |
| 2162 | 1 | 0.4266 | 0.5027 | 0.4548 | 0.4438 | 0.4408 |
| 2162 | 4 | 1.1579 | 1.1509 | 1.1624 | 1.4740 | 1.1814 |
| 2162 | 8 | 1.9864 | 2.0272 | 2.0236 | 2.1972 | 2.0018 |
| 4096 | 1 | 0.7697 | 0.8962 | 0.7691 | 0.7162 | 0.7594 |
| 4096 | 4 | 1.7405 | 1.9367 | 2.3383 | 2.5269 | 2.0574 |
| 4096 | 8 | 3.5515 | 4.0800 | 4.0024 | 4.1370 | 3.7493 |

## Decision

Rejected. A dense lazy Metal decode kernel can win selected rows, but it is not
stable enough to replace MLX fast attention in the no-radix batched decode path.
The no-radix runtime should keep using dense MLX fast attention unless a later
kernel clears the full shape matrix.
