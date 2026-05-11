# P1 Prefix Steel Bridge Microbenchmark

Generated: `2026-05-10`

This note records the focused follow-up after the materialized-prefix serving
experiment regressed radix long-partial and delayed-short rows.

## Raw Kernel Finding

Shape: Qwen3-0.6B attention, `dtype=float16`, `heads=16`, `kv_heads=8`,
`head_dim=128`, `batch=1`, `prefix=433`, suffix as listed.

Before widening the dense-prefix Steel bridge, p1 paged-prefix prefill used the
generic paged prefill kernel:

| Prefix | Suffix | Block | Metal paged prefix ms | Dense MLX fast ms | Ratio |
|---:|---:|---:|---:|---:|---:|
| 433 | 1729 | 1 | 11302.385 | 38.652 | 292.411 |
| 433 | 1729 | 16 | 18.203 | 37.942 | 0.480 |
| 433 | 1 | 1 | 0.839 | 0.274 | 3.062 |
| 433 | 1 | 16 | 0.737 | 0.294 | 2.503 |

After allowing the dense-prefix Steel bridge for p1 caches and gating it to
suffixes with at least 128 query tokens:

| Prefix | Suffix | Block | Metal paged prefix ms | Dense MLX fast ms | Ratio |
|---:|---:|---:|---:|---:|---:|
| 433 | 1729 | 1 | 18.417 | 34.112 | 0.540 |
| 433 | 1 | 1 | 1.401 | 0.321 | 4.369 |

The p1 large-suffix raw kernel gap is fixed at the wrapper level. The small
query row remains unfavorable, so runtime delayed-short routing should not be
expanded to use the dense-prefix bridge for `q_len=1`.

## Serving Check

Forcing the long-partial serving row (`prefix=433`, `new=1729`) to prefer paged
prefill after this raw-kernel fix was still rejected:

| Probe | Cached tokens | E2E latency s | Result |
|---|---:|---:|---|
| short first | 0 | 1.5910 | coherent |
| short hit 1 | 433 | 1.1976 | coherent |
| short hit 2 | 433 | 1.0159 | coherent |
| long partial, forced paged | 433 | 4.1032 | coherent but regressed |
| long full | 2161 | 0.9612 | coherent |
| delayed short | 433 | 2.6380 | coherent but variance remains |

Profile timing for the forced long-partial row showed the serving loss was not
the raw attention kernel alone: `model_ms=2316.65`, `materialize_ms=23.78`,
`eval_ms=877.96`, `total_ms=3219.45`. The runtime routing was reverted to the
accepted dense recompute path for this shape.

Final post-revert radix probe on the same short/long sequence:

| Probe | Cached tokens | E2E latency s | Result |
|---|---:|---:|---|
| short first | 0 | 1.6487 | coherent |
| short hit 1 | 433 | 0.9621 | coherent |
| short hit 2 | 433 | 0.6979 | coherent |
| long partial, dense recompute | 433 | 2.9215 | coherent |
| long full | 2161 | 0.9311 | coherent |
| delayed short | 433 | 2.4485 | coherent, still variance-sensitive |

No-radix final check with the server stopped:

| Batch | Total throughput tok/s |
|---:|---:|
| 1 | 193.32 |
| 4 | 302.96 |
| 8 | 419.86 |

Conclusion: keep the p1 dense-prefix Steel bridge for large suffixes already
routed through paged prefill, but do not let it override the radix small-prefix
dense-recompute guard. The next radix serving gap is model/eval overhead for
prefix-hit prefill, not the raw p1 prefix attention kernel alone.
