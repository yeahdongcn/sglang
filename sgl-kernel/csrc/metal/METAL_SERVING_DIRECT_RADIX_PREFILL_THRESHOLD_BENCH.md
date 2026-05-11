# Direct Radix Prefill Threshold Benchmark

Generated: `2026-05-09`

Shape: prefix tokens `1392`, new suffix tokens `1781`, total prompt tokens `3173`. Same `MlxModelRunner`, same loaded Qwen3-0.6B model, same paged prefix slots. The old pool-backed path is measured by temporarily raising `_RADIX_PAGED_PREFILL_MIN_NEW_TOKENS`; the new path uses the default threshold `384`.

| Variant | Wall s | Next token |
|---|---:|---:|
| Seed no-prefix prefill | 4.8070 | 2784 |
| Pool-backed prefix hit, threshold disabled | 4.3153 | 4784 |
| Paged Metal prefix hit, default threshold | 3.3812 | 4784 |

Paged/old speedup: `1.276x`.
Same next token for old vs paged: `True`.
