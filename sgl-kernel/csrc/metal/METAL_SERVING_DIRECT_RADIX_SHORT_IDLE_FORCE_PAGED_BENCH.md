# Direct Radix Short Idle Forced-Paged Benchmark

Generated: `2026-05-09`

Shape: prefix tokens `368`, new suffix tokens `12`, total prompt tokens `380`. Same `MlxModelRunner`, same loaded Qwen3-0.6B model, same paged prefix slots. Pool-backed rows temporarily raise `_RADIX_PAGED_PREFILL_MIN_NEW_TOKENS`; forced paged row sets it to `1`.

| Variant | Wall s | Next token |
|---|---:|---:|
| Seed no-prefix prefill | 0.6709 | 198 |
| Pool-backed immediate prefix hit | 0.4058 | 198 |
| Pool-backed prefix hit after 10s idle | 0.2913 | 198 |
| Forced paged prefix hit after 10s idle | 0.4033 | 198 |

Forced-paged/delayed-pool speedup: `0.722x`.
Same next token for delayed pool vs forced paged: `True`.
