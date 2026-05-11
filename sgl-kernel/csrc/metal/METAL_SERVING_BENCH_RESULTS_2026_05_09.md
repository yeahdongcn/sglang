# MLX/Metal Serving Benchmark Results - 2026-05-09

Workload:

```text
python -m sglang.bench_one_batch
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
  --trust-remote-code --tp-size 1
  --batch-size 1 4 8 --input-len 256 --output-len 32 --warmups 1
```

The benchmark runner uses `disable_radix_cache=True`, so these numbers measure
the no-radix MLX path. They are useful as a regression guard, but they do not
prove the paged Metal radix-cache path is faster than MLX fast attention.

## Baseline Commit `c4903e67bf50f50286406273dedc52190fe9011f`

Refreshed from `/tmp/sglang-c490` on 2026-05-09 14:28.

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 1163.72 | 35.15 | 259.88 |
| 4 | 1064.77 | 63.51 | 393.71 |
| 8 | 1096.54 | 110.37 | 558.80 |

Earlier same-day baseline snapshot, before the refresh:

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 1285.83 | 39.65 | 292.60 |
| 4 | 1206.72 | 71.69 | 445.74 |
| 8 | 1217.15 | 107.43 | 575.07 |

## Current Branch Before No-Radix Guard

Current paged-cache path with page size 16, before restoring the no-radix
contiguous-cache guard:

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 582.39 | 6.60 | 56.05 |
| 4 | 696.77 | 24.59 | 172.89 |
| 8 | 745.71 | 47.85 | 289.77 |

## Current Branch After No-Radix Guard

Current branch after routing `disable_radix_cache=True` through the contiguous
MLX cache path again, with benchmark-only scheduler slot overhead skipped.
Refreshed on 2026-05-09 14:29.

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 1154.74 | 35.05 | 257.14 |
| 4 | 1050.64 | 64.43 | 399.39 |
| 8 | 1103.41 | 109.20 | 555.08 |

## Current Branch After Hybrid Fast-Path Cleanup

Current branch after:

- keeping the no-radix path on contiguous MLX caches,
- disabling unused paged-context dispatch checks in runner-installed attention
  wrappers, and
- preserving compact/eager radix side-store sync only for radix-enabled serving.

Refreshed on 2026-05-09 15:55.

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 1223.30 | 35.09 | 261.91 |
| 4 | 1136.21 | 66.30 | 410.38 |
| 8 | 1114.56 | 113.13 | 567.82 |

Same-window baseline checks used for comparison:

| Baseline run | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| 2026-05-09 15:53 | 1 | 1189.64 | 34.78 | 257.66 |
| 2026-05-09 15:38 | 4 | 963.54 | 49.15 | 308.89 |
| 2026-05-09 15:38 | 8 | 816.43 | 81.64 | 408.21 |

## Current Branch After Batched Radix Sync

Current branch after batching radix side-store K/V sync across all layers and
recomputing from the full prompt when a radix partial hit has a small cached
prefix relative to the uncached suffix. No-radix remains on the contiguous MLX
cache path; the first full sweep B1 was a low outlier, so B1 below is the
immediate same-window repeat.

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 1 | 1143.64 | 34.27 | 257.21 |
| 4 | 977.24 | 59.22 | 366.55 |
| 8 | 990.37 | 101.41 | 509.12 |

Same-window baseline checks used for comparison:

| Baseline run | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| 2026-05-09 16:10 | 1 | 988.62 | 34.39 | 244.09 |
| 2026-05-09 16:10 | 4 | 900.88 | 58.98 | 355.94 |
| 2026-05-09 16:10 | 8 | 900.24 | 98.97 | 469.61 |

Post page-alignment spot check for the no-radix path:

| Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---:|---:|---:|---:|
| 4 | 854.34 | 67.74 | 362.03 |

Same-window rerun after rebuilding the Metal extension with a consistent
128-thread decode dispatch setting. The no-radix path still routes active
compute through contiguous MLX caches.

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` | 1 | 1289.02 | 40.82 | 296.02 |
| current | 1 | 1299.82 | 41.31 | 300.01 |
| baseline `c4903e67` | 4 | 1202.46 | 74.50 | 450.60 |
| current | 4 | 1220.13 | 74.71 | 458.39 |
| baseline `c4903e67` | 8 | 1229.92 | 122.91 | 625.42 |
| current | 8 | 1236.39 | 125.29 | 627.31 |

## Radix Cache Serving Probe

Probe command used `SGLANG_USE_MLX=1` with radix cache enabled and
`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` so allocator warnings did not
abort the server. Prompt was:

```text
Explain why paged attention caches need slot mappings and block tables in a language model server.
```

repeated 24 times for the short probe and 120 times for the long probe, with
`temperature=0` and `max_new_tokens=20`.

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first | 434 | 0 | 2.52 | coherent |
| current short hit 1 | 434 | 432 | 1.46 | coherent |
| current short hit 2 | 434 | 432 | 1.42 | coherent |
| baseline short first | 434 | 0 | 3.14 | coherent |
| baseline short hit 1 | 434 | 433 | 4.54 | coherent |
| baseline short hit 2 | 434 | 433 | 1.97 | coherent |
| current long partial hit | 2162 | 432 | 7.07 | coherent |
| current long full hit | 2162 | 2160 | 2.25 | coherent |
| baseline long partial hit | 2162 | 433 | 5.08 | coherent |
| baseline long full hit | 2162 | 2161 | 2.53 | coherent |

After the batched sync and small-prefix recompute changes:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first, post-batch-sync | 433 | 0 | 2.47 | coherent |
| current short hit 1, post-batch-sync | 433 | 432 | 1.24 | coherent |
| current short hit 2, post-batch-sync | 433 | 432 | 0.84 | coherent |
| current long partial hit, post-batch-sync | 2161 | 448 | 3.35 | coherent |
| current long full hit, post-batch-sync | 2161 | 2160 | 1.46 | coherent |

After page-aligning the MLX scheduler-visible pool size, the same probe was
rerun with strict idle memory checks left enabled. No idle leak warning appeared
after the server returned to idle.

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first, strict-idle on | 434 | 0 | 2.61 | coherent |
| current short hit 1, strict-idle on | 434 | 432 | 1.51 | coherent |
| current short hit 2, strict-idle on | 434 | 432 | 1.48 | coherent |
| current long partial hit, strict-idle on | 2162 | 432 | 3.87 | coherent |
| current long full hit, strict-idle on | 2162 | 2160 | 1.51 | coherent |

Latest radix-enabled rerun after the 128-thread Metal rebuild, again with
strict idle checks left enabled:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first, latest | 434 | 0 | 2.33 | coherent |
| current short hit 1, latest | 434 | 432 | 0.93 | coherent |
| current short hit 2, latest | 434 | 432 | 0.74 | coherent |
| current long partial hit, latest | 2162 | 432 | 2.91 | coherent |
| current long full hit, latest | 2162 | 2160 | 0.94 | coherent |

After routing long/batched h128/b16 radix decode through direct Metal paged
decode, while keeping single-request short hits on gathered MLX fast decode, a
fresh radix-enabled probe on port 43446 remained coherent:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first, direct-metal guard | 434 | 0 | 2.42 | coherent |
| current short hit 1, direct-metal guard | 434 | 432 | 1.40 | coherent |
| current short hit 2, direct-metal guard | 434 | 432 | 0.96 | coherent |
| current long partial hit, direct-metal guard | 2162 | 432 | 3.35 | coherent |
| current long full hit, direct-metal guard | 2162 | 2160 | 1.48 | coherent |

A later short hit after an idle gap was an outlier at 4.74 s, so the short-hit
serving path still needs repeatability work even though the routing guard avoids
the always-direct-Metal short B1 regression seen in raw microbenchmarks.

After removing the all-layer `mx.stack` temporary from radix side-store sync
and passing per-layer K/V arrays directly to `set_kv_all_layers`, a fresh
same-window no-radix comparison remained ahead of `/tmp/sglang-c490` despite
low available memory:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` | 1 | 1013.06 | 30.57 | 229.27 |
| current no-stack sync | 1 | 1091.52 | 31.83 | 236.52 |
| baseline `c4903e67` | 4 | 1001.40 | 58.04 | 364.63 |
| current no-stack sync | 4 | 1005.61 | 62.36 | 380.94 |
| baseline `c4903e67` | 8 | 989.70 | 94.01 | 485.59 |
| current no-stack sync | 8 | 1022.87 | 103.17 | 521.31 |

The radix-enabled probe on port 43449 was coherent with strict idle checks left
at their default setting:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current short first, no-stack sync | 434 | 0 | 2.56 | coherent |
| current short hit 1, no-stack sync | 434 | 432 | 1.14 | coherent |
| current short hit 2, no-stack sync | 434 | 432 | 0.99 | coherent |
| current long partial hit, no-stack sync | 2162 | 432 | 3.42 | coherent |
| current long full hit, no-stack sync | 2162 | 2160 | 1.13 | coherent |
| current short hit after 10s idle, no-stack sync | 434 | 432 | 4.65 | coherent |

The delayed short hit still reproduces as an outlier. A zero-copy materialized
prefix-cache experiment was rejected because it regressed the same serving probe
severely: short hits rose to about 3 s, long full hit to about 16 s, and the
delayed short hit to about 22 s.

After releasing active per-request MLX contiguous caches on scheduler idle
without clearing the radix paged KV side-store, a fresh radix-enabled probe on
port 43460 improved the delayed-idle behavior. The server was launched with
`SGLANG_USE_MLX=1`, radix cache enabled, default strict idle checks, and
`--enable-request-time-stats-logging`.

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current idle-cleanup short first | 434 | 0 | 1.78 | coherent |
| current idle-cleanup short hit 1 | 434 | 432 | 1.13 | coherent |
| current idle-cleanup short hit 2 | 434 | 432 | 0.77 | coherent |
| current idle-cleanup long partial hit | 2162 | 432 | 2.91 | coherent |
| current idle-cleanup long full hit | 2162 | 2160 | 0.94 | coherent |
| current idle-cleanup short hit after 10s idle | 434 | 432 | 1.84 | coherent |
| current idle-cleanup short hit after second 10s idle | 434 | 432 | 2.97 | coherent |

A current-only no-radix regression check after idle cleanup measured:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| current idle-cleanup check | 1 | 1292.13 | 39.32 | 290.97 |
| current idle-cleanup check | 4 | 1228.69 | 73.20 | 455.66 |
| current idle-cleanup check | 8 | 1234.58 | 121.66 | 623.80 |

A narrow batch-1 long-context GQA2 raw decode dispatch gate was tested after
this no-radix run and rejected: the first B1/S2162 run looked better, but a
length sweep plus focused rerun did not hold the gain, so the runtime selection
gate was reverted.

A follow-up delayed-idle control on port 43461 called `/freeze_gc` before the
10 s idle gap. It did not remove the residual variance: the delayed short hit
still measured 3.20 s with `cached_tokens=432`, so Python GC is not a sufficient
explanation for the remaining delayed cached-prefill jitter.

Focused profiling on port 43462 showed the delayed short-hit outlier is inside
the `self.model(...)` phase of the pool-backed prefix path, not materialization
or side-store sync. Immediate short hit (`prefix=432`, `new=2`) measured
`model_ms=17.94`, `materialize_ms=0.37`, `eval_ms=368.96`, `sync_ms=4.78`;
the delayed hit after 10 s idle measured `model_ms=1881.68`,
`materialize_ms=0.26`, `eval_ms=177.59`, `sync_ms=16.81`.

The focused raw short-hit microbench in
`METAL_MICROBENCH_RESULTS_RADIX_SHORT_HIT_Q2.md` shows why short prefix hits
must not be blindly moved to Metal paged prefill: for B1 prefix 432 and query
length 2, `metal_prefill_paged_prefix_432` is `0.9885 ms` while
`mlx_fast_dense_prefix_432` is `0.2701 ms`. For a larger query length 434 with
the same prefix, the direction flips: Metal paged prefix prefill is
`2.7881 ms` versus MLX dense prefix `3.3111 ms`.

The runtime now uses this evidence as a thresholded radix path: short prefix
hits stay on the pool-backed MLX dense route, while large radix prefix/suffix
shapes use the paged Metal prefill context and materialize the resulting paged
slots back to the active contiguous decode cache. A direct
same-run comparison in `METAL_SERVING_DIRECT_RADIX_PREFILL_THRESHOLD_BENCH.md`
used prefix 1392, new suffix 1781, and the same loaded Qwen3-0.6B runner:

| Variant | Wall s | Next token |
|---|---:|---:|
| Pool-backed prefix hit, threshold disabled | 4.3153 | 4784 |
| Paged Metal prefix hit, thresholded route | 3.3812 | 4784 |

This is a `1.276x` same-process speedup for that large radix prefix-hit shape,
with identical next token. The first serving probe after the change confirmed
the routing split: short repeat hits remained `paged=False`, while a
`prompt_tokens=3173`, `cached_tokens=1392`, `new=1781` partial-prefix hit used
`paged=True`.

The same change exposed and fixed a cache-pool growth bug: reused contiguous
caches could have a 4096-token physical buffer while their logical
`max_seq_len` had been raised above that size, causing large paged-slot
materialization to skip growth. `ContiguousKVCache.update_and_fetch` and
`write_token` now grow based on actual buffer capacity.

A direct short-idle control in
`METAL_SERVING_DIRECT_RADIX_SHORT_IDLE_FORCE_PAGED_BENCH.md` did not reproduce
the server delayed-hit outlier in a standalone runner and rejected forcing paged
Metal prefill for short hits after idle: prefix 368/new 12 pool-backed delayed
hit was `0.2913 s`, while forced paged after the same 10 s idle gap was
`0.4033 s`, with the same next token.

The broad GQA2 decode rerun in `METAL_MICROBENCH_RESULTS_GQA2_BROAD_RERUN.md`
was also rejected. It did not clear dense MLX and regressed important batched
long decode cases: B4/S2162 `metal_paged` was `1.2338 ms` versus dense MLX
`1.0815 ms`, and B8/S2162 was `1.9925 ms` versus dense MLX `1.9349 ms`.
The experimental dispatch was reverted and the Metal extension rebuilt.

A follow-up shared-query decode shader attempt is recorded in
`METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md`. It preserved correctness but
regressed the long batched decode gate: B4/S2162 `metal_paged` was `1.3122 ms`
and B8/S2162 was `2.1160 ms`, versus dense MLX fast `1.0204 ms` and
`1.6015 ms`. The shader change was reverted and the extension rebuilt.

A Qwen-shape two-pass long paged decode attempt is recorded in
`METAL_MICROBENCH_RESULTS_DECODE_2PASS.md`. It matched dense MLX in a direct
B2/S1031 correctness probe (`max_abs_diff=0.000122`) but regressed raw decode:
B1/B4/B8 S2162 `metal_paged` was `0.7473/1.2245/2.2871 ms`, so that dispatch
was also reverted.

The radix prefill threshold was refined with
`METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md`. The old
`new_token_count >= 384` gate was too broad for single-request prefix256
hits: prefix256/new384 and prefix256/new512 were still faster on dense MLX.
The runtime now selects paged Metal prefill for suffixes of at least 768 tokens,
or for prefix/suffix pairs with both prefix and suffix at least 384 tokens.

## Lazy MLX Metal Decode Wrapper

`METAL_MICROBENCH_RESULTS_DECODE_LAZY_MLX_KERNEL.md` records a follow-up
implementation that expresses the h128/block16 fp16 paged decode kernel through
`mx.fast.metal_kernel` instead of the native Python wrapper. This lets MLX own
the output allocation lazily and avoids materializing a zero-filled output array
before dispatch.

The new path improves shuffled radix-style direct paged decode in the measured
B1/B4/B8 rows. After the direct block-table indexing fix, S2162 lazy radix
decode was `0.4773/1.2585/2.7048 ms` versus the native wrapper at
`0.6801/2.0629/2.9883 ms`, and much faster than gathered radix MLX fast at
`1.3844/7.1466/14.0863 ms`. The batch-1 radix length sweep now routes supported
fp16/head_dim=128/block_size=16 decode shapes through lazy direct paged decode
at S32 and above; it beats gather+MLX from S32 through S2162, while the tiny
S16 row remains on gather+MLX. It still does not clear dense MLX fast for every
contiguous no-radix shape, so this is an
accepted wrapper-overhead reduction and radix decode improvement but not a
reason to remove the serving hybrid guard.

A fresh no-radix same-window check after this thresholded paged-prefill change
still beats `/tmp/sglang-c490` under the slower current machine state:

After the lazy decode address-space fix and all-supported-shape radix routing,
the no-radix `bench_one_batch --batch-size 1 4 8 --input-len 256
--output-len 32 --warmups 1` rerun was variance-sensitive but did not show a
stable regression. In the full same-window sweep, current was ahead at B1/B4
and behind at B8; immediate isolated B8 reruns flipped back ahead.

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline full sweep | 1 | 901.10 | 23.91 | 187.10 |
| current full sweep | 1 | 859.07 | 25.65 | 192.07 |
| baseline full sweep | 4 | 753.14 | 52.00 | 302.59 |
| current full sweep | 4 | 853.09 | 50.37 | 315.98 |
| baseline full sweep | 8 | 917.77 | 86.60 | 454.49 |
| current full sweep | 8 | 883.85 | 83.61 | 438.43 |
| baseline isolated rerun | 8 | 890.13 | 86.06 | 445.43 |
| current isolated rerun | 8 | 931.96 | 89.90 | 464.46 |

The current radix-enabled serving probe on port 43463 remained coherent after
the lazy decode route was expanded to tiny batch-1 block tables:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current lazy-index short first | 578 | 0 | 2.56 | coherent |
| current lazy-index short hit 1 | 578 | 576 | 1.20 | coherent |
| current lazy-index short hit 2 | 578 | 576 | 0.93 | coherent |
| current lazy-index long partial hit | 3266 | 576 | 5.65 | coherent |
| current lazy-index long full hit | 3266 | 3264 | 2.78 | coherent |

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` | 1 | 713.27 | 20.52 | 157.75 |
| current thresholded radix prefill | 1 | 731.59 | 21.49 | 159.17 |
| baseline `c4903e67` | 4 | 688.52 | 37.27 | 237.92 |
| current thresholded radix prefill | 4 | 798.85 | 46.95 | 295.75 |
| baseline `c4903e67` | 8 | 655.84 | 75.68 | 358.89 |
| current thresholded radix prefill | 8 | 858.27 | 80.68 | 423.71 |

After deferring the no-radix batched attention wrapper until batch decode and
keeping it installed across the decode burst, the current no-radix path again
beats a same-window `/tmp/sglang-c490` baseline:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` | 1 | 804.73 | 23.05 | 170.96 |
| current lazy no-radix wrapper | 1 | 902.42 | 23.75 | 184.62 |
| baseline `c4903e67` | 4 | 808.44 | 46.05 | 281.33 |
| current lazy no-radix wrapper | 4 | 893.17 | 49.36 | 319.36 |
| baseline `c4903e67` | 8 | 802.01 | 73.06 | 384.61 |
| current lazy no-radix wrapper | 8 | 909.61 | 87.32 | 447.54 |

The current radix path now survives the short and long probes that previously
crashed with Metal command-buffer OOM. After batching K/V sync and avoiding
cache-gather/materialization for small partial prefixes, both full-prefix and
long partial-prefix probes are ahead of the `c4903e67` baseline in this session.

After replacing the radix side-store all-layer MLX indexed assignment with a
native all-layer scatter loop, the focused sync microbench improved again:

| Sync path | Tokens | Layers | Median ms |
|---|---:|---:|---:|
| previous list assignment | 1 | 28 | 1.4097 |
| native all-layer scatter | 1 | 28 | 0.8832 |
| previous list assignment | 4 | 28 | 1.4123 |
| native all-layer scatter | 4 | 28 | 1.1346 |
| previous list assignment | 256 | 28 | 11.1655 |
| native all-layer scatter | 256 | 28 | 9.5193 |

No-radix same-window checks under the current low-memory run were noisy but did
not show a stable regression. The full B1/B4/B8 current run measured
`162.80 / 257.24 / 364.49` total tok/s versus `/tmp/sglang-c490`
`163.78 / 259.48 / 357.58`; the focused reruns measured current B1 at `165.82`
versus baseline B1 at `162.35`, and current B4 at `261.07` versus the full-run
baseline B4 at `259.48`.

The first E2E radix probe after native all-layer scatter exposed a dtype
mismatch: Qwen-side K/V can arrive as `bfloat16`, while the Metal-compatible
paged side-store is `float16`. The helper now casts each layer to the cache
dtype before native scatter. Focused tests passed after the fix:
`sgl-kernel/tests/test_metal.py` passed 37 tests, and the MLX
attention/cache/model-runner subset passed 85 tests.

A mixed-dtype sync spot check, recorded in
`METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md`, confirms the native
route remains faster for the serving-shaped `bfloat16 -> float16` sync:
1-token sync is `1.2174 ms` vs list assignment `1.3566 ms`, 4-token is
`1.5727 ms` vs `1.6968 ms`, and 256-token is `12.6477 ms` vs `14.6236 ms`.

The radix-enabled serving probe on port 43465 was coherent after the dtype fix
with strict idle checks enabled:

| Branch | Prompt tokens | Cached tokens | E2E latency (s) | Output |
|---|---:|---:|---:|---|
| current native-scatter dtype-fix short first | 434 | 0 | 2.58 | coherent |
| current native-scatter dtype-fix short hit 1 | 434 | 432 | 1.17 | coherent |
| current native-scatter dtype-fix short hit 2 | 434 | 432 | 0.78 | coherent |
| current native-scatter dtype-fix long partial hit | 2162 | 432 | 2.88 | coherent |
| current native-scatter dtype-fix long full hit | 2162 | 2160 | 0.95 | coherent |
| current native-scatter dtype-fix short hit after 10s idle | 434 | 432 | 3.09 | coherent |

The immediate radix hit path is still improved and coherent, but the delayed
short-hit outlier remains a watch item. It is not a scatter correctness issue;
earlier profiling placed the outlier inside the pool-backed model forward.

A follow-up idle-gap materialized-prefix experiment was tested and rejected. A
profiled short-only probe on port 43466 improved the delayed short hit to
`1.52 s` and moved the expensive work out of `model_ms`, but the real
short/long probe sequence without profiling did not hold the gain: after a long
full-prefix hit, the delayed short hit measured `3.40 s` on port 43467. The
experiment was reverted because it was sequence-sensitive and could worsen the
normal radix probe.

A fresh same-window no-radix check after the dtype fix was again
variance-sensitive but did not show a stable regression. B1/B4 were ahead in
the full sweep, while B8 flipped ahead in an isolated rerun:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` full sweep | 1 | 679.74 | 21.91 | 160.66 |
| current native-scatter dtype-fix full sweep | 1 | 808.12 | 20.59 | 161.27 |
| baseline `c4903e67` full sweep | 4 | 680.66 | 40.37 | 255.71 |
| current native-scatter dtype-fix full sweep | 4 | 702.67 | 40.50 | 258.43 |
| baseline `c4903e67` full sweep | 8 | 710.73 | 68.15 | 360.24 |
| current native-scatter dtype-fix full sweep | 8 | 697.61 | 67.20 | 354.95 |
| baseline `c4903e67` isolated rerun | 8 | 715.32 | 69.34 | 361.71 |
| current native-scatter dtype-fix isolated rerun | 8 | 720.70 | 69.29 | 363.97 |

After adding an equal-length no-radix batched contiguous-cache route, the
multi-request no-radix guard improved, but batch-1 did not clear the refreshed
baseline in the final rerun:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| baseline `c4903e67` same-window sweep | 1 | 990.99 | 25.30 | 197.93 |
| current batched-contiguous final sweep | 1 | 780.65 | 25.50 | 184.92 |
| baseline `c4903e67` same-window sweep | 4 | 847.89 | 47.09 | 305.29 |
| current batched-contiguous final sweep | 4 | 1167.15 | 56.82 | 375.76 |
| baseline `c4903e67` same-window sweep | 8 | 831.90 | 81.58 | 421.64 |
| current batched-contiguous final sweep | 8 | 1223.51 | 83.72 | 496.65 |

The kept no-radix direction is therefore the equal-length batched cache for B4
and B8. The B1 guard remains open/variance-sensitive rather than accepted.

A same-window radix comparison against `/tmp/sglang-c490` showed that the
current default `page_size=16` is still not the right default for short radix
hits. It is coherent, but it only reuses 432 prompt tokens for the 434-token
short prompt and recomputes a 16-token page, while the baseline `page_size=1`
reuses 433 and recomputes one token:

| Branch | Probe | Cached tokens | E2E latency (s) | Output |
|---|---|---:|---:|---|
| baseline `c4903e67` | short first | 0 | 1.40 | coherent |
| current default p16 | short first | 0 | 1.79 | coherent |
| baseline `c4903e67` | short hit 1 | 433 | 2.55 | coherent |
| current default p16 | short hit 1 | 432 | 1.28 | coherent |
| baseline `c4903e67` | short hit 2 | 433 | 0.65 | coherent |
| current default p16 | short hit 2 | 432 | 1.17 | coherent |
| baseline `c4903e67` | long partial hit | 433 | 2.41 | coherent |
| current default p16 | long partial hit | 432 | 3.16 | coherent |
| baseline `c4903e67` | long full hit | 2161 | 0.81 | coherent |
| current default p16 | long full hit | 2160 | 1.01 | coherent |
| baseline `c4903e67` | short hit after 10s idle | 433 | 1.77 | coherent |
| current default p16 | short hit after 10s idle | 432 | 2.98 | coherent |

Current `--page-size 1` restores the token-granular cached-token counts and
improves the default p16 short-hit shape, but it still does not broadly beat the
baseline (`2.24/1.52/0.76/2.81/0.97/2.07 s` for short
first/hit1/hit2, long partial/full, and delayed short). A pure-MLX
block-size-1 side-store/dtype experiment was rejected because it improved one
short hit while regressing long partial and delayed hits.

A decoupled scheduler/storage page-size experiment was also rejected. It kept
the MLX side-store block size at 16 while making scheduler allocation
token-granular. It restored cached-token counts (`433/2161`) but regressed
latency (`2.03/1.28/1.13 s` short first/hit1/hit2, `3.61/1.00 s` long
partial/full, `2.13 s` delayed short), so the change was reverted.

After the accidental interruption, the accepted serving changes were narrowed:

- The no-radix equal-length `BatchedContiguousKVCache` decode route was rejected
  after direct B4/B8 comparisons showed the wrapper-based
  `BatchedDecodeContext` route was faster. No-radix multi-request decode stays
  on the wrapper route, with batch-1 decode using the raw mlx-lm cache path.
- The unchecked full-block `PoolBackedCache` gather was rejected. A same-probe
  run regressed short first/hit1/hit2 from `1.62/1.06/0.97 s` on the checked
  route to `1.82/1.26/1.18 s`, and long partial from `3.04` to `3.87 s`.
- Deferring prompt-side radix side-store sync was rejected. It kept cache-hit
  metadata but shifted pending K/V work into later requests and produced a
  delayed short hit around `3.43 s` in the probe. Prompt K/V sync is immediate
  again; only decode-token side-store sync remains deferred until flush/remove.
- The default MLX page size is now token-granular (`page_size=1`) because the
  p16 route consistently lost one cacheable token per short/long prompt and
  hurt short radix hits.

Current no-radix guard rerun after restoring the wrapper route and the
token-granular default:

| Branch | Batch | Prefill tok/s | Median decode tok/s | Total tok/s |
|---|---:|---:|---:|---:|
| current wrapper route full sweep | 1 | 887.00 | 25.80 | 189.37 |
| current wrapper route full sweep | 4 | 831.58 | 45.23 | 298.17 |
| current wrapper route full sweep | 8 | 821.37 | 78.44 | 413.02 |
| current wrapper route isolated rerun | 4 | 886.21 | 46.17 | 304.27 |

The no-radix result is still variance-sensitive. It is ahead of the nearest
refreshed same-window baseline at B1/B8 and ahead on the isolated B4 rerun, but
the full B4 sweep remains slightly below the same-window baseline.

The latest accepted radix-enabled probe used default `page_size=1`,
immediate prompt sync, deferred decode sync, and a new idle-gap guard that
routes short prefix hits through paged Metal prefill only after the runner has
been idle for at least one second. The server stayed coherent:

| Probe | Cached tokens | E2E latency (s) | Route |
|---|---:|---:|---|
| current p1 short first | 0 | 1.71 | dense no-prefix prefill |
| current p1 short hit 1 | 433 | 0.93 | pool-backed MLX |
| current p1 short hit 2 | 433 | 0.68 | pool-backed MLX |
| current p1 long partial hit | 433 | 2.77 | dense recompute |
| current p1 long full hit | 2161 | 0.89 | pool-backed MLX |
| current p1 delayed short hit | 433 | 1.31 | idle paged prefill |

The idle guard specifically addressed the delayed short-hit outlier: the same
probe before this guard had `model_ms=3207.40` and `e2e_latency=4.21 s`; with
`idle_paged=True`, the delayed hit measured `model_ms=617.34` and
`e2e_latency=1.31 s`.

Two follow-up long-partial experiments were rejected:

- Prioritizing paged Metal prefill for the `prefix=433/new=1729` long partial
  row did not return promptly in serving and was reverted.
- Narrowing the thresholds so that the same row used the pool-backed prefix
  path instead of dense recompute regressed `long_partial` to `4.38 s`. The
  dense recompute route remains the accepted path for this shape.

## 2026-05-10 Warmup Continuation

The retained follow-up change warms the radix Metal all-layer scatter path and a
q=1 paged-prefix prefill primitive when the MLX paged KV pool is initialized.
This targets first-use Metal/MLX materialization cost, not the steady-state
attention algorithm. In a profiled startup comparison, the built-in 6-token
serving warmup side-store sync dropped from about `895 ms` before the change to
about `15 ms` after it. The warmup can be disabled for diagnosis with
`SGLANG_MLX_DISABLE_RADIX_KERNEL_WARMUP=1`.

Clean post-warmup radix-enabled probe on port `43531`:

| Probe | Cached tokens | E2E latency (s) | Route |
|---|---:|---:|---|
| current warmup short first | 0 | 2.02 | dense no-prefix prefill |
| current warmup short hit 1 | 433 | 0.98 | pool-backed MLX |
| current warmup short hit 2 | 433 | 0.69 | pool-backed MLX |
| current warmup long partial hit | 433 | 2.70 | dense recompute |
| current warmup long full hit | 2161 | 0.93 | pool-backed MLX |
| current warmup delayed short hit | 433 | 1.66 | idle paged prefill |

The refreshed `/tmp/sglang-c490` baseline from the same investigation window was
short first/hit1/hit2 `1.50/2.97/0.77 s`, long partial/full `2.61/0.89 s`, and
delayed short `2.35 s`. The current path is therefore ahead on short cache hits
and delayed reuse, but cold first, long partial, and long full are still not
conclusively beyond baseline.

Two additional serving experiments were rejected after profiling:

- Non-blocking side-store scatter (`eager=False`) reduced the logged sync bucket
  but shifted work into later evaluation/decode. The probe regressed to short
  first/hit1/hit2 `2.47/1.01/0.70 s`, long partial/full `2.74/0.98 s`, and
  delayed short `2.38 s`.
- Combined token+cache evaluation for no-prefix radix prefill improved only
  cold short first (`1.72 s`) and regressed the rest of the probe:
  short hit1/hit2 `1.29/1.07 s`, long partial/full `3.56/1.02 s`, and delayed
  short `2.86 s`.
- A warmed retest of temporary radix-wrapper removal for non-paged prefill is
  still rejected. It measured short first/hit1/hit2 `2.18/1.01/0.67 s`, long
  partial/full `2.79/0.88 s`, and delayed short `1.43 s`: long full and delayed
  improved, but cold first and long partial regressed.

Env-gated decode timing (`SGLANG_MLX_PROFILE_TIMING=1`) shows the current cold
short gap is mainly in radix dense prefill evaluation, not single-token decode.
For the 434-token short prompt, radix decode tokens were typically
`40-55 ms/token`, while the dense no-prefix prefill eval was about `1.2 s` in
the profiled run.

No-radix guard after the warmup-only change remains variance-sensitive. The full
B1/B4/B8 sweep measured total throughput `177.72/272.93/389.12 tok/s`; an
isolated B4/B8 rerun recovered to `285.19/392.56 tok/s`, ahead of the saved
same-window baseline rows `281.07/378.43 tok/s`.

## Interpretation

- The large end-to-end regression is not explained by scheduler allocation
  alone. Raw paged Metal attention is slower than `mx.fast.scaled_dot_product_attention`,
  and the paged serving path also pays per-layer scatter/gather integration cost.
- Restoring and tightening the no-radix contiguous-cache path removes the severe
  `bench_one_batch` regression. The current no-radix guard is variance-sensitive
  but has ahead-of-baseline B1/B8 rows and an ahead isolated B4 rerun.
- The radix path is now a hybrid: contiguous MLX caches are used for active
  request compute, while generated K/V is synced into the paged side-store for
  radix reuse. This avoids the raw paged-kernel performance cliff and fixes the
  observed OOM by evaluating radix side-store sync eagerly after batching it
  across layers.
- The raw paged Metal kernels are still not accepted as complete performance
  kernels. The hybrid serving probes are coherent and avoid the original OOM,
  but the default p16 radix path is not ahead of the refreshed p1 baseline.
- Follow-up raw microbenchmarks added h128 online packed-varlen and prefix-aware
  paged prefill kernels. They substantially improve the no-prefix and
  radix-prefix raw prefill cases, but remain several times slower than
  `mx.fast.scaled_dot_product_attention`; the serving path should therefore stay
  hybrid until a Q-block tiled prefill kernel clears that gate.
- Releasing MLX active request caches on idle reduced the delayed short-hit
  outlier from 4.65 s to 1.84-2.97 s in the fresh probe, but the delayed cached
  prefill still shows variance and should remain a follow-up watch item.
- The idle-gap paged short-hit guard further reduced the delayed short hit to
  1.31 s in the latest probe. It is deliberately limited to post-idle short
  prefix hits because normal q=2 short hits are faster on the pool-backed MLX
  route.
- Thresholded paged Metal prefill improves large radix partial-prefix hits
  without regressing short q=2 hits into the slower raw Metal path. This is
  still a hybrid route: active decode continues to use contiguous caches, and
  direct raw paged decode remains behind dense MLX fast attention.
- The lazy MLX Metal decode wrapper reduces direct paged decode overhead and
  improves radix-style block-table rows, but it does not yet justify routing
  all no-radix contiguous decode through paged attention.
