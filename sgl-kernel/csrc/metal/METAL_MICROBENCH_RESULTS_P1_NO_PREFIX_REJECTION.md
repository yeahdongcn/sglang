# P1 No-Prefix Prefill Follow-up

Generated during the 2026-05-10 continuation after the radix kernel warmup
change.

## Raw P1 Microbench

Command:

```bash
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python \
  sgl-kernel/csrc/metal/bench_metal_attention_micro.py \
  --block-size 1 --decode-batches 1 --decode-seq-lens 16 \
  --prefill-batches 1 --prefill-seq-lens 434 2162 \
  --prefill-iters 5 --iters 5 --warmups 3 \
  --scatter-tokens 1 --output /tmp/p1_prefill_micro.md
```

Key p1 no-prefix rows:

| Case | Variant | Batch | Seq Len | Median ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|
| prefill | metal_prefill_paged_no_prefix | 1 | 434 | 1.2707 | 1.1998 | 1.6409 |
| prefill | metal_flash_varlen | 1 | 434 | 1.1816 | 1.1406 | 1.6396 |
| prefill | mlx_fast_dense | 1 | 434 | 1.4035 | 1.3576 | 4.8005 |
| prefill | metal_prefill_paged_no_prefix | 1 | 2162 | 22.1840 | 21.6464 | 22.9300 |
| prefill | metal_flash_varlen | 1 | 2162 | 21.5775 | 18.8893 | 27.1503 |
| prefill | mlx_fast_dense | 1 | 2162 | 34.6772 | 29.1891 | 44.8956 |

The raw p1 no-prefix attention rows are now faster than MLX fast attention, so
this looked like a plausible cold-radix direction.

## Serving Experiment Rejection

A temporary env-gated serving patch forced prefix-0 radix prefill through the
paged context and allowed the wrapper to call `prefill_attention_paged` for
no-prefix contexts:

```bash
SGLANG_MLX_USE_PAGED_NO_PREFIX_PREFILL=1
```

It was rejected. Serving latencies on port `43543` were:

| Probe | Cached tokens | E2E latency (s) | Notes |
|---|---:|---:|---|
| short first | 0 | 3.8550 | `paged=True`, worse than retained path |
| short hit 1 | 433 | 0.7201 | good row |
| short hit 2 | 433 | 2.4037 | decode outlier |
| long partial | 433 | 2.6628 | coherent |
| long full | 2161 | 2.9478 | regressed badly |

Profiled cold short prefill showed the raw route did not translate to serving:
`model_ms=866.19`, `materialize_ms=31.25`, `eval_ms=401.10`, total prefill
`1298.70 ms`. More importantly, active decode after materializing from the
paged side store often jumped to `100-260 ms/token`. The runtime therefore keeps
the dense no-prefix prefill route and uses paged Metal prefill only for accepted
prefix-hit cases.

## Other Rejected Follow-ups

- Capacity-aware cache-pool reuse that shrank overlarge active caches on reuse
  regressed the profiled sequence. The delayed short hit measured `4.27 s`, so
  the runtime keeps the existing reuse behavior.
- 128-token contiguous-cache capacity alignment improved neither the radix gate
  nor delayed stability. No-radix `bench_one_batch` measured
  `180.10/281.11/388.38 tok/s`, but radix serving measured short
  first/hit1/hit2 `2.177/1.006/0.672 s`, long partial/full
  `2.809/0.897 s`, and delayed short `5.073 s`.
- Forcing active contiguous caches to float16 is slower for this bfloat16 model:
  a direct 434-token model call measured about `770.8 ms` median with fp16 cache
  versus `559.1 ms` with the existing bfloat16 active cache.
- Standard MLX `KVCache` was only a small direct-call improvement
  (`546.0 ms` median versus `554.0 ms` for the current contiguous cache on the
  434-token synthetic row) and does not support the current batched decode
  wrapper contract.
- `--skip-server-warmup` is not a cold-latency fix. It moved paged-pool
  initialization into the first user request and measured short first/hit
  `2.42/2.83 s`.

## Current Interpretation

Direct `MlxModelRunner.prefill()` calls do not reproduce the serving cold gap:
with the 434-token probe, no-radix prefill was about `602 ms`, radix with a
4096-slot p1 pool was about `577 ms`, and radix with a 65000-slot p1 pool was
about `410 ms` in the direct harness. The remaining cold serving gap is therefore
more likely in serving warmup/cleanup/scheduler state around the first real
request than in the core runner prefill call.
