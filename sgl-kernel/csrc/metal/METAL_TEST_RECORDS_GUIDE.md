# Metal Test Records Guide

Date: 2026-05-11

Purpose: classify benchmark, correctness, and experiment records so future
agents can add evidence consistently and retrieve the right artifact quickly.

Read this after `AGENTS.md` and before writing a new benchmark or test result.

## Record Categories

| Category | Use for | Primary files | Required fields |
|---|---|---|---|
| `correctness-gate` | Unit tests, focused pytest gates, coherent generation smokes | `METAL_COMPLETION_AUDIT_2026_05_10.md`, `METAL_FINAL_PERF_RESULTS_2026_05_11.md`, specific result artifact | Command, environment, target commit/worktree, pass/fail count, warnings, covered paths |
| `serving-benchmark` | `bench_one_batch`, server latency probes, radix on/off serving comparisons | `METAL_FINAL_PERF_RESULTS_2026_05_11.md`, `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`, `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` | Baseline source, current source, workload, radix on/off, batch/probe, raw output files, percent deltas |
| `raw-microbench` | Direct Metal kernels versus dense MLX/gathered MLX microbench rows | Specific `METAL_MICROBENCH_RESULTS_*.md`, `FULL_FLASH_ATTENTION_METAL_PLAN.md` if the gate changes | Dtype, heads, KV heads, head dim, block size, sequence length, batch, layout, iterations, median/mean/min/max |
| `cache-behavior` | Radix cache, paged KV cache, p1/p16 cache layout, scatter/sync behavior | `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`, p1/radix/scatter-specific artifacts | Page size, prefix lengths, cache capacity, sync strategy, hit/miss shape, dtype, serving impact |
| `rejection-record` | Probes that were tested, failed a gate, and were reverted or left unrouted | Most specific result artifact, then `METAL_DOC_INDEX.md` `Do Not Retry Without New Evidence` | Hypothesis, changed route/kernel, failed row, regression size, output path, revert/routing status |
| `api-contract` | Public wrapper behavior, unsupported options, shape/dtype coverage | `README.md`, `sgl-kernel/tests/`, relevant plan/result artifact | API name, supported layout, unsupported options, tests added/updated |
| `historical-reference` | Older plans and benchmark commands that explain provenance | `FLASH_ATTENTION_METAL_PLAN.md`, `BENCHMARK_RESULTS.md` | Date, why it is historical, current replacement doc |

## Category Crosswalk For Existing Records

| Artifact family | Main category | Secondary categories |
|---|---|---|
| `METAL_FINAL_PERF_RESULTS_2026_05_11.md` | `serving-benchmark` | `correctness-gate` |
| `METAL_COMPLETION_AUDIT_2026_05_10.md` | `correctness-gate` | `serving-benchmark`, `rejection-record` |
| `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` | `serving-benchmark` | `raw-microbench`, `cache-behavior`, `rejection-record` |
| `METAL_SERVING_*.md` | `serving-benchmark` | `cache-behavior`, `rejection-record` |
| `METAL_MICROBENCH_RESULTS_P1_*.md` | `cache-behavior` | `raw-microbench`, `rejection-record` |
| `METAL_MICROBENCH_RESULTS_RADIX_*.md` | `cache-behavior` | `serving-benchmark`, `rejection-record` |
| `METAL_MICROBENCH_RESULTS_DECODE_*.md` | `raw-microbench` | `rejection-record` |
| `METAL_MICROBENCH_RESULTS_GQA2_*.md` | `raw-microbench` | `rejection-record` |
| `METAL_MICROBENCH_RESULTS_STEEL_*.md` | `raw-microbench` | `cache-behavior` |
| `FULL_FLASH_ATTENTION_METAL_PLAN.md` | `historical-reference` | `raw-microbench`, `rejection-record` |
| `FLASH_ATTENTION_METAL_PLAN.md` | `historical-reference` | `api-contract` |
| `README.md` | `api-contract` | `historical-reference` |

## Placement Rules

- Add a focused new file when the result is a new independent kernel family,
  route, or cache strategy: `METAL_MICROBENCH_RESULTS_<TOPIC>.md` or
  `METAL_SERVING_<TOPIC>.md`.
- Append a dated section when extending an active ledger such as
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`.
- Update `METAL_FINAL_PERF_RESULTS_2026_05_11.md` only when the accepted final
  serving claim changes.
- Update `FULL_FLASH_ATTENTION_METAL_PLAN.md` when a result changes an
  acceptance gate or remaining-work direction.
- Update `METAL_DOC_INDEX.md` whenever a new artifact should be discoverable or
  a rejected idea should be added to `Do Not Retry Without New Evidence`.
- Update `AGENTS.md` only when the operating instructions or current accepted
  state change.

## Required Record Template

Use this section shape for new records or dated continuation notes:

```markdown
## <YYYY-MM-DD> <Short Result Name>

- Category: `<category>`
- Status: `accepted-serving` | `accepted-raw-subpath` | `rejected` |
  `historical` | `open-raw-gate`
- Target state: `<commit plus local-change description>`
- Baseline: `<commit/path/run id, or none>`
- Workload: `<batch/probe/seq/prefix/radix/page-size details>`
- Command: `<exact command or script>`
- Output files: `<raw jsonl/md/log paths>`
- Correctness: `<command and pass/fail, or why not applicable>`
- Metrics: `<table or bullets with baseline/current/percent>`
- Decision: `<what changed, what remains open, what was reverted>`
- Search tags: `<short comma-separated tags>`
```

For long commands, use fenced `text` blocks. For tables, keep columns stable:
`Row`, `Baseline`, `Current`, `Change`, `Status`.

## Percent Rules

- Throughput metrics: positive means current is faster.
  `(current - baseline) / baseline * 100`.
- Latency metrics: positive means current is faster.
  `(baseline - current) / baseline * 100`.
- Mixed metrics: state the unit in the column name and avoid combining
  throughput and latency in one average.
- Report very small wins with caution. If the result is within run variance,
  mark it `mixed`, `on-par`, or `needs repeat` instead of `accepted`.

## Agent Workflow

1. Classify the result with `Record Categories`.
2. Pick the destination with `Placement Rules`.
3. Record raw paths and exact commands before summarizing.
4. Compare against the right baseline in the same load window when possible.
5. Tie performance claims to correctness evidence.
6. If the result changes accepted state, update the final/audit/plan/index docs
   listed in the `AGENTS.md` update matrix.
7. If the probe is rejected, record the failed row and add a one-line entry to
   `METAL_DOC_INDEX.md` so future agents do not retry it without new evidence.

## Retrieval Tags

Use these tags consistently in new records:

`correctness-gate`, `serving-benchmark`, `raw-microbench`,
`cache-behavior`, `rejection-record`, `api-contract`,
`historical-reference`, `accepted-serving`, `accepted-raw-subpath`,
`open-raw-gate`, `radix-cache`, `paged-kv`, `p1`, `bf16`, `steel`,
`no-radix`.
