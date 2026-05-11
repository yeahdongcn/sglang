# MLX/Metal Documentation Retrieval Index

Date: 2026-05-11

Purpose: make the Metal kernel and MLX KV-cache investigation notes easy for AI
agents to retrieve, route, and update during future optimization work.

## Agent Retrieval Contract

- Agents should load `AGENTS.md` first when their harness supports it. Treat
  this file as the secondary routing index for Metal kernel, MLX attention,
  radix-cache, paged KV-cache, and Apple Silicon benchmark notes.
- Prefer the `Fast Path For Agents` section for current state before opening
  older artifacts.
- Do not treat `historical`, `partially-met`, or `rejected` artifacts as current
  implementation guidance unless a newer benchmark explicitly changes the
  decision.
- Keep performance and correctness linked. The final serving result is accepted
  only with the recorded pytest gate and paired same-window benchmark evidence.
- Search tags: `mlx-metal`, `sglang`, `apple-silicon`, `flashattention`,
  `paged-kv`, `radix-cache`, `metal-kernel`, `serving-baseline`.

## Fast Path For Agents

Start here for most future work:

1. `AGENTS.md` - scoped operating instructions for agents in this directory.
2. `METAL_FINAL_PERF_RESULTS_2026_05_11.md` - final accepted serving-baseline
   numbers and correctness gate.
3. `METAL_COMPLETION_AUDIT_2026_05_10.md` - prompt-to-artifact checklist and
   final serving completion update.
4. `METAL_TEST_RECORDS_GUIDE.md` - categories, templates, and update rules for
   future benchmark/test records.
5. `MLX_METAL_REGRESSION_INVESTIGATION.md` - chronological decision log and
   latest accepted/rejected state.
6. `FULL_FLASH_ATTENTION_METAL_PLAN.md` - architecture plan, acceptance gates,
   and remaining raw-kernel direction.
7. `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` - densest single benchmark
   artifact for final radix/no-radix serving evidence and rejected probes.
8. `BENCHMARK_RESULTS.md` - older command patterns and pre-created environment
   usage.
9. `FLASH_ATTENTION_METAL_PLAN.md` - historical proof-of-concept plan; do not
   treat old per-call env flags as current architecture.

Current headline:

- Serving baseline is met by the retained hybrid MLX/Metal route.
- Final compacted current commit is
  `5b674a5a7d71f386f074396dfab3974ec93055fc`, compared against baseline
  `c4903e67bf50f50286406273dedc52190fe9011f`.
- Raw direct Metal kernels are still not a universal replacement for dense MLX
  fast attention on every contiguous decode/prefill microbenchmark row.
- Correctness gate for the final audit: `156 passed, 6 warnings`.

## Query Routing

| If the agent asks... | Open first | Then open |
|---|---|---|
| How should I work in this directory? | `AGENTS.md` | `METAL_DOC_INDEX.md` |
| How should I categorize a new test record? | `METAL_TEST_RECORDS_GUIDE.md` | `AGENTS.md` |
| What is the final perf result? | `METAL_FINAL_PERF_RESULTS_2026_05_11.md` | `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` |
| Is the overall task complete? | `METAL_COMPLETION_AUDIT_2026_05_10.md` | `MLX_METAL_REGRESSION_INVESTIGATION.md` |
| What should not be retried? | `FULL_FLASH_ATTENTION_METAL_PLAN.md` remaining-kernel section | `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` rejected lists |
| How do I run the same environment? | `BENCHMARK_RESULTS.md` | `FULL_FLASH_ATTENTION_METAL_PLAN.md` |
| What is accepted in serving? | `METAL_FINAL_PERF_RESULTS_2026_05_11.md` | `METAL_COMPLETION_AUDIT_2026_05_10.md` |
| What raw kernel gap remains? | `FULL_FLASH_ATTENTION_METAL_PLAN.md` | dense/GQA2/paged decode result files below |
| What changed for radix/KV cache? | `MLX_METAL_REGRESSION_INVESTIGATION.md` | radix and p1 files below |
| Where are wrapper/API limits? | `README.md` | `FLASH_ATTENTION_METAL_PLAN.md` |

## Status Tags

- `accepted-serving`: retained runtime route or final serving evidence.
- `accepted-raw-subpath`: useful raw kernel or micro path, but not necessarily
  a default serving route.
- `rejected`: tested and reverted or not routed.
- `historical`: useful for provenance, not the current target architecture.
- `open-raw-gate`: raw direct kernel still trails dense MLX on some target rows.
- `command-reference`: environment or benchmark command reference.
- `agent-instructions`: scoped instructions that coding agents should load
  before applying the index.
- `test-record-guide`: taxonomy and templates for future benchmark/test notes.

## Test Record Categories

Use `METAL_TEST_RECORDS_GUIDE.md` before adding or updating benchmark evidence.
The category set is:

- `correctness-gate`
- `serving-benchmark`
- `raw-microbench`
- `cache-behavior`
- `rejection-record`
- `api-contract`
- `historical-reference`

## Canonical Documents

| File | Tags | Retrieval use |
|---|---|---|
| `AGENTS.md` | `agent-instructions`, `accepted-serving`, `open-raw-gate` | Scoped instructions and current-state guardrails for AI agents in this directory. |
| `METAL_DOC_INDEX.md` | `index`, `agent-routing` | This file. Route future agents before opening many artifacts. |
| `METAL_TEST_RECORDS_GUIDE.md` | `test-record-guide`, `agent-routing`, `command-reference` | Category taxonomy, templates, placement rules, and percent conventions for future records. |
| `METAL_FINAL_PERF_RESULTS_2026_05_11.md` | `accepted-serving`, `final-results`, `correctness` | Final percentage table, evidence files, and correctness gate. |
| `METAL_COMPLETION_AUDIT_2026_05_10.md` | `accepted-serving`, `audit`, `checklist`, `open-raw-gate` | Objective checklist and final completion update. |
| `MLX_METAL_REGRESSION_INVESTIGATION.md` | `decision-log`, `accepted-serving`, `rejected`, `open-raw-gate` | Chronological evidence trail and latest state. |
| `FULL_FLASH_ATTENTION_METAL_PLAN.md` | `plan`, `remaining-work`, `rejected`, `open-raw-gate` | Architecture plan, acceptance gates, and remaining kernel direction. |
| `FLASH_ATTENTION_METAL_PLAN.md` | `historical`, `plan` | Historical staged plan; old per-call env flags are not current. |
| `BENCHMARK_RESULTS.md` | `command-reference`, `historical` | Pre-created venv and older benchmark command patterns. |
| `README.md` | `api-reference`, `wrapper-limits` | Public Metal APIs, kernel entry points, compatibility wrapper limits. |

## Serving And Audit Artifacts

| File | Tags | Retrieval use |
|---|---|---|
| `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` | `serving`, `historical`, `rejected` | Long serving-probe history, idle behavior, threshold changes, rejected serving ideas. |
| `METAL_MICROBENCH_RESULTS_SERVING.md` | `serving`, `historical` | Early serving microbenchmark result table. |
| `METAL_SERVING_DIRECT_RADIX_PREFILL_THRESHOLD_BENCH.md` | `serving`, `radix-prefill`, `threshold` | Direct runner evidence for when paged radix prefill helps. |
| `METAL_SERVING_DIRECT_RADIX_SHORT_IDLE_FORCE_PAGED_BENCH.md` | `serving`, `idle`, `rejected` | Short-idle forced-paged prefill evidence; useful to avoid redoing that route. |
| `METAL_COMPLETION_AUDIT_2026_05_09.md` | `audit`, `historical`, `partially-met` | Earlier incomplete audit and rejected-direction table. |
| `METAL_COMPLETION_AUDIT_2026_05_10.md` | `audit`, `accepted-serving`, `open-raw-gate` | Current completion audit; start here for formal status. |

## Radix, Paged KV, And p1 Cache Artifacts

| File | Tags | Retrieval use |
|---|---|---|
| `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` | `accepted-serving`, `radix`, `no-radix`, `rejected`, `open-raw-gate` | Main consolidated radix/no-radix evidence and final serving refresh. |
| `METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md` | `radix-prefill`, `threshold`, `accepted-subpath` | Prefix/suffix threshold evidence for paged Metal prefill routing. |
| `METAL_MICROBENCH_RESULTS_RADIX_SHORT_HIT_Q2.md` | `radix-prefill`, `short-hit`, `rejected` | Evidence against blanket short-hit paged prefill. |
| `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` | `kv-cache`, `radix-sync`, `accepted-subpath` | Native all-layer scatter sync improvement evidence. |
| `METAL_MICROBENCH_RESULTS_BF16_SCATTER.md` | `kv-cache`, `bf16`, `scatter` | BF16 scatter behavior and dtype coverage. |
| `METAL_MICROBENCH_RESULTS_P1_LAZY_DECODE.md` | `p1`, `decode`, `accepted-raw-subpath` | Float16 p1 lazy decode wins versus p1 gather+MLX in raw rows. |
| `METAL_MICROBENCH_RESULTS_P1_BF16_LAZY_DECODE.md` | `p1`, `bf16`, `decode`, `opt-in` | BF16 p1 lazy decode raw wins; default serving route remains guarded. |
| `METAL_MICROBENCH_RESULTS_P1_FLAT_CACHE_REJECTION.md` | `p1`, `kv-cache`, `rejected` | Flat p1 storage/view rejection. |
| `METAL_MICROBENCH_RESULTS_P1_NO_PREFIX_REJECTION.md` | `p1`, `prefill`, `rejected` | No-prefix p1 prefill follow-up rejection. |
| `METAL_MICROBENCH_RESULTS_P1_PREFIX_STEEL_BRIDGE.md` | `p1`, `prefill`, `steel` | P1 prefix Steel bridge data. |

## Decode Kernel Artifacts

| File | Tags | Retrieval use |
|---|---|---|
| `METAL_MICROBENCH_RESULTS.md` | `decode`, `prefill`, `historical` | Baseline early raw Metal microbenchmark table. |
| `METAL_MICROBENCH_RESULTS_4096_SCORE_CACHE.md` | `decode`, `score-cache`, `accepted-subpath` | Larger score cache improvement history. |
| `METAL_MICROBENCH_RESULTS_B1_SHORT_DECODE_THRESHOLD_RERUN.md` | `decode`, `b1`, `threshold`, `rerun` | Short B1 decode threshold rerun evidence. |
| `METAL_MICROBENCH_RESULTS_THREADGROUP_128.md` | `decode`, `threadgroup`, `accepted-subpath` | 128-thread dispatch alignment evidence. |
| `METAL_MICROBENCH_RESULTS_ONLINE_DECODE.md` | `decode`, `online-softmax` | Online decode baseline. |
| `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC.md` | `decode`, `online-softmax`, `vectorized` | Vectorized online decode evidence. |
| `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC_CACHED_Q.md` | `decode`, `online-softmax`, `cached-q` | Cached-Q online decode evidence. |
| `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_128_THREADS.md` | `decode`, `threadgroup`, `rejected` | 128-thread online variant rejection. |
| `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_192_THREADS.md` | `decode`, `threadgroup`, `rejected` | 192-thread online variant rejection. |
| `METAL_MICROBENCH_RESULTS_ONLINE_512_THREADS_QUICK.md` | `decode`, `prefill`, `threadgroup`, `rejected` | 512-thread quick check rejection. |
| `METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md` | `decode`, `gqa`, `rejected` | Shared-query/GQA decode rejection. |
| `METAL_MICROBENCH_RESULTS_DECODE_2PASS.md` | `decode`, `two-pass`, `rejected` | Two-pass long decode rejection. |
| `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE.md` | `decode`, `regscore`, `rejected` | Register-score decode experiment. |
| `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE_COMPARE.md` | `decode`, `regscore`, `compare`, `rejected` | Register-score comparison data. |
| `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE_SWEEP.md` | `decode`, `regscore`, `sweep`, `rejected` | Register-score sweep data. |
| `METAL_MICROBENCH_RESULTS_DECODE_FAST_EXP_COMPARE.md` | `decode`, `fast-exp`, `compare` | Fast-exp compare data. |
| `METAL_MICROBENCH_RESULTS_DECODE_LAZY_MLX_KERNEL.md` | `decode`, `lazy`, `accepted-raw-subpath` | Lazy direct paged decode evidence versus gather+MLX. |
| `METAL_MICROBENCH_RESULTS_DECODE_LAZY_HARNESS_RERUN.md` | `decode`, `lazy`, `rerun` | Harness rerun for lazy decode. |
| `METAL_MICROBENCH_RESULTS_DECODE_NATIVE_LOWER_BOUND.md` | `decode`, `lower-bound`, `native` | Native dispatch lower-bound data. |
| `METAL_MICROBENCH_RESULTS_DENSE_DECODE_LAZY.md` | `decode`, `dense`, `rejected`, `open-raw-gate` | Dense-cache lazy decode benchmark-only result. |
| `METAL_MICROBENCH_RESULTS_GQA2_DECODE_FOCUSED.md` | `decode`, `gqa2`, `focused`, `rejected` | Focused GQA2 decode data. |
| `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_GATE.md` | `decode`, `gqa2`, `gate`, `rejected` | Initial B1 long gate that did not hold. |
| `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LENGTH_SWEEP.md` | `decode`, `gqa2`, `sweep`, `rejected` | Length sweep rejecting B1 GQA2 gate. |
| `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_RERUN.md` | `decode`, `gqa2`, `rerun`, `rejected` | Rerun confirming GQA2 gate rejection. |
| `METAL_MICROBENCH_RESULTS_GQA2_BROAD_RERUN.md` | `decode`, `gqa2`, `rerun`, `rejected` | Broad GQA2 rerun rejection. |
| `METAL_MICROBENCH_RESULTS_CONTIGUOUS_DECODE_SPECIALIZATION.md` | `decode`, `contiguous`, `rejected` | Contiguous paged decode specialization result. |
| `METAL_MICROBENCH_RESULTS_FUSED_KV_DECODE.md` | `decode`, `fused-kv`, `rejected` | Fused scatter+decode h128/block16 rejection. |
| `METAL_MICROBENCH_RESULTS_GATHERED_DECODE.md` | `decode`, `gathered`, `compare` | Gathered decode comparison data. |
| `METAL_MICROBENCH_RESULTS_VEC4_DECODE_FOCUSED.md` | `decode`, `vec4`, `focused` | Vec4 focused decode experiment. |
| `METAL_MICROBENCH_RESULTS_PAGED_SDPA_VECTOR_DECODE.md` | `decode`, `paged`, `sdpa-vector`, `rejected` | Paged SDPA-vector adaptation rejection. |
| `METAL_MICROBENCH_RESULTS_PAGED_2PASS_VECTOR_DECODE.md` | `decode`, `paged`, `two-pass-vector`, `rejected` | Paged two-pass vector split rejection. |

## Prefill And Steel Artifacts

| File | Tags | Retrieval use |
|---|---|---|
| `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL.md` | `prefill`, `online-softmax` | Early online prefill data. |
| `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX.md` | `prefill`, `prefix`, `online-softmax` | Prefix-aware online prefill evidence. |
| `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX_RADIX.md` | `prefill`, `prefix`, `radix`, `online-softmax` | Radix-prefix online prefill data. |
| `METAL_MICROBENCH_RESULTS_STEEL_PREFILL.md` | `prefill`, `steel`, `accepted-subpath` | Steel no-prefix prefill bridge data. |
| `METAL_MICROBENCH_RESULTS_STEEL_PREFIX.md` | `prefill`, `steel`, `prefix`, `accepted-subpath` | Steel prefix-aware prefill bridge data. |
| `METAL_MICROBENCH_RESULTS_STEEL_PREFIX_FAST_EXP.md` | `prefill`, `steel`, `fast-exp` | Steel prefix fast-exp experiment. |

## Do Not Retry Without New Evidence

These ideas have documented regressions or did not hold under rerun:

- Default direct paged bf16 decode serving route.
- One-token p1 prefix-hit prefill through the p1 decode kernel.
- Pre-materializing p1 radix prefixes into contiguous caches for short hits.
- 128-thread bf16 p1 lazy decode replacement.
- Flat p1 authoritative cache storage.
- Tight request-local radix cache capacity for short hits.
- Compact-only radix B1 wrapper eval.
- Per-token slice eval for short B1 wrapper decode.
- Full-state eval for all single-request radix fallback decode.
- Active short-prefix contiguous-cache reuse.
- Raw-attention fallback for radix single-request fallback compute.
- Dense contiguous lazy Metal decode as a broad replacement.
- Dense/GQA2 shared-query decode routes.
- Paged SDPA-vector and paged two-pass vector decode.
- Register-score broad replacement.
- Native fused scatter+decode h128/block16 route.
- Default-on idle paged-prefill guard.

When considering any of these, open the corresponding artifact above and write
down what materially changed before re-running it.

## Update Rules For Future Agents

1. If a new benchmark changes the final serving claim, update
   `METAL_FINAL_PERF_RESULTS_2026_05_11.md`, `METAL_COMPLETION_AUDIT_2026_05_10.md`,
   and this index together.
2. If a new raw kernel clears a previously open row, update
   `FULL_FLASH_ATTENTION_METAL_PLAN.md` and add the artifact to the relevant
   table here.
3. If an idea is rejected, add it to the most specific artifact, then add a
   one-line `Do Not Retry` entry here.
4. Keep tags short and searchable: `decode`, `prefill`, `radix`, `p1`,
   `kv-cache`, `serving`, `bf16`, `steel`, `rejected`, `accepted-serving`,
   `open-raw-gate`.
5. Record command lines and raw output paths. Future agents should not rely on
   prose summaries without evidence file pointers.
