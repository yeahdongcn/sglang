# MPS model-operator contribution pattern

The MPS backend has one execution backend and one authoritative runner:
SGLang's Torch `ModelRunner`. The standard scheduler owns scheduling, request
lifecycle, and Radix state. The Torch `ModelRunner` and standard SRT components
remain authoritative for parameters, request-to-token tables, KV pools, logits,
sampling, and weight updates. Metal or MLX providers are implementation choices
for selected semantic operations; they are not model loaders, workers, cache
managers, or secondary runners.

## Choose the smallest integration level

Use the first level that can deliver the required speedup:

1. Add a generic semantic implementation to an existing `MultiPlatformOp` or
   `sglang/kernels/ops/<domain>` dispatcher. This is preferred for operations
   such as RMSNorm or SiLU because every compatible model can reuse it.
2. Add a model-family provider for an operation whose contract depends on
   typed model modules, such as fused QK normalization, RoPE, KV store, or a
   model-specific attention layout.
3. Add a whole-model provider only when per-layer framework transitions make
   the smaller integrations slower. A whole-model island must still borrow the
   Torch-owned weights and caches, and it must preserve a normal Torch path for
   requests outside its measured contract.

Kernel implementation files belong under `sglang/kernels/ops/<domain>`.
Model discovery, storage contracts, request eligibility, lifecycle, and
telemetry belong in this directory. Do not put SRT model or KV ownership in a
kernel module.

## Register a model family

Add one lazy `MpsModelOperatorSpec` entry in `registry.py`. Architecture names
are only import-routing hints; they are not proof of support. The installer
must validate the exact SGLang model type and every assumption it consumes,
including model dimensions, dtype, quantization, attention mode, KV layout,
contiguity, and pool identity.

The dependency direction is deliberately one-way:

`platform -> router -> registry/base -> lazy family installer -> providers`

`base.py` owns the lifecycle protocol, stable serving lock, generic fallback,
and rollback-capable binding transaction. `router.py` is the only public model
dispatcher. A family installer must not import `router.py` or `registry.py`, and
the package `__init__.py` must not import a family provider. The router validates
the returned plan's `forward_lock`, `close()`, `invalidate_views()`, and
`get_state()` contract before publishing it to `ModelRunner`.

Every plan must use the shared `MPS_OPERATOR_FORWARD_LOCK`. A private `RLock`
still breaks lock identity across online replacement, while a plain lock can
deadlock during nested lifecycle work. The installer validates its complete
plan before binding any provider; the router repeats that validation at the
family boundary. Unknown models parse only model-neutral operator gates, so a
family-specific environment setting cannot affect an unregistered family.

An installer follows this order:

1. Parse and validate provider priorities once at worker startup.
2. Discover typed model modules; never match class names alone.
3. Resolve optional static providers independently for each semantic op.
4. Validate all borrowed Torch storage before constructing a provider.
5. Compile and warm every selected provider while it is unpublished.
6. Validate the complete lifecycle contract, then atomically bind the plan
   with `MpsBindingPublication`. If any
   contributor-owned setter raises, roll back earlier bindings and close the
   unpublished plan.
7. No environment lookup or provider reselection is allowed on a request path.

The Qwen3 dense plan is the reference implementation. A new model family
should normally add its own installer module and registry entry; it should not
add another branch to `ModelRunner` or create another worker/model runner.

## Three different gates

Keep these decisions separate:

- Platform capability is a startup contract. Unsupported execution modes that
  cannot be made correct on MPS must fail before model loading.
- Model static eligibility is evaluated while building the plan. A missing
  optional provider may advance to the next explicitly listed priority. Once a
  provider is selected, view creation, compilation, warmup, or OOM failure is a
  startup error, not a silent downgrade.
- Request eligibility is a pure, synchronization-free check before device work
  launches. A known request-shape or feature miss may run the complete request
  through Torch. After a provider launches or writes KV, failure is fatal; it
  must never fall back after partial execution.

Whole-model eligibility and output-tail eligibility are independent. For
example, a provider may run the transformer body but return hidden states to
the standard Torch logits processor and sampler when logprobs, grammar,
penalties, or non-greedy sampling are requested.

## Feature ownership matrix

Use these terms consistently. **Inherited** means the ordinary SGLang/Torch
owner remains in charge; it does not by itself prove that the path is valid on
MPS. **Static fallback** selects the Torch model path while building the plan.
**Request fallback** declines before provider work launches. **Startup gate**
rejects a server-wide mode for which no correct MPS path exists. Never recover
through Torch after a provider has launched or partially mutated KV state.

| Feature area | Authoritative owner | Provider policy and required evidence |
| --- | --- | --- |
| Loading, parameters, tokenizer, scheduler, and request lifecycle | Standard loader, `ModelRunner`, and scheduler | Providers only borrow state. Require a Torch-reference HTTP smoke plus exact-type and static-contract tests. |
| Radix cache, request-to-token tables, KV pools, and continuous batching | Standard scheduler and Torch pools | Unsupported shapes/layouts are pre-launch operation or request fallbacks. Test cold and chunked prefill, prefix hits, batch 1 and greater than 1, concurrency, and KV/output parity. |
| Non-exact prefill, including chunked, batched, and prefix-hit cases | Standard Torch forward | A whole-model provider admits only its measured contract and falls back for the complete forward otherwise. E2E must assert route telemetry, not only output. |
| Logits processing, sampling, penalties, logit bias, custom processors, logprobs, and grammar | Standard Torch logits processor and sampler | The transformer may run while an optimized token tail falls back. Every advertised mode needs its own E2E; grammar also needs a device-valid mask implementation. |
| Embeddings, pooling, hidden capture, multimodal inputs, and replacement embeddings | Standard model path | Decline before launch unless the complete input/output contract is implemented. Provider fallback alone does not prove that the inherited path is MPS-safe. |
| Hooks, Dumper/debug capture, memory saver, TorchAO, CPU offload, and weight-cache modes | Standard Torch module and lifecycle path | Use static fallback when a coarse island would bypass hooks or borrow unstable storage. Add lifecycle E2E or a startup gate. |
| LoRA | Standard LoRA manager plus a device-valid backend | Whole-model islands use static fallback. MPS requires the Torch-native backend and pageable host metadata; initial load and adapter-bearing prefill/decode need E2E. CUDA/Triton backends and overlap loading are startup errors. |
| Online weight update and model replacement | `ModelRunner` and `WeightUpdater` | Serialize mutation, invalidate borrowed views, and atomically rebuild or refresh. Test disk/tensor updates followed by inference; reject unsupported transports before collectives. |
| Quantization, speculative decode, DLLM, `torch.compile`, TP/PP/DP, HND KV, unsupported KV dtype or backend | Platform startup contract | These are not provider misses when the ordinary MPS path cannot execute them. Keep an actionable startup gate until a dedicated MPS E2E exists. |
| Torch/MLX versions and required Metal APIs | MPS platform runtime | No fallback. Validate before model loading. |

For the Qwen3 reference plan, prefix-hit or incomplete/batched prefill is a
whole-model request fallback; logprobs, grammar, penalties, and non-greedy
sampling are greedy-tail fallbacks; hook-dependent modes are whole-model static
fallbacks; unsupported global execution modes are startup gates.

## Storage and lifecycle rules

- Borrowed DLPack views never transfer ownership away from Torch.
- Record pool identity, tensor data pointers, shapes, dtypes, layouts, and
  contiguity for storage that must remain stable.
- Keep producer storage alive until the consumer stream is fenced. `copy=True`
  is a correctness decision as well as a performance decision.
- Size weights and KV pools against the smaller of host-available unified
  memory and the remaining Metal recommended working set. Count driver
  residency, not only live Torch tensor allocations.
- Serialize forward and online weight mutation through the plan's stable lock.
- Invalidate borrowed views after in-place weight mutation.
- Close providers idempotently before model or KV teardown, and do not clear a
  newer replacement plan's bindings.
- A 16 GB unified-memory machine must not retain old and replacement compiled
  providers simultaneously.

## New model-family checklist

- [ ] Keep the standard Scheduler as owner of scheduling, request lifecycle,
  and Radix state. Keep Torch `ModelRunner` and standard SRT components as the
  owners of parameters, request tables, KV pools, logits, sampling, and
  updates. Add no loader, worker, runner, or cache implementation.
- [ ] Choose the smallest useful integration: generic semantic op first,
  typed model-family op second, whole-model island only with measured evidence.
- [ ] Add one lazy registry spec and family installer. Treat HF architecture
  names only as import hints and validate the exact SGLang model type.
- [ ] Classify each feature as inherited, static fallback, request/operation
  fallback, or startup gate. Trace inherited paths for CUDA, Triton, pinned
  memory, collective, and graph-runner assumptions.
- [ ] Validate dimensions, dtype, effective quantization, attention mode,
  exact modules, KV layout, contiguity, pool identity, data pointers, and
  storage lifetime before publication.
- [ ] Give each semantic operation an immutable startup priority ending in
  Torch. Read environment configuration once; a forced provider is strict.
- [ ] Keep every request gate pure, synchronization-free, and complete. It
  decides before bridge, kernel, or KV work; post-launch exceptions are fatal.
- [ ] Compile and warm providers while unpublished, then bind the complete
  plan atomically. Expose selections, fallback reasons, and route counters.
- [ ] Serialize forward and mutation, invalidate borrowed views, close
  idempotently, and prove replacement cannot retain two compiled generations.
- [ ] Add CPU-safe registry, duplicate, exact/static/dynamic gate, priority,
  atomic-publication, close, invalidation, and negative startup tests.
- [ ] Add device parity tests for kernels, including unsupported shape, dtype,
  layout, and AOT/JIT equivalence where both exist.
- [ ] Run HTTP E2E against Torch for cold/chunked prefill, Radix hits, decode
  batching, concurrency, sampling/logprobs/grammar, newly allowed features,
  and post-update inference. Assert telemetry so an unused provider cannot pass.
- [ ] Benchmark both run orders after warmup, report peak RSS/available memory
  and swap, reap every process, and do not present eager diagnostics as final
  compiled-provider performance.

## Required evidence

Every model-family contribution needs:

- CPU-safe registry, duplicate-registration, exact-type, static-gate,
  priority-fallback, atomic-publication, close, and invalidation tests;
- device tests for every new AOT/JIT/MLX semantic kernel, including unsupported
  shapes and dtypes;
- output and KV parity against the ordinary Torch path for prefill and decode;
- an HTTP E2E that exercises cold prefill, Radix prefix hits, decode batching,
  logprobs, request-level fallback, and disk weight refresh;
- an explicit feature matrix separating “standard SGLang path works while the
  custom provider falls back” from “MPS startup rejects this feature”;
- forward-order and reverse-order performance trials, process cleanup, peak
  RSS/available-memory/swap reporting, and no eager-mode result presented as
  final graph performance.

Only advertise a feature after its ordinary Torch fallback path has an MPS E2E
test. Reusing `ModelRunner` is necessary, but it does not prove that a CUDA-only
sampler, grammar backend, LoRA backend, or speculative verification kernel is
valid on MPS.
