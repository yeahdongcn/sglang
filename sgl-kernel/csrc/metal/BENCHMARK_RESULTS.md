# Apple Silicon Metal Attention Benchmark Results

Date: 2026-05-07
Branch: `xd/sgl_kernel_metal`
Model: `/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`

## Summary

Correctness tests pass after the paged-cache dtype normalization fix, and local Apple Silicon MLX/Metal server smoke generation now works for no-prefix, partial-prefix, and full-prefix cache-hit prompts. The latest repeat `bench_one_batch --batch-size 1 4 8 --input-len 256 --output-len 32` comparison shows Metal ahead on total throughput for batch sizes 1, 4, and 8, with decode throughput ahead for batch sizes 4 and 8 and slightly behind for batch size 1.

Latest repeat post-dtype-fix verification results:

- MLX/Metal server smoke: no-prefix, partial-prefix, and full-prefix prompts completed successfully on port `43436`.
- `bench_one_batch --batch-size 1 --input-len 256 --output-len 32`: Metal improved prefill throughput by 8.79% and total throughput by 2.74%; median decode throughput was 2.07% lower.
- `bench_one_batch --batch-size 4 --input-len 256 --output-len 32`: Metal improved prefill throughput by 14.82%, median decode throughput by 7.49%, and total throughput by 7.57%.
- `bench_one_batch --batch-size 8 --input-len 256 --output-len 32`: Metal improved prefill throughput by 8.87%, median decode throughput by 4.18%, and total throughput by 6.77%.

Previous post-routing verification results:

- `bench_offline_throughput --num-prompts 1`: Metal improved total token throughput by 4.23% and last-generation throughput by 5.46%.
- `bench_one_batch --batch-size 1`: Metal changed median decode throughput by -0.64% and total throughput by -0.87%; this is effectively tied and still much better than the earlier offline regression.

Previous post-revert verification results:

- `bench_offline_throughput --num-prompts 1`: Metal regressed total token throughput by 4.42% and last-generation throughput by 5.63%.
- `bench_one_batch --batch-size 1`: Metal improved median decode throughput by 9.14% and total throughput by 27.08%.
- `bench_one_batch --batch-size 4`: Metal improved median decode throughput by 1.40% and total throughput by 1.49%.

Previous accepted implementation results after dense routing:

- `bench_offline_throughput --num-prompts 1`: Metal improved total token throughput by 3.11%, but last-generation throughput regressed by 9.28%.
- `bench_one_batch --batch-size 1`: Metal improved median decode throughput by 3.47% and total throughput by 5.19%.
- `bench_one_batch --batch-size 4`: Metal improved total throughput by 1.34%, but median decode throughput was effectively tied/slightly lower by 0.04%.

Discarded query-cache experiment:

- Caching query vectors in threadgroup memory regressed `bench_offline_throughput --num-prompts 1` total token throughput by 16.79% and last-generation throughput by 30.81%.
- The same experiment was effectively tied for `bench_one_batch --batch-size 1` median decode throughput (+0.44%) but regressed `bench_one_batch --batch-size 4` median decode throughput by 2.26% and total throughput by 1.89%.
- The query-cache shader change was reverted; the current kernel is back to direct QK reads with score/weight caching.

Current conclusion: functional E2E smoke passes after normalizing unsupported model/cache dtypes to Metal-supported dtypes. Repeat performance verification now shows a local Metal win on total throughput for batch sizes 1, 4, and 8, with the clearest gains at batch sizes 4 and 8. Batch-size-1 decode remains slightly behind, but the combined smoke and repeat benchmark results are sufficient to close the local E2E/performance verification gate and proceed to removing the old guarded implementation.

## Environment

Common environment prefix:

```bash
PYTHONPATH="python:sgl-kernel/python"
```

Fallback MLX path:

```bash
SGLANG_USE_MLX=1
```

Metal-enabled MLX path:

```bash
SGLANG_USE_MLX=1
SGLANG_MLX_USE_METAL_ATTENTION=1
```

Observed runtime notes in benchmark output:

- Platform detection fell back to base SRT platform.
- CUDA graph was disabled because the MLX path uses torch-native attention as the server-side placeholder backend.
- MLX model loading was used for inference.
- `RLIMIT_STACK` warning was printed during `bench_one_batch` but did not stop the benchmark.

## Correctness verification

`sgl-kernel` Metal wrapper/kernel tests:

```bash
PYTHONPATH="sgl-kernel/python" uv run --active --with pytest pytest "sgl-kernel/tests/test_metal.py" -q
```

Result:

```text
20 passed in 0.11s
```

SRT/MLX Metal integration unit tests:

```bash
PYTHONPATH="python:sgl-kernel/python" \
"/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin/python" \
  -m unittest \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_attention_wrapper \
  test.registered.unit.layers.attention.test_metal_backend \
  test.registered.unit.server_args.test_server_args.TestAttentionBackendArgs \
  -v
```

Result:

```text
Ran 19 tests in 0.077s
OK
```

## Latest run: repeat post-dtype-fix paged backend performance

This run repeats the fallback-vs-Metal comparison after the successful server smoke, extending the focused one-batch benchmark to batch sizes 1, 4, and 8.

Executed fallback and Metal commands sequentially to avoid resource contention:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:$PYTHONPATH" \
SGLANG_USE_MLX=1 \
"./sglang-mlx/bin/python" -m sglang.bench_one_batch \
  --model-path "/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --batch-size 1 4 8 \
  --input-len 256 \
  --output-len 32 \
  --warmups 1 \
  --run-name mlx_fallback_repeat_20260507 \
  --result-filename /tmp/sglang_mlx_fallback_repeat_20260507.jsonl

PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:$PYTHONPATH" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
"./sglang-mlx/bin/python" -m sglang.bench_one_batch \
  --model-path "/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --batch-size 1 4 8 \
  --input-len 256 \
  --output-len 32 \
  --warmups 1 \
  --run-name mlx_metal_repeat_20260507 \
  --result-filename /tmp/sglang_mlx_metal_repeat_20260507.jsonl
```

### Repeat batch size 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.37697 s | 0.34649 s | -8.08% faster |
| Prefill throughput | 679.11 tok/s | 738.83 tok/s | +8.79% |
| Median decode latency | 0.04641 s | 0.04739 s | +2.11% slower |
| Median decode throughput | 21.55 tok/s | 21.10 tok/s | -2.07% |
| Total latency | 1.809 s | 1.761 s | -2.67% faster |
| Total throughput | 159.17 tok/s | 163.54 tok/s | +2.74% |

Result: Metal wins total throughput for batch size 1 in the repeat run, though median decode throughput remains slightly behind fallback.

### Repeat batch size 4

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 1.58660 s | 1.38180 s | -12.91% faster |
| Prefill throughput | 645.40 tok/s | 741.06 tok/s | +14.82% |
| Median decode latency | 0.10054 s | 0.09353 s | -6.97% faster |
| Median decode throughput | 39.79 tok/s | 42.77 tok/s | +7.49% |
| Total latency | 4.610 s | 4.285 s | -7.04% faster |
| Total throughput | 249.91 tok/s | 268.84 tok/s | +7.57% |

Result: Metal is ahead for batch size 4 on prefill, median decode, and total throughput.

### Repeat batch size 8

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 2.96128 s | 2.71996 s | -8.15% faster |
| Prefill throughput | 691.59 tok/s | 752.95 tok/s | +8.87% |
| Median decode latency | 0.11407 s | 0.10949 s | -4.01% faster |
| Median decode throughput | 70.13 tok/s | 73.06 tok/s | +4.18% |
| Total latency | 6.437 s | 6.029 s | -6.34% faster |
| Total throughput | 357.90 tok/s | 382.15 tok/s | +6.77% |

Result: Metal is ahead for batch size 8 on prefill, median decode, and total throughput.



### Server smoke: no-prefix, partial-prefix, and full-prefix prompts

Executed with local checkout packages and the pre-created `sglang-mlx` environment on port `43436`:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:$PYTHONPATH" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
"./sglang-mlx/bin/python" -m sglang.launch_server \
  --model-path "/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --tp-size 1 \
  --port 43436
```

| Prompt case | Elapsed | Result |
|---|---:|---|
| No prefix | 4.926 s | Completed |
| Partial prefix A | 8.057 s | Completed |
| Partial prefix B | 11.617 s | Completed |
| Full prefix first request | 5.165 s | Completed |
| Full prefix second request | 2.735 s | Completed |

Result: local MLX/Metal server generation completed for no-prefix, partial-prefix, and full-prefix workloads. The repeated full-prefix request was faster than the first request in this smoke run.

### Benchmark: one-batch, batch sizes 1 and 4

Executed fallback and Metal commands sequentially to avoid resource contention:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:$PYTHONPATH" \
SGLANG_USE_MLX=1 \
"./sglang-mlx/bin/python" -m sglang.bench_one_batch \
  --model-path "/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --batch-size 1 4 \
  --input-len 256 \
  --output-len 32 \
  --warmups 1 \
  --run-name mlx_fallback_20260507 \
  --result-filename /tmp/sglang_mlx_fallback_20260507.jsonl

PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:$PYTHONPATH" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
"./sglang-mlx/bin/python" -m sglang.bench_one_batch \
  --model-path "/Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --batch-size 1 4 \
  --input-len 256 \
  --output-len 32 \
  --warmups 1 \
  --run-name mlx_metal_20260507 \
  --result-filename /tmp/sglang_mlx_metal_20260507.jsonl
```

#### Batch size 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.20704 s | 0.20100 s | -2.92% faster |
| Prefill throughput | 1236.47 tok/s | 1273.63 tok/s | +3.01% |
| Median decode latency | 0.02855 s | 0.02828 s | -0.94% faster |
| Median decode throughput | 35.03 tok/s | 35.36 tok/s | +0.95% |
| Total latency | 1.092 s | 1.112 s | +1.81% slower |
| Total throughput | 263.72 tok/s | 259.03 tok/s | -1.78% |

Result: Metal is effectively tied with fallback for batch size 1 on this longer-input benchmark.

#### Batch size 4

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.86530 s | 0.86513 s | -0.02% faster |
| Prefill throughput | 1183.41 tok/s | 1183.63 tok/s | +0.02% |
| Median decode latency | 0.05740 s | 0.05697 s | -0.74% faster |
| Median decode throughput | 69.69 tok/s | 70.21 tok/s | +0.75% |
| Total latency | 2.673 s | 2.628 s | -1.71% faster |
| Total throughput | 430.94 tok/s | 438.42 tok/s | +1.74% |

Result: Metal is ahead for batch size 4 on this focused run, but the gain is still small enough that repeat measurements are needed before claiming a robust performance win.

## Latest run: after single-request Metal routing

This run includes the `MlxModelRunner.decode_batch` routing change that keeps the direct single-request cache path when `SGLANG_MLX_USE_METAL_ATTENTION` is disabled, but routes `batch_size == 1` through `BatchedDecodeContext` when Metal attention is enabled.

Focused routing tests were also added for both paths:

```text
7 passed, 6 warnings in 8.29s
```

### Benchmark 1: Offline throughput, batch-size equivalent 1

Executed fallback command:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python" \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
SGLANG_USE_MLX=1 \
uv run --active --no-project python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

Executed Metal command:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python" \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
uv run --active --no-project python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 7.18 s | 6.88 s | -4.18% faster |
| Last generation throughput | 34.63 tok/s | 36.52 tok/s | +5.46% |
| Output token throughput | 34.00 tok/s | 35.44 tok/s | +4.24% |
| Total token throughput | 36.37 tok/s | 37.91 tok/s | +4.23% |

Result: Metal is now faster on the requested single-prompt offline benchmark after routing Metal-enabled `batch_size == 1` decode through the context path.

### Benchmark 2: One-batch, batch size 1

Executed fallback command:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python" \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
SGLANG_USE_MLX=1 \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

Executed Metal command:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python" \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

#### Benchmark-phase metrics

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.06120 s | 0.05785 s | -5.47% faster |
| Prefill throughput | 980.33 tok/s | 1037.24 tok/s | +5.81% |
| Median decode latency | 0.02660 s | 0.02677 s | +0.64% slower |
| Median decode throughput | 37.59 tok/s | 37.35 tok/s | -0.64% |
| Total latency | 2.742 s | 2.766 s | +0.88% slower |
| Total throughput | 58.35 tok/s | 57.84 tok/s | -0.87% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.02448 s | 40.86 tok/s | 0.02899 s | 34.49 tok/s |
| 1 | 0.02930 s | 34.13 tok/s | 0.02581 s | 38.75 tok/s |
| 2 | 0.02399 s | 41.68 tok/s | 0.02990 s | 33.45 tok/s |
| 3 | 0.02479 s | 40.34 tok/s | 0.02460 s | 40.64 tok/s |
| 4 | 0.02681 s | 37.29 tok/s | 0.02791 s | 35.83 tok/s |

Result: Metal is effectively tied with fallback on this requested one-batch batch-size-1 run. The offline regression is fixed, but this microbenchmark should not be treated as a robust Metal win.

## Latest run: post-revert refresh

This run uses the current direct-QK kernel with score/weight caching after reverting the threadgroup query-cache experiment. Commands were executed with `uv run --active --no-project` and the pre-created `sglang-mlx` virtual environment.

### Benchmark 1: Offline throughput, batch-size equivalent 1

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 6.91 s | 7.23 s | +4.63% slower |
| Total input tokens | 17 | 17 | 0.00% |
| Total generated tokens | 244 | 244 | 0.00% |
| Last generation throughput | 36.58 tok/s | 34.52 tok/s | -5.63% |
| Request throughput | 0.14 req/s | 0.14 req/s | 0.00% |
| Input token throughput | 2.46 tok/s | 2.35 tok/s | -4.47% |
| Output token throughput | 35.30 tok/s | 33.74 tok/s | -4.42% |
| Total token throughput | 37.76 tok/s | 36.09 tok/s | -4.42% |

Result: Metal regressed on the single-prompt offline benchmark in this refresh.

### Benchmark 2: One-batch, batch size 1

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

#### Benchmark-phase metrics

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.05930 s | 0.05904 s | -0.44% faster |
| Prefill throughput | 1011.77 tok/s | 1016.18 tok/s | +0.44% |
| Median decode latency | 0.03347 s | 0.03067 s | -8.37% faster |
| Median decode throughput | 29.88 tok/s | 32.61 tok/s | +9.14% |
| Total latency | 3.994 s | 3.143 s | -21.31% faster |
| Total throughput | 40.06 tok/s | 50.91 tok/s | +27.08% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.03514 s | 28.46 tok/s | 0.02529 s | 39.54 tok/s |
| 1 | 0.02987 s | 33.48 tok/s | 0.02581 s | 38.74 tok/s |
| 2 | 0.03347 s | 29.88 tok/s | 0.02843 s | 35.17 tok/s |
| 3 | 0.03143 s | 31.82 tok/s | 0.02529 s | 39.54 tok/s |
| 4 | 0.03533 s | 28.30 tok/s | 0.03031 s | 32.99 tok/s |

Result: Metal improved median decode throughput, total latency, and total throughput for this requested batch-size-1 run.

### Benchmark 3: One-batch, batch size 4

This benchmark targets the plan's batched decode performance gate.

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 4 \
  --input-len 60 \
  --output-len 100 \
  --port 43440 \
  --mem-fraction-static 0.5 \
  --disable-radix-cache
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 4 \
  --input-len 60 \
  --output-len 100 \
  --port 43440 \
  --mem-fraction-static 0.5 \
  --disable-radix-cache
```

#### Benchmark-phase metrics

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.24513 s | 0.24538 s | +0.10% slower |
| Prefill throughput | 979.08 tok/s | 978.09 tok/s | -0.10% |
| Median decode latency | 0.05447 s | 0.05371 s | -1.40% faster |
| Median decode throughput | 73.44 tok/s | 74.47 tok/s | +1.40% |
| Total latency | 5.660 s | 5.577 s | -1.47% faster |
| Total throughput | 113.07 tok/s | 114.76 tok/s | +1.49% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.05211 s | 76.76 tok/s | 0.04817 s | 83.04 tok/s |
| 1 | 0.05366 s | 74.54 tok/s | 0.05024 s | 79.62 tok/s |
| 2 | 0.05115 s | 78.20 tok/s | 0.05066 s | 78.96 tok/s |
| 3 | 0.05712 s | 70.02 tok/s | 0.05153 s | 77.62 tok/s |
| 4 | 0.05187 s | 77.11 tok/s | 0.05045 s | 79.29 tok/s |

Result: Metal improved the batch-size-4 median decode throughput and total throughput in this refresh, satisfying the current batched decode performance gate locally. The margin is still small, so repeat runs are useful before making a broad performance claim.

## Latest run: after dense Metal routing for uniform MLX decode batches

This run includes routing uniform-length MLX batched decode requests to dense `decode_attention` instead of ragged `decode_attention_ragged`, reducing per-request Metal dispatch overhead for uniform batches.

### Benchmark 1: Offline throughput, batch-size equivalent 1

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
uv run python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
uv run python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 7.59 s | 7.36 s | -3.03% faster |
| Total input tokens | 17 | 17 | 0.00% |
| Total generated tokens | 244 | 244 | 0.00% |
| Last generation throughput | 36.85 tok/s | 33.43 tok/s | -9.28% |
| Request throughput | 0.13 req/s | 0.14 req/s | +7.69% |
| Input token throughput | 2.24 tok/s | 2.31 tok/s | +3.13% |
| Output token throughput | 32.16 tok/s | 33.16 tok/s | +3.11% |
| Total token throughput | 34.40 tok/s | 35.47 tok/s | +3.11% |

Result: Metal improved aggregate throughput, but the last-generation metric regressed.

### Benchmark 2: One-batch, batch size 1

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
uv run python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
uv run python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

#### Benchmark-phase metrics

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.05629 s | 0.05882 s | +4.49% slower |
| Prefill throughput | 1065.87 tok/s | 1020.00 tok/s | -4.30% |
| Median decode latency | 0.02800 s | 0.02706 s | -3.36% faster |
| Median decode throughput | 35.71 tok/s | 36.95 tok/s | +3.47% |
| Total latency | 2.906 s | 2.762 s | -4.96% faster |
| Total throughput | 55.06 tok/s | 57.92 tok/s | +5.19% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.02661 s | 37.57 tok/s | 0.02820 s | 35.47 tok/s |
| 1 | 0.02474 s | 40.42 tok/s | 0.02718 s | 36.80 tok/s |
| 2 | 0.02505 s | 39.92 tok/s | 0.03097 s | 32.29 tok/s |
| 3 | 0.02617 s | 38.21 tok/s | 0.02812 s | 35.56 tok/s |
| 4 | 0.02566 s | 38.97 tok/s | 0.02988 s | 33.47 tok/s |

Result: Metal improved the benchmark-phase median decode and total throughput for this run, although the first five printed decode steps were slower.

### Benchmark 3: One-batch, batch size 4

This benchmark targets the plan's batched decode performance gate.

Executed fallback command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
uv run python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 4 \
  --input-len 60 \
  --output-len 100 \
  --port 43440 \
  --mem-fraction-static 0.5 \
  --disable-radix-cache
```

Executed Metal command:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
SGLANG_MLX_USE_METAL_ATTENTION=1 \
uv run python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 4 \
  --input-len 60 \
  --output-len 100 \
  --port 43440 \
  --mem-fraction-static 0.5 \
  --disable-radix-cache
```

#### Benchmark-phase metrics

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.24579 s | 0.25148 s | +2.32% slower |
| Prefill throughput | 976.43 tok/s | 954.35 tok/s | -2.26% |
| Median decode latency | 0.05397 s | 0.05399 s | +0.04% slower |
| Median decode throughput | 74.12 tok/s | 74.09 tok/s | -0.04% |
| Total latency | 5.736 s | 5.660 s | -1.33% faster |
| Total throughput | 111.58 tok/s | 113.07 tok/s | +1.34% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.05721 s | 69.92 tok/s | 0.05577 s | 71.72 tok/s |
| 1 | 0.05376 s | 74.41 tok/s | 0.05400 s | 74.08 tok/s |
| 2 | 0.05149 s | 77.69 tok/s | 0.05698 s | 70.20 tok/s |
| 3 | 0.05330 s | 75.04 tok/s | 0.05139 s | 77.83 tok/s |
| 4 | 0.05002 s | 79.96 tok/s | 0.05165 s | 77.44 tok/s |

Result: Metal was slightly faster in total throughput, but decode median was effectively tied and not a robust win.

## Reverted experiment: query caching in threadgroup memory

This experiment loaded each query vector into threadgroup memory before the QK dot products, using `vllm-metal/vllm_metal/metal/kernels_v2/` as inspiration for reducing repeated reads. It passed correctness before benchmarking, but it was slower on the target workloads and was reverted.

### Offline throughput, batch-size equivalent 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 6.92 s | 8.32 s | +20.23% slower |
| Total input tokens | 17 | 17 | 0.00% |
| Total generated tokens | 244 | 244 | 0.00% |
| Last generation throughput | 37.10 tok/s | 25.67 tok/s | -30.81% |
| Request throughput | 0.14 req/s | 0.12 req/s | -14.29% |
| Input token throughput | 2.46 tok/s | 2.04 tok/s | -17.07% |
| Output token throughput | 35.24 tok/s | 29.32 tok/s | -16.80% |
| Total token throughput | 37.70 tok/s | 31.37 tok/s | -16.79% |

Result: Metal regressed substantially; the query-cache optimization was not viable for this benchmark.

### One-batch, batch size 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.06306 s | 0.05963 s | -5.44% faster |
| Prefill throughput | 951.53 tok/s | 1006.25 tok/s | +5.75% |
| Median decode latency | 0.02934 s | 0.02921 s | -0.44% faster |
| Median decode throughput | 34.08 tok/s | 34.23 tok/s | +0.44% |
| Total latency | 3.156 s | 3.048 s | -3.42% faster |
| Total throughput | 50.70 tok/s | 52.50 tok/s | +3.55% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.03162 s | 31.62 tok/s | 0.03197 s | 31.28 tok/s |
| 1 | 0.02608 s | 38.35 tok/s | 0.03445 s | 29.03 tok/s |
| 2 | 0.02704 s | 36.98 tok/s | 0.03612 s | 27.69 tok/s |
| 3 | 0.02746 s | 36.42 tok/s | 0.03723 s | 26.86 tok/s |
| 4 | 0.02870 s | 34.84 tok/s | 0.04043 s | 24.74 tok/s |

Result: Summary metrics were slightly faster, but the printed early decode steps were slower and the gain was too small to offset regressions elsewhere.

### One-batch, batch size 4

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.23487 s | 0.23117 s | -1.58% faster |
| Prefill throughput | 1021.85 tok/s | 1038.18 tok/s | +1.60% |
| Median decode latency | 0.05115 s | 0.05233 s | +2.31% slower |
| Median decode throughput | 78.20 tok/s | 76.43 tok/s | -2.26% |
| Total latency | 5.347 s | 5.450 s | +1.93% slower |
| Total throughput | 119.70 tok/s | 117.44 tok/s | -1.89% |

#### Per-step decode metrics

| Decode step | Fallback latency | Fallback throughput | Metal latency | Metal throughput |
|---:|---:|---:|---:|---:|
| 0 | 0.04880 s | 81.96 tok/s | 0.05168 s | 77.40 tok/s |
| 1 | 0.05087 s | 78.63 tok/s | 0.05664 s | 70.62 tok/s |
| 2 | 0.05114 s | 78.21 tok/s | 0.05537 s | 72.24 tok/s |
| 3 | 0.05011 s | 79.83 tok/s | 0.05024 s | 79.61 tok/s |
| 4 | 0.05059 s | 79.07 tok/s | 0.05042 s | 79.33 tok/s |

Result: The batched decode target regressed, so the query-cache optimization was reverted.



### Offline throughput

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 9.01 s | 8.98 s | -0.33% faster |
| Last generation throughput | 26.61 tok/s | 28.65 tok/s | +7.67% |
| Output token throughput | 27.10 tok/s | 27.18 tok/s | +0.30% |
| Total token throughput | 28.98 tok/s | 29.07 tok/s | +0.31% |

Result: essentially tied; total throughput improved only 0.31%.

### One-batch, batch size 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.05901 s | 0.06032 s | +2.22% slower |
| Prefill throughput | 1016.79 tok/s | 994.71 tok/s | -2.17% |
| Median decode latency | 0.02580 s | 0.02674 s | +3.64% slower |
| Median decode throughput | 38.76 tok/s | 37.39 tok/s | -3.53% |
| Total latency | 2.703 s | 2.746 s | +1.59% slower |
| Total throughput | 59.19 tok/s | 58.27 tok/s | -1.55% |

Result: Metal was slightly slower.

### One-batch, batch size 4

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.23530 s | 0.24447 s | +3.90% slower |
| Prefill throughput | 1019.99 tok/s | 981.70 tok/s | -3.75% |
| Median decode latency | 0.05292 s | 0.05335 s | +0.81% slower |
| Median decode throughput | 75.58 tok/s | 74.97 tok/s | -0.81% |
| Total latency | 5.507 s | 5.554 s | +0.85% slower |
| Total throughput | 116.21 tok/s | 115.22 tok/s | -0.85% |

Result: Metal was slightly slower, which motivated the dense routing optimization for uniform batches.

## Initial run: before the latest decode performance work

These are the earlier baseline numbers from the first benchmark report.

### Offline throughput, batch-size equivalent 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Benchmark duration | 7.01 s | 7.42 s | +5.85% slower |
| Total input tokens | 17 | 17 | 0.00% |
| Total generated tokens | 244 | 244 | 0.00% |
| Last generation throughput | 35.54 tok/s | 34.77 tok/s | -2.17% |
| Request throughput | 0.14 req/s | 0.13 req/s | -7.14% |
| Input token throughput | 2.42 tok/s | 2.29 tok/s | -5.37% |
| Output token throughput | 34.79 tok/s | 32.90 tok/s | -5.43% |
| Total token throughput | 37.21 tok/s | 35.19 tok/s | -5.43% |

Result: Metal regressed on this benchmark.

### One-batch, batch size 1

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.05633 s | 0.05979 s | +6.14% slower |
| Prefill throughput | 1065.17 tok/s | 1003.59 tok/s | -5.78% |
| Median decode latency | 0.02698 s | 0.02677 s | -0.78% faster |
| Median decode throughput | 37.07 tok/s | 37.36 tok/s | +0.78% |
| Total latency | 2.805 s | 2.706 s | -3.53% faster |
| Total throughput | 57.05 tok/s | 59.14 tok/s | +3.66% |

Result: Metal showed a very small decode improvement, but the difference was too small and noisy to claim a meaningful win.

### One-batch, batch size 4

| Metric | Fallback MLX | Metal enabled | Metal delta |
|---|---:|---:|---:|
| Prefill latency | 0.26190 s | 0.25341 s | -3.24% faster |
| Prefill throughput | 916.39 tok/s | 947.08 tok/s | +3.35% |
| Median decode latency | 0.05645 s | 0.05559 s | -1.52% faster |
| Median decode throughput | 70.86 tok/s | 71.96 tok/s | +1.55% |
| Total latency | 5.881 s | 5.766 s | -1.96% faster |
| Total throughput | 108.82 tok/s | 110.99 tok/s | +1.99% |

Result: Metal was slightly faster in aggregate, but the gain was small.

## Current conclusion

Correctness tests pass and the functional plan tracker is complete. The dense MLX routing change improved the latest `bench_one_batch --batch-size 1` run and kept batch-size-4 total throughput slightly ahead, but batched decode median throughput is still effectively tied with the MLX fallback.

Next target: continue improving Metal decode performance while preserving correctness. The reference implementation to inspect is `vllm-metal/vllm_metal/metal/kernels_v2/`.
