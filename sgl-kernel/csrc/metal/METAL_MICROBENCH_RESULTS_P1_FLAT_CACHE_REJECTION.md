# P1 Flat Cache Rejection

Generated: `2026-05-10 01:00 Asia/Shanghai`

This experiment tested making `MlxPagedKVCache(block_size=1)` authoritative
storage flat, with shape `(num_blocks, n_kv_heads, head_dim)`, and teaching the
Metal wrappers to accept that p1 shape as equivalent to
`(num_blocks, 1, n_kv_heads, head_dim)`.

The native 3-D p1 wrapper support is correct, but using flat p1 as the runtime
cache storage is rejected. Serving regressions outweigh the microbench wins.

## Microbench

Shape: 28 layers, 8 KV heads, head dim 128, float16. Median over 12 measured
iterations after 3 warmups.

Before the runtime flat-storage patch, current 4-D p1 cache storage measured:

| Tokens | 4-D p1 Metal sync ms | 4-D p1 MLX list sync ms | Flat MLX list sync ms | Dual 4-D+flat sync ms | 4-D p1 materialize ms | Flat materialize ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8765 | 2.3214 | 0.9073 | 1.7895 | 2.7103 | 0.5315 |
| 2 | 1.2594 | 2.0281 | 1.0669 | 2.2857 | 21.8338 | 0.7217 |
| 16 | 1.3969 | 2.4596 | 1.0831 | 2.5938 | 20.1824 | 0.6088 |
| 432 | 7.7918 | 17.4030 | 11.7032 | 20.6901 | 27.0939 | 8.4455 |
| 1729 | 34.4034 | 63.6872 | 36.9765 | 72.7109 | 59.3335 | 20.3386 |

After making p1 storage flat in `MlxPagedKVCache`, the runtime cache measured:

| Tokens | Flat p1 Metal sync ms | Flat p1 MLX sync ms | Flat p1 materialize ms |
|---:|---:|---:|---:|
| 1 | 0.8060 | 0.9966 | 2.8756 |
| 2 | 1.0510 | 1.3836 | 18.0673 |
| 16 | 1.1550 | 2.3310 | 18.1996 |
| 432 | 8.3142 | 10.4787 | 20.0536 |
| 1729 | 23.3632 | 37.3269 | 42.8201 |

Interpretation: flat p1 improves selected synthetic write/read rows, especially
large all-layer sync, but not enough to justify the serving-path regressions.
The dual side-store direction is also unattractive because it adds write cost on
the hot prompt-sync path.

## Serving Probe

Server command shape:

```bash
PYTHONPATH="$PWD:$PWD/python:$PWD/sgl-kernel/python" \
SGLANG_USE_MLX=1 \
VIRTUAL_ENV="$PWD/sglang-mlx" \
PATH="$PWD/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.launch_server \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.6 \
  --port 43491
```

Probe prompt:

```text
Explain why paged attention caches need slot mappings and block tables in a language model server.
```

The short probe repeats that text 24 times; the long probe repeats it 120
times. Sampling used `temperature=0` and `max_new_tokens=20`.

First flat-p1 run:

| Probe | Prompt tokens | Cached tokens | E2E latency s | Notes |
|---|---:|---:|---:|---|
| short first | 434 | 0 | 1.4848 | coherent |
| short hit 1 | 434 | 433 | 1.2029 | regressed vs accepted p1 0.93 |
| short hit 2 | 434 | 433 | 1.0217 | regressed vs accepted p1 0.68 |
| long partial | 2162 | 433 | 4.1002 | regressed vs accepted p1 2.77 |
| long full | 2162 | 2161 | 1.1293 | regressed vs accepted p1 0.89 |
| delayed short hit | 434 | 433 | 1.1203 | improved vs accepted p1 1.31 |

After `/flush_cache`, the same server measured:

| Probe | Prompt tokens | Cached tokens | E2E latency s | Notes |
|---|---:|---:|---:|---|
| short first | 434 | 0 | 3.5315 | regressed |
| short hit 1 | 434 | 433 | 0.7559 | improved |
| short hit 2 | 434 | 433 | 0.6825 | tied with accepted p1 |
| long partial | 2162 | 433 | 3.2995 | still regressed vs accepted p1 2.77 |
| long full | 2162 | 2161 | 0.8395 | improved |
| delayed short hit | 434 | 433 | 3.4238 | regressed badly |

Decision: revert `MlxPagedKVCache` runtime p1 storage to 4-D. Keep pursuing a
different radix direction; do not retry flat-authoritative p1 storage without a
way to avoid the long-partial and delayed-short regressions.
