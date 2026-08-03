# sglang-kernel Metal kernels

Custom Apple Metal kernels for the Torch MPS backend on Apple Silicon. The
Metal wheel is a Python package plus `sgl_metal_kernels.metallib`; Torch 2.13
loads the archive directly with `torch.mps.load_metallib`. There is no MLX
primitive, tensor bridge, nanobind module, or C++ host extension on this path.

[`setup_metal.py`](setup_metal.py) compiles the canonical shader source
shared with the JIT path, and
[`python/sgl_kernel/metal.py`](python/sgl_kernel/metal.py) owns lazy
loading and process-local library caching.

## Kernels

| Kernel | Description | Tested on |
| --- | --- | --- |
| `qwen3_qknorm_rope_store_bf16` | Qwen3-0.6B Q/K RMSNorm, NeoX RoPE, and Torch-owned NHD KV-pool store. | Apple Silicon / Torch MPS bf16 |
| `qwen3_radix_decode_bf16` | Qwen3-0.6B GQA decode over SGLang's Radix request/slot tables. | Apple Silicon / Torch MPS bf16 |

## Adding a new Metal kernel

1. Define the vendor-neutral callable in `sglang.kernels.ops.<group>`.
2. Put the Metal implementation beside that semantic operator. If JIT and AOT
   have the same fixed contract, keep one canonical MSL source for both paths.
3. Add the source to [`setup_metal.py`](setup_metal.py), which compiles it
   into the packaged metallib.
4. Add a lazy `forward_metal_aot` adapter; model/runtime code must call the
   semantic operator rather than importing a private Metal module.
5. Add parity and packaging tests under `test/registered/kernels/ops/<group>`
   and update the table above.
