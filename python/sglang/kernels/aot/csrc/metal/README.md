# sgl-kernel Metal kernels

Custom Apple Metal kernels for the MLX backend on Apple Silicon. Shader sources
(`*.metal`) and C++ host / nanobind sources (`*.cpp`) in this directory are
compiled by [`sgl-kernel/setup_metal.py`](../../setup_metal.py) through MLX's
`mlx.extension.MetalExtension` API. The build produces the
`sgl_metal_kernels` native module, its `sgl_metal_kernels.metallib` sidecar,
and a generated stub file, which are exposed through the public Python
wrappers in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py).

The API landed in MLX PR #4004. Until the next MLX release, install an MLX
build from the MLX `main` branch before compiling these optional kernels.

## Kernels

| Kernel | Description | Tested on |
| --- | --- | --- |
| `rope_pool_fused` | Fused NeoX RoPE for Q/K plus K/V scatter into the MLX KV pool. | Apple Silicon / MLX |

## Adding a new Metal kernel

1. Add the shader under `csrc/metal/<kernel>.metal`.
2. Add the C++ host / nanobind binding under `csrc/metal/<kernel>.cpp`, exporting the native entry point for the wrapper in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py).
3. Add both files to the `sources` list in [`sgl-kernel/setup_metal.py`](../../setup_metal.py).
4. Add a Python wrapper in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py) that validates input shapes/dtypes and invokes the native AOT entry point without forcing MLX evaluation.
5. Add a test under [`sgl-kernel/tests/`](../../tests) and update the **Kernels** table above with a short description and the hardware / OS / MLX version the kernel was validated on.
