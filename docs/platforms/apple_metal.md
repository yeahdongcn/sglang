# Apple Silicon with Metal

This document describes how run SGLang on Apple Silicon using [Metal](https://developer.apple.com/metal/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Prerequisites

Building the native Metal kernels in `sgl-kernel` requires the Apple
toolchain (`clang++`, the Metal framework headers, and `xcrun`). These ship
with the **Xcode Command Line Tools**, which cannot be installed via `pip`:

```bash
xcode-select --install
```

If you have the full Xcode app installed, the Command Line Tools are already
available. You can verify with `xcode-select -p && xcrun --find metal`.

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create and activate a virtual environment
uv venv -p 3.12 sglang-metal
source sglang-metal/bin/activate

# (Optional) Compile sgl-kernel
uv pip install --upgrade pip
uv run sgl-kernel/setup_metal.py install

# Install sglang python package along with diffusion support
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```
