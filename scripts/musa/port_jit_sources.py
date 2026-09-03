#!/usr/bin/env python3
"""Apply torchada's CUDA-to-MUSA source mapping to the in-image JIT tree.

The JIT builder bypasses torch.utils.cpp_extension's normal MusaBuildExtension,
so it otherwise feeds CUDA-spelled headers directly to mcc. This helper is
intended for a copied checkout (for example, a Docker image), never a shared
source checkout. A sentinel makes the operation idempotent across worker
processes and records the torchada version used for the transformation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

_SKIP_NAMES = {"cxx17_compat.h", "utils.h", "utils.cuh", "tensor.h", "type.cuh"}


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".h", ".cuh", ".cu", ".cpp", ".cc"}:
            continue
        if path.name in _SKIP_NAMES:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    sentinel = args.sentinel.resolve()
    if not root.is_dir():
        raise SystemExit(f"JIT root does not exist: {root}")

    import torchada
    from torchada.utils.cpp_extension import _port_cuda_source

    source_fingerprint = _fingerprint(root)
    marker_prefix = f"torchada={getattr(torchada, '__version__', 'unknown')}\n"
    marker = marker_prefix + f"sha256={source_fingerprint}\n"
    if sentinel.is_file() and sentinel.read_text() == marker:
        print(f"MUSA_JIT_PORT_REUSED root={root}")
        return 0

    count = 0
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".h", ".cuh", ".cu", ".cpp", ".cc"}:
            continue
        if path.name in _SKIP_NAMES:
            continue
        original = path.read_text()
        ported = _port_cuda_source(original)
        if ported != original:
            path.write_text(ported)
            count += 1

    sentinel.parent.mkdir(parents=True, exist_ok=True)
    final_fingerprint = _fingerprint(root)
    sentinel.write_text(marker_prefix + f"sha256={final_fingerprint}\n")
    print(f"MUSA_JIT_PORTED root={root} files={count} sha256={source_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
