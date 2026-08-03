"""Import smoke tests for the diffusion Torch path."""

import subprocess
import sys
import unittest


class TestDiffusionImportIsolation(unittest.TestCase):
    def test_diffusion_import_does_not_import_mlx(self):
        """Diffusion modules must not import the optional SRT MLX providers."""
        script = """
import sys
from sglang.kernels.ops.diffusion import norm_infer
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
assert norm_infer is not None and RMSNorm is not None
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
