import subprocess
import sys
import unittest

import torch

from medusa.model import triton_kernels as tk


class TritonAvailabilityTest(unittest.TestCase):
    def test_triton_imports_and_wrappers_are_available(self):
        self.assertTrue(tk.TRITON_AVAILABLE, "Triton should import in the project environment.")
        for name in (
            "qjl_path_scores_triton",
            "packed_kv_qjl_node_scores_triton",
            "turbo_qjl_select_paths_triton",
            "turbo_vq_append_triton",
            "compressed_kv_attention_turbo_vq_triton",
            "hybrid_kv_attention_turbo_vq_triton",
        ):
            self.assertTrue(callable(getattr(tk, name, None)), f"{name} is not callable")

    def test_check_script_runs_without_cuda(self):
        result = subprocess.run(
            [sys.executable, "scripts/check_triton_kernels.py"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("triton: available", result.stdout)
        if not torch.cuda.is_available():
            self.assertIn("skipped Triton kernel launches", result.stdout)


if __name__ == "__main__":
    unittest.main()
