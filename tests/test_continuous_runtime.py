"""CPU source/version identity fixtures; never query an accelerator."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from w1a1_eagle.continuous_runtime import MATH_FILES, training_runtime_identity


class RuntimeIdentityTests(unittest.TestCase):
    def test_versions_math_hashes_and_docs_only_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in MATH_FILES:
                (root / name).write_text("# CPU math fixture " + name)
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
                identity = training_runtime_identity("cpu", source_root=root)
            self.assertEqual(identity["torch_version"], str(torch.__version__))
            self.assertIn("python_version", identity)
            self.assertIn("numpy_version", identity)
            self.assertNotIn("cuda_math", identity)
            self.assertEqual(set(identity["math_source_sha256"]), set(MATH_FILES))
            (root / "unrelated-documentation.md").write_text("documentation commit fixture")
            self.assertEqual(identity, training_runtime_identity("cpu", source_root=root))
            (root / "recurrent_qat.py").write_text("# changed math fixture")
            updated = training_runtime_identity("cpu", source_root=root)
            self.assertNotEqual(
                identity["math_source_sha256"]["recurrent_qat.py"],
                updated["math_source_sha256"]["recurrent_qat.py"],
            )


if __name__ == "__main__":
    unittest.main()
