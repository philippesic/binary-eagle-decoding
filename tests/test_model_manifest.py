"""A recorded model snapshot must not silently change before evaluation."""

import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_pytorch_w1a1 import sha256_file, verify_model_snapshot


class ModelManifestTests(unittest.TestCase):
    def test_changed_or_extra_model_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            file = directory / "model.safetensors"
            file.write_bytes(b"original")
            entry = {
                "directory": str(directory),
                "files": [
                    {
                        "path": file.name,
                        "bytes": file.stat().st_size,
                        "sha256": sha256_file(file),
                    }
                ],
            }
            verify_model_snapshot(directory, entry)

            file.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                verify_model_snapshot(directory, entry)

            file.write_bytes(b"original")
            (directory / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_model_snapshot(directory, entry)


if __name__ == "__main__":
    unittest.main()
