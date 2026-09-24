"""CPU checks for the derived head-only drafter export."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_qat_head.py"
spec = importlib.util.spec_from_file_location("export_qat_head", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ExportQATHeadTests(unittest.TestCase):
    def test_replaces_only_head_and_creates_compatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "models/hf/Qwen3-4B_eagle3"
            source.mkdir(parents=True)
            original = {
                "lm_head.weight": torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16),
                "fc.weight": torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16),
                "d2t": torch.tensor([1.0], dtype=torch.float32),
            }
            save_file(original, source / "model.safetensors", metadata={"format": "pt"})
            (source / "config.json").write_text("{}")
            manifest_path = root / "original.json"
            original_manifest = {
                "models": {
                    "draft": {"directory": str(source), "files": module.snapshot_files(source)},
                    "target": {
                        "directory": str(root / "models/hf/Qwen3-4B"),
                        "repo": "Qwen/Qwen3-4B",
                        "revision": "pinned",
                        "files": [],
                    },
                }
            }
            manifest_path.write_text(json.dumps(original_manifest))
            config_path = root / "configs/pytorch_w1a1_qat_head.toml"
            config_path.parent.mkdir()
            config_path.write_text(
                '[models]\ntarget_repo="Qwen/Qwen3-4B"\ntarget_revision="pinned"\n'
                'draft_repo="local/qat-head-pilot"\ndraft_revision="qat-head-pilot-v1"\n'
                'draft_dir="models/qat-head-pilot-v1"\n'
            )
            trained = torch.tensor([[5.0, -6.0], [-7.0, 8.0]], dtype=torch.bfloat16)
            head_path = root / "best-head.pt"
            torch.save({"weight": trained}, head_path)
            summary_path = root / "training-summary.json"
            summary_path.write_text("{}")
            output = root / "models/qat-head-pilot-v1"
            result_manifest_path = root / "results/export/model-manifest.json"
            result = module.export_checkpoint(
                root,
                config_path,
                manifest_path,
                head_path,
                summary_path,
                output,
                result_manifest_path,
            )
            derived = load_file(output / "model.safetensors")
            self.assertEqual(set(derived), set(original))
            self.assertTrue(torch.equal(derived["lm_head.weight"], trained))
            for name in original.keys() - {"lm_head.weight"}:
                self.assertTrue(torch.equal(derived[name], original[name]))
                self.assertEqual(derived[name].dtype, original[name].dtype)
            with safe_open(output / "model.safetensors", framework="pt") as reader:
                self.assertEqual(reader.metadata(), {"format": "pt"})
            self.assertEqual(result["config_sha256"], module.sha256_file(config_path))
            self.assertEqual(result["models"]["draft"]["files"], module.snapshot_files(output))
            self.assertEqual(json.loads(result_manifest_path.read_text()), result)
            with self.assertRaises(FileExistsError):
                module.export_checkpoint(
                    root,
                    config_path,
                    manifest_path,
                    head_path,
                    summary_path,
                    output,
                    result_manifest_path,
                )

    def test_rejects_wrong_head_shape_and_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            destination = root / "derived.safetensors"
            save_file({"lm_head.weight": torch.ones(2, 3, dtype=torch.bfloat16)}, source)
            head = root / "head.pt"
            for tensor in (torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bfloat16)):
                torch.save({"weight": tensor}, head)
                with self.assertRaises(ValueError):
                    module.replace_head(source, head, destination)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
