"""Bad source/capture evidence refuses before a native/GPU query."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w1a1_eagle.nine_model_pipeline import sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "eagle_smoke_test", ROOT / "scripts/check_eagle_binary_native.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inputs = {}
        for key in ("binary", "model", "target", "export_audit", "train_prompts", "train_capture"):
            path = self.root / key
            path.write_text("fixture")
            self.inputs[key] = {"path": str(path), "sha256": sha256(path)}
        self.config = {
            "schema": "eagle_native_graph_smoke_config_v1",
            "inputs": self.inputs,
            "activation_bits": 8,
        }
        self.config_path = self.root / "config.json"

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def test_changed_model_refuses_before_gpu(self):
        self.write()
        Path(self.inputs["model"]["path"]).write_text("changed")
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                smoke.smoke(self.config_path, self.root / "receipt.json")
        self.assertFalse((self.root / "receipt.json").exists())

    def test_export_failed_cannot_be_reinterpreted_as_native_ready(self):
        path = Path(self.inputs["export_audit"]["path"])
        path.write_text(json.dumps({"serialization_audit_passed": False, "activation_bits": 8}))
        self.inputs["export_audit"]["sha256"] = sha256(path)
        self.write()
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "packed EAGLE export"):
                smoke.smoke(self.config_path, self.root / "receipt.json")

    def test_development_capture_cannot_substitute_train_prefix(self):
        audit = Path(self.inputs["export_audit"]["path"])
        audit.write_text(
            json.dumps(
                {
                    "serialization_audit_passed": True,
                    "activation_bits": 8,
                    "output": {"sha256": self.inputs["model"]["sha256"]},
                }
            )
        )
        capture = Path(self.inputs["train_capture"]["path"])
        capture.write_text(json.dumps({"split": "development"}))
        for key in ("export_audit", "train_capture"):
            self.inputs[key]["sha256"] = sha256(self.inputs[key]["path"])
        self.write()
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "TRAIN prefix"):
                smoke.smoke(self.config_path, self.root / "receipt.json")


if __name__ == "__main__":
    unittest.main()
