"""Frozen builder refuses pending or synthetic production dependencies."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from w1a1_eagle.nine_model_pipeline import CELLS, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "bundle_builder_test", ROOT / "scripts/prepare_nine_model_bundle.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def test_draft_exposes_missing_data_and_never_creates_bundle(self):
        result = builder.build(
            {
                "schema": "nine_model_bundle_inputs_v1",
                "pending_dependencies": ["actual five-tap capture unavailable"],
            },
            None,
            inspect_draft=True,
        )
        self.assertEqual(result["status"], "PENDING")
        self.assertIn("actual five-tap capture unavailable", result["pending"])
        self.assertFalse(result["production_bundle_created"])
        self.assertFalse(result["gpu_queried"])

    def test_production_refuses_pending_before_loading_training_api(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bundle.json"
            with self.assertRaisesRegex(ValueError, "production preparation PENDING"):
                builder.build({"schema": "nine_model_bundle_inputs_v1"}, output)
            self.assertFalse(output.exists())

    def test_cpu_synthetic_only_cannot_grant_data_model_export_readiness(self):
        ledger = {"schema": "nine_model_qa_ledger_v1", "profiles": {}}
        for cell in CELLS:
            ledger["profiles"][cell] = {
                "status": "PASS",
                "requirements": {
                    name: {"status": "PASS", "evidence": [{"scope": "cpu_synthetic"}]}
                    for name in builder.PORTABLE
                },
            }
        pending = builder.ledger_pending(ledger)
        self.assertIn("dspark_a8: data has only synthetic/unscoped evidence", pending)
        self.assertIn("eagle_a1: hard_forward has only synthetic/unscoped evidence", pending)
        self.assertIn("dflash_q4: export has only synthetic/unscoped evidence", pending)

    def test_fresh_sm120_pending_stays_separate_from_portable_ledger(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "fresh_sm120": {"status": "PENDING"},
            "profiles": {
                cell: {
                    "status": "PASS",
                    "requirements": {
                        name: {"status": "PASS", "evidence": [{"scope": "native_real_model_sm75"}]}
                        for name in builder.PORTABLE
                    },
                }
                for cell in CELLS
            },
        }
        self.assertEqual(builder.ledger_pending(ledger), [])

    def test_changed_ledger_refuses_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            ledger = root / "ledger.json"
            ledger.write_text(json.dumps({"schema": "nine_model_qa_ledger_v1", "profiles": {}}))
            pin = {"path": str(ledger), "sha256": sha256(ledger)}
            ledger.write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                builder.build(
                    {"schema": "nine_model_bundle_inputs_v1", "qa_ledger": pin},
                    None,
                    inspect_draft=True,
                )


if __name__ == "__main__":
    unittest.main()
