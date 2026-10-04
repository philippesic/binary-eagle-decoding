"""Frozen builder refuses pending or synthetic production dependencies."""

import dataclasses
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from w1a1_eagle.nine_model_pipeline import CANDIDATES, CELLS, sha256

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


class SixSourceConfigTests(unittest.TestCase):
    def test_six_real_source_profiles_parse_without_model_or_gpu(self):
        from w1a1_eagle.block_qat import BlockQATConfig
        from w1a1_eagle.continuous_qat import ContinuousConfig

        if "latent_initialization" not in {
            f.name for f in dataclasses.fields(BlockQATConfig)
        } or "initialization_sha256" not in {f.name for f in dataclasses.fields(ContinuousConfig)}:
            self.skipTest("reference-magnitude source integration PENDING")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()

            def pin(path):
                return {"path": str(path), "sha256": sha256(path)}

            artifact = root / "source-fixture"
            artifact.write_bytes(b"fixture metadata only; never model or GPU loaded")
            artifact_pin = pin(artifact)
            candidates = {}
            budgets = {}
            for candidate in CANDIDATES:
                family = candidate.split("_")[0]
                bits = 8 if candidate.endswith("a8") else 1
                candidates[candidate] = {
                    "profile": "fixed_reference" if bits == 8 else "direct_a1",
                    "initialization": {
                        **artifact_pin,
                        "activation_bits": bits,
                        "latent_initialization": {
                            "policy": "preserve_reference_magnitudes",
                            "reference_kind": "eagle_fixed_reference_0.5"
                            if family == "eagle"
                            else "block_source_weight_magnitudes",
                            "reference_sha256": "a" * 64,
                        },
                    },
                    "base_model": artifact_pin,
                    "data": artifact_pin,
                    "checkpoint_every": 1,
                    "deployment_coverage": {"profile": "ffn15_fusion"},
                    "eagle_config_template": pin(ROOT / "configs/continuous_w1ax.json"),
                    "prepared": {
                        "run_dir": str(root / "missing-production-data"),
                        "ready_sha256": "a" * 64,
                    },
                }
                budgets[candidate] = {
                    "training_limits": {
                        "max_steps": 1,
                        ("max_tokens" if family == "eagle" else "max_supervised_tokens"): None,
                        "max_seconds": None,
                        "max_epochs": None,
                    }
                }
            budget = root / "budget.json"
            budget.write_text(
                json.dumps(
                    {
                        "schema": "nine_model_selected_budget_v1",
                        "human_selected": True,
                        "candidates": budgets,
                    }
                )
            )
            receipt = builder.materialize_configs(
                {
                    "schema": "nine_model_bundle_inputs_v1",
                    "candidates": candidates,
                    "budget": pin(budget),
                },
                root / "configs",
            )
            self.assertEqual(set(receipt["configs"]), set(CANDIDATES))
            self.assertEqual(receipt["source_validation"], "PASS")
            self.assertFalse(receipt["production_preparation_ready"])
            self.assertFalse(receipt["gpu_queried"])
            for name, record in receipt["configs"].items():
                spec = json.loads(Path(record["path"]).read_text())
                self.assertEqual(spec["candidate"], name)
                self.assertEqual(
                    spec["initialization"]["latent_initialization"]["policy"],
                    "preserve_reference_magnitudes",
                )


if __name__ == "__main__":
    unittest.main()
