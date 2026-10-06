"""Frozen builder refuses pending or synthetic production dependencies."""

import dataclasses
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                "status": "PENDING",
                "prelaunch_status": "PASS",
                "prelaunch_requirements": {
                    name: {"status": "PASS", "evidence": [{"scope": "cpu_synthetic"}]}
                    for name in builder.PORTABLE
                },
            }
        pending = builder.ledger_pending(ledger)
        self.assertIn("dspark_a8: data lacks explicit actual production evidence scope", pending)
        self.assertIn(
            "eagle_a1: hard_forward lacks explicit actual production evidence scope", pending
        )
        self.assertIn(
            "dflash_q4: initial_export lacks explicit actual production evidence scope", pending
        )

    def test_current_synthetic_scope_variants_and_empty_scopes_cannot_grant_prelaunch(self):
        for scope in (
            "cpu_synthetic_qat_and_native_graph",
            "native_cpu_synthetic_15_ffn_plus_optional_fusion",
            "",
            "   ",
            "native_mock",
            "actual_fixture",
            "toy_model",
            None,
            "source_only",
        ):
            with self.subTest(scope=scope):
                self.assertFalse(builder.actual_evidence_scope(scope))
        for scope in (
            "actual_model_cpu_native_TRAIN",
            "native_real_model_sm75",
            "production_initial_export",
            "actual_calibrated_initializer",
        ):
            self.assertTrue(builder.actual_evidence_scope(scope))

    def test_fresh_sm120_pending_stays_separate_from_portable_ledger(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "fresh_sm120": {"status": "PENDING"},
            "profiles": {
                cell: {
                    "status": "PENDING",
                    "training": "PENDING",
                    "quality_evaluation": "PENDING",
                    "prelaunch_status": "PASS",
                    "prelaunch_requirements": {
                        name: {"status": "PASS", "evidence": [{"scope": "native_real_model_sm75"}]}
                        for name in builder.PORTABLE
                    },
                }
                for cell in CELLS
            },
        }
        self.assertEqual(builder.ledger_pending(ledger), [])

    def test_aggregate_pass_cannot_replace_explicit_prelaunch_contract(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "profiles": {
                cell: {
                    "status": "PASS",
                    "requirements": {
                        name: {
                            "status": "PASS",
                            "evidence": [{"scope": "actual_native_initializer"}],
                        }
                        for name in builder.PORTABLE
                    },
                }
                for cell in CELLS
            },
        }
        pending = builder.ledger_pending(ledger)
        self.assertIn("eagle_a8: prelaunch production dependencies PENDING", pending)
        self.assertIn("eagle_a8: initial_export PENDING", pending)

    def test_missing_data_or_calibration_still_refuses_prelaunch_pass(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "profiles": {
                cell: {
                    "status": "PENDING",
                    "prelaunch_status": "PASS",
                    "prelaunch_requirements": {
                        name: {
                            "status": "PASS",
                            "evidence": [{"scope": "actual_native_initializer"}],
                        }
                        for name in builder.PORTABLE
                    },
                }
                for cell in CELLS
            },
        }
        ledger["profiles"]["dspark_a8"]["prelaunch_requirements"]["data"]["status"] = "PENDING"
        ledger["profiles"]["eagle_a1"]["prelaunch_requirements"]["calibration_initialization"][
            "status"
        ] = "PENDING"
        pending = builder.ledger_pending(ledger)
        self.assertIn("dspark_a8: data PENDING", pending)
        self.assertIn("eagle_a1: calibration_initialization PENDING", pending)

    def test_all_declared_source_files_must_match_not_only_minimum_list(self):
        path = "src/w1a1_eagle/nine_model_report.py"
        pending = builder.source_pending({"source_files": {path: "f" * 64}})
        self.assertIn("independent current source evidence PENDING: " + path, pending)
        with self.assertRaisesRegex(ValueError, "repository relative"):
            builder.source_pending({"source_files": {"../outside": "f" * 64}})

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


class PausedLaunchTests(unittest.TestCase):
    def test_paused_start_refuses_before_source_audit_or_gpu_query(self):
        spec = importlib.util.spec_from_file_location(
            "paused_campaign_cli", ROOT / "scripts/run_nine_model_campaign.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            control = root / "gpu-control.json"
            control.write_text(json.dumps({"rtx5080": {"pause_requested": True}}))
            with (
                patch.object(cli, "validate_bundle", side_effect=AssertionError("source audit")),
                patch.object(cli, "LinuxResources", side_effect=AssertionError("GPU query")),
            ):
                with self.assertRaisesRegex(InterruptedError, "paused; no GPU query"):
                    cli.main(
                        [
                            "--start",
                            "--bundle",
                            str(root / "not-opened.json"),
                            "--bundle-sha256",
                            "a" * 64,
                            "--gpu-control",
                            str(control),
                        ]
                    )


class SixSourceConfigTests(unittest.TestCase):
    def set_timed_budget(self, descriptor, budgets):
        for entry in budgets.values():
            entry["training_limits"]["max_seconds"] = 43200
        path = Path(descriptor["budget"]["path"])
        content = json.loads(path.read_text())
        content["candidates"] = budgets
        path.write_text(json.dumps(content))
        descriptor["budget"]["sha256"] = sha256(path)

    def fixture(self, root):
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
        return {
            "schema": "nine_model_bundle_inputs_v1",
            "candidates": candidates,
            "budget": pin(budget),
        }, budgets

    def test_six_real_source_profiles_parse_without_model_or_gpu(self):
        from w1a1_eagle.block_qat import BlockQATConfig
        from w1a1_eagle.continuous_qat import ContinuousConfig

        if "latent_initialization" not in {
            f.name for f in dataclasses.fields(BlockQATConfig)
        } or "initialization_sha256" not in {f.name for f in dataclasses.fields(ContinuousConfig)}:
            self.skipTest("reference-magnitude source integration PENDING")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()

            descriptor, _ = self.fixture(root)
            receipt = builder.materialize_configs(
                descriptor,
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
                for field in (
                    "checkpoint_retention",
                    "evaluation_milestones_seconds",
                    "evaluation_protocol",
                ):
                    self.assertNotIn(field, spec)

    def test_explicit_execution_controls_reach_real_source_configs(self):
        import train_nine_model_qat as trainer

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            descriptor, budgets = self.fixture(root)
            self.set_timed_budget(descriptor, budgets)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"schema": "source-fixture-protocol"}))
            locator = {"path": str(protocol), "sha256": sha256(protocol)}
            retention = {"keep_recent": 3, "max_checkpoints": 8, "max_bytes": 40 * 1024**3}
            for name, selected in descriptor["candidates"].items():
                selected.update(
                    evaluation_milestones_seconds=[14400, 28800, 43200],
                    evaluation_protocol=locator,
                )
                if not name.startswith("eagle"):
                    selected["checkpoint_retention"] = retention
            result = builder.materialize_configs(descriptor, root / "configs")
            for name, pin in result["configs"].items():
                spec = trainer.load_spec(pin["path"])
                self.assertEqual(pin["sha256"], sha256(Path(pin["path"])))
                self.assertEqual(spec["evaluation_milestones_seconds"], [14400, 28800, 43200])
                self.assertEqual(spec["evaluation_protocol"], locator)
                if name.startswith("eagle"):
                    self.assertNotIn("checkpoint_retention", spec)
                else:
                    self.assertEqual(spec["checkpoint_retention"], retention)

    def test_invalid_optional_controls_refuse_even_without_source_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            descriptor, budgets = self.fixture(root)
            self.set_timed_budget(descriptor, budgets)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            locator = {"path": str(protocol), "sha256": sha256(protocol)}
            good = {
                "evaluation_milestones_seconds": [14400, 28800, 43200],
                "evaluation_protocol": locator,
            }
            cases = [
                {"checkpoint_retention": None},
                {"checkpoint_retention": {"keep_recent": 1, "max_checkpoints": 1}},
                {
                    "checkpoint_retention": {
                        "keep_recent": True,
                        "max_checkpoints": 2,
                        "max_bytes": 99,
                    }
                },
                {"checkpoint_retention": {"keep_recent": 3, "max_checkpoints": 2, "max_bytes": 99}},
                {"evaluation_protocol": locator},
                {"evaluation_milestones_seconds": [14400, 28800, 43200]},
                *[
                    {**good, "evaluation_milestones_seconds": values}
                    for values in (
                        None,
                        [],
                        [True, 43200],
                        [14400, 14400, 43200],
                        [28800, 14400, 43200],
                        [14400, float("inf"), 43200],
                        [14400, 28800],
                        [1, 2, 43200],
                        [14400, 28800, 43201],
                    )
                ],
                {**good, "evaluation_protocol": {**locator, "extra": True}},
                {**good, "evaluation_protocol": {**locator, "sha256": "0" * 64}},
                {**good, "profile": "a8_to_a1_reset"},
            ]
            original = dict(descriptor["candidates"]["dspark_a8"])
            for index, change in enumerate(cases):
                with self.subTest(change=change):
                    descriptor["candidates"]["dspark_a8"] = original | change
                    output = root / f"invalid-{index}"
                    with self.assertRaises(ValueError):
                        builder.materialize_configs(descriptor, output, source_validate=False)
                    self.assertFalse(output.exists())
            descriptor["candidates"]["dspark_a8"] = original
            descriptor["candidates"]["eagle_a8"]["checkpoint_retention"] = {
                "keep_recent": 1,
                "max_checkpoints": 2,
                "max_bytes": 99,
            }
            with self.assertRaisesRegex(ValueError, "only by the block saver"):
                builder.materialize_configs(
                    descriptor, root / "eagle-retention", source_validate=False
                )

    def test_three_warm_profiles_use_real_curriculum_and_block_source_apis(self):
        import train_nine_model_qat as trainer

        if not hasattr(trainer, "run_eagle_curriculum"):
            self.skipTest("EAGLE warm source integration PENDING")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            descriptor, budgets = self.fixture(root)
            for candidate in ("eagle_a1", "dspark_a1", "dflash_a1"):
                selected = descriptor["candidates"][candidate]
                selected["profile"] = "a8_to_a1_reset"
                selected["initialization"]["activation_bits"] = 8
                selected["a8_warmup_steps"] = 1
                budgets[candidate]["training_limits"]["max_steps"] = 5
                budgets[candidate]["min_a1_updates"] = 2
                if candidate == "eagle_a1":
                    budgets[candidate]["curriculum"] = {
                        "optimizer_transition": "fresh",
                        "stages": [
                            {"activation_bits": 8, "gpu_seconds": 100, "max_updates": 2},
                            {"activation_bits": 1, "gpu_seconds": 200, "max_updates": 3},
                        ],
                    }
            budget = Path(descriptor["budget"]["path"])
            content = json.loads(budget.read_text())
            content["candidates"] = budgets
            budget.write_text(json.dumps(content))
            descriptor["budget"]["sha256"] = sha256(budget)
            receipt = builder.materialize_configs(descriptor, root / "warm-configs")
            self.assertEqual(receipt["source_validation"], "PASS")
            for candidate in ("eagle_a1", "dspark_a1", "dflash_a1"):
                spec = json.loads(Path(receipt["configs"][candidate]["path"]).read_text())
                self.assertEqual(spec["precision_stage"], "a8_to_a1")
                self.assertEqual(spec["min_a1_updates"], 2)
            self.assertFalse(receipt["production_preparation_ready"])


if __name__ == "__main__":
    unittest.main()
