"""One-lane preparation retains authentic admission and operational provenance."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import test_nine_model_admission_plan_builder as plan_fixtures  # noqa: E402
import test_nine_model_bundle_builder as config_fixtures  # noqa: E402
import test_nine_model_sm120_admission as admission_fixtures  # noqa: E402
from run_nine_model_lane import committed_training_checkpoint  # noqa: E402
from run_nine_model_lane import main as run_lane  # noqa: E402

from w1a1_eagle.nine_model_admission import Admission, validate_plan  # noqa: E402
from w1a1_eagle.nine_model_pipeline import GATES, Files, sha256  # noqa: E402

builder = config_fixtures.builder


def pin(path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


class StagedLaneTests(unittest.TestCase):
    def test_eagle_and_block_resume_use_actual_checkpoint_publication_paths(self):
        root = Path("/runs/staged")
        self.assertEqual(
            committed_training_checkpoint(root, "eagle_a8"), root / "training/latest.json"
        )
        self.assertEqual(
            committed_training_checkpoint(root, "dspark_a8"),
            root / "training/checkpoints/latest.json",
        )

    def test_single_operational_config_does_not_claim_human_numeric_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            descriptor, budgets = config_fixtures.SixSourceConfigTests().fixture(root)
            descriptor["schema"] = "nine_model_lane_inputs_v1"
            descriptor["candidates"] = {"eagle_a8": descriptor["candidates"]["eagle_a8"]}
            authorization = root / "authorization.md"
            authorization.write_text("Human delegated relevant overnight QAT and health repair.")
            budget_path = Path(descriptor["budget"]["path"])
            budgets["eagle_a8"]["training_limits"].update(max_steps=None, max_seconds=86400)
            budget_path.write_text(
                json.dumps(
                    {
                        "schema": "nine_model_selected_budget_v1",
                        "human_selected": False,
                        "authorization": {
                            "kind": "human_delegated_operational_settings",
                            "instruction": "Start relevant QAT and continue healthy runs.",
                            "record": pin(authorization),
                        },
                        "candidates": {"eagle_a8": budgets["eagle_a8"]},
                    }
                )
            )
            descriptor["budget"] = pin(budget_path)
            receipt = builder.materialize_configs(descriptor, root / "configs")
            self.assertEqual(set(receipt["configs"]), {"eagle_a8"})
            spec = json.loads(Path(receipt["configs"]["eagle_a8"]["path"]).read_text())
            config = json.loads(Path(spec["eagle_config"]["path"]).read_text())
            self.assertEqual(config["training"]["max_seconds"], 86400)
            self.assertIsNone(config["training"]["max_steps"])
            self.assertFalse(receipt["production_preparation_ready"])
            bad = json.loads(budget_path.read_text())
            bad["human_selected"] = True
            with self.assertRaisesRegex(ValueError, "delegated provenance"):
                builder.selected_budget(bad, ("eagle_a8",), staged=True)

    def test_staged_plan_requires_only_selected_family_and_preserves_full_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            descriptor = plan_fixtures.AdmissionPlanBuilderTests().descriptor(root)
            descriptor["schema"] = "nine_model_lane_inputs_v1"
            descriptor["candidates"] = {"eagle_a8": descriptor["candidates"]["eagle_a8"]}
            descriptor.pop("controls")
            descriptor["inputs"].pop("block_native_binary")
            descriptor["preflight"]["portability"] = {
                "eagle": descriptor["preflight"]["portability"]["eagle"]
            }
            output = root / "lane-plan.json"
            builder.materialize_admission_plan(descriptor, output, fixture=True)
            plan, _ = validate_plan(output, fixture=True)
            self.assertEqual(set(plan["candidates"]), {"eagle_a8"})
            self.assertEqual(set(plan["portability"]), {"eagle"})
            self.assertEqual(plan["controls"], {})
            plan["schema"] = "nine_model_sm120_plan_v1"
            output.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "six candidate"):
                validate_plan(output, fixture=True)

    def test_staged_missing_or_wrong_portability_never_passes(self):
        for names, families in (
            ({}, {}),
            ({"eagle_a8": {}}, {"dspark": {}}),
            ({"unknown_a8": {}}, {"unknown": {}}),
        ):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "plan.json"
                output.write_text(
                    json.dumps(
                        {
                            "schema": "nine_model_lane_sm120_plan_v1",
                            "artifact_kind": "production",
                            "candidates": names,
                            "portability": families,
                        }
                    )
                )
                with self.assertRaises(ValueError):
                    validate_plan(output)

    def test_one_lane_runs_every_gate_and_retains_fixture_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = admission_fixtures.AdmissionTests().fixture(root)
            plan["candidates"] = {"eagle_a8": plan["candidates"]["eagle_a8"]}
            plan["portability"] = {"eagle": plan["portability"]["eagle"]}
            runner = admission_fixtures.Runner(plan)
            result = Admission(
                plan,
                Files(),
                runner,
                admission_fixtures.Resources(),
                "a" * 64,
                root / "run",
                fixture=True,
            ).execute(root / "receipt.json")
            self.assertEqual(result["gates"], dict.fromkeys(sorted(GATES), "PASS"))
            self.assertEqual(len(runner.events), 4)
            record = json.loads(
                Path(result["candidate_admissions"]["eagle_a8"]["path"]).read_text()
            )
            self.assertEqual(
                set(record["checks"]),
                {
                    "source",
                    "resource",
                    "kernel",
                    "model",
                    "backward",
                    "memory",
                    "capture_portability",
                },
            )
            self.assertEqual(record["artifact_kind"], "fixture")

    def test_pause_refuses_lane_before_resource_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            control = root / "control.json"
            control.write_text(json.dumps({"rtx5080": {"pause_requested": True}}))
            lane = {"gpu_control_path": str(control)}
            with (
                patch("run_nine_model_lane.validate_lane", return_value=(lane, None, None)),
                patch("run_nine_model_lane.require_available", return_value={}),
                patch("run_nine_model_lane.LinuxResources") as resources,
            ):
                with self.assertRaisesRegex(InterruptedError, "paused"):
                    run_lane(
                        [
                            "--lane",
                            str(root / "lane"),
                            "--lane-sha256",
                            "a" * 64,
                            "--run-dir",
                            str(root / "run"),
                            "--availability",
                            str(root / "lease"),
                            "--supervisor-state",
                            str(root / "supervisor"),
                            "--start",
                        ]
                    )
                resources.assert_not_called()

    def test_portable_qa_other_lanes_can_stay_pending(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "profiles": {
                "eagle_a8": {
                    "prelaunch_status": "PASS",
                    "prelaunch_requirements": {
                        name: {"status": "PASS", "evidence": [{"scope": "actual_production"}]}
                        for name in builder.PORTABLE
                    },
                }
            },
        }
        self.assertEqual(builder.ledger_pending(ledger, ("eagle_a8",)), [])
        self.assertTrue(builder.ledger_pending(ledger))
        wrong = copy.deepcopy(ledger)
        wrong["profiles"]["eagle_a8"]["prelaunch_requirements"]["data"]["evidence"] = [
            {"scope": "actual_fixture"}
        ]
        self.assertTrue(builder.ledger_pending(wrong, ("eagle_a8",)))


if __name__ == "__main__":
    unittest.main()
