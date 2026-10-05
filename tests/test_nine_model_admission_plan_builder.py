"""Actual source CLI plan construction; fixtures never admit production training."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from w1a1_eagle.nine_model_pipeline import CANDIDATES, FAMILIES, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import test_nine_model_bundle_builder as fixtures  # noqa: E402

builder = fixtures.builder


def api():
    from w1a1_eagle import nine_model_admission

    return nine_model_admission


def pin(path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


class AdmissionPlanBuilderTests(unittest.TestCase):
    def descriptor(self, root, *, warm=False):
        descriptor, budgets = fixtures.SixSourceConfigTests().fixture(root)
        binary = root / "source-binary-fixture"
        binary.write_bytes(b"source bindings only; not executable or native proof")
        target = pin(binary)
        descriptor["inputs"] = {
            role: target
            for role in (
                "backend_binary",
                "block_native_binary",
                "teacher_binary",
                "binary",
                "target",
            )
        }
        descriptor["controls"] = {
            family: {"model": target, "frozen_original": True} for family in FAMILIES
        }
        descriptor["gpu_uuid"] = "GPU-source-fixture"
        descriptor["resource_policy"] = {
            "host_floor_bytes": 1,
            "gpu_floor_bytes": 1,
            "host_return_tolerance_bytes": 0,
            "gpu_return_tolerance_bytes": 0,
        }
        descriptor["preflight"] = {"native_source_revision": "a" * 40, "portability": {}}
        prompts = root / "train-prompts.jsonl"
        prompts.write_text(
            json.dumps(
                {
                    "id": "dolly:line-fixture",
                    "split": "train",
                    "messages": [{"role": "user", "content": "source fixture"}],
                }
            )
        )
        capture = root / "train-capture.json"
        capture.write_text(
            json.dumps(
                {
                    "split": "train",
                    "target_sha256": target["sha256"],
                    "prompts_sha256": sha256(prompts),
                }
            )
        )
        descriptor["inputs"].update(
            eagle_smoke_prompts=pin(prompts), eagle_smoke_capture=pin(capture)
        )
        prepared = root / "prepared"
        prepared.mkdir()
        ready = prepared / "preparation-ready.json"
        ready.write_text(
            json.dumps(
                {
                    "schema": "continuous_w1ax_preparation_ready_v1",
                    "preparation_complete": True,
                    "optimization_started": False,
                    "teacher_coverage": {
                        "source": {
                            "common_source_sha256": {"target_gguf": target["sha256"]},
                            "round_count": 1,
                        }
                    },
                }
            )
        )
        for name in CANDIDATES:
            selected = descriptor["candidates"][name]
            family, precision = name.split("_")
            bits = int(precision[1:])
            selected["initial_model"] = target
            export = root / (name + "-export.json")
            export.write_text(
                json.dumps(
                    {
                        "serialization_audit_passed": True,
                        "activation_bits": bits,
                        "output": {"sha256": target["sha256"]},
                        "projections": {"fc": {"shape": [8, 40]}},
                    }
                )
            )
            selected["initial_export_audit"] = pin(export)
            if family == "eagle":
                selected["prepared"] = {"run_dir": str(prepared), "ready_sha256": sha256(ready)}
                selected["data_admission"] = pin(ready)
            else:
                data = root / (name + "-data.json")
                data.write_text(json.dumps({"producer": {"target_sha256": target["sha256"]}}))
                selected["data"] = pin(data)
                completed = root / (name + "-data-admission.json")
                completed.write_text(
                    json.dumps(
                        {
                            "schema": "block_data_completed_admission_v1",
                            "manifest_sha256": sha256(data),
                        }
                    )
                )
                selected["data_admission"] = pin(completed)
            if warm and bits == 1:
                selected["profile"] = "a8_to_a1_reset"
                selected["initialization"]["activation_bits"] = 8
                selected["a8_warmup_steps"] = 1
                budgets[name]["training_limits"]["max_steps"] = 5
                budgets[name]["min_a1_updates"] = 2
                if family == "eagle":
                    budgets[name]["curriculum"] = {
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
        for family in FAMILIES:
            golden = root / (family + "-golden.json")
            golden.write_text(
                json.dumps(
                    {
                        "schema": "nine_model_train_capture_goldens_v1",
                        "family": family,
                        "target_sha256": target["sha256"],
                    }
                )
            )
            descriptor["preflight"]["portability"][family] = {"golden_manifest": pin(golden)}
        builder.materialize_configs(descriptor, root / "configs")
        return json.loads((root / "configs/resolved-inputs.json").read_text())

    def test_missing_production_dependencies_exposed_without_gpu_or_models(self):
        report = builder.materialize_admission_plan({}, None, inspect_draft=True)
        self.assertTrue(report["pending"])
        self.assertEqual(report["status"], "PENDING")
        self.assertFalse(report["gpu_queried"])
        with self.assertRaisesRegex(ValueError, "admission preparation PENDING"):
            builder.materialize_admission_plan({}, Path("must-not-be-created.json"))

    def test_direct_six_and_warm_three_plans_bind_real_clis_without_circular_hash(self):
        module = api()
        for warm in (False, True):
            with self.subTest(warm=warm), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                descriptor = self.descriptor(root, warm=warm)
                path = root / "admission-plan.json"
                receipt = builder.materialize_admission_plan(
                    descriptor, path, fixture=True, api=module
                )
                plan, _ = module.validate_plan(path, fixture=True)
                self.assertFalse(receipt["production_ready"])
                self.assertFalse(receipt["gpu_queried"])
                resolved = json.loads(Path(receipt["resolved_inputs"]["path"]).read_text())
                self.assertEqual(resolved["inputs"]["admission_plan"], receipt["plan"])
                self.assertEqual(set(plan["candidates"]), set(CANDIDATES))
                self.assertEqual(plan["python_invocation"], sys.executable)
                self.assertEqual(plan["python"]["path"], str(Path(sys.executable).resolve()))
                self.assertEqual(plan["python"]["sha256"], sha256(Path(sys.executable)))
                self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-source-fixture")
                for name, candidate in plan["candidates"].items():
                    self.assertNotIn("bundle_sha256", candidate["source_bindings"])
                    self.assertIn("--smoke-zero-updates", candidate["backward"]["argv"])
                    self.assertNotIn("--resume", candidate["backward"]["argv"])
                    self.assertIn("{bundle_sha256}", candidate["backward"]["argv"])
                    self.assertIn("{receipt}", candidate["native"]["argv"])
                    if warm and name.endswith("a1"):
                        self.assertEqual(candidate["profile"], "a8_to_a1_reset")
                        self.assertEqual(candidate["precision_stage"], "a8_to_a1")
                    else:
                        self.assertEqual(candidate["precision_stage"], "direct")
                self.assertEqual(
                    Path(plan["portability"]["eagle"]["producer"]["path"]).name,
                    "check_eagle_capture_portability.py",
                )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    module.validate_plan(path)

    def test_fixture_golden_metadata_cannot_emit_production_preflight_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            descriptor = self.descriptor(root)
            output = root / "production-plan.json"
            with self.assertRaisesRegex(ValueError, "bounded native TRAIN golden"):
                builder.materialize_admission_plan(descriptor, output, api=api())
            self.assertFalse(output.exists())

    def test_plan_flags_exist_in_actual_cpu_only_cli_help(self):
        module = api()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            descriptor = self.descriptor(root)
            path = root / "admission-plan.json"
            builder.materialize_admission_plan(descriptor, path, fixture=True, api=module)
            plan, _ = module.validate_plan(path, fixture=True)
            specs = [s for c in plan["candidates"].values() for s in (c["native"], c["backward"])]
            specs.extend(plan["portability"].values())
            help_cache = {}
            for spec in specs:
                producer = spec["producer"]["path"]
                if producer not in help_cache:
                    result = subprocess.run(
                        [sys.executable, producer, "--help"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    help_cache[producer] = result.stdout
                for argument in spec["argv"]:
                    if argument.startswith("--"):
                        self.assertIn(argument, help_cache[producer])


if __name__ == "__main__":
    unittest.main()
