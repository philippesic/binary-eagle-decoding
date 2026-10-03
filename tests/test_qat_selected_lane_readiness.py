"""Selected-lane CPU contracts, without loading models or discovering CUDA."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_qat_optimization_readiness as common  # noqa: E402
import collect_qat_native_evidence as collector  # noqa: E402

from w1a1_eagle.qat_readiness import recipe_identity  # noqa: E402


def spec(bits=None):
    training = {"device": "cuda:0"}
    if bits is not None:
        training["activation_bits"] = bits
    return {
        "schema": "continuous_w1ax_experiment_v1",
        "training": training,
        "model_shapes": list(common.PINNED_SHAPES.values()),
        "native": {"expected_commit": "c" * 40},
        "hardware": {"device_name": "CPU fixture", "compute_capability": [12, 0]},
    }


class SelectedLaneReadinessTests(unittest.TestCase):
    def test_plan_selects_a8_without_runtime_and_preserves_old_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for bits, expected in (([8], ["A8"]), (None, ["A8", "A1"])):
                path.write_text(json.dumps(spec(bits)))
                with (
                    patch.object(common, "runtime_api", side_effect=AssertionError("runtime")),
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    self.assertEqual(common.main(["--config", str(path)]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["planned_cases"]["paired_lanes"], expected)
                self.assertFalse(plan["cuda_discovered"])
                self.assertFalse(plan["model_loaded"])

    def test_invalid_lane_selection_fails_in_cpu_preflight(self):
        for bits in ([], [8, 8], [True], [2], "8", 8):
            with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, "activation_bits"):
                common.validate_config_spec(spec(bits))

    def test_provider_only_instantiates_selected_a8_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "descriptor.json"
            manifest.write_text("{}")
            source = {"split": "train", "execution_manifest_sha256": "d" * 64}
            provider = SimpleNamespace(
                training_eligible=True,
                split="train",
                data_split="train",
                full_body_qat_eligible=True,
                allowed_prompt_ids={"train"},
                source_metadata=source,
                provider_manifest_sha256=common.sha256(manifest),
            )
            qat = Mock(side_effect=lambda bits: bits)
            factory = Mock(return_value=provider)
            config = SimpleNamespace(activation_bits=(8,), qat=qat)
            with patch.object(
                common.importlib, "import_module", return_value=SimpleNamespace(make=factory)
            ):
                result = common.make_provider("fixture:make", manifest, config)
            self.assertIs(result, provider)
            qat.assert_called_once_with(8)
            factory.assert_called_once_with(8, manifest)
            self.assertEqual(source, {"split": "train", "execution_manifest_sha256": "d" * 64})
            provider.provider_manifest_sha256 = "e" * 64
            with patch.object(
                common.importlib, "import_module", return_value=SimpleNamespace(make=factory)
            ):
                with self.assertRaisesRegex(ValueError, "manifest bytes"):
                    common.make_provider("fixture:make", manifest, config)

    def test_bootstrap_recipe_uses_selected_config_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(spec([8])))
            api = SimpleNamespace(
                ContinuousConfig=lambda **training: SimpleNamespace(**training),
                recipe_identity=lambda config: {"activation_bits": config.activation_bits},
                training_runtime_identity=lambda device: {"device": device},
            )
            with patch.object(common, "cuda_environment", return_value={"stub": True}):
                result = collector.recipe_context(
                    SimpleNamespace(config=path, curriculum_config=None), api
                )
            self.assertEqual(result[2], (8,))
            self.assertEqual(result[3], {"activation_bits": [8]})

    def test_a8_native_evidence_keeps_exact_lane_and_numeric_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bound = {
                "recipe": {"activation_bits": [8]},
                "native_commit": "c" * 40,
                "backend": "cuda",
                "hardware": {"fixture": True},
                "training_runtime": {"fixture": True},
                "source_sha256": "s" * 64,
            }
            state = "8" * 64
            artifact = {
                "schema": common.MEASUREMENT_SCHEMA,
                "kind": "native_decisions",
                "split": "train",
                "activation_bits": 8,
                "deployment_state_sha256": state,
                **bound,
                "cases": [
                    {
                        "prompt_id": "train",
                        "capture_id": "capture",
                        "round_index": 0,
                        "state_relative_rms": 0.01,
                        "logit_relative_rms": 0.01,
                        "torch_choice": 3,
                        "native_choice": 3,
                        "native_margin": 0.04,
                    }
                ],
            }
            path = base / "decision.json"
            path.write_text(json.dumps(artifact))
            evidence = {
                "schema": common.NATIVE_SCHEMA,
                "split": "train",
                "fixture_only": False,
                "optimizer_updates": 0,
                **bound,
                "lanes": {
                    "A8": {
                        "deployment_state_sha256": state,
                        "native_decisions": collector.file_record(path),
                    }
                },
            }
            common.preflight_native(evidence, spec([8]), base, expected_lanes=("A8",))
            gates = common.validate_native_evidence(
                evidence, base, bound, {"A8": state}, {"train"}, expected_lanes=("A8",)
            )
            self.assertEqual(gates["native_decisions"]["cases"], 1)
            unexpected = copy.deepcopy(evidence)
            unexpected["lanes"]["A1"] = unexpected["lanes"]["A8"]
            with self.assertRaisesRegex(ValueError, "mismatched"):
                common.preflight_native(unexpected, spec([8]), base, expected_lanes=("A8",))
            artifact["cases"][0].update(torch_choice=4)
            path.write_text(json.dumps(artifact))
            evidence["lanes"]["A8"]["native_decisions"] = collector.file_record(path)
            with self.assertRaisesRegex(ValueError, "material native decision"):
                common.validate_native_evidence(
                    evidence, base, bound, {"A8": state}, {"train"}, expected_lanes=("A8",)
                )

    def test_standalone_evaluation_does_not_change_math_recipe(self):
        reference = {"activation_bits": [8], "development_lifecycle": "in_process"}
        self.assertEqual(
            recipe_identity(reference),
            recipe_identity({**reference, "development_lifecycle": "standalone"}),
        )
        self.assertNotEqual(
            recipe_identity(reference), recipe_identity({"activation_bits": [8, 1]})
        )


if __name__ == "__main__":
    unittest.main()
