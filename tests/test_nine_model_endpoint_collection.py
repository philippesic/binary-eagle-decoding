"""Collection software fixtures only: no weights, CUDA, training or readiness proof."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import evaluate_nine_model_native as evaluator  # noqa: E402
import prepare_nine_model_endpoint_collection as preparer  # noqa: E402

from w1a1_eagle import nine_model_endpoint_collection as collection  # noqa: E402
from w1a1_eagle.nine_model_pipeline import CANDIDATES, FAMILIES, sha256  # noqa: E402


def pin(path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pin(path)


class CollectionTests(unittest.TestCase):
    def fixture(self, root):
        target = write(root, "target", {"fixture": "opaque bytes, not a model"})
        prompts = write(root, "prompts", {"fixture": "not opened by argument inspection"})
        protocol = write(
            root,
            "protocol",
            {
                "schema": "nine_model_native_protocol_v1",
                "split": "development",
                "repetitions": 5,
                "warmups_per_cell": 2,
                "port": 18000,
                "diagnostic_pass_separate_from_timing": True,
                "context_tokens": 2048,
                "batch_tokens": 32,
                "microbatch_tokens": 32,
                "draft_lengths": {"eagle": 5, "dspark": 7, "dflash": 7},
            },
        )
        policy = {"immutable": True, "weights": "f16", "kv": "f16"}
        binary = write(root, "binary", {"fixture": "not executable"})
        source = {
            "target": target,
            "prompts": prompts,
            "protocol": protocol,
            "runtime": {"binary": binary},
            "source": {"source": binary},
            "controls": {},
            "target_policy": policy,
        }
        candidates, contexts, exports, controls = {}, {}, {}, {}
        for name in CANDIDATES:
            selected = {
                key: write(root, name + "/" + key, {"fixture": name + key})
                for key in collection.CANDIDATE_FIELDS - {"heldout_admission"}
            }
            bindings = {"fixture_source": name, "synthetic": False}
            selected["heldout_admission"] = write(
                root,
                name + "/heldout",
                {
                    "schema": "nine_model_lane_development_admission_v1",
                    "split": "development",
                    "training_disjoint": True,
                    "prompts": prompts,
                    "frozen_training_source": bindings,
                    "evidence": [selected["frozen_lane"]],
                },
            )
            candidates[name] = selected
            contexts[name] = {
                "lane": {"candidate": name, "target_policy": policy, "gpu_uuid": "fixture-device"},
                "target": target,
                "source_bindings": bindings,
            }
            exports[name] = {"candidate": name, "model": write(root, name + "/model", name)}
        for family in FAMILIES:
            model = write(root, family + "/original-q4", family)
            coverage = {"fixture": "explicit original normal-precision exceptions"}
            provenance = write(
                root,
                family + "/provenance",
                {
                    "schema": "nine_model_original_q4_reference_v1",
                    "family": family,
                    "precision": "Q4_0",
                    "frozen_original": True,
                    "model": model,
                    "evidence": [binary],
                    "deployment_coverage": coverage,
                },
            )
            controls[family] = {
                "family": family,
                "precision": "Q4_0",
                "frozen_original": True,
                "model": model,
                "provenance": provenance,
                "deployment_coverage": coverage,
            }
        source["controls"] = copy.deepcopy(controls)
        value = {
            "schema": collection.SCHEMA,
            "artifact_kind": "production",
            "campaign_complete": False,
            "candidates": candidates,
            "controls": controls,
            "evaluation_source": write(root, "source-origin", {"fixture": "origin"}),
            "hardware_stage": {
                "gpu_uuid": "fixture-device",
                "compute_capability": [12, 0],
                "runtime_gate": "PENDING_fresh_lease_load_dispatch_release",
            },
        }

        def endpoint(lane, state, supervisor, training, *, files):
            name = next(n for n in candidates if candidates[n]["frozen_lane"] == lane)
            # The genuine staged validator owns serialized optimizer/RNG/cursor/source proof.
            self.assertEqual(
                (state, supervisor, training),
                tuple(
                    candidates[name][key]
                    for key in ("lane_state", "supervisor_state", "training_receipt")
                ),
            )
            for record in (lane, state, supervisor, training):
                files.check(record)
            return contexts[name]

        def export(context, record, *, files):
            name = context["lane"]["candidate"]
            self.assertEqual(record, candidates[name]["export_receipt"])
            files.check(record)
            files.check(exports[name]["model"])
            return exports[name]

        return value, source, contexts, exports, (endpoint, export)

    def validate(self, value, source, validators):
        # Fixture EAGLE bytes have no released model hash. The real production
        # path still compares the immutable known EAGLE hash, tested separately.
        with patch(
            "prepare_nine_model_lane_endpoint.ORIGINAL_EAGLE_Q4_SHA256",
            value["controls"]["eagle"]["model"]["sha256"],
        ):
            return collection.validate_collection(
                value, validators=validators, source_loader=lambda locator, files: source
            )

    def test_independent_lane_hashes_preserved_and_native_argv_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            value, source, _, exports, validators = self.fixture(Path(tmp))
            before = copy.deepcopy(value)
            context = self.validate(value, source, validators)
            self.assertEqual(value, before)
            proof = collection.inspection(context)
            self.assertEqual(len(set(proof["training_bundle_sha256"].values())), 6)
            self.assertFalse(proof["production_ready"])
            self.assertFalse(proof["execution_allowed"])
            view, models = collection.evaluation_view(context)
            for name in (*evaluator.CELLS, "target_only"):
                argv = evaluator.native_command(
                    view, context["protocol"], name, models.get(name), 18000
                )
                self.assertEqual(argv[argv.index("-m") + 1], source["target"]["path"])
                self.assertEqual(argv[argv.index("--cache-type-k") + 1], "f16")
                if name == "target_only":
                    self.assertEqual(argv[-2:], ["--spec-type", "none"])
                elif name in exports:
                    self.assertEqual(argv[argv.index("-md") + 1], exports[name]["model"]["path"])

    def test_mixed_target_hardware_relabelled_model_and_private_heldout_rejected(self):
        for mutation in ("target", "gpu", "name", "model", "heldout", "control"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                value, source, contexts, exports, validators = self.fixture(Path(tmp))
                if mutation == "target":
                    contexts["eagle_a8"]["target"] = value["evaluation_source"]
                elif mutation == "gpu":
                    contexts["eagle_a8"]["lane"]["gpu_uuid"] = "foreign"
                elif mutation == "name":
                    contexts["eagle_a8"]["lane"]["candidate"] = "eagle_a1"
                elif mutation == "model":
                    exports["eagle_a1"]["model"] = exports["eagle_a8"]["model"]
                elif mutation == "control":
                    value["controls"]["dspark"]["model"] = value["controls"]["dflash"]["model"]
                else:
                    value["candidates"]["eagle_a8"]["heldout_admission"] = value["candidates"][
                        "eagle_a1"
                    ]["heldout_admission"]
                with self.assertRaises((ValueError, AssertionError)):
                    self.validate(value, source, validators)

    def test_fixture_readiness_foreign_fields_missing_and_hash_tampering_rejected(self):
        for mutation in ("fixture", "ready", "extra", "missing", "tamper"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                value, source, _, _, validators = self.fixture(Path(tmp))
                if mutation == "fixture":
                    value["artifact_kind"] = "synthetic"
                elif mutation == "ready":
                    value["hardware_stage"]["runtime_gate"] = "PASS"
                elif mutation == "extra":
                    value["candidates"]["eagle_a8"]["bundle_sha256"] = "f" * 64
                elif mutation == "missing":
                    del value["controls"]["dflash"]
                else:
                    Path(value["candidates"]["dspark_a8"]["training_receipt"]["path"]).write_text(
                        "x"
                    )
                with self.assertRaises(ValueError):
                    self.validate(value, source, validators)

    def test_draft_cannot_claim_validation_or_create_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "collection.json"
            result = preparer.prepare(
                {"schema": "nine_model_endpoint_collection_inputs_v1"}, output, inspect_draft=True
            )
            self.assertEqual(result["status"], "PENDING")
            self.assertIn("eagle_a8/training_receipt", result["missing"])
            self.assertFalse(output.exists())
            self.assertFalse(result["gpu_queried"])

    def test_execution_route_fails_closed_before_resource_observer(self):
        args = SimpleNamespace(collection=Path("/fixture"), inspect_collection=False)
        with (
            patch.object(evaluator, "inspect_collection", return_value={}),
            patch.object(
                evaluator, "LinuxResources", side_effect=AssertionError("must not query GPU")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "fresh supervised collection"):
                evaluator.run(args)

    def test_collection_and_bundle_routes_must_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collection"
            path.write_text("{}")
            args = SimpleNamespace(
                collection=path,
                collection_sha256=sha256(path),
                bundle=Path("/foreign"),
                bundle_sha256=None,
                evaluation_inputs=None,
            )
            with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                evaluator.inspect_collection(args)

    def test_single_lane_source_cannot_self_assert_missing_block_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            value, source, _, _, validators = self.fixture(Path(tmp))
            del source["controls"]["dspark"]
            with self.assertRaisesRegex(ValueError, "original control source PENDING: dspark"):
                self.validate(value, source, validators)

    def test_original_eagle_hash_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            value, source, _, _, validators = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "original EAGLE hash differs"):
                collection.validate_collection(
                    value, validators=validators, source_loader=lambda locator, files: source
                )


if __name__ == "__main__":
    unittest.main()
