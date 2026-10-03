"""Bound evaluator actors to current native bytes and immutable teacher ancestry."""

import copy
import hashlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_continuous_launcher as launcher_fixtures
import test_native_label_audit_receipts as receipt_fixtures
import train_continuous_w1ax as launcher
import train_prepared_continuous_w1ax as prepared
import w1ax_continuous_stages as stages

from w1a1_eagle.continuous_qat import ContinuousConfig
from w1a1_eagle.qat_recipe_audit import load_comparison_manifest, resolve_comparison_config

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9e2c7a90051e738751aab7d7bd7c2d8201fb76e3"


def actor(root):
    def pin(name):
        path = root / name
        path.write_bytes(name.encode())
        return {"path": str(path), "sha256": launcher.sha256(path)}

    binary, manifest, library = pin("llama-server"), pin("manifest.json"), pin("libggml.so")
    return {
        "binary": binary["path"],
        "binary_sha256": binary["sha256"],
        "native_commit": COMMIT,
        "native_runtime": {
            "schema": "qat_current_native_runtime_v1",
            "immutable_manifest": manifest,
            "libraries": [library],
            "ld_library_path": str(root),
        },
    }


class StandalonePlanTests(unittest.TestCase):
    def test_native_override_keeps_teacher_ancestry_and_freezes_both_shared_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = actor(root)
            original = {
                "schema": "w1ax_continuous_development_v1",
                "split": "development",
                "max_wall_seconds": 1200,
                "sources": {
                    "binary": "frozen-b4",
                    "sha256": {"binary": "b4", "target_gguf": "f16"},
                },
            }
            launcher.atomic_json(
                root / "resolved_config.json",
                {"development": original, "evaluation_native": current},
            )
            effective = launcher.frozen_development_config(root)
            self.assertEqual(effective["teacher_capture_sources"], original["sources"])
            self.assertEqual(effective["sources"]["sha256"]["target_gguf"], "f16")
            self.assertEqual(effective["sources"]["binary"], current["binary"])
            self.assertEqual(effective["sources"]["evaluation_native_commit"], COMMIT)
            self.assertEqual(
                effective["sources"]["evaluation_env"],
                {"GGML_EAGLE_SHARED_PACK": "1", "GGML_EAGLE_PRUNE_UNUSED_HEAD": "1"},
            )
            self.assertEqual(original["sources"]["binary"], "frozen-b4")
            Path(current["native_runtime"]["libraries"][0]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "library/manifest identity differs"):
                launcher.frozen_development_config(root)

    def test_old_native_actor_and_comparison_without_override_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = actor(root)
            current["native_commit"] = "b4" * 20
            with self.assertRaisesRegex(ValueError, "current supported"):
                launcher.validate_evaluation_native(current)
            launcher.atomic_json(
                root / "resolved_config.json",
                {
                    "comparison": {"arm": "candidate"},
                    "development": {
                        "schema": "w1ax_continuous_development_v1",
                        "split": "development",
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "explicit supported"):
                launcher.frozen_development_config(root)

    def test_materialized_both_arms_load_with_hard_ce_and_a8_only(self):
        base = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())
        manifest = load_comparison_manifest(ROOT / "configs/qat_a8_comparison.json")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for arm in ("reference", "candidate"):
                launcher.atomic_json(path, resolve_comparison_config(base, manifest, arm))
                _, config = launcher.load_config(path)
                self.assertEqual(config.activation_bits, (8,))
                self.assertEqual(config.objective, "hard_ce")
                self.assertEqual(config.a1_computation, "reference")
                self.assertEqual(config.max_seconds, 7200)
            invalid = resolve_comparison_config(base, manifest, "candidate")
            invalid["training"]["objective"] = "different"
            launcher.atomic_json(path, invalid)
            with self.assertRaises(ValueError):
                launcher.load_config(path)

    def test_candidate_teacher_provider_does_not_claim_fixed_capture_is_actor_readiness(self):
        manifest = load_comparison_manifest(ROOT / "configs/qat_a8_comparison.json")
        base = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())
        resolved = resolve_comparison_config(base, manifest, "candidate")
        cfg = ContinuousConfig(**resolved["training"]).qat(8)
        with patch(
            "w1ax_capture_provider.NativeCaptureProvider", return_value=types.SimpleNamespace()
        ) as factory:
            child = prepared.create_current_native_child(cfg, "authenticated-manifest")
        teacher = factory.call_args.args[0]
        self.assertEqual(teacher.activation_quantization, "fixed")
        self.assertIsNone(teacher.affine_weights)
        self.assertEqual(child.current_actor_config, cfg)
        self.assertEqual(cfg.activation_quantization, "learned")
        self.assertIsNotNone(cfg.affine_weights)

    def test_pinned_historical_receipt_wrapper_change_reuses_unchanged_actual_capture(self):
        fixture = receipt_fixtures.NativeLabelReceiptTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build(receipt=True)
        receipt = json.loads(fixture.receipt.read_text())
        wrapper = "scripts/w1ax_continuous_stages.py"
        source6f = subprocess.run(
            ["git", "-C", str(ROOT), "show", "6f1444b86dd01862da878c2d5d2434a1d9165c29:" + wrapper],
            check=True,
            capture_output=True,
        ).stdout
        receipt["binding"]["audit_source"]["files"][wrapper] = hashlib.sha256(source6f).hexdigest()
        receipt["audit_origin"] = {"fixture": "authenticated historical full-pass origin"}
        launcher.atomic_json(fixture.receipt, receipt)
        before = fixture.receipt.read_bytes()
        with (
            patch.object(stages, "_validate_historical_receipt_origin") as ancestry,
            patch.object(
                stages, "audit_native_labels", side_effect=AssertionError("semantic re-audit")
            ),
        ):
            loaded = fixture.load()
            self.assertEqual(loaded.report, receipt["report"])
            ancestry.assert_called_once()
        self.assertEqual(before, fixture.receipt.read_bytes())
        # A random old source hash has no compatibility authority.
        receipt["binding"]["audit_source"]["files"][wrapper] = "0" * 64
        launcher.atomic_json(fixture.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "source/input binding differs"):
            fixture.load()

    def test_raw_failed_evaluations_are_never_pruned(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            failures = []
            for index in range(5):
                output = parent / str(index)
                output.mkdir()
                launcher.atomic_json(
                    output / "ownership.json", {"schema": "continuous_development_owned_v1"}
                )
                (output / "raw-error.txt").write_text("raw failure")
                failures.append(output)
            stages.prune_owned_evaluations(parent, 1, preserve_failures=True)
            self.assertTrue(all((output / "raw-error.txt").exists() for output in failures))


class CurrentRecipePreparedReuseTests(unittest.TestCase):
    def test_comparison_and_current_actor_metadata_do_not_change_corpus_declarations(self):
        fixture = launcher_fixtures.PreparedCorpusReuseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.completed_fixture()
        comparison = load_comparison_manifest(ROOT / "configs/qat_a8_comparison.json")
        old = copy.deepcopy(fixture.spec)
        fixture.spec["comparison"] = resolve_comparison_config(old, comparison, "candidate")[
            "comparison"
        ]
        fixture.spec["evaluation_native"] = actor(fixture.root)
        binding, _, development = launcher.prepared_corpus_inputs(
            fixture.spec, fixture.run_dir, fixture.old, fixture.ready_sha
        )
        self.assertEqual(binding["prepared_ready_sha256"], fixture.ready_sha)
        self.assertEqual(development["sources"], old["stages"]["sources"])
