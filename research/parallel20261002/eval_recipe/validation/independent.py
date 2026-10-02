"""Independent adversarial checks for the development deployment recipe adapter."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import check_continuous_w1ax_readiness as gate
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "tests"),
    str(ROOT / "research/parallel20261002/eval_recipe/reference"),
    str(ROOT / "third_party/llama.cpp/gguf-py"),
]

import adapter
import test_recipe as owner_fixture
import w1ax_continuous_stages as stages
from production_preflight import (
    _development_checkpoint_preflight,
    _development_checkpoint_unchanged,
)

from w1a1_eagle.qat_state import deployment_state_sha256
from w1a1_eagle.recurrent_qat import install_joint_linears


class IndependentRecipeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = Path(cls._run_dir)
        cls.run_dir.mkdir(parents=True, exist_ok=False)
        cls.results = []

    def case_dir(self, name):
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_schema2_fixed_roundtrip_and_combined_effective_state(self):
        root = self.case_dir("schema2-and-combined")
        records = {}
        for bits, profile in ((8, "baseline"), (1, "combined-contract-smoke")):
            folder = root / f"A{bits}"
            folder.mkdir()
            source, checkpoint, manifest = owner_fixture.produce(folder, bits, profile)
            prepared = adapter.preflight_checkpoint(checkpoint, manifest, bits, "a" * 64)
            drafter, target = owner_fixture.tiny()
            restored = adapter.construct_and_load(prepared, drafter, target)
            self.assertEqual(deployment_state_sha256(source), deployment_state_sha256(restored))
            for path, module in source.items():
                values = torch.linspace(-0.71, 0.83, 2 * module.in_features).reshape(2, -1)
                with torch.no_grad():
                    torch.testing.assert_close(
                        module(values), restored[path](values), rtol=3e-6, atol=3e-5
                    )
            with torch.no_grad():
                torch.testing.assert_close(
                    owner_fixture.graph_logits(source),
                    owner_fixture.graph_logits(restored),
                    rtol=3e-6,
                    atol=3e-5,
                )
            state = {"deployment_state_sha256": deployment_state_sha256(restored)}
            if profile == "combined-contract-smoke":
                source_correction = source["fc"].fusion_correction
                restored_correction = restored["fc"].fusion_correction
                with np.load(checkpoint, allow_pickle=False) as archive:
                    self.assertEqual(archive["fc.correction_u.weight"].dtype, np.float16)
                    self.assertEqual(archive["fc.correction_v.weight"].dtype, np.float16)
                    self.assertTrue(
                        np.array_equal(
                            archive["fc.correction_u.weight"],
                            source_correction.u.detach().numpy().astype(np.float16),
                        )
                    )
                expected_effective_u = source_correction.u.detach().half().float()
                torch.testing.assert_close(
                    restored_correction.u, expected_effective_u, rtol=0, atol=0
                )
                self.assertFalse(
                    torch.equal(source_correction.u.detach(), restored_correction.u.detach())
                )
                state["effective_u_dtype"] = str(restored_correction.u.dtype)
                state["exported_u_dtype"] = "float16"
            records[str(bits)] = state
        self.results.append({"check": "schema2_and_combined", "lanes": records})

    def test_invalid_recipe_fails_before_capture_consumer_action(self):
        root = self.case_dir("missing-recipe")
        for bits in (8, 1):
            folder = root / f"A{bits}"
            folder.mkdir()
            owner_fixture.produce(folder, bits, "combined-contract-smoke")
        owner_fixture.publish(root)
        manifest = root / "A1/joint.json"
        data = json.loads(manifest.read_text())
        del data["affine_weights"]
        manifest.write_text(json.dumps(data, sort_keys=True))
        expensive_actions = []

        def proposed_consumer_order():
            prepared = adapter.preflight_pair(root, "a" * 64)
            self.assertEqual(set(prepared), {8, 1})
            expensive_actions.append("native_capture")
            stages.native_capture(None, None, None)

        with patch.object(stages, "native_capture", side_effect=AssertionError("capture called")):
            with self.assertRaisesRegex(ValueError, "export hash mismatch"):
                proposed_consumer_order()
        self.assertEqual(expensive_actions, [])
        self.results.append(
            {"check": "missing_recipe_preflight", "result": "rejected_before_consumer_action"}
        )

    def test_source_bound_pair_helper_reconstructs_both_graphs(self):
        root = self.case_dir("proposed-pair-helper")
        source = {}
        for bits in (8, 1):
            folder = root / f"A{bits}"
            folder.mkdir()
            source[bits], _, _ = owner_fixture.produce(folder, bits, "combined-contract-smoke")
        owner_fixture.publish(root)
        configs, identities = _development_checkpoint_preflight(root, "a" * 64)
        digests = {}
        for bits in (8, 1):
            _development_checkpoint_unchanged(root, bits, identities)
            drafter, target = owner_fixture.tiny()
            restored = install_joint_linears(drafter, target, configs[bits])
            gate._load_checkpoint(
                root / f"A{bits}/joint.npz",
                root / f"A{bits}/joint.json",
                restored,
                bits,
                "a" * 64,
            )
            self.assertEqual(
                deployment_state_sha256(source[bits]), deployment_state_sha256(restored)
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    owner_fixture.graph_logits(source[bits]),
                    owner_fixture.graph_logits(restored),
                    rtol=3e-6,
                    atol=3e-5,
                )
            digests[str(bits)] = deployment_state_sha256(restored)
        self.results.append({"check": "proposed_pair_helper_graph_logits", "lanes": digests})

    def test_published_identity_rejects_learned_recipe_downgrade(self):
        root = self.case_dir("published-downgrade")
        for bits in (8, 1):
            folder = root / f"A{bits}"
            folder.mkdir()
            owner_fixture.produce(folder, bits, "learned-activations")
        owner_fixture.publish(root)
        manifest = root / "A1/joint.json"
        data = json.loads(manifest.read_text())
        del data["activation_quantizers"]
        data["schema_version"] = 2
        data["activation_rule"] = "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
        manifest.write_text(json.dumps(data, sort_keys=True))
        # The bare arrays lack the learned scalars and can be mislabelled fixed.
        standalone = adapter.preflight_checkpoint(root / "A1/joint.npz", manifest, 1, "a" * 64)
        self.assertEqual(standalone.config.activation_quantization, "fixed")
        with self.assertRaisesRegex(ValueError, "export hash mismatch"):
            adapter.preflight_pair(root, "a" * 64)
        self.results.append(
            {
                "check": "published_learned_recipe_downgrade",
                "result": "rejected_by_producer_identity",
            }
        )

    def test_clip_contract_and_mutated_checkpoint_fail_closed(self):
        root = self.case_dir("clip-and-checkpoint-mutation")
        source, checkpoint, manifest = owner_fixture.produce(root, 8, "learned-activations")
        del source
        data = json.loads(manifest.read_text())
        data["activation_quantizers"]["boundaries"]["fc"]["clip_ratio"] = 2**-17
        manifest.write_text(json.dumps(data, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "clip parameter"):
            adapter.preflight_checkpoint(checkpoint, manifest, 8, "a" * 64)

        # Restore a valid producer checkpoint, preflight it, then mutate bytes.
        manifest.unlink()
        checkpoint.unlink()
        _, checkpoint, manifest = owner_fixture.produce(root, 8, "baseline")
        prepared = adapter.preflight_checkpoint(checkpoint, manifest, 8, "a" * 64)
        with checkpoint.open("ab") as stream:
            stream.write(b"changed after preflight")
        install_calls = []
        original_install = adapter.install_joint_linears

        def observed_install(*args, **kwargs):
            install_calls.append(True)
            return original_install(*args, **kwargs)

        with patch.object(adapter, "install_joint_linears", side_effect=observed_install):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                adapter.construct_and_load(prepared, *owner_fixture.tiny())
        self.assertEqual(install_calls, [])
        self.results.append(
            {"check": "clip_and_mutation", "result": "both_rejected_before_module_install"}
        )

    def test_manifest_json_corruption_rejected_before_model_construction(self):
        root = self.case_dir("manifest-identity-mutation")
        _, checkpoint, manifest = owner_fixture.produce(root, 1, "baseline")
        prepared = adapter.preflight_checkpoint(checkpoint, manifest, 1, "a" * 64)
        manifest.write_text(manifest.read_text() + " ")
        installs = []
        original_install = adapter.install_joint_linears

        def observed_install(*args, **kwargs):
            installs.append(True)
            return original_install(*args, **kwargs)

        with patch.object(adapter, "install_joint_linears", side_effect=observed_install):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                adapter.construct_and_load(prepared, *owner_fixture.tiny())
        self.assertEqual(installs, [])
        self.results.append(
            {"check": "manifest_mutation", "result": "rejected_before_module_install"}
        )

    @classmethod
    def tearDownClass(cls):
        report = {
            "schema": "independent_eval_recipe_validation_v1",
            "python": sys.version,
            "torch": torch.__version__,
            "device_policy": "CPU-only; accelerator APIs were not called",
            "results": cls.results,
        }
        (cls.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    IndependentRecipeChecks._run_dir = str(Path(args.run_dir).resolve())
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentRecipeChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
