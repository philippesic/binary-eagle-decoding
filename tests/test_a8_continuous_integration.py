"""Actual tiny CPU A8 update/resume/export tests; no GPU claims."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_continuous_qat import FixtureProvider, config, make
from test_development_checkpoint_preflight import TinyProvider

import train_continuous_w1ax as launcher
import w1ax_continuous_stages as stages
from w1a1_eagle.continuous_qat import (
    ContinuousTrainer,
    atomic_json,
    build_lanes,
    sha256,
    validate_lane_ownership,
)
from w1a1_eagle.recurrent_qat import joint_parameter_families


def a8(**kwargs):
    return config(activation_bits=(8,), **kwargs)


def candidate(**kwargs):
    return a8(
        activation_quantization="learned",
        affine_weights={"enabled": True, "coverage": "all"},
        binary_optimization={"latent_magnitude": 0.1, "sign_lr": 0.01, "scale_lr": 0.001},
        optimize_cache=True,
        optimize_head=True,
        persistent_sign_diagnostics=True,
        **kwargs,
    )


class A8IntegrationTests(unittest.TestCase):
    def test_a8_does_not_construct_second_model(self):
        provider = FixtureProvider()
        with patch("w1a1_eagle.continuous_qat.copy.deepcopy", side_effect=AssertionError("A1 copied")):
            lanes = build_lanes(provider, a8())
        self.assertEqual([lane.name for lane in lanes], ["A8"])

    def test_optional_parameters_move_and_exact_positive_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            direct = make(root / "direct", candidate(max_steps=5))
            lane = direct.lanes[0]
            before = {name: [p.detach().clone() for p in params]
                      for name, params in joint_parameter_families(lane.linears).items()}
            direct.run(require_smoke=False)
            for family in ("sign", "scale", "activation", "midpoint"):
                actual = joint_parameter_families(lane.linears)[family]
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[family], actual)), family)
            first = make(root / "resume", candidate(max_steps=2))
            first.run(require_smoke=False)
            resumed = make(root / "resume", candidate(max_steps=5))
            resumed.resume()
            self.assertEqual(resumed.step, 2)
            resumed.run(require_smoke=False)
            self.assertEqual(resumed.unique_rows, direct.unique_rows)
            self.assertEqual(resumed.cursor, direct.cursor)
            for name, expected in direct.lanes[0].linears.items():
                actual = resumed.lanes[0].linears[name]
                for key, value in expected.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
            for expected, actual in zip(direct.lanes[0].optimizer.param_groups,
                                        resumed.lanes[0].optimizer.param_groups):
                for pa, pb in zip(expected["params"], actual["params"]):
                    for key, value in direct.lanes[0].optimizer.state[pa].items():
                        torch.testing.assert_close(value, resumed.lanes[0].optimizer.state[pb][key], rtol=0, atol=0)
            publication = json.loads((Path(resumed.checkpoint["path"]).parent / "manifest.json").read_text())
            self.assertEqual(set(publication["exports"]), {"A8"})
            self.assertEqual(lane.config.contract.activation_bits, 8)
            self.assertEqual(direct.metrics["A8"]["execution"]["reason"], "trainable_learned_activation")
            self.assertGreater(direct.metrics["A8"]["execution"]["context_cache_calls"], 0)

    def test_missing_midpoint_optimizer_family_fails_before_update(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), candidate(max_steps=1))
            lane = trainer.lanes[0]
            lane.optimizer.param_groups[:] = [g for g in lane.optimizer.param_groups if g["family"] != "midpoint"]
            with self.assertRaisesRegex(ValueError, "parameter families differ"):
                validate_lane_ownership(lane)

    def test_standalone_boundary_releases_by_exit_and_requires_exact_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = replace(a8(max_steps=3, development_lifecycle="standalone"), development_every=2)
            trainer = make(root, cfg)
            trainer.evaluator = lambda *_: self.fail("in-process evaluation called")
            trainer.run(require_smoke=False)
            self.assertEqual(trainer.step, 2)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "awaiting_development")
            with self.assertRaisesRegex(ValueError, "must finish"):
                launcher.require_development_result(root, trainer.checkpoint)
            report = {"checkpoint": str(Path(trainer.checkpoint["path"]).parent),
                      "split": "development", "sealed_test_accessed": False}
            atomic_json(root / "development.json", report)
            receipt = {"checkpoint": trainer.checkpoint, "completed": True,
                       "report_path": str(root / "development.json"),
                       "report_sha256": sha256(root / "development.json")}
            atomic_json(root / "development-result.json", receipt)
            launcher.require_development_result(root, trainer.checkpoint)
            resumed = make(root, cfg)
            resumed.resume()
            elapsed = resumed.elapsed_seconds
            resumed.run(require_smoke=False)
            self.assertEqual(resumed.step, 3)
            self.assertGreaterEqual(resumed.elapsed_seconds, elapsed)
            launcher.require_development_result(root, resumed.checkpoint)
            receipt["checkpoint"] = {**trainer.checkpoint, "sha256": "0" * 64}
            atomic_json(root / "development-result.json", receipt)
            with self.assertRaisesRegex(ValueError, "authenticate"):
                launcher.require_development_result(root, resumed.checkpoint)

    def test_a8_manifest_preflight_replays_optional_recipe_without_a1(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = TinyProvider()
            cfg = candidate(max_steps=1)
            trainer = ContinuousTrainer(provider, build_lanes(provider, cfg), cfg, Path(folder))
            trainer.save()
            checkpoint_dir = Path(trainer.checkpoint["path"]).parent
            with patch.object(stages, "host_admission", return_value={}):
                configs, identities = stages._development_checkpoint_preflight(checkpoint_dir, "a" * 64)
            self.assertEqual(set(configs), {8})
            self.assertEqual(set(identities["lanes"]), {8})
            self.assertEqual(configs[8].affine_weights, cfg.affine_weights)
            self.assertEqual(configs[8].activation_quantization, "learned")

    def test_unowned_trainable_model_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), a8(max_steps=1))
            trainer.lanes[0].drafter.register_parameter("accidental", torch.nn.Parameter(torch.ones(1)))
            with self.assertRaisesRegex(ValueError, "parameter families differ"):
                validate_lane_ownership(trainer.lanes[0])


if __name__ == "__main__":
    unittest.main()
