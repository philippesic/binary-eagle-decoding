"""CPU recurrent fixtures; CUDA/Metal and native target execution are prohibited."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_recurrent_provider import dense_drafter, provider_round
from torch import nn

from w1a1_eagle.continuous_qat import (
    ContinuousConfig,
    ContinuousTrainer,
    build_lanes,
    memory_estimate,
)
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, shared_round_hard_signs


class FixtureProvider:
    split = "train"
    training_eligible = True
    allowed_prompt_ids = {"p", "q"}
    target_vocab_size = 4
    draft_vocab_size = 3
    max_depth = 2
    d2t_offsets = (0, 1, 1)
    base_gguf_sha256 = "a" * 64
    candidate_d = None
    source_metadata = {"schema": "CPU_fixture", "capture_sha256": "b" * 64}

    def load_models(self):
        return dense_drafter(native_shape=True), nn.Linear(4, 4)

    def make_step_adapter(self, drafter):
        return NativeStepAdapter(drafter)

    def rounds(self):
        first = replace(
            provider_round(),
            raw_target_features=torch.tensor([[0.2, -0.1, 0.3, 0.4] * 3]),
            teacher_arrays=None,
        )
        yield first
        yield replace(
            first,
            anchor=replace(first.anchor, prompt_id="q"),
            rows=tuple(dict(row, prompt_id="q") for row in first.rows),
        )


def config(**kwargs):
    return ContinuousConfig(
        checkpoint_every=1, min_free_disk_bytes=0, development_every=100, warmup_steps=0, **kwargs
    )


def make(root, cfg):
    torch.random.default_generator.manual_seed(77)
    provider = FixtureProvider()
    return ContinuousTrainer(provider, build_lanes(provider, cfg), cfg, root)


class ContinuousTests(unittest.TestCase):
    def test_quantized_sign_reuse_preserves_forward_and_accumulated_gradients(self):
        module = RowBinaryLinear(
            torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([0.2, 0.3]), W1AxContract(8)
        )
        x = torch.tensor([0.4, -0.5])
        (module(x) + module(x * 2)).sum().backward()
        expected = module.latent_sign.grad.clone()
        module.zero_grad(set_to_none=True)
        with shared_round_hard_signs({"fixture": module}):
            identity = id(module._round_hard_signs)
            value = (module(x) + module(x * 2)).sum()
            self.assertEqual(id(module._round_hard_signs), identity)
        self.assertIsNone(module._round_hard_signs)
        value.backward()
        torch.testing.assert_close(module.latent_sign.grad, expected)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with shared_round_hard_signs({"fixture": module}):
                raise RuntimeError("fixture")
        self.assertIsNone(module._round_hard_signs)

    def test_both_models_interleave_same_coverage_with_distinct_optimizers(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), config(max_steps=3))
            self.assertIs(
                trainer.lanes[0].drafter.embed_tokens.weight,
                trainer.lanes[1].drafter.embed_tokens.weight,
            )
            self.assertIsNot(
                trainer.lanes[0].linears["fc"].latent_sign,
                trainer.lanes[1].linears["fc"].latent_sign,
            )
            trainer.run(require_smoke=False)
            status = json.loads((Path(folder) / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["models"]["A8"]["step"], 3)
            self.assertEqual(status["models"]["A1"]["step"], 3)
            self.assertEqual(status["presented_supervised_tokens"], 6)
            self.assertEqual(status["unique_supervised_rows"], 4)
            self.assertEqual(status["unique_prompts"], 2)
            self.assertIsNone(status["models"]["A1"]["saturation_mean"])
            self.assertTrue(
                all(
                    p.grad is None
                    for lane in trainer.lanes
                    for module in lane.linears.values()
                    for p in module.parameters()
                )
            )

    def test_exact_optimizer_rng_cursor_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = make(root / "full", config(max_steps=5))
            full.run(require_smoke=False)
            first = make(root / "resume", config(max_steps=3))
            first.run(require_smoke=False)
            resumed = make(root / "resume", config(max_steps=5))
            resumed.resume()
            self.assertEqual(resumed.cursor, 1)
            resumed.run(require_smoke=False)
            self.assertEqual(resumed.tokens, full.tokens)
            for expected_lane, actual_lane in zip(full.lanes, resumed.lanes):
                for name in expected_lane.linears:
                    for key, value in expected_lane.linears[name].state_dict().items():
                        torch.testing.assert_close(
                            actual_lane.linears[name].state_dict()[key], value, rtol=0, atol=0
                        )
                for expected, actual in zip(
                    expected_lane.optimizer.state.values(), actual_lane.optimizer.state.values()
                ):
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual_lane.rng["torch"], expected_lane.rng["torch"], rtol=0, atol=0
                )

    def test_stop_file_commits_pair_and_checkpoint_retention_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = make(root, config(max_steps=5, keep_checkpoints=2))
            original = trainer.log

            def stop_after_pair(item):
                original(item)
                if item["step"] == 3:
                    (root / "STOP").touch()

            trainer.log = stop_after_pair
            trainer.run(require_smoke=False)
            self.assertEqual(trainer.step, 3)
            self.assertEqual(len(list((root / "checkpoints").glob("step-*"))), 2)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "stopped")

    def test_finite_failure_preserves_prior_paired_checkpoint(self):
        from w1a1_eagle.continuous_qat import joint_train_step

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = make(root / "crashed", config(max_steps=4))
            count = 0

            def fail_second_a1(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 4:
                    raise ValueError("nonfinite joint QAT gradient fixture")
                return joint_train_step(*args, **kwargs)

            with patch("w1a1_eagle.continuous_qat.joint_train_step", side_effect=fail_second_a1):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    trainer.run(require_smoke=False)
            status = json.loads((root / "crashed/status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["models"]["A8"]["step"], 2)
            self.assertEqual(status["models"]["A1"]["step"], 1)
            self.assertTrue(status["resume_from_last_committed_pair"])
            self.assertEqual(json.loads((root / "crashed/latest.json").read_text())["step"], 1)
            resumed = make(root / "crashed", config(max_steps=4))
            resumed.resume()
            resumed.run(require_smoke=False)
            full = make(root / "full", config(max_steps=4))
            full.run(require_smoke=False)
            for expected, actual in zip(full.lanes, resumed.lanes):
                for name in expected.linears:
                    for key, value in expected.linears[name].state_dict().items():
                        torch.testing.assert_close(
                            actual.linears[name].state_dict()[key], value, rtol=0, atol=0
                        )
                for wanted, got in zip(
                    expected.optimizer.state.values(), actual.optimizer.state.values()
                ):
                    for key in wanted:
                        torch.testing.assert_close(got[key], wanted[key], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.rng["torch"], expected.rng["torch"], rtol=0, atol=0
                )

    def test_hash_contract_and_disk_gate_fail_safely(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = make(root, config(max_steps=1))
            trainer.run(require_smoke=False)
            resumed = make(root, config(max_steps=2))
            resumed.source = "changed"
            with self.assertRaisesRegex(ValueError, "immutable"):
                resumed.resume()
            checkpoint = Path(json.loads((root / "latest.json").read_text())["path"])
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"):
                resumed.resume()
            with patch(
                "w1a1_eagle.continuous_qat.shutil.disk_usage",
                return_value=type("Usage", (), {"free": 0})(),
            ):
                with self.assertRaisesRegex(RuntimeError, "disk"):
                    make(root / "disk", config(max_steps=1)).save()

    def test_development_callback_is_serial_and_checkpoint_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = replace(config(max_steps=4), development_every=2)
            trainer = make(root / "evaluated", cfg)
            observed = []

            def evaluator(checkpoint, lanes):
                status = json.loads((root / "evaluated/status.json").read_text())
                self.assertEqual(status["status"], "development_evaluation")
                self.assertEqual(lanes[0].name, "A8")
                self.assertEqual(lanes[1].name, "A1")
                self.assertEqual(status["models"]["A8"]["step"], checkpoint["step"])
                self.assertEqual(status["models"]["A1"]["step"], checkpoint["step"])
                for lane in lanes:
                    self.assertTrue(
                        (Path(checkpoint["path"]).parent / lane.name / "joint.npz").exists()
                    )
                    self.assertTrue(
                        all(
                            p.grad is None
                            for group in lane.optimizer.param_groups
                            for p in group["params"]
                        )
                    )
                observed.append(checkpoint["step"])
                return {"execution": "CPU_callback_fixture", "split": "development"}

            trainer.evaluator = evaluator
            trainer.run(require_smoke=False)
            self.assertEqual(observed, [2, 4])
            baseline = make(root / "baseline", cfg)
            baseline.run(require_smoke=False)
            for expected, actual in zip(baseline.lanes, trainer.lanes):
                for name in expected.linears:
                    for key, value in expected.linears[name].state_dict().items():
                        torch.testing.assert_close(
                            actual.linears[name].state_dict()[key], value, rtol=0, atol=0
                        )

    def test_graceful_interrupt_during_development_keeps_paired_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = replace(config(max_steps=4), development_every=2)
            trainer = make(root, cfg)

            def evaluator(checkpoint, lanes):
                trainer.stop_requested = True
                raise InterruptedError("intentional native stage signal fixture")

            trainer.evaluator = evaluator
            trainer.run(require_smoke=False)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "stopped")
            self.assertTrue(status["intentional_stop_during_development"])
            self.assertEqual(status["models"]["A8"]["step"], 2)
            self.assertEqual(status["models"]["A1"]["step"], 2)
            self.assertEqual(json.loads((root / "latest.json").read_text())["step"], 2)

    def test_memory_estimator_uses_cpu_shape_arithmetic_only(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
            result = memory_estimate([(4, 12), (3, 4)], frozen_bytes=100, graph_budget_bytes=200)
        self.assertEqual(result["trainable_parameters_per_model"], 67)
        self.assertEqual(result["dual_persistent_bytes"], 2 * (12 * 67 + 4 * 7) + 100)

    def test_smoke_is_required_and_calibration_is_ineligible(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), config(max_steps=1))
            with self.assertRaisesRegex(ValueError, "smoke"):
                trainer.run()
            provider = FixtureProvider()
            provider.readiness_scope = "row_a16_hard_ce_100_steps"
            with self.assertRaisesRegex(ValueError, "calibration"):
                ContinuousTrainer(provider, trainer.lanes, trainer.config, Path(folder))

    def test_later_only_gradient_gates_both_native_structural_fixtures(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), config(max_steps=1))
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
                report = trainer.smoke(next(trainer.provider.rounds()))
            for name in ("A8", "A1"):
                for field in (
                    "later_state_gradient_norm",
                    "later_k_gradient_norm",
                    "later_v_gradient_norm",
                ):
                    self.assertGreater(report[name][field], 0)
            trainer.run()


if __name__ == "__main__":
    unittest.main()
