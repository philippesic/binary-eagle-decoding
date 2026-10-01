"""CPU fixtures only: no real data, CUDA, native server or sealed-final access."""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_continuous_qat import FixtureProvider

from w1a1_eagle.fusion_correction import FusionCorrectionConfig
from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage
from w1a1_eagle.qat_curriculum_runner import CurriculumRunner, RunnerConfig
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract


class TrainProvider(FixtureProvider):
    full_body_qat_eligible = True
    readiness_scope = "full_body_qat"


def stages(bits=(8, 4, 1), updates=2, seconds=1000):
    return CurriculumConfig(tuple(PrecisionStage(b, seconds, updates) for b in bits))


def make(root, curriculum=None, *, provider=None, recipe=False, **kwargs):
    torch.random.default_generator.manual_seed(77)
    curriculum = curriculum or stages()
    cfg = JointQATConfig(
        W1AxContract(curriculum.stages[0].activation_bits),
        sign_lr=0.01,
        activation_quantization="learned" if recipe else "fixed",
        fusion_correction=FusionCorrectionConfig(enabled=True, output_bias=True)
        if recipe
        else None,
    )
    runner_config = kwargs.pop(
        "config",
        RunnerConfig(
            checkpoint_every=1,
            min_free_disk_bytes=0,
            warmup_updates=6,
            update_upper_bound_seconds=0.01,
        ),
    )
    return CurriculumRunner(
        provider or TrainProvider(),
        curriculum,
        cfg,
        root,
        config=runner_config,
        start=True,
        **kwargs,
    )


def compare_models(left, right):
    for path, expected in left.linears.items():
        actual = right.linears[path]
        for key, value in expected.state_dict().items():
            other = actual.state_dict()[key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other, value, rtol=0, atol=0)
            else:
                assert other == value
    for expected, actual in zip(left.optimizer.state.values(), right.optimizer.state.values()):
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            else:
                assert actual[key] == value


class CurriculumRunnerTests(unittest.TestCase):
    def test_three_stages_one_model_fresh_optimizers_and_global_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder))
            model_id = id(trainer.drafter)
            parameters = {
                p: (id(m.latent_sign), id(m.scale_offset)) for p, m in trainer.linears.items()
            }
            trainer.run(require_smoke=False)
            self.assertEqual(id(trainer.drafter), model_id)
            self.assertTrue(trainer.state.complete)
            self.assertEqual(trainer.state.global_updates, 6)
            self.assertEqual([p["supported_rows"] for p in trainer.state.phases], [4, 4, 4])
            self.assertEqual(len(trainer.state.transitions), 2)
            for path, module in trainer.linears.items():
                self.assertEqual(
                    (id(module.latent_sign), id(module.scale_offset)), parameters[path]
                )
                self.assertIsNone(module._round_hard_signs)
            self.assertTrue(all(int(s["step"]) == 2 for s in trainer.optimizer.state.values()))
            metrics = [
                json.loads(line)
                for line in (Path(folder) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([m["warmup_factor"] for m in metrics], [i / 6 for i in range(1, 7)])
            self.assertGreaterEqual(
                trainer.occupancy_seconds, sum(p["gpu_seconds"] for p in trainer.state.phases)
            )

    def test_exact_midphase_and_boundary_resume_with_learned_and_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = make(root / "full", recipe=True)
            full.run(require_smoke=False)
            for count in (2, 3):
                partial = make(root / str(count), recipe=True)
                partial.run(require_smoke=False, max_new_updates=count)
                resumed = make(root / str(count), recipe=True)
                resumed.resume()
                resumed.smoke_current()
                resumed.run()
                compare_models(full, resumed)
                self.assertEqual(resumed.unique_rows, full.unique_rows)
                self.assertEqual(resumed.unique_prompts, full.unique_prompts)
                self.assertEqual(resumed.state.global_updates, 6)
                torch.testing.assert_close(resumed.rng["torch"], full.rng["torch"], rtol=0, atol=0)

    def test_prepare_smokes_every_stage_without_optimizer_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True)
            before = {p: m.latent_sign.detach().clone() for p, m in trainer.linears.items()}
            with patch.object(
                torch.optim.AdamW, "step", side_effect=AssertionError("update forbidden")
            ):
                report = trainer.prepare(next(trainer.provider.rounds()))
            self.assertEqual([r["activation_bits"] for r in report], [8, 4, 1])
            self.assertEqual(trainer.state.global_updates, 0)
            self.assertEqual(trainer.optimizer.state, {})
            self.assertTrue(trainer.smoke_passed)
            for path, module in trainer.linears.items():
                torch.testing.assert_close(module.latent_sign, before[path], rtol=0, atol=0)
            trainer.resume()
            self.assertEqual(trainer.state.global_updates, 0)

    def test_transition_preserves_weight_magnitudes_scales_and_fc(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True)
            trainer.run(require_smoke=False, max_new_updates=1)
            old = {
                p: (m.latent_sign.detach().clone(), m.effective_scales().detach().clone())
                for p, m in trainer.linears.items()
            }
            correction = trainer.linears["fc"].fusion_correction
            correction_values = {k: v.clone() for k, v in correction.state_dict().items()}
            trainer.state.finish_phase()
            trainer._transition()
            self.assertIs(trainer.linears["fc"].fusion_correction, correction)
            self.assertEqual(trainer.optimizer.state, {})
            for path, module in trainer.linears.items():
                torch.testing.assert_close(module.latent_sign, old[path][0], rtol=0, atol=0)
                torch.testing.assert_close(module.effective_scales(), old[path][1], rtol=0, atol=0)
            for key, value in correction_values.items():
                torch.testing.assert_close(correction.state_dict()[key], value, rtol=0, atol=0)

    def test_direct_a1_and_a8_to_a1_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            for bits in ((1,), (8, 1)):
                trainer = make(Path(folder) / str(len(bits)), stages(bits, updates=3))
                trainer.run(require_smoke=False)
                self.assertTrue(trainer.state.complete)
                self.assertEqual(trainer.state.global_updates, 3 * len(bits))

    def test_source_and_immutable_config_resume_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = make(root)
            original.run(require_smoke=False, max_new_updates=1)
            changed = make(root, config=replace(original.config, warmup_updates=7))
            with self.assertRaisesRegex(ValueError, "contract"):
                changed.resume()
            provider = TrainProvider()
            provider.source_metadata = {**provider.source_metadata, "new": True}
            changed = make(root, provider=provider)
            with self.assertRaisesRegex(ValueError, "contract"):
                changed.resume()

    def test_overrun_keeps_last_committed_checkpoint(self):
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            trainer = make(Path(folder), stages((1,), updates=3, seconds=1), clock=clock)
            from w1a1_eagle.qat_curriculum_runner import joint_train_step

            def slow(*args, **kwargs):
                result = joint_train_step(*args, **kwargs)
                clock.now += 2
                return result

            with patch("w1a1_eagle.qat_curriculum_runner.joint_train_step", side_effect=slow):
                with self.assertRaisesRegex(ValueError, "exceeds current phase GPU budget"):
                    trainer.run(require_smoke=False)
            pointer = json.loads((Path(folder) / "latest.json").read_text())
            self.assertEqual(pointer["global_updates"], 0)
            self.assertEqual(
                json.loads((Path(folder) / "status.json").read_text())["status"], "failed"
            )
            recovered = make(Path(folder), stages((1,), updates=3, seconds=1), clock=Clock())
            with self.assertRaisesRegex(ValueError, "budget-overrun"):
                recovered.resume()

    def test_fresh_stage_source_revalidation_refuses_changed_ancestry(self):
        with tempfile.TemporaryDirectory() as folder:

            def revalidate(qat):
                provider = TrainProvider()
                if qat.contract.activation_bits == 4:
                    provider.source_metadata = {**provider.source_metadata, "changed": True}
                return provider

            trainer = make(Path(folder), source_revalidator=revalidate)
            with self.assertRaisesRegex(ValueError, "source revalidation"):
                trainer.run(require_smoke=False)
            self.assertEqual(trainer.state.global_updates, 2)
            self.assertEqual(trainer.model_phase, 0)

    def test_budget_can_finish_phase_early_without_resetting_exposures(self):
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            curriculum = CurriculumConfig((PrecisionStage(8, 1, 100), PrecisionStage(1, 10, 1)))
            trainer = make(
                Path(folder),
                curriculum,
                clock=clock,
                config=RunnerConfig(
                    min_free_disk_bytes=0, checkpoint_every=1, update_upper_bound_seconds=0.6
                ),
            )
            from w1a1_eagle.qat_curriculum_runner import joint_train_step

            def measured(*args, **kwargs):
                result = joint_train_step(*args, **kwargs)
                clock.now += 0.5
                return result

            with patch("w1a1_eagle.qat_curriculum_runner.joint_train_step", side_effect=measured):
                trainer.run(require_smoke=False)
            self.assertEqual([p["updates"] for p in trainer.state.phases], [1, 1])
            self.assertEqual(trainer.state.global_updates, 2)
            self.assertEqual(trainer.state.transitions[0]["global_updates"], 1)

    def test_atomic_pointer_failure_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder))
            first = trainer.save()
            from w1a1_eagle.qat_curriculum_runner import atomic_json

            def fail_pointer(path, value):
                if path.name == "latest.json":
                    raise OSError("pointer interrupted")
                atomic_json(path, value)

            with patch("w1a1_eagle.qat_curriculum_runner.atomic_json", side_effect=fail_pointer):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    trainer.save()
            self.assertEqual(json.loads((Path(folder) / "latest.json").read_text()), first)
            trainer.resume()

    def test_calibration_and_unauthorized_construction_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "explicit start"):
                CurriculumRunner(
                    TrainProvider(), stages((1,)), JointQATConfig(W1AxContract(1)), Path(folder)
                )
            provider = TrainProvider()
            provider.readiness_scope = "row_a16_hard_ce_100_steps"
            with self.assertRaisesRegex(ValueError, "calibration"):
                make(Path(folder), provider=provider)

    def test_resume_rejects_duplicate_activation_alias_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True)
            model = trainer._model_payload()
            model["linears"]["fc"]["activation_quantizer.clip_ratio"] = torch.tensor(0.5)
            with self.assertRaisesRegex(ValueError, "shared aliases"):
                trainer._load_model(model)

    def test_cli_default_validates_only_without_model_or_cuda(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_qat_curriculum.py"
        spec = importlib.util.spec_from_file_location("curriculum_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        qat = JointQATConfig(W1AxContract(8))
        with (
            patch.object(
                cli, "load_config", return_value=({"provider": {}}, stages(), qat, RunnerConfig())
            ),
            patch.object(cli, "provider_factory", return_value=lambda _: TrainProvider()),
            patch.object(
                TrainProvider, "load_models", side_effect=AssertionError("model load forbidden")
            ),
            patch.object(TrainProvider, "load_models_cpu", create=True, return_value=None),
            patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA forbidden")),
            patch("builtins.print") as output,
        ):
            cli.main(["--config", "/unused"])
            report = json.loads(output.call_args.args[0])
            self.assertFalse(report["optimization_started"])
            self.assertFalse(report["model_loaded"])
            self.assertFalse(report["cuda_started"])


if __name__ == "__main__":
    unittest.main()
