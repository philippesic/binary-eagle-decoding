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
from test_recurrent_provider import dense_drafter
from torch import nn

from w1a1_eagle.affine_binary import AffineBinaryConfig
from w1a1_eagle.fusion_correction import FusionCorrectionConfig
from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage
from w1a1_eagle.qat_curriculum_runner import CurriculumRunner, RunnerConfig
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract


class TrainProvider(FixtureProvider):
    full_body_qat_eligible = True
    readiness_scope = "full_body_qat"

    def load_models(self):
        # Eight-wide native topology avoids the exact four-wide binary
        # cancellation that legitimately fails a strict scale-gradient gate.
        model = dense_drafter(native_shape=True)
        model.config.hidden_size = 8
        model.config.head_dim = 4
        model.config.intermediate_size = 9
        model.embed_tokens = nn.Embedding(5, 8, dtype=torch.float16)
        shapes = {
            "fc": (8, 24),
            "midlayer.self_attn.q_proj": (8, 16),
            "midlayer.self_attn.k_proj": (4, 16),
            "midlayer.self_attn.v_proj": (4, 16),
            "midlayer.self_attn.o_proj": (8, 8),
            "midlayer.mlp.gate_proj": (9, 8),
            "midlayer.mlp.up_proj": (9, 8),
            "midlayer.mlp.down_proj": (8, 9),
            "lm_head": (3, 8),
        }
        generator = torch.Generator().manual_seed(0)
        for path, (out_features, in_features) in shapes.items():
            parent, _, name = path.rpartition(".")
            module = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
            with torch.no_grad():
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.2)
            setattr(model.get_submodule(parent) if parent else model, name, module)
        with torch.no_grad():
            model.embed_tokens.weight.copy_(torch.randn((5, 8), generator=generator) * 0.2)
        for path in (
            "midlayer.input_layernorm",
            "midlayer.hidden_norm",
            "midlayer.post_attention_layernorm",
            "norm",
        ):
            model.get_submodule(path).weight = nn.Parameter(torch.ones(8), requires_grad=False)
        return model, nn.Linear(4, 4)

    def rounds(self):
        features = torch.randn((1, 24), generator=torch.Generator().manual_seed(0)) * 0.2
        for batch in super().rounds():
            yield replace(batch, raw_target_features=features)


def stages(bits=(8, 4, 1), updates=2, seconds=1000):
    return CurriculumConfig(tuple(PrecisionStage(b, seconds, updates) for b in bits))


def make(root, curriculum=None, *, provider=None, recipe=False, affine=False, **kwargs):
    torch.random.default_generator.manual_seed(77)
    curriculum = curriculum or stages()
    cfg = JointQATConfig(
        W1AxContract(curriculum.stages[0].activation_bits),
        sign_lr=0.01,
        affine_weights=AffineBinaryConfig(enabled=True, coverage="all", midpoint_bound=0.1)
        if affine
        else None,
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

    def test_combined_learned_raw_fusion_affine_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = make(root / "full-affine", recipe=True, affine=True)
            full.run(require_smoke=False)
            for count in (1, 2, 3, 4):
                partial = make(root / str(count), recipe=True, affine=True)
                partial.run(require_smoke=False, max_new_updates=count)
                resumed = make(root / str(count), recipe=True, affine=True)
                resumed.resume()
                report = resumed.smoke_current()
                self.assertEqual(report["optional_gradients"]["midpoint"]["parameter_tensors"], 9)
                resumed.run()
                compare_models(full, resumed)
                self.assertEqual(resumed.state.global_updates, 6)
                self.assertEqual(resumed.unique_rows, full.unique_rows)
                torch.testing.assert_close(resumed.rng["torch"], full.rng["torch"], rtol=0, atol=0)

    def test_combined_prepare_reports_all_nine_and_later_state_kv(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True, affine=True)
            report = trainer.prepare(next(trainer.provider.rounds()))
            self.assertEqual(trainer.state.global_updates, 0)
            self.assertEqual(trainer.optimizer.state, {})
            self.assertEqual([item["activation_bits"] for item in report], [8, 4, 1])
            for item in report:
                self.assertTrue(item["all_nine_binary_gradients_passed"])
                self.assertTrue(item["later_state_kv_passed"])
                self.assertEqual(set(item["binary_gradients"]), set(trainer.linears))
                for values in item["binary_gradients"].values():
                    self.assertGreater(values["sign_gradient_norm"], 0)
                    self.assertGreater(values["scale_gradient_norm"], 0)
                for key in (
                    "later_state_gradient_norm",
                    "later_k_gradient_norm",
                    "later_v_gradient_norm",
                ):
                    self.assertGreater(item[key], 0)
                self.assertEqual(
                    item["optional_gradients"]["activation"]["finite_gradient_tensors"], 6
                )
                self.assertEqual(item["optional_gradients"]["fusion"]["finite_gradient_tensors"], 3)
                self.assertEqual(
                    item["optional_gradients"]["midpoint"]["finite_gradient_tensors"], 9
                )
                # Raw low-rank V legitimately has an attached zero gradient at U=0.
                self.assertIn(0.0, item["optional_gradients"]["fusion"]["gradient_norms"])

    def test_zero_scale_preparation_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder))
            with torch.no_grad():
                trainer.linears["fc"].initial_scale.zero_()
                trainer.linears["fc"].scale_offset.zero_()
            with self.assertRaisesRegex(ValueError, "(all-nine|attached later-position)"):
                trainer.prepare(next(trainer.provider.rounds()))
            self.assertFalse(trainer.smoke_passed)
            self.assertEqual(trainer.state.global_updates, 0)
            self.assertFalse((Path(folder) / "preparation-ready.json").exists())

    def test_later_kv_disconnect_gate_refuses_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder))
            from w1a1_eagle.qat_curriculum_runner import later_gradient

            def disconnected(*args):
                diagnostics = later_gradient(*args)
                diagnostics["later_k_gradient_norm"] = 0.0
                return diagnostics

            with patch("w1a1_eagle.qat_curriculum_runner.later_gradient", side_effect=disconnected):
                with self.assertRaisesRegex(ValueError, "attached later-position"):
                    trainer.smoke_current()
            self.assertFalse(trainer.smoke_passed)
            self.assertEqual(trainer.state.global_updates, 0)

    def test_frozen_declared_midpoint_refuses_smoke_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True, affine=True)
            trainer.linears["fc"].affine_binary.midpoint.requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "must remain trainable"):
                trainer.smoke_current()
            self.assertFalse(trainer.smoke_passed)

    def test_one_disconnected_binary_projection_fails_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True, affine=True)
            module = trainer.linears["midlayer.mlp.gate_proj"]

            def eliminate(gradient):
                return torch.zeros_like(gradient)

            hook = module.scale_offset.register_hook(eliminate)
            try:
                with self.assertRaisesRegex(ValueError, "gate_proj.scale: all-nine"):
                    trainer.smoke_current()
            finally:
                hook.remove()
            self.assertFalse(trainer.smoke_passed)
            self.assertEqual(trainer.state.global_updates, 0)

    def test_affine_midpoint_canonical_bounds_and_transition_preservation(self):
        from w1a1_eagle.affine_binary import _tensor_sha256

        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), recipe=True, affine=True)
            trainer.run(require_smoke=False, max_new_updates=1)
            bank = trainer.drafter.qat_affine_bank
            before = {
                path: (id(row.midpoint), row.midpoint.detach().clone())
                for path, row in bank.midpoints.items()
            }
            trainer.state.finish_phase()
            trainer._transition()
            self.assertIs(trainer.drafter.qat_affine_bank, bank)
            for path, row in bank.midpoints.items():
                self.assertEqual(id(row.midpoint), before[path][0])
                torch.testing.assert_close(row.midpoint, before[path][1], rtol=0, atol=0)
            model = trainer._model_payload()
            self.assertEqual(set(model["affine_bank"]["state"]), set(trainer.linears))
            self.assertTrue(
                all(
                    not key.startswith("affine_binary.")
                    for values in model["linears"].values()
                    for key in values
                )
            )
            model["affine_bank"]["state"]["fc"].fill_(0.2)
            model["affine_bank"]["state_sha256"]["fc"] = _tensor_sha256(
                model["affine_bank"]["state"]["fc"]
            )
            with self.assertRaisesRegex(ValueError, "exceeds explicit bound"):
                trainer._load_model(model)
            for path, row in bank.midpoints.items():
                torch.testing.assert_close(row.midpoint, before[path][1], rtol=0, atol=0)

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
