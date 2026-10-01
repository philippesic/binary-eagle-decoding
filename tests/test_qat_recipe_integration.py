"""End-to-end tiny optimizer/checkpoint tests for optional QAT recipes."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_continuous_qat import config, make

from w1a1_eagle.continuous_qat import ContinuousConfig
from w1a1_eagle.continuous_runtime import MATH_FILES
from w1a1_eagle.learned_activation import LearnedActivationBank, native_order_linear
from w1a1_eagle.qat_state import recipe_state, validate_resume_state
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    joint_parameter_families,
)


def optimized(**kwargs):
    return config(
        activation_quantization="learned",
        a1_computation="single_forward",
        optimize_cache=True,
        optimize_head=True,
        fusion_correction={"enabled": True, "rank": 1, "output_bias": True},
        binary_optimization={"latent_magnitude": 0.1, "sign_lr": 0.01, "scale_lr": 0.001},
        **kwargs,
    )


class RecipeIntegrationTests(unittest.TestCase):
    def test_all_parameter_families_are_independent_and_shared_once(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            first, second = trainer.lanes
            for lane in trainer.lanes:
                families = joint_parameter_families(lane.linears)
                self.assertEqual([len(x) for x in families.values()], [9, 9, 6, 3, 0])
                owned = [p for g in lane.optimizer.param_groups for p in g["params"]]
                self.assertEqual(len(owned), len({id(p) for p in owned}))
                LearnedActivationBank.from_attached(lane.linears).validate_attachment(lane.linears)
                self.assertTrue(all(p.requires_grad for p in owned))
                self.assertFalse(lane.drafter.embed_tokens.weight.requires_grad)
            self.assertIs(first.drafter.embed_tokens.weight, second.drafter.embed_tokens.weight)
            self.assertIsNot(
                first.linears["fc"].fusion_correction.u, second.linears["fc"].fusion_correction.u
            )
            self.assertEqual(first.linears["fc"].activation_quantizer.bits, 8)
            self.assertEqual(second.linears["fc"].activation_quantizer.bits, 1)

    def test_learned_single_forward_matches_explicit_surrogate_gradients(self):
        for bits in (1, 4, 8):
            from w1a1_eagle.learned_activation import LearnedActivationQuantizer

            baseline = RowBinaryLinear(
                torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.2]]),
                torch.tensor([0.2, 0.7]),
                W1AxContract(bits),
            )
            baseline.activation_quantizer = LearnedActivationQuantizer(bits, "fc", 3)
            baseline.activation_quantizer.parameter.data.fill_(0.2 if bits == 1 else 0.7)
            candidate = copy.deepcopy(baseline)
            candidate.a1_computation = "single_forward"
            inputs = [
                torch.tensor([[0.1, -0.4, 0.3], [0.7, 0.2, -0.3]], requires_grad=True)
                for _ in range(2)
            ]
            outputs = [m(x) for m, x in zip((baseline, candidate), inputs)]
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            with torch.no_grad():
                q = baseline.activation_quantizer(inputs[0])
                expected = native_order_linear(
                    q, torch.where(baseline.latent_sign < 0, -1.0, 1.0), baseline.effective_scales()
                )
            torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
            for output in outputs:
                (output * torch.tensor([0.4, -0.7])).sum().backward()
            torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=1e-4, atol=1e-6)
            for a, b in zip(baseline.parameters(), candidate.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)

    def test_combined_recipe_exact_resume_and_schema4_export_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            direct = make(root / "direct", optimized(max_steps=4))
            direct.run(require_smoke=False)
            partial = make(root / "resume", optimized(max_steps=2))
            partial.run(require_smoke=False)
            resumed = make(root / "resume", optimized(max_steps=4))
            resumed.resume()
            resumed.run(require_smoke=False)
            self.assertEqual(direct.step, resumed.step)
            for a, b in zip(direct.lanes, resumed.lanes):
                for name in a.linears:
                    for key, value in a.linears[name].state_dict().items():
                        other = b.linears[name].state_dict()[key]
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, other, rtol=0, atol=0)
                        else:
                            self.assertEqual(value, other)
                for ga, gb in zip(a.optimizer.param_groups, b.optimizer.param_groups):
                    for pa, pb in zip(ga["params"], gb["params"]):
                        for key, value in a.optimizer.state[pa].items():
                            torch.testing.assert_close(
                                value, b.optimizer.state[pb][key], rtol=0, atol=0
                            )
                manifest = json.loads(
                    (Path(resumed.checkpoint["path"]).parent / a.name / "joint.json").read_text()
                )
                self.assertEqual(manifest["schema_version"], 4)
                self.assertEqual(
                    manifest["activation_quantizers"]["boundaries"]["fc"]["bits"],
                    a.config.contract.activation_bits,
                )
                self.assertEqual(manifest["fusion_correction"]["rank"], 1)

    def test_resume_rejects_inconsistent_tied_alias_before_mutating(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            lane = trainer.lanes[0]
            before = recipe_state(lane.linears)
            states = {name: copy.deepcopy(m.state_dict()) for name, m in lane.linears.items()}
            states["midlayer.self_attn.k_proj"]["activation_quantizer.clip_ratio"].fill_(0.5)
            with self.assertRaisesRegex(ValueError, "inconsistent shared"):
                validate_resume_state(lane.linears, states, before)
            torch.testing.assert_close(
                lane.linears["midlayer.self_attn.q_proj"].activation_quantizer.parameter,
                torch.tensor(1.0),
                rtol=0,
                atol=0,
            )

    def test_changed_recipe_or_frozen_operand_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            lane = trainer.lanes[0]
            states = {name: copy.deepcopy(m.state_dict()) for name, m in lane.linears.items()}
            states["fc"]["initial_scale"][0] += 1
            with self.assertRaisesRegex(ValueError, "changed frozen"):
                validate_resume_state(lane.linears, states, recipe_state(lane.linears))
            trainer.run(require_smoke=False)
            changed = make(Path(folder), replace(optimized(max_steps=2), depth_loss_decay=0.8))
            with self.assertRaisesRegex(ValueError, "immutable"):
                changed.resume()

    def test_cpu_recipe_smoke_never_steps_or_discovers_cuda(self):
        for optimizer in ("adamw", "sgd"):
            with (
                tempfile.TemporaryDirectory() as folder,
                patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("accelerator query")
                ),
            ):
                cfg = replace(optimized(max_steps=1), binary_optimization={"optimizer": optimizer})
                trainer = make(Path(folder), cfg)
                # Choose nonsingular toy binary dots: the width-four default
                # has legal exact zero gate/up scale gradients and fails smoke.
                generator = torch.Generator().manual_seed(0)
                for module in trainer.lanes[1].linears.values():
                    module.latent_sign.data.copy_(
                        torch.randn(module.latent_sign.shape, generator=generator).sign() * 0.1
                    )
                before = [
                    {k: p.detach().clone() for k, p in lane.drafter.named_parameters()}
                    for lane in trainer.lanes
                ]
                with patch.object(
                    torch.optim.Optimizer, "step", side_effect=AssertionError("update")
                ):
                    trainer.smoke(next(iter(trainer.provider.rounds())))
                self.assertEqual(trainer.step, 0)
                for lane, saved in zip(trainer.lanes, before):
                    for name, p in lane.drafter.named_parameters():
                        torch.testing.assert_close(p, saved[name], rtol=0, atol=0)
                    self.assertTrue(
                        all(
                            float(state.get("step", 0)) == 0
                            for state in lane.optimizer.state.values()
                        )
                    )

    def test_affine_all_families_and_exact_resume_preserve_midpoints(self):
        for coverage in ("fusion", "all"):
            with tempfile.TemporaryDirectory() as folder:
                cfg = replace(
                    optimized(max_steps=3),
                    affine_weights={
                        "enabled": True,
                        "coverage": coverage,
                        "midpoint_lr": 0.01,
                        "mild_l2": 1e-4,
                    },
                )
                root = Path(folder)
                direct = make(root / "direct", cfg)
                direct.run(require_smoke=False)
                partial = make(root / "resume", replace(cfg, max_steps=1))
                partial.run(require_smoke=False)
                resumed = make(root / "resume", cfg)
                resumed.resume()
                resumed.run(require_smoke=False)
                for a, b in zip(direct.lanes, resumed.lanes):
                    families = joint_parameter_families(b.linears)
                    self.assertEqual(len(families["midpoint"]), 1 if coverage == "fusion" else 9)
                    self.assertTrue(all(p.requires_grad for p in families["midpoint"]))
                    for path, ma in a.linears.items():
                        mb = b.linears[path]
                        for key, value in ma.state_dict().items():
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(
                                    value, mb.state_dict()[key], rtol=0, atol=0
                                )
                    for ga, gb in zip(a.optimizer.param_groups, b.optimizer.param_groups):
                        for pa, pb in zip(ga["params"], gb["params"]):
                            for key, value in a.optimizer.state[pa].items():
                                torch.testing.assert_close(
                                    value, b.optimizer.state[pb][key], rtol=0, atol=0
                                )
                    manifest = json.loads(
                        (
                            Path(resumed.checkpoint["path"]).parent / b.name / "joint.json"
                        ).read_text()
                    )
                    self.assertEqual(manifest["schema_version"], 5)
                    self.assertEqual(manifest["affine_weights"]["coverage"], coverage)
                    self.assertGreater(
                        sum(float(p.detach().abs().sum()) for p in families["midpoint"]), 0
                    )

    def test_affine_zero_initialization_matches_native_symmetric_a1(self):
        from w1a1_eagle.affine_binary import AffineBinaryConfig, AffineBinaryMidpoint

        baseline = RowBinaryLinear(
            torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.2]]),
            torch.tensor([0.0, 0.7]),
            W1AxContract(1),
        )
        affine = copy.deepcopy(baseline)
        affine.affine_binary = AffineBinaryMidpoint(2, AffineBinaryConfig(enabled=True))
        for x in (
            torch.tensor([0.0, -0.0, 0.0]),
            torch.tensor([0.1, -0.4, 0.3]),
            torch.tensor([1e-40, -1e-40, 0.0]),
        ):
            with torch.no_grad():
                torch.testing.assert_close(baseline(x), affine(x), rtol=0, atol=0)
        x = torch.tensor([0.1, -0.4, 0.3], requires_grad=True)
        affine.affine_binary.midpoint.data.fill_(0.2)
        affine(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(affine.affine_binary.midpoint.grad.abs().sum()), 0)
        self.assertGreater(float(affine.scale_offset.grad[0].abs()), 0)

    def test_deployment_fingerprint_tracks_effective_values_not_latent_magnitudes(self):
        from w1a1_eagle.qat_state import deployment_state_sha256

        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            linears = trainer.lanes[0].linears
            current = deployment_state_sha256(linears)
            changed = copy.deepcopy(linears)
            for module in changed.values():
                module.latent_sign.data.mul_(2)
            self.assertEqual(current, deployment_state_sha256(changed))
            changed["fc"].scale_offset.data[0] += 0.01
            self.assertNotEqual(current, deployment_state_sha256(changed))

    def test_shared_sum_does_not_reuse_no_grad_result_in_attached_work(self):
        from w1a1_eagle.affine_binary import affine_input_sum, shared_affine_input_sums

        for keyed in (False, True):
            values = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
            codes = values.detach().clone()
            beta = torch.ones((1, 1))
            kwargs = {"cache_key": (values, 8, None)} if keyed else {}
            with shared_affine_input_sums():
                with torch.no_grad():
                    detached = affine_input_sum(values, codes=codes, beta=beta, **kwargs)
                attached = affine_input_sum(values, codes=codes, beta=beta, **kwargs)
                self.assertFalse(detached.surrogate.requires_grad)
                self.assertTrue(attached.surrogate.requires_grad)
            attached.surrogate.sum().backward()
            torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)

    def test_fusion_bias_addition_matches_declared_native_order(self):
        from w1a1_eagle.fusion_correction import FusionCorrection, FusionCorrectionConfig

        correction = FusionCorrection(
            1, 1, FusionCorrectionConfig(enabled=True, output_bias=True, bias_bound=2)
        )
        correction.v.data.fill_(4096)
        correction.u.data.fill_(-8192)
        correction.output_bias.data.fill_(1)
        result = correction.add_to(torch.ones(1), torch.tensor([2.0**25]))
        torch.testing.assert_close(result, torch.ones(1), rtol=0, atol=0)

    def test_fixed_affine_safe_tiny_multibit_codes_match_native_reference(self):
        from w1a1_eagle.learned_activation import learned_activation_reference
        from w1a1_eagle.recurrent_qat import hard_activation_with_codes

        for bits in (4, 8):
            x = torch.tensor([[1e-40, -1e-40, 0.0]], requires_grad=True)
            values, beta, _, codes = hard_activation_with_codes(x, bits)
            expected = learned_activation_reference(x, bits, torch.tensor(1.0))
            self.assertTrue(torch.isfinite(values).all())
            torch.testing.assert_close(codes.to(torch.int8), expected.codes, rtol=0, atol=0)
            torch.testing.assert_close(beta, expected.scale, rtol=0, atol=0)
            values.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    def test_zero_update_boundary_rejects_nonzero_optimizer_moments(self):
        from scripts.train_continuous_w1ax import require_zero_optimizer_progress

        for name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            with tempfile.TemporaryDirectory() as folder:
                trainer = make(Path(folder), config(max_steps=1))
                parameter = trainer.lanes[0].optimizer.param_groups[0]["params"][0]
                trainer.lanes[0].optimizer.state[parameter][name] = torch.ones_like(parameter)
                with self.assertRaisesRegex(ValueError, "nonzero optimizer"):
                    require_zero_optimizer_progress(trainer)

    def test_learned_smoke_rejects_actual_zero_scale_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            with self.assertRaisesRegex(ValueError, "all-nine finite/nonzero"):
                trainer.smoke(next(iter(trainer.provider.rounds())))
            self.assertEqual(trainer.step, 0)

    def test_recipe_sources_are_bound_and_guards_reject_invalid_choices(self):
        self.assertTrue(
            {
                "qat_optimization.py",
                "qat_curriculum.py",
                "learned_activation.py",
                "fusion_correction.py",
                "qat_state.py",
            }.issubset(MATH_FILES)
        )
        for kwargs in (
            {"activation_quantization": "unknown"},
            {"a1_computation": "unknown"},
            {"depth_loss_decay": 0},
            {"context_chunk_size": 0},
        ):
            with self.assertRaises(ValueError):
                ContinuousConfig(**kwargs)
        with self.assertRaises(ValueError):
            JointQATConfig(W1AxContract(16), activation_quantization="learned")
        for kwargs in (
            {"sign_lr": True},
            {"warmup_steps": True},
            {"max_steps": 1.5},
            {"depth_loss_decay": True},
            {"seeds": (True, 2)},
        ):
            with self.assertRaises(ValueError):
                ContinuousConfig(**kwargs)
        with self.assertRaises(ValueError):
            W1AxContract(True)
        with self.assertRaises(ValueError):
            JointQATConfig(W1AxContract(1), activation_lr=True)
        for value in (1, "false"):
            with self.assertRaises(ValueError):
                JointQATConfig(W1AxContract(1), allow_accelerator=value)


if __name__ == "__main__":
    unittest.main()
