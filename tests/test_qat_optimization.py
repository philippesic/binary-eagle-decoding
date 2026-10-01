"""Synthetic CPU optimizer/control checks; no accelerator discovery."""

import copy
import io
import unittest
from dataclasses import replace

import torch

from w1a1_eagle.qat_optimization import (
    BinaryOptimizationConfig, SignFlipDiagnostics, initialize_latents_,
    load_optimizer_checkpoint, make_binary_optimizer, optimizer_checkpoint,
    project_binary_parameters_, transform_binary_gradients_, validate_optimizer_ownership,
)
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract


def linears():
    return {"z": RowBinaryLinear(torch.tensor([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]),
                                  torch.tensor([0.2, 0.0]), W1AxContract(1)),
            "a": RowBinaryLinear(torch.tensor([[0.5, -0.5, 0.5]]),
                                  torch.tensor([0.4]), W1AxContract(1))}


def step(modules, optimizer, config):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(module(torch.tensor([[1.0, -2.0, 0.5]])).square().sum() for module in modules.values())
    loss.backward()
    transform_binary_gradients_(modules, config)
    optimizer.step()
    project_binary_parameters_(modules)


class OptimizationTests(unittest.TestCase):
    def test_finite_config_and_rejected_rules(self):
        baseline = BinaryOptimizationConfig()
        for name in ("latent_magnitude", "sign_lr", "scale_lr", "eps", "scale_floor", "max_grad_norm"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(baseline, **{name: value})
        for values in ({"optimizer": "bop"}, {"latent_magnitude": 1.01}, {"betas": (0.9, 1.0)},
                       {"momentum": 1}, {"clip_policy": "none"}, {"sign_gradient_rule": "raw"}):
            with self.assertRaises(ValueError):
                replace(baseline, **values)

    def test_lower_inertia_preserves_initial_hard_forward(self):
        modules = linears()
        with torch.no_grad():
            modules["a"].latent_sign[0, 0] = -0.0
        old = [m(torch.tensor([[1.0, -2.0, 0.5]])).detach().clone() for m in modules.values()]
        initialize_latents_(modules, replace(BinaryOptimizationConfig(), latent_magnitude=0.1))
        for module, expected in zip(modules.values(), old):
            torch.testing.assert_close(module(torch.tensor([[1.0, -2.0, 0.5]])), expected, rtol=0, atol=0)
            torch.testing.assert_close(module.latent_sign.abs(), torch.full_like(module.latent_sign, 0.1))
        self.assertGreater(modules["a"].latent_sign[0, 0].item(), 0)

    def test_baseline_matches_manual_adamw_joint_clip(self):
        first, second = linears(), linears()
        config = BinaryOptimizationConfig()
        optimizer = make_binary_optimizer(first, config)
        manual = torch.optim.AdamW([{"params": [second[k].latent_sign for k in sorted(second)], "lr": config.sign_lr},
                                   {"params": [second[k].scale_offset for k in sorted(second)], "lr": config.scale_lr}],
                                  weight_decay=0, foreach=False)
        for _ in range(3):
            step(first, optimizer, config)
            manual.zero_grad(set_to_none=True)
            sum(m(torch.tensor([[1.0, -2.0, 0.5]])).square().sum() for m in second.values()).backward()
            torch.nn.utils.clip_grad_norm_([second[k].latent_sign for k in sorted(second)] +
                                          [second[k].scale_offset for k in sorted(second)], 1)
            manual.step()
            project_binary_parameters_(second)
        for name in first:
            for p, q in zip(first[name].parameters(), second[name].parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_gradient_rules_zero_scale_and_extras(self):
        modules = linears()
        extra = torch.nn.Parameter(torch.tensor([3.0]))
        for module in modules.values():
            module.latent_sign.grad = torch.ones_like(module.latent_sign)
            module.scale_offset.grad = torch.ones_like(module.scale_offset)
        extra.grad = torch.tensor([4.0])
        config = replace(BinaryOptimizationConfig(), sign_gradient_rule="weight_unit",
                         scale_gradient_rule="fan_in_rsqrt", max_grad_norm=100)
        result = transform_binary_gradients_(modules, config, additional_parameters=(extra,))
        torch.testing.assert_close(modules["z"].latent_sign.grad[0], torch.full((3,), 5.0))
        torch.testing.assert_close(modules["z"].latent_sign.grad[1], torch.zeros(3))
        torch.testing.assert_close(modules["z"].scale_offset.grad, torch.full((2,), 1 / (3 ** .5)))
        self.assertEqual(result["additional_gradient_norm"], 4)
        with self.assertRaises(ValueError):
            make_binary_optimizer(modules, config, extra_parameters=(extra,))
        optimizer = make_binary_optimizer(modules, config, extra_parameters=(extra,), extra_lr=.01)
        validate_optimizer_ownership(modules, optimizer, additional_parameters=(extra,))
        with self.assertRaises(ValueError):
            validate_optimizer_ownership(modules, optimizer)

    def test_per_family_clips_and_rejects_nonfinite_before_mutation(self):
        modules = linears()
        config = replace(BinaryOptimizationConfig(), clip_policy="per_family", sign_max_grad_norm=.1, scale_max_grad_norm=.2)
        extra = torch.nn.Parameter(torch.tensor([1.0]))
        extra.grad = torch.tensor([2.0])
        for module in modules.values():
            module.latent_sign.grad = torch.ones_like(module.latent_sign)
            module.scale_offset.grad = torch.ones_like(module.scale_offset)
        transform_binary_gradients_(modules, config, additional_parameters=(extra,))
        self.assertLessEqual(torch.cat([m.latent_sign.grad.flatten() for m in modules.values()]).norm(), .100001)
        self.assertLessEqual(torch.cat([m.scale_offset.grad for m in modules.values()]).norm(), .200001)
        self.assertLessEqual(extra.grad.norm(), 1.000001)
        before = modules["a"].latent_sign.grad.clone()
        modules["z"].scale_offset.grad[0] = float("nan")
        with self.assertRaises(ValueError):
            transform_binary_gradients_(modules, config)
        torch.testing.assert_close(modules["a"].latent_sign.grad, before)

    def test_sgd_projection_and_ownership(self):
        modules = linears()
        config = replace(BinaryOptimizationConfig(), optimizer="sgd", momentum=.9)
        optimizer = make_binary_optimizer(modules, config)
        step(modules, optimizer, config)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        with torch.no_grad():
            modules["z"].latent_sign.fill_(2)
            modules["z"].scale_offset.fill_(-2)
        project_binary_parameters_(modules)
        self.assertTrue(bool((modules["z"].latent_sign == 1).all()))
        self.assertTrue(bool((modules["z"].effective_scales() == 0).all()))
        optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
        with self.assertRaises(ValueError):
            validate_optimizer_ownership(modules, optimizer)

    def test_optimizer_serialized_exact_resume_and_contract_guards(self):
        modules = linears()
        config = BinaryOptimizationConfig()
        optimizer = make_binary_optimizer(modules, config)
        step(modules, optimizer, config)
        checkpoint = optimizer_checkpoint(modules, optimizer, config, contract={"data": "fixed"})
        resumed = copy.deepcopy(modules)
        resumed_optimizer = make_binary_optimizer(resumed, config)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        load_optimizer_checkpoint(resumed, resumed_optimizer, config, restored, contract={"data": "fixed"})
        step(modules, optimizer, config)
        step(resumed, resumed_optimizer, config)
        for name in modules:
            for p, q in zip(modules[name].parameters(), resumed[name].parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_optimizer_checkpoint(resumed, resumed_optimizer, config, restored, contract={"data": "changed"})
        corrupt = copy.deepcopy(restored)
        corrupt["optimizer"]["param_groups"][0]["lr"] = .1
        with self.assertRaises(ValueError):
            load_optimizer_checkpoint(resumed, resumed_optimizer, config, corrupt, contract={"data": "fixed"})
        corrupt = copy.deepcopy(restored)
        corrupt["optimizer"]["state"][0]["exp_avg"] = torch.ones(7)
        with self.assertRaises(ValueError):
            load_optimizer_checkpoint(resumed, resumed_optimizer, config, corrupt, contract={"data": "fixed"})

    def test_packed_flip_history_and_exact_resume(self):
        modules = linears()
        tracker = SignFlipDiagnostics(modules, contract={"data": "fixed"})
        with torch.no_grad():
            modules["a"].latent_sign[0, 0].mul_(-1)
            modules["z"].latent_sign[0, 0].mul_(-1)
        one = tracker.observe(modules, step=1)
        self.assertEqual(one["sign_flips"], 2)
        self.assertEqual(one["unique_flipped_signs"], 2)
        state = tracker.state_dict()
        resumed = SignFlipDiagnostics(modules, contract={"data": "fixed"})
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        resumed.load_state_dict(torch.load(buffer, weights_only=False), modules)
        with torch.no_grad():
            modules["a"].latent_sign[0, 0].mul_(-1)
        two = tracker.observe(modules, step=2)
        self.assertEqual(two, resumed.observe(modules, step=2))
        self.assertEqual(two["flip_backs"], 1)
        self.assertEqual(two["net_sign_disagreement"], 1)
        self.assertEqual(two["sustained_disagreement"], 1)
        self.assertEqual(two["cumulative_sign_flips"], 3)
        self.assertEqual(two["unique_flipped_signs"], 2)
        with self.assertRaises(ValueError):
            resumed.load_state_dict(state, modules)
        with self.assertRaises(ValueError):
            tracker.observe(modules, step=2)
        # A sparse observer can resume between samples without pretending that
        # its previous sampled signs are the current model signs.
        sampled = SignFlipDiagnostics(modules, contract={"data": "fixed"})
        sampled.load_state_dict(state, modules, resumed_step=2)
        self.assertEqual(sampled.observe(modules, step=3)["sign_flips"], 1)


if __name__ == "__main__":
    unittest.main()
