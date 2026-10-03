"""Selected A8 production round/update/recipe gates on tiny CPU fixtures only."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_learned_head_batching import chain
from test_qat_cache_head import drafter
from torch import nn

from w1a1_eagle.affine_binary import install_affine_binary
from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.qat_optimization import initialize_latents_
from w1a1_eagle.qat_recipe_audit import (
    QATRecipeTelemetry,
    QATUpdateProbe,
    audit_a8_recipe,
    comparison_joint_recipe,
    comparison_training,
    load_comparison_manifest,
    resolve_comparison_config,
)
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_provider import forward_torch_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_optimizer,
    joint_parameter_families,
    joint_train_step,
    shared_round_hard_signs,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/qat_a8_comparison.json"


def fixture(arm="candidate"):
    manifest = load_comparison_manifest(MANIFEST)
    expected = comparison_joint_recipe(manifest, arm)
    config = JointQATConfig(W1AxContract(8), **expected)
    torch.manual_seed(19)
    model = drafter(8)
    linears = {name: model.get_submodule(name) for name in CANDIDATE_D_BASE_TO_PATH.values()}
    for module in linears.values():
        module.a1_computation = config.a1_computation
    initialize_latents_(linears, config.binary_optimization)
    if config.activation_quantization == "learned":
        bank = LearnedActivationBank(8, {p: m.in_features for p, m in linears.items()})
        bank.attach(linears)
        model.qat_activation_bank = bank
    if config.affine_weights is not None:
        model.qat_affine_bank = install_affine_binary(
            linears, target=nn.Linear(1, 1), config=config.affine_weights
        )
    return config, expected, linears, joint_optimizer(linears, config), NativeStepAdapter(model)


def round_forward(adapter, config, *, training=True):
    batch, audit = chain(3)
    evidence = {}
    calls = []
    original = adapter.build_context_cache

    def actual_cache(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    adapter.build_context_cache = actual_cache
    try:
        with torch.set_grad_enabled(training), shared_round_hard_signs(adapter.linears):
            logits = forward_torch_round(
                batch,
                adapter,
                7,
                optimize_cache=True,
                optimize_head=True,
                context_chunk_size=config.context_chunk_size,
                execution_metadata=evidence,
            )
    finally:
        adapter.build_context_cache = original
    assert calls == [3], calls
    evidence.update(context_cache_calls=len(calls), context_chunk_size=config.context_chunk_size)
    return logits, audit, evidence


class A8RecipeTests(unittest.TestCase):
    def test_materialization_preserves_provider_binding_and_embeds_independent_expectation(self):
        manifest = load_comparison_manifest(MANIFEST)
        base = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())
        base["provider"]["manifest"] = "authenticated-provider.json"
        original = copy.deepcopy(base)
        for arm in ("reference", "candidate"):
            result = resolve_comparison_config(base, manifest, arm)
            self.assertEqual(result["training"]["activation_bits"], [8])
            self.assertEqual(result["training"]["max_seconds"], 7200)
            self.assertEqual(result["provider"], original["provider"])
            self.assertEqual(
                result["comparison"]["expected_recipe"], comparison_joint_recipe(manifest, arm)
            )
        self.assertEqual(base, original)

    def test_rejects_silent_budget_lane_math_or_feature_change(self):
        cases = [
            ("shared_training", "activation_bits", [8, 1]),
            ("shared_training", "max_seconds", 7300),
            ("shared_training", "objective", "compact_probability"),
            ("shared_training", "optimize_cache", False),
            ("shared_training", "optimize_head", False),
            ("shared_training", "fusion_correction", {"enabled": True}),
            ("shared_training", "depth_loss_decay", 0.9),
            ("shared_training", "curriculum", {}),
            ("shared_training", "refresh", {}),
            ("budget", "sealed_final_allowed", True),
            ("budget", "a1_held_out", False),
            ("budget", "training_seconds_total", 15000),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                manifest = load_comparison_manifest(MANIFEST)
                manifest[section][key] = value
                with self.assertRaises(ValueError):
                    comparison_training(manifest, "candidate")
        for key, value in (
            ("activation_quantization", "fixed"),
            ("affine_weights", None),
            ("binary_optimization", {"optimizer": "sgd"}),
        ):
            manifest = load_comparison_manifest(MANIFEST)
            manifest["arms"]["candidate"][key] = value
            with self.assertRaises(ValueError):
                comparison_training(manifest, "candidate")

    def test_initial_deployed_signs_scales_and_logits_match(self):
        c, ce, cl, co, ca = fixture("candidate")
        r, re, rl, ro, ra = fixture("reference")
        for name in cl:
            self.assertTrue(torch.equal(cl[name].latent_sign < 0, rl[name].latent_sign < 0))
            self.assertTrue(torch.equal(cl[name].effective_scales(), rl[name].effective_scales()))
        audit_a8_recipe(cl, co, c, ce, fresh=True)
        audit_a8_recipe(rl, ro, r, re, fresh=True)
        # Same serial invocation boundaries avoid an irrelevant batching rounding difference.
        batch, _ = chain(3)
        a = forward_torch_round(batch, ca, 7)
        b = forward_torch_round(batch, ra, 7)
        # Zero affine arithmetic takes a slightly different floating-point path.
        # Preserve exact deployed signs/scales and require unchanged decisions.
        torch.testing.assert_close(a, b, rtol=1e-6, atol=2e-7)
        self.assertTrue(torch.equal(a.argmax(-1), b.argmax(-1)))

    def test_integrated_real_gradients_movement_and_effective_execution_both_arms(self):
        for arm in ("reference", "candidate"):
            with self.subTest(arm=arm):
                config, expected, linears, optimizer, adapter = fixture(arm)
                probe = QATUpdateProbe(linears, optimizer, require_all=True)
                logits, audit, evidence = round_forward(adapter, config)
                checked = audit_a8_recipe(
                    linears, optimizer, config, expected, fresh=True, execution_evidence=evidence
                )
                self.assertEqual(checked["execution"]["status"], "observed")
                self.assertEqual(
                    checked["execution"]["head"]["path"],
                    "serial" if arm == "candidate" else "batched",
                )
                joint_train_step(linears, logits, audit, optimizer, config)
                report = probe.last_report
                counts = {k: v["parameter_tensors"] for k, v in report["families"].items()}
                self.assertEqual(
                    counts,
                    {"sign": 9, "scale": 9}
                    if arm == "reference"
                    else {"sign": 9, "scale": 9, "activation": 6, "midpoint": 9},
                )
                for family in report["families"].values():
                    self.assertEqual(
                        family["parameter_tensors"], family["nonzero_gradient_tensors"]
                    )
                    self.assertEqual(family["parameter_tensors"], family["sampled_moved_tensors"])
                    self.assertGreater(family["sampled_parameter_displacement_l1"], 0)
                audit_a8_recipe(linears, optimizer, config, expected, execution_evidence=evidence)
                probe.close()

    def test_candidate_inference_observes_batched_head(self):
        config, expected, linears, optimizer, adapter = fixture()
        _, _, evidence = round_forward(adapter, config, training=False)
        result = audit_a8_recipe(
            linears, optimizer, config, expected, training=False, execution_evidence=evidence
        )
        self.assertTrue(result["execution"]["head"]["effective_batched"])
        with self.assertRaisesRegex(ValueError, "head execution"):
            audit_a8_recipe(linears, optimizer, config, expected, execution_evidence=evidence)

    def test_rejects_unobserved_cache_or_undeclared_head_fallback(self):
        config, expected, linears, optimizer, adapter = fixture()
        _, _, evidence = round_forward(adapter, config)
        for key, value in (
            ("context_cache_calls", 0),
            ("path", "batched"),
            ("reason", "adapter_unavailable"),
            ("requested", False),
            ("context_chunk_size", 1),
        ):
            invalid = {**evidence, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_a8_recipe(linears, optimizer, config, expected, execution_evidence=invalid)

    def test_rejects_config_attachment_optimizer_or_initialization_mismatch(self):
        for kind in ("config", "attachment", "optimizer", "rate", "fresh", "frozen", "group_rule"):
            config, expected, linears, optimizer, _ = fixture()
            if kind == "config":
                config = replace(config, optimize_head=False)
            elif kind == "attachment":
                linears["fc"].affine_binary = None
            elif kind == "optimizer":
                optimizer.param_groups[-1]["params"].pop()
            elif kind == "rate":
                optimizer.param_groups[0]["lr"] *= 2
            elif kind == "fresh":
                linears["fc"].scale_offset.data[0] = 0.1
            elif kind == "frozen":
                linears["fc"].latent_sign.requires_grad_(False)
            else:
                optimizer.param_groups[0]["weight_decay"] = 0.1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_a8_recipe(linears, optimizer, config, expected, fresh=True)

    def test_warmup_audit_accepts_only_exact_effective_factor(self):
        config, expected, linears, optimizer, _ = fixture()
        for group in optimizer.param_groups:
            group["lr"] *= 0.01
        audit_a8_recipe(linears, optimizer, config, expected, lr_factor=0.01)
        with self.assertRaisesRegex(ValueError, "rules/LR"):
            audit_a8_recipe(linears, optimizer, config, expected)

    def test_probe_pre_step_rejects_missing_or_zero_gradients_without_update(self):
        for value in (None, 0.0, float("nan")):
            _, _, linears, optimizer, _ = fixture()
            before = linears["fc"].latent_sign.detach().clone()
            for parameters in joint_parameter_families(linears).values():
                for p in parameters:
                    p.grad = None if value is None else torch.full_like(p, value)
            probe = QATUpdateProbe(linears, optimizer, require_all=True)
            with self.subTest(value=value), self.assertRaises(ValueError):
                optimizer.step()
            self.assertTrue(torch.equal(before, linears["fc"].latent_sign))
            probe.close()

    def test_exact_positive_step_optimizer_and_telemetry_resume(self):
        config, expected, linears, optimizer, adapter = fixture()
        telemetry = QATRecipeTelemetry(linears, contract=expected)
        probe = QATUpdateProbe(linears, optimizer, require_all=True)
        logits, audit, _ = round_forward(adapter, config)
        joint_train_step(linears, logits, audit, optimizer, config)
        telemetry.observe(linears, step=1)
        state = copy.deepcopy(
            {
                "linears": {name: m.state_dict() for name, m in linears.items()},
                "optimizer": optimizer.state_dict(),
                "probe": probe.state_dict(),
                "telemetry": telemetry.state_dict(),
            }
        )
        rc, re, rl, ro, ra = fixture()
        for name in rl:
            rl[name].load_state_dict(state["linears"][name])
        ro.load_state_dict(state["optimizer"])
        restored = QATRecipeTelemetry(rl, contract=re)
        restored.load_state_dict(state["telemetry"], rl, resumed_step=1)
        audit_a8_recipe(rl, ro, rc, re)
        with self.assertRaisesRegex(ValueError, "fresh latent"):
            audit_a8_recipe(rl, ro, rc, re, fresh=True)
        rp = QATUpdateProbe(rl, ro, require_all=True)
        rp.load_state_dict(state["probe"])
        for ls, opt, cfg, ad in ((linears, optimizer, config, adapter), (rl, ro, rc, ra)):
            logits, audit, _ = round_forward(ad, cfg)
            joint_train_step(ls, logits, audit, opt, cfg)
        self.assertEqual(telemetry.observe(linears, step=2), restored.observe(rl, step=2))
        self.assertEqual(probe.last_report, rp.last_report)
        self.assertEqual(probe.admission_report(), rp.admission_report())
        for name in rl:
            for key, value in rl[name].state_dict().items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        value, linears[name].state_dict()[key], rtol=0, atol=0
                    )
        probe.close()
        rp.close()

    def test_telemetry_rejects_nonfinite_or_changed_saved_layout(self):
        _, expected, linears, _, _ = fixture()
        telemetry = QATRecipeTelemetry(linears, contract=expected)
        for kind in ("nan", "shape", "family", "threshold"):
            state = telemetry.state_dict()
            if kind == "nan":
                state["initial"]["activation"][0].fill_(float("nan"))
            elif kind == "shape":
                state["initial"]["scale"][0] = torch.zeros(1)
            elif kind == "family":
                del state["initial"]["midpoint"]
            else:
                state["near_zero"] = 0.2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                telemetry.load_state_dict(state, linears, resumed_step=0)

    def test_family_admission_accumulates_through_float32_warmup_rounding(self):
        _, _, linears, optimizer, _ = fixture()
        probe = QATUpdateProbe(linears, optimizer)
        for group in optimizer.param_groups:
            group["lr"] = 1e-12
            for p in group["params"]:
                p.grad = torch.full_like(p, 0.2)
        optimizer.step()
        pending = probe.admission_report()
        self.assertFalse(pending["passed"])
        self.assertEqual(pending["families"]["activation"]["nonzero_gradient_tensors_ever"], 6)
        self.assertEqual(pending["families"]["activation"]["sampled_moved_tensors_ever"], 0)
        with self.assertRaisesRegex(ValueError, "lacks observed"):
            probe.admission_report(require_passed=True)
        state = probe.state_dict()
        probe.close()
        restored = QATUpdateProbe(linears, optimizer)
        restored.load_state_dict(state)
        self.assertEqual(restored.admission_report(), pending)
        for group in optimizer.param_groups:
            group["lr"] = 1e-5
        optimizer.step()
        passed = restored.admission_report(require_passed=True)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["observed_updates"], 2)
        self.assertEqual(passed["families"]["activation"]["sampled_moved_tensors_ever"], 6)
        self.assertEqual(len(restored.last_report["families"]["activation"]["per_tensor"]), 6)
        restored.close()

    def test_probe_state_rejects_corrupt_counter_layout_or_coverage(self):
        _, _, linears, optimizer, _ = fixture()
        probe = QATUpdateProbe(linears, optimizer)
        for kind in ("counter", "layout", "coverage", "bool", "contradiction"):
            state = probe.state_dict()
            if kind == "counter":
                state["observed_updates"] = -1
            elif kind == "layout":
                state["layout"]["activation"][0] = [1]
            elif kind == "coverage":
                state["admission"]["activation"]["moved"].pop()
            elif kind == "bool":
                state["admission"]["activation"]["moved"][0] = 1
            else:
                state["admission"]["activation"]["moved"][0] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                probe.load_state_dict(state)
        probe.close()

    def test_probe_can_disable_and_remove_hooks(self):
        config, _, linears, optimizer, adapter = fixture()
        probe = QATUpdateProbe(linears, optimizer)
        probe.enabled = False
        logits, audit, _ = round_forward(adapter, config)
        joint_train_step(linears, logits, audit, optimizer, config)
        self.assertIsNone(probe.last_report)
        probe.close()
        self.assertEqual(len(optimizer._optimizer_step_pre_hooks), 0)
        self.assertEqual(len(optimizer._optimizer_step_post_hooks), 0)


if __name__ == "__main__":
    unittest.main()
