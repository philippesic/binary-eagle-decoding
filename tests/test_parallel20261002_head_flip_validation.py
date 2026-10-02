"""Independent dense CPU validation for the parallel head flip oracle."""

from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace

import torch

from research.parallel20261002.head_flip_oracle.reference.oracle import (
    FlipCandidate,
    capture_head,
    evaluate_candidates,
    overshoot_counterexample,
)
from research.parallel20261002.head_flip_oracle.validation.independent_oracle import (
    check_control,
    dense_reference,
)
from w1a1_eagle.affine_binary import AffineBinaryConfig, AffineBinaryMidpoint
from w1a1_eagle.learned_activation import LearnedActivationQuantizer
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

F32_EPS = torch.finfo(torch.float32).eps
LOGITS_FACTOR = 8 * F32_EPS


def fixture(bits: int, activation_rule: str):
    torch.manual_seed(7200 + bits)
    width, vocab, tokens = 131, 5, 5
    weight = torch.randint(0, 2, (vocab, width), dtype=torch.int64).float().mul_(2).sub_(1)
    weight[0, 0] = 0.0  # hard weight sign at zero is positive
    weight[1].fill_(1.0)
    weight[2].copy_(weight[1])  # exact winning-row tie for diagnostic convention
    scales = torch.tensor([0.0, 1.0, 1.0, 0.5, 0.75])
    bias = torch.tensor([-0.1, 0.0, 0.0, -0.2, -0.1])
    module = RowBinaryLinear(weight, scales, W1AxContract(bits), bias=bias).eval()
    if activation_rule == "learned":
        module.activation_quantizer = LearnedActivationQuantizer(bits, "fc", width)
    elif activation_rule == "fixed_affine_v1":
        module.affine_binary = AffineBinaryMidpoint(
            vocab, AffineBinaryConfig(enabled=True, coverage="all")
        )
        with torch.no_grad():
            module.affine_binary.midpoint.copy_(torch.tensor([0.35, -0.07, -0.07, 0.15, -0.2]))
    elif activation_rule != "fixed_symmetric":
        raise AssertionError(activation_rule)

    inputs = torch.zeros(tokens, width, dtype=torch.float32)
    inputs[:, 0] = torch.tensor([-0.0, 0.0, -torch.finfo(torch.float32).tiny / 2, 1.0, -1.0])
    inputs[:, 1] = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5])
    inputs[:, 2] = torch.tensor([1.0, -1.0, 0.125, -0.125, 0.0])
    inputs[:, 128] = torch.tensor([0.25, -0.25, 0.75, -0.75, 1.0])
    inputs[:, 129] = torch.tensor([-0.25, 0.25, -0.5, 0.5, -1.0])
    inputs[:, 130] = torch.tensor([0.0, -0.0, 0.5, -0.5, 0.25])
    inputs[:, 3:128] = torch.linspace(-0.3, 0.3, 125).unsqueeze(0)
    inputs[0].zero_()
    inputs[0, 0] = -0.0  # zero-scale row still preserves the defined positive A1 sign
    # Deliberately permuted absolute target IDs expose row/id confusion.
    d2t = torch.tensor([41, 7, 88, 13, 101], dtype=torch.int64)
    labels = torch.tensor([41, 101, 999, 7, 13], dtype=torch.int64)
    return module, inputs, d2t, labels


class DenseIndependentHeadOracleTests(unittest.TestCase):
    metrics: dict = {
        "configurations": 0,
        "candidates": 0,
        "logit_values_checked": 0,
        "max_logit_abs_error": 0.0,
        "max_logit_gate_ratio": 0.0,
        "finite_ce_values_checked": 0,
        "max_ce_abs_error": 0.0,
        "decisions_above_margin_gate": 0,
        "decisive_top1_matches": 0,
        "explicit_tie_tokens": 0,
        "baseline_drift_by_configuration": [],
    }

    def setUp(self):
        check_control()

    @classmethod
    def tearDownClass(cls):
        print("VALIDATION_METRICS=" + json.dumps(cls.metrics, sort_keys=True))

    def test_fixed_learned_and_affine_rows_match_full_dense_flip_reconstruction(self):
        candidates = [
            FlipCandidate(0, (0, 128, 130)),  # zero-scale row and unaligned tail
            FlipCandidate(3, (1, 2, 129)),
            FlipCandidate(4, (0, 33, 130)),
        ]
        for bits in (1, 4, 8):
            for rule in ("fixed_symmetric", "learned", "fixed_affine_v1"):
                with self.subTest(bits=bits, rule=rule):
                    check_control()
                    module, inputs, d2t, labels = fixture(bits, rule)
                    parameters_before = copy.deepcopy(module.state_dict())
                    snapshot = capture_head(module, inputs, d2t, labels)
                    observed = evaluate_candidates(snapshot, candidates)
                    self.metrics["configurations"] += 1
                    self.metrics["baseline_drift_by_configuration"].append(
                        {"bits": bits, "rule": rule, "max_abs": snapshot.baseline_max_drift}
                    )
                    self.assertEqual(snapshot.activation_bits, bits)
                    self.assertEqual(snapshot.rule, rule)
                    self.assertEqual(float(snapshot.beta[0]), 0.0)
                    self.assertEqual(float(snapshot.alpha[0]), 0.0)
                    if bits == 1:
                        self.assertEqual(int(snapshot.codes[0, 0]), 1)
                    self.assertLessEqual(
                        snapshot.baseline_max_drift,
                        LOGITS_FACTOR * max(1.0, float(snapshot.baseline_logits.abs().max())),
                    )
                    self.assertEqual(int(snapshot.label_rows[2]), -1)
                    self.assertTrue(torch.equal(snapshot.d2t, d2t))
                    for key, before in parameters_before.items():
                        after = module.state_dict()[key]
                        if isinstance(before, torch.Tensor):
                            self.assertTrue(torch.equal(before, after), key)
                        else:
                            self.assertEqual(before, after, key)

                    for candidate, actual in zip(candidates, observed, strict=True):
                        self.metrics["candidates"] += 1
                        expected = dense_reference(snapshot, candidate)
                        torch.testing.assert_close(actual.dots, expected["dots"], rtol=0, atol=0)
                        gate = LOGITS_FACTOR * torch.maximum(
                            torch.ones_like(expected["logits"]), expected["logits"].abs()
                        )
                        logit_error = (actual.logits - expected["logits"]).abs()
                        self.metrics["logit_values_checked"] += logit_error.numel()
                        self.metrics["max_logit_abs_error"] = max(
                            self.metrics["max_logit_abs_error"], float(logit_error.max())
                        )
                        self.metrics["max_logit_gate_ratio"] = max(
                            self.metrics["max_logit_gate_ratio"],
                            float((logit_error / gate).max()),
                        )
                        self.assertTrue(bool((logit_error <= gate).all()))
                        self.assertTrue(
                            bool(
                                (
                                    (actual.ce_delta - expected["ce_delta"]).abs()
                                    <= 2e-10 + 2e-12 * expected["ce_delta"].abs()
                                )[snapshot.label_rows >= 0].all()
                            )
                        )
                        ce_mask = snapshot.label_rows >= 0
                        ce_error = (actual.ce_delta - expected["ce_delta"]).abs()[ce_mask]
                        self.metrics["finite_ce_values_checked"] += ce_error.numel()
                        self.metrics["max_ce_abs_error"] = max(
                            self.metrics["max_ce_abs_error"], float(ce_error.max())
                        )
                        decision_gate = LOGITS_FACTOR * torch.maximum(
                            torch.ones_like(expected["top1_margin"]),
                            expected["full_logits"].abs().amax(dim=1),
                        )
                        decisive = expected["top1_margin"] > decision_gate
                        self.metrics["decisions_above_margin_gate"] += int(decisive.sum())
                        self.metrics["decisive_top1_matches"] += int(
                            (actual.top1_rows[decisive] == expected["top1_rows"][decisive]).sum()
                        )
                        self.metrics["explicit_tie_tokens"] += int((~decisive).sum())
                        self.assertTrue(
                            torch.equal(actual.top1_rows[decisive], expected["top1_rows"][decisive])
                        )
                        ties = ~decisive
                        self.assertTrue(bool(actual.rounding_tie[ties].all()))
                        torch.testing.assert_close(
                            actual.top1_target_ids, expected["top1_target_ids"], rtol=0, atol=0
                        )
                        self.assertTrue(
                            bool(
                                ((actual.label_margin - expected["label_margin"]).abs() <= gate)[
                                    snapshot.label_rows >= 0
                                ].all()
                            )
                        )
                        self.assertTrue(
                            bool(
                                (actual.top1_margin - expected["top1_margin"]).abs().le(gate).all()
                            )
                        )
                        self.assertTrue(math.isnan(float(actual.ce_delta[2])))
                        self.assertTrue(math.isnan(float(actual.label_margin[2])))
                        self.assertEqual(actual.avoided_candidate_head_gemms, 1)

                    # Both equal top rows remain a tie; lowest dense draft row wins.
                    self.assertTrue(bool(observed[0].rounding_tie[0]))
                    self.assertEqual(int(observed[0].top1_rows[0]), 1)
                    self.assertEqual(int(observed[0].top1_target_ids[0]), 7)
                    # Flipping a zero-scale row leaves its projection unchanged.
                    self.assertTrue(torch.equal(observed[0].logits, snapshot.baseline_logits[:, 0]))

    def test_dense_reference_uses_native_f32_epilogue_order(self):
        module, inputs, d2t, labels = fixture(4, "fixed_affine_v1")
        with torch.no_grad():
            module.effective_scales()  # preserve the module's detached F32 scale source
        snapshot = capture_head(module, inputs, d2t, labels)
        candidate = FlipCandidate(4, (128, 130))
        result = evaluate_candidates(snapshot, [candidate])[0]
        reference = dense_reference(snapshot, candidate)
        torch.testing.assert_close(result.logits, reference["logits"], rtol=0, atol=0)
        self.assertTrue(torch.equal(snapshot.input_sum, snapshot.codes.sum(-1, keepdim=True)))

    def test_extreme_logsumexp_and_unsupported_labels_are_stable(self):
        module, inputs, d2t, labels = fixture(8, "learned")
        snapshot = capture_head(module, inputs, d2t, labels)
        # A large row scale drives logits past the range where naive exp(logit)
        # remains finite, while preserving a self-consistent dense baseline.
        alpha = torch.full_like(snapshot.alpha, 1.0e28)
        base_logits = (snapshot.dots.float() * alpha) * snapshot.beta
        if snapshot.midpoint is not None:
            base_logits = (
                base_logits + (snapshot.input_sum.float() * snapshot.midpoint) * snapshot.beta
            )
        if snapshot.bias is not None:
            base_logits = base_logits + snapshot.bias
        snapshot = replace(snapshot, alpha=alpha, baseline_logits=base_logits + 0.0)
        candidate = FlipCandidate(3, (0, 130))
        actual = evaluate_candidates(snapshot, [candidate])[0]
        expected = dense_reference(snapshot, candidate)
        self.assertTrue(bool(torch.isfinite(actual.ce_delta[snapshot.label_rows >= 0]).all()))
        self.assertTrue(bool(torch.isfinite(actual.top1_margin).all()))
        self.assertTrue(
            bool(
                (
                    (actual.ce_delta - expected["ce_delta"]).abs()
                    <= 2e-10 + 2e-12 * expected["ce_delta"].abs()
                )[snapshot.label_rows >= 0].all()
            )
        )
        self.assertTrue(math.isnan(float(actual.ce_delta[2])))

    def test_invalid_legacy_subnormal_reciprocal_is_reported(self):
        tiny = torch.finfo(torch.float32).tiny / 16
        module, inputs, d2t, labels = fixture(4, "fixed_symmetric")
        inputs.zero_()
        inputs[0, 0] = tiny
        inputs[0, 1] = -tiny
        # The fixed legacy A4 path forms qmax/absmax in F32; the learned v1 path
        # has a defined safe reciprocal fallback for this represented subnormal.
        with self.assertRaisesRegex(ValueError, "invalid/nonfinite codes"):
            capture_head(module, inputs, d2t, labels)
        learned, _, _, _ = fixture(4, "learned")
        safe = capture_head(learned, inputs, d2t, labels)
        self.assertTrue(bool(torch.isfinite(safe.baseline_logits).all()))
        self.assertEqual(safe.rule, "learned")

    def test_overshoot_can_reverse_the_complete_sign_flip_outcome(self):
        result = overshoot_counterexample()
        self.assertLess(result["linear_prediction_delta"], 0)
        self.assertGreater(result["actual_delta"], 0)
        self.assertEqual(result["old_accuracy"], 0.75)
        self.assertEqual(result["new_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
