"""Acceptance checks for bounded fixed-state native-order head telemetry."""

import unittest
from unittest.mock import patch

import torch

from research.parallel20261002.head_flip_oracle.reference.oracle import (
    FlipCandidate,
    capture_head,
    evaluate_candidates,
    overshoot_counterexample,
)
from w1a1_eagle.affine_binary import AffineBinaryConfig, AffineBinaryMidpoint
from w1a1_eagle.learned_activation import LearnedActivationQuantizer
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract


class HeadFlipOracleTests(unittest.TestCase):
    def fixture(self, bits=1, learned=False, affine=False):
        module = RowBinaryLinear(
            torch.tensor([[1., -1., 0., -0., 1.], [-1., 1., -1., 1., -1.],
                          [1., 1., -1., -1., 1.]]),
            torch.tensor([0.3, 0., 0.7]), W1AxContract(bits),
            bias=torch.tensor([0.1, -0.2, 0.]),
        )
        if learned:
            module.activation_quantizer = LearnedActivationQuantizer(bits, "head", 5)
            with torch.no_grad():
                module.activation_quantizer.parameter.fill_(0.25 if bits == 1 else 0.5)
        if affine:
            module.affine_binary = AffineBinaryMidpoint(3, AffineBinaryConfig(True, "all"))
            with torch.no_grad():
                module.affine_binary.midpoint.copy_(torch.tensor([0.4, -0.1, 0.2]))
        x = torch.tensor([[0., -0., -1., 0.5, 1.], [0., 0., 0., 0., 0.],
                          [1., -2., 3., -4., 5.]])
        snapshot = capture_head(module, x, torch.tensor([7, 2, 9]), torch.tensor([7, 3, 9]))
        return module, snapshot

    def test_native_epilogue_and_integer_flip_all_rules(self):
        for bits in (1, 4, 8):
            for learned, affine in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(bits=bits, learned=learned, affine=affine):
                    _, snapshot = self.fixture(bits, learned, affine)
                    candidate = FlipCandidate(0, (0, 2, 4))
                    result, = evaluate_candidates(snapshot, [candidate])
                    signs = snapshot.signs.clone()
                    signs[0, list(candidate.columns)] *= -1
                    dot = snapshot.codes @ signs[0]
                    self.assertTrue(torch.equal(result.dots, dot))
                    expected = (dot.float() * snapshot.alpha[0]) * snapshot.beta[:, 0]
                    if affine:
                        expected += (snapshot.input_sum[:, 0].float()
                                     * snapshot.midpoint[0]) * snapshot.beta[:, 0]
                    expected += snapshot.bias[0]
                    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
                    changed = snapshot.baseline_logits.double().clone()
                    changed[:, 0] = expected.double()
                    old = torch.logsumexp(snapshot.baseline_logits.double(), -1)
                    new = torch.logsumexp(changed, -1)
                    label = snapshot.label_rows.clamp_min(0)
                    expected_ce = new - old - (
                        changed[torch.arange(3), label]
                        - snapshot.baseline_logits.double()[torch.arange(3), label]
                    )
                    supported = snapshot.label_rows >= 0
                    torch.testing.assert_close(result.ce_delta[supported], expected_ce[supported],
                                               rtol=2e-12, atol=2e-10)
                    self.assertTrue(torch.isnan(result.ce_delta[1]))

    def test_source_capture_and_evaluation_are_nondestructive(self):
        module, _ = self.fixture(4, True, True)
        before = {key: value.clone() for key, value in module.state_dict().items()
                  if isinstance(value, torch.Tensor)}
        saturation = module.last_saturation_fraction
        snapshot = capture_head(module, torch.zeros(2, 5), torch.tensor([7, 2, 9]),
                                torch.tensor([7, 9]))
        captured = snapshot.baseline_logits.clone()
        with patch("torch.nn.functional.linear", side_effect=AssertionError("candidate GEMM")):
            activities = [torch.profiler.ProfilerActivity.CPU]
            with torch.profiler.profile(activities=activities) as profile:
                results = evaluate_candidates(snapshot, [FlipCandidate(0, (0,)),
                                                         FlipCandidate(0, (0, 1))])
        self.assertFalse({"aten::mm", "aten::addmm", "aten::bmm"}
                         & {event.key for event in profile.key_averages()})
        self.assertEqual(sum(r.avoided_candidate_head_gemms for r in results), 2)
        self.assertTrue(torch.equal(snapshot.baseline_logits, captured))
        for key, value in before.items():
            self.assertTrue(torch.equal(module.state_dict()[key], value))
        torch.testing.assert_close(module.last_saturation_fraction, saturation)

    def test_tie_mapping_and_zero_scale(self):
        module = RowBinaryLinear(torch.ones(3, 3), torch.zeros(3), W1AxContract(1))
        snapshot = capture_head(module, torch.tensor([[0., -0., 0.]]),
                                torch.tensor([9, 2, 7]), torch.tensor([7]))
        self.assertEqual(snapshot.codes.tolist(), [[1, 1, 1]])
        result, = evaluate_candidates(snapshot, [FlipCandidate(0, (0, 1))])
        self.assertEqual(result.top1_rows.tolist(), [0])
        self.assertEqual(result.top1_target_ids.tolist(), [9])
        self.assertEqual(result.ce_delta.tolist(), [0.])
        self.assertTrue(result.rounding_tie.item())

    def test_candidates_are_independent_and_bounded(self):
        _, snapshot = self.fixture()
        candidates = [FlipCandidate(0, (0,)), FlipCandidate(0, (1,))]
        together = evaluate_candidates(snapshot, candidates)
        for candidate, result in zip(candidates, together, strict=True):
            separate, = evaluate_candidates(snapshot, [candidate])
            self.assertTrue(torch.equal(result.logits, separate.logits))
        for invalid in (FlipCandidate(0, (0, 0)), FlipCandidate(-1, (0,)),
                        FlipCandidate(0, (5,)), FlipCandidate(0, ())):
            with self.assertRaises(ValueError):
                evaluate_candidates(snapshot, [invalid])
        with self.assertRaisesRegex(ValueError, "32"):
            evaluate_candidates(snapshot, [candidates[0]] * 33)

    def test_legacy_reciprocal_overflow_fails_closed(self):
        module = RowBinaryLinear(torch.ones(2, 2), torch.ones(2), W1AxContract(8))
        with self.assertRaisesRegex(ValueError, "invalid/nonfinite"):
            capture_head(module, torch.tensor([[1e-44, 0.]]),
                         torch.tensor([0, 1]), torch.tensor([0]))

    def test_valid_derivative_can_overshoot(self):
        case = overshoot_counterexample()
        self.assertAlmostEqual(case["gradient"], 0.26159415595576463)
        self.assertLess(case["linear_prediction_delta"], 0)
        self.assertAlmostEqual(case["actual_delta"], 1.)
        self.assertEqual(case["old_accuracy"], 0.75)
        self.assertEqual(case["new_accuracy"], 0.25)
        module = RowBinaryLinear(torch.ones(2, 1), torch.tensor([2., 0.]), W1AxContract(1))
        labels = torch.tensor([0, 0, 0, 1])
        inputs = torch.ones(4, 1)
        loss = torch.nn.functional.cross_entropy(module(inputs), labels)
        loss.backward()
        self.assertAlmostEqual(float(module.latent_sign.grad[0, 0]), case["gradient"], places=6)
        snapshot = capture_head(module, inputs, torch.tensor([0, 1]), labels)
        result, = evaluate_candidates(snapshot, [FlipCandidate(0, (0,))])
        self.assertAlmostEqual(float(result.ce_delta.mean()), case["actual_delta"])
        self.assertEqual(float((result.top1_rows == labels).float().mean()), 0.25)


if __name__ == "__main__":
    unittest.main()
