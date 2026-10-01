"""CPU synthetic arithmetic, gradient, ownership, and resume acceptance checks."""

import copy
import dataclasses
import io
import math
import unittest

import numpy as np
import torch

from w1a1_eagle.learned_activation import (
    BOUNDARY_PATHS,
    CLIP_FLOOR,
    LearnedActivationBank,
    LearnedActivationQuantizer,
    NativeActivationEvidence,
    learned_activation,
    learned_activation_reference,
    native_order_linear,
)
from w1a1_eagle.recurrent_binary import pack_signs, unpack_signs
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, hard_activation


def fixture(bits=4):
    paths = [p for consumers in BOUNDARY_PATHS.values() for p in consumers]
    linears = {p: RowBinaryLinear(torch.ones(3, 4), torch.ones(3), W1AxContract(bits)) for p in paths}
    bank = LearnedActivationBank(bits, {p: m.in_features for p, m in linears.items()})
    return bank, linears


class LearnedActivationTests(unittest.TestCase):
    def test_initialization_exact_baseline_ordinary_inputs(self):
        generator = torch.Generator().manual_seed(17)
        x = torch.cat([torch.randn(11, 129, generator=generator), torch.zeros(1, 129)])
        for bits in (1, 4, 8):
            with self.subTest(bits=bits):
                quantizer = LearnedActivationQuantizer(bits, "fc", 129)
                result = quantizer(x)
                expected, scales, saturated = hard_activation(x, bits)
                torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
                torch.testing.assert_close(result.scale, scales, rtol=0, atol=0)
                torch.testing.assert_close(result.saturated, saturated)
                self.assertFalse(result.scale.requires_grad)
                self.assertFalse(result.codes.requires_grad)
                self.assertFalse(result.clipped.requires_grad)

    def test_clip_changes_codes_and_distinguishes_actual_clipping(self):
        x = torch.tensor([[1., .6, .1, -.1, -.6, -1., 0.]])
        for bits in (4, 8):
            baseline = learned_activation_reference(x, bits, torch.tensor(1.))
            learned = learned_activation_reference(x, bits, torch.tensor(.5))
            self.assertTrue(bool((learned.codes != baseline.codes).any()))
            self.assertFalse(bool(baseline.clipped.any()))
            self.assertTrue(bool(baseline.saturated[0, 0]))
            self.assertTrue(bool(learned.clipped[0, 0]))
            torch.testing.assert_close(learned.scale, baseline.scale * .5, rtol=0, atol=0)
            self.assertLess(float(learned.values[0, 0]), float(baseline.values[0, 0]))
            self.assertEqual(int(learned.codes[0, -1]), 0)
            self.assertGreaterEqual(int(learned.codes.min()), -((1 << (bits - 1)) - 1))

    def test_round_even_native_f32_order(self):
        # Native reciprocal is exactly 1 here; positive and negative half ties.
        x = torch.tensor([[7., .5, 1.5, 2.5, -.5, -1.5, -2.5]])
        result = learned_activation_reference(x, 4, torch.tensor(1.))
        torch.testing.assert_close(result.codes, torch.tensor([[7, 0, 2, 2, 0, -2, -2]], dtype=torch.int8))
        torch.testing.assert_close(result.scale, torch.ones(1, 1))

    def test_threshold_changes_bits_retains_beta_and_ties_positive(self):
        x = torch.tensor([[1., -1., .1, -.1]])
        before = learned_activation_reference(x, 1, torch.tensor(0.))
        after = learned_activation_reference(x, 1, torch.tensor(.5))
        torch.testing.assert_close(before.codes, torch.tensor([[1, -1, 1, -1]], dtype=torch.int8))
        torch.testing.assert_close(after.codes, torch.tensor([[1, -1, -1, -1]], dtype=torch.int8))
        torch.testing.assert_close(after.scale, before.scale, rtol=0, atol=0)
        ties = learned_activation_reference(torch.tensor([[1., -1., .5, -.5]]), 1, torch.tensor(2/3))
        self.assertEqual(int(ties.codes[0, 2]), 1)
        negative_tie = learned_activation_reference(torch.tensor([[1., -1., .5, -.5]]), 1, torch.tensor(-2/3))
        self.assertEqual(int(negative_tie.codes[0, 3]), 1)

    def test_a1_signed_zeros_negative_subnormals_and_tail_bits(self):
        tiny = torch.finfo(torch.float32).tiny / 2
        x = torch.zeros(1, 129)
        x[0, 0], x[0, 1], x[0, 127], x[0, 128] = -0., -tiny, tiny, -tiny
        result = learned_activation_reference(x, 1, torch.tensor(0.))
        self.assertEqual(int(result.codes[0, 0]), 1)
        self.assertEqual(int(result.codes[0, 1]), -1)
        self.assertEqual(int(result.codes[0, 127]), 1)
        self.assertEqual(int(result.codes[0, 128]), -1)
        packed = pack_signs(result.codes.float().numpy())
        torch.testing.assert_close(torch.from_numpy(unpack_signs(packed, 129)), result.codes.float())
        self.assertEqual(int(np.uint32(packed[0, -1])), 0)
        # A negative threshold creates an exactly equal negative subnormal tie.
        ties = learned_activation_reference(torch.tensor([[tiny, -tiny]]), 1, torch.tensor(-1.))
        torch.testing.assert_close(ties.codes, torch.ones(1, 2, dtype=torch.int8))

    def test_uniform_subnormal_overflow_fallback_is_finite_and_symmetric(self):
        tiny = torch.finfo(torch.float32).tiny / 16
        x = torch.tensor([[tiny, -tiny, 0., tiny / 2]], requires_grad=True)
        for bits in (4, 8):
            parameter = torch.tensor(1., requires_grad=True)
            result = learned_activation(x, bits, parameter)
            qmax = (1 << (bits - 1)) - 1
            self.assertEqual(int(result.codes[0, 0]), qmax)
            self.assertEqual(int(result.codes[0, 1]), -qmax)
            self.assertEqual(int(result.codes[0, 2]), 0)
            self.assertTrue(bool(torch.isfinite(result.values).all()))
            result.values.sum().backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertTrue(bool(torch.isfinite(parameter.grad)))
            x.grad = None
        # A representable nonzero input can become an exactly zero F32 limit.
        smallest = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
        result = learned_activation_reference(smallest.reshape(1, 1), 4, torch.tensor(CLIP_FLOOR))
        self.assertEqual(int(result.codes[0, 0]), 0)
        self.assertEqual(float(result.values[0, 0]), 0)

    def test_zero_rows_no_amplitude_invention_input_gradient_retained(self):
        for bits in (1, 4, 8):
            x = torch.tensor([[0., -0., 0.]], requires_grad=True)
            parameter = torch.tensor(.25 if bits == 1 else .5, requires_grad=True)
            result = learned_activation(x, bits, parameter)
            self.assertEqual(float(result.values.detach().abs().sum()), 0)
            self.assertEqual(float(result.scale.sum()), 0)
            self.assertFalse(bool(result.saturated.any()))
            result.values.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
            self.assertEqual(float(parameter.grad), 0)

    def test_clip_gradient_matches_normalized_lsq_interior_and_saturation(self):
        x = torch.tensor([[1., .2, -.3, -.9]], requires_grad=True)
        parameter = torch.tensor(.5, requires_grad=True)
        weights = torch.tensor([[2., -1., .5, 3.]])
        result = learned_activation(x, 4, parameter)
        (result.values * weights).sum().backward()
        qmax, limit = 7, .5
        support = x.detach().abs() <= limit
        normalized = x.detach() * (qmax / limit)
        codes = result.codes.float()
        expected_local = torch.where(support, (codes - normalized) / qmax, codes / qmax)
        expected = (weights * expected_local).sum() / math.sqrt(x.numel() * qmax)
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(x.grad, weights * support)
        self.assertGreater(abs(float(parameter.grad)), .01)

    def test_threshold_normalized_gradient_and_clipped_support(self):
        x = torch.tensor([[1., -.1, .1, -.2]], requires_grad=True)
        parameter = torch.tensor(.2, requires_grad=True)
        result = learned_activation(x, 1, parameter)
        result.values.sum().backward()
        beta = x.detach().abs().double().mean().float()
        support = (x.detach() - parameter.detach() * beta).abs() <= beta
        torch.testing.assert_close(x.grad, support.float())
        torch.testing.assert_close(parameter.grad, -beta * support.sum() / math.sqrt(x.numel()))
        self.assertGreater(abs(float(parameter.grad)), .1)

    def test_valid_mask_excludes_padding_and_chunk_normalization(self):
        x = torch.tensor([[1., .1, -.2, .3], [1., .2, -.3, .4]], requires_grad=True)
        parameter = torch.tensor(.5, requires_grad=True)
        result = learned_activation(x, 4, parameter, valid_mask=torch.tensor([True, False]))
        result.values.sum().backward()
        self.assertEqual(float(x.grad[1].abs().sum()), 0)
        single = torch.tensor(.5, requires_grad=True)
        learned_activation(x.detach()[:1], 4, single).values.sum().backward()
        torch.testing.assert_close(parameter.grad, single.grad, rtol=0, atol=0)
        full = torch.tensor(.5, requires_grad=True)
        chunk = torch.tensor(.5, requires_grad=True)
        learned_activation(x.detach(), 4, full).values.sum().backward()
        for row in x.detach().split(1):
            learned_activation(row, 4, chunk, normalization_count=x.numel()).values.sum().backward()
        torch.testing.assert_close(full.grad, chunk.grad, rtol=1e-6, atol=1e-8)

    def test_invalid_inputs_parameters_and_masks_fail_early(self):
        for input in (torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                      torch.tensor([1]), torch.empty(1, 0), torch.tensor([[1e300]], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                learned_activation(input, 4, torch.tensor(1.))
        for parameter in (torch.tensor(0.), torch.tensor(-1.), torch.tensor(1.1), torch.tensor(float("nan")),
                          torch.tensor([1.]), torch.tensor(1., dtype=torch.float64)):
            with self.assertRaises(ValueError):
                learned_activation(torch.ones(1, 3), 4, parameter)
        with self.assertRaises(ValueError):
            learned_activation(torch.ones(1, 3), 16, torch.tensor(1.))
        with self.assertRaises(ValueError):
            learned_activation(torch.ones(1, 3), 4, torch.tensor(1.), valid_mask=torch.ones(1))
        with self.assertRaises(ValueError):
            learned_activation(torch.ones(1, 3), 4, torch.tensor(1.), normalization_count=2)
        with self.assertRaises(ValueError):
            learned_activation(torch.full((1, 3), 1e38), 1, torch.tensor(10.))

    def test_native_order_integer_dot_then_row_then_token_scale(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(3, 129, generator=generator)
        signs = torch.where(torch.randn(4, 129, generator=generator) < 0, -1., 1.)
        scales = torch.tensor([.5, .7, .9, 1.1])
        bias = torch.tensor([.1, -.2, .3, -.4])
        for bits in (1, 4, 8):
            result = learned_activation_reference(x, bits, torch.tensor(.2 if bits == 1 else .5))
            expected = torch.tensor(result.codes.numpy().astype(np.int64) @ signs.numpy().astype(np.int64).T,
                                    dtype=torch.float32)
            expected = (expected * scales) * result.scale + bias
            torch.testing.assert_close(native_order_linear(result, signs, scales, bias), expected, rtol=0, atol=0)

    def test_shared_boundary_attachment_unique_optimizer_ownership(self):
        bank, linears = fixture()
        bank.attach(linears)
        bank.validate_attachment(linears)
        self.assertEqual(len(list(bank.parameters())), 6)
        q, k, v = [linears[p].activation_quantizer for p in BOUNDARY_PATHS["qkv"]]
        self.assertIs(q, k)
        self.assertIs(k, v)
        gate, up = [linears[p].activation_quantizer for p in BOUNDARY_PATHS["gate_up"]]
        self.assertIs(gate, up)
        recovered = LearnedActivationBank.from_attached(linears)
        self.assertEqual(recovered.identity(), bank.identity())
        self.assertEqual([id(p) for p in recovered.parameters()], [id(p) for p in bank.parameters()])
        self.assertIs(recovered.quantizers["qkv"], q)
        with self.assertRaises(ValueError):
            bank.attach(linears)
        linears[BOUNDARY_PATHS["qkv"][1]].activation_quantizer = LearnedActivationQuantizer(4, "qkv", 4)
        with self.assertRaisesRegex(ValueError, "ownership"):
            bank.validate_attachment(linears)
        with self.assertRaisesRegex(ValueError, "ownership"):
            LearnedActivationBank.from_attached(linears)

    def test_from_attached_rejects_unknown_and_cross_boundary_aliases(self):
        bank, linears = fixture()
        with self.assertRaises(ValueError):
            LearnedActivationBank.from_attached(linears)
        bank.attach(linears)
        bank.quantizers["head"].clip_ratio = bank.quantizers["fc"].clip_ratio
        with self.assertRaisesRegex(ValueError, "distinct"):
            LearnedActivationBank.from_attached(linears)

    def test_attachment_rejects_incompatible_contract_before_mutation(self):
        bank, linears = fixture()
        linears["fc"] = RowBinaryLinear(torch.ones(3, 4), torch.ones(3), W1AxContract(1))
        with self.assertRaises(ValueError):
            bank.attach(linears)
        self.assertTrue(all(getattr(m, "activation_quantizer", None) is None for m in linears.values()))
        widths = {p: 4 for paths in BOUNDARY_PATHS.values() for p in paths}
        widths[BOUNDARY_PATHS["qkv"][0]] = 5
        with self.assertRaisesRegex(ValueError, "identical"):
            LearnedActivationBank(4, widths)

    def test_projected_clip_initializer_has_recoverable_gradient(self):
        quantizer = LearnedActivationQuantizer(4, "fc", 4)
        optimizer = torch.optim.SGD(quantizer.parameters(), lr=.1)
        before = quantizer.parameter.detach().clone()
        result = quantizer(torch.tensor([[1., .1, .2, .3]]))
        (result.values * torch.tensor([[0., 1., 0., 0.]])).sum().backward()
        self.assertGreater(float(quantizer.parameter.grad), 0)
        optimizer.step()
        quantizer.project_()
        self.assertLess(float(quantizer.parameter.detach()), float(before))
        with torch.no_grad():
            quantizer.parameter.fill_(-1)
        quantizer.project_()
        self.assertEqual(float(quantizer.parameter.detach()), CLIP_FLOOR)
        with torch.no_grad():
            quantizer.parameter.fill_(2)
        quantizer.project_()
        self.assertEqual(float(quantizer.parameter.detach()), 1)

    def test_tiny_synthetic_updates_change_threshold_bits_and_clip_codes(self):
        for bits, rate, x, weights in (
            (1, 1., [[1., -1., .1, -.1]], [[1., 1., 1., 1.]]),
            (4, 60., [[1., .1, .2, .3]], [[0., 1., 0., 0.]]),
        ):
            quantizer = LearnedActivationQuantizer(bits, "fc", 4)
            optimizer = torch.optim.SGD(quantizer.parameters(), lr=rate)
            before = quantizer(torch.tensor(x)).codes.clone()
            result = quantizer(torch.tensor(x))
            (result.values * torch.tensor(weights)).sum().backward()
            optimizer.step()
            quantizer.project_()
            after = quantizer(torch.tensor(x)).codes
            self.assertTrue(bool((before != after).any()))

    def test_reusing_shared_input_equals_consumer_grad_accumulation(self):
        x = torch.tensor([[1., .1, -.2, .3]], requires_grad=True)
        parameter = torch.tensor(.5, requires_grad=True)
        shared = learned_activation(x, 4, parameter)
        weights = [torch.tensor([[1., -.5, 2., .2]]), torch.tensor([[-.2, .5, -.3, .1]])]
        sum((shared.values * w).sum() for w in weights).backward()
        expected_x, expected_param = x.grad.clone(), parameter.grad.clone()
        x.grad, parameter.grad = None, None
        sum((learned_activation(x, 4, parameter).values * w).sum() for w in weights).backward()
        torch.testing.assert_close(x.grad, expected_x, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(parameter.grad, expected_param, rtol=1e-6, atol=1e-8)

    def test_checkpoint_and_optimizer_resume_exact(self):
        for bits in (1, 4, 8):
            bank, _ = fixture(bits)
            optimizer = torch.optim.AdamW(bank.parameters(), lr=.01, weight_decay=0)

            def step(active, opt):
                opt.zero_grad(set_to_none=True)
                x = torch.tensor([[1., .1, -.2, .3]])
                loss = sum((q(x).values * torch.tensor([[0., 1., -.5, .1]])).sum()
                           for q in active.quantizers.values())
                loss.backward()
                opt.step()
                active.project_()

            step(bank, optimizer)
            stream = io.BytesIO()
            torch.save({"bank": bank.checkpoint(), "optimizer": optimizer.state_dict()}, stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=True)
            resumed, _ = fixture(bits)
            resumed.load_checkpoint(saved["bank"])
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01, weight_decay=0)
            resumed_optimizer.load_state_dict(saved["optimizer"])
            step(bank, optimizer)
            step(resumed, resumed_optimizer)
            for p, r in zip(bank.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, r, rtol=0, atol=0)
            self.assertEqual(bank.native_parameters(), resumed.native_parameters())
            self.assertEqual(bank.parameters_sha256(), resumed.parameters_sha256())
            for state, resumed_state in zip(optimizer.state.values(), resumed_optimizer.state.values()):
                for key in state:
                    torch.testing.assert_close(state[key], resumed_state[key], rtol=0, atol=0)

    def test_checkpoint_rejects_changed_recipe_identity_ownership_nonfinite_atomically(self):
        bank, _ = fixture()
        baseline = bank.checkpoint()
        invalid = copy.deepcopy(baseline)
        invalid["parameters"]["fc"] = torch.tensor(.5)
        invalid["parameters"]["head"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            bank.load_checkpoint(invalid)
        self.assertEqual(float(bank.quantizers["fc"].parameter.detach()), 1)
        for key, value in (("recipe", "other"), ("version", 2)):
            invalid = copy.deepcopy(baseline)
            invalid["identity"][key] = value
            with self.assertRaisesRegex(ValueError, "identity"):
                bank.load_checkpoint(invalid)
        invalid = copy.deepcopy(baseline)
        del invalid["parameters"]["qkv"]
        with self.assertRaisesRegex(ValueError, "ownership"):
            bank.load_checkpoint(invalid)
        other_bits, _ = fixture(1)
        with self.assertRaisesRegex(ValueError, "identity"):
            other_bits.load_checkpoint(baseline)
        quantizer = bank.quantizers["fc"]
        state = copy.deepcopy(quantizer.state_dict())
        state["_extra_state"]["in_features"] = 5
        state["clip_ratio"] = torch.tensor(.5)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            quantizer.load_state_dict(state)
        self.assertEqual(float(quantizer.parameter.detach()), 1)
        restored = LearnedActivationQuantizer(4, "fc", 4)
        restored.load_state_dict(quantizer.state_dict())
        torch.testing.assert_close(restored.parameter, quantizer.parameter, rtol=0, atol=0)

    def test_native_payload_and_readiness_fail_closed_and_invalidate_on_update(self):
        bank, _ = fixture()
        payload = bank.native_parameters()
        self.assertEqual(payload["version"], 1)
        self.assertEqual(set(payload["boundaries"]), set(BOUNDARY_PATHS))
        self.assertEqual(payload["boundaries"]["qkv"], {"bits": 4, "threshold_delta": 0., "clip_ratio": 1.})
        with self.assertRaisesRegex(ValueError, "requires"):
            bank.require_native_ready(None, backend="cuda")
        # Synthetic evidence exercises plumbing; this test asserts no actual
        # native, CUDA, quality, or throughput validation.
        evidence = NativeActivationEvidence(bank.parameters_sha256(), "a" * 40, "cpu", "synthetic fixture",
                                            True, .05, 0)
        bank.require_native_ready(evidence, backend="cpu")
        for changed in (dataclasses.replace(evidence, backend="cuda"),
                        dataclasses.replace(evidence, exact_pack=False),
                        dataclasses.replace(evidence, max_relative_rms=.11),
                        dataclasses.replace(evidence, max_relative_rms=float("nan")),
                        dataclasses.replace(evidence, changed_choices_above_margin=1)):
            with self.assertRaises(ValueError):
                bank.require_native_ready(changed, backend="cpu")
        with self.assertRaises(ValueError):
            bank.require_native_ready(evidence, backend="cuda")
        with torch.no_grad():
            bank.quantizers["fc"].parameter.fill_(.5)
        with self.assertRaisesRegex(ValueError, "stale"):
            bank.require_native_ready(evidence, backend="cpu")


if __name__ == "__main__":
    unittest.main()
