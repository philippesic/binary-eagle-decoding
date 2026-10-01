"""Tiny CPU references for the opt-in A1 VJP and native-only no-grad path."""

import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import (
    RowBinaryLinear,
    W1AxContract,
    _A1SingleForward,
    hard_activation,
    shared_round_hard_signs,
)


def reference_projection(module, x):
    """Historical surrogate graph with an independent integer native forward."""
    quantized, beta, _ = hard_activation(x, 1)
    signs = (hard_sign_ste(module.latent_sign) if module._round_hard_signs is None
             else module._round_hard_signs)
    alpha = module.effective_scales()
    surrogate = F.linear(quantized, signs) * alpha
    if module.frozen_bias is not None:
        surrogate = surrogate + module.frozen_bias
    with torch.no_grad():
        raw = x.float().contiguous().view(torch.int32)
        # Raw bits, integer accumulation and separate F32 scales are independent
        # of the implementation under test, including signed zero/subnormals.
        a = torch.where((raw < 0) & ((raw & 0x7fffffff) != 0), -1, 1)
        s = torch.where(module.latent_sign < 0, -1, 1)
        native = ((a.to(torch.int64) @ s.to(torch.int64).T).float() * alpha) * beta
        if module.frozen_bias is not None:
            native = native + module.frozen_bias
    return native + (surrogate - surrogate.detach())


def fixture(mode="single_forward"):
    gen = torch.Generator(device="cpu").manual_seed(43)
    latent = torch.randn(5, 17, generator=gen)
    latent[0, :7] = torch.tensor([-1.01, -1., -0., 0., 1., 1.01, 0.5])
    module = RowBinaryLinear(
        latent, torch.tensor([0.25, 0.5, 0.75, 0.37, 0.213]), W1AxContract(1),
        bias=torch.tensor([0.1, -0.25, 0.0, 0.3, -0.7]), a1_computation=mode,
    )
    with torch.no_grad():
        module.scale_offset[:3].copy_(torch.tensor([-0.3, -0.5, -0.75]))
    return module


class QATComputationTests(unittest.TestCase):
    def assert_gradient_gate(self, actual, expected):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # Predeclared F32 gate: relative L2 <=1e-5, absolute near-zero <=1e-6.
        error = (actual - expected).norm()
        self.assertLessEqual(float(error), 1e-5 * float(expected.norm()) + 1e-6)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_default_and_invalid_selection(self):
        self.assertEqual(fixture("reference").a1_computation, "reference")
        with self.assertRaisesRegex(ValueError, "unknown A1"):
            fixture("invalid")

    def test_single_forward_counts_and_exact_native_values(self):
        x = torch.randn(2, 3, 17, generator=torch.Generator().manual_seed(1))
        for mode, count in (("reference", 2), ("single_forward", 1)):
            module = fixture(mode)
            expected = reference_projection(module, x).detach()
            with mock.patch("w1a1_eagle.recurrent_qat.F.linear", wraps=F.linear) as linear:
                actual = module(x)
            self.assertEqual(linear.call_count, count)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with torch.no_grad(), mock.patch(
                "w1a1_eagle.recurrent_qat.F.linear", wraps=F.linear
            ) as linear, mock.patch(
                "w1a1_eagle.recurrent_qat.hard_activation",
                side_effect=AssertionError("unused Q must not be built"),
            ):
                actual = module(x)
            self.assertEqual(linear.call_count, 1)
            self.assertFalse(actual.requires_grad)
            self.assertEqual(float(module.last_saturation_fraction), 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_raw_bits_zeros_subnormals_scale_order_and_noncontiguous_rows(self):
        tiny = torch.tensor([1], dtype=torch.int32).view(torch.float32)[0]
        x = torch.tensor([[-0., 0., -tiny, tiny, 1.], [tiny, -tiny, -1., 0., -0.]])
        x = x.T.contiguous().T
        self.assertFalse(x.is_contiguous())
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for mode in ("reference", "single_forward"):
                module = RowBinaryLinear(
                    torch.tensor([[0., -0., 0.5, 1., -0.5]], dtype=torch.float32),
                    torch.tensor([0.1234567], dtype=torch.float32), W1AxContract(1),
                    bias=torch.tensor([0.001234567], dtype=torch.float32), a1_computation=mode,
                )
                expected = reference_projection(module, x).detach()
                for enabled in (True, False):
                    with torch.set_grad_enabled(enabled):
                        actual = module(x)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_zero_output_bits_match_historical_cancellation_epilogue(self):
        for mode in ("reference", "single_forward"):
            module = RowBinaryLinear(torch.ones(1, 3), torch.zeros(1),
                                     W1AxContract(1), a1_computation=mode)
            x = torch.tensor([[-1., -1., -1.], [0., -0., 0.]])
            expected = reference_projection(module, x).detach().view(torch.int32)
            for enabled in (True, False):
                with torch.set_grad_enabled(enabled):
                    actual = module(x).detach().view(torch.int32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_surrogate_vjp_shapes_clip_boundaries_and_zero_scales(self):
        for shape in ((17,), (3, 17), (2, 3, 17)):
            for zero_input in (False, True):
                module = fixture()
                reference = copy.deepcopy(module)
                x = torch.randn(shape, generator=torch.Generator().manual_seed(7))
                if zero_input:
                    x.zero_()
                x.requires_grad_()
                xr = x.detach().clone().requires_grad_()
                upstream = torch.randn((*shape[:-1], 5),
                                       generator=torch.Generator().manual_seed(8))
                module(x).backward(upstream)
                reference_projection(reference, xr).backward(upstream)
                for actual, expected in ((x.grad, xr.grad),
                                         (module.latent_sign.grad, reference.latent_sign.grad),
                                         (module.scale_offset.grad, reference.scale_offset.grad)):
                    self.assert_gradient_gate(actual, expected)
                self.assertTrue(bool((module.latent_sign.grad[:3] == 0).all()))
                self.assertEqual(float(module.scale_offset.grad[0]), 0.)
                self.assertTrue(bool((module.latent_sign.grad[
                    module.latent_sign.abs() > 1] == 0).all()))
                if zero_input:
                    self.assertGreater(float(x.grad.abs().sum()), 0.)
                    self.assertTrue(bool((module.scale_offset.grad == 0).all()))
                else:
                    self.assertGreater(float(module.scale_offset.grad[1:3].abs().sum()), 0.)

    def test_backward_uses_actual_quantized_values_and_optional_bias(self):
        # Q deliberately differs from beta*A: raw-bit native signs must never
        # substitute for actual surrogate Q in the backward (e.g. CUDA FTZ).
        q = torch.tensor([[0.2, -0.2, 0.2]], requires_grad=True)
        s = torch.tensor([[1., -1., 1.], [-1., 1., 1.]], requires_grad=True)
        alpha = torch.tensor([0., 0.25], requires_grad=True)
        bias = torch.tensor([0.1, -0.3], requires_grad=True)
        raw = torch.tensor([[-1., -1., 1.]], requires_grad=True)
        beta = torch.tensor([[0.2]], requires_grad=True)
        upstream = torch.tensor([[0.7, -0.4]])
        actual = _A1SingleForward.apply(q, s, alpha, raw, beta, bias)
        expected = F.linear(q, s) * alpha + bias
        ga = torch.autograd.grad(actual, (q, s, alpha, raw, beta, bias), upstream,
                                 allow_unused=True)
        ge = torch.autograd.grad(expected, (q, s, alpha, bias), upstream)
        for index, expected_gradient in zip((0, 1, 2, 5), ge, strict=True):
            self.assert_gradient_gate(ga[index], expected_gradient)
        self.assertIsNone(ga[3])
        self.assertIsNone(ga[4])
        self.assertNotEqual(float(ga[2][0]), 0.)

    def test_shared_signs_later_only_recurrent_state_and_f16_kv(self):
        from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout
        from w1a1_eagle.recurrent_qat import JointQATConfig

        linears, _ = tiny_joint_fixture(JointQATConfig(W1AxContract(1)))
        for module in linears.values():
            module.a1_computation = "single_forward"
        hooks = [linears[path].register_forward_hook(
            lambda module, inputs, output: output.to(torch.float16).float()
        ) for path in ("midlayer.self_attn.k_proj", "midlayer.self_attn.v_proj")]
        try:
            with shared_round_hard_signs(linears):
                logits, cache, states = tiny_rollout(linears, "cpu")
                # Only the final position is supervised. Repeated attached signs
                # must aggregate gradients without breaking earlier state/K/V.
                F.cross_entropy(logits[-1:], torch.tensor([1])).backward()
        finally:
            for hook in hooks:
                hook.remove()
        for node in (states[0], cache[0][0], cache[0][1]):
            self.assertGreater(float(node.grad.abs().sum()), 0.)
        for module in linears.values():
            self.assertIsNone(module._round_hard_signs)
            self.assertTrue(bool(torch.isfinite(module.latent_sign.grad).all()))

    def test_one_clipped_synthetic_adamw_update_preserves_signs_and_scales(self):
        actual = fixture()
        reference = copy.deepcopy(actual)
        x = torch.randn(3, 17, generator=torch.Generator().manual_seed(9))
        label = torch.tensor([1, 3, 4])
        for module, forward in ((actual, lambda m, v: m(v)),
                                (reference, reference_projection)):
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.003, weight_decay=0,
                                           foreach=False)
            F.cross_entropy(forward(module, x), label).backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.7)
            optimizer.step()
            module.project_scales_()
        torch.testing.assert_close(actual.latent_sign.sign(), reference.latent_sign.sign(),
                                   rtol=0, atol=0)
        for parameter, expected in zip(actual.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.effective_scales(), reference.effective_scales(),
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual(x).argmax(-1), reference(x).argmax(-1),
                                   rtol=0, atol=0)

    def test_requested_operand_gradient_subsets(self):
        for enabled in ((True, False, False), (False, True, False), (False, False, True)):
            q = torch.tensor([[0.25, -0.25]], requires_grad=enabled[0])
            s = torch.tensor([[1., -1.]], requires_grad=enabled[1])
            alpha = torch.tensor([0.], requires_grad=enabled[2])
            result = _A1SingleForward.apply(q, s, alpha, q.detach(), torch.ones(1, 1), None)
            surrogate = F.linear(q, s) * alpha
            inputs = tuple(v for v in (q, s, alpha) if v.requires_grad)
            actual = torch.autograd.grad(result.sum(), inputs)
            expected = torch.autograd.grad(surrogate.sum(), inputs)
            for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
                self.assert_gradient_gate(actual_gradient, expected_gradient)

    def test_custom_double_backward_matches_defined_surrogate(self):
        gen = torch.Generator().manual_seed(5)
        q = torch.randn(2, 4, generator=gen, requires_grad=True)
        s = torch.randn(3, 4, generator=gen, requires_grad=True)
        alpha = torch.randn(3, generator=gen, requires_grad=True)
        g = torch.randn(2, 3, generator=gen, requires_grad=True)
        actual = _A1SingleForward.apply(q, s, alpha, q.detach(),
                                        torch.ones(2, 1), None)
        reference = F.linear(q, s) * alpha
        inputs = (q, s, alpha, g)
        gradients = []
        for output in (actual, reference):
            first = torch.autograd.grad(output, inputs[:3], g, create_graph=True)
            gradients.append(torch.autograd.grad(sum(v.square().sum() for v in first),
                                                  inputs, retain_graph=True))
        for actual_gradient, reference_gradient in zip(*gradients, strict=True):
            self.assert_gradient_gate(actual_gradient, reference_gradient)

    def test_no_grad_validation_matches_activation_boundary(self):
        for bad in (torch.ones(17, dtype=torch.int64), torch.full((17,), float("nan"))):
            with torch.no_grad(), self.assertRaisesRegex(ValueError, "finite floating"):
                fixture()(bad)


if __name__ == "__main__":
    unittest.main()
