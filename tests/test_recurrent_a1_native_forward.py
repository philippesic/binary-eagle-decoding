"""CPU checks for native-order A1 values and unchanged QAT surrogate gradients."""

import copy
import unittest
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import (
    RowBinaryLinear,
    W1AxContract,
    hard_activation,
    shared_round_hard_signs,
)


def original_surrogate(module, x):
    quantized, _, _ = hard_activation(x, module.contract.activation_bits)
    signs = (hard_sign_ste(module.latent_sign) if module._round_hard_signs is None
             else module._round_hard_signs)
    return F.linear(quantized, signs) * module.effective_scales() + (
        0 if module.frozen_bias is None else module.frozen_bias
    )


def integer_reference(module, x):
    # Independent integer accumulation, followed by two separate F32 multiplies.
    inputs = torch.where(x.float() < 0, -1, 1).to(torch.int64)
    weights = torch.where(module.latent_sign.detach() < 0, -1, 1).to(torch.int64)
    dot = inputs @ weights.T
    activation_scale = x.float().abs().double().mean(dim=-1, keepdim=True).float()
    scales = (module.initial_scale + module.scale_offset.detach()).clamp(min=0)
    result = (dot.float() * scales) * activation_scale
    return result if module.frozen_bias is None else result + module.frozen_bias


class A1NativeForwardTests(unittest.TestCase):
    def test_real_width_exact_zero_and_nonzero_integer_dots(self):
        generator = torch.Generator(device="cpu").manual_seed(31)
        for width in (2560, 7680):
            with self.subTest(width=width):
                weights = torch.ones(16, width)
                for row in range(16):
                    # Balanced rows, then small and large nonzero integer dots.
                    negatives = width // 2 if row < 8 else width // 2 - (row - 7)
                    weights[row, :negatives] = -1
                    weights[row] = weights[row, torch.randperm(width, generator=generator)]
                module = RowBinaryLinear(
                    weights, torch.linspace(0.0021, 0.21, 16), W1AxContract(1)
                )
                x = torch.full((2, width), 0.1234567)
                x[1] = -x[1]
                actual = module(x)
                torch.testing.assert_close(actual, integer_reference(module, x), rtol=0, atol=0)
                self.assertTrue(bool((actual[:, :8] == 0).all()))
                self.assertTrue(bool((actual[:, 8:] != 0).all()))

    def test_tail_zero_positive_bias_and_clamped_weight_scales(self):
        for width in (1, 129, 2560, 7680):
            with self.subTest(width=width):
                weights = torch.arange(3 * width).reshape(3, width).remainder(3).float() - 1
                module = RowBinaryLinear(
                    weights, torch.tensor([0.2, 0.3, 0.4]), W1AxContract(1),
                    bias=torch.tensor([0.123, -0.75, 0.0]),
                )
                with torch.no_grad():
                    module.scale_offset.copy_(torch.tensor([-0.3, -0.3, 0.1]))
                x = torch.arange(2 * width).reshape(2, width).remainder(5).float() - 2
                x[0, 0] = -0.0
                actual = module(x)
                torch.testing.assert_close(actual, integer_reference(module, x), rtol=0, atol=0)
                torch.testing.assert_close(actual[:, 0], module.frozen_bias[0].expand(2),
                                           rtol=0, atol=0)

    def assert_surrogate_gradients(self, bits, *, zero_input=False, shared=False):
        generator = torch.Generator(device="cpu").manual_seed(19)
        weight = torch.randn(4, 129, generator=generator)
        weight[0, :4] = torch.tensor([-1.5, -0.0, 0.0, 1.5])
        module = RowBinaryLinear(weight, torch.tensor([0.3, 0.4, 0.5, 0.6]),
                                 W1AxContract(bits), bias=torch.tensor([0.1, 0.2, -0.4, 0.0]))
        with torch.no_grad():
            module.scale_offset.copy_(torch.tensor([-0.4, -0.4, 0.1, -0.1]))
        reference = copy.deepcopy(module)
        x = (torch.zeros(2, 129) if zero_input
             else torch.randn(2, 129, generator=generator)).requires_grad_()
        x_reference = x.detach().clone().requires_grad_()
        upstream = torch.randn(2, 4, generator=generator)
        outputs = []
        for model, inputs, forward in (
            (module, x, lambda m, v: m(v)),
            (reference, x_reference, original_surrogate),
        ):
            context = shared_round_hard_signs({"fixture": model}) if shared else nullcontext()
            with context:
                output = forward(model, inputs) + forward(model, inputs * 1.7)
                if shared:
                    self.assertIsNotNone(model._round_hard_signs)
            self.assertIsNone(model._round_hard_signs)
            output.backward(upstream)
            outputs.append(output)
        for actual, expected in ((x.grad, x_reference.grad),
                                 (module.latent_sign.grad, reference.latent_sign.grad),
                                 (module.scale_offset.grad, reference.scale_offset.grad)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(bool((module.latent_sign.grad[weight.abs() > 1] == 0).all()))
        if zero_input:
            self.assertGreater(float(x.grad.abs().sum()), 0)
        if bits != 1:
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_a1_fixed_upstream_gradients_match_original_surrogate(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                self.assert_surrogate_gradients(1, shared=shared)

    def test_zero_activation_scale_preserves_input_ste(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                self.assert_surrogate_gradients(1, zero_input=True, shared=shared)
        module = RowBinaryLinear(torch.ones(2, 2560), torch.tensor([0.2, 0.0]),
                                 W1AxContract(1))
        torch.testing.assert_close(module(torch.zeros(2560)), torch.zeros(2), rtol=0, atol=0)

    def test_other_activation_width_values_and_gradients_are_unchanged(self):
        for bits in (4, 8, 16):
            for shared in (False, True):
                with self.subTest(bits=bits, shared=shared):
                    self.assert_surrogate_gradients(bits, shared=shared)


if __name__ == "__main__":
    unittest.main()
