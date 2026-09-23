"""Numerical checks for the inference-only W1A1 linear simulation."""

import itertools
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle import W1A1Config, W1A1Linear, fake_binary_linear  # noqa: E402


def reference_scale(values: list[float], rule: str) -> float:
    if rule == "mean_abs":
        return sum(abs(value) for value in values) / len(values)
    if rule == "rms":
        return math.sqrt(sum(value * value for value in values) / len(values))
    return 1.0


def reference_linear(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    config: W1A1Config,
) -> torch.Tensor:
    """Scalar Python reference independent of PyTorch's matrix operation."""
    result = torch.empty((*inputs.shape[:-1], weights.shape[0]), dtype=inputs.dtype)
    for position in itertools.product(*(range(size) for size in inputs.shape[:-1])):
        values = inputs[position].tolist()
        input_scale = reference_scale(values, config.activation_scale)
        for row_index, row in enumerate(weights.tolist()):
            weight_scale = reference_scale(row, config.weight_scale)
            sign_dot = sum(
                (1 if a > 0 or (a == 0 and config.zero_sign == 1) else -1)
                * (1 if w > 0 or (w == 0 and config.zero_sign == 1) else -1)
                for a, w in zip(values, row, strict=True)
            )
            offset = 0.0 if bias is None else bias[row_index].item()
            result[position + (row_index,)] = sign_dot * input_scale * weight_scale + offset
    return result


class FakeBinaryLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.tensor(
            [[[0.0, -2.0, 1.0], [3.0, 0.0, -4.0]], [[-1.0, 5.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        self.weight = torch.tensor([[0.0, 2.0, -3.0], [-4.0, 0.0, 1.0]], dtype=torch.float64)
        self.bias = torch.tensor([0.25, -1.0], dtype=torch.float64)

    def test_batched_bias_and_scale_broadcast_match_scalar_reference(self) -> None:
        for zero_sign in (-1, 1):
            for weight_scale, activation_scale in itertools.product(
                ("mean_abs", "rms", "unit"), repeat=2
            ):
                with self.subTest(
                    zero_sign=zero_sign, weight=weight_scale, activation=activation_scale
                ):
                    config = W1A1Config(zero_sign, weight_scale, activation_scale)
                    actual = fake_binary_linear(self.input, self.weight, self.bias, config)
                    expected = reference_linear(self.input, self.weight, self.bias, config)
                    self.assertEqual(actual.shape, (2, 2, 2))
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_unbatched_without_bias(self) -> None:
        actual = fake_binary_linear(self.input[0, 0], self.weight)
        expected = reference_linear(self.input[0, 0], self.weight, None, W1A1Config())
        torch.testing.assert_close(actual, expected)

    def test_zero_sign_changes_bit_products(self) -> None:
        values = torch.tensor([0.0, 2.0], dtype=torch.float64)
        weights = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        positive = fake_binary_linear(
            values,
            weights,
            config=W1A1Config(zero_sign=1, weight_scale="unit", activation_scale="unit"),
        )
        negative = fake_binary_linear(
            values,
            weights,
            config=W1A1Config(zero_sign=-1, weight_scale="unit", activation_scale="unit"),
        )
        self.assertEqual(positive.item(), 0.0)
        self.assertEqual(negative.item(), -2.0)

    def test_wrapper_can_return_to_original_linear_without_changing_weights(self) -> None:
        linear = nn.Linear(3, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            linear.bias.copy_(self.bias)
        original_weight = linear.weight.detach().clone()
        original_bias = linear.bias.detach().clone()
        dense = linear(self.input)

        wrapper = W1A1Linear(linear)
        quantized = wrapper(self.input)
        torch.testing.assert_close(
            quantized, fake_binary_linear(self.input, self.weight, self.bias)
        )
        self.assertFalse(torch.equal(quantized, dense))

        wrapper.set_enabled(False)
        torch.testing.assert_close(wrapper(self.input), dense, rtol=0, atol=0)
        wrapper.set_enabled(True)
        torch.testing.assert_close(wrapper(self.input), quantized, rtol=0, atol=0)
        torch.testing.assert_close(linear.weight, original_weight, rtol=0, atol=0)
        torch.testing.assert_close(linear.bias, original_bias, rtol=0, atol=0)
        self.assertIs(wrapper.linear, linear)

    def test_wrapper_refreshes_cached_weight_sign_after_weight_change(self) -> None:
        linear = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(self.weight)
        wrapper = W1A1Linear(linear)
        wrapper(self.input)
        with torch.no_grad():
            linear.weight[0, 0] = -7.0
        expected = fake_binary_linear(self.input, linear.weight)
        torch.testing.assert_close(wrapper(self.input), expected)

    def test_invalid_configuration_or_shape_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            W1A1Config(zero_sign=0)
        with self.assertRaises(ValueError):
            W1A1Config(weight_scale="not_a_rule")
        with self.assertRaises(ValueError):
            fake_binary_linear(self.input, self.weight[:, :2])
        with self.assertRaises(ValueError):
            fake_binary_linear(self.input, self.weight, torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
