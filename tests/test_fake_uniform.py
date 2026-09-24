"""Independent scalar checks for the fake INT4/INT8 linear simulation."""

import itertools
import struct
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.fake_uniform import (  # noqa: E402
    FakeUniformLinear,
    UniformQuantConfig,
    _integer_codes,
    fake_uniform_linear,
)


def scalar_codes(values: list[float], bits: int) -> tuple[list[int], float]:
    """Python reference using nearest-even ``round`` and an absmax scale."""

    def f32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    qmax = 2 ** (bits - 1) - 1
    scale = f32(max(abs(value) for value in values) / qmax)
    if scale == 0:
        return [0] * len(values), 0.0
    return [max(-qmax, min(qmax, round(f32(value / scale)))) for value in values], scale


def scalar_linear(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    config: UniformQuantConfig,
) -> torch.Tensor:
    """Reference without PyTorch quantization or matrix multiplication."""
    result = torch.empty((*inputs.shape[:-1], weights.shape[0]), dtype=inputs.dtype)
    positions = itertools.product(*(range(size) for size in inputs.shape[:-1]))
    for position in positions:
        values = inputs[position].tolist()
        if config.activation_bits is None:
            input_codes, input_scale = values, 1.0
        else:
            input_codes, input_scale = scalar_codes(values, config.activation_bits)
        for row_index, row in enumerate(weights.tolist()):
            weight_codes, weight_scale = scalar_codes(row, config.weight_bits)
            product = sum(a * w for a, w in zip(input_codes, weight_codes, strict=True))
            offset = 0.0 if bias is None else bias[row_index].item()
            result[position + (row_index,)] = product * weight_scale * input_scale + offset
    return result


class FakeUniformLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.tensor(
            [[[0.0, -2.0, 1.0], [3.0, 0.0, -4.0]], [[-1.0, 5.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        self.weight = torch.tensor([[0.0, 2.0, -3.0], [-4.0, 0.0, 1.0]], dtype=torch.float32)
        self.bias = torch.tensor([0.25, -1.0], dtype=torch.float32)

    def test_batched_scale_broadcast_matches_scalar_reference(self) -> None:
        for weight_bits, activation_bits in itertools.product((4, 8), (4, 8, None)):
            with self.subTest(weight_bits=weight_bits, activation_bits=activation_bits):
                config = UniformQuantConfig(weight_bits, activation_bits)
                actual = fake_uniform_linear(self.input, self.weight, self.bias, config)
                expected = scalar_linear(self.input, self.weight, self.bias, config)
                self.assertEqual(actual.shape, (2, 2, 2))
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)

    def test_unbatched_and_zero_vectors(self) -> None:
        inputs = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 3.0]])
        weights = torch.tensor([[0.0, 0.0, 0.0], [-4.0, 1.0, 3.0]])
        for bits in (4, 8):
            for activation_bits in (bits, None):
                with self.subTest(bits=bits, activation_bits=activation_bits):
                    config = UniformQuantConfig(bits, activation_bits)
                    actual = fake_uniform_linear(inputs, weights, config=config)
                    expected = scalar_linear(inputs, weights, None, config)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(actual[0].tolist(), [0.0, 0.0])
                    self.assertEqual(actual[:, 0].tolist(), [0.0, 0.0])
                    one = fake_uniform_linear(inputs[1], weights, config=config)
                    torch.testing.assert_close(one, expected[1], rtol=0, atol=2e-5)

    def test_nearest_even_and_saturation_at_fixed_scale(self) -> None:
        values = torch.tensor([[0.5, 1.5, 2.5, -2.5, 200.0, -200.0], [0.0] * 6])
        scale = torch.tensor([[1.0], [0.0]])
        for bits, qmax in ((4, 7), (8, 127)):
            with self.subTest(bits=bits):
                actual = _integer_codes(values, scale, qmax)
                self.assertEqual(actual[0].tolist(), [0.0, 2.0, 2.0, -2.0, qmax, -qmax])
                self.assertEqual(actual[1].tolist(), [0.0] * 6)
                self.assertTrue(torch.isfinite(actual).all())

    def test_bfloat16_result_and_weight_cache_refresh(self) -> None:
        linear = nn.Linear(3, 2, bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            linear.bias.copy_(self.bias)
        inputs = self.input.to(torch.bfloat16)
        wrapper = FakeUniformLinear(linear, UniformQuantConfig(4, 4))
        initial = wrapper(inputs)
        self.assertEqual(initial.dtype, torch.bfloat16)
        torch.testing.assert_close(
            initial, fake_uniform_linear(inputs, linear.weight, linear.bias, wrapper.config)
        )
        with torch.no_grad():
            linear.weight[0, 0] = 6.0
        torch.testing.assert_close(
            wrapper(inputs), fake_uniform_linear(inputs, linear.weight, linear.bias, wrapper.config)
        )
        wrapper.config = UniformQuantConfig(8, None)
        torch.testing.assert_close(
            wrapper(inputs), fake_uniform_linear(inputs, linear.weight, linear.bias, wrapper.config)
        )
        wrapper.to(dtype=torch.float32)
        self.assertEqual(wrapper.weight.dtype, torch.float32)
        self.assertEqual(wrapper(inputs.float()).dtype, torch.float32)
        torch.testing.assert_close(
            wrapper(inputs.float()),
            fake_uniform_linear(inputs.float(), linear.weight, linear.bias, wrapper.config),
        )

        replacement = nn.Parameter(torch.tensor([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]))
        linear.weight = replacement
        torch.testing.assert_close(
            wrapper(inputs.float()),
            fake_uniform_linear(inputs.float(), replacement, linear.bias, wrapper.config),
        )

    def test_wrapper_disable_reenable_has_exact_dense_parity(self) -> None:
        linear = nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            linear.bias.copy_(self.bias)
        weight_before = linear.weight.detach().clone()
        bias_before = linear.bias.detach().clone()
        wrapper = FakeUniformLinear(linear, UniformQuantConfig(4, 4))
        quantized = wrapper(self.input)
        dense = linear(self.input)
        self.assertFalse(torch.equal(quantized, dense))
        wrapper.set_enabled(False)
        torch.testing.assert_close(wrapper(self.input), dense, rtol=0, atol=0)
        wrapper.set_enabled(True)
        torch.testing.assert_close(wrapper(self.input), quantized, rtol=0, atol=0)
        self.assertIs(wrapper.weight, linear.weight)
        self.assertIs(wrapper.bias, linear.bias)
        torch.testing.assert_close(wrapper.weight, weight_before, rtol=0, atol=0)
        torch.testing.assert_close(wrapper.bias, bias_before, rtol=0, atol=0)

    def test_invalid_config_and_shapes(self) -> None:
        with self.assertRaises(ValueError):
            UniformQuantConfig(3, 4)
        with self.assertRaises(ValueError):
            UniformQuantConfig(4, 3)
        with self.assertRaises(ValueError):
            UniformQuantConfig(4.0, 4)
        with self.assertRaises(FrozenInstanceError):
            UniformQuantConfig(4, 4).weight_bits = 8
        with self.assertRaises(ValueError):
            fake_uniform_linear(self.input, self.weight[:, :2])
        with self.assertRaises(ValueError):
            fake_uniform_linear(self.input, self.weight, torch.zeros(3))
        with self.assertRaises(TypeError):
            FakeUniformLinear(nn.Identity())


if __name__ == "__main__":
    unittest.main()
