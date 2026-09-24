"""Inference-only numerical simulation of signed INT4/INT8 linear operands.

This uses FP32 PyTorch matrix multiplication on integer-valued floats. It does
not pack integers, use native INT4/INT8 kernels, or predict hardware speed.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UniformQuantConfig:
    """Symmetric signed weight and optional activation quantization widths.

    ``activation_bits=None`` leaves activations at their original numerical
    values (converted to FP32 for accumulation): W4A16 or W8A16 when the
    original drafter activations are BF16. Otherwise both operands are quantized
    independently using absmax scales along their last dimension.
    """

    weight_bits: int
    activation_bits: int | None

    def __post_init__(self) -> None:
        if type(self.weight_bits) is not int or self.weight_bits not in (4, 8):
            raise ValueError("weight_bits must be 4 or 8")
        if self.activation_bits is not None and (
            type(self.activation_bits) is not int or self.activation_bits not in (4, 8)
        ):
            raise ValueError("activation_bits must be 4, 8, or None")


def _integer_codes(values: Tensor, scale: Tensor, qmax: int) -> Tensor:
    """Nearest-even signed codes as FP32, including fixed-scale saturation.

    Zero-scale rows or vectors map to all-zero codes. An explicit scale argument
    also lets numerical tests exercise saturation independently of absmax.
    """
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(values.float() / divisor).clamp(-qmax, qmax)
    return torch.where(scale == 0, torch.zeros_like(codes), codes)


def _quantize(values: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = (1 << (bits - 1)) - 1
    scale = values.float().abs().amax(dim=-1, keepdim=True) / qmax
    return _integer_codes(values, scale, qmax), scale


def _linear_from_codes(
    input: Tensor,
    weight_codes: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    activation_bits: int | None,
) -> Tensor:
    if activation_bits is None:
        input_values = input.float()
        output = F.linear(input_values, weight_codes) * weight_scale.squeeze(-1)
    else:
        input_codes, input_scale = _quantize(input, activation_bits)
        output = F.linear(input_codes, weight_codes)
        output = output * weight_scale.squeeze(-1) * input_scale
    if bias is not None:
        output = output + bias.float()
    return output.to(input.dtype)


def fake_uniform_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    config: UniformQuantConfig = UniformQuantConfig(4, 4),
) -> Tensor:
    """Simulate per-row weight and per-vector activation INT4/INT8 linear.

    Codes, scales, matrix accumulation, and bias addition are FP32. The final
    result is cast to the input dtype. This is a post-training numerical
    simulation, not native quantized execution or a speed measurement.
    """
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("expected input (..., in_features) and weight (out_features, in_features)")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have shape (out_features,)")

    with torch.no_grad():
        weight_codes, weight_scale = _quantize(weight, config.weight_bits)
        return _linear_from_codes(input, weight_codes, weight_scale, bias, config.activation_bits)


class FakeUniformLinear(nn.Module):
    """Reversible inference wrapper retaining an existing ``nn.Linear``.

    Cached FP32 weight codes and scales are invalidated on parameter version,
    identity, storage, dtype, device, or configuration changes. Disabling the
    wrapper calls the original linear directly and exactly.
    """

    def __init__(
        self,
        linear: nn.Linear,
        config: UniformQuantConfig = UniformQuantConfig(4, 4),
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an nn.Linear")
        self.linear = linear
        self.config = config
        self.enabled = enabled
        self.register_buffer("_weight_codes", None, persistent=False)
        self.register_buffer("_weight_scale", None, persistent=False)
        self._weight_cache_key: tuple[object, ...] | None = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.linear.bias

    def forward(self, input: Tensor) -> Tensor:
        if not self.enabled:
            return self.linear(input)
        weight = self.linear.weight
        key = (
            id(weight),
            weight.data_ptr(),
            weight._version,
            weight.dtype,
            weight.device,
            self.config,
        )
        with torch.no_grad():
            if (
                self._weight_cache_key != key
                or self._weight_codes is None
                or self._weight_scale is None
                or self._weight_codes.dtype != torch.float32
                or self._weight_scale.dtype != torch.float32
                or self._weight_codes.device != weight.device
                or self._weight_scale.device != weight.device
            ):
                self._weight_codes, self._weight_scale = _quantize(weight, self.config.weight_bits)
                self._weight_cache_key = key
            return _linear_from_codes(
                input,
                self._weight_codes,
                self._weight_scale,
                self.linear.bias,
                self.config.activation_bits,
            )
