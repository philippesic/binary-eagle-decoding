"""Inference-only simulation of a linear layer with one-bit operands.

This module deliberately uses ordinary PyTorch matrix multiplication. It measures
the numerical effect of W1A1; it does not pack bits or provide a speedup.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ScaleRule = Literal["mean_abs", "rms", "unit"]


@dataclass(frozen=True)
class W1A1Config:
    """Sign and scale rules for both operands of a selected linear operation.

    A zero is mapped to ``zero_sign``. Each output row of the weight matrix gets
    one scale; each input vector (one token for sequence inputs) gets one scale.
    ``unit`` disables magnitude scaling and is useful as a quality ablation.
    """

    zero_sign: Literal[-1, 1] = 1
    weight_scale: ScaleRule = "mean_abs"
    activation_scale: ScaleRule = "mean_abs"

    def __post_init__(self) -> None:
        if self.zero_sign not in (-1, 1):
            raise ValueError("zero_sign must be -1 or 1")
        for name in ("weight_scale", "activation_scale"):
            if getattr(self, name) not in ("mean_abs", "rms", "unit"):
                raise ValueError(f"{name} must be mean_abs, rms, or unit")


def _sign(values: Tensor, zero_sign: int) -> Tensor:
    if zero_sign == 1:
        return torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    return torch.where(values > 0, torch.ones_like(values), -torch.ones_like(values))


def _scale(values: Tensor, rule: ScaleRule) -> Tensor:
    if rule == "mean_abs":
        return values.abs().mean(dim=-1, keepdim=True)
    if rule == "rms":
        return values.square().mean(dim=-1, keepdim=True).sqrt()
    return torch.ones_like(values[..., :1])


def fake_binary_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    config: W1A1Config = W1A1Config(),
) -> Tensor:
    """Approximate ``F.linear`` with sign operands and row/token scales.

    ``input`` has shape ``(..., in_features)`` and ``weight`` has shape
    ``(out_features, in_features)``. The output is floating point, with shape
    ``(..., out_features)``. This function is for post-training evaluation;
    ``torch.where`` does not supply a useful gradient through the sign rule.
    """
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("expected input (..., in_features) and weight (out_features, in_features)")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have shape (out_features,)")

    weight_sign = _sign(weight, config.zero_sign)
    input_sign = _sign(input, config.zero_sign)
    weight_scale = _scale(weight, config.weight_scale).squeeze(-1)
    input_scale = _scale(input, config.activation_scale)
    output = F.linear(input_sign, weight_sign)
    output = output * weight_scale * input_scale
    if bias is not None:
        output = output + bias
    return output


class W1A1Linear(nn.Module):
    """Reversible wrapper around an already-loaded ``nn.Linear``.

    The wrapped module and its parameters are retained by reference. Call
    ``set_enabled(False)`` to recover the ordinary ``nn.Linear`` computation.
    Inference-only weight signs and scales are cached and refreshed when the
    underlying parameter changes.
    """

    def __init__(
        self,
        linear: nn.Linear,
        config: W1A1Config = W1A1Config(),
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an nn.Linear")
        self.linear = linear
        self.config = config
        self.enabled = enabled
        self.register_buffer("_weight_sign", None, persistent=False)
        self.register_buffer("_weight_scale", None, persistent=False)
        self._weight_version = -1
        self._weight_dtype = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @property
    def weight(self) -> Tensor:
        """Preserve ``nn.Linear.weight`` reads in the pinned EAGLE forward."""
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.linear.bias

    def forward(self, input: Tensor) -> Tensor:
        if not self.enabled:
            return self.linear(input)
        weight = self.linear.weight
        if (
            self._weight_sign is None
            or self._weight_version != weight._version
            or self._weight_dtype != weight.dtype
            or self._weight_sign.device != weight.device
        ):
            self._weight_sign = _sign(weight.detach(), self.config.zero_sign)
            self._weight_scale = _scale(weight.detach(), self.config.weight_scale).squeeze(-1)
            self._weight_version = weight._version
            self._weight_dtype = weight.dtype
        input_sign = _sign(input, self.config.zero_sign)
        input_scale = _scale(input, self.config.activation_scale)
        output = F.linear(input_sign, self._weight_sign)
        output = output * self._weight_scale * input_scale
        if self.linear.bias is not None:
            output = output + self.linear.bias
        return output
