"""Head-only PyTorch simulation of the packed W1A1 arithmetic contract.

This is an acceptance simulation, not a packed kernel or a speed measurement.
For the EAGLE head K=2560, float32 matmul of exact +/-1 operands returns the
same integer dot as K-2*popcount(XOR): every partial integer is exactly
representable. The GGUF converter computes weight scales with float32 means;
the native CPU/CUDA activation pack uses a float64 absolute-value sum and
division, rounded once to float32. Scaling is ordered as dot * weight_scale *
token_scale in float32. The head returns float32 logits to match the native
GGML graph; AngelSlim's log-softmax, top-k selection, and d2t indexing accept
those logits. The rest of this PyTorch model remains BF16, whereas GGUF may
use F16 upstream.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapter import DrafterW1A1Adapter, _validated_linears

MAX_EXACT_INTEGER_DOT_K = 1 << 24


def native_contract_linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Simulate sign(0)=+1, native scale reductions, and F32 head output."""
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("expected input (..., in_features) and weight (out_features, in_features)")
    if weight.shape[-1] < 1 or weight.shape[-1] > MAX_EXACT_INTEGER_DOT_K:
        raise ValueError("K must be positive and within the exact F32 integer-dot range")
    if input.dtype != weight.dtype or input.dtype != torch.bfloat16:
        raise ValueError("native head simulation requires BF16 input and weight")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must have shape (out_features,)")
    weight_f32 = weight.float()
    input_f32 = input.float()
    weight_scale = weight_f32.abs().mean(dim=-1)
    input_scale = activation_scale_f64_to_f32(input_f32)
    dot = integer_sign_dot(input_f32, weight_f32)
    output = dot * weight_scale
    output = output * input_scale
    if bias is not None:
        output = output + bias.float()
    return output


def torch_sign_f32(values: Tensor) -> Tensor:
    """Return exact F32 +/-1 signs, mapping both zero encodings to +1."""
    return torch.where(values < 0, -1.0, 1.0)


def activation_scale_f64_to_f32(values: Tensor) -> Tensor:
    """Match native abs(F32) -> F64 sum/mean -> F32 token scale."""
    return values.float().abs().double().mean(dim=-1, keepdim=True).float()


def integer_sign_dot(input: Tensor, weight: Tensor) -> Tensor:
    """Compute the exact integer sign dot via F32 matmul for K <= 2**24.

    This equality holds for the tested head shape because all products and
    partial sums are integers in the exactly representable float32 range.
    CUDA callers must disable TF32 matmul in the acceptance runner.
    """
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("expected input (..., in_features) and weight (out_features, in_features)")
    if weight.shape[-1] < 1 or weight.shape[-1] > MAX_EXACT_INTEGER_DOT_K:
        raise ValueError("K must be positive and within the exact F32 integer-dot range")
    return F.linear(torch_sign_f32(input.float()), torch_sign_f32(weight.float()))


class NativeContractHead(nn.Module):
    """Reversible BF16 EAGLE head wrapper with cached F32 weight operands."""

    def __init__(self, linear: nn.Linear, *, enabled: bool = True) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be nn.Linear")
        self.linear = linear
        self.enabled = enabled
        self.register_buffer("_weight_sign", None, persistent=False)
        self.register_buffer("_weight_scale", None, persistent=False)
        self._weight_version = -1
        self._weight_object = None

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.linear.bias

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def forward(self, input: Tensor) -> Tensor:
        if not self.enabled:
            return self.linear(input)
        weight = self.linear.weight
        if input.dtype != weight.dtype or input.dtype != torch.bfloat16:
            raise ValueError("native head simulation requires BF16 input and weight")
        if weight.shape[1] < 1 or weight.shape[1] > MAX_EXACT_INTEGER_DOT_K:
            raise ValueError("K must be positive and within the exact F32 integer-dot range")
        if (
            self._weight_sign is None
            or self._weight_object is not weight
            or self._weight_version != weight._version
            or self._weight_sign.device != weight.device
        ):
            weight_f32 = weight.detach().float()
            self._weight_sign = torch_sign_f32(weight_f32)
            self._weight_scale = weight_f32.abs().mean(dim=-1)
            self._weight_version = weight._version
            self._weight_object = weight
        input_f32 = input.float()
        dot = F.linear(torch_sign_f32(input_f32), self._weight_sign)
        output = dot * self._weight_scale
        output = output * activation_scale_f64_to_f32(input_f32)
        if self.linear.bias is not None:
            output = output + self.linear.bias.float()
        return output


def install_native_contract_head(
    drafter: nn.Module, *, enabled: bool = True, target: nn.Module | None = None
) -> DrafterW1A1Adapter:
    """Install only the drafter-owned lm_head, preserving adapter safety checks."""
    pending = _validated_linears(drafter, ["lm_head"], target, "native W1A1")
    parent, name, linear = pending["lm_head"]
    wrapper = NativeContractHead(linear, enabled=enabled)
    setattr(parent, name, wrapper)
    return DrafterW1A1Adapter(drafter, {"lm_head": wrapper})
