"""Trainable W1A1 vocabulary head for the bounded, cached-input QAT pilot.

Only the head weights are trained. The BF16 activation vectors are constants;
the target and EAGLE body are not owned by this module.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fake_binary import W1A1Config, _scale, _sign


class _WeightSignSTE(torch.autograd.Function):
    """BF16 sign forward with a clipped, row-scale-normalized FP32 backward.

    Let ``s = max(detached_row_scale, eps)``. The surrogate derivative is
    ``1 / s`` where ``abs(BF16(weight) / s) <= clip`` and zero elsewhere.
    The BF16 view and scale are fixed in backward; no gradient passes through
    the per-row magnitude scale. The default clip is 1 and eps is 1e-3.
    """

    @staticmethod
    def forward(
        ctx,
        latent_weight: Tensor,
        zero_sign: int,
        row_scale: Tensor,
        clip: float,
        eps: float,
    ) -> Tensor:
        weight_bf16 = latent_weight.to(torch.bfloat16)
        ctx.save_for_backward(weight_bf16, row_scale)
        ctx.clip = clip
        ctx.eps = eps
        return _sign(weight_bf16, zero_sign)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        weight_bf16, row_scale = ctx.saved_tensors
        denominator = row_scale.float().clamp_min(ctx.eps)
        within_clip = (weight_bf16.float().abs() / denominator) <= ctx.clip
        gradient = grad_output.float() * within_clip.float() / denominator
        return gradient, None, None, None, None


class TrainableW1A1Head(nn.Module):
    """Bias-free W1A1 head with FP32 latent weights and BF16-exact inference.

    ``dense_weight`` must be the original BF16 ``lm_head.weight``. Forward
    requires BF16 inputs and reproduces ``fake_binary_linear`` on the current
    BF16 weight view, including its multiplication order. ``export_bf16_weight``
    returns a detached copy for replacing only ``lm_head.weight`` in a fresh
    checkpoint. This module never stores or modifies the target model.
    """

    def __init__(
        self,
        dense_weight: Tensor,
        config: W1A1Config = W1A1Config(),
        *,
        ste_clip: float = 1.0,
        ste_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if dense_weight.ndim != 2 or dense_weight.dtype != torch.bfloat16:
            raise ValueError("dense_weight must be a BF16 matrix (out_features, in_features)")
        if not torch.isfinite(dense_weight).all():
            raise ValueError("dense_weight must be finite")
        if not isinstance(config, W1A1Config):
            raise TypeError("config must be a W1A1Config")
        if not (0 < ste_clip < float("inf") and 0 < ste_eps < float("inf")):
            raise ValueError("ste_clip and ste_eps must be finite positive numbers")
        self.latent_weight = nn.Parameter(dense_weight.detach().float().clone())
        self.config = config
        self.ste_clip = ste_clip
        self.ste_eps = ste_eps

    def export_bf16_weight(self) -> Tensor:
        """Return a detached BF16 copy of the head weight for inference."""
        return self.latent_weight.detach().to(torch.bfloat16).clone()

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != torch.bfloat16:
            raise ValueError("input must be BF16 cached head activations")
        if input.ndim < 1 or input.shape[-1] != self.latent_weight.shape[-1]:
            raise ValueError("input must have shape (..., in_features)")
        weight_bf16 = self.latent_weight.to(torch.bfloat16)
        weight_scale = _scale(weight_bf16.detach(), self.config.weight_scale).squeeze(-1)
        weight_sign = _WeightSignSTE.apply(
            self.latent_weight,
            self.config.zero_sign,
            weight_scale.unsqueeze(-1),
            self.ste_clip,
            self.ste_eps,
        )
        input_constant = input.detach()
        input_sign = _sign(input_constant, self.config.zero_sign)
        input_scale = _scale(input_constant, self.config.activation_scale)
        output = F.linear(input_sign, weight_sign)
        output = output * weight_scale * input_scale
        return output
