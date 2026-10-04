"""Efficient joint W1Ax training simulation and explicit representation contracts.

The forward always uses hard binary weights and native-style activation values.
Dense F32 matmul computes their dot product; it does not predict kernel latency.
Clipped identity is the weight-sign surrogate. Activation codes use an identity
surrogate through the dequantized value; dynamic absmax/mean-absolute scales
are detached in the backward pass. A16 uses an F16
boundary cast and PyTorch's cast derivative. No accelerator is used by default.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapter import GROUP_PATHS, _validated_linears
from .recurrent_binary import (
    CANDIDATE_D_BASE_TO_PATH,
    GroupedBinaryLinear,
    hard_sign_ste,
    install_candidate_d_linears,
)
from .recurrent_loss import supported_prefix_ce
from .recurrent_trace import TraceAudit

ActivationBits = Literal[1, 4, 8, 16]
ScaleLayout = Literal["row", "group128"]


@dataclass(frozen=True)
class W1AxContract:
    activation_bits: ActivationBits
    scale_layout: ScaleLayout = "row"

    def __post_init__(self) -> None:
        if type(self.activation_bits) is not int or self.activation_bits not in (1, 4, 8, 16):
            raise ValueError("activation bits must be A1, A4, A8, or A16")
        if self.scale_layout not in ("row", "group128"):
            raise ValueError("scale layout must be row or group128")
        if self.scale_layout == "group128" and self.activation_bits != 16:
            raise ValueError("native group128 supports A16 only")

    @property
    def export_status(self) -> str:
        if self.scale_layout == "group128":
            return "candidate_d_group128_a16_exporter"
        return "row_w1ax_requires_native_validation"


class _HardActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, bits: int) -> tuple[Tensor, Tensor, Tensor]:
        x = x.float()
        if bits == 1:
            scale = x.abs().double().mean(dim=-1, keepdim=True).float()
            hard = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
            saturation = torch.zeros_like(x, dtype=torch.bool)
        else:
            qmax = (1 << (bits - 1)) - 1
            absmax = x.abs().amax(dim=-1, keepdim=True)
            scale = absmax / qmax
            # Native uses x * (qmax / absmax), then round-to-nearest-even.
            normalized = x * torch.where(absmax > 0, qmax / absmax, 0)
            hard = torch.round(normalized).clamp(-qmax, qmax)
            saturation = hard.abs() == qmax
        codes = hard.detach()
        if bits == 1:
            raw = x.contiguous().view(torch.int32)
            negative = ((raw & -2147483648) != 0) & ((raw & 2147483647) != 0)
            codes = torch.where(negative, -torch.ones_like(x), torch.ones_like(x))
        ctx.mark_non_differentiable(scale, saturation, codes)
        return hard * scale, scale, saturation, codes

    @staticmethod
    def backward(
        ctx, grad_values: Tensor, grad_scale: Tensor, grad_saturation: Tensor, grad_codes: Tensor
    ):
        # Identity through the dequantized value; dynamic scale is detached.
        return grad_values, None


def _validate_activation_input(input: Tensor) -> None:
    if not input.is_floating_point() or (
        input.device.type == "cpu" and not bool(torch.isfinite(input).all())
    ):
        raise ValueError("activations must be finite floating point")


def hard_activation(input: Tensor, bits: ActivationBits) -> tuple[Tensor, Tensor, Tensor]:
    """Return hard dequantized values, per-token scale and saturation mask."""
    if type(bits) is not int or bits not in (1, 4, 8, 16):
        raise ValueError("unsupported activation width")
    _validate_activation_input(input)
    if bits == 16:
        cast = input.float().to(torch.float16).float()
        if input.device.type == "cpu" and not bool(torch.isfinite(cast).all()):
            raise ValueError("A16 boundary cast overflow")
        return cast, torch.ones_like(cast[..., :1]), torch.zeros_like(cast, dtype=torch.bool)
    values, scale, saturation, _ = _HardActivationSTE.apply(input, bits)
    return values, scale, saturation


class _HardAffineActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, bits: int):
        # Affine v1 uses the native safe reciprocal-overflow rule; the legacy
        # symmetric quantizer remains unchanged. Only input has an identity STE.
        from .learned_activation import learned_activation_reference

        parameter = input.new_tensor(0.0 if bits == 1 else 1.0, dtype=torch.float32)
        result = learned_activation_reference(input, bits, parameter)
        ctx.mark_non_differentiable(result.scale, result.saturated, result.codes)
        return result.values, result.scale, result.saturated, result.codes

    @staticmethod
    def backward(ctx, grad_values, grad_scale, grad_saturation, grad_codes):
        return grad_values, None


def hard_activation_with_codes(input: Tensor, bits: ActivationBits):
    """One quantization call; native codes and original attached surrogate Q."""
    if bits == 16:
        values, scale, saturation = hard_activation(input, bits)
        return values, scale, saturation, None
    if type(bits) is not int or bits not in (1, 4, 8):
        raise ValueError("unsupported activation width")
    _validate_activation_input(input)
    return _HardAffineActivationSTE.apply(input, bits)


def _native_a1_projection(
    input: Tensor,
    signs: Tensor,
    weight_scales: Tensor,
    activation_scale: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """Integer-valued F32 dot, weight scale, activation scale, then bias.

    The caller must disable gradients. Raw F32 sign bits preserve negative
    subnormals even on a device that flushes floating comparisons; both signed
    zeros are positive. Do not replace the two multiplies with combined scales.
    """
    raw = input.float().contiguous().view(torch.int32)
    negative = ((raw & -2147483648) != 0) & ((raw & 2147483647) != 0)
    activation_signs = torch.where(negative, -1.0, 1.0).to(torch.float32)
    native = F.linear(activation_signs, signs) * weight_scales
    native = native * activation_scale
    return native if bias is None else native + bias


class _A1SingleForward(torch.autograd.Function):
    """Native A1 forward and the dequantized-identity surrogate VJP.

    Saving the attached inputs delegates the activation STE, inclusive latent
    clipping and scale derivative at zero to their original autograd nodes.
    The reassociated scale contraction is numerically, not bitwise, equivalent
    to the reference; callers must explicitly select this implementation.
    """

    @staticmethod
    def forward(ctx, quantized, signs, scales, input, activation_scale, bias):
        ctx.save_for_backward(quantized, signs, scales)
        # Historical native + (surrogate - surrogate.detach()) adds positive
        # zero. Preserve its signed-zero output bits without the surrogate GEMM.
        return _native_a1_projection(input, signs, scales, activation_scale, bias) + 0.0

    @staticmethod
    def backward(ctx, grad_output):
        quantized, signs, scales = ctx.saved_tensors
        rows = quantized.reshape(-1, signs.shape[1])
        grad = grad_output.reshape(-1, signs.shape[0])
        grad_quantized = grad_signs = grad_scales = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_quantized = ((grad * scales) @ signs).reshape_as(quantized)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            contraction = grad.T @ rows
            if ctx.needs_input_grad[2]:
                # Row dot lowers to batched reductions, avoiding a dense T*S
                # temporary. Never divide by scales: zero rows can revive.
                grad_scales = torch.bmm(contraction.unsqueeze(1), signs.unsqueeze(2)).reshape_as(
                    scales
                )
            if ctx.needs_input_grad[1]:
                grad_signs = contraction * scales[:, None]
        if ctx.needs_input_grad[5]:
            grad_bias = grad.sum(dim=0)
        # Raw input and detached activation scale contribute no native gradient.
        return grad_quantized, grad_signs, grad_scales, None, None, grad_bias


class _LearnedSingleForward(_A1SingleForward):
    @staticmethod
    def forward(ctx, quantized, signs, scales, codes, activation_scale, bias):
        ctx.save_for_backward(quantized, signs, scales)
        native = (F.linear(codes.float(), signs) * scales) * activation_scale
        return native + (0 if bias is None else bias)


class RowBinaryLinear(nn.Module):
    """Trainable one-bit row-scale linear for A1/A4/A8/A16 simulation.

    The symmetric default trains `latent_sign` and `scale_offset`. Explicit
    recipes can attach activation quantizers, row midpoints and an FC correction.
    Frozen biases are retained. A zero weight scale is legal and projects to zero.
    A row checkpoint from this module is not a deployable GGUF.
    `a1_computation="single_forward"` opts into a reassociated surrogate VJP;
    the default reference retains bit-exact historical training arithmetic.
    """

    def __init__(
        self,
        weight: Tensor,
        scales: Tensor,
        contract: W1AxContract,
        *,
        bias: Tensor | None = None,
        a1_computation: Literal["reference", "single_forward"] = "reference",
    ) -> None:
        super().__init__()
        if a1_computation not in ("reference", "single_forward"):
            raise ValueError("unknown A1 computation implementation")
        if contract.scale_layout != "row":
            raise ValueError("RowBinaryLinear needs row-scale contract")
        if weight.ndim != 2 or min(weight.shape) < 1 or scales.shape != (weight.shape[0],):
            raise ValueError("weight and row scale shapes disagree")
        if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(scales).all()):
            raise ValueError("initial weight and scales must be finite")
        if bool((scales < 0).any()):
            raise ValueError("weight scales must be nonnegative")
        if bias is not None and (
            bias.shape != (weight.shape[0],) or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("frozen bias shape or values invalid")
        self.contract = contract
        self.a1_computation = a1_computation
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.latent_sign = nn.Parameter(weight.detach().float().clone())
        self.register_buffer("initial_scale", scales.detach().float().clone())
        self.scale_offset = nn.Parameter(torch.zeros_like(self.initial_scale))
        self.register_buffer("frozen_bias", None if bias is None else bias.detach().float().clone())
        self.last_saturation_fraction = 0.0
        self._round_hard_signs = None

    def effective_scales(self) -> Tensor:
        raw = self.initial_scale + self.scale_offset
        return torch.where(raw >= 0, raw, torch.zeros_like(raw))

    @torch.no_grad()
    def project_scales_(self) -> None:
        raw = self.initial_scale + self.scale_offset
        if raw.device.type == "cpu" and not bool(torch.isfinite(raw).all()):
            raise ValueError("weight scales became nonfinite")
        self.scale_offset.copy_(torch.where(raw < 0, -self.initial_scale, self.scale_offset))

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError("input last dimension differs from weight")
        if input.device != self.latent_sign.device:
            raise ValueError("input and linear must share device")
        signs = (
            hard_sign_ste(self.latent_sign)
            if self._round_hard_signs is None
            else self._round_hard_signs
        )
        weight_scales = self.effective_scales()
        quantizer = getattr(self, "activation_quantizer", None)
        affine = getattr(self, "affine_binary", None)
        if affine is not None:
            from .affine_binary import affine_binary_projection

            if quantizer is None:
                values, scale, saturated, codes = hard_activation_with_codes(
                    input, self.contract.activation_bits
                )
                identity = None
            else:
                result = quantizer(input)
                values, scale, saturated, codes = (
                    result.values,
                    result.scale,
                    result.saturated,
                    result.codes,
                )
                identity = (id(quantizer), quantizer.parameter._version)
            self.last_saturation_fraction = saturated.float().mean().detach()
            return affine_binary_projection(
                values,
                signs,
                weight_scales,
                affine.midpoint,
                codes=codes,
                beta=None if codes is None else scale,
                bias=self.frozen_bias,
                single_forward=True,
                cache_key=(input, self.contract.activation_bits, identity),
            )
        if quantizer is not None:
            if quantizer.bits != self.contract.activation_bits:
                raise ValueError("learned quantizer precision differs from linear contract")
            result = quantizer(input)
            self.last_saturation_fraction = result.saturated.float().mean().detach()
            if not torch.is_grad_enabled() or self.a1_computation == "single_forward":
                return _LearnedSingleForward.apply(
                    result.values,
                    signs,
                    weight_scales,
                    result.codes,
                    result.scale,
                    self.frozen_bias,
                )
            surrogate = F.linear(result.values, signs) * weight_scales
            surrogate = surrogate + (0 if self.frozen_bias is None else self.frozen_bias)
            with torch.no_grad():
                native = (
                    F.linear(result.codes.float(), signs.detach()) * weight_scales.detach()
                ) * result.scale
                native = native + (0 if self.frozen_bias is None else self.frozen_bias)
            return native.detach() + (surrogate - surrogate.detach())
        if self.contract.activation_bits == 1 and not torch.is_grad_enabled():
            _validate_activation_input(input)
            activation_scale = input.float().abs().double().mean(dim=-1, keepdim=True).float()
            self.last_saturation_fraction = input.new_zeros((), dtype=torch.float32)
            # Prefix reconstruction needs neither the surrogate GEMM nor Q.
            return (
                _native_a1_projection(
                    input, signs, weight_scales, activation_scale, self.frozen_bias
                )
                + 0.0
            )
        quantized, activation_scale, saturated = hard_activation(
            input, self.contract.activation_bits
        )
        self.last_saturation_fraction = saturated.float().mean().detach()
        if self.contract.activation_bits == 1 and self.a1_computation == "single_forward":
            return _A1SingleForward.apply(
                quantized, signs, weight_scales, input, activation_scale, self.frozen_bias
            )
        surrogate = F.linear(quantized, signs) * weight_scales + (
            0 if self.frozen_bias is None else self.frozen_bias
        )
        if self.contract.activation_bits != 1:
            return surrogate
        # Native A1 reduces unscaled signs to an integer dot before applying
        # weight scale, then activation scale. F32 sums of +/-1 are exact for
        # the deployed widths (<2**24, F32 without autocast); scaling before the dot can leave a
        # cancellation residual that changes a later A1 sign. Keep the original
        # dequantized-value STE, including meaningful gradients at scale zero.
        with torch.no_grad():
            native = _native_a1_projection(
                input, signs.detach(), weight_scales.detach(), activation_scale, self.frozen_bias
            )
        return native.detach() + (surrogate - surrogate.detach())


@contextmanager
def shared_round_hard_signs(linears):
    """Reuse attached hard signs inside exactly one forward/backward round.

    Every recurrent call adds a gradient path to this tensor. Never reuse the
    tensor across an optimizer update or backward. Clearing it does not detach
    recurrent state/K/V; graphs own their references until backward completes.
    """
    modules = list(linears.values())
    if any(not isinstance(module, RowBinaryLinear) for module in modules):
        raise TypeError("shared signs require row W1Ax modules")
    if any(module._round_hard_signs is not None for module in modules):
        raise ValueError("nested or stale round sign cache")
    try:
        for module in modules:
            module._round_hard_signs = hard_sign_ste(module.latent_sign)
        from .affine_binary import shared_affine_input_sums

        with shared_affine_input_sums():
            yield
    finally:
        for module in modules:
            module._round_hard_signs = None


@dataclass(frozen=True)
class JointQATConfig:
    contract: W1AxContract
    device: str = "cpu"
    allow_accelerator: bool = False
    objective: Literal["hard_ce", "compact_probability"] = "hard_ce"
    sign_lr: float = 1e-4
    scale_lr: float = 1e-5
    max_grad_norm: float = 1.0
    seed: int = 0
    a1_computation: str = "reference"
    activation_quantization: str = "fixed"
    activation_lr: float = 1e-5
    binary_optimization: object | None = None
    fusion_correction: object | None = None
    fusion_lr: float = 1e-4
    depth_loss_decay: float = 1.0
    optimization_readiness: dict | None = None
    affine_weights: object | None = None
    optimize_cache: bool = False
    optimize_head: bool = False
    context_chunk_size: int = 64

    def __post_init__(self) -> None:
        if type(self.allow_accelerator) is not bool:
            raise ValueError("allow_accelerator must be an explicit boolean")
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
            object.__setattr__(self, "device", str(device))
        if device.type != "cpu" and not self.allow_accelerator:
            raise ValueError("accelerator use requires explicit allow_accelerator")
        if self.contract.scale_layout == "group128" and device.type != "cpu":
            raise ValueError("group128 reference projection is CPU-only")
        if self.objective not in ("hard_ce", "compact_probability"):
            raise ValueError("unknown joint QAT objective")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0,2**64)")
        if any(
            isinstance(x, bool) or not math.isfinite(x) or x <= 0
            for x in (
                self.sign_lr,
                self.scale_lr,
                self.max_grad_norm,
                self.activation_lr,
                self.fusion_lr,
            )
        ):
            raise ValueError("learning rates and gradient bound must be finite and positive")
        if self.a1_computation not in ("reference", "single_forward"):
            raise ValueError("unknown A1 computation implementation")
        if self.activation_quantization not in ("fixed", "learned"):
            raise ValueError("unknown activation quantization recipe")
        if (
            isinstance(self.depth_loss_decay, bool)
            or not math.isfinite(self.depth_loss_decay)
            or not 0 < self.depth_loss_decay <= 1
        ):
            raise ValueError("depth loss decay must be finite in (0,1]")
        if self.activation_quantization == "learned" and self.contract.activation_bits == 16:
            raise ValueError("learned activations support A1/A4/A8 only")
        if self.binary_optimization is not None:
            from .qat_optimization import BinaryOptimizationConfig

            if isinstance(self.binary_optimization, dict):
                object.__setattr__(
                    self,
                    "binary_optimization",
                    BinaryOptimizationConfig(**self.binary_optimization),
                )
            elif not isinstance(self.binary_optimization, BinaryOptimizationConfig):
                raise ValueError("binary optimization requires a validated recipe")
        if self.fusion_correction is not None:
            from .fusion_correction import FusionCorrectionConfig

            if isinstance(self.fusion_correction, dict):
                object.__setattr__(
                    self, "fusion_correction", FusionCorrectionConfig(**self.fusion_correction)
                )
            elif not isinstance(self.fusion_correction, FusionCorrectionConfig):
                raise ValueError("fusion correction requires a validated recipe")
        if type(self.optimize_cache) is not bool or type(self.optimize_head) is not bool:
            raise ValueError("cache/head controls must be boolean")
        if type(self.context_chunk_size) is not int or self.context_chunk_size < 1:
            raise ValueError("context chunks must be positive integers")
        if self.affine_weights is not None:
            from .affine_binary import AffineBinaryConfig

            if isinstance(self.affine_weights, dict):
                object.__setattr__(
                    self, "affine_weights", AffineBinaryConfig(**self.affine_weights)
                )
            elif not isinstance(self.affine_weights, AffineBinaryConfig):
                raise ValueError("affine weights require a validated recipe")
        if self.optimization_readiness is not None:
            import re

            value = self.optimization_readiness
            if (
                not isinstance(value, dict)
                or set(value) != {"path", "sha256"}
                or not isinstance(value["path"], str)
                or not Path(value["path"]).is_absolute()
                or not isinstance(value["sha256"], str)
                or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None
            ):
                raise ValueError("optimization readiness requires an absolute path and SHA256")
        if self.contract.scale_layout != "row" and (
            self.activation_quantization != "fixed"
            or self.binary_optimization is not None
            or self.fusion_correction is not None
            or self.affine_weights is not None
            or self.a1_computation != "reference"
        ):
            raise ValueError("optimization options require row W1Ax")


def install_joint_linears(
    drafter: nn.Module,
    target: nn.Module,
    config: JointQATConfig,
    *,
    candidate_d: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, RowBinaryLinear | GroupedBinaryLinear]:
    """Install all nine paths after the pinned official drafter has loaded.

    Row weights initialize from each original dense F16 weight: hard signs and
    F32 mean-absolute row scales. This is a fresh row representation, not a
    conversion of candidate-D group scales. The group path requires the exact
    candidate-D packed arrays and retains its existing Q/K inverse permutation.
    All paths and target aliases are checked before mutation.
    """
    if target is None:
        raise ValueError("target is required for alias protection")
    if config.contract.scale_layout == "group128":
        if candidate_d is None:
            raise ValueError("group128 initialization needs candidate-D arrays")
        if config.device != "cpu":
            raise ValueError("group128 installation is CPU-only")
        return install_candidate_d_linears(
            drafter, candidate_d, target=target, arithmetic="group_matmul"
        )
    if candidate_d is not None:
        raise ValueError("row initialization uses original dense weights, not candidate D")
    pending = _validated_linears(drafter, GROUP_PATHS.keys(), target, "joint row W1Ax")
    if set(pending) != set(CANDIDATE_D_BASE_TO_PATH.values()):
        raise ValueError("pinned drafter must expose exactly nine selected linears")
    target_parameter_ids = {id(parameter) for parameter in target.parameters()}
    replacements = {}
    for path, (_, _, linear) in pending.items():
        if any(id(parameter) in target_parameter_ids for parameter in linear.parameters()):
            raise ValueError(f"{path}: target owns a selected drafter parameter")
        weight = linear.weight.detach().to(dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(f"{path}: original dense weight is nonfinite")
        initial_scale = weight.abs().mean(dim=1)
        latent = torch.where(weight < 0, -torch.full_like(weight, 0.5), 0.5)
        bias = None if linear.bias is None else linear.bias.detach().to(torch.float32, device="cpu")
        replacements[path] = RowBinaryLinear(
            latent, initial_scale, config.contract, bias=bias, a1_computation=config.a1_computation
        ).to(config.device)
    for path, replacement in replacements.items():
        parent, name, _ = pending[path]
        setattr(parent, name, replacement)
    if config.binary_optimization is not None:
        from .qat_optimization import initialize_latents_

        initialize_latents_(replacements, config.binary_optimization)
    if config.activation_quantization == "learned":
        from .learned_activation import LearnedActivationBank

        bank = LearnedActivationBank(
            config.contract.activation_bits,
            {name: m.in_features for name, m in replacements.items()},
        )
        bank.to(config.device)
        bank.attach(replacements)
        drafter.qat_activation_bank = bank
    if config.fusion_correction is not None:
        from .fusion_correction import install_fusion_correction

        install_fusion_correction(
            replacements["fc"], target=target, config=config.fusion_correction
        )
    if config.affine_weights is not None:
        from .affine_binary import install_affine_binary

        drafter.qat_affine_bank = install_affine_binary(
            replacements, target=target, config=config.affine_weights
        )
    return replacements


def validate_joint_linears(
    linears: Mapping[str, RowBinaryLinear | GroupedBinaryLinear], config: JointQATConfig
) -> None:
    if set(linears) != set(CANDIDATE_D_BASE_TO_PATH.values()):
        raise ValueError("joint training needs exactly the nine candidate-D linears")
    expected = RowBinaryLinear if config.contract.scale_layout == "row" else GroupedBinaryLinear
    if any(not isinstance(module, expected) for module in linears.values()):
        raise TypeError("nine linears do not match declared scale layout")
    if config.contract.scale_layout == "row":
        if any(module.contract != config.contract for module in linears.values()):
            raise ValueError("row linears must share the W1Ax contract")
    elif any(module.group_size != 128 for module in linears.values()):
        raise ValueError("group contract needs group size 128")
    if any(module.latent_sign.device != torch.device(config.device) for module in linears.values()):
        raise ValueError("joint linears are not on configured device")
    attached = [getattr(module, "activation_quantizer", None) for module in linears.values()]
    if config.activation_quantization == "learned":
        from .learned_activation import LearnedActivationBank

        bank = LearnedActivationBank.from_attached(linears)
        bank.validate_attachment(linears)
    elif any(q is not None for q in attached):
        raise ValueError("undeclared learned activation parameters")
    correction = getattr(linears["fc"], "fusion_correction", None)
    if config.fusion_correction is None:
        if correction is not None:
            raise ValueError("undeclared fusion correction")
    elif correction is None or correction.config != config.fusion_correction:
        raise ValueError("fusion correction config differs from attachment")
    actual = {path for path, m in linears.items() if getattr(m, "affine_binary", None) is not None}
    expected = (
        set(linears)
        if config.affine_weights is not None
        and config.affine_weights.enabled
        and config.affine_weights.coverage == "all"
        else {"fc"}
        if config.affine_weights is not None and config.affine_weights.enabled
        else set()
    )
    if actual != expected or any(
        linears[p].affine_binary.config != config.affine_weights for p in actual
    ):
        raise ValueError("affine midpoint coverage or recipe differs")


def joint_parameter_families(linears) -> dict[str, list[nn.Parameter]]:
    """Explicit trainable ownership; shared activation parameters appear once."""
    families = {"sign": [], "scale": [], "activation": [], "fusion": [], "midpoint": []}
    seen = set()
    for path, module in linears.items():
        candidates = [("sign", module.latent_sign), ("scale", module.scale_offset)]
        quantizer = getattr(module, "activation_quantizer", None)
        correction = getattr(module, "fusion_correction", None)
        if quantizer is not None:
            from .learned_activation import LearnedActivationQuantizer

            if not isinstance(quantizer, LearnedActivationQuantizer):
                raise ValueError("unknown trainable activation module")
            candidates.extend(("activation", p) for p in quantizer.parameters())
        if correction is not None:
            from .fusion_correction import FusionCorrection

            if path != "fc" or not isinstance(correction, FusionCorrection):
                raise ValueError("correction is restricted to feature fusion")
            candidates.extend(("fusion", p) for p in correction.parameters())
        affine = getattr(module, "affine_binary", None)
        if affine is not None:
            from .affine_binary import AffineBinaryMidpoint

            if type(affine) is not AffineBinaryMidpoint:
                raise ValueError("unknown affine weight module")
            candidates.append(("midpoint", affine.midpoint))
        for family, parameter in candidates:
            if parameter.requires_grad and id(parameter) not in seen:
                families[family].append(parameter)
                seen.add(id(parameter))
    return families


def joint_optimizer(linears, config: JointQATConfig) -> torch.optim.Optimizer:
    validate_joint_linears(linears, config)
    families = joint_parameter_families(linears)
    if config.binary_optimization is not None:
        from .qat_optimization import make_binary_optimizer

        optimizer = make_binary_optimizer(linears, config.binary_optimization)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": families["sign"], "lr": config.sign_lr, "family": "sign"},
                {"params": families["scale"], "lr": config.scale_lr, "family": "scale"},
            ],
            weight_decay=0,
            foreach=False,
        )
    rates = [("activation", config.activation_lr), ("fusion", config.fusion_lr)]
    if config.affine_weights is not None:
        rates.append(("midpoint", config.affine_weights.midpoint_lr))
    for family, lr in rates:
        if families[family]:
            optimizer.add_param_group({"params": families[family], "lr": lr, "family": family})
    return optimizer


def compact_probability_loss(
    logits: Tensor, audit: TraceAudit, teacher: Mapping[str, Tensor]
) -> Tensor:
    """Conditional mapped-vocab CE with uniform unseen tail approximation.

    Teacher `draft_topk_probs` and `draft_tail_mass` are unconditional target
    softmax masses. `outside_draft_mass` remains visible and is excluded from
    the conditional objective. Mapped tail is distributed uniformly among
    draft IDs absent from top-k. This approximation must not be called full KL.
    """
    if logits.ndim != 2 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("student logits must be finite floating [rows, draft vocab]")
    n, vocab = logits.shape
    if n != len(audit.valid_mask) or vocab != sum(x >= 0 for x in audit.target_to_draft):
        raise ValueError("student logits disagree with audited rows or vocabulary")
    ids = teacher["draft_topk_ids"].to(device=logits.device)
    probs = teacher["draft_topk_probs"].to(device=logits.device)
    tail = teacher["draft_tail_mass"].to(device=logits.device)
    outside = teacher["outside_draft_mass"].to(device=logits.device)
    if ids.ndim != 2 or ids.shape != probs.shape or ids.shape[0] != n or ids.shape[1] >= vocab:
        raise ValueError("compact teacher top-k shape invalid")
    if tail.shape != (n,) or outside.shape != (n,):
        raise ValueError("compact teacher tail shape invalid")
    if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= vocab)).any()):
        raise ValueError("compact teacher IDs outside draft vocabulary")
    if bool((ids.sort(dim=1).values[:, 1:] == ids.sort(dim=1).values[:, :-1]).any()):
        raise ValueError("duplicate compact teacher IDs")
    if not all(bool(torch.isfinite(x).all()) for x in (probs, tail, outside)):
        raise ValueError("compact teacher masses must be finite")
    if bool((probs < 0).any()) or bool((tail < 0).any()) or bool((outside < 0).any()):
        raise ValueError("compact teacher masses must be nonnegative")
    total = probs.sum(dim=1) + tail + outside
    if not bool(torch.allclose(total, torch.ones_like(total), rtol=0, atol=1e-4)):
        raise ValueError("compact teacher mass does not sum to one")
    mapped = 1 - outside
    mask = torch.tensor(audit.valid_mask, dtype=torch.bool, device=logits.device) & (mapped > 0)
    if not bool(mask.any()):
        raise ValueError("no valid row with mapped teacher mass")
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.gather(1, ids.long())
    # Sum of log-probabilities over all other draft IDs; no dense teacher
    # distribution is materialized even for the full 32k vocabulary.
    tail_log_sum = logp.sum(dim=1) - selected.sum(dim=1)
    numerator = (probs * selected).sum(dim=1) + tail * tail_log_sum / (vocab - ids.shape[1])
    return -(numerator[mask] / mapped[mask]).mean()


def typed_diagnostics(counts: Mapping, values: Mapping, *, device: torch.device) -> dict:
    """Two typed transfers: exact I64 counters and loss/norm scalars.

    Never pack token/sign counts in F32: deployed layers exceed its exact
    integer range. Pre-update loss/gradient finite guards remain synchronous.
    """
    count_tensors = []
    for value in counts.values():
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim != 0 or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("diagnostic counters must be scalar exact integers")
        count_tensors.append(tensor.to(torch.int64))
    value_tensors = [
        torch.as_tensor(value, device=device, dtype=torch.float64).detach()
        for value in values.values()
    ]
    if any(value.ndim != 0 for value in value_tensors):
        raise ValueError("diagnostic values must be scalar")
    result = dict(zip(counts, torch.stack(count_tensors).cpu().tolist())) if counts else {}
    if values:
        result.update(zip(values, torch.stack(value_tensors).cpu().tolist()))
    return result


def joint_train_step(
    linears: Mapping[str, RowBinaryLinear | GroupedBinaryLinear],
    logits: Tensor,
    audit: TraceAudit,
    optimizer: torch.optim.Optimizer,
    config: JointQATConfig,
    *,
    teacher: Mapping[str, Tensor] | None = None,
) -> dict[str, float | int]:
    """Update all nine linears from an attached current-student unroll.

    Caller constructs that unroll and its cache. Later loss must retain the
    earlier state/K/V graph; this step never detaches logits. Trace auditing
    remains mandatory at capture ingestion.
    """
    validate_joint_linears(linears, config)
    if logits.device != torch.device(config.device):
        raise ValueError("logits are not on configured device")
    families = joint_parameter_families(linears)
    params = [p for family in families.values() for p in family]
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(params) or {id(p) for p in owned} != {id(p) for p in params}:
        raise ValueError("optimizer must own exactly declared binary/activation/fusion parameters")
    before_signs = [m.latent_sign.detach() < 0 for m in linears.values()]
    before_scales = [m.effective_scales().detach().clone() for m in linears.values()]
    optimizer.zero_grad(set_to_none=True)
    if config.objective == "hard_ce":
        if teacher is not None:
            raise ValueError("hard CE does not take compact teacher")
        if config.depth_loss_decay == 1:
            loss = supported_prefix_ce(logits, audit)
        else:
            from .qat_curriculum import depth_weighted_supported_ce

            loss = depth_weighted_supported_ce(
                logits, audit, tuple(range(len(audit.ce_mask))), decay=config.depth_loss_decay
            )
    else:
        if teacher is None:
            raise ValueError("compact probability objective needs teacher")
        loss = compact_probability_loss(logits, audit, teacher)
    token_loss = loss
    midpoint_penalty = sum(
        (
            m.affine_binary.regularization_loss()
            for m in linears.values()
            if getattr(m, "affine_binary", None) is not None
        ),
        loss.new_zeros(()),
    )
    if families["midpoint"]:
        loss = loss + midpoint_penalty
    if not loss.requires_grad or not bool(torch.isfinite(loss)):
        raise ValueError("joint loss must be finite and differentiable")
    loss.backward()
    active = torch.stack(
        [
            (p.grad != 0).any()
            if p.grad is not None
            else torch.zeros((), dtype=torch.bool, device=logits.device)
            for p in params
        ]
    )
    gradient_tensors = active.sum()
    finite_grads = torch.stack(
        [
            torch.isfinite(p.grad).all()
            if p.grad is not None
            else torch.ones((), dtype=torch.bool, device=logits.device)
            for p in params
        ]
    )
    if not bool(finite_grads.all()):
        raise ValueError("nonfinite joint QAT gradient")
    if config.binary_optimization is None:
        norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm, error_if_nonfinite=True)
    else:
        from .qat_optimization import transform_binary_gradients_

        gradient_metrics = transform_binary_gradients_(
            linears,
            config.binary_optimization,
            additional_parameters=families["activation"]
            + families["fusion"]
            + families["midpoint"],
        )
        norm = gradient_metrics["gradient_norm"]
    optimizer.step()
    for module in linears.values():
        # The clipped sign surrogate has zero derivative outside [-1, 1].
        # Project after AdamW so a packed +/-1 initialization cannot drift
        # permanently outside its trainable interval.
        with torch.no_grad():
            module.latent_sign.clamp_(-1, 1)
        module.project_scales_()
    projected = set()
    for module in linears.values():
        for name in ("activation_quantizer", "fusion_correction", "affine_binary"):
            child = getattr(module, name, None)
            if child is not None and id(child) not in projected:
                child.project_()
                projected.add(id(child))
    finite_parameters = torch.stack([torch.isfinite(p).all() for p in params])
    if not bool(finite_parameters.all()):
        raise ValueError("joint QAT update produced nonfinite parameters")
    sign_flips = torch.stack(
        [
            ((m.latent_sign.detach() < 0) != old).sum()
            for m, old in zip(linears.values(), before_signs)
        ]
    ).sum()
    scale_movement = torch.stack(
        [
            (m.effective_scales().detach() - old).abs().sum()
            for m, old in zip(linears.values(), before_scales)
        ]
    ).sum()
    latent_outside_clip = torch.stack(
        [(m.latent_sign.detach().abs() > 1).sum() for m in linears.values()]
    ).sum()
    saturation_mean = torch.stack(
        [
            torch.as_tensor(getattr(m, "last_saturation_fraction", 0.0), device=logits.device)
            for m in linears.values()
        ]
    ).mean()
    return typed_diagnostics(
        {
            "gradient_tensors": gradient_tensors,
            "sign_flips": sign_flips,
            "latent_outside_clip": latent_outside_clip,
        },
        {
            "loss": loss.detach(),
            "token_loss": token_loss.detach(),
            "midpoint_regularization": midpoint_penalty.detach(),
            "gradient_norm": norm,
            "scale_l1_movement": scale_movement,
            "saturation_mean": saturation_mean,
        },
        device=logits.device,
    )


def save_joint_checkpoint(
    linears: Mapping[str, RowBinaryLinear | GroupedBinaryLinear],
    config: JointQATConfig,
    base_gguf_sha256: str,
    checkpoint_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Save row-scale training state; block accidental group/row export mixups."""
    validate_joint_linears(linears, config)
    if config.contract.scale_layout != "row":
        raise ValueError("group128 uses save_training_checkpoint and its existing exporter")
    if len(base_gguf_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_gguf_sha256):
        raise ValueError("base GGUF SHA256 must be lowercase hex")
    checkpoint_path, manifest_path = Path(checkpoint_path), Path(manifest_path)
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError("checkpoint and manifest must be new paths")
    arrays: dict[str, np.ndarray] = {}
    projections = {}
    from .recurrent_training import CHECKPOINT_NAMES

    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        module = linears[path]
        name = CHECKPOINT_NAMES[base]
        arrays[name + ".latent"] = (
            module.latent_sign.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        arrays[name + ".scale"] = (
            module.effective_scales().detach().cpu().numpy().astype(np.float32, copy=True)
        )
        projections[base] = {"checkpoint_name": name, "shape": list(module.latent_sign.shape)}
    activation_quantizers = None
    if config.activation_quantization == "learned":
        from .learned_activation import LearnedActivationBank

        activation_quantizers = LearnedActivationBank.from_attached(linears).native_parameters()
    fusion_descriptor = None
    correction = getattr(linears["fc"], "fusion_correction", None)
    if correction is not None:
        fusion_descriptor, tensors = correction.native_payload()
        arrays.update(
            {name: tensor.detach().cpu().numpy().copy() for name, tensor in tensors.items()}
        )
    affine_descriptor = None
    if config.affine_weights is not None and config.affine_weights.enabled:
        from .affine_binary import AffineBinaryBank

        bank = AffineBinaryBank.from_attached(linears)
        affine_descriptor, tensors = bank.native_payload()
        arrays.update(
            {name: tensor.detach().cpu().numpy().copy() for name, tensor in tensors.items()}
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint_path, **arrays)
    hasher = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    manifest = {
        "schema_version": 2,
        "base_gguf_sha256": base_gguf_sha256,
        "checkpoint_sha256": digest,
        "scale_layout": "row",
        "activation_bits": config.contract.activation_bits,
        "activation_rule": "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive",
        "weight_rule": "hard_sign_zero_positive_clipped_identity_ste",
        "qk_row_order": "original_checkpoint",
        "export_status": config.contract.export_status,
        "objective": config.objective,
        "projections": projections,
    }
    if activation_quantizers is not None:
        manifest.update(
            schema_version=3,
            activation_rule="learned_scalar_a1_threshold_a4a8_clip_v1",
            activation_quantizers=activation_quantizers,
        )
    if fusion_descriptor is not None:
        manifest.update(
            schema_version=4,
            activation_quantizers=activation_quantizers,
            fusion_correction=fusion_descriptor,
        )
    if affine_descriptor is not None:
        manifest.update(
            schema_version=5,
            activation_quantizers=activation_quantizers,
            fusion_correction=fusion_descriptor,
            affine_weights=affine_descriptor,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "checkpoint_sha256": digest,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
