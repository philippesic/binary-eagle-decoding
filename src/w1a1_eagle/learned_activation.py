"""Learned scalar activation quantizers with an explicit native contract.

A4/A8 learn a relative clip ratio, not a post-quantization amplitude. A1
learns a meanabs-relative decision threshold and retains the original amplitude.
The bank owns one parameter per input boundary; Q/K/V and gate/up share it.
This is training/reference arithmetic. Native readiness needs evidence bound
to the *current* parameter values, even when all values equal baseline defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONTRACT_VERSION = 1
CLIP_FLOOR = 2.0**-16
RECIPE = "relative_clip_lsq_a4a8_meanabs_threshold_a1_v1"
BOUNDARY_PATHS = MappingProxyType({
    "fc": ("fc",),
    "qkv": ("midlayer.self_attn.q_proj", "midlayer.self_attn.k_proj", "midlayer.self_attn.v_proj"),
    "attn_output": ("midlayer.self_attn.o_proj",),
    "gate_up": ("midlayer.mlp.gate_proj", "midlayer.mlp.up_proj"),
    "down": ("midlayer.mlp.down_proj",),
    "head": ("lm_head",),
})


@dataclass(frozen=True)
class ActivationResult:
    values: Tensor
    scale: Tensor
    codes: Tensor  # I8: +/-1 for A1, symmetric signed integer codes for A4/A8.
    saturated: Tensor  # endpoint code; different from actual clipping
    clipped: Tensor  # abs(x) > learned limit; A1 surrogate support, not clipping


def _validate_input(input: Tensor) -> Tensor:
    if not input.is_floating_point() or input.ndim < 1 or input.shape[-1] < 1:
        raise ValueError("activations need a nonempty floating-point feature dimension")
    x = input.float()
    if x.device.type == "cpu" and not bool(torch.isfinite(x).all()):
        raise ValueError("activations must remain finite after F32 cast")
    return x


def _positive_sign(x: Tensor) -> Tensor:
    # Comparisons under CUDA fast math can flush negative subnormals. A raw-bit
    # test preserves their sign while assigning both signed zeros positive.
    raw = x.contiguous().view(torch.int32)
    negative = ((raw & -2147483648) != 0) & ((raw & 2147483647) != 0)
    return torch.where(negative, -torch.ones_like(x), torch.ones_like(x))


@torch.no_grad()
def learned_activation_reference(input: Tensor, bits: int, parameter: Tensor) -> ActivationResult:
    """Hard reference: native F32 operation order, with defined overflow fallback.

    ``parameter`` is a scalar F32 threshold_delta for A1 or clip_ratio for A4/A8.
    A1: beta=F32(mean_F64(abs(x))); t=F32(delta*beta); bits=sign_F32(x-t),
    tie positive. Delta zero bypasses subtraction. A4/A8: L=F32(M*c),
    s=F32(L/qmax), u=F32(x*F32(qmax/L)), round-even and symmetric clamp.
    If the reciprocal overflows, u=F32((F64(x)/F64(L))*qmax). This versioned
    learned rule defines subnormal rows without native invalid-int sentinels.
    """
    x = _validate_input(input)
    _validate_parameter(parameter, bits)
    if parameter.device != x.device:
        raise ValueError("activation parameter and input must share device")
    if bits == 1:
        scale = x.abs().double().mean(dim=-1, keepdim=True).float()
        threshold = parameter * scale
        # F32 subtraction, not a normalized comparison: exact ties are positive.
        # torch.where also keeps the zero-delta branch raw on accelerator.
        shifted = torch.where(parameter == 0, x, x - threshold)
        codes = _positive_sign(shifted).to(torch.int8)
        clipped = shifted.abs() > scale
        saturated = torch.zeros_like(x, dtype=torch.bool)
    else:
        qmax = (1 << (bits - 1)) - 1
        absmax = x.abs().amax(dim=-1, keepdim=True)
        limit = absmax * parameter
        scale = limit / qmax
        inverse = torch.where(limit > 0, qmax / limit, torch.zeros_like(limit))
        normalized = x * inverse
        # Do not change ordinary F32 multiply ordering at round-even ties.
        if x.device.type != "cpu" or not bool(torch.isfinite(inverse).all()):
            fallback = ((x.double() / torch.where(limit > 0, limit, 1).double()) * qmax).float()
            normalized = torch.where(torch.isfinite(inverse), normalized, fallback)
        codes = torch.round(normalized).clamp(-qmax, qmax).to(torch.int8)
        saturated = codes.abs() == qmax
        clipped = x.abs() > limit
    if x.device.type == "cpu" and (not bool(torch.isfinite(scale).all()) or
                                   (bits == 1 and not bool(torch.isfinite(threshold).all()))):
        raise ValueError("learned quantizer scale or threshold overflowed F32")
    return ActivationResult(codes.float() * scale, scale, codes, saturated, clipped)


def _validate_parameter(parameter: Tensor, bits: int) -> None:
    if type(bits) is not int or bits not in (1, 4, 8):
        raise ValueError("learned activation supports A1, A4, A8 only")
    if parameter.dtype != torch.float32 or parameter.ndim != 0:
        raise ValueError("learned activation parameter must be scalar F32")
    if parameter.device.type == "cpu":
        value = float(parameter.detach())
        if not math.isfinite(value):
            raise ValueError("learned activation parameter must be finite")
        if bits != 1 and not CLIP_FLOOR <= value <= 1:
            raise ValueError("clip_ratio outside projected [2^-16, 1] range")


class _LearnedActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, parameter, bits, valid, normalization_count):
        hard = learned_activation_reference(x, bits, parameter)
        x = x.float()
        if bits == 1:
            threshold = parameter * hard.scale
            shifted = torch.where(parameter == 0, x, x - threshold)
            support = (shifted.abs() <= hard.scale) | (hard.scale == 0)
            derivative = -hard.scale * support
            normalizer = math.sqrt(normalization_count)
        else:
            qmax = (1 << (bits - 1)) - 1
            absmax = x.abs().amax(dim=-1, keepdim=True)
            limit = absmax * parameter
            support = (x.abs() <= limit)
            # dQ/dc = (M/qmax)*(q-u) inside, (M/qmax)*signed_qmax
            # outside. Algebraically use Q/c-x/c inside to avoid inf*0 on
            # subnormal reciprocal rows. Dynamic M is detached.
            derivative = torch.where(support,
                                     (hard.values - x) / parameter,
                                     hard.values / parameter)
            normalizer = math.sqrt(normalization_count * qmax)
        valid = valid.unsqueeze(-1)
        ctx.save_for_backward(support & valid, derivative * valid / normalizer)
        ctx.mark_non_differentiable(hard.scale, hard.codes, hard.saturated, hard.clipped)
        return hard.values, hard.scale, hard.codes, hard.saturated, hard.clipped

    @staticmethod
    def backward(ctx, grad_values, _scale, _codes, _saturated, _clipped):
        support, derivative = ctx.saved_tensors
        return grad_values * support, (grad_values * derivative).sum(), None, None, None


def learned_activation(input: Tensor, bits: int, parameter: Tensor, *,
                       valid_mask: Tensor | None = None,
                       normalization_count: int | None = None) -> ActivationResult:
    """Attached hard quantization with normalized LSQ/threshold surrogates.

    Input STE is one in the unclipped interval (A1: |x-t| <= meanabs), zero
    outside. Zero rows retain input derivative one, parameter derivative zero.
    Parameter gradients divide by sqrt(N*qmax) for A4/A8 or sqrt(N) for A1.
    N counts valid feature elements of this invocation; QKV consumer count is
    never included. Pass the same larger N across chunks to preserve full-batch
    normalization. Masked rows contribute neither input nor parameter gradients.
    Dynamic token scales and masks never differentiate.
    """
    x = _validate_input(input)
    _validate_parameter(parameter, bits)
    if valid_mask is None:
        valid = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        count = x.numel()
    else:
        if valid_mask.dtype != torch.bool or valid_mask.shape != x.shape[:-1] or valid_mask.device != x.device:
            raise ValueError("valid_mask must be boolean token-shape on input device")
        valid = valid_mask
        count = int(valid.sum()) * x.shape[-1]
    if normalization_count is None:
        normalization_count = max(count, 1)
    if type(normalization_count) is not int or normalization_count < max(count, 1):
        raise ValueError("normalization_count must cover valid feature elements")
    return ActivationResult(*_LearnedActivationSTE.apply(x, parameter, bits, valid, normalization_count))


@torch.no_grad()
def native_order_linear(result: ActivationResult, weight_signs: Tensor,
                        row_scales: Tensor, bias: Tensor | None = None) -> Tensor:
    """Hard reference epilogue: integer dot -> row scale -> token scale -> bias."""
    if weight_signs.dtype != torch.float32 or row_scales.dtype != torch.float32 or weight_signs.ndim != 2:
        raise ValueError("native-order weights and row scales must be F32")
    if row_scales.shape != (weight_signs.shape[0],) or result.codes.shape[-1] != weight_signs.shape[-1]:
        raise ValueError("native-order projection dimensions disagree")
    if result.codes.dtype != torch.int8 or result.scale.dtype != torch.float32:
        raise ValueError("native-order activation codes and scales must be I8/F32")
    if weight_signs.device != result.codes.device or row_scales.device != result.codes.device:
        raise ValueError("native-order projection must share activation device")
    if not bool(torch.isfinite(row_scales).all()) or bool((row_scales < 0).any()):
        raise ValueError("native-order row scales must be finite and nonnegative")
    if bias is not None and (bias.shape != row_scales.shape or bias.dtype != torch.float32 or
                             bias.device != row_scales.device or not bool(torch.isfinite(bias).all())):
        raise ValueError("native-order frozen bias must be finite F32 with row shape")
    if not bool(((weight_signs == 1) | (weight_signs == -1)).all()):
        raise ValueError("native-order weight signs must be +/-1")
    if result.codes.shape[-1] * 127 >= 2**24:
        raise ValueError("F32 reference integer dot could lose exactness")
    with torch.autocast(device_type=weight_signs.device.type, enabled=False):
        dot = F.linear(result.codes.float(), weight_signs)
    output = (dot * row_scales) * result.scale
    return output if bias is None else output + bias


class LearnedActivationQuantizer(nn.Module):
    def __init__(self, bits: int, boundary: str, in_features: int) -> None:
        super().__init__()
        if type(bits) is not int or bits not in (1, 4, 8) or boundary not in BOUNDARY_PATHS or type(in_features) is not int or in_features < 1:
            raise ValueError("invalid learned activation boundary, precision or width")
        self.bits, self.boundary, self.in_features = bits, boundary, in_features
        if bits == 1:
            self.threshold_delta = nn.Parameter(torch.tensor(0.0))
        else:
            self.clip_ratio = nn.Parameter(torch.tensor(1.0))

    @property
    def parameter(self) -> nn.Parameter:
        return self.threshold_delta if self.bits == 1 else self.clip_ratio

    def forward(self, input: Tensor, **kwargs) -> ActivationResult:
        if input.ndim < 1 or input.shape[-1] != self.in_features:
            raise ValueError("input width differs from shared boundary")
        return learned_activation(input, self.bits, self.parameter, **kwargs)

    @torch.no_grad()
    def project_(self) -> None:
        if not bool(torch.isfinite(self.parameter).all()):
            raise ValueError("nonfinite learned activation update")
        if self.bits != 1:
            self.clip_ratio.clamp_(CLIP_FLOOR, 1)

    def identity(self) -> dict:
        return {"version": CONTRACT_VERSION, "recipe": RECIPE, "bits": self.bits,
                "boundary": self.boundary, "in_features": self.in_features,
                "clip_floor": CLIP_FLOOR, "normalization": "valid_feature_count_per_invocation",
                "input_ste": "clipped_identity_zero_row_identity"}

    def get_extra_state(self):
        return self.identity()

    def set_extra_state(self, state):
        if state != self.identity():
            raise ValueError("learned activation checkpoint identity mismatch")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        name = "threshold_delta" if self.bits == 1 else "clip_ratio"
        value = state_dict.get(prefix + name)
        if state_dict.get(prefix + "_extra_state") != self.identity():
            error_msgs.append("learned activation checkpoint identity mismatch: " + prefix)
            return
        try:
            if not isinstance(value, Tensor):
                raise ValueError("learned activation parameter missing")
            _validate_parameter(value.detach().cpu(), self.bits)
        except ValueError as exc:
            error_msgs.append(str(exc))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


@dataclass(frozen=True)
class NativeActivationEvidence:
    parameters_sha256: str
    native_revision: str
    backend: str
    hardware: str
    exact_pack: bool
    max_relative_rms: float
    changed_choices_above_margin: int


class LearnedActivationBank(nn.Module):
    """Canonical six-boundary ownership, serialization, and deployment gates.

    Construct after installing row linears, attach, then explicitly include
    ``bank.parameters()`` once in the optimizer with no weight decay. Optimizer
    steps must call project_(). Attachment alone does not modify linear forward
    code or an existing optimizer allowlist. Quantizer sharing is parameter
    sharing; reuse one result per shared input if avoiding duplicate pack work.
    """
    def __init__(self, bits: int, in_features: Mapping[str, int]) -> None:
        super().__init__()
        paths = {p for consumers in BOUNDARY_PATHS.values() for p in consumers}
        if set(in_features) != paths:
            raise ValueError("learned bank requires exactly all nine projection widths")
        self.quantizers = nn.ModuleDict()
        for boundary, consumers in BOUNDARY_PATHS.items():
            widths = {in_features[p] for p in consumers}
            if len(widths) != 1:
                raise ValueError("shared boundary consumers need identical input widths")
            self.quantizers[boundary] = LearnedActivationQuantizer(bits, boundary, widths.pop())

    def identity(self) -> dict:
        return {"version": CONTRACT_VERSION, "recipe": RECIPE,
                "boundaries": {b: {**q.identity(), "consumers": list(BOUNDARY_PATHS[b])}
                               for b, q in self.quantizers.items()}}

    @classmethod
    def from_attached(cls, linears: Mapping[str, nn.Module]) -> LearnedActivationBank:
        """Recover canonical ownership references without creating parameters.

        Useful to checkpoint APIs that receive only projection modules. Requires
        the complete nine-projection graph; independent or aliased parameters at
        any of the six boundaries are rejected before returning a bank.
        """
        if set(linears) != {p for c in BOUNDARY_PATHS.values() for p in c}:
            raise ValueError("learned attachment requires exactly nine row projections")
        references = {}
        for boundary, consumers in BOUNDARY_PATHS.items():
            quantizer = getattr(linears[consumers[0]], "activation_quantizer", None)
            if type(quantizer) is not LearnedActivationQuantizer or quantizer.boundary != boundary:
                raise ValueError("unknown or mismatched attached activation quantizer")
            references[boundary] = quantizer
        if len({q.bits for q in references.values()}) != 1:
            raise ValueError("learned bank boundaries must share activation precision")
        if len({id(q.parameter) for q in references.values()}) != len(BOUNDARY_PATHS):
            raise ValueError("different activation boundaries must own distinct parameters")
        bank = cls.__new__(cls)
        nn.Module.__init__(bank)
        bank.quantizers = nn.ModuleDict(references)
        bank.validate_attachment(linears)
        return bank

    def attach(self, linears: Mapping[str, nn.Module]) -> None:
        self._validate_linears(linears)
        for boundary, consumers in BOUNDARY_PATHS.items():
            for path in consumers:
                if getattr(linears[path], "activation_quantizer", None) is not None:
                    raise ValueError("activation quantizer already attached")
        for boundary, consumers in BOUNDARY_PATHS.items():
            for path in consumers:
                linears[path].activation_quantizer = self.quantizers[boundary]

    def _validate_linears(self, linears):
        if set(linears) != {p for c in BOUNDARY_PATHS.values() for p in c}:
            raise ValueError("learned attachment requires exactly nine row projections")
        for boundary, consumers in BOUNDARY_PATHS.items():
            quantizer = self.quantizers[boundary]
            for path in consumers:
                module = linears[path]
                contract = getattr(module, "contract", None)
                if (getattr(module, "in_features", None) != quantizer.in_features or
                    getattr(contract, "activation_bits", None) != quantizer.bits or
                    getattr(contract, "scale_layout", None) != "row"):
                    raise ValueError("learned quantizer needs matching row projection contract")
                if getattr(module, "latent_sign", quantizer.parameter).device != quantizer.parameter.device:
                    raise ValueError("bank and projections must share device")

    def validate_attachment(self, linears: Mapping[str, nn.Module]) -> None:
        self._validate_linears(linears)
        for boundary, consumers in BOUNDARY_PATHS.items():
            for path in consumers:
                if getattr(linears[path], "activation_quantizer", None) is not self.quantizers[boundary]:
                    raise ValueError("shared activation parameter ownership changed")

    @torch.no_grad()
    def project_(self) -> None:
        for quantizer in self.quantizers.values():
            quantizer.project_()

    def checkpoint(self) -> dict:
        for q in self.quantizers.values():
            _validate_parameter(q.parameter.detach().cpu(), q.bits)
        return {"identity": self.identity(), "parameters": {
            b: q.parameter.detach().cpu().clone() for b, q in self.quantizers.items()}}

    @torch.no_grad()
    def load_checkpoint(self, checkpoint: Mapping) -> None:
        if set(checkpoint) != {"identity", "parameters"} or checkpoint["identity"] != self.identity():
            raise ValueError("learned bank checkpoint identity mismatch")
        parameters = checkpoint["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != set(self.quantizers):
            raise ValueError("learned bank parameter ownership mismatch")
        # Validate every entry before mutating any parameter.
        for b, q in self.quantizers.items():
            value = parameters[b]
            if not isinstance(value, Tensor):
                raise ValueError("learned bank checkpoint needs scalar F32 tensors")
            _validate_parameter(value.detach().cpu(), q.bits)
        for b, q in self.quantizers.items():
            q.parameter.copy_(parameters[b])

    def native_parameters(self) -> dict:
        """Encoding payload only; this does not assert native deployment support."""
        boundaries = {}
        for b, q in self.quantizers.items():
            _validate_parameter(q.parameter.detach().cpu(), q.bits)
            boundaries[b] = {"bits": q.bits,
                             "threshold_delta": float(q.parameter.detach()) if q.bits == 1 else 0.0,
                             "clip_ratio": float(q.parameter.detach()) if q.bits != 1 else 1.0}
        return {"version": CONTRACT_VERSION, "boundaries": boundaries}

    def parameters_sha256(self) -> str:
        encoded = json.dumps(self.native_parameters(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def require_native_ready(self, evidence: NativeActivationEvidence | None, *, backend: str) -> None:
        """Fail closed without current exact-pack and bounded native decision checks.

        A CPU check can authorize CPU only. No flag or default initialization
        authorizes CUDA. Evidence must come from the orchestrator's real native
        tests, including relative RMS <=0.10 and no changed choice margin >0.02.
        """
        if not isinstance(evidence, NativeActivationEvidence):
            raise ValueError("learned quantizer requires parameter-bound native validation")
        revision = evidence.native_revision
        if (evidence.parameters_sha256 != self.parameters_sha256() or
            backend not in ("cpu", "cuda") or evidence.backend != backend or
            not isinstance(evidence.hardware, str) or not evidence.hardware or
            not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision) or
            evidence.exact_pack is not True or type(evidence.max_relative_rms) not in (int, float) or
            not math.isfinite(evidence.max_relative_rms) or
            not 0 <= evidence.max_relative_rms <= 0.10 or
            type(evidence.changed_choices_above_margin) is not int or evidence.changed_choices_above_margin != 0):
            raise ValueError("learned quantizer native validation is missing, stale or failed")
