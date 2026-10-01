"""Optional raw-input fusion residual, with an explicit native arithmetic contract.

The binary core remains unchanged. This is a binary-core hybrid when enabled:
delta = U(V raw), with F32 master factors, hard F16 forward factors, and two
F32 reductions. Neither raw input nor the intermediate is activation-quantized.
CPU simulation is not evidence of native acceptance or accelerator throughput.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ARITHMETIC = "raw_f32_v_f16_dot_f32_u_f16_dot_f32_add_base_f32_bias_f32"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FusionCorrectionConfig:
    enabled: bool = False
    rank: int = 1
    output_bias: bool = False
    bias_bound: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.output_bias) is not bool:
            raise ValueError("correction switches must be booleans")
        if type(self.rank) is not int or self.rank not in (1, 4):
            raise ValueError("fusion correction rank must be 1 or 4")
        if not isinstance(self.bias_bound, (int, float)) or isinstance(self.bias_bound, bool):
            raise ValueError("bias bound must be a finite positive number")
        if not math.isfinite(self.bias_bound) or self.bias_bound <= 0:
            raise ValueError("bias bound must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.output_bias and not self.enabled:
            raise ValueError("output bias requires enabled fusion correction")


class _F16FactorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        rounded = value.to(torch.float16).float()
        if value.device.type == "cpu" and not bool(torch.isfinite(rounded).all()):
            raise ValueError("fusion factor is nonfinite or overflows F16")
        return rounded

    @staticmethod
    def backward(ctx, gradient: Tensor) -> Tensor:
        return gradient


def _tensor_sha256(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    # NumPy dtype strings encode width and byte order in the identity.
    array = value.numpy()
    header = json.dumps({"shape": list(value.shape), "dtype": array.dtype.str}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + array.tobytes()).hexdigest()


class FusionCorrection(nn.Module):
    """Drafter fusion-only low-rank residual, default off with no parameters.

    U is zero and V is deterministic nonzero at installation. Thus U gets a
    useful first-step gradient while the correction is exactly zero. V and raw
    input gradients through the correction become nonzero after U changes;
    nonzero first-step V/input gradients would contradict a zero U forward.
    The binary base still supplies its own raw-input surrogate gradient.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FusionCorrectionConfig = FusionCorrectionConfig(),
    ) -> None:
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (in_features, out_features)):
            raise ValueError("fusion dimensions must be positive integers")
        if config.enabled and config.rank > min(in_features, out_features):
            raise ValueError("rank exceeds fusion dimensions")
        self.in_features, self.out_features, self.config = in_features, out_features, config
        if config.enabled:
            # Local CPU generator leaves all global and accelerator RNGs untouched.
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            self.u = nn.Parameter(torch.zeros(out_features, config.rank, dtype=torch.float32))
            self.v = nn.Parameter(
                torch.randn(config.rank, in_features, generator=generator) / math.sqrt(in_features)
            )
        else:
            self.register_parameter("u", None)
            self.register_parameter("v", None)
        self.register_parameter(
            "output_bias",
            nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
            if config.output_bias
            else None,
        )

    def _validate_masters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name}: fusion master parameters must remain F32")
            if parameter.device.type == "cpu" and not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"{name}: fusion master parameter is nonfinite")

    def effective_bias(self) -> Tensor | None:
        if self.output_bias is None:
            return None
        return self.output_bias.clamp(-self.config.bias_bound, self.config.bias_bound)

    def _raw_delta(self, raw_input: Tensor) -> Tensor:
        if (
            not raw_input.is_floating_point()
            or raw_input.ndim < 1
            or raw_input.shape[-1] != self.in_features
        ):
            raise ValueError("raw fusion input must be floating point with declared width")
        if raw_input.device.type == "cpu" and not bool(torch.isfinite(raw_input).all()):
            raise ValueError("raw fusion input must be finite")
        if not self.config.enabled:
            return raw_input.new_zeros(
                (*raw_input.shape[:-1], self.out_features),
                dtype=torch.float32,
            )
        self._validate_masters()
        if raw_input.device != self.u.device:
            raise ValueError("raw input and correction must share device")
        # Disabling autocast prevents an implicit F16/BF16 intermediate/output.
        with torch.autocast(device_type=raw_input.device.type, enabled=False):
            latent = F.linear(raw_input.float(), _F16FactorSTE.apply(self.v))
            return F.linear(latent, _F16FactorSTE.apply(self.u))

    def forward(self, raw_input: Tensor) -> Tensor:
        delta = self._raw_delta(raw_input)
        bias = self.effective_bias()
        return delta if bias is None else delta + bias

    def add_to(self, raw_input: Tensor, binary_output: Tensor) -> Tensor:
        if not self.config.enabled:
            return binary_output
        delta = self._raw_delta(raw_input)
        if delta.shape != binary_output.shape or delta.device != binary_output.device:
            raise ValueError("binary output and fusion correction shape/device disagree")
        combined = binary_output.float() + delta
        bias = self.effective_bias()
        return combined if bias is None else combined + bias

    @torch.no_grad()
    def project_(self) -> None:
        """Bound bias after updates; reject invalid masters and F16 overflow."""
        self._validate_masters()
        if self.config.enabled:
            for factor in (self.u, self.v):
                if not bool(torch.isfinite(factor.to(torch.float16)).all()):
                    raise ValueError("fusion factor overflows F16")
        if self.output_bias is not None:
            self.output_bias.clamp_(-self.config.bias_bound, self.config.bias_bound)

    def identity(self) -> dict[str, Any]:
        contract = {
            "version": SCHEMA_VERSION,
            "path": "fc",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "config": asdict(self.config),
            "arithmetic": ARITHMETIC,
            "kind": "binary_core_hybrid",
        }
        contract["sha256"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return contract

    def state_payload(self) -> dict[str, Any]:
        """Exact F32 masters and strict config identity for training resume."""
        self._validate_masters()
        if self.config.enabled and any(
            not bool(torch.isfinite(t.to(torch.float16)).all()) for t in (self.u, self.v)
        ):
            raise ValueError("fusion factor overflows F16")
        state = {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}
        return {
            "identity": self.identity(),
            "state": state,
            "state_sha256": {name: _tensor_sha256(tensor) for name, tensor in state.items()},
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        """Validate every field before mutating; caller restores optimizer separately."""
        if set(payload) != {"identity", "state", "state_sha256"}:
            raise ValueError("fusion state payload fields differ")
        if payload["identity"] != self.identity():
            raise ValueError("fusion correction identity differs")
        state, hashes = payload["state"], payload["state_sha256"]
        expected = self.state_dict()
        if not isinstance(state, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("fusion state and hashes must be mappings")
        if set(state) != set(expected) or set(hashes) != set(expected):
            raise ValueError("fusion state parameter names differ")
        for name, tensor in state.items():
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"{name}: invalid fusion master tensor")
            if hashes[name] != _tensor_sha256(tensor):
                raise ValueError(f"{name}: fusion state hash differs")
            if name in ("u", "v") and not bool(torch.isfinite(tensor.to(torch.float16)).all()):
                raise ValueError(f"{name}: fusion factor overflows F16")
        self.load_state_dict(state, strict=True)

    def native_payload(self) -> tuple[dict[str, Any] | None, dict[str, Tensor]]:
        """Effective deployment tensors. Descriptor never asserts native readiness."""
        self._validate_masters()
        if not self.config.enabled:
            return None, {}
        if any(not bool(torch.isfinite(t.to(torch.float16)).all()) for t in (self.u, self.v)):
            raise ValueError("fusion factor overflows F16")
        arrays = {
            "fc.correction_u.weight": self.u.detach().cpu().to(torch.float16).contiguous(),
            "fc.correction_v.weight": self.v.detach().cpu().to(torch.float16).contiguous(),
        }
        bias = self.effective_bias()
        if bias is not None:
            arrays["fc.correction_bias"] = bias.detach().cpu().clone()
        descriptor = {
            "version": SCHEMA_VERSION,
            "rank": self.config.rank,
            "u_name": "fc.correction_u.weight",
            "v_name": "fc.correction_v.weight",
            "bias_name": None if bias is None else "fc.correction_bias",
            "bias_bound": self.config.bias_bound if bias is not None else None,
            "arithmetic": ARITHMETIC,
        }
        return descriptor, arrays

    def manifest_payload(self) -> dict[str, Any]:
        descriptor, arrays = self.native_payload()
        return {
            "identity": self.identity(),
            "native_descriptor": descriptor,
            "native_tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": _tensor_sha256(tensor),
                }
                for name, tensor in arrays.items()
            },
            "export_status": "requires_native_validation" if self.config.enabled else "disabled",
        }


def _storage_keys(module: nn.Module) -> set[tuple[str, int]]:
    return {
        (str(t.device), t.untyped_storage().data_ptr())
        for t in (*module.parameters(), *module.buffers())
        if t.numel()
    }


def validate_target_separation(module: nn.Module, target: nn.Module) -> None:
    """Protect target aliases, including views sharing a tensor storage."""
    if not isinstance(target, nn.Module):
        raise ValueError("target is required for fusion alias protection")
    target_modules = {id(m) for m in target.modules()}
    if any(id(m) in target_modules for m in module.modules()):
        raise ValueError("target owns fusion module")
    if _storage_keys(module) & _storage_keys(target):
        raise ValueError("fusion parameters/buffers alias target storage")


def _fusion_correction_hook(module, args, kwargs, output):
    # Read the registered child from the invoked module: deepcopy lanes never
    # capture or accidentally update the original correction/graph.
    if len(args) == 1 and not kwargs:
        raw = args[0]
    elif not args and set(kwargs) == {"input"}:
        raw = kwargs["input"]
    else:
        raise ValueError("fusion hook expects exactly one raw input")
    return module.fusion_correction.add_to(raw, output)


def install_fusion_correction(
    fc: nn.Module,
    *,
    target: nn.Module,
    config: FusionCorrectionConfig = FusionCorrectionConfig(),
) -> FusionCorrection:
    """Register one residual on RowBinaryLinear.fc and hook its original input.

    Root integration owns selecting exactly drafter.fc (never the vocabulary
    head), optimizer inclusion, and checkpoint fields. Registering the module
    makes its ownership and state visible to the drafter. A disabled install
    registers no trainables and no hook. Reinstallation fails closed.
    """
    from .recurrent_qat import RowBinaryLinear

    if not isinstance(fc, RowBinaryLinear):
        raise TypeError("fusion correction requires RowBinaryLinear.fc")
    validate_target_separation(fc, target)
    if hasattr(fc, "fusion_correction"):
        raise ValueError("fusion correction already attached")
    correction = FusionCorrection(fc.in_features, fc.out_features, config).to(fc.latent_sign.device)
    fc.add_module("fusion_correction", correction)
    if config.enabled:
        fc.register_forward_hook(_fusion_correction_hook, with_kwargs=True)
    return correction


def correction_parameter_group(
    correction: FusionCorrection,
    *,
    target: nn.Module,
    lr: float,
) -> dict[str, Any] | None:
    validate_target_separation(correction, target)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("fusion correction learning rate must be finite positive")
    parameters = list(correction.parameters())
    return {"params": parameters, "lr": lr, "weight_decay": 0.0} if parameters else None


def validate_correction_optimizer(
    optimizer: torch.optim.Optimizer,
    corrections: Iterable[FusionCorrection],
    *,
    base_parameters: Iterable[nn.Parameter] = (),
    target: nn.Module,
) -> None:
    """Require exact base+correction ownership once, with no target storage."""
    if not isinstance(target, nn.Module):
        raise ValueError("target is required for fusion optimizer alias protection")
    expected = list(base_parameters)
    for correction in corrections:
        validate_target_separation(correction, target)
        expected.extend(correction.parameters())
    expected_ids = [id(p) for p in expected]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    actual_ids = [id(p) for p in actual]
    if (
        len(set(expected_ids)) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected_ids)
    ):
        raise ValueError("optimizer must own exactly base and fusion parameters once")
    target_storage = _storage_keys(target)
    if any((str(p.device), p.untyped_storage().data_ptr()) in target_storage for p in actual):
        raise ValueError("optimizer parameter aliases target storage")
