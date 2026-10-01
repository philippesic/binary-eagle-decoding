"""Opt-in affine two-level rows, sharing the existing quantized input boundary.

The binary dot stays unchanged: (D * alpha) * beta + (S * mu) * beta,
then frozen bias. A16 uses F16-rounded input values and omits beta. Midpoints
are F32 masters, not a second sign plane or a dense weight shadow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SCHEMA_VERSION = 1
ARITHMETIC = "integer_dot_alpha_beta_plus_integer_sum_midpoint_beta_before_bias_f32"


@dataclass(frozen=True)
class AffineBinaryConfig:
    enabled: bool = False
    coverage: str = "fusion"
    midpoint_lr: float = 1e-5
    mild_l2: float = 1e-6
    midpoint_bound: float | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or self.coverage not in ("fusion", "all"):
            raise ValueError("affine enabled must be boolean and coverage fusion or all")
        for name, value in (("midpoint_lr", self.midpoint_lr), ("mild_l2", self.mild_l2)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "midpoint_lr" and value == 0)
            ):
                requirement = "positive" if name == "midpoint_lr" else "nonnegative"
                raise ValueError(f"{name} must be finite {requirement}")
        if self.midpoint_bound is not None and (
            isinstance(self.midpoint_bound, bool)
            or not isinstance(self.midpoint_bound, (int, float))
            or not math.isfinite(self.midpoint_bound)
            or self.midpoint_bound <= 0
        ):
            raise ValueError("explicit midpoint bound must be finite positive")


class AffineBinaryMidpoint(nn.Module):
    def __init__(self, out_features: int, config: AffineBinaryConfig) -> None:
        super().__init__()
        if type(out_features) is not int or out_features < 1:
            raise ValueError("midpoint rows must be positive integer")
        self.config = config
        self.midpoint = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def validate(self) -> None:
        if self.midpoint.dtype != torch.float32 or not bool(torch.isfinite(self.midpoint).all()):
            raise ValueError("affine midpoint must remain finite F32")

    def validate_bound(self) -> None:
        self.validate()
        bound = self.config.midpoint_bound
        if bound is not None and bool((self.midpoint.abs() > bound).any()):
            raise ValueError("affine midpoint exceeds explicit bound")

    @torch.no_grad()
    def project_(self) -> None:
        self.validate()
        if self.config.midpoint_bound is not None:
            self.midpoint.clamp_(-self.config.midpoint_bound, self.config.midpoint_bound)

    def regularization_loss(self) -> Tensor:
        return self.config.mild_l2 * self.midpoint.square().sum()


class AffineInputSum(NamedTuple):
    hard: Tensor
    surrogate: Tensor


class _InputSumSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, codes, beta):
        ctx.shape = values.shape
        hard = (values if codes is None else codes.float()).sum(dim=-1, keepdim=True)
        ctx.mark_non_differentiable(hard)
        # Actual Q can differ from codes*beta in rounding/subnormal behavior.
        # Save one attached Q sum per shared boundary, rather than recomputing
        # independently for every output projection during backward.
        return hard, hard.clone() if codes is None else values.sum(dim=-1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_hard, grad_surrogate):
        return grad_surrogate.expand(ctx.shape), None, None


_SUM_CACHE: ContextVar[dict | None] = ContextVar("affine_input_sums", default=None)


def _version(tensor: Tensor) -> int | None:
    # Inference tensors have no mutation counter. Avoid caching them rather
    # than accidentally preserving a stale sum after an in-scope mutation.
    try:
        return tensor._version
    except RuntimeError:
        return None


@contextmanager
def shared_affine_input_sums():
    """One forward scope only; callers must share the actual quantizer result.

    A Q/K/V or gate/up boundary can reuse an identical values/codes/beta tuple.
    Tensor identity and mutation versions are keys. No cache persists across
    scopes or updates; unrelated quantizer results never share their gradients.
    """
    if _SUM_CACHE.get() is not None:
        raise ValueError("nested affine sum scope")
    token = _SUM_CACHE.set({})
    try:
        yield
    finally:
        _SUM_CACHE.reset(token)


def _validate_boundary(values: Tensor, codes: Tensor | None, beta: Tensor | None) -> None:
    if (codes is None) != (beta is None):
        raise ValueError("codes and beta must be supplied together; omit both for A16")
    if values.ndim < 1 or values.shape[-1] < 1 or not values.is_floating_point():
        raise ValueError("quantized values require a nonempty floating input boundary")
    if codes is not None and (
        codes.shape != values.shape
        or codes.device != values.device
        or beta.shape != (*values.shape[:-1], 1)
        or beta.device != values.device
        or beta.dtype != torch.float32
    ):
        raise ValueError("codes/beta shape or device differs from quantized values")
    if values.device.type == "cpu" and (
        not bool(torch.isfinite(values).all())
        or (
            codes is not None
            and (
                not bool(torch.isfinite(codes).all())
                or not bool(torch.isfinite(beta).all())
                or bool((beta < 0).any())
                or bool((codes.float() != codes.float().trunc()).any())
            )
        )
    ):
        raise ValueError(
            "affine sum needs finite values, integer codes and nonnegative finite beta"
        )


def affine_input_sum(
    values: Tensor,
    *,
    codes: Tensor | None = None,
    beta: Tensor | None = None,
    cache_key: tuple | None = None,
) -> AffineInputSum:
    _validate_boundary(values, codes, beta)
    tensors = (values, codes, beta)
    key = (
        "result",
        torch.is_grad_enabled(),
        values.requires_grad,
        tuple(None if t is None else (id(t), _version(t)) for t in tensors),
    )
    if cache_key is not None:
        if (
            not isinstance(cache_key, tuple)
            or not cache_key
            or not isinstance(cache_key[0], Tensor)
        ):
            raise ValueError("affine cache key must start with original source tensor")
        source = cache_key[0]
        if source.shape != values.shape or source.device != values.device:
            raise ValueError("affine cache source shape/device differs")
        # Remaining fields declare precision, quantizer identity and all its
        # parameter versions. The caller owns their semantic completeness.
        key = (
            "boundary",
            torch.is_grad_enabled(),
            values.requires_grad,
            id(source),
            _version(source),
            *(None if t is None else _version(t) for t in (values, codes, beta)),
            *cache_key[1:],
        )
        try:
            hash(key)
        except TypeError as error:
            raise ValueError("affine cache key metadata must be hashable") from error
        tensors = (*tensors, source)
    cache = _SUM_CACHE.get()
    if any(t is not None and _version(t) is None for t in tensors):
        cache = None
    if cache is not None and key in cache:
        # References in the cache prevent Python identity reuse within a scope.
        return cache[key][1]
    result = AffineInputSum(*_InputSumSTE.apply(values, codes, beta))
    if cache is not None:
        cache[key] = (tensors, result)
    return result


def _native(values, signs, alpha, midpoint, codes, beta, input_sum, bias):
    native = F.linear(values if codes is None else codes.float(), signs) * alpha
    correction = input_sum.hard * midpoint
    if beta is not None:
        native = native * beta
        correction = correction * beta
    native = native + correction
    return native if bias is None else native + bias


class _AffineSingleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, signs, alpha, midpoint, surrogate_sum, hard_sum, codes, beta, bias):
        ctx.save_for_backward(values, signs, alpha, midpoint, surrogate_sum)
        return (
            _native(
                values,
                signs,
                alpha,
                midpoint,
                codes,
                beta,
                AffineInputSum(hard_sum, surrogate_sum),
                bias,
            )
            + 0.0
        )

    @staticmethod
    def backward(ctx, grad_output):
        values, signs, alpha, midpoint, surrogate_sum = ctx.saved_tensors
        rows = values.reshape(-1, signs.shape[1])
        grad = grad_output.reshape(-1, signs.shape[0])
        grad_values = grad_signs = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_values = ((grad * alpha) @ signs).reshape_as(values)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            contraction = grad.T @ rows
            if ctx.needs_input_grad[1]:
                grad_signs = contraction * alpha[:, None]
            if ctx.needs_input_grad[2]:
                grad_alpha = torch.bmm(contraction.unsqueeze(1), signs.unsqueeze(2)).reshape_as(
                    alpha
                )
        # Use the shared actual-Q sum, never codes*beta: native A1 subnormal
        # signs can differ from a device's surrogate Q.
        grad_midpoint = (
            (grad * surrogate_sum.reshape(-1, 1)).sum(dim=0) if ctx.needs_input_grad[3] else None
        )
        grad_sum = (
            (grad * midpoint).sum(dim=-1, keepdim=True).reshape_as(surrogate_sum)
            if ctx.needs_input_grad[4]
            else None
        )
        grad_bias = grad.sum(dim=0) if ctx.needs_input_grad[8] else None
        return (
            grad_values,
            grad_signs,
            grad_alpha,
            grad_midpoint,
            grad_sum,
            None,
            None,
            None,
            grad_bias,
        )


def affine_binary_projection(
    values: Tensor,
    signs: Tensor,
    alpha: Tensor,
    midpoint: Tensor,
    *,
    codes: Tensor | None = None,
    beta: Tensor | None = None,
    bias: Tensor | None = None,
    single_forward: bool = True,
    input_sum: AffineInputSum | None = None,
    cache_key: tuple | None = None,
) -> Tensor:
    """Consume already-quantized values/codes, preserving their attached STE.

    No quantization, input rescaling, dense affine weight, or extra midpoint
    GEMM is performed. The sum's VJP goes through values, including learned
    quantizer parameters and input gradients when alpha=0 and mu is nonzero.
    Use no codes/beta for fixed A16; values must already be F16 rounded.
    """
    if (
        values.ndim < 1
        or signs.ndim != 2
        or values.shape[-1] != signs.shape[1]
        or alpha.shape != (signs.shape[0],)
        or midpoint.shape != alpha.shape
    ):
        raise ValueError("affine row/input dimensions disagree")
    if type(single_forward) is not bool:
        raise ValueError("affine single_forward must be boolean")
    if (codes is None) != (beta is None):
        raise ValueError("codes and beta must be supplied together; omit both for A16")
    if any(t.dtype != torch.float32 for t in (values, signs, alpha, midpoint)) or any(
        t.device != values.device for t in (signs, alpha, midpoint)
    ):
        raise ValueError("affine arithmetic operands must share device and be F32")
    if values.device.type == "cpu" and (
        any(not bool(torch.isfinite(t).all()) for t in (values, signs, alpha, midpoint))
        or bool((alpha < 0).any())
        or bool((signs.abs() != 1).any())
    ):
        raise ValueError("affine operands must be finite and alpha nonnegative")
    if bias is not None and (
        bias.shape != alpha.shape
        or bias.device != values.device
        or bias.dtype != torch.float32
        or (bias.device.type == "cpu" and not bool(torch.isfinite(bias).all()))
    ):
        raise ValueError("affine bias shape/device/dtype differs")
    if input_sum is not None:
        _validate_boundary(values, codes, beta)
    sums = (
        affine_input_sum(values, codes=codes, beta=beta, cache_key=cache_key)
        if input_sum is None
        else input_sum
    )
    if not isinstance(sums, AffineInputSum) or any(
        t.shape != (*values.shape[:-1], 1) or t.device != values.device or t.dtype != torch.float32
        for t in sums
    ):
        raise ValueError("affine shared sum shape/device/dtype differs")
    with torch.autocast(device_type=values.device.type, enabled=False):
        if single_forward or not torch.is_grad_enabled():
            return _AffineSingleForward.apply(
                values, signs, alpha, midpoint, sums.surrogate, sums.hard, codes, beta, bias
            )
        surrogate = F.linear(values, signs) * alpha + sums.surrogate * midpoint
        surrogate = surrogate if bias is None else surrogate + bias
        with torch.no_grad():
            native = _native(values, signs, alpha, midpoint, codes, beta, sums, bias)
        return native.detach() + (surrogate - surrogate.detach())


def _storage_keys(module: nn.Module) -> set[tuple[str, int]]:
    return {
        (str(t.device), t.untyped_storage().data_ptr())
        for t in (*module.parameters(), *module.buffers())
        if t.numel()
    }


def _protect_target(module: nn.Module, target: nn.Module) -> None:
    if not isinstance(target, nn.Module):
        raise ValueError("target required for affine alias protection")
    if {id(m) for m in module.modules()} & {id(m) for m in target.modules()} or _storage_keys(
        module
    ) & _storage_keys(target):
        raise ValueError("affine drafter module aliases target")


def _tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps({"shape": list(tensor.shape), "dtype": array.dtype.str}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + array.tobytes()).hexdigest()


class AffineBinaryBank(nn.Module):
    """Registered midpoint inventory; root owns base optimizer/checkpoint wiring."""

    def __init__(
        self,
        config: AffineBinaryConfig,
        midpoints: Mapping[str, AffineBinaryMidpoint],
        dimensions: Mapping[str, tuple[int, int]],
    ) -> None:
        super().__init__()
        self.config = config
        self.declared_paths = tuple(midpoints)
        self.midpoints = dict(midpoints)
        self.dimensions = dict(dimensions)
        # Numeric keys allow actual projection paths to contain dots.
        self.rows = nn.ModuleList(midpoints.values())

    @classmethod
    def from_attached(cls, linears):
        rows = {
            p: m.affine_binary
            for p, m in linears.items()
            if getattr(m, "affine_binary", None) is not None
        }
        if not rows or any(type(row) is not AffineBinaryMidpoint for row in rows.values()):
            raise ValueError("affine bank needs declared midpoint attachments")
        config = next(iter(rows.values())).config
        expected = set(linears) if config.coverage == "all" else {"fc"}
        if (
            not config.enabled
            or set(rows) != expected
            or any(row.config != config for row in rows.values())
        ):
            raise ValueError("affine midpoint coverage or recipe differs")
        if len({id(row.midpoint) for row in rows.values()}) != len(rows):
            raise ValueError("affine midpoint parameters alias")
        return cls(
            config, rows, {p: (linears[p].in_features, linears[p].out_features) for p in rows}
        )

    @torch.no_grad()
    def project_(self) -> None:
        for row in self.rows:
            row.project_()

    def regularization_loss(self) -> Tensor:
        zero = torch.tensor(0.0) if not self.rows else self.rows[0].midpoint.new_zeros(())
        return sum((row.regularization_loss() for row in self.rows), zero)

    def parameter_group(self, *, target: nn.Module) -> dict[str, Any] | None:
        _protect_target(self, target)
        params = list(self.parameters())
        return (
            {"params": params, "lr": self.config.midpoint_lr, "weight_decay": 0.0}
            if params
            else None
        )

    def identity(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "kind": "affine_binary_core",
            "config": asdict(self.config),
            "arithmetic": ARITHMETIC,
            "dimensions": {path: list(shape) for path, shape in self.dimensions.items()},
        }

    def state_payload(self) -> dict[str, Any]:
        for row in self.rows:
            row.validate_bound()
        state = {path: row.midpoint.detach().cpu().clone() for path, row in self.midpoints.items()}
        return {
            "identity": self.identity(),
            "state": state,
            "state_sha256": {path: _tensor_sha256(t) for path, t in state.items()},
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if (
            set(payload) != {"identity", "state", "state_sha256"}
            or payload["identity"] != self.identity()
        ):
            raise ValueError("affine state identity/fields differ")
        state, hashes = payload["state"], payload["state_sha256"]
        if (
            not isinstance(state, Mapping)
            or not isinstance(hashes, Mapping)
            or set(state) != set(self.midpoints)
            or set(hashes) != set(state)
        ):
            raise ValueError("affine state inventory differs")
        for path, tensor in state.items():
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != torch.float32
                or tensor.shape != self.midpoints[path].midpoint.shape
                or not bool(torch.isfinite(tensor).all())
                or hashes[path] != _tensor_sha256(tensor)
            ):
                raise ValueError(f"{path}: invalid affine midpoint state")
            if self.config.midpoint_bound is not None and bool(
                (tensor.abs() > self.config.midpoint_bound).any()
            ):
                raise ValueError(f"{path}: affine midpoint exceeds explicit bound")
        with torch.no_grad():
            for path, tensor in state.items():
                self.midpoints[path].midpoint.copy_(tensor)

    def native_payload(self) -> tuple[dict[str, Any] | None, dict[str, Tensor]]:
        from .recurrent_binary import CANDIDATE_D_BASE_TO_PATH

        if not self.config.enabled:
            return None, {}
        by_path = {path: base for base, path in CANDIDATE_D_BASE_TO_PATH.items()}
        tensors, arrays = {}, {}
        for path, row in self.midpoints.items():
            row.validate_bound()
            base = by_path[path]
            tensors[base] = base + ".w1ax_midpoint"
            arrays[tensors[base]] = row.midpoint.detach().cpu().contiguous().clone()
        return {
            "version": SCHEMA_VERSION,
            "coverage": self.config.coverage,
            "arithmetic": ARITHMETIC,
            "tensors": tensors,
        }, arrays

    def manifest_payload(self) -> dict[str, Any]:
        descriptor, tensors = self.native_payload()
        return {
            "identity": self.identity(),
            "native_descriptor": descriptor,
            "native_tensors": {
                name: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": _tensor_sha256(t)}
                for name, t in tensors.items()
            },
            "storage_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "export_status": "requires_native_validation" if self.config.enabled else "disabled",
        }


def install_affine_binary(
    linears: Mapping[str, nn.Module],
    *,
    target: nn.Module,
    config: AffineBinaryConfig = AffineBinaryConfig(),
) -> AffineBinaryBank:
    from .recurrent_binary import CANDIDATE_D_BASE_TO_PATH
    from .recurrent_qat import RowBinaryLinear

    expected = set(CANDIDATE_D_BASE_TO_PATH.values())
    if set(linears) != expected:
        raise ValueError("affine installation needs exactly nine declared linears")
    selected = tuple(
        path for path in linears if config.enabled and (config.coverage == "all" or path == "fc")
    )
    for path, module in linears.items():
        if not isinstance(module, RowBinaryLinear):
            raise TypeError("affine weights require row binary linears")
        _protect_target(module, target)
        if hasattr(module, "affine_binary"):
            raise ValueError(f"{path}: affine midpoint already installed")
    if len({id(module) for module in linears.values()}) != len(linears):
        raise ValueError("affine selected projection modules alias")
    rows = {
        path: AffineBinaryMidpoint(linears[path].out_features, config).to(
            linears[path].latent_sign.device
        )
        for path in selected
    }
    bank = AffineBinaryBank(
        config,
        rows,
        {path: (linears[path].in_features, linears[path].out_features) for path in selected},
    )
    for path, row in rows.items():
        linears[path].add_module("affine_binary", row)
    return bank


def validate_affine_optimizer(
    optimizer: torch.optim.Optimizer,
    bank: AffineBinaryBank,
    *,
    target: nn.Module,
    base_parameters: Iterable[nn.Parameter] = (),
) -> None:
    _protect_target(bank, target)
    expected = list(base_parameters) + list(bank.parameters())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    expected_ids, actual_ids = [id(p) for p in expected], [id(p) for p in actual]
    if (
        len(set(expected_ids)) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
        or set(expected_ids) != set(actual_ids)
    ):
        raise ValueError("optimizer must own exactly base and affine parameters once")
    target_storage = _storage_keys(target)
    if any((str(p.device), p.untyped_storage().data_ptr()) in target_storage for p in actual):
        raise ValueError("optimizer parameter aliases target storage")
