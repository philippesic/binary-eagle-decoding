"""Ephemeral learned activation reuse for explicit NativeStep sibling groups.

There is no persistent tensor cache or quantizer installation. Only the exact
input and quantizer declared by an immediate QKV/gate-up scope can participate.
Unversioned inference tensors and unrecognized options take the ordinary path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableSequence, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .learned_activation import ActivationResult, LearnedActivationQuantizer


@dataclass
class ActivationReuseScope:
    boundary: str
    input: Tensor | None
    quantizer: LearnedActivationQuantizer | None
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


_SCOPE: ContextVar[ActivationReuseScope | None] = ContextVar(
    "native_activation_reuse", default=None
)


def shared_quantizer(boundary: str, quantizers: Sequence[object]) -> object | None:
    """A group is eligible only when every sibling owns one exact quantizer."""
    from .learned_activation import LearnedActivationQuantizer

    if boundary not in ("qkv", "gate_up"):
        raise ValueError("activation reuse is limited to immediate qkv/gate_up")
    expected = 3 if boundary == "qkv" else 2
    if len(quantizers) != expected:
        raise ValueError("activation reuse needs every immediate sibling")
    first = quantizers[0]
    if (
        type(first) is LearnedActivationQuantizer
        and first.boundary == boundary
        and all(quantizer is first for quantizer in quantizers)
    ):
        return first
    return None


@contextmanager
def activation_reuse_group(
    boundary: str,
    input: Tensor,
    quantizers: Sequence[object],
    *,
    enabled: bool,
    events: MutableSequence[dict] | None = None,
) -> Iterator[ActivationReuseScope | None]:
    """Release all owned results/references on normal and exceptional exit.

    Diagnostics contain counts only. A disabled or incompatible group never
    creates cache state, and must not inherit an outer enabled group's state.
    """
    if type(enabled) is not bool:
        raise ValueError("activation reuse enabled must be boolean")
    if _SCOPE.get() is not None:
        raise ValueError("nested activation reuse group")
    quantizer = shared_quantizer(boundary, quantizers) if enabled else None
    if quantizer is None:
        yield None
        return
    state = ActivationReuseScope(boundary, input, quantizer)
    token = _SCOPE.set(state)
    try:
        yield state
    finally:
        state.entries.clear()
        # Even a retained diagnostic scope must not own graph/parameter tensors.
        state.input = None
        state.quantizer = None
        _SCOPE.reset(token)
        if events is not None:
            events.append(
                {
                    "boundary": boundary,
                    "hits": state.hits,
                    "misses": state.misses,
                    "retained_entries": 0,
                }
            )


def _tensor_key(tensor: Tensor) -> tuple | None:
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (
        id(tensor),
        version,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
        tensor.requires_grad,
    )


def _result_key(result: ActivationResult) -> tuple | None:
    keys = tuple(
        _tensor_key(value)
        for value in (result.values, result.scale, result.codes, result.saturated, result.clipped)
    )
    return None if any(key is None for key in keys) else keys


def reuse_activation(
    quantizer: LearnedActivationQuantizer,
    input: Tensor,
    compute: Callable[..., ActivationResult],
    **kwargs,
) -> ActivationResult:
    """Share an attached result only under identical same-N execution state.

    Strong input/parameter/mask references prevent identity recycling. Output
    versions also guard instrumentation that mutates a returned result.
    Validation errors and unsupported kwargs remain the compute function's job.
    """
    state = _SCOPE.get()
    if state is None or quantizer is not state.quantizer or input is not state.input:
        return compute(input, quantizer.bits, quantizer.parameter, **kwargs)
    mask = kwargs.get("valid_mask")
    count = kwargs.get("normalization_count")
    if (
        set(kwargs) - {"valid_mask", "normalization_count"}
        or (mask is not None and not isinstance(mask, Tensor))
        or (count is not None and type(count) is not int)
    ):
        return compute(input, quantizer.bits, quantizer.parameter, **kwargs)
    tensors = (input, quantizer.parameter) + (() if mask is None else (mask,))
    keys = tuple(_tensor_key(tensor) for tensor in tensors)
    state.misses += 1
    if any(key is None for key in keys):
        return compute(input, quantizer.bits, quantizer.parameter, **kwargs)
    key = (
        id(quantizer),
        type(quantizer.bits),
        quantizer.bits,
        quantizer.boundary,
        type(quantizer.in_features),
        quantizer.in_features,
        quantizer.training,
        keys,
        count,
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled(input.device.type),
        torch.get_autocast_dtype(input.device.type),
    )
    cached = state.entries.get(key)
    if cached is not None and _result_key(cached[0]) == cached[1]:
        state.misses -= 1
        state.hits += 1
        return cached[0]
    result = compute(input, quantizer.bits, quantizer.parameter, **kwargs)
    result_key = _result_key(result)
    # A mutation/context change retires the previous entry; memory is bounded
    # to one result, even if a caller invokes a quantizer repeatedly in a scope.
    state.entries.clear()
    if result_key is not None:
        state.entries[key] = (result, result_key, tensors)
    return result
