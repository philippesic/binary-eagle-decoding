"""Source-bound research adapter: reuse only immediate learned sibling groups."""

from __future__ import annotations

import inspect
import textwrap
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MethodType

import torch

from w1a1_eagle.native_step import NativeStepAdapter


@dataclass
class Group:
    label: str
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


_GROUP: ContextVar[Group | None] = ContextVar("immediate_activation_group", default=None)


@contextmanager
def immediate_group(label, events=None):
    """Release all cache-owned tensors before the next projection/step."""
    if _GROUP.get() is not None:
        raise ValueError("nested immediate activation group")
    state = Group(label)
    token = _GROUP.set(state)
    try:
        yield state
    finally:
        state.entries.clear()
        _GROUP.reset(token)
        if events is not None:
            events.append(dict(label=label, hits=state.hits, misses=state.misses, retained=0))


def _tensor_key(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        return None  # Inference tensors cannot prove mutation identity.
    return (id(tensor), version, tuple(tensor.shape), tensor.dtype, tensor.device,
            tensor.requires_grad)


def quantize(quantizer, input, **kwargs):
    """Same-N reuse; fail closed for unrecognized options or unversioned tensors."""
    original = getattr(quantizer, "_reuse_original_forward", quantizer.forward)
    state = _GROUP.get()
    if state is None:
        return original(input, **kwargs)
    if set(kwargs) - {"valid_mask", "normalization_count"}:
        return original(input, **kwargs)
    mask = kwargs.get("valid_mask")
    count = kwargs.get("normalization_count")
    if ((mask is not None and not isinstance(mask, torch.Tensor))
            or (count is not None and type(count) is not int)):
        return original(input, **kwargs)  # Preserve the original validation error.
    keys = [_tensor_key(input), _tensor_key(quantizer.parameter)]
    if mask is not None:
        keys.append(_tensor_key(mask))
    if any(key is None for key in keys):
        state.misses += 1
        return original(input, **kwargs)
    # N's default is input.numel() (or mask count); mask identity/version covers
    # its contents without an extra device-to-host reduction. Explicit N differs.
    key = (id(quantizer), type(quantizer.bits), quantizer.bits, quantizer.boundary,
           type(quantizer.in_features), quantizer.in_features,
           tuple(keys), input.numel(), count,
           torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
           torch.is_autocast_enabled(input.device.type),
           torch.get_autocast_dtype(input.device.type))
    if key in state.entries:
        state.hits += 1
        return state.entries[key][0]
    state.misses += 1
    result = original(input, **kwargs)
    # Strong references prevent id recycling until this immediate group ends.
    state.entries[key] = (result, input, quantizer, quantizer.parameter, mask)
    return result


QKV = """    q = attn.q_proj(fused)
    k = attn.k_proj(fused)
    v = attn.v_proj(fused)
"""
GATE = """    gate = mlp.gate_proj(post_attention)
    activated = (
        self.native_cpu_operators.silu(gate)
        if self.attention_mode == "native_cpu_diagnostic"
        else F.silu(gate)
    )
    ffn = mlp.down_proj(activated * mlp.up_proj(post_attention))
"""


def transformed_decode_source():
    """Reject source drift rather than silently rewriting a different graph."""
    source = textwrap.dedent(inspect.getsource(NativeStepAdapter.decode_step))
    if source.count(QKV) != 1 or source.count(GATE) != 1:
        raise RuntimeError("NativeStep immediate sibling source changed; re-review required")
    source = source.replace(
        QKV,
        "    with immediate_group('qkv', self.reuse_events):\n" + textwrap.indent(QKV, "    "),
    )
    group = GATE.replace(
        "    ffn = mlp.down_proj(activated * mlp.up_proj(post_attention))\n",
        "    up = mlp.up_proj(post_attention)\n",
    )
    return source.replace(
        GATE,
        "    with immediate_group('gate_up', self.reuse_events):\n"
        + textwrap.indent(group, "    ")
        + "    ffn = mlp.down_proj(activated * up)\n",
    )


def install_reuse(adapter):
    """Research-only binding; production source files are never modified."""
    if hasattr(adapter, "reuse_events"):
        raise ValueError("reuse already installed")
    adapter.reuse_events = []
    namespace = dict(NativeStepAdapter.decode_step.__globals__, immediate_group=immediate_group)
    exec(compile(transformed_decode_source(), "<immediate-sibling-reuse>", "exec"), namespace)
    adapter.decode_step = MethodType(namespace["decode_step"], adapter)
    seen = set()
    for module in adapter.linears.values():
        quantizer = getattr(module, "activation_quantizer", None)
        if quantizer is not None and id(quantizer) not in seen:
            seen.add(id(quantizer))
            quantizer._reuse_original_forward = quantizer.forward

            def forward(self, input, **kwargs):
                return quantize(self, input, **kwargs)

            quantizer.forward = MethodType(forward, quantizer)
    return adapter
