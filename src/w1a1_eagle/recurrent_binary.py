"""CPU reference for trainable W1A16 grouped binary draft linears.

This is a correctness reference, not a fast kernel or a complete recurrent
drafter. Every forward uses hard signs and nonnegative group scales. Latent F32
weights exist only for optimization; ``export_arrays`` exposes little-endian
I32 signs and F32 scales in the module's row order. The GGUF exporter must
apply its Q/K RoPE row permutation after a PyTorch training checkpoint.

The sign surrogate is clipped identity: d sign(w) / d w = 1 for |w| <= 1,
and zero outside. It changes only the backward pass. The activation cast
F32 -> F16 -> F32 is included in the graph, with PyTorch's cast derivative.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .adapter import GROUP_PATHS, _validated_linears

CANDIDATE_D_BASE_TO_PATH = {
    "fc": "fc",
    "blk.0.attn_q": "midlayer.self_attn.q_proj",
    "blk.0.attn_k": "midlayer.self_attn.k_proj",
    "blk.0.attn_v": "midlayer.self_attn.v_proj",
    "blk.0.attn_output": "midlayer.self_attn.o_proj",
    "blk.0.ffn_gate": "midlayer.mlp.gate_proj",
    "blk.0.ffn_up": "midlayer.mlp.up_proj",
    "blk.0.ffn_down": "midlayer.mlp.down_proj",
    "output": "lm_head",
}


class _HardSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor) -> Tensor:
        ctx.save_for_backward(weight)
        # Both +0 and -0 have the positive sign in the frozen export contract.
        return torch.where(weight < 0, -torch.ones_like(weight), torch.ones_like(weight))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (weight,) = ctx.saved_tensors
        return grad_output * (weight.abs() <= 1).to(grad_output.dtype)


def hard_sign_ste(weight: Tensor) -> Tensor:
    """Hard +/-1 F32 forward, clipped-identity surrogate backward."""
    if weight.dtype != torch.float32:
        raise ValueError("latent signs must be F32")
    return _HardSignSTE.apply(weight)


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """Pack row-major +/-1 signs into little-endian 32-bit GGUF words.

    A positive sign sets its bit. Tail bits are zero. The returned dtype is
    little-endian signed I32, matching ``.w1a1_packed`` in candidate D.
    """
    signs = np.asarray(signs)
    if signs.ndim != 2 or signs.shape[1] < 1:
        raise ValueError("signs must have shape (out_features, positive in_features)")
    if not np.all((signs == 1) | (signs == -1)):
        raise ValueError("signs must contain only +1 or -1")
    positive = signs > 0
    tail = (-positive.shape[1]) % 32
    if tail:
        positive = np.pad(positive, ((0, 0), (0, tail)))
    return np.ascontiguousarray(np.packbits(positive, axis=1, bitorder="little")).view("<i4")


def unpack_signs(packed: np.ndarray, in_features: int) -> np.ndarray:
    """Decode packed signs; reject nonzero unused bits."""
    packed = np.asarray(packed)
    if in_features < 1 or packed.ndim != 2 or packed.shape[1] != math.ceil(in_features / 32):
        raise ValueError("packed shape does not match in_features")
    if packed.dtype != np.dtype("<i4"):
        raise ValueError("packed signs must be little-endian I32")
    bits = np.unpackbits(np.ascontiguousarray(packed).view(np.uint8), axis=1, bitorder="little")
    if np.any(bits[:, in_features:]):
        raise ValueError("packed tail bits must be zero")
    return np.where(bits[:, :in_features] != 0, 1, -1).astype(np.float32)


def _check_scales(scales: np.ndarray, out_features: int, in_features: int, group_size: int) -> None:
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if scales.shape != (out_features, math.ceil(in_features / group_size)):
        raise ValueError("scales must have one value per output row and input group")
    if not np.isfinite(scales).all() or not np.all(scales >= 0):
        raise ValueError("scales must be finite and nonnegative")


class GroupedBinaryLinear(nn.Module):
    """Trainable CPU W1A16 linear with hard signs and groupwise F32 scales.

    Input of any leading shape is cast to F16, then F32. For each output row,
    input features are added sequentially within each group in F32. Each group
    sum is multiplied by its F32 scale, then products are added in group order
    in F32. Zero scales are valid in candidate D; a trainable additive F32
    offset is clamped in forward and can be projected after each optimizer
    step. No dense floating weight is used by the forward pass.
    """

    def __init__(
        self,
        weight: Tensor,
        scales: Tensor,
        *,
        group_size: int = 128,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if weight.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("GroupedBinaryLinear is CPU-only")
        if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] < 1:
            raise ValueError("weight must have shape (out_features, in_features)")
        if not torch.isfinite(weight).all():
            raise ValueError("weight must be finite")
        initial_scales = scales.detach().to(dtype=torch.float32, device="cpu")
        _check_scales(initial_scales.numpy(), weight.shape[0], weight.shape[1], group_size)
        self.group_size = group_size
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.latent_sign = nn.Parameter(
            weight.detach().to(dtype=torch.float32, device="cpu").clone()
        )
        # Additive zero offset gives exactly the supplied F32 scales, including
        # exact zeros, on the first forward/export.
        self.register_buffer("initial_scale", initial_scales.clone())
        self.scale_offset = nn.Parameter(torch.zeros_like(initial_scales))
        if bias is not None:
            if bias.device.type != "cpu" or bias.shape != (self.out_features,):
                raise ValueError("bias must be a CPU vector of out_features")
            if not torch.isfinite(bias).all():
                raise ValueError("bias must be finite")
            self.register_buffer("frozen_bias", bias.detach().to(torch.float32).clone())
        else:
            self.register_buffer("frozen_bias", None)

    @classmethod
    def from_packed(
        cls,
        packed: np.ndarray,
        scales: np.ndarray,
        *,
        in_features: int,
        group_size: int = 128,
        bias: Tensor | None = None,
    ) -> GroupedBinaryLinear:
        """Initialize directly from candidate-D arrays, without F16 signs."""
        signs = unpack_signs(packed, in_features)
        scales = np.asarray(scales)
        if scales.dtype != np.float32:
            raise ValueError("candidate-D scales must be F32")
        return cls(
            torch.from_numpy(signs.copy()),
            torch.from_numpy(np.ascontiguousarray(scales).copy()),
            group_size=group_size,
            bias=bias,
        )

    @property
    def bias(self) -> Tensor | None:
        """Frozen original bias, if present; deployment must retain it."""
        return self.frozen_bias

    def effective_scales(self) -> Tensor:
        """Nonnegative F32 scales; the surrogate derivative at exact zero is one."""
        raw = self.initial_scale + self.scale_offset
        return torch.where(raw >= 0, raw, torch.zeros_like(raw))

    @torch.no_grad()
    def project_scales_(self) -> None:
        """Project additive scales after an optimizer step, including zero."""
        raw = self.initial_scale + self.scale_offset
        if not torch.isfinite(raw).all():
            raise ValueError("trained scales must be finite")
        self.scale_offset.copy_(torch.where(raw < 0, -self.initial_scale, self.scale_offset))

    def forward(self, input: Tensor) -> Tensor:
        if (
            input.device.type != "cpu"
            or self.latent_sign.device.type != "cpu"
            or self.scale_offset.device.type != "cpu"
            or self.initial_scale.device.type != "cpu"
        ):
            raise ValueError("GroupedBinaryLinear forward is CPU-only")
        if input.ndim < 1 or input.shape[-1] != self.in_features:
            raise ValueError("input last dimension must equal in_features")
        if not torch.is_floating_point(input):
            raise ValueError("input must be floating point")
        x = input.to(torch.float16).to(torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("post-F16-cast input must be finite")
        if not torch.isfinite(self.latent_sign).all():
            raise ValueError("latent signs must be finite")
        signs = hard_sign_ste(self.latent_sign)
        scales = self.effective_scales()
        if not torch.isfinite(scales).all():
            raise ValueError("trained scales must be finite")
        result = torch.zeros((*x.shape[:-1], self.out_features), dtype=torch.float32, device="cpu")
        for group, start in enumerate(range(0, self.in_features, self.group_size)):
            subtotal = torch.zeros_like(result)
            for feature in range(start, min(start + self.group_size, self.in_features)):
                subtotal = subtotal + x[..., feature, None] * signs[:, feature]
            result = result + subtotal * scales[:, group]
        if self.frozen_bias is not None:
            result = result + self.frozen_bias
        return result

    @torch.no_grad()
    def export_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return I32 packed signs and F32 scales in the module's row order.

        A Q/K module installed from candidate D has PyTorch checkpoint row
        order. The GGUF exporter applies native Q/K RoPE row permutation.
        """
        if self.latent_sign.device.type != "cpu" or self.scale_offset.device.type != "cpu":
            raise ValueError("GroupedBinaryLinear export is CPU-only")
        if not torch.isfinite(self.latent_sign).all():
            raise ValueError("latent signs must be finite")
        signs = torch.where(self.latent_sign < 0, -1, 1).numpy()
        scales = self.effective_scales().numpy().astype(np.float32, copy=True)
        _check_scales(scales, self.out_features, self.in_features, self.group_size)
        return pack_signs(signs), scales

    @torch.no_grad()
    def training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy F32 latent signs and effective scales in PyTorch row order.

        These arrays are for a CPU training checkpoint or the later GGUF
        exporter input. They are not a deployable artifact: ``export_arrays``
        is the packed inference representation. If installed from candidate D,
        Q/K rows have already been restored to original checkpoint order.
        """
        if self.latent_sign.device.type != "cpu" or self.scale_offset.device.type != "cpu":
            raise ValueError("GroupedBinaryLinear training export is CPU-only")
        latent = self.latent_sign.detach().numpy().astype(np.float32, copy=True)
        scales = self.effective_scales().detach().numpy().astype(np.float32, copy=True)
        if not np.isfinite(latent).all():
            raise ValueError("latent signs must be finite")
        _check_scales(scales, self.out_features, self.in_features, self.group_size)
        return latent, scales


def replay_packed_linear(
    input: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    in_features: int,
    *,
    group_size: int = 128,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Independent NumPy scalar replay of the candidate-D ordered F32 math."""
    x = np.asarray(input)
    if x.ndim < 1 or x.shape[-1] != in_features or not np.issubdtype(x.dtype, np.floating):
        raise ValueError("input must be floating with matching last dimension")
    signs = unpack_signs(packed, in_features)
    scales = np.asarray(scales)
    if scales.dtype != np.float32:
        raise ValueError("scales must be F32")
    _check_scales(scales, signs.shape[0], in_features, group_size)
    cast = x.astype(np.float16).astype(np.float32)
    if not np.isfinite(cast).all():
        raise ValueError("post-F16-cast input must be finite")
    flattened = cast.reshape(-1, in_features)
    output = np.zeros((len(flattened), len(signs)), dtype=np.float32)
    for row in range(len(flattened)):
        for column in range(len(signs)):
            total = np.float32(0)
            for group, start in enumerate(range(0, in_features, group_size)):
                subtotal = np.float32(0)
                for feature in range(start, min(start + group_size, in_features)):
                    product = np.float32(flattened[row, feature] * signs[column, feature])
                    subtotal = np.float32(subtotal + product)
                total = np.float32(total + np.float32(subtotal * scales[column, group]))
            output[row, column] = total
    if bias is not None:
        bias = np.asarray(bias)
        if bias.shape != (len(signs),) or bias.dtype != np.float32:
            raise ValueError("bias must be F32 with one value per output row")
        output += bias
    return output.reshape((*x.shape[:-1], len(signs)))


def _native_qk_to_python_rows(array: np.ndarray, heads: int) -> np.ndarray:
    """Undo candidate-D GGUF Q/K RoPE row order for the PyTorch drafter."""
    count = array.shape[0]
    if count % (2 * heads):
        raise ValueError("Q/K row count is incompatible with pinned head count")
    tail = array.shape[1:]
    return array.reshape(heads, count // heads // 2, 2, *tail).swapaxes(1, 2).reshape(count, *tail)


def install_candidate_d_linears(
    drafter: nn.Module,
    candidate_d: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    target: nn.Module,
) -> dict[str, GroupedBinaryLinear]:
    """Replace the nine drafter-owned linears from candidate-D GGUF arrays.

    ``candidate_d`` maps the nine GGUF base names in
    ``CANDIDATE_D_BASE_TO_PATH`` to ``(packed_i32, scales_f32)`` pairs. The
    target is mandatory for alias checking. All arrays and shapes are checked
    before any drafter mutation. Q/K signs and scales are inverse-permuted from
    GGUF RoPE row order to the pinned PyTorch checkpoint row order. Existing
    biases are frozen and retained as declared nonbinary exceptions.
    """
    if target is None:
        raise TypeError("target is required for ownership checks")
    if set(candidate_d) != set(CANDIDATE_D_BASE_TO_PATH):
        raise ValueError("candidate_d must contain exactly the nine selected GGUF bases")
    pending = _validated_linears(drafter, GROUP_PATHS.keys(), target, "recurrent W1A16")
    target_parameter_ids = {id(parameter) for parameter in target.parameters()}
    replacements = {}
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        _, _, linear = pending[path]
        if any(id(parameter) in target_parameter_ids for parameter in linear.parameters()):
            raise ValueError(f"refusing target-shared parameter at {path}")
        packed, scales = candidate_d[base]
        signs = unpack_signs(packed, linear.in_features)
        if signs.shape[0] != linear.out_features:
            raise ValueError(f"{base}: output row count differs from pinned linear")
        scales = np.asarray(scales)
        if scales.dtype != np.float32:
            raise ValueError(f"{base}: scales must be F32")
        _check_scales(scales, linear.out_features, linear.in_features, 128)
        if base in ("blk.0.attn_q", "blk.0.attn_k"):
            heads = 32 if base.endswith("_q") else 8
            signs = _native_qk_to_python_rows(signs, heads)
            scales = _native_qk_to_python_rows(scales, heads)
        replacements[path] = GroupedBinaryLinear(
            torch.from_numpy(np.ascontiguousarray(signs)),
            torch.from_numpy(np.ascontiguousarray(scales)),
            group_size=128,
            bias=linear.bias,
        )
    for path, replacement in replacements.items():
        parent, name, _ = pending[path]
        setattr(parent, name, replacement)
    return replacements
