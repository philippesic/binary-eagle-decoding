"""Exact ggml CPU attention forward with an explicit F32 surrogate backward.

This is a diagnostic boundary, not an assertion that ggml Flash Attention has
the same derivative as PyTorch attention. The forward invokes the pinned native
helper on a 256-slot physical cache; the backward differentiates the existing
F32 scores/softmax/value expression on the unpadded live prefix.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

NATIVE_ATTENTION_SLOTS = 256
NATIVE_ATTENTION_HELPER_SHA256 = "f63177876148da8e285afc39c021a9d28b9cc75bd020d92471a3d58bcdc3487f"

NativeAttentionForward = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def _require_f32(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.shape != shape
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be one finite CPU F32 tensor of shape {shape}")


class NativeCPUAttentionOracle:
    """Run the exact pinned Apple CPU ggml Flash Attention helper.

    The binary hash is from the validated 46-execution CPU oracle replay. A
    different build must be audited separately before use in this diagnostic.
    """

    def __init__(self, helper_path: Path, *, threads: int = 10) -> None:
        self.helper_path = Path(helper_path).resolve()
        if type(threads) is not int or not 1 <= threads <= 128:
            raise ValueError("native attention helper threads must be in 1..128")
        self.threads = threads
        self._verify_helper()

    def _verify_helper(self) -> None:
        if not self.helper_path.is_file():
            raise ValueError("native attention helper is missing")
        digest = hashlib.sha256(self.helper_path.read_bytes()).hexdigest()
        if digest != NATIVE_ATTENTION_HELPER_SHA256:
            raise ValueError("native attention helper identity differs from pinned CPU oracle")

    def __call__(self, query: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        _require_f32("query", query, (32, 128))
        _require_f32("keys", keys, (8, NATIVE_ATTENTION_SLOTS, 128))
        _require_f32("values", values, (8, NATIVE_ATTENTION_SLOTS, 128))
        if not torch.equal(keys, keys.to(torch.float16).to(torch.float32)) or not torch.equal(
            values, values.to(torch.float16).to(torch.float32)
        ):
            raise ValueError("native attention cache K/V must be F16-exact")
        if (
            not isinstance(mask, Tensor)
            or mask.device.type != "cpu"
            or mask.dtype != torch.float16
            or mask.shape != (1, NATIVE_ATTENTION_SLOTS)
            or not torch.all((mask == 0) | torch.isneginf(mask))
        ):
            raise ValueError("native attention mask must be F16 zero/-inf over 256 slots")
        visible = mask[0] == 0
        count = int(visible.sum())
        if count < 1 or not visible[:count].all() or visible[count:].any():
            raise ValueError("native attention mask must expose one contiguous prefix")
        self._verify_helper()
        with tempfile.TemporaryDirectory(prefix="native-recurrent-attn-") as temporary:
            operands = Path(temporary)
            np.asarray(query.detach().numpy(), dtype="<f4").tofile(operands / "query.f32")
            np.asarray(keys.detach().permute(1, 0, 2).numpy(), dtype="<f2").tofile(
                operands / "keys.f16"
            )
            np.asarray(values.detach().permute(1, 0, 2).numpy(), dtype="<f2").tofile(
                operands / "values.f16"
            )
            np.asarray(mask.detach().numpy(), dtype="<f2").tofile(operands / "mask.f16")
            subprocess.run(
                [
                    str(self.helper_path),
                    str(operands),
                    "1",
                    str(NATIVE_ATTENTION_SLOTS),
                    str(self.threads),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_path = operands / "attention.f32"
            if not output_path.is_file() or output_path.stat().st_size != 32 * 128 * 4:
                raise ValueError("native attention helper returned the wrong output size")
            output = torch.from_numpy(np.fromfile(output_path, dtype="<f4").reshape(32, 128))
        _require_f32("native attention output", output, (32, 128))
        return output


def f32_attention_surrogate(query: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    """The original adapter's differentiable F32 attention arithmetic."""
    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("F32 attention surrogate geometry is invalid")
    heads, width = query.shape
    kv_heads, _, kv_width = keys.shape
    if kv_heads < 1 or width != kv_width or heads % kv_heads:
        raise ValueError("F32 attention surrogate head geometry is invalid")
    repeat = heads // kv_heads
    repeated_keys = keys.repeat_interleave(repeat, dim=0)
    repeated_values = values.repeat_interleave(repeat, dim=0)
    scores = torch.einsum("hd,htd->ht", query, repeated_keys) / math.sqrt(width)
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum("ht,htd->hd", probabilities, repeated_values)


def _native_interleaved_qk(value: Tensor) -> Tensor:
    """Map Python half-split RoPE channels to ggml's interleaved Q/K rows."""
    if not isinstance(value, Tensor) or value.ndim not in (2, 3) or value.shape[-1] != 128:
        raise ValueError("native Q/K row conversion requires head_dim=128")
    return value.reshape(*value.shape[:-1], 2, 64).transpose(-1, -2).reshape(value.shape)


class _NativeForwardF32Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query: Tensor, keys: Tensor, values: Tensor, oracle: NativeAttentionForward
    ) -> Tensor:
        if (
            not isinstance(query, Tensor)
            or not isinstance(keys, Tensor)
            or not isinstance(values, Tensor)
            or query.ndim != 2
            or keys.ndim != 3
            or values.ndim != 3
        ):
            raise ValueError("native attention prefix or head geometry is invalid")
        heads, width = query.shape
        kv_heads, prefix, kv_width = keys.shape
        if width != 128:
            raise ValueError("native attention forward requires head_dim=128")
        if (
            values.shape != keys.shape
            or width != kv_width
            or kv_heads < 1
            or heads % kv_heads
            or not 1 <= prefix <= NATIVE_ATTENTION_SLOTS
        ):
            raise ValueError("native attention prefix or head geometry is invalid")
        _require_f32("query", query, (heads, width))
        _require_f32("keys", keys, (kv_heads, prefix, width))
        _require_f32("values", values, (kv_heads, prefix, width))
        if not torch.equal(keys, keys.to(torch.float16).to(torch.float32)) or not torch.equal(
            values, values.to(torch.float16).to(torch.float32)
        ):
            raise ValueError("native attention cache K/V must be F16-exact")
        padded_keys = torch.zeros((kv_heads, NATIVE_ATTENTION_SLOTS, width), dtype=torch.float32)
        padded_values = torch.zeros_like(padded_keys)
        padded_keys[:, :prefix] = _native_interleaved_qk(keys)
        padded_values[:, :prefix] = values
        mask = torch.full((1, NATIVE_ATTENTION_SLOTS), -torch.inf, dtype=torch.float16)
        mask[:, :prefix] = 0
        native_query = _native_interleaved_qk(query).detach().clone()
        result = oracle(native_query, padded_keys, padded_values, mask)
        _require_f32("native attention output", result, (heads, width))
        ctx.save_for_backward(query, keys, values)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        query, keys, values = ctx.saved_tensors
        with torch.enable_grad():
            surrogate_query = query.detach().requires_grad_(True)
            surrogate_keys = keys.detach().requires_grad_(True)
            surrogate_values = values.detach().requires_grad_(True)
            output = f32_attention_surrogate(surrogate_query, surrogate_keys, surrogate_values)
            gradients = torch.autograd.grad(
                output,
                (surrogate_query, surrogate_keys, surrogate_values),
                grad_output,
            )
        return *gradients, None


def native_forward_f32_backward(
    query: Tensor, keys: Tensor, values: Tensor, oracle: NativeAttentionForward
) -> Tensor:
    """Return exact native forward values with declared F32 surrogate gradients."""
    if not callable(oracle):
        raise ValueError("native attention oracle must be callable")
    return _NativeForwardF32Backward.apply(query, keys, values, oracle)
