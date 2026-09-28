"""Pinned ggml CPU forward operators for parity diagnostics only.

This path requires torch.no_grad(). It supplies no derivative for RoPE or
SiLU and does not select the attention surrogate backward for training.
"""

from __future__ import annotations

import ctypes
import hashlib
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .native_attention_oracle import (
    NativeCPUAttentionOracle,
    _native_interleaved_qk,
    native_forward_f32_backward,
)

NATIVE_ROPE_HELPER_SHA256 = "d44d710f4debd138ec4f790ab7317626650d9ff69e9770aeec19c478a718def1"
NATIVE_SILU_LIBRARY_SHA256 = "0d499359a40900596b172556bebd1aad1fa69aaf1cb3e46bf962a98ccf2a0600"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f32(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.shape != shape
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite CPU F32 {shape}")


class NativeCPUDiagnosticOperators:
    """Apple CPU ggml forward, with explicit no-backward/no-training scope."""

    def __init__(
        self,
        rope_helper: Path,
        silu_library: Path,
        attention_helper: Path,
        *,
        threads: int = 10,
    ) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("pinned native CPU diagnostic requires macOS arm64")
        self.rope_helper = Path(rope_helper).resolve()
        self.silu_library = Path(silu_library).resolve()
        if (
            not self.rope_helper.is_file()
            or not self.silu_library.is_file()
            or _sha256(self.rope_helper) != NATIVE_ROPE_HELPER_SHA256
            or _sha256(self.silu_library) != NATIVE_SILU_LIBRARY_SHA256
        ):
            raise ValueError("pinned native RoPE or SiLU operator identity differs")
        self.attention_oracle = NativeCPUAttentionOracle(attention_helper, threads=threads)
        self._library = ctypes.CDLL(str(self.silu_library))
        self._silu = self._library.ggml_vec_silu_f32
        pointer = ctypes.POINTER(ctypes.c_float)
        self._silu.argtypes = (ctypes.c_int, pointer, pointer)
        self._silu.restype = None

    @staticmethod
    def _require_no_grad() -> None:
        if torch.is_grad_enabled():
            raise ValueError("native CPU diagnostic is forward-only; use torch.no_grad()")

    def rope(self, query: Tensor, key: Tensor, position: int) -> tuple[Tensor, Tensor]:
        """Rotate 32 Q heads and eight K heads in one five-row ggml graph."""
        self._require_no_grad()
        _f32("query", query, (32, 128))
        _f32("key", key, (8, 128))
        if type(position) is not int or not 0 <= position < 256:
            raise ValueError("native CPU RoPE position must be in 0..255")
        native_q = _native_interleaved_qk(query).numpy().reshape(4, 1024)
        native_k = _native_interleaved_qk(key).numpy().reshape(1, 1024)
        source = np.concatenate((native_q, native_k), axis=0).astype("<f4", copy=False)
        with tempfile.TemporaryDirectory(prefix="native-cpu-diagnostic-rope-") as directory:
            operands = Path(directory)
            source.tofile(operands / "key_raw.f32")
            np.full(5, position, dtype="<i4").tofile(operands / "positions.i32")
            subprocess.run(
                [str(self.rope_helper), str(operands), "5"],
                check=True,
                capture_output=True,
                text=True,
            )
            output = operands / "key_rope.f32"
            if not output.is_file() or output.stat().st_size != source.nbytes:
                raise ValueError("native CPU RoPE returned a truncated F32 tensor")
            rotated = np.fromfile(output, dtype="<f4").reshape(5, 8, 128)
        if not np.isfinite(rotated).all():
            raise ValueError("native CPU RoPE returned nonfinite values")

        def python_rows(value: np.ndarray, heads: int) -> Tensor:
            rows = value.reshape(heads, 128).reshape(heads, 64, 2)
            return torch.from_numpy(rows.transpose(0, 2, 1).reshape(heads, 128).copy())

        return python_rows(rotated[:4], 32), python_rows(rotated[4:], 8)

    def attention(self, query: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        self._require_no_grad()
        return native_forward_f32_backward(query, keys, values, self.attention_oracle)

    def silu(self, gate: Tensor) -> Tensor:
        self._require_no_grad()
        _f32("gate", gate, (9728,))
        source = np.ascontiguousarray(gate.numpy(), dtype="<f4")
        result = np.empty_like(source)
        pointer = ctypes.POINTER(ctypes.c_float)
        self._silu(
            source.size,
            result.ctypes.data_as(pointer),
            source.ctypes.data_as(pointer),
        )
        if not np.isfinite(result).all():
            raise ValueError("native CPU SiLU returned nonfinite values")
        return torch.from_numpy(result)
