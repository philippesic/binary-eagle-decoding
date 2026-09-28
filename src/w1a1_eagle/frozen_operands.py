"""Read-only CPU operands borrowed by the recurrent binary EAGLE drafter.

The pinned target's token embedding is too large to copy for a CPU training
adapter. GGUFReader memory-maps it; each lookup copies just one F16 row into a
CPU tensor. The four small candidate-D draft norms are copied from GGUF.
Neither object exposes a trainable parameter or a view of mapped weight bytes.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

_GGUF_PY = Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp" / "gguf-py"
if str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))

from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

TARGET_F16_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
DRAFT_D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"

TARGET_EMBEDDING = "token_embd.weight"
DRAFT_NORMS = (
    "output_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_norm_2.weight",
    "blk.0.ffn_norm.weight",
)


def _check_sha256(path: Path, expected: str) -> None:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected GGUF SHA256 must be 64 lowercase hex characters")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"{path.name}: GGUF SHA256 differs from frozen source")


def _unique_tensor(reader: GGUFReader, name: str):
    tensors = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(tensors) != 1:
        raise ValueError(f"GGUF requires exactly one {name} tensor")
    return tensors[0]


class FrozenOperands:
    """Validated F16 target embedding lookup and F32 draft norm source.

    Expected hashes default to the pinned FP16 target and candidate D. Explicit
    hashes and dimensions allow tiny synthetic GGUF fixtures without weakening
    the production check. The target mapping stays alive with this object.
    ``norm_arrays`` returns fresh CPU tensor copies so callers cannot alter
    its source.
    """

    def __init__(
        self,
        target_gguf: Path,
        draft_gguf: Path,
        *,
        target_sha256: str = TARGET_F16_SHA256,
        draft_sha256: str = DRAFT_D_SHA256,
        vocab_size: int = 151_936,
        hidden_size: int = 2_560,
    ) -> None:
        if type(vocab_size) is not int or type(hidden_size) is not int:
            raise ValueError("vocab_size and hidden_size must be integers")
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        target_gguf, draft_gguf = Path(target_gguf), Path(draft_gguf)
        _check_sha256(target_gguf, target_sha256)
        _check_sha256(draft_gguf, draft_sha256)

        target = GGUFReader(target_gguf, mode="r")
        draft = GGUFReader(draft_gguf, mode="r")
        if target.fields.get("general.architecture") is None or (
            target.fields["general.architecture"].contents() != "qwen3"
        ):
            raise ValueError("target GGUF must have qwen3 architecture")
        if draft.fields.get("general.architecture") is None or (
            draft.fields["general.architecture"].contents() != "eagle3"
        ):
            raise ValueError("draft GGUF must have eagle3 architecture")

        embedding = _unique_tensor(target, TARGET_EMBEDDING)
        if embedding.tensor_type != GGMLQuantizationType.F16 or embedding.data.shape != (
            vocab_size,
            hidden_size,
        ):
            raise ValueError(f"{TARGET_EMBEDDING} must be F16 [{vocab_size}, {hidden_size}]")
        if not isinstance(target.data, np.memmap) or not np.shares_memory(
            embedding.data, target.data
        ):
            raise ValueError("target embedding must remain file memory-mapped")

        norms: dict[str, np.ndarray] = {}
        for name in DRAFT_NORMS:
            tensor = _unique_tensor(draft, name)
            if tensor.tensor_type != GGMLQuantizationType.F32 or tensor.data.shape != (
                hidden_size,
            ):
                raise ValueError(f"{name} must be F32 [{hidden_size}]")
            value = np.array(tensor.data, dtype=np.float32, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            norms[name] = value

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._target_reader = target
        self._embedding = embedding.data
        self._norms = MappingProxyType(norms)

    @property
    def norm_arrays(self) -> Mapping[str, torch.Tensor]:
        """Fresh, non-trainable CPU F32 tensors keyed by exact GGUF names."""
        copied: dict[str, torch.Tensor] = {}
        for name, value in self._norms.items():
            copied[name] = torch.from_numpy(value.copy())
        return MappingProxyType(copied)

    def __call__(self, token: int) -> torch.Tensor:
        """Return one independent CPU F16 embedding row for a target token."""
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise ValueError("token must be a target-vocabulary integer")
        row = np.array(self._embedding[token], dtype=np.float16, copy=True)
        if not np.isfinite(row).all():
            raise ValueError(f"{TARGET_EMBEDDING}[{token}] must be finite")
        result = torch.from_numpy(row)
        if result.device.type != "cpu" or result.dtype != torch.float16 or result.requires_grad:
            raise RuntimeError("frozen embedding lookup must return non-trainable CPU F16")
        return result
