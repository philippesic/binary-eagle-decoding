"""Small CPU-only GGUF fixtures for borrowed frozen EAGLE operands."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from w1a1_eagle.frozen_operands import DRAFT_NORMS, FrozenOperands

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
from gguf import GGUFWriter  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, architecture: str, arrays: dict[str, np.ndarray]) -> None:
    writer = GGUFWriter(path, architecture)
    for name, array in arrays.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class FrozenOperandsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.target = directory / "target.gguf"
        self.draft = directory / "draft.gguf"
        self.embedding = (np.arange(15).reshape(5, 3) / 2).astype(np.float16)
        self.norms = {
            name: np.array([index + 0.5, index + 1.5, index + 2.5], dtype=np.float32)
            for index, name in enumerate(DRAFT_NORMS)
        }
        self.write()

    def write(
        self,
        *,
        embedding: np.ndarray | None = None,
        norms: dict[str, np.ndarray] | None = None,
        target_arch: str = "qwen3",
        draft_arch: str = "eagle3",
    ) -> None:
        _write(
            self.target,
            target_arch,
            {"token_embd.weight": self.embedding if embedding is None else embedding},
        )
        _write(self.draft, draft_arch, self.norms if norms is None else norms)

    def load(self, **kwargs) -> FrozenOperands:
        return FrozenOperands(
            self.target,
            self.draft,
            target_sha256=_sha256(self.target),
            draft_sha256=_sha256(self.draft),
            vocab_size=5,
            hidden_size=3,
            **kwargs,
        )

    def test_row_lookup_is_cpu_f16_copy_and_norms_are_copied(self) -> None:
        operands = self.load()
        self.assertEqual((operands.vocab_size, operands.hidden_size), (5, 3))
        self.assertIsInstance(operands._target_reader.data, np.memmap)
        self.assertTrue(np.shares_memory(operands._embedding, operands._target_reader.data))
        row = operands(2)
        self.assertEqual(
            (row.device.type, row.dtype, row.requires_grad), ("cpu", torch.float16, False)
        )
        torch.testing.assert_close(row, torch.tensor(self.embedding[2].copy()))
        row.fill_(-100)
        torch.testing.assert_close(operands(2), torch.tensor(self.embedding[2].copy()))
        self.assertEqual(set(operands.norm_arrays), set(DRAFT_NORMS))
        for name, value in operands.norm_arrays.items():
            self.assertEqual(
                (value.device.type, value.dtype, value.requires_grad), ("cpu", torch.float32, False)
            )
            torch.testing.assert_close(value, torch.from_numpy(self.norms[name]))
        copied = operands.norm_arrays[DRAFT_NORMS[0]]
        copied[0] = -100
        torch.testing.assert_close(
            operands.norm_arrays[DRAFT_NORMS[0]], torch.from_numpy(self.norms[DRAFT_NORMS[0]])
        )

    def test_rejects_invalid_token_and_hash(self) -> None:
        operands = self.load()
        for token in (-1, 5, 1.0, True):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "token"):
                operands(token)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            FrozenOperands(
                self.target,
                self.draft,
                target_sha256="0" * 64,
                draft_sha256=_sha256(self.draft),
                vocab_size=5,
                hidden_size=3,
            )
        with self.assertRaisesRegex(ValueError, "SHA256"):
            FrozenOperands(
                self.target,
                self.draft,
                target_sha256=_sha256(self.target),
                draft_sha256="0" * 64,
                vocab_size=5,
                hidden_size=3,
            )
        with self.assertRaisesRegex(ValueError, "64 lowercase"):
            FrozenOperands(self.target, self.draft, target_sha256="wrong")

    def test_rejects_embedding_type_and_shape(self) -> None:
        for array in (
            self.embedding.astype(np.float32),
            np.zeros((4, 3), dtype=np.float16),
            np.zeros((5, 4), dtype=np.float16),
        ):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                self.write(embedding=array)
                with self.assertRaisesRegex(ValueError, "token_embd.weight must be F16"):
                    self.load()

    def test_rejects_norm_missing_type_shape_and_nonfinite(self) -> None:
        name = DRAFT_NORMS[1]
        variants = (
            ({key: value for key, value in self.norms.items() if key != name}, "exactly one"),
            ({**self.norms, name: self.norms[name].astype(np.float16)}, "must be F32"),
            ({**self.norms, name: np.ones(4, np.float32)}, "must be F32"),
            ({**self.norms, name: np.array([1, np.nan, 3], np.float32)}, "must be finite"),
        )
        for norms, error in variants:
            with self.subTest(error=error):
                self.write(norms=norms)
                with self.assertRaisesRegex(ValueError, error):
                    self.load()

    def test_rejects_wrong_architecture_and_nonfinite_embedding_row(self) -> None:
        self.write(target_arch="llama")
        with self.assertRaisesRegex(ValueError, "qwen3 architecture"):
            self.load()
        self.write(draft_arch="llama")
        with self.assertRaisesRegex(ValueError, "eagle3 architecture"):
            self.load()
        embedding = self.embedding.copy()
        embedding[1, 0] = np.inf
        self.write(embedding=embedding)
        operands = self.load()
        with self.assertRaisesRegex(ValueError, "must be finite"):
            operands(1)


if __name__ == "__main__":
    unittest.main()
