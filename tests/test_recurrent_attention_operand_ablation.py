"""Shape, row-order and physical-slot checks for the attention operand ablation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from check_recurrent_attention_operand_ablation import (  # noqa: E402
    MODE_LABELS,
    _load_student_operands,
    build_modes,
    native_qk_to_python_rows,
    python_qk_to_native_rows,
)
from check_recurrent_real_step import _post_seed_cache_arrays  # noqa: E402

from w1a1_eagle.native_step import NativeStepCache  # noqa: E402


class AttentionOperandAblationTests(unittest.TestCase):
    def test_qk_permutation_and_inverse_on_query_and_batched_cache(self):
        for shape, heads in (((32, 128), 32), ((3, 8, 128), 8)):
            source = np.arange(np.prod(shape), dtype="<f4").reshape(shape)
            native = python_qk_to_native_rows(source, heads)
            np.testing.assert_array_equal(native_qk_to_python_rows(native, heads), source)
            self.assertFalse(np.array_equal(native, source))
        with self.assertRaisesRegex(ValueError, "geometry"):
            python_qk_to_native_rows(np.zeros((8, 127), dtype="<f4"), 8)

    def test_export_requires_f16_exact_pinned_cache_geometry(self):
        key = torch.arange(8 * 2 * 128, dtype=torch.float32).reshape(8, 2, 128) / 1024
        value = -key
        exported = _post_seed_cache_arrays(NativeStepCache(key, value), 2)
        self.assertEqual(set(exported), {"key", "value"})
        self.assertEqual(exported["key"].shape, (8, 2, 128))
        self.assertEqual(exported["key"].dtype, np.dtype("<f4"))
        key[0, 0, 0] = 0.25
        self.assertEqual(exported["key"][0, 0, 0], 0.0)  # copied, not a view
        with self.assertRaisesRegex(ValueError, "F16-exact"):
            _post_seed_cache_arrays(NativeStepCache(key + 1e-7, value), 2)
        with self.assertRaisesRegex(ValueError, "positions"):
            _post_seed_cache_arrays(NativeStepCache(key, value), 257)

    def test_loader_rejects_ambiguous_or_truncated_operands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            taps = path / "taps.npz"
            cache = path / "cache.npz"
            np.savez(taps, **{"Qcur_rope-0": np.zeros(4096, dtype="<f4")})
            keys = np.zeros((8, 2, 128), dtype="<f4")
            np.savez(cache, key=keys, value=keys)
            q, k, v = _load_student_operands(taps, cache)
            self.assertEqual((q.shape, k.shape, v.shape), ((32, 128), (8, 2, 128), (8, 2, 128)))
            np.savez(cache, key=keys, value=keys, extra=keys)
            with self.assertRaisesRegex(ValueError, "exactly key and value"):
                _load_student_operands(taps, cache)
            np.savez(cache, key=keys[:, :1, :127], value=keys[:, :1, :127])
            with self.assertRaisesRegex(ValueError, "F32 \\[8,T,128\\]"):
                _load_student_operands(taps, cache)

    def test_modes_preserve_reserve_slots_and_v_row_order(self):
        native_q = np.zeros((32, 128), dtype="<f4")
        student_q = np.arange(4096, dtype="<f4").reshape(32, 128)
        native_k = np.zeros((256, 1024), dtype="<u2")
        native_v = np.zeros_like(native_k)
        native_k[2:] = 0x3C00
        native_v[2:] = 0x4000
        student_k = np.arange(8 * 2 * 128, dtype="<f4").reshape(8, 2, 128) / 1024
        student_v = -student_k
        modes = build_modes(native_q, native_k, native_v, student_q, student_k, student_v)
        self.assertEqual(tuple(modes), MODE_LABELS)
        _, converted_k, converted_v = modes["student_qk_native_rows_student_v"]
        np.testing.assert_array_equal(converted_k[2:], native_k[2:])
        np.testing.assert_array_equal(converted_v[2:], native_v[2:])
        np.testing.assert_array_equal(
            converted_k[:2].view("<f2").astype("<f4").reshape(2, 8, 128),
            python_qk_to_native_rows(student_k.transpose(1, 0, 2), 8),
        )
        np.testing.assert_array_equal(
            converted_v[:2].view("<f2").astype("<f4").reshape(2, 8, 128),
            student_v.transpose(1, 0, 2),
        )
        np.testing.assert_array_equal(modes["student_q_native_rows_native_kv"][1], native_k)


if __name__ == "__main__":
    unittest.main()
