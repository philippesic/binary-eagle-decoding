"""Small arithmetic checks for the model-free attention ablation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_recurrent_attention_precision import _neon_f16_dot, _online_attention  # noqa: E402


class AttentionPrecisionTests(unittest.TestCase):
    def test_neon_dot_lanes_cover_each_element(self) -> None:
        query = np.ones(128, dtype=np.float16)
        key = np.zeros(128, dtype=np.float16)
        key[::8] = np.float16(2.0)
        self.assertEqual(_neon_f16_dot(query, key), 32.0)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            _neon_f16_dot(query[:-1], key[:-1])

    def test_online_attention_rescales_when_maximum_increases(self) -> None:
        scores = np.tile(np.array([0.0, np.log(3.0)], dtype=np.float32), (4, 1))
        values = np.zeros((1, 2, 2), dtype=np.float16)
        values[0, 0, 0] = 1.0
        values[0, 1, 1] = 1.0
        result = _online_attention(scores, values, f16_values=True)
        np.testing.assert_allclose(result, np.tile([0.25, 0.75], (4, 1)), atol=4e-4)

    def test_f16_online_numerator_has_distinct_rounding(self) -> None:
        scores = np.zeros((4, 3), dtype=np.float32)
        values = np.zeros((1, 3, 1), dtype=np.float16)
        values[0, :, 0] = [1000.0, 0.125, 0.125]
        f32 = _online_attention(scores, values, f16_values=False)
        f16 = _online_attention(scores, values, f16_values=True)
        self.assertGreater(float(np.max(np.abs(f32 - f16))), 0.01)


if __name__ == "__main__":
    unittest.main()
