"""CPU-only same-input FFN replay and strict native graph-join checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from check_recurrent_ffn_from_graph import (  # noqa: E402
    FFN_BASES,
    _metrics,
    join_first_seed,
    replay_ffn,
)

from w1a1_eagle.recurrent_binary import pack_signs  # noqa: E402


def _graph_row(name: str, execution: int, offset: int, width: int = 2560) -> dict:
    return {
        "group_kind": "decoder",
        "group_execution": execution,
        "tensor_name": name,
        "dtype": "f32",
        "n_tokens": 1,
        "token_width": width,
        "f32_offset": offset,
    }


class FirstSeedJoinTests(unittest.TestCase):
    def setUp(self):
        self.prenorm = np.zeros(2560, dtype=np.float32)
        self.prenorm[0:2] = [3.0, 4.0]
        self.norm = np.ones(2560, dtype=np.float32)
        normalized = self.prenorm.astype(np.float64) / np.sqrt(
            np.mean(self.prenorm.astype(np.float64) ** 2) + 1e-6
        )
        self.head = normalized.astype(np.float32)
        self.input = np.linspace(-1, 1, 2560, dtype=np.float32)
        self.output = self.input * 2
        unrelated = self.prenorm + 1
        self.values = np.concatenate((unrelated, self.prenorm, self.input, self.output))
        self.rows = [
            _graph_row("eagle3_prenorm-0", 1, 0),
            _graph_row("eagle3_prenorm-0", 2, 2560),
            _graph_row("post_attn_norm-0", 2, 5120),
            _graph_row("ffn_out-0", 2, 7680),
        ]

    def test_reconstructs_unique_head_and_reads_same_seed_column(self):
        native_input, native_output, join = join_first_seed(
            self.rows, self.values, self.head, self.norm
        )
        np.testing.assert_array_equal(native_input, self.input)
        np.testing.assert_array_equal(native_output, self.output)
        self.assertEqual(join["native_group_execution"], 2)
        self.assertEqual(join["native_token_column"], 0)
        self.assertEqual(join["join_method"], "f32_output_norm_reconstruction")
        self.assertLess(join["join_error"]["rms"], 1e-4)

    def test_rejects_ambiguous_or_wrong_execution_join(self):
        duplicate = self.prenorm.copy()
        values = self.values.copy()
        values[:2560] = duplicate
        with self.assertRaisesRegex(ValueError, "unique reconstructed"):
            join_first_seed(self.rows, values, self.head, self.norm)
        rows = [dict(row) for row in self.rows]
        rows[1]["group_execution"] = 3
        rows[2]["group_execution"] = 3
        rows[3]["group_execution"] = 3
        with self.assertRaisesRegex(ValueError, "execution 2, column 0"):
            join_first_seed(rows, self.values, self.head, self.norm)

    def test_rejects_missing_or_mismatched_ffn_taps(self):
        missing = self.rows[:-1]
        with self.assertRaisesRegex(ValueError, "lacks FFN input or output"):
            join_first_seed(missing, self.values, self.head, self.norm)
        rows = [dict(row) for row in self.rows]
        rows[-1]["n_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "different token counts"):
            join_first_seed(rows, self.values, self.head, self.norm)


class FFNArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.input = np.array([1.001, -0.751, 0.333, 1.117], dtype=np.float32)
        gate_signs = np.tile(np.array([1, -1, 1, 1], np.float32), (32, 1))
        up_signs = np.tile(np.array([-1, 1, 1, 1], np.float32), (32, 1))
        gate_signs[1::2] *= -1
        up_signs[2::3] *= -1
        down_signs = np.ones((4, 32), np.float32)
        down_signs[:, 1::2] = -1
        down_signs[1::2] *= -1
        self.signs = (gate_signs, up_signs, down_signs)
        self.scales = (
            np.linspace(0.11, 0.42, 32, dtype=np.float32)[:, None],
            np.linspace(0.22, 0.38, 32, dtype=np.float32)[:, None],
            np.array([[0.35], [0.27], [0.4], [0.19]], np.float32),
        )
        self.arrays = {
            base: (pack_signs(signs), scales)
            for base, signs, scales in zip(FFN_BASES, self.signs, self.scales, strict=True)
        }

    def test_three_projection_ffn_matches_manual_f16_boundaries(self):
        with torch.no_grad():
            x = torch.from_numpy(self.input).to(torch.float16).to(torch.float32)
            gate = F.linear(x, torch.from_numpy(self.signs[0])) * torch.from_numpy(
                self.scales[0][:, 0]
            )
            up = F.linear(x, torch.from_numpy(self.signs[1])) * torch.from_numpy(
                self.scales[1][:, 0]
            )
            fused = (F.silu(gate) * up).to(torch.float16).to(torch.float32)
            expected = F.linear(fused, torch.from_numpy(self.signs[2])) * torch.from_numpy(
                self.scales[2][:, 0]
            )
        for arithmetic in ("native_order", "group_matmul"):
            with self.subTest(arithmetic=arithmetic):
                actual, summaries = replay_ffn(
                    self.input, self.arrays, arithmetic, expected_intermediate=32
                )
                np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-6, atol=2e-6)
                self.assertEqual(summaries["gate"]["elements"], 32)
                self.assertEqual(summaries["up"]["elements"], 32)

    def test_rejects_missing_projection_or_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "exactly gate/up/down"):
            replay_ffn(self.input, {FFN_BASES[0]: self.arrays[FFN_BASES[0]]}, "native_order")
        malformed = dict(self.arrays)
        malformed[FFN_BASES[2]] = malformed[FFN_BASES[0]]
        with self.assertRaisesRegex(ValueError, "geometry"):
            replay_ffn(self.input, malformed, "native_order", expected_intermediate=32)

    def test_metrics_count_bitwise_f32_matches(self):
        expected = np.array([1.0, -2.0, 0.0], np.float32)
        actual = np.array([1.0, -1.5, -0.0], np.float32)
        metrics = _metrics(actual, expected)
        self.assertEqual(metrics["exact_elements"], 1)
        self.assertEqual(metrics["max_abs"], 0.5)
        self.assertGreater(metrics["relative_l2"], 0)


if __name__ == "__main__":
    unittest.main()
