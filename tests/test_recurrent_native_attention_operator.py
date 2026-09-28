"""Physical K/V ledger and native ggml Flash Attention fixture tests."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_recurrent_native_attention_operator import _metrics, reconstruct_slots  # noqa: E402


def _fixture():
    executions = [
        {
            "execution": 0,
            "n_tokens": 1,
            "n_kv": 4,
            "mask_dtype": "f16",
            "mask_offset": 0,
            "mask_bytes": 8,
        },
        {
            "execution": 1,
            "n_tokens": 1,
            "n_kv": 4,
            "mask_dtype": "f16",
            "mask_offset": 8,
            "mask_bytes": 8,
        },
    ]
    rows = [
        {"execution": 0, "column": 0, "slot": 0, "position": 0, "row_offset": 0},
        {"execution": 1, "column": 0, "slot": 1, "position": 1, "row_offset": 4096},
    ]
    payload = np.arange(4096, dtype="<u2")
    mask = np.array([0, -np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], dtype="<f2")
    return executions, rows, payload, mask.view("<u2")


class RecurrentNativeAttentionTests(unittest.TestCase):
    def test_ledger_uses_chronological_physical_slots(self):
        executions, rows, payload, mask = _fixture()
        keys, values, selected_mask, info = reconstruct_slots(executions, rows, payload, mask, 1)
        self.assertEqual(keys.shape, (4, 1024))
        self.assertEqual(values.shape, (4, 1024))
        np.testing.assert_array_equal(keys[0], payload[:1024])
        np.testing.assert_array_equal(values[0], payload[1024:2048])
        np.testing.assert_array_equal(keys[1], payload[2048:3072])
        np.testing.assert_array_equal(values[1], payload[3072:4096])
        self.assertEqual(info["visible_slots_per_query"], [2])
        self.assertEqual(info["known_slots"], 2)
        self.assertTrue(np.isneginf(selected_mask.view("<f2")[0, 2]))

    def test_ledger_rejects_mask_visible_unknown_slot(self):
        executions, rows, payload, mask = _fixture()
        mask[-2] = np.float16(0).view("<u2")
        with self.assertRaisesRegex(ValueError, "without a captured write"):
            reconstruct_slots(executions, rows, payload, mask, 1)

    def test_ledger_rejects_wrong_physical_offset_and_query_shape(self):
        executions, rows, payload, mask = _fixture()
        rows[1]["row_offset"] = 0
        with self.assertRaisesRegex(ValueError, "physical cache write"):
            reconstruct_slots(executions, rows, payload, mask, 1)
        with self.assertRaisesRegex(ValueError, "geometry"):
            _metrics(np.zeros((1, 32, 127), dtype="<f4"), np.zeros((1, 32, 128), dtype="<f4"))

    def test_output_metrics_count_exact_f32_bits_and_head_error(self):
        native = np.ones((1, 32, 128), dtype="<f4")
        replay = native.copy()
        replay[0, 0, 0] = 2.0
        result = _metrics(replay, native)
        self.assertEqual(result["elements"], 4096)
        self.assertEqual(result["bitwise_equal_elements"], 4095)
        self.assertEqual(result["max_abs_difference"], 1.0)
        self.assertGreater(result["per_head_relative_l2"][0], 0)
        self.assertEqual(result["per_head_relative_l2"][1:], [0.0] * 31)

    def test_native_helper_one_visible_value_per_head(self):
        helper_name = os.environ.get("NATIVE_RECURRENT_ATTENTION_HELPER")
        if not helper_name:
            self.skipTest("set NATIVE_RECURRENT_ATTENTION_HELPER to run compiled ggml fixture")
        helper = Path(helper_name).resolve()
        self.assertTrue(helper.is_file())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.zeros((1, 32, 128), dtype="<f4").tofile(path / "query.f32")
            np.zeros((1, 8, 128), dtype="<f2").tofile(path / "keys.f16")
            values = np.arange(1024, dtype=np.float32).reshape(1, 8, 128) / 1024
            values.astype("<f2").tofile(path / "values.f16")
            np.zeros((1, 1), dtype="<f2").tofile(path / "mask.f16")
            subprocess.run([str(helper), str(path), "1", "1", "1"], check=True, capture_output=True)
            actual = np.fromfile(path / "attention.f32", dtype="<f4").reshape(32, 128)
            expected = np.repeat(values.astype("<f2").astype("<f4")[0], 4, axis=0)
            np.testing.assert_array_equal(actual, expected)

    def test_native_helper_rejects_truncated_cache_fixture(self):
        helper_name = os.environ.get("NATIVE_RECURRENT_ATTENTION_HELPER")
        if not helper_name:
            self.skipTest("set NATIVE_RECURRENT_ATTENTION_HELPER to run compiled ggml fixture")
        helper = Path(helper_name).resolve()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.zeros((1, 32, 128), dtype="<f4").tofile(path / "query.f32")
            np.zeros((1, 8, 127), dtype="<f2").tofile(path / "keys.f16")
            np.zeros((1, 8, 128), dtype="<f2").tofile(path / "values.f16")
            np.zeros((1, 1), dtype="<f2").tofile(path / "mask.f16")
            result = subprocess.run(
                [str(helper), str(path), "1", "1", "1"], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("wrong-size operand", result.stderr)


if __name__ == "__main__":
    unittest.main()
