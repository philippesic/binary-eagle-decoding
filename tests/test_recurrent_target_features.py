"""CPU-only checks for independent target-feature row alignment and metrics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_recurrent_target_features import WIDTH, _capture_metrics  # noqa: E402


class TargetFeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.prefix = [10, 11]
        self.metadata = [
            {
                "event": "decoded_row",
                "phase": "prefill",
                "feature_row": index,
                "position": index,
                "token_id": token,
                "prefix_token_ids": self.prefix[: index + 1],
                "target_layer_ids": [2, 18, 33],
            }
            for index, token in enumerate(self.prefix)
        ]
        self.values = np.ones((2, WIDTH), dtype="<f4")
        self.save()

    def save(self):
        (self.directory / "heads.target_features.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.metadata)
        )
        self.values.tofile(self.directory / "heads.target_features.f32")

    def test_exact_rows_have_zero_drift_for_all_taps(self):
        metrics, hashes = _capture_metrics(self.directory, self.prefix, self.values.copy())
        self.assertEqual(set(metrics), {"2", "18", "33"})
        self.assertTrue(all(row["max_abs"] == 0 for row in metrics.values()))
        self.assertEqual(len(hashes["values"]), 64)

    def test_wrong_token_ancestry_and_payload_size_fail(self):
        self.metadata[1]["prefix_token_ids"] = [10, 12]
        self.save()
        with self.assertRaisesRegex(ValueError, "ancestry"):
            _capture_metrics(self.directory, self.prefix, self.values)
        self.metadata[1]["prefix_token_ids"] = [10, 11]
        self.save()
        (self.directory / "heads.target_features.f32").write_bytes(b"\0" * 4)
        with self.assertRaisesRegex(ValueError, "payload size"):
            _capture_metrics(self.directory, self.prefix, self.values)

    def test_second_tap_drift_isolated_from_other_taps(self):
        self.values[1, 2560] = 2
        self.save()
        reference = np.ones((2, WIDTH), dtype=np.float32)
        metrics, _ = _capture_metrics(self.directory, self.prefix, reference)
        self.assertEqual(metrics["18"]["max_abs"], 1)
        self.assertGreater(metrics["18"]["max_relative_row_l2"], 0)
        self.assertEqual(metrics["2"]["max_abs"], 0)
        self.assertEqual(metrics["33"]["max_abs"], 0)


if __name__ == "__main__":
    unittest.main()
