"""Accepted-prefix feature selection for real CPU drafter replay."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_recurrent_real_step import _retained_feature_indices  # noqa: E402


class RetainedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        for row, prefix, retained in (
            (0, [10], True),
            (1, [10, 11], True),
            (2, [10, 11, 12], True),
            (3, [10, 11, 99], False),
        ):
            self.events.extend(
                (
                    {
                        "event": "decoded_row",
                        "feature_row": row,
                        "task_id": 5,
                        "position": len(prefix) - 1,
                        "token_id": prefix[-1],
                        "prefix_token_ids": prefix,
                    },
                    {"event": "disposition", "feature_row": row, "retained_input": retained},
                )
            )

    def test_selects_only_exact_retained_ancestry(self):
        self.assertEqual(_retained_feature_indices(self.events, [10, 11, 12], 5), [0, 1, 2])

    def test_missing_or_duplicate_retained_prefix_fails(self):
        missing = [row for row in self.events if row.get("feature_row") != 2]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _retained_feature_indices(missing, [10, 11, 12], 5)
        duplicate = self.events + [
            {**self.events[4], "feature_row": 4},
            {**self.events[5], "feature_row": 4},
        ]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _retained_feature_indices(duplicate, [10, 11, 12], 5)


if __name__ == "__main__":
    unittest.main()
