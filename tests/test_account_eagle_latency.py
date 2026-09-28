"""Small contract checks for disjoint latency accounting."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "account_eagle_latency.py"
SPEC = importlib.util.spec_from_file_location("account_eagle_latency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LatencyAccountingTests(unittest.TestCase):
    def test_round_overlap_and_unassigned_reconcile(self):
        row = {
            "round_start_us": 100,
            "round_end_us": 200,
            "round_us": 100,
            "spans_us": {"draft": [110, 150], "process": [140, 170]},
        }
        parts = MODULE.round_partition(row)
        self.assertEqual(parts["draft"], 30)
        self.assertEqual(parts["process"], 20)
        self.assertEqual(parts["shared_overlap"], 10)
        self.assertEqual(parts["unassigned"], 40)
        self.assertEqual(sum(parts.values()), 100)

    def test_stage_requires_contiguous_partition(self):
        row = {
            "schema": "eagle_draft_stage_v1",
            "clock": "ggml_time_us_cpu_wall",
            "start_us": 10,
            "end_us": 20,
            "total_us": 10,
            "partition_us": 10,
            "unassigned_us": 0,
            "stage_totals_us": {"seed": 10},
            "spans": [{"stage": "seed", "start_us": 10, "end_us": 20, "duration_us": 10}],
        }
        self.assertEqual(MODULE.checked_stage(row), {"seed": 10})
        row["spans"][0]["start_us"] = 11
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            MODULE.checked_stage(row)


if __name__ == "__main__":
    unittest.main()
