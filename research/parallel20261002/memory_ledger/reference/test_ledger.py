"""Bounded arithmetic checks; configured shapes are integers only."""

import importlib.util
import json
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("ledger.py")
SPEC = importlib.util.spec_from_file_location("memory_ledger_reference", MODULE)
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


class LedgerChecks(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ledger.ROOT / "configs/continuous_w1ax.json").read_text())
        self.result = ledger.configured_ledger(self.spec)

    def test_configured_counts(self):
        self.assertEqual(self.result["weights_per_lane"], 218_234_880)
        self.assertEqual(self.result["rows_per_lane"], 65_280)
        self.assertEqual(self.result["head_weights"], 81_920_000)
        self.assertEqual(self.result["largest_f32_tensor_bytes"], 327_680_000)

    def test_cpu_alias_source_has_only_one_new_clone(self):
        self.assertEqual(
            ledger.ordered_cpu_copy_peak([(8, False), (20, False)]),
            {"final_clone_bytes": 28, "additional_peak_bytes": 28},
        )

    def test_transfer_lifetime_is_order_sensitive(self):
        self.assertEqual(
            ledger.ordered_cpu_copy_peak([(8, True), (20, True)]),
            {"final_clone_bytes": 28, "additional_peak_bytes": 48},
        )
        self.assertEqual(
            ledger.ordered_cpu_copy_peak([(20, True), (8, True)]),
            {"final_clone_bytes": 28, "additional_peak_bytes": 40},
        )

    def test_current_admission_omits_live_transfer(self):
        copy = self.result["curriculum_save_core_and_adam"]
        self.assertEqual(copy["logical_excess_over_current_allowance_bytes"], 310_380_508)
        self.assertGreater(copy["proposed_allowance_bytes"], copy["additional_peak_bytes"])

    def test_subcap_arithmetic_cannot_establish_fit(self):
        self.assertLess(
            self.result["existing_estimated_peak_bytes"], self.result["configured_cuda_cap_bytes"]
        )
        self.assertEqual(self.result["existing_estimated_peak_bytes"], 10_112_064_512)
        # The supplied graph allowance remains an assumption, not a measured bound.
        self.assertEqual(self.spec["assumed_graph_budget_bytes"], 3 * 1024**3)


if __name__ == "__main__":
    unittest.main()
