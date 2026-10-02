"""CPU synthetic receipts; no captured development or final payloads."""

import unittest
from dataclasses import replace

from research.parallel20261002.refresh_contract.reference.contract import (
    CurvePoint,
    RoundCounts,
    decision_receipt,
)
from research.parallel20261002.refresh_contract.reference.demo import (
    handoff_fixture,
    paired_receipts,
)


class ReceiptChecks(unittest.TestCase):
    def test_caps_eos_units_and_actual_gate(self):
        receipts = paired_receipts()
        rise = receipts["rate_rises_drafts_per_round_falls"]
        fall = receipts["rate_falls_drafts_per_round_rises"]
        self.assertEqual(rise["previous"]["counts"], {
            "accepted": 5, "proposed": 10, "rounds": 2, "emitted": 5, "accepted_emitted": 4})
        self.assertEqual(rise["current"]["counts"], {
            "accepted": 2, "proposed": 2, "rounds": 2, "emitted": 3, "accepted_emitted": 2})
        self.assertEqual(rise["gate"]["status"], "pass")
        self.assertEqual(fall["gate"]["reasons"], ["native_acceptance_regression"])
        self.assertEqual(fall["selected_unit"], "accepted_draft_tokens/proposed_draft_tokens")
        self.assertFalse(rise["training_authorized"])

    def test_weight_raw_counts_not_mean_prompt_ratios(self):
        # Unweighted prompt means: old .5, new .375 (regression).
        # Pooled rates: old 1/6, new 2/6 (improvement).
        old = CurvePoint("0" * 64, "1" * 64, 100, 20, 10, (
            RoundCounts("p0", 1, 1, 2, 1), RoundCounts("p1", 0, 5, 1, 5)))
        new = replace(old, checkpoint_sha256="2" * 64, export_sha256="3" * 64,
                      completed_steps=200, ce_loss_sum=18, rounds=(
                          RoundCounts("p0", 1, 2, 2, 2), RoundCounts("p1", 1, 4, 2, 4)))
        base = paired_receipts()["rate_rises_drafts_per_round_falls"]
        policy = {"development_sample_sha256": "b" * 64, "caps": {"max_rounds": 2},
                  "learning_curve": {"min_completed_steps": 200,
                                     "min_relative_ce_improvement": 0.05,
                                     "max_acceptance_regression": 0.02,
                                     "min_changed_prefix_fraction": 0.2}}
        result = decision_receipt(policy, old, new, changed_fraction=.5, round_index=0,
                                  comparison_contract_sha256=base[
                                      "comparison_contract_sha256"])
        self.assertEqual(result["legacy_curve_input"]["previous"]["native_acceptance"], 1/6)
        self.assertEqual(result["legacy_curve_input"]["current"]["native_acceptance"], 2/6)
        self.assertEqual(result["gate"]["status"], "pass")

    def test_actual_plan_and_simulated_ledger_handoff(self):
        fixture = handoff_fixture()
        plans = fixture["planner_manifests"]
        self.assertTrue(plans["planned"]["capture_queue_ready"])
        self.assertEqual(plans["planned"]["counts"]["new_label_rows"], 3)
        self.assertEqual(plans["after_fabricated_capture"]["counts"]["new_label_rows"], 0)
        for plan in (plans["planned"], plans["after_fabricated_capture"]):
            self.assertFalse(plan["training_eligible"])
        self.assertFalse(fixture["planned_only"]["admissible_proposal"])
        self.assertTrue(fixture["audited_simulation"]["admissible_proposal"])
        self.assertEqual(fixture["audited_simulation"]["remaining_gpu_seconds"], 3)
        self.assertIn("cumulative_gpu_budget_exhausted", fixture["spent_budget"]["reasons"])
        self.assertIn("original_research_allowance_closed_or_unknown",
                      fixture["fresh_research_window"]["reasons"])
        for key in ("planned_only", "audited_simulation", "spent_budget",
                    "fresh_research_window"):
            self.assertFalse(fixture[key]["training_authorized"])
            self.assertEqual(fixture[key]["new_experiment_updates"], 0)


if __name__ == "__main__":
    unittest.main()
