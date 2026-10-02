"""Independent CPU checks of refresh metric units and handoff boundaries."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from research.parallel20261002.refresh_contract.reference.contract import (
    BudgetLedger,
    CorpusAdmission,
    CurvePoint,
    LedgerEntry,
    RoundCounts,
    decision_receipt,
    new_experiment_proposal,
)
from w1a1_eagle.qat_curriculum import CurriculumConfig, CurriculumState, PrecisionStage
from w1a1_eagle.qat_curriculum_runner import require_provider
from w1a1_eagle.trajectory_refresh import learning_gate

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "refresh_validation_continuous_stages", ROOT / "scripts/w1ax_continuous_stages.py"
)
continuous_stages = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(continuous_stages)


def quality(rounds, accepted, proposed, emitted, accepted_emitted):
    return {
        "rounds": rounds,
        "accepted": accepted,
        "proposed": proposed,
        "emitted": emitted,
        "accepted_emitted": accepted_emitted,
    }


class RefreshContractValidation(unittest.TestCase):
    def test_actual_producer_aggregates_raw_counts_and_preserves_distinct_units(self):
        # Unequal prompt proposal counts make the pooled ratio differ from a
        # per-prompt mean. Each request also includes emitted/EOS accounting.
        cell = {
            "requests": [
                {"quality": quality(4, 2, 4, 7, 3)},
                {"quality": quality(1, 1, 8, 2, 1)},
            ]
        }
        metrics = continuous_stages.native_acceptance_metrics(cell)
        self.assertEqual(metrics["native_accepted_drafts"], 3)
        self.assertEqual(metrics["native_rounds"], 5)
        self.assertEqual(metrics["native_proposed_drafts"], 12)
        self.assertEqual(metrics["native_emitted_tokens"], 9)
        self.assertEqual(metrics["native_accepted_emitted"], 4)
        self.assertAlmostEqual(metrics["native_acceptance_rate"], 3 / 12)
        self.assertAlmostEqual(metrics["native_accepted_per_round"], 3 / 5)
        self.assertAlmostEqual(metrics["native_mean_emitted_per_round"], 9 / 5)
        self.assertNotAlmostEqual(metrics["native_acceptance_rate"], (2 / 4 + 1 / 8) / 2)

    def test_acceptance_rate_and_accepted_per_round_can_move_oppositely(self):
        # The fixed development producer reports both measures. In this
        # synthetic pair rate improves while accepted drafts per round fall.
        previous = continuous_stages.native_acceptance_metrics(
            {"requests": [{"quality": quality(20, 40, 100, 60, 45)}]}
        )
        current = continuous_stages.native_acceptance_metrics(
            {"requests": [{"quality": quality(20, 32, 40, 52, 37)}]}
        )
        self.assertGreater(current["native_acceptance_rate"], previous["native_acceptance_rate"])
        self.assertLess(current["native_accepted_per_round"], previous["native_accepted_per_round"])
        # Conversely, accepted-per-round can improve as the rate falls.
        previous = continuous_stages.native_acceptance_metrics(
            {"requests": [{"quality": quality(10, 30, 50, 40, 33)}]}
        )
        current = continuous_stages.native_acceptance_metrics(
            {"requests": [{"quality": quality(12, 48, 120, 64, 51)}]}
        )
        self.assertLess(current["native_acceptance_rate"], previous["native_acceptance_rate"])
        self.assertGreater(
            current["native_accepted_per_round"], previous["native_accepted_per_round"]
        )

    def test_learning_gate_consumes_one_bounded_rate_field_not_other_units(self):
        policy = {
            "caps": {"max_rounds": 3},
            "learning_curve": {
                "min_completed_steps": 200,
                "min_relative_ce_improvement": 0.05,
                "max_acceptance_regression": 0.02,
                "min_changed_prefix_fraction": 0.2,
            },
            "development_sample_sha256": "b" * 64,
        }

        def curve(rate):
            return {
                "schema": "w1ax_refresh_learning_curve_v1",
                "split": "development",
                "development_sample_sha256": "b" * 64,
                "checkpoint_sha256": "2" * 64,
                "previous": {
                    "checkpoint_sha256": "1" * 64,
                    "completed_steps": 100,
                    "hard_ce": 2.0,
                    "native_acceptance": 0.5,
                },
                "current": {
                    "checkpoint_sha256": "2" * 64,
                    "completed_steps": 200,
                    "hard_ce": 1.8,
                    "native_acceptance": rate,
                },
                # Richer fields proposed by the owner must not silently affect
                # the current gate until explicitly wired into it.
                "current_counts": {
                    "accepted": 32,
                    "proposed": 40,
                    "rounds": 20,
                    "emitted": 52,
                },
                "current_accepted_per_round": 1.6,
            }

        improves_per_round = curve(0.52)
        result = learning_gate(policy, improves_per_round, "2" * 64, 0.3, 1)
        self.assertEqual(result["status"], "pass")
        # Changing only the auxiliary count/rate-units leaves the gate output
        # identical; native_acceptance is its sole acceptance input.
        improves_rate = curve(0.52)
        improves_rate["current_counts"] = {
            "accepted": 48,
            "proposed": 120,
            "rounds": 12,
            "emitted": 64,
        }
        improves_rate["current_accepted_per_round"] = 4.0
        self.assertEqual(learning_gate(policy, improves_rate, "2" * 64, 0.3, 1), result)
        regresses_rate = curve(0.47)
        self.assertEqual(
            learning_gate(policy, regresses_rate, "2" * 64, 0.3, 1)["status"], "stop"
        )

    def test_plan_and_provider_admission_are_separate_handoff_gates(self):
        planned = SimpleNamespace(
            training_eligible=False,
            full_body_qat_eligible=False,
            split="train",
            readiness_scope="full_body_qat",
            source_metadata={"plan_sha256": "c" * 64},
        )
        with self.assertRaisesRegex(ValueError, "audited full-body train eligibility"):
            require_provider(planned)

        admitted = SimpleNamespace(
            training_eligible=True,
            full_body_qat_eligible=True,
            split="train",
            readiness_scope="full_body_qat",
            source_metadata={"audited_corpus_sha256": "d" * 64},
        )
        require_provider(admitted)

    def test_curriculum_resume_preserves_spent_budget_and_rejects_budget_change(self):
        config = CurriculumConfig(
            (PrecisionStage(8, 3.0, 3), PrecisionStage(1, 5.0, 3))
        )
        data, model = {"corpus": "audited-v1"}, {"checkpoint": "student-v1"}
        state = CurriculumState(config, data_contract=data, model_contract=model)
        state.record_update(
            activation_bits=8,
            gpu_seconds=1.25,
            rows=5,
            tokens=19,
            supported_rows=4,
        )
        resumed = CurriculumState(config, data_contract=data, model_contract=model)
        resumed.load_state_dict(state.state_dict())
        self.assertEqual(resumed.global_updates, 1)
        self.assertEqual(resumed.remaining_gpu_seconds, 1.75)
        resumed.record_update(
            activation_bits=8,
            gpu_seconds=1.0,
            rows=4,
            tokens=15,
            supported_rows=3,
        )
        resumed.finish_phase()
        resumed.record_transition(
            source_bits=8,
            target_bits=1,
            source_checkpoint_sha256="e" * 64,
            target_layout_sha256="f" * 64,
        )
        resumed.record_update(
            activation_bits=1,
            gpu_seconds=2.0,
            rows=2,
            tokens=7,
            supported_rows=2,
        )
        self.assertEqual(resumed.global_updates, 3)
        self.assertEqual(resumed.phases[0]["gpu_seconds"], 2.25)
        self.assertEqual(resumed.phases[1]["gpu_seconds"], 2.0)
        changed_budget = CurriculumConfig(
            (PrecisionStage(8, 4.0, 3), PrecisionStage(1, 5.0, 3))
        )
        changed = CurriculumState(changed_budget, data_contract=data, model_contract=model)
        with self.assertRaisesRegex(ValueError, "resume changes curriculum budget"):
            changed.load_state_dict(resumed.state_dict())

    def test_zero_proposal_round_is_counted_by_producer_and_receipt(self):
        raw = {"requests": [{"quality": quality(2, 1, 1, 3, 1)}]}
        metrics = continuous_stages.native_acceptance_metrics(raw)
        self.assertEqual(metrics["native_rounds"], 2)
        self.assertEqual(metrics["native_proposed_drafts"], 1)
        # Empty proposal rounds are possible producer inputs, and must still
        # contribute to the accepted/round denominator.
        previous = CurvePoint(
            "1" * 64,
            "a" * 64,
            100,
            200.0,
            100,
            (RoundCounts("p", 0, 0, 1, 3), RoundCounts("p", 1, 1, 2, 3)),
        )
        current = CurvePoint(
            "2" * 64,
            "b" * 64,
            200,
            180.0,
            100,
            (RoundCounts("p", 0, 0, 1, 2), RoundCounts("p", 1, 1, 2, 2)),
        )
        self.assertEqual(previous.counts()["rounds"], metrics["native_rounds"])
        self.assertEqual(previous.counts()["proposed"], metrics["native_proposed_drafts"])
        self.assertEqual(previous.counts()["accepted"], metrics["native_accepted_drafts"])
        self.assertEqual(previous.evidence()["metrics"]["accepted_per_round"]["value"], 0.5)
        self.assertEqual(current.counts()["rounds"], 2)
        self.assertEqual(current.counts()["accepted"], 1)

    def test_typed_receipt_uses_pooled_counts_and_replays_exact_gate(self):
        policy = {
            "caps": {"max_rounds": 3},
            "learning_curve": {
                "min_completed_steps": 200,
                "min_relative_ce_improvement": 0.05,
                "max_acceptance_regression": 0.02,
                "min_changed_prefix_fraction": 0.2,
            },
            "development_sample_sha256": "b" * 64,
        }
        previous = CurvePoint(
            "1" * 64,
            "a" * 64,
            100,
            200.0,
            100,
            (
                RoundCounts("short", 2, 4, 3, 4),
                RoundCounts("long", 1, 5, 2, 5),
            ),
        )
        current = CurvePoint(
            "2" * 64,
            "c" * 64,
            200,
            180.0,
            100,
            (
                RoundCounts("short", 4, 5, 5, 5),
                RoundCounts("long", 1, 5, 2, 5),
            ),
        )
        receipt = decision_receipt(
            policy,
            previous,
            current,
            changed_fraction=0.3,
            round_index=1,
            comparison_contract_sha256="d" * 64,
        )
        self.assertEqual(receipt["previous"]["metrics"]["accepted_per_proposed"]["value"], 3 / 9)
        self.assertEqual(receipt["previous"]["metrics"]["accepted_per_round"]["value"], 3 / 2)
        self.assertEqual(receipt["selected_unit"], "accepted_draft_tokens/proposed_draft_tokens")
        self.assertEqual(receipt["gate"], learning_gate(
            policy,
            receipt["legacy_curve_input"],
            current.checkpoint_sha256,
            0.3,
            1,
        ))
        self.assertFalse(receipt["training_authorized"])

    def test_new_experiment_handoff_reuses_cumulative_ledger_and_original_window(self):
        ledger = BudgetLedger(
            authorization_sha256="9" * 64,
            authorized_gpu_seconds=1000.0,
            original_research_reset_unix=1791049896,
            entries=(
                LedgerEntry("old", "training_residency", 650.0),
                LedgerEntry("old", "capture", 100.0),
                LedgerEntry("old", "audit", 50.0),
            ),
        )
        admission = CorpusAdmission(
            corpus_sha256="3" * 64,
            plan_sha256="4" * 64,
            synthetic_provider_audit_sha256="5" * 64,
            missing_requirements=0,
            full_body_train_eligible=True,
        )
        control = {
            "research_stop": False,
            "reset_observed": False,
            "last_reset_unix": 1791049896,
            "last_weekly_used_percent": 82,
        }
        common = {
            "old_experiment_id": "old",
            "new_experiment_id": "new",
            "old_corpus_sha256": "6" * 64,
            "source_checkpoint_sha256": "7" * 64,
            "admission": admission,
            "ledger": ledger,
            "control": control,
        }
        fits = new_experiment_proposal(**common, requested_gpu_seconds=150.0)
        self.assertTrue(fits["admissible_proposal"])
        self.assertFalse(fits["training_authorized"])
        self.assertFalse(fits["research_window_resets"])
        self.assertEqual(fits["remaining_gpu_seconds"], 200.0)
        over = new_experiment_proposal(**common, requested_gpu_seconds=250.0)
        self.assertIn("cumulative_gpu_budget_exhausted", over["reasons"])
        reset_control = {**control, "last_reset_unix": 1791049999, "reset_observed": True}
        reset = new_experiment_proposal(
            **{**common, "control": reset_control}, requested_gpu_seconds=150.0
        )
        self.assertIn("original_research_allowance_closed_or_unknown", reset["reasons"])


if __name__ == "__main__":
    unittest.main()
