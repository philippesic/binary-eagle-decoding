"""Write fabricated receipt/handoff fixtures; uses existing synthetic test data."""

import json
from dataclasses import replace
from pathlib import Path

from test_trajectory_refresh import RefreshTests

from research.parallel20261002.refresh_contract.reference.contract import (
    BudgetLedger,
    CorpusAdmission,
    CurvePoint,
    LedgerEntry,
    RoundCounts,
    decision_receipt,
    new_experiment_proposal,
)
from w1a1_eagle.qat_curriculum import plan_curriculum_refresh
from w1a1_eagle.trajectory_refresh import digest

OUTPUT = Path(__file__).resolve().parents[4] / "experiments/parallel20261002/refresh_contract"


def paired_receipts():
    policy = {
        "development_sample_sha256": "b" * 64,
        "caps": {"max_rounds": 2},
        "learning_curve": {"min_completed_steps": 200, "min_relative_ce_improvement": 0.05,
                           "max_acceptance_regression": 0.02,
                           "min_changed_prefix_fraction": 0.2},
    }
    wide = (RoundCounts("p0", 3, 5, 4, 5), RoundCounts("p1", 2, 5, 1, 5, True))
    narrow = (RoundCounts("p0", 1, 1, 2, 1), RoundCounts("p1", 1, 1, 1, 1, True))
    first = CurvePoint("0" * 64, "1" * 64, 100, 20.0, 10, wide)
    second = CurvePoint("2" * 64, "3" * 64, 200, 18.0, 10, narrow)
    kwargs = {"changed_fraction": 0.5, "round_index": 0,
              "comparison_contract_sha256": "4" * 64}
    return {
        "rate_rises_drafts_per_round_falls": decision_receipt(policy, first, second, **kwargs),
        "rate_falls_drafts_per_round_rises": decision_receipt(
            policy, replace(first, rounds=narrow), replace(second, rounds=wide), **kwargs),
    }


def handoff_fixture():
    # This reuses fabricated prompts/teacher bytes from an existing CPU test.
    fixture = RefreshTests()
    fixture.setUp()
    try:
        bindings = {
            "expected_checkpoint_sha256": fixture.identity["checkpoint_sha256"],
            "expected_export_sha256": fixture.identity["export_sha256"],
            "expected_train_prompts_sha256": fixture.policy["train_prompts_sha256"],
            "expected_native_teacher_contract": fixture.contract,
            "expected_refresh_round": 0,
        }
        planned = plan_curriculum_refresh(fixture.spec, **bindings)
        # Fabricate the newly requested labels, then rerun the actual planner.
        for row in planned["capture_requests"]:
            fixture.teacher_rows.append({
                "kind": row["kind"], "prompt_id": row["prompt_id"], "split": "train",
                "prefix_token_ids": row["prefix_token_ids"], "artifact": "native",
                "row": len(fixture.teacher_rows), "capture_id": "synthetic-refresh",
                "next_target_id": 7,
            })
        fixture.sync()
        complete = plan_curriculum_refresh(fixture.spec, **bindings)
        summary = {
            "schema": "synthetic_planner_pair_manifest_v1", "synthetic": True,
            "student": fixture.identity,
            "planned": {k: planned[k] for k in (
                "counts", "capture_queue_ready", "training_eligible", "readiness")},
            "after_fabricated_capture": {k: complete[k] for k in (
                "counts", "capture_queue_ready", "training_eligible", "readiness")},
            "provider_audit": "separate synthetic simulation; no payload audited",
        }
    finally:
        fixture.doCleanups()
    ledger = BudgetLedger("5" * 64, 20.0, 1791049896, (
        LedgerEntry("old", "training_residency", 12),
        LedgerEntry("refresh-preflight", "capture", 2),
        LedgerEntry("refresh-preflight", "audit", 1),
        LedgerEntry("refresh-preflight", "export", 1),
        LedgerEntry("old", "development", 1),
    ))
    control = {"research_stop": False, "reset_observed": False,
               "last_reset_unix": 1791049896, "last_weekly_used_percent": 82}
    admission = CorpusAdmission("7" * 64, digest(summary),
                                digest({"synthetic_provider_pass": True}), 0, True)
    kwargs = {
        "old_experiment_id": "old", "new_experiment_id": "new",
        "old_corpus_sha256": "6" * 64,
        "source_checkpoint_sha256": summary["student"]["checkpoint_sha256"],
        "ledger": ledger, "requested_gpu_seconds": 3, "control": control,
    }
    return {
        "synthetic": True, "planner_manifests": summary,
        "planned_only": new_experiment_proposal(**kwargs, admission=None),
        "audited_simulation": new_experiment_proposal(**kwargs, admission=admission),
        "spent_budget": new_experiment_proposal(
            **{**kwargs, "ledger": replace(ledger, authorized_gpu_seconds=17)},
            admission=admission),
        "fresh_research_window": new_experiment_proposal(
            **{**kwargs, "control": {**control, "last_reset_unix": 1791654696}},
            admission=admission),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = {"paired_policy_fixtures.json": paired_receipts(),
              "handoff_fixtures.json": handoff_fixture()}
    for name, value in values.items():
        (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"synthetic": True, "files": list(values),
                      "paired_gate_status": {k: v["gate"]["status"]
                                             for k, v in values[
                                                 "paired_policy_fixtures.json"].items()}}))


if __name__ == "__main__":
    main()
