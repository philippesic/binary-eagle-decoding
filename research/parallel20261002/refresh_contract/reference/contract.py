"""Synthetic CPU receipt proposal; no training/capture/authorization side effects.

The adapter preserves the existing accepted/proposed refresh decision. Changing
the selected metric or accepting this proposal into production is owner-owned.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from w1a1_eagle.trajectory_refresh import checked_hash, digest, learning_gate


@dataclass(frozen=True)
class RoundCounts:
    prompt_id: str
    accepted: int
    proposed: int
    emitted: int
    draft_cap: int
    eos: bool = False

    def __post_init__(self):
        if not self.prompt_id or type(self.eos) is not bool:
            raise ValueError("named synthetic prompt and EOS disposition required")
        if any(type(x) is not int for x in (
            self.accepted, self.proposed, self.emitted, self.draft_cap
        )) or not (
            0 <= self.accepted <= self.proposed <= self.draft_cap <= 5
            and self.draft_cap >= 1 and 0 <= self.emitted <= self.accepted + 1
        ):
            raise ValueError("invalid synthetic speculative-round counts")
        if not self.eos and self.emitted != self.accepted + 1:
            raise ValueError("this fixture scope only clips emitted tokens at EOS")


@dataclass(frozen=True)
class CurvePoint:
    checkpoint_sha256: str
    export_sha256: str
    completed_steps: int
    ce_loss_sum: float
    ce_rows: int
    rounds: tuple[RoundCounts, ...]

    def __post_init__(self):
        checked_hash(self.checkpoint_sha256)
        checked_hash(self.export_sha256)
        if type(self.completed_steps) is not int or self.completed_steps < 1:
            raise ValueError("positive completed steps required")
        if (type(self.ce_rows) is not int or self.ce_rows < 1
                or type(self.ce_loss_sum) not in (int, float)
                or not math.isfinite(self.ce_loss_sum) or self.ce_loss_sum < 0):
            raise ValueError("finite supported-row CE evidence required")
        if not self.rounds:
            raise ValueError("nonempty synthetic quality rounds required")
        if sum(r.proposed for r in self.rounds) == 0:
            raise ValueError("aggregate synthetic quality evidence has no proposals")

    def counts(self) -> dict:
        return {
            "accepted": sum(r.accepted for r in self.rounds),
            "proposed": sum(r.proposed for r in self.rounds),
            "rounds": len(self.rounds),
            "emitted": sum(r.emitted for r in self.rounds),
            "accepted_emitted": sum(min(r.accepted, r.emitted) for r in self.rounds),
        }

    def evidence(self) -> dict:
        counts = self.counts()
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "export_sha256": self.export_sha256,
            "completed_steps": self.completed_steps,
            "ce_loss_sum": self.ce_loss_sum,
            "ce_supported_rows": self.ce_rows,
            "raw_rounds": [asdict(r) for r in self.rounds],
            "counts": counts,
            "metrics": {
                "accepted_per_proposed": {
                    "value": counts["accepted"] / counts["proposed"],
                    "unit": "accepted_draft_tokens/proposed_draft_tokens",
                },
                "accepted_per_round": {
                    "value": counts["accepted"] / counts["rounds"],
                    "unit": "accepted_draft_tokens/speculative_round",
                },
                "emitted_per_round": {
                    "value": counts["emitted"] / counts["rounds"],
                    "unit": "emitted_tokens/speculative_round",
                },
                "hard_ce": {
                    "value": self.ce_loss_sum / self.ce_rows,
                    "unit": "nats/supported_label_row",
                },
            },
        }


def decision_receipt(policy: dict, previous: CurvePoint, current: CurvePoint, *,
                     changed_fraction: float, round_index: int,
                     comparison_contract_sha256: str) -> dict:
    """Reproduce the actual callable gate from weighted raw counts, never ratios.

    comparison_contract_sha256 identifies frozen sample, target/verifier,
    hardware and execution settings. It permits stated draft-cap differences;
    synthetic fixtures do not prove a real execution contract or measurement.
    """
    checked_hash(comparison_contract_sha256)
    if {r.prompt_id for r in previous.rounds} != {r.prompt_id for r in current.rounds}:
        raise ValueError("paired synthetic prompt sets differ")
    if type(changed_fraction) not in (int, float) or not 0 <= changed_fraction <= 1:
        raise ValueError("changed prefix fraction must be bounded")
    if type(round_index) is not int or round_index < 0:
        raise ValueError("refresh round must be nonnegative")
    old, new = previous.evidence(), current.evidence()
    curve = {
        "schema": "w1ax_refresh_learning_curve_v1", "split": "development",
        "development_sample_sha256": policy["development_sample_sha256"],
        "checkpoint_sha256": current.checkpoint_sha256,
        **{name: {
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "completed_steps": evidence["completed_steps"],
            "hard_ce": evidence["metrics"]["hard_ce"]["value"],
            "native_acceptance": evidence["metrics"]["accepted_per_proposed"]["value"],
        } for name, evidence in (("previous", old), ("current", new))},
    }
    thresholds = policy["learning_curve"]
    lower = curve["previous"]["native_acceptance"] - thresholds["max_acceptance_regression"]
    ce_improvement = ((curve["previous"]["hard_ce"] - curve["current"]["hard_ce"])
                      / max(curve["previous"]["hard_ce"], 1e-12))
    return {
        "schema": "synthetic_refresh_decision_receipt_proposal_v1", "synthetic": True,
        "training_authorized": False, "policy_sha256": digest(policy),
        "gate_policy": policy,
        "comparison_contract_sha256": comparison_contract_sha256,
        "selected_metric": "accepted_per_proposed",
        "selected_unit": "accepted_draft_tokens/proposed_draft_tokens",
        "aggregation": "sum_raw_counts_then_divide",
        "previous": old, "current": new, "legacy_curve_input": curve,
        "comparisons": {
            "acceptance_regression": {
                "left": curve["current"]["native_acceptance"], "operator": "<",
                "right": lower, "unit": "accepted_draft_tokens/proposed_draft_tokens",
                "result": curve["current"]["native_acceptance"] < lower,
            },
            "insufficient_steps": current.completed_steps < thresholds["min_completed_steps"],
            "insufficient_ce": ce_improvement < thresholds["min_relative_ce_improvement"],
            "relative_ce_improvement": ce_improvement,
            "ce_threshold": thresholds["min_relative_ce_improvement"],
            "insufficient_changed_prefix": changed_fraction
            < thresholds["min_changed_prefix_fraction"],
            "changed_prefix_fraction": changed_fraction,
            "changed_prefix_threshold": thresholds["min_changed_prefix_fraction"],
            "step_threshold": thresholds["min_completed_steps"],
            "refresh_round": round_index, "refresh_round_cap": policy["caps"]["max_rounds"],
            "round_cap_exhausted": round_index >= policy["caps"]["max_rounds"],
        },
        "gate": learning_gate(policy, curve, current.checkpoint_sha256,
                              changed_fraction, round_index),
    }


@dataclass(frozen=True)
class LedgerEntry:
    experiment_id: str
    kind: str
    gpu_seconds: float

    def __post_init__(self):
        if not self.experiment_id or self.kind not in {
            "training_residency", "capture", "audit", "export", "development"
        } or type(self.gpu_seconds) not in (int, float) or not (
            math.isfinite(self.gpu_seconds) and self.gpu_seconds >= 0
        ):
            raise ValueError("named nonoverlapping GPU charge required")


@dataclass(frozen=True)
class BudgetLedger:
    authorization_sha256: str
    authorized_gpu_seconds: float
    original_research_reset_unix: int
    entries: tuple[LedgerEntry, ...]

    def __post_init__(self):
        checked_hash(self.authorization_sha256)
        if type(self.authorized_gpu_seconds) not in (int, float) or not (
            math.isfinite(self.authorized_gpu_seconds) and self.authorized_gpu_seconds >= 0
        ) or type(self.original_research_reset_unix) is not int:
            raise ValueError("explicit synthetic allowance and original window required")

    @property
    def remaining(self):
        return self.authorized_gpu_seconds - sum(e.gpu_seconds for e in self.entries)


@dataclass(frozen=True)
class CorpusAdmission:
    corpus_sha256: str
    plan_sha256: str
    synthetic_provider_audit_sha256: str
    missing_requirements: int
    full_body_train_eligible: bool

    def __post_init__(self):
        for value in (self.corpus_sha256, self.plan_sha256,
                      self.synthetic_provider_audit_sha256):
            checked_hash(value)
        if type(self.missing_requirements) is not int or self.missing_requirements < 0:
            raise ValueError("bounded missing requirement count required")
        if type(self.full_body_train_eligible) is not bool:
            raise ValueError("explicit synthetic provider admission disposition required")


def new_experiment_proposal(*, old_experiment_id: str, new_experiment_id: str,
                            old_corpus_sha256: str, source_checkpoint_sha256: str,
                            admission: CorpusAdmission | None, ledger: BudgetLedger,
                            requested_gpu_seconds: float, control: dict) -> dict:
    """Synthetic admission simulation; never creates a runner or grants training.

    Corpus audit and budget are separate. A new identity/fresh counters alone
    never replenish the cumulative ledger or the original research allowance.
    """
    checked_hash(old_corpus_sha256)
    checked_hash(source_checkpoint_sha256)
    if type(requested_gpu_seconds) not in (int, float) or not (
        math.isfinite(requested_gpu_seconds) and requested_gpu_seconds > 0
    ):
        raise ValueError("positive requested bounded GPU reservation required")
    reasons = []
    if not old_experiment_id or not new_experiment_id or old_experiment_id == new_experiment_id:
        reasons.append("distinct_new_experiment_required")
    if (admission is None or admission.missing_requirements != 0
            or admission.full_body_train_eligible is not True):
        reasons.append("audited_new_corpus_required")
    elif admission.corpus_sha256 == old_corpus_sha256:
        reasons.append("changed_corpus_required_for_this_handoff")
    if requested_gpu_seconds > ledger.remaining:
        reasons.append("cumulative_gpu_budget_exhausted")
    if (control.get("research_stop") is not False or control.get("reset_observed") is not False
            or control.get("last_reset_unix") != ledger.original_research_reset_unix
            or type(control.get("last_weekly_used_percent")) not in (int, float)
            or not math.isfinite(control["last_weekly_used_percent"])
            or not 0 <= control["last_weekly_used_percent"] < 99):
        reasons.append("original_research_allowance_closed_or_unknown")
    return {
        "schema": "synthetic_refresh_handoff_proposal_v1", "synthetic": True,
        "training_authorized": False, "admissible_proposal": not reasons,
        "reasons": reasons, "old_experiment_id": old_experiment_id,
        "new_experiment_id": new_experiment_id, "resume_mode": "new_experiment",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "old_corpus_sha256": old_corpus_sha256,
        "admission": asdict(admission) if admission else None,
        "initialization": "explicit_parent_weight_import_after_separate_model_gates",
        "optimizer_state": "fresh", "new_experiment_updates": 0,
        "cumulative_ledger": asdict(ledger), "remaining_gpu_seconds": ledger.remaining,
        "requested_gpu_seconds": requested_gpu_seconds,
        "research_window_resets": False,
        "remaining_real_gates": ["human_experiment_scope_and_budget",
                                 "audited_provider_factory", "native_cuda_readiness",
                                 "all_stage_zero_update_smoke", "exclusive_gpu_ownership"],
    }
