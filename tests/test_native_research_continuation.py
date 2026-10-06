"""Genuine-shaped CPU provenance fixtures; no native quality/admission claims."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import evaluate_nine_model_timed_checkpoint as evaluator  # noqa: E402

from w1a1_eagle import continuous_budget as api  # noqa: E402


class ResearchContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = 0

    def write(self, value, name=None):
        self.index += 1
        path = self.root / (name or f"artifact-{self.index}.json")
        path.write_text(json.dumps(value))
        return api.artifact_locator(path)

    def fixture(self):
        candidate = "dspark_a8"
        known = "gsm8k:train-006474"
        prompts = [
            {
                "id": known if i == 23 else f"dev-{i}",
                "messages": [{"role": "user", "content": "CPU fixture"}],
            }
            for i in range(24)
        ]
        prompt_file = self.root / "prompts.jsonl"
        prompt_file.write_text("\n".join(json.dumps(r) for r in prompts))
        target, model, q4, binary = [
            self.write({"CPU_fixture": name}) for name in ("target", "initial", "q4", "runtime")
        ]
        plan = {
            "candidate": candidate,
            "target": target,
            "target_policy": {"immutable": True, "weights": "f16", "kv": "f16"},
            "runtime": {"binary": binary},
            "protocol": self.write({"repetitions": 5}),
            "prompts": api.artifact_locator(prompt_file),
            "initial": {"model": model},
            "control": {"model": q4},
            "gpu_uuid": "GPU-CPU-fixture",
            "frozen_lane": self.write({"fixture": True}),
            "remaining_candidates": ["other"],
        }
        records, diagnostics = [], []
        for diagnostic, repeats in ((False, range(5)), (True, range(1))):
            for rep in repeats:
                for prompt in prompts:
                    for cell in (candidate, "initial", "eagle_q4", "target_only"):
                        tokens = (
                            [1, 3] if prompt["id"] == known and cell != "target_only" else [1, 2]
                        )
                        counters = (
                            None
                            if cell == "target_only"
                            else {"proposed": 4, "accepted": 2, "rounds": 2}
                        )
                        raw = self.write(
                            {
                                "generated_token_ids": tokens,
                                "completion_tokens": 2,
                                "request_wall_s": 1.0,
                                "finish_reason": "stop",
                                "speculative": counters,
                            }
                        )
                        row = {
                            "cell": cell,
                            "repetition": rep,
                            "prompt_id": prompt["id"],
                            "output_tokens": 2,
                            "latency_s": 1.0,
                            "generated_token_ids": tokens,
                            "speculative": counters,
                            "raw_result": raw,
                        }
                        if diagnostic and counters:
                            row["round_summary"] = {"rounds": 2}
                        (diagnostics if diagnostic else records).append(row)
        progress = self.write({"clean_records": records, "diagnostic_records": diagnostics})
        binding = api.research_bindings(plan)
        hardware = {"compute_capability": [12, 0], "gpu_uuid": plan["gpu_uuid"]}
        resource = self.write(
            {
                "release": {
                    "owned_process_groups_absent": True,
                    "owned_cuda_pids_absent": True,
                    "other_context_pids": [],
                },
                "resources": {"gpu_uuid": plan["gpu_uuid"], "dxg_holders": []},
            }
        )
        dispatch = self.write(
            {
                "schema": "nine_model_observed_w1_dispatch_v1",
                "status": "PASS",
                "activation_bits": 8,
                "packed_names": [f"fixture{i}" for i in range(16)],
            }
        )
        evidence = self.write(
            {
                "schema": "native_zero_measurement_evidence_v1",
                "progress": progress,
                "bindings": binding,
                "hardware": hardware,
                "failed_status": self.write(
                    {"status": "FAILED", "reason": "strict parity CPU fixture"}
                ),
                "zero_preparation": self.write(
                    {
                        "schema": "nine_model_preparation_v1",
                        "status": "PASS",
                        "optimizer_updates": 0,
                        "checkpoint": {"cursor": {"step": 0}},
                    }
                ),
                "model_ancestry": {candidate: model, "initial": model, "eagle_q4": q4},
                "resource_returns": [self.write(api._research_json(resource)) for _ in range(24)],
                "dispatch_proofs": [self.write(api._research_json(dispatch)) for _ in range(2)],
            }
        )
        baseline = evaluator.collect_failed_zero_baseline(
            plan, progress, evidence, self.root / "baseline.json"
        )
        report = api._research_json(baseline)
        sources = {
            name: api.artifact_locator(ROOT / name)
            for name in (
                "scripts/run_nine_model_lane_endpoint.py",
                "scripts/evaluate_nine_model_timed_checkpoint.py",
                "scripts/run_nine_model_lane.py",
                "src/w1a1_eagle/continuous_budget.py",
            )
        }
        body = {
            "schema": "native_research_continuation_policy_v1",
            "scope": "research_only_no_deployment_admission",
            "strict_quality_status": "FAILED",
            "bindings": binding,
            "baseline_report": baseline,
            "allowed_cases": report["allowed_cases"],
            "controller_sources": sources,
        }
        review = self.write(
            {
                "schema": "native_research_operational_review_v1",
                "status": "REVIEWED",
                "policy_body_sha256": api._research_digest(body),
                "new_scientific_approval_claimed": False,
                "approved_plan": self.write({"fixture": "approved existing plan"}),
                "standing_start_authorization": self.write({"fixture": "existing human start"}),
                "historical_failure": self.write(
                    {"strict_quality_status": "FAILED", "allowed_cases": report["allowed_cases"]}
                ),
            }
        )
        policy = self.write({**body, "review": review})
        return plan, binding, records, diagnostics, policy, baseline

    def test_failed_science_retained_but_opt_in_research_can_continue(self):
        plan, binding, records, diagnostics, policy, baseline = self.fixture()
        report = api._research_json(baseline)
        self.assertEqual(report["strict_quality_status"], "FAILED")
        self.assertEqual(report["continuation_status"], "NOT_AUTHORIZED")
        decision = api.research_continuation_decision(policy, binding, records, diagnostics)
        self.assertEqual(decision["strict_quality_status"], "FAILED")
        self.assertEqual(decision["continuation_status"], "RESEARCH_CONTINUATION_ALLOWED")
        with self.assertRaisesRegex(ValueError, "greedy verifier"):
            evaluator.matched_report(
                plan,
                records,
                diagnostics,
                {"compute_capability": [12, 0], "gpu_uuid": plan["gpu_uuid"]},
            )
        allowed = evaluator.matched_report(
            {**plan, "research_continuation_policy": policy},
            records,
            diagnostics,
            {"compute_capability": [12, 0], "gpu_uuid": plan["gpu_uuid"]},
        )
        self.assertEqual(allowed["strict_quality_status"], "FAILED")

    def mutate_raw(self, row, *, tokens=None, finish=None):
        raw = api._research_json(row["raw_result"])
        if tokens is not None:
            row["generated_token_ids"] = tokens
            raw["generated_token_ids"] = tokens
        if finish is not None:
            raw["finish_reason"] = finish
        row["raw_result"] = self.write(raw)

    def test_new_mismatch_changed_controls_third_branch_finish_and_missing_rows_stop(self):
        _, binding, records, diagnostics, policy, _ = self.fixture()
        for kind in ("outside", "third", "control", "finish", "missing", "counter"):
            with self.subTest(kind=kind):
                altered = copy.deepcopy(records)
                if kind == "missing":
                    altered.pop()
                else:
                    known = kind in {"third", "control"}
                    cell = "eagle_q4" if kind == "control" else binding["candidate"]
                    row = next(
                        r
                        for r in altered
                        if r["cell"] == cell and (r["prompt_id"] == "gsm8k:train-006474") == known
                    )
                    if kind == "finish":
                        self.mutate_raw(row, finish="length")
                    elif kind == "counter":
                        row["speculative"] = None
                    else:
                        self.mutate_raw(row, tokens=[1, 99])
                with self.assertRaises(ValueError):
                    api.research_continuation_decision(policy, binding, altered, diagnostics)

    def test_changed_model_runtime_protocol_policy_review_and_raw_pin_stop(self):
        _, binding, records, diagnostics, policy, _ = self.fixture()
        for field in ("initial_model", "q4_model", "target", "protocol", "runtime"):
            altered = copy.deepcopy(binding)
            new = self.write({"different": field})
            altered[field] = {"binary": new} if field == "runtime" else new
            with self.assertRaises(ValueError):
                api.research_continuation_decision(policy, altered, records, diagnostics)
        data = api._research_json(policy)
        data["allowed_cases"]["gsm8k:train-006474"]["first_divergence_index"] += 1
        with self.assertRaises(ValueError):
            api.validate_research_policy(self.write(data), binding)
        Path(records[0]["raw_result"]["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            api.validate_research_policy(policy, binding)

    def test_receipt_consumer_actively_rechecks_failed_policy_and_rejects_scientific_rename(self):
        plan, binding, records, diagnostics, policy, _ = self.fixture()
        report = api.research_continuation_decision(policy, binding, records, diagnostics)
        report["measurements"] = self.write(
            {"clean_records": records, "diagnostic_records": diagnostics}
        )
        evaluation = self.write(report)
        result = {
            **report,
            "schema": "nine_model_timed_evaluation_receipt_v1",
            "status": "FAILED",
            "completed": True,
            "evaluation": evaluation,
            "evaluation_plan": self.write({**plan, "research_continuation_policy": policy}),
            "candidate": plan["candidate"],
            "protocol": plan["protocol"],
        }
        self.assertTrue(api.validate_evaluation_continuation(result))
        with self.assertRaises(ValueError):
            api.validate_evaluation_continuation({**result, "continuation_status": "PASS"})
        with self.assertRaises(ValueError):
            api.validate_evaluation_continuation({**result, "status": "PASS"})
        # A strict failed receipt without opt-in cannot resume.
        with self.assertRaises(ValueError):
            api.validate_evaluation_continuation({"status": "FAILED", "completed": True})


if __name__ == "__main__":
    unittest.main()
