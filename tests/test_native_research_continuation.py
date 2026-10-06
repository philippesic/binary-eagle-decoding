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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return api.artifact_locator(path)

    def fixture(self):
        candidate = "dspark_a8"
        known = "gsm8k:train-006474"
        prompts = [
            {
                "id": known if i == 23 else f"dev-{i}",
                "messages": [{"role": "user", "content": f"CPU fixture {i}"}],
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
            "protocol": self.write(
                {
                    "repetitions": 5,
                    "context_tokens": 2048,
                    "batch_tokens": 32,
                    "microbatch_tokens": 32,
                    "max_output_tokens": 128,
                    "seed": 42,
                    "draft_lengths": {"eagle": 5, "dspark": 7},
                }
            ),
            "prompts": api.artifact_locator(prompt_file),
            "initial": {"model": model},
            "control": {"model": q4},
            "gpu_uuid": "GPU-CPU-fixture",
            "frozen_lane": self.write({"fixture": True}),
            "remaining_candidates": ["other"],
        }
        from benchmark_native_eagle import request_body

        from w1a1_eagle.nine_model_pipeline import validate_cuda_dispatch

        models = {candidate: model, "initial": model, "eagle_q4": q4}
        producer_sources = {
            name: api.artifact_locator(ROOT / name)
            for name in (
                "scripts/evaluate_nine_model_timed_checkpoint.py",
                "scripts/benchmark_native_eagle.py",
                "src/w1a1_eagle/nine_model_pipeline.py",
            )
        }
        stages = []
        audits = {}
        for cell in (candidate, "initial"):
            audits[cell] = self.write(
                {
                    "serialization_audit_passed": True,
                    "output": models[cell],
                    "projections": {f"fixture{i}": {"shape": [2, 32]} for i in range(16)},
                }
            )
        request_config = {
            "evaluation": {
                "max_output_tokens": 128,
                "temperature": 0.0,
                "seed": 42,
                "enable_thinking": False,
            }
        }
        records, diagnostics = [], []
        for diagnostic, repeats in ((False, range(5)), (True, range(1))):
            for rep in repeats:
                for cell in (candidate, "initial", "eagle_q4", "target_only"):
                    stage_path = (
                        self.root / ("diagnostic" if diagnostic else f"rep-{rep:02d}") / cell
                    )
                    stage_path.mkdir(parents=True)
                    pid = 100 + len(stages)
                    identity = {
                        "pid": pid,
                        "boot_id": "CPU-fixture-boot",
                        "start_ticks": 1000 + pid,
                    }
                    argv = [
                        binary["path"],
                        "-m",
                        target["path"],
                        "--ctx-size",
                        "2048",
                        "--batch-size",
                        "32",
                        "--ubatch-size",
                        "32",
                        "--cache-type-k",
                        "f16",
                        "--cache-type-v",
                        "f16",
                        "--spec-type",
                        "none" if cell == "target_only" else "draft-eagle3",
                    ]
                    if cell != "target_only":
                        argv += [
                            "-md",
                            models[cell]["path"],
                            "--spec-draft-n-max",
                            "5" if cell == "eagle_q4" else "7",
                            "--spec-draft-p-min",
                            "0",
                            "--spec-draft-type-k",
                            "f16",
                            "--spec-draft-type-v",
                            "f16",
                        ]
                    process = self.write(
                        {"pid": pid, "pgid": pid, "kernel_identity": identity, "argv": argv},
                        str(stage_path / "process.json"),
                    )
                    lineage = self.write(
                        {"kernel_identities": [identity]}, str(stage_path / "process-lineage.json")
                    )
                    resource = self.write(
                        {
                            "release": {
                                "owned_process_groups_absent": True,
                                "owned_cuda_pids_absent": True,
                                "other_context_pids": [],
                            },
                            "resources": {
                                "gpu_uuid": plan["gpu_uuid"],
                                "dxg_holders": [],
                                "boot_id": "CPU-fixture-boot",
                            },
                        },
                        str(stage_path / "resource-return.json"),
                    )
                    log_path = stage_path / "server.log"
                    trace = [
                        {
                            "schema": "w1ax_cuda_dispatch_v1",
                            "backend": "CUDA",
                            "device": 0,
                            "activation_bits": 8,
                            "packed": f"fixture{i}.w1a1_packed",
                            "logical_k": 32,
                            "rows": 2,
                            "packed_type": "i32",
                            "packed_words": 1,
                            "tokens": 7,
                        }
                        for i in range(16)
                    ]
                    log_path.write_text(
                        "\n".join("W1AX_ADMISSION_TRACE " + json.dumps(r) for r in trace)
                    )
                    stage = {
                        "cell": cell,
                        "repetition": rep,
                        "diagnostic": diagnostic,
                        "process": process,
                        "lineage": lineage,
                        "resource_return": resource,
                        "server_log": api.artifact_locator(log_path),
                        "model": target if cell == "target_only" else models[cell],
                    }
                    if diagnostic and cell in audits:
                        dispatch = self.write(
                            validate_cuda_dispatch(
                                log_path.read_text(),
                                api._research_json(audits[cell]),
                                activation_bits=8,
                            ),
                            str(stage_path / "actual-dispatch.json"),
                        )
                        stage.update(dispatch=dispatch, export_audit=audits[cell])
                    stages.append(stage)
                for ordinal, prompt in enumerate(prompts):
                    for cell in (candidate, "initial", "eagle_q4", "target_only"):
                        tokens = (
                            [1, 3] if prompt["id"] == known and cell != "target_only" else [1, 2]
                        )
                        counters = (
                            None
                            if cell == "target_only"
                            else {"proposed": 4, "accepted": 2, "rounds": 2}
                        )
                        stage_path = (
                            self.root / ("diagnostic" if diagnostic else f"rep-{rep:02d}") / cell
                        )
                        request_directory = stage_path / f"prompt-{ordinal:04d}"
                        raw_request = self.write(
                            request_body(request_config, prompt),
                            str(request_directory / "request.json"),
                        )
                        raw = self.write(
                            {
                                "generated_token_ids": tokens,
                                "completion_tokens": 2,
                                "request_wall_s": 1.0,
                                "finish_reason": "stop",
                                "speculative": counters,
                            },
                            str(request_directory / "measurement.json"),
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
                            "raw_request": raw_request,
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
                "stages": stages,
                "producer_sources": producer_sources,
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
        directory = self.root / f"mutated-{self.index}"
        row["raw_request"] = self.write(
            api._research_json(row["raw_request"]), str(directory / "request.json")
        )
        row["raw_result"] = self.write(raw, str(directory / "measurement.json"))

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

    def receipt_fixture(self, plan, binding, records, diagnostics, policy):
        baseline = api._research_json(api._research_json(policy)["baseline_report"])
        evidence = api._research_json(baseline["collection_evidence"])
        report = api.research_continuation_decision(policy, binding, records, diagnostics)
        checkpoint = self.write({"fixture": "actual CPU checkpoint bytes"})
        config = self.write({"fixture": "actual config"})
        counters = {"step": 1, "elapsed_seconds": 14400.0}
        request = self.write(
            {"checkpoint": checkpoint, "protocol": plan["protocol"], "counters": counters}
        )
        projection_manifest = self.write({"projections": {f"fixture{i}": {} for i in range(16)}})
        npz = self.write({"fixture": "actual projection input bytes"})
        training = self.write(
            {
                "schema": "nine_model_stage_receipt_v1",
                "status": "PASS",
                "committed": True,
                "checkpoint": checkpoint,
                "bundle_sha256": plan["frozen_lane"]["sha256"],
                "config_sha256": config["sha256"],
                "timed_evaluation_request": request,
                "counters": counters,
                "exports": {
                    plan["candidate"]: {"checkpoint": npz, "manifest": projection_manifest}
                },
            }
        )
        model = plan["initial"]["model"]
        audit = self.write(
            {
                "serialization_audit_passed": True,
                "family": "dspark",
                "output": model,
                "checkpoint": {"sha256": npz["sha256"]},
                "manifest": {"sha256": projection_manifest["sha256"]},
                "projections": {f"fixture{i}": {"shape": [2, 32]} for i in range(16)},
            }
        )
        exported = {
            "schema": "nine_model_lane_endpoint_export_v1",
            "artifact_kind": "production",
            "candidate": plan["candidate"],
            "checkpoint": checkpoint,
            "training_receipt": training,
            "frozen_lane": plan["frozen_lane"],
            "config": config,
            "model": model,
            "audit": audit,
        }
        report.update(
            measurements=self.write({"clean_records": records, "diagnostic_records": diagnostics}),
            export_receipt=self.write(exported),
            trained_export=exported,
            model_ancestry=evidence["model_ancestry"],
            native_stages=evidence["stages"],
            producer_sources=evidence["producer_sources"],
        )
        evaluation = self.write(report)
        return {
            **report,
            "schema": "nine_model_timed_evaluation_receipt_v1",
            "status": "FAILED",
            "completed": True,
            "evaluation": evaluation,
            "evaluation_plan": self.write({**plan, "research_continuation_policy": policy}),
            "candidate": plan["candidate"],
            "protocol": plan["protocol"],
            "checkpoint": checkpoint,
            "training_receipt": training,
            "request": request,
            "bundle_sha256": plan["frozen_lane"]["sha256"],
            "config_sha256": config["sha256"],
        }

    def test_stage_relabel_duplicate_process_wrong_argv_input_and_old_checkpoint_replay_stop(self):
        plan, binding, records, diagnostics, policy, baseline = self.fixture()
        baseline_value = api._research_json(baseline)
        evidence = api._research_json(baseline_value["collection_evidence"])
        progress = baseline_value["progress"]
        for change in ("process", "resource", "dispatch", "model"):
            altered = copy.deepcopy(evidence)
            if change == "process":
                altered["stages"][1]["process"] = altered["stages"][0]["process"]
            elif change == "resource":
                altered["stages"][1]["resource_return"] = altered["stages"][0]["resource_return"]
            elif change == "dispatch":
                pair = [st for st in altered["stages"] if "dispatch" in st]
                pair[1]["dispatch"] = pair[0]["dispatch"]
            else:
                altered["stages"][0]["model"] = plan["control"]["model"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                api.validate_collection_evidence(binding, self.write(altered), progress)
        altered = copy.deepcopy(records)
        row = altered[0]
        wrong = self.write({"messages": [{"role": "user", "content": "different prompt"}]})
        row["raw_request"] = wrong
        with self.assertRaises(ValueError):
            api.research_continuation_decision(policy, binding, altered, diagnostics)
        result = self.receipt_fixture(plan, binding, records, diagnostics, policy)
        self.assertTrue(api.validate_evaluation_continuation(result))
        with self.assertRaisesRegex(ValueError, "export/checkpoint"):
            api.validate_evaluation_continuation(
                {**result, "checkpoint": self.write({"new": "unmeasured checkpoint"})}
            )

    def test_receipt_consumer_actively_rechecks_failed_policy_and_rejects_scientific_rename(self):
        plan, binding, records, diagnostics, policy, _ = self.fixture()
        result = self.receipt_fixture(plan, binding, records, diagnostics, policy)
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
