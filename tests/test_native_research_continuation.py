"""Genuine-shaped CPU provenance fixtures; no native quality/admission claims."""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import evaluate_nine_model_timed_checkpoint as evaluator  # noqa: E402
import export_nine_model_lane_candidate as exporter  # noqa: E402
import test_nine_model_lane_candidate_export as export_fixtures  # noqa: E402

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
        self.serialized = export_fixtures.LaneExportTests().fixture(
            self.root / "serialized", "dspark", 8
        )
        target, model, q4, binary = [
            self.write({"CPU_fixture": name}) for name in ("target", "initial", "q4", "runtime")
        ]
        target = self.serialized["plan"]["target"]
        names = {"fc"} | {
            f"blk.{layer}.ffn_{part}" for layer in range(5) for part in ("gate", "up", "down")
        }
        self.projection_names = sorted(names)
        initial_audit = self.write(
            {
                "serialization_audit_passed": True,
                "output": model,
                "family": "dspark",
                "profile": "ffn15_fusion",
                "activation_bits": 8,
                "base_gguf": {"sha256": self.serialized["saved"]["source"]["base_gguf_sha256"]},
                "projections": {name: {"shape": [2, 32]} for name in names},
            }
        )
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
            "initial": {"model": model, "audit": initial_audit},
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
                "scripts/evaluate_nine_model_native.py",
            )
        }
        stages = []
        audits = {candidate: initial_audit, "initial": initial_audit}
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
                    from evaluate_nine_model_native import native_command

                    argv = native_command(
                        {"inputs": {"binary": binary, "target": target}},
                        api._research_json(plan["protocol"]),
                        candidate if cell == "initial" else cell,
                        None if cell == "target_only" else models[cell],
                        18290,
                    )
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
                            "packed": name + ".w1a1_packed",
                            "logical_k": 32,
                            "rows": 2,
                            "packed_type": "i32",
                            "packed_words": 1,
                            "tokens": 7,
                        }
                        for name in self.projection_names
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
                "hardware": {**hardware, "boot_id": "CPU-fixture-boot"},
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
                "scripts/export_nine_model_lane_candidate.py",
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
        f = self.serialized
        write = f["write"]
        frozen, state, supervisor, old_training = f["locators"]
        lane = f["lane"]
        spec = api._research_json(lane["config"])
        spec.update(evaluation_protocol=plan["protocol"], evaluation_milestones_seconds=[10])
        lane["config"] = write(Path(lane["config"]["path"]), spec)
        frozen = write(Path(frozen["path"]), lane)
        checkpoint_path = Path(f["receipt"]["checkpoint"]["path"])
        saved = f["saved"]
        saved["source"]["bundle_sha256"] = frozen["sha256"]
        torch.save(saved, checkpoint_path)
        checkpoint = api.artifact_locator(checkpoint_path)
        side = api._research_json(
            api.artifact_locator(checkpoint_path.with_suffix(".receipt.json"))
        )
        side.update(**checkpoint, source_sha256=exporter.digest(saved["source"]))
        write(checkpoint_path.with_suffix(".receipt.json"), side)
        counters = f["receipt"]["counters"]
        request = self.write(
            {
                "schema": "nine_model_timed_evaluation_request_v1",
                "candidate": plan["candidate"],
                "bundle_sha256": frozen["sha256"],
                "config_sha256": lane["config"]["sha256"],
                "checkpoint": checkpoint,
                "protocol": plan["protocol"],
                "counters": counters,
                "elapsed_seconds": counters["elapsed_seconds"],
                "milestone_seconds": 10,
                "milestone_index": 0,
                "budget_ledger": self.write(
                    {"schema": "continuous_training_budget_v1", "training_seconds": 10.0}
                ),
                "final_training_complete": True,
            }
        )
        training_value = {
            **f["receipt"],
            "checkpoint": checkpoint,
            "bundle_sha256": frozen["sha256"],
            "config_sha256": lane["config"]["sha256"],
            "timed_evaluation_request": request,
        }
        training = write(Path(old_training["path"]), training_value)
        state_value = api._research_json(state)
        state_value.update(
            status="awaiting_evaluation", bundle_sha256=frozen["sha256"], train_receipt=training
        )
        state = write(Path(state["path"]), state_value)
        supervisor = write(
            Path(supervisor["path"]), {"status": "running", "pid": 98765, "supervisor_pid": 98764}
        )
        patcher = patch.object(
            exporter, "validate_lane", return_value=(lane, exporter.Files(), f["plan"])
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        context = exporter.validate_endpoint(
            frozen, state, supervisor, training, checkpoint_mode="timed_evidence"
        )
        model = plan["initial"]["model"]
        initial_audit = api._research_json(plan["initial"]["audit"])
        projection_inputs = training_value["exports"][plan["candidate"]]
        audit = self.write(
            {
                **initial_audit,
                "schema": "block_binary_export_v1",
                "checkpoint": {"sha256": projection_inputs["checkpoint"]["sha256"]},
                "manifest": {"sha256": projection_inputs["manifest"]["sha256"]},
            }
        )
        exported = {
            "schema": "nine_model_lane_endpoint_export_v1",
            "artifact_kind": "production",
            "candidate": plan["candidate"],
            "checkpoint": checkpoint,
            "training_receipt": training,
            "frozen_lane": frozen,
            "config": lane["config"],
            "model": model,
            "audit": audit,
            "lane_state": state,
            "supervisor_state": supervisor,
            "campaign_complete": False,
        }
        export_receipt = self.write(exported)
        exporter.validate_export(context, export_receipt)
        future = self.root / "future-attempt" / "evaluation"
        future.mkdir(parents=True)
        execution = {
            "schema": "native_timed_evaluation_attempt_v1",
            "request": request,
            "training_receipt": training,
            "checkpoint": checkpoint,
            "model": model,
            "attempt_nonce": "future-attempt",
            "owner_identity": {"pid": 98765, "boot_id": "CPU-fixture-boot", "start_ticks": 987651},
            "evaluation_directory": str(future.resolve()),
        }
        stages = copy.deepcopy(evidence["stages"])
        for stage in stages:
            old_directory = Path(stage["process"]["path"]).parent
            new_directory = future / old_directory.parent.name / old_directory.name
            shutil.copytree(old_directory, new_directory)
            for key in ("process", "lineage", "resource_return", "server_log", "dispatch"):
                if key in stage:
                    stage[key] = api.artifact_locator(new_directory / Path(stage[key]["path"]).name)
            process = api._research_json(stage["process"])
            process["pid"] += 1000
            process["pgid"] += 1000
            process["kernel_identity"]["pid"] += 1000
            process["kernel_identity"]["start_ticks"] += 1000
            process["execution_context"] = execution
            stage["process"] = write(Path(stage["process"]["path"]), process)
            stage["lineage"] = write(
                Path(stage["lineage"]["path"]), {"kernel_identities": [process["kernel_identity"]]}
            )
            if stage["cell"] == plan["candidate"] and stage["diagnostic"]:
                stage["export_audit"] = audit
        future_records, future_diagnostics = copy.deepcopy(records), copy.deepcopy(diagnostics)
        for row in future_records + future_diagnostics:
            for key in ("raw_result", "raw_request"):
                old = Path(row[key]["path"])
                row[key] = api.artifact_locator(future / old.relative_to(self.root.resolve()))
        report.update(
            measurements=self.write(
                {"clean_records": future_records, "diagnostic_records": future_diagnostics}
            ),
            export_receipt=export_receipt,
            trained_export=exported,
            model_ancestry=evidence["model_ancestry"],
            native_stages=stages,
            producer_sources=evidence["producer_sources"],
            execution_context=execution,
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
            "bundle_sha256": frozen["sha256"],
            "config_sha256": lane["config"]["sha256"],
        }

    def test_historical_evidence_cannot_execute_export_or_callbacks(self):
        plan, binding, records, diagnostics, policy, _ = self.fixture()
        receipt = self.receipt_fixture(plan, binding, records, diagnostics, policy)
        report = api._research_json(receipt["evaluation"])
        exported = report["trained_export"]
        context = exporter.validate_endpoint(
            exported["frozen_lane"],
            exported["lane_state"],
            exported["supervisor_state"],
            exported["training_receipt"],
            checkpoint_mode="timed_evidence",
        )
        self.assertEqual(api._research_json(exported["supervisor_state"])["pid"], 98765)
        output = self.root.resolve() / "forbidden-evidence-export"
        calls = []

        def forbidden(name):
            def callback(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f"evidence mode reached {name}")

            return callback

        with self.assertRaisesRegex(ValueError, "read-only timed evidence"):
            exporter.export_endpoint(
                context,
                output,
                release_check=forbidden("release check"),
                cpu_admission=forbidden("CPU admission"),
                run=forbidden("serializer"),
            )
        self.assertEqual(calls, [])
        self.assertFalse(output.exists())

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

    def test_empty_projection_audit_counterfeit_serialized_weights_and_old_attempt_stop(self):
        plan, binding, records, diagnostics, policy, baseline = self.fixture()
        baseline_value = api._research_json(baseline)
        evidence = api._research_json(baseline_value["collection_evidence"])
        initial = api._research_json(plan["initial"]["audit"])
        empty = self.write({**initial, "projections": {}})
        altered = copy.deepcopy(evidence)
        for stage in altered["stages"]:
            if stage["diagnostic"] and "export_audit" in stage:
                stage["export_audit"] = empty
        with self.assertRaisesRegex(ValueError, "audit"):
            api.validate_collection_evidence(
                binding, self.write(altered), baseline_value["progress"]
            )
        result = self.receipt_fixture(plan, binding, records, diagnostics, policy)
        self.assertTrue(api.validate_evaluation_continuation(result))
        report = api._research_json(result["evaluation"])
        reused = {**report, "native_stages": evidence["stages"]}
        with self.assertRaisesRegex(ValueError, "request/evaluation attempt"):
            api.validate_evaluation_continuation({**result, "evaluation": self.write(reused)})
        # All JSON assertions are coherently rebound; only actual serialized
        # latent bytes vs old projection NPZ distinguish the counterfeit.
        saved = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=False)
        saved["linears"]["fc"]["latent_sign"][0, 0] += 0.25
        checkpoint_path = Path(result["checkpoint"]["path"])
        torch.save(saved, checkpoint_path)
        changed = api.artifact_locator(checkpoint_path)
        side_path = checkpoint_path.with_suffix(".receipt.json")
        side = api._research_json(api.artifact_locator(side_path))
        side.update(**changed)
        self.write(side, str(side_path))
        request = self.write({**api._research_json(result["request"]), "checkpoint": changed})
        training_old = api._research_json(result["training_receipt"])
        training = self.write(
            {**training_old, "checkpoint": changed, "timed_evaluation_request": request},
            str(Path(result["training_receipt"]["path"])),
        )
        exported = dict(report["trained_export"], checkpoint=changed, training_receipt=training)
        lane_state = api._research_json(exported["lane_state"])
        exported["lane_state"] = self.write(
            {**lane_state, "train_receipt": training}, str(Path(exported["lane_state"]["path"]))
        )
        modified_report = {
            **report,
            "trained_export": exported,
            "export_receipt": self.write(exported),
        }
        counterfeit = {
            **result,
            "checkpoint": changed,
            "request": request,
            "training_receipt": training,
            "evaluation": self.write(modified_report),
        }
        with self.assertRaisesRegex(ValueError, "serialized|latent|projection|NPZ"):
            api.validate_evaluation_continuation(counterfeit)

    def test_actual_zero_failure_status_schema_is_preserved_and_validated(self):
        plan, binding, _, _, _, baseline = self.fixture()
        baseline_value = api._research_json(baseline)
        evidence = api._research_json(baseline_value["collection_evidence"])
        status = {
            "schema": "dspark_zero_measurement_status_v1",
            "strict_quality_gate": "FAIL",
            "clean_records": 480,
            "diagnostic_records": 96,
            "optimizer_updates": 0,
            "current_and_initial_same_exact_model": True,
            "measurement_complete": True,
            "failure": {"type": "ValueError", "message": "greedy verifier token sequence differs"},
            "owner": {"pid": 88488, "start_ticks": 26169014, "boot_id": "CPU-fixture-boot"},
            "owned_release": {
                "owned_process_groups_absent": True,
                "owned_cuda_pids_absent": True,
                "other_context_pids": [],
            },
            "source_plan": plan,
        }
        original = self.write(status)
        derived = {
            **evidence,
            "failed_status": original,
            "remote_job_state": self.write({"status": "finished", "exit_code": 1}),
        }
        api.validate_collection_evidence(binding, self.write(derived), baseline_value["progress"])
        self.assertEqual(api._research_json(original), status)
        for key, value in (
            ("measurement_complete", False),
            ("optimizer_updates", 1),
            ("clean_records", 479),
        ):
            bad = {**derived, "failed_status": self.write({**status, key: value})}
            with self.assertRaises(ValueError):
                api.validate_collection_evidence(
                    binding, self.write(bad), baseline_value["progress"]
                )

    def test_changed_fifth_export_validator_rejects_before_import_validation(self):
        _, binding, _, _, policy, _ = self.fixture()
        original = api.artifact_locator

        def changed(path):
            value = original(path)
            if Path(path).name == "export_nine_model_lane_candidate.py":
                return {**value, "sha256": "0" * 64}
            return value

        with (
            patch.object(api, "artifact_locator", side_effect=changed),
            self.assertRaisesRegex(ValueError, "deployed source"),
        ):
            api.validate_research_policy(policy, binding)

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
