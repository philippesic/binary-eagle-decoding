#!/usr/bin/env python3
"""One released timed checkpoint: CPU serializer, four matched native cells, receipt.

Invoked by the admitted lane controller under its GPU lock, never a campaign
sequencer. Final-only historical endpoint validation remains the default.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_native_eagle import (  # noqa: E402
    available_port,
    execute_request,
    request_body,
    wait_ready,
)
from evaluate_nine_model_native import native_command  # noqa: E402
from export_nine_model_lane_candidate import export_endpoint, pin, validate_endpoint  # noqa: E402
from prepare_nine_model_lane_endpoint import (  # noqa: E402
    ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
    ORIGINAL_EAGLE_Q4_SHA256,
    validate_protocol,
)
from run_nine_model_lane_endpoint import (  # noqa: E402
    aggregate,
    round_summary,
    rows,
    stop_owned_server,
)

from w1a1_eagle.continuous_budget import (  # noqa: E402
    baseline_cases,
    baseline_control_outputs,
    research_bindings,
    research_continuation_decision,
    validate_collection_evidence,
    validate_research_policy,
)
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    Files,
    atomic_json,
    cleanup_descendants,
    deferred_termination,
    descendant_identities,
    load_opaque_prompts,
    native_environment,
    process_identity,
    require,
    resource_gate,
    validate_cuda_dispatch,
)


def validate_plan(locator, lane, admission, files):
    plan = json.loads(files.check(locator).read_text())
    selected = admission["candidates"][lane["candidate"]]
    require(
        plan.get("schema") == "nine_model_timed_evaluation_plan_v1",
        "timed evaluation plan required",
    )
    for key in (
        "candidate",
        "config",
        "gpu_uuid",
        "target_policy",
        "resource_policy",
        "environment",
        "gpu_control_path",
    ):
        require(plan.get(key) == lane[key], "timed plan/lane differs: " + key)
    require(
        plan["target"] == admission["target"]
        and plan["training_source"] == selected["source_bindings"],
        "timed target/TRAIN ancestry differs",
    )
    require(
        plan["initial"] == {"model": selected["model"], "audit": selected["export"]},
        "exact calibrated untrained drafter required",
    )
    expected_runtime = {
        key[len("artifact:") :]: record
        for key, record in admission["source"].items()
        if key == "artifact:binary" or key.startswith("artifact:runtime_library_")
    }
    require(plan["runtime"] == expected_runtime, "original admitted runtime differs")
    for record in plan["source"].values():
        files.check(record)
    # The helper plus both reused producers must be hash-bound, not merely mentioned.
    for name in (
        "scripts/evaluate_nine_model_timed_checkpoint.py",
        "scripts/export_nine_model_lane_candidate.py",
        "scripts/evaluate_nine_model_native.py",
        "scripts/run_nine_model_lane_endpoint.py",
    ):
        require(
            plan["source"].get(name) == pin(ROOT / name),
            "timed executable source pin differs: " + name,
        )
    for key in ("protocol", "prompts", "development_admission", "target"):
        files.check(plan[key])
    for group in (plan["runtime"], plan["initial"]):
        for record in group.values():
            files.check(record)
    control = plan["control"]
    require(
        control.get("frozen_original") is True
        and control.get("precision") == "Q4_0"
        and control["model"]["sha256"] == ORIGINAL_EAGLE_Q4_SHA256,
        "original EAGLE Q4 control required",
    )
    files.check(control["model"])
    files.check(control["provenance"])
    require(
        plan["prompts"]["sha256"] == ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
        "original held-out development prompts required",
    )
    spec = json.loads(files.check(lane["config"]).read_text())
    require(
        spec.get("evaluation_milestones_seconds") == [14400, 28800, 43200]
        and spec.get("evaluation_protocol") == plan["protocol"],
        "frozen 4/8/12h protocol differs",
    )
    limits = json.loads(files.check(lane["budget"]).read_text())["candidates"][lane["candidate"]][
        "training_limits"
    ]
    require(limits["max_seconds"] == 43200, "twelve cumulative trainer-hour cap required")
    protocol = json.loads(files.check(plan["protocol"]).read_text())
    validate_protocol(protocol, {"evaluation_wall_seconds": protocol["evaluation_wall_seconds"]})
    family = lane["candidate"].split("_")[0]
    require(
        type(protocol["draft_lengths"].get(family)) is int
        and protocol["draft_lengths"][family] > 0,
        "candidate native draft length required",
    )
    require(
        type(plan["export_wall_seconds"]) in (int, float)
        and math.isfinite(plan["export_wall_seconds"])
        and plan["export_wall_seconds"] > 0,
        "bounded CPU export required",
    )
    dev = json.loads(files.check(plan["development_admission"]).read_text())
    require(
        dev.get("schema") == "nine_model_lane_development_admission_v1"
        and dev.get("split") == "development"
        and dev.get("training_disjoint") is True
        and dev.get("prompts") == plan["prompts"]
        and dev.get("frozen_training_source") == selected["source_bindings"]
        and dev.get("evidence"),
        "authenticated TRAIN-disjoint development admission required",
    )
    for record in dev["evidence"]:
        files.check(record)
    if plan.get("research_continuation_policy"):
        validate_research_policy(plan["research_continuation_policy"], research_bindings(plan))
    return plan, protocol


def cell_order(candidate, repetition):
    cells = (candidate, "initial", "eagle_q4", "target_only")
    shifted = cells[repetition % 4 :] + cells[: repetition % 4]
    return shifted if repetition % 2 == 0 else tuple(reversed(shifted))


def matched_report(plan, records, diagnostics, hardware, *, fixture=False, measurement_only=False):
    candidate = plan["candidate"]
    require(
        {row["cell"] for row in records} == {candidate, "initial", "eagle_q4", "target_only"},
        "four matched timed evaluation cells required",
    )
    research = plan.get("research_continuation_policy")
    collecting = measurement_only or research is not None
    report = aggregate(
        plan,
        [r for r in records if r["cell"] != "initial"],
        diagnostics,
        hardware,
        fixture=fixture,
        measurement_only=collecting,
    )
    initial_plan = dict(plan, candidate="initial")
    initial = aggregate(
        initial_plan,
        [r for r in records if r["cell"] != candidate],
        diagnostics,
        hardware,
        fixture=fixture,
        measurement_only=collecting,
    )
    report["cells"]["initial"] = initial["cells"]["initial"]
    # Both aggregates independently check pairing against the same target rows.
    for cell in report["cells"].values():
        if "native_counts" in cell:
            counts = cell["native_counts"]
            require(counts["proposed"] > 0, "positive proposal denominator required")
            cell["acceptance_rate"] = counts["accepted"] / counts["proposed"]
    report["schema"] = "nine_model_timed_matched_report_v1"
    report["initial_model"] = plan["initial"]
    if collecting:
        report["strict_quality_status"] = (
            "FAILED"
            if "FAILED" in (report["strict_quality_status"], initial["strict_quality_status"])
            else "PASS"
        )
    if research and not measurement_only:
        report.update(
            research_continuation_decision(research, research_bindings(plan), records, diagnostics)
        )
    return report


def evaluate(
    plan, files, protocol, exported, directory, stop_path, authorization, observer, runner
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    prompts = load_opaque_prompts(files.check(plan["prompts"]))
    require(
        prompts
        and all(p.get("split") in (None, "development") for p in prompts)
        and len({p["id"] for p in prompts}) == len(prompts),
        "unique admitted development prompts required",
    )
    baseline = observer.snapshot()
    candidate = plan["candidate"]
    family = candidate.split("_")[0]
    bits = int(candidate.split("_a")[1])
    models = {
        candidate: exported["model"],
        "initial": plan["initial"]["model"],
        "eagle_q4": plan["control"]["model"],
    }
    audits = {
        candidate: json.loads(files.check(exported["audit"]).read_text()),
        "initial": json.loads(files.check(plan["initial"]["audit"]).read_text()),
    }
    for model in models.values():
        files.check(model)
    bundle = {"inputs": {"binary": plan["runtime"]["binary"], "target": plan["target"]}}
    records, diagnostics, native_stages = [], [], []
    started = time.monotonic()
    request_config = {
        "evaluation": {
            "max_output_tokens": protocol["max_output_tokens"],
            "temperature": 0.0,
            "seed": protocol["seed"],
            "enable_thinking": False,
        }
    }
    for rep, diagnostic in [(r, False) for r in range(protocol["repetitions"])] + [(0, True)]:
        for cell in cell_order(candidate, rep):
            authorization()
            require(not Path(stop_path).exists(), "timed evaluation STOP requested")
            stage = directory / ("diagnostic" if diagnostic else f"rep-{rep:02d}") / cell
            stage.mkdir(parents=True)
            quantized = cell in (candidate, "initial")
            native_cell = candidate if cell == "initial" else cell
            environment = native_environment(
                family if quantized else ("target_only" if cell == "target_only" else "eagle"),
                bits if quantized else None,
                plan["environment"],
            )
            environment["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
            if family in {"dspark", "dflash"} and quantized:
                environment["DSPARK_REQUIRE_AUTHOR_LAYOUT"] = "1"
            trace = stage / "rounds.jsonl"
            if diagnostic:
                environment["W1AX_ROUND_TRACE_JSONL"] = str(trace)
                if quantized:
                    environment["GGML_W1AX_ADMISSION_TRACE"] = "1"
            port = protocol["port"]
            require(available_port("127.0.0.1", port), "native port occupied")
            command = native_command(bundle, protocol, native_cell, models.get(cell), port)
            verbosity = (
                protocol.get("diagnostic_log_verbosity", 4)
                if diagnostic
                else protocol.get("clean_log_verbosity", 3)
            )
            command += ["--log-verbosity", str(verbosity)]
            proc = None
            with (stage / "server.log").open("wb") as log:
                try:
                    mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                    prior = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
                    try:
                        proc = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, prior),
                        )
                        identity = process_identity(proc.pid)
                        runner.process_groups.append(proc.pid)
                        runner.process_identities.append(identity)
                        atomic_json(
                            stage / "process.json",
                            {
                                "pid": proc.pid,
                                "pgid": proc.pid,
                                "kernel_identity": identity,
                                "argv": command,
                                **(
                                    {"execution_context": plan["execution_context"]}
                                    if plan.get("execution_context")
                                    else {}
                                ),
                            },
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, prior)
                    remaining = protocol["evaluation_wall_seconds"] - (time.monotonic() - started)
                    require(remaining > 0, "timed evaluation wall cap exhausted")
                    wait_ready(
                        proc,
                        f"http://127.0.0.1:{port}",
                        min(protocol["startup_wall_seconds"], remaining),
                    )
                    cases = [
                        (True, i, prompts[i % len(prompts)])
                        for i in range(protocol["warmups_per_cell"])
                    ] + [(False, i, p) for i, p in enumerate(prompts)]
                    for warmup, ordinal, prompt in cases:
                        authorization()
                        require(not Path(stop_path).exists(), "timed evaluation STOP requested")
                        discovered = descendant_identities(proc.pid)
                        runner.process_identities.extend(
                            i for i in discovered if i not in runner.process_identities
                        )
                        atomic_json(
                            stage / "process-lineage.json",
                            {"kernel_identities": runner.process_identities},
                        )
                        remaining = protocol["evaluation_wall_seconds"] - (
                            time.monotonic() - started
                        )
                        require(remaining > 0, "timed evaluation wall cap exhausted")
                        before = len(rows(trace)) if diagnostic else 0
                        destination = stage / (
                            f"warmup-{ordinal:04d}" if warmup else f"prompt-{ordinal:04d}"
                        )
                        result = execute_request(
                            f"http://127.0.0.1:{port}",
                            request_body(request_config, prompt),
                            min(180, remaining),
                            destination,
                        )
                        require(
                            result.get("generated_token_ids") is not None,
                            "exact native token IDs absent",
                        )
                        if warmup:
                            continue
                        row = {
                            "cell": cell,
                            "repetition": rep,
                            "prompt_id": prompt["id"],
                            "output_tokens": result["completion_tokens"],
                            "latency_s": result["request_wall_s"],
                            "generated_token_ids": result["generated_token_ids"],
                            "speculative": result["speculative"],
                            "raw_result": pin(destination / "measurement.json"),
                            "raw_request": pin(destination / "request.json")
                            if (destination / "request.json").exists()
                            else None,
                        }
                        if row["raw_request"] is None:
                            del row["raw_request"]
                        if diagnostic and cell != "target_only":
                            row["round_summary"] = round_summary(
                                rows(trace)[before:],
                                protocol["draft_lengths"][family if quantized else "eagle"],
                            )
                            require(
                                row["round_summary"]["rounds"] > 0,
                                "native diagnostic round trace absent",
                            )
                        (diagnostics if diagnostic else records).append(row)
                        atomic_json(
                            directory / "progress.json",
                            {"clean_records": records, "diagnostic_records": diagnostics},
                        )
                finally:
                    if proc is not None:
                        with deferred_termination():
                            try:
                                discovered = descendant_identities(proc.pid)
                                runner.process_identities.extend(
                                    i for i in discovered if i not in runner.process_identities
                                )
                                atomic_json(
                                    stage / "process-lineage.json",
                                    {"kernel_identities": runner.process_identities},
                                )
                            finally:
                                try:
                                    stop_owned_server(proc)
                                finally:
                                    cleanup_descendants(runner.process_identities)
            release = observer.require_released(runner.process_groups, runner.process_identities)
            observed = observer.snapshot()
            resource_gate(observed, baseline, plan["resource_policy"])
            require(
                release.get("other_context_pids") == [] and observed.get("dxg_holders") == [],
                "foreign GPU contexts forbid timed continuation",
            )
            atomic_json(stage / "resource-return.json", {"release": release, "resources": observed})
            if quantized and diagnostic:
                text = (stage / "server.log").read_text(errors="replace")
                require("dense fallback" not in text.lower(), "native dense fallback prohibited")
                actual = validate_cuda_dispatch(text, audits[cell], activation_bits=bits)
                atomic_json(stage / "actual-dispatch.json", actual)
            stage_record = {
                "cell": cell,
                "repetition": rep,
                "diagnostic": diagnostic,
                "process": pin(stage / "process.json"),
                "lineage": pin(stage / "process-lineage.json"),
                "server_log": pin(stage / "server.log"),
                "resource_return": pin(stage / "resource-return.json"),
                "model": plan["target"] if cell == "target_only" else models[cell],
            }
            if quantized and diagnostic:
                stage_record.update(
                    dispatch=pin(stage / "actual-dispatch.json"),
                    export_audit=(
                        exported["audit"] if cell == candidate else plan["initial"]["audit"]
                    ),
                )
            native_stages.append(stage_record)
    report = matched_report(plan, records, diagnostics, baseline)
    report.update(
        model_ancestry=models,
        target=plan["target"],
        trained_export=exported,
        development_admission=plan["development_admission"],
        native_stages=native_stages,
        producer_sources={
            name: pin(ROOT / name)
            for name in (
                "scripts/evaluate_nine_model_timed_checkpoint.py",
                "scripts/benchmark_native_eagle.py",
                "src/w1a1_eagle/nine_model_pipeline.py",
                "scripts/evaluate_nine_model_native.py",
            )
        },
    )
    if plan.get("research_continuation_policy"):
        report["execution_context"] = plan["execution_context"]
        report["export_receipt"] = plan["research_export_receipt"]
    atomic_json(
        directory / "measurements.json",
        {"clean_records": records, "diagnostic_records": diagnostics},
    )
    report["measurements"] = pin(directory / "measurements.json")
    atomic_json(directory / "report.json", report)
    return pin(directory / "report.json")


def recover_boundary_receipt(lane, lane_locator, run_dir, attempt, admission_path, state, files):
    """Recover a published boundary without reconstructing a GPU trainer.

    The trainer request can precede its state rename and outer receipt. Preserve
    both original paths; derive a new recovery receipt only from authenticated
    settled accounting, checkpoint counters and existing serialized exports.
    Full tensor/source validation still precedes export in evaluate_checkpoint.
    """
    from w1a1_eagle.continuous_budget import TimedEvaluation

    training_dir = Path(run_dir) / "training"
    if not (training_dir / "timed-evaluation-state.json").exists() and not any(
        training_dir.glob("timed-evaluation/milestone-*/request.json")
    ):
        return None
    spec = json.loads(files.check(lane["config"]).read_text())
    budget = json.loads(files.check(lane["budget"]).read_text())
    maximum = budget["candidates"][lane["candidate"]]["training_limits"]["max_seconds"]
    boundary = TimedEvaluation(
        training_dir,
        spec["evaluation_milestones_seconds"],
        maximum,
        candidate=lane["candidate"],
        bundle_sha256=lane_locator["sha256"],
        config_sha256=lane["config"]["sha256"],
        protocol=spec["evaluation_protocol"],
        atomic_write=atomic_json,
    )
    request_locator = boundary.state["pending"]
    if request_locator is None:
        index = len(boundary.state["completed"])
        orphan = training_dir / "timed-evaluation" / f"milestone-{index:02d}" / "request.json"
        if not orphan.exists():
            return None
        request_locator = pin(orphan)
    request = boundary._request(request_locator)
    require(
        request["milestone_index"] == len(boundary.state["completed"]),
        "recovered pending milestone order differs",
    )
    if state.get("evaluation_receipt") is not None:
        result = json.loads(files.check(state["evaluation_receipt"]).read_text())
        if result.get("request") == request_locator:
            boundary._result(state["evaluation_receipt"], request_locator, request)
            # First consumption remains trainer-owned and requires exact checkpoint.
            return None
    current_ledger = json.loads((training_dir / "budget-used.json").read_text())
    frozen_ledger = json.loads(files.check(request["budget_ledger"]).read_text())
    require(
        all(
            current_ledger.get(key) == frozen_ledger.get(key)
            for key in (
                "schema",
                "source_sha256",
                "max_seconds",
                "active_attempt",
                "training_seconds",
            )
        ),
        "recovery requires unchanged settled training ledger",
    )
    original_plan = json.loads(files.check(lane["admission_plan"]).read_text())
    source = original_plan["candidates"][lane["candidate"]]["source_bindings"]
    source_digest = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_budget_source = (
        source_digest if lane["candidate"].startswith("eagle_") else lane_locator["sha256"]
    )
    require(
        frozen_ledger.get("source_sha256") == expected_budget_source,
        "recovery settled ledger source differs",
    )
    checkpoint = files.check(request["checkpoint"])
    require(
        training_dir / "checkpoints" in checkpoint.parents, "recovery checkpoint outside trainer"
    )
    family, precision = lane["candidate"].split("_")
    counters = request["counters"]
    latest_path = training_dir / ("latest.json" if family == "eagle" else "checkpoints/latest.json")
    latest = json.loads(latest_path.read_text())
    require(
        {key: latest[key] for key in ("path", "sha256")} == request["checkpoint"],
        "recovery latest checkpoint differs from pending boundary",
    )
    if family == "eagle":
        outer = json.loads((checkpoint.parent / "manifest.json").read_text())
        require(
            outer.get("schema") == "continuous_joint_w1ax_v1"
            and outer.get("optimizer_rng_cursor_exact") is True
            and outer.get("source_sha256") == source_digest
            and outer.get("sha256") == request["checkpoint"]["sha256"]
            and all(outer.get(key) == counters.get(key) for key in ("step", "epoch", "cursor")),
            "recovery EAGLE checkpoint counter join differs",
        )
        lane_name = "A" + precision[1:]
        exported = {
            "checkpoint": {
                "path": str(checkpoint.parent / lane_name / "joint.npz"),
                "sha256": outer["exports"][lane_name]["joint.npz"],
            },
            "manifest": {
                "path": str(checkpoint.parent / lane_name / "joint.json"),
                "sha256": outer["exports"][lane_name]["joint.json"],
            },
        }
    else:
        sidecar = checkpoint.with_suffix(".receipt.json")
        side = json.loads(sidecar.read_text()) if sidecar.exists() else latest
        require(
            side.get("schema") == "block_qat_checkpoint_v1"
            and side.get("committed") is True
            and side.get("source_sha256")
            == hashlib.sha256(
                json.dumps(
                    {**source, "bundle_sha256": lane_locator["sha256"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            and side.get("cursor") == counters
            and {key: side[key] for key in ("path", "sha256")} == request["checkpoint"],
            "recovery block checkpoint counter join differs",
        )
        pairs = [
            (directory / "binary.npz", directory / "binary.json")
            for directory in Path(request_locator["path"]).parent.glob("export-*")
            if directory.is_dir()
            and not directory.is_symlink()
            and (directory / "binary.npz").is_file()
            and (directory / "binary.json").is_file()
        ]
        require(len(pairs) == 1, "recovery requires one complete preserved block export pair")
        exported = {"checkpoint": pin(pairs[0][0]), "manifest": pin(pairs[0][1])}
    for record in exported.values():
        files.check(record)
    joint = json.loads(files.check(exported["manifest"]).read_text())
    require(
        joint.get("checkpoint_sha256") == exported["checkpoint"]["sha256"]
        and joint.get("activation_bits") == int(precision[1:]),
        "recovery serializer hash/precision differs",
    )
    expected_base = (
        source["common_source_sha256"]["base_draft_gguf"]
        if family == "eagle"
        else source["base_gguf_sha256"]
    )
    require(
        joint.get("base_gguf_sha256") == expected_base, "recovery serializer base ancestry differs"
    )
    exported["base_gguf_sha256"] = joint["base_gguf_sha256"]
    originals = []
    for path in (Path(run_dir) / "attempts").glob("*/training-attempts/*/train-receipt.json"):
        value = json.loads(path.read_text())
        if value.get("timed_evaluation_request") != request_locator:
            continue
        require(
            value.get("schema") == "nine_model_stage_receipt_v1"
            and value.get("status") == "PASS"
            and value.get("artifact_kind") == "production"
            and value.get("committed") is True
            and value.get("candidate", lane["candidate"]) == lane["candidate"]
            and value.get("stage") == lane["candidate"] + "/train"
            and value.get("bundle_sha256") == lane_locator["sha256"]
            and value.get("config_sha256") == lane["config"]["sha256"]
            and value.get("checkpoint") == request["checkpoint"]
            and value.get("counters") == counters
            and value.get("exports") == {lane["candidate"]: exported}
            and value.get("hardware", {}).get("gpu_uuid") == lane["gpu_uuid"]
            and value["hardware"].get("compute_capability") == [12, 0]
            and value.get("completion_reason")
            == (
                "approved_budget_complete"
                if request["final_training_complete"]
                else "timed_evaluation_boundary"
            ),
            "retained boundary receipt provenance differs",
        )
        originals.append(pin(path))
    require(len(originals) <= 1, "duplicate retained boundary receipts require explicit recovery")
    if originals:
        receipt = originals[0]
    else:
        admission_locator = pin(admission_path)
        admitted = json.loads(files.check(admission_locator).read_text())
        require(
            admitted.get("schema") == "nine_model_training_admission_v1"
            and admitted.get("status") == "PASS"
            and admitted.get("artifact_kind") == "production"
            and admitted.get("candidate") == lane["candidate"]
            and admitted.get("bundle_sha256") == lane_locator["sha256"]
            and admitted.get("config_sha256") == lane["config"]["sha256"]
            and admitted.get("gpu_uuid") == lane["gpu_uuid"]
            and admitted.get("compute_capability") == [12, 0],
            "recovered boundary admission/hardware provenance differs",
        )
        directory = Path(attempt) / "boundary-recovery" / __import__("uuid").uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        recovered = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": lane["candidate"] + "/train",
            "status": "PASS",
            "artifact_kind": "production",
            "committed": True,
            "bundle_sha256": lane_locator["sha256"],
            "config_sha256": lane["config"]["sha256"],
            "checkpoint": request["checkpoint"],
            "counters": counters,
            "exports": {lane["candidate"]: exported},
            "hardware": {
                "gpu_uuid": admitted["gpu_uuid"],
                "compute_capability": admitted["compute_capability"],
            },
            "completion_reason": "approved_budget_complete"
            if request["final_training_complete"]
            else "timed_evaluation_boundary",
            "timed_evaluation_request": request_locator,
            "training_time_policy": request["training_time_policy"],
            "recovery": {
                "original_outer_receipt": "ABSENT",
                "request": request_locator,
                "admission": admission_locator,
                "optimizer_updates_performed": 0,
            },
        }
        atomic_json(directory / "train-receipt.json", recovered)
        receipt = pin(directory / "train-receipt.json")
    if boundary.state["pending"] is None:
        boundary.state["pending"] = request_locator
        boundary.write(boundary.path, boundary.state)
    return receipt


def evaluate_checkpoint(
    lane_locator,
    state_locator,
    supervisor_locator,
    training_locator,
    plan_locator,
    directory,
    *,
    files,
    observer,
    runner,
    release,
    authorization,
    stop_path,
):
    context = validate_endpoint(
        lane_locator, state_locator, supervisor_locator, training_locator, checkpoint_mode="timed"
    )
    plan, protocol = validate_plan(plan_locator, context["lane"], context["admission_plan"], files)
    plan = dict(
        plan, frozen_lane=lane_locator, remaining_candidates=context["lane"]["remaining_candidates"]
    )
    request_locator = context["receipt"]["timed_evaluation_request"]
    request = json.loads(files.check(request_locator).read_text())
    require(request["protocol"] == plan["protocol"], "timed request protocol differs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    def fresh_release(_context):
        authorization()
        release()
        return True

    def cpu_admission(_context, _directory):
        observed = observer.snapshot()
        resource_gate(observed, observed, plan["resource_policy"])
        # Serializer inputs and output coexist. Conservative additional host/disk
        # allowance; expensive exporter remains bounded and CPU-only.
        size = sum(
            Path(context[key]["path"]).stat().st_size for key in ("checkpoint", "base_model")
        )
        import shutil

        require(
            observed["host_available_bytes"] >= size * 2, "CPU export peak RAM admission failed"
        )
        require(shutil.disk_usage(directory).free >= size * 2, "CPU export disk admission failed")
        return True

    def cpu_run(command, **kwargs):
        previous = runner.environment
        runner.environment = kwargs["env"]
        try:
            runner.run(
                command,
                directory=directory / "cpu-process",
                stop_path=stop_path,
                wall_seconds=kwargs["timeout"],
            )
        finally:
            runner.environment = previous

    exported = export_endpoint(
        context,
        directory / "export",
        release_check=fresh_release,
        cpu_admission=cpu_admission,
        wall_seconds=plan["export_wall_seconds"],
        run=cpu_run,
    )
    authorization()
    release()
    if plan.get("research_continuation_policy"):
        # Snapshot historical source state, without changing checkpoint or model bytes.
        exported_value = json.loads((directory / "export/receipt.json").read_text())
        exported_value["snapshot_ancestry"] = {}
        exported_value["source_run_directory"] = str(
            Path(exported_value["lane_state"]["path"]).parent.resolve()
        )
        for key in ("lane_state", "supervisor_state"):
            original_locator = exported_value[key]
            source_path = files.check(exported_value[key])
            destination = directory / "export" / ("immutable-" + key + ".json")
            destination.write_bytes(source_path.read_bytes())
            exported_value[key] = pin(destination)
            exported_value["snapshot_ancestry"][key] = {
                "original": original_locator,
                "snapshot": exported_value[key],
            }
        exported_value["original_export_receipt"] = pin(directory / "export/receipt.json")
        immutable_export = directory / "export/research-receipt.json"
        atomic_json(immutable_export, exported_value)
        plan["research_export_receipt"] = pin(immutable_export)
        exported = exported_value
        plan["execution_context"] = {
            "schema": "native_timed_evaluation_attempt_v1",
            "request": request_locator,
            "training_receipt": training_locator,
            "checkpoint": context["checkpoint"],
            "model": exported["model"],
            "attempt_nonce": directory.name,
            "owner_identity": process_identity(os.getpid()),
            "evaluation_directory": str((directory / "evaluation").resolve()),
        }
    report = evaluate(
        plan,
        files,
        protocol,
        exported,
        directory / "evaluation",
        stop_path,
        authorization,
        observer,
        runner,
    )
    authorization()
    proof = release()
    report_value = json.loads(files.check(report).read_text())
    receipt = {
        "schema": "nine_model_timed_evaluation_receipt_v1",
        "status": "FAILED" if report_value.get("strict_quality_status") == "FAILED" else "PASS",
        "completed": True,
        "candidate": context["lane"]["candidate"],
        "bundle_sha256": lane_locator["sha256"],
        "config_sha256": context["lane"]["config"]["sha256"],
        "request": request_locator,
        "checkpoint": context["checkpoint"],
        "protocol": plan["protocol"],
        "evaluation": report,
        "owned_release": proof,
        "evaluation_plan": plan_locator,
        "training_receipt": training_locator,
        "elapsed_seconds": request["elapsed_seconds"],
        "milestone_seconds": request["milestone_seconds"],
        "campaign_complete": False,
    }
    if plan.get("research_continuation_policy"):
        for key in (
            "measurement_status",
            "strict_quality_status",
            "continuation_status",
            "research_only",
            "research_continuation_policy",
            "research_bindings",
            "execution_context",
        ):
            receipt[key] = report_value[key]
    # Recheck source/request pins after the full comparison before publishing.
    files.check(request_locator)
    validate_plan(plan_locator, context["lane"], context["admission_plan"], files)
    atomic_json(directory / "receipt.json", receipt)
    return pin(directory / "receipt.json")


def collect_failed_zero_baseline(plan, progress_locator, evidence_locator, output):
    """CPU-only honest reconstruction from already completed raw 0h collection.

    This emits no continuation authority. A separately reviewed, hash-bound
    operational policy must explicitly authorize its exact original failure cases.
    """
    bindings = research_bindings(plan)
    from w1a1_eagle.continuous_budget import checked_locator

    progress = json.loads(checked_locator(progress_locator).read_text())
    evidence = validate_collection_evidence(bindings, evidence_locator, progress_locator)
    records, diagnostics = progress["clean_records"], progress["diagnostic_records"]
    cases = baseline_cases(bindings, records, diagnostics)
    report = matched_report(plan, records, diagnostics, evidence["hardware"], measurement_only=True)
    require(
        report["strict_quality_status"] == "FAILED", "genuine strict zero parity failure required"
    )
    report.update(
        schema="native_zero_research_measurement_v1",
        bindings=bindings,
        progress=progress_locator,
        collection_evidence=evidence_locator,
        allowed_cases=cases,
        control_outputs=baseline_control_outputs(bindings, records, diagnostics),
        continuation_status="NOT_AUTHORIZED",
        optimizer_updates=0,
    )
    require(not Path(output).exists(), "preserve honest zero collection evidence")
    atomic_json(Path(output), report)
    return pin(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CPU-only strict-failed zero baseline reconstruction"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--progress-sha256", required=True)
    parser.add_argument("--collection-evidence", type=Path, required=True)
    parser.add_argument("--collection-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = Files()
    plan = json.loads(files.check({"path": str(args.plan), "sha256": args.plan_sha256}).read_text())
    print(
        json.dumps(
            collect_failed_zero_baseline(
                plan,
                {"path": str(args.progress), "sha256": args.progress_sha256},
                {"path": str(args.collection_evidence), "sha256": args.collection_evidence_sha256},
                args.output,
            )
        )
    )
