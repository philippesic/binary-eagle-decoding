#!/usr/bin/env python3
"""One released timed checkpoint: CPU serializer, four matched native cells, receipt.

Invoked by the admitted lane controller under its GPU lock, never a campaign
sequencer. Final-only historical endpoint validation remains the default.
"""

from __future__ import annotations

import json
import math
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

from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
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
    return plan, protocol


def cell_order(candidate, repetition):
    cells = (candidate, "initial", "eagle_q4", "target_only")
    shifted = cells[repetition % 4 :] + cells[: repetition % 4]
    return shifted if repetition % 2 == 0 else tuple(reversed(shifted))


def matched_report(plan, records, diagnostics, hardware, *, fixture=False):
    candidate = plan["candidate"]
    require(
        {row["cell"] for row in records} == {candidate, "initial", "eagle_q4", "target_only"},
        "four matched timed evaluation cells required",
    )
    report = aggregate(
        plan, [r for r in records if r["cell"] != "initial"], diagnostics, hardware, fixture=fixture
    )
    initial_plan = dict(plan, candidate="initial")
    initial = aggregate(
        initial_plan,
        [r for r in records if r["cell"] != candidate],
        diagnostics,
        hardware,
        fixture=fixture,
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
    return report


def evaluate(
    plan, files, protocol, exported, directory, stop_path, authorization, observer, runner
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    prompts = load_opaque_prompts(files.check(plan["prompts"]))
    require(
        prompts
        and all(p.get("split") == "development" for p in prompts)
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
    records, diagnostics = [], []
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
                        }
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
    report = matched_report(plan, records, diagnostics, baseline)
    report.update(
        model_ancestry=models,
        target=plan["target"],
        trained_export=exported,
        development_admission=plan["development_admission"],
    )
    atomic_json(
        directory / "measurements.json",
        {"clean_records": records, "diagnostic_records": diagnostics},
    )
    atomic_json(directory / "report.json", report)
    return pin(directory / "report.json")


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
    receipt = {
        "schema": "nine_model_timed_evaluation_receipt_v1",
        "status": "PASS",
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
    # Recheck source/request pins after the full comparison before publishing.
    files.check(request_locator)
    validate_plan(plan_locator, context["lane"], context["admission_plan"], files)
    atomic_json(directory / "receipt.json", receipt)
    return pin(directory / "receipt.json")
