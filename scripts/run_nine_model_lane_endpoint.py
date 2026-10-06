#!/usr/bin/env python3
"""Wait for a released positive EAGLE budget endpoint, then export/compare separately.

Never signals or modifies the original trainer. Source/metadata inspection is
the default; execution requires a separately selected policy, QA and fresh lease.
"""

from __future__ import annotations

import argparse
import fcntl
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
from benchmark_dspark_screen import round_summary, rows, stop_owned_server  # noqa: E402
from benchmark_native_eagle import (  # noqa: E402
    available_port,
    execute_request,
    request_body,
    wait_ready,
)
from prepare_nine_model_lane_endpoint import (  # noqa: E402
    ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
    ORIGINAL_EAGLE_Q4_SHA256,
    pin,
    validate_protocol,
)
from run_nine_model_lane import validate_lane  # noqa: E402

from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    DiagnosticMemorySampler,
    Files,
    LinuxResources,
    SubprocessRunner,
    atomic_json,
    identity_active,
    load_opaque_prompts,
    native_environment,
    process_identity,
    require,
    require_available,
    resource_gate,
    sha256,
    stop_signals,
)


def validate_plan(path, expected):
    require(sha256(path) == expected, "frozen endpoint plan changed")
    plan = json.loads(path.read_text())
    require(
        plan.get("schema") == "nine_model_lane_endpoint_plan_v1"
        and plan.get("artifact_kind") == "production"
        and plan.get("campaign_complete") is False
        and plan.get("candidate") in {"eagle_a8", "eagle_a1"},
        "production EAGLE endpoint plan required",
    )
    files = Files()
    for record in plan["source"].values():
        files.check(record)
    lane, _, admission = validate_lane(
        files.check(plan["frozen_lane"]), plan["frozen_lane"]["sha256"]
    )
    require(
        lane["candidate"] == plan["candidate"]
        and lane["gpu_uuid"] == plan["gpu_uuid"]
        and lane["target_policy"] == plan["target_policy"]
        and lane["remaining_candidates"] == plan["remaining_candidates"]
        and admission["target"] == plan["target"],
        "endpoint differs from original frozen lane/target",
    )
    selected = admission["candidates"][plan["candidate"]]
    require(
        plan["serializer"] == lane["source"]["scripts/export_recurrent_binary.py"]
        and plan["serializer_python"] == admission["python"]
        and plan["serializer_python_invocation"] == admission["python_invocation"],
        "original pinned serializer/interpreter differs",
    )
    files.check(plan["serializer"])
    files.check(plan["serializer_python"])
    initial = json.loads(Path(selected["export"]["path"]).read_text())
    require(
        plan["base_model"] == initial["base_gguf"]
        and plan["training_source"] == selected["source_bindings"]
        and plan["training_source_sha256"]
        == hashlib.sha256(
            json.dumps(selected["source_bindings"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "endpoint base/training source changed",
    )
    require(
        plan["training_limits"]
        == json.loads(Path(lane["budget"]["path"]).read_text())["candidates"][plan["candidate"]][
            "training_limits"
        ],
        "endpoint changes original training allocation",
    )
    expected_runtime = {
        key[len("artifact:") :]: record
        for key, record in admission["source"].items()
        if key.startswith("artifact:")
        and (key == "artifact:binary" or key.startswith("artifact:runtime_library_"))
    }
    require(
        plan["runtime"] == expected_runtime
        and plan["resource_policy"] == lane["resource_policy"]
        and plan["environment"] == lane["environment"]
        and plan["gpu_control_path"] == lane["gpu_control_path"],
        "endpoint changes native/resource/ownership binding",
    )
    require(
        plan["control"].get("frozen_original") is True
        and plan["control"].get("precision") == "Q4_0"
        and plan["control"]["model"]["sha256"] == ORIGINAL_EAGLE_Q4_SHA256
        and plan["prompts"]["sha256"] == ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
        "original Q4/development scope differs",
    )
    for key in (
        "base_model",
        "target",
        "protocol",
        "prompts",
        "development_admission",
        "policy",
        "qa_ledger",
    ):
        files.check(plan[key])
    for record in plan["runtime"].values():
        files.check(record)
    files.check(plan["control"]["model"])
    files.check(plan["control"]["provenance"])
    policy = json.loads(Path(plan["policy"]["path"]).read_text())
    require(
        policy.get("schema") == "nine_model_lane_endpoint_policy_v1"
        and policy.get("authorization", {}).get("kind") == "human_delegated_operational_settings"
        and all(
            type(policy.get(key)) in (int, float) and math.isfinite(policy[key]) and policy[key] > 0
            for key in (
                "wait_wall_seconds",
                "export_wall_seconds",
                "evaluation_wall_seconds",
                "poll_seconds",
            )
        )
        and policy["poll_seconds"] <= 1800,
        "selected bounded endpoint policy differs",
    )
    files.check(policy["authorization"]["record"])
    protocol = json.loads(Path(plan["protocol"]["path"]).read_text())
    validate_protocol(protocol, policy)
    qa = json.loads(Path(plan["qa_ledger"]["path"]).read_text())
    require(
        qa.get("schema") == "nine_model_lane_endpoint_qa_v1"
        and qa.get("status") == "PASS"
        and qa.get("lane") == plan["frozen_lane"]
        and qa.get("source_files")
        == {name: record["sha256"] for name, record in plan["source"].items()}
        and qa.get("protocol") == plan["protocol"]
        and qa.get("prompts") == plan["prompts"]
        and qa.get("control_model") == plan["control"]["model"],
        "endpoint independent QA binding differs",
    )
    return plan, files, policy, protocol


def training_endpoint(plan, files, *, active=identity_active):
    """None while a confirmed live trainer/controller remains; never stop it."""
    supervisor = json.loads(Path(plan["training_supervisor_state"]).read_text())
    require(
        supervisor.get("pid") == plan["controller_identity"]["pid"]
        and supervisor.get("supervisor_pid") == plan["supervisor_identity"]["pid"],
        "original supervisor/controller identity differs",
    )
    if supervisor.get("status") in {"starting", "running", "stop_requested"}:
        require(
            active(plan["controller_identity"]) or active(plan["supervisor_identity"]),
            "training handle absent; recovery needed, no endpoint execution",
        )
        return None
    require(
        supervisor.get("status") == "finished"
        and supervisor.get("exit_code") == 0
        and supervisor.get("received_signal") is None,
        "original supervisor not a successful natural endpoint",
    )
    if active(plan["controller_identity"]) or active(plan["supervisor_identity"]):
        return None  # terminal receipt may precede process exit by milliseconds
    run = Path(plan["training_run_dir"])
    state = json.loads((run / "state.json").read_text())
    require(
        state.get("schema") == "nine_model_lane_state_v1"
        and state.get("status") == "training_complete"
        and state.get("candidate") == plan["candidate"]
        and state.get("bundle_sha256") == plan["frozen_lane"]["sha256"]
        and state.get("campaign_complete") is False
        and state.get("owned_release", {}).get("owned_process_groups_absent") is True,
        "successful original lane/release endpoint absent",
    )
    receipt = json.loads(files.check(state["train_receipt"]).read_text())
    frozen = json.loads(Path(plan["frozen_lane"]["path"]).read_text())
    counters = receipt.get("counters", {})
    require(
        receipt.get("schema") == "nine_model_stage_receipt_v1"
        and receipt.get("stage") == plan["candidate"] + "/train"
        and receipt.get("status") == "PASS"
        and receipt.get("artifact_kind") == "production"
        and receipt.get("committed") is True
        and receipt.get("completion_reason") == "approved_budget_complete"
        and receipt.get("bundle_sha256") == plan["frozen_lane"]["sha256"]
        and receipt.get("config_sha256") == frozen["config"]["sha256"]
        and type(counters.get("step")) is int
        and counters["step"] > 0,
        "positive committed exact-config training endpoint absent",
    )
    require(
        receipt.get("hardware", {}).get("gpu_uuid") == plan["gpu_uuid"]
        and receipt["hardware"].get("compute_capability") == [12, 0],
        "training hardware differs",
    )
    checkpoint = files.check(receipt["checkpoint"])
    require(
        run / "training/checkpoints" in checkpoint.parents, "checkpoint outside original trainer"
    )
    outer = json.loads((checkpoint.parent / "manifest.json").read_text())
    require(
        outer.get("schema") == "continuous_joint_w1ax_v1"
        and all(
            type(outer.get(key)) is int and outer[key] >= 0 and type(counters.get(key)) is int
            for key in ("epoch", "cursor")
        ),
        "exact continuous checkpoint schema/integer cursor required",
    )
    require(
        outer.get("source_sha256") == plan["training_source_sha256"],
        "checkpoint frozen training data/source differs",
    )
    require(
        type(counters.get("elapsed_seconds")) in (int, float)
        and math.isfinite(counters["elapsed_seconds"])
        and counters["elapsed_seconds"] >= plan["training_limits"]["max_seconds"],
        "genuine cumulative training allocation not reached",
    )
    require(
        outer.get("step") == counters["step"]
        and outer.get("epoch") == counters["epoch"]
        and outer.get("cursor") == counters["cursor"]
        and outer.get("sha256") == receipt["checkpoint"]["sha256"]
        and outer.get("optimizer_rng_cursor_exact") is True,
        "committed optimizer/RNG/cursor checkpoint differs",
    )
    lane_name = "A" + plan["candidate"].split("_a")[1]
    require(set(outer["exports"]) == {lane_name}, "single-lane checkpoint export inventory differs")
    exported = receipt["exports"][plan["candidate"]]
    require(
        exported["base_gguf_sha256"] == plan["base_model"]["sha256"], "trained export base differs"
    )
    for role, name in (("checkpoint", "joint.npz"), ("manifest", "joint.json")):
        require(
            exported[role]
            == {
                "path": str(checkpoint.parent / lane_name / name),
                "sha256": outer["exports"][lane_name][name],
            },
            "trained joint export does not join checkpoint",
        )
        files.check(exported[role])
    joint = json.loads(Path(exported["manifest"]["path"]).read_text())
    require(
        joint.get("checkpoint_sha256") == exported["checkpoint"]["sha256"]
        and joint.get("base_gguf_sha256") == plan["base_model"]["sha256"]
        and joint.get("activation_bits") == int(plan["candidate"].split("_a")[1])
        and len(joint.get("projections", {})) == 9,
        "trained all-nine projection manifest differs",
    )
    return {
        "state": state,
        "receipt": receipt,
        "train_receipt": state["train_receipt"],
        "outer_manifest": pin(checkpoint.parent / "manifest.json"),
    }


def owned_training_processes(plan):
    groups, identities = [], [plan["controller_identity"], plan["supervisor_identity"]]
    for path in Path(plan["training_run_dir"]).glob("**/process.json"):
        record = json.loads(path.read_text())
        require(record.get("kernel_identity"), "original process identity missing")
        identities.append(record["kernel_identity"])
        groups.append(record.get("pgid", record["pid"]))
    for path in Path(plan["training_run_dir"]).glob("**/process-lineage.json"):
        identities.extend(json.loads(path.read_text())["kernel_identities"])
    return groups, identities


def export_endpoint(plan, files, endpoint, directory, *, wall_seconds):
    require(not directory.exists(), "preserve endpoint export")
    directory.mkdir(parents=True)
    exported = endpoint["receipt"]["exports"][plan["candidate"]]
    files.check(plan["serializer"])
    files.check(plan["serializer_python"])
    command = [
        plan["serializer_python_invocation"],
        plan["serializer"]["path"],
        "--base",
        str(files.check(plan["base_model"])),
        "--checkpoint",
        str(files.check(exported["checkpoint"])),
        "--manifest",
        str(files.check(exported["manifest"])),
        "--output",
        str(directory / "trained.gguf"),
        "--audit",
        str(directory / "export-audit.json"),
    ]
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, timeout=wall_seconds)
    audit = json.loads((directory / "export-audit.json").read_text())
    require(
        audit.get("serialization_audit_passed") is True
        and audit.get("activation_bits") == int(plan["candidate"].split("_a")[1])
        and len(audit.get("projections", {})) == 9
        and audit.get("base_gguf") == plan["base_model"]
        and audit.get("checkpoint") == exported["checkpoint"]
        and audit.get("checkpoint_manifest") == exported["manifest"]
        and audit.get("output") == pin(directory / "trained.gguf"),
        "trained native serialization failed",
    )
    result = {
        "schema": "nine_model_lane_endpoint_export_v1",
        "artifact_kind": "production",
        "candidate": plan["candidate"],
        "frozen_lane": plan["frozen_lane"],
        "training_receipt": endpoint["train_receipt"],
        "checkpoint": endpoint["receipt"]["checkpoint"],
        "model": pin(directory / "trained.gguf"),
        "audit": pin(directory / "export-audit.json"),
        "campaign_complete": False,
    }
    atomic_json(directory / "receipt.json", result)
    return result


def cell_order(candidate, repetition):
    cells = (candidate, "eagle_q4", "target_only")
    shifted = cells[repetition % 3 :] + cells[: repetition % 3]
    return shifted if repetition % 2 == 0 else tuple(reversed(shifted))


def require_held_owner_lock(lock_path, owner):
    require(json.loads(lock_path.read_text()) == owner, "endpoint actual GPU lock owner differs")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
    raise ValueError("endpoint shared GPU lock is not actually held")


def issue_continuation(
    plan,
    plan_sha256,
    original_lease,
    endpoint,
    directory,
    stage,
    observer,
    groups,
    identities,
    baseline,
    *,
    now=None,
):
    """Fresh agent verification under standing authorization, never a new announcement."""
    issued = time.time() if now is None else now
    release = observer.require_released(groups, identities)
    require(
        release.get("owned_process_groups_absent") is True
        and release.get("owned_cuda_pids_absent") is True,
        "all prior owned release proof required",
    )
    observed = observer.snapshot()
    resource_gate(observed, baseline, plan["resource_policy"])
    require(
        observed["boot_id"] == plan["controller_identity"]["boot_id"]
        and observed["gpu_uuid"] == plan["gpu_uuid"],
        "endpoint continuation boot/device changed",
    )
    require(
        release.get("other_context_pids") == [] and observed.get("dxg_holders") == [],
        "foreign CUDA/DXG holders forbid endpoint continuation",
    )
    control = json.loads(Path(plan["gpu_control_path"]).read_text())
    require(control.get("rtx5080", {}).get("pause_requested") is False, "endpoint paused")
    lock_path = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    owner = process_identity(os.getpid())
    require_held_owner_lock(lock_path, owner)
    release_path = directory / ("continuation-release-" + stage + ".json")
    require(not release_path.exists(), "preserve continuation release proof")
    atomic_json(
        release_path,
        {
            "schema": "nine_model_lane_endpoint_release_v1",
            "training_receipt": endpoint["train_receipt"],
            "checkpoint": endpoint["receipt"]["checkpoint"],
            "terminal_state": pin(Path(plan["training_run_dir"]) / "state.json"),
            "terminal_supervisor": pin(Path(plan["training_supervisor_state"])),
            "actual_release": release,
            "resources": observed,
            "owner": owner,
            "shared_lock": str(lock_path),
            "unpaused_control": pin(Path(plan["gpu_control_path"])),
        },
    )
    result = {
        "schema": "nine_model_lane_endpoint_continuation_v1",
        "stage": stage,
        "endpoint_plan_sha256": plan_sha256,
        "training_lane_sha256": plan["frozen_lane"]["sha256"],
        "original_startup_lease": original_lease,
        "standing_authorization": json.loads(Path(plan["policy"]["path"]).read_text())[
            "authorization"
        ]["record"],
        "release_proof": pin(release_path),
        "issued_unix": issued,
        "expires_unix": issued + 300,
        "gpu_uuid": plan["gpu_uuid"],
        "boot_id": observed["boot_id"],
        "owner": owner,
        "shared_lock": str(lock_path),
        "new_human_announcement": False,
    }
    path = directory / ("endpoint-continuation-" + stage + ".json")
    require(not path.exists(), "preserve endpoint continuation")
    atomic_json(path, result)
    return pin(path)


def verify_continuation(
    plan, plan_sha256, original_lease, locator, stage, files, *, now=None, parent_pid=None
):
    current = time.time() if now is None else now
    record = json.loads(files.check(locator).read_text())
    require(
        record.get("schema") == "nine_model_lane_endpoint_continuation_v1"
        and record.get("stage") == stage
        and record.get("endpoint_plan_sha256") == plan_sha256
        and record.get("training_lane_sha256") == plan["frozen_lane"]["sha256"]
        and record.get("original_startup_lease") == original_lease
        and record.get("new_human_announcement") is False
        and record.get("gpu_uuid") == plan["gpu_uuid"]
        and record.get("boot_id") == plan["controller_identity"]["boot_id"],
        "typed endpoint continuation provenance differs",
    )
    require(
        type(record.get("issued_unix")) in (int, float)
        and type(record.get("expires_unix")) in (int, float)
        and record["issued_unix"]
        <= current
        < record["expires_unix"]
        <= record["issued_unix"] + 300,
        "endpoint continuation freshness expired",
    )
    files.check(original_lease)
    files.check(record["standing_authorization"])
    require(
        record["standing_authorization"]
        == json.loads(Path(plan["policy"]["path"]).read_text())["authorization"]["record"],
        "standing endpoint authorization changed",
    )
    owner = record["owner"]
    require(
        owner["pid"] == (os.getppid() if parent_pid is None else parent_pid)
        and identity_active(owner)
        and process_identity(owner["pid"]) == owner,
        "endpoint continuation parent owner is not live",
    )
    lock_path = Path(record["shared_lock"])
    require(
        lock_path == Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
        and json.loads(lock_path.read_text()) == owner,
        "endpoint actual GPU lock ownership differs",
    )
    require_held_owner_lock(lock_path, owner)
    proof = json.loads(files.check(record["release_proof"]).read_text())
    require(
        proof.get("owner") == owner
        and proof.get("actual_release", {}).get("owned_process_groups_absent") is True
        and proof.get("actual_release", {}).get("owned_cuda_pids_absent") is True
        and proof.get("actual_release", {}).get("other_context_pids") == []
        and proof.get("resources", {}).get("dxg_holders") == []
        and proof["resources"]["gpu_uuid"] == plan["gpu_uuid"]
        and proof["resources"]["boot_id"] == record["boot_id"],
        "endpoint release/foreign context proof differs",
    )
    for role in ("terminal_state", "terminal_supervisor", "unpaused_control"):
        files.check(proof[role])
    require(
        json.loads(Path(plan["gpu_control_path"]).read_text())
        .get("rtx5080", {})
        .get("pause_requested")
        is False,
        "endpoint continuation paused",
    )
    return record


def native_command(plan, protocol, cell, model, port, *, diagnostic=False):
    command = [
        plan["runtime"]["binary"]["path"],
        "--log-verbosity",
        "4" if diagnostic else "3",
        "-m",
        plan["target"]["path"],
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        str(protocol["context_tokens"]),
        "--batch-size",
        str(protocol["batch_tokens"]),
        "--ubatch-size",
        str(protocol["microbatch_tokens"]),
        "--parallel",
        "1",
        "--fit",
        "off",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--jinja",
        "--metrics",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if cell == "target_only":
        return command + ["--spec-type", "none"]
    return command + [
        "-md",
        model["path"],
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        str(protocol["draft_lengths"]["eagle"]),
        "--spec-draft-p-min",
        "0",
        "--spec-draft-ngl",
        "all",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
    ]


def aggregate(plan, records, diagnostics, hardware, *, fixture=False, measurement_only=False):
    require(
        fixture
        or (
            hardware.get("compute_capability") == [12, 0]
            and hardware.get("gpu_uuid") == plan["gpu_uuid"]
        ),
        "actual bound SM120 comparison hardware required",
    )
    cells = {plan["candidate"], "eagle_q4", "target_only"}
    require(
        {row["cell"] for row in records} == cells,
        "matched staged candidate/Q4/target cells required",
    )
    pairs, tables = {}, {}
    for cell in cells:
        selected = [row for row in records if row["cell"] == cell]
        keys = [(row["repetition"], row["prompt_id"]) for row in selected]
        require(len(keys) == len(set(keys)), "duplicate paired result")
        pairs[cell] = set(keys)
        require(
            len({row["repetition"] for row in selected}) >= 5, "five paired repetitions required"
        )
        require(
            all(
                type(row["output_tokens"]) is int
                and row["output_tokens"] > 0
                and type(row["latency_s"]) in (int, float)
                and math.isfinite(row["latency_s"])
                and row["latency_s"] > 0
                for row in selected
            ),
            "valid output counts/request latency required",
        )
        tokens, seconds = (
            sum(row["output_tokens"] for row in selected),
            sum(row["latency_s"] for row in selected),
        )
        tables[cell] = {
            "requests": len(selected),
            "output_tokens": tokens,
            "request_seconds": seconds,
            "request_tokens_per_second": tokens / seconds,
            "mean_request_latency_s": seconds / len(selected),
        }
        if cell != "target_only":
            require(
                all(
                    isinstance(row.get("speculative"), dict)
                    and all(
                        type(row["speculative"].get(key)) is int and row["speculative"][key] >= 0
                        for key in ("proposed", "accepted", "rounds")
                    )
                    and row["speculative"]["rounds"] > 0
                    and row["speculative"]["accepted"] <= row["speculative"]["proposed"]
                    for row in selected
                ),
                "native speculative acceptance counters required",
            )
            counts = {
                key: sum(row["speculative"][key] for row in selected)
                for key in ("proposed", "accepted", "rounds")
            }
            tables[cell].update(
                native_counts=counts, accepted_per_round=counts["accepted"] / counts["rounds"]
            )
    require(
        len({frozenset(pair) for pair in pairs.values()}) == 1, "prompt/repetition pairing differs"
    )
    target = {
        (row["repetition"], row["prompt_id"]): row["generated_token_ids"]
        for row in records
        if row["cell"] == "target_only"
    }
    mismatches = [
        row
        for row in records
        if row["generated_token_ids"] != target[(row["repetition"], row["prompt_id"])]
    ]
    require(measurement_only or not mismatches, "greedy verifier token sequence differs")
    reference = tables["eagle_q4"]
    candidate = tables[plan["candidate"]]
    candidate["speed_vs_original_eagle_q4"] = (
        candidate["request_tokens_per_second"] / reference["request_tokens_per_second"]
    )
    candidate["acceptance_vs_original_eagle_q4"] = (
        candidate["accepted_per_round"] / reference["accepted_per_round"]
        if reference["accepted_per_round"]
        else None
    )
    return {
        "schema": "nine_model_lane_matched_report_v1",
        "artifact_kind": "fixture" if fixture else "production",
        "native": True,
        "campaign_complete": False,
        "candidate": plan["candidate"],
        "primary_baseline": "original Q4_0 EAGLE",
        "target_policy": plan["target_policy"],
        "frozen_lane": plan["frozen_lane"],
        "protocol": plan["protocol"],
        "hardware": hardware,
        "cells": tables,
        "diagnostic_records": diagnostics,
        "instrumentation_in_clean_timing": False,
        "native_execution_proof_scope": "matching fresh diagnostic pass only",
        "remaining_candidates": plan["remaining_candidates"],
        **(
            {
                "measurement_status": "MEASUREMENT_COMPLETE",
                "strict_quality_status": "FAILED" if mismatches else "PASS",
                "continuation_status": "NOT_AUTHORIZED",
            }
            if measurement_only
            else {}
        ),
    }


def evaluate_endpoint(plan, files, protocol, exported, directory, stop_path, authorization):
    require(not directory.exists(), "preserve endpoint native results")
    directory.mkdir(parents=True)
    prompts = load_opaque_prompts(files.check(plan["prompts"]))
    require(
        prompts and all(prompt.get("split") == "development" for prompt in prompts),
        "only authorized development prompts allowed",
    )
    observer = LinuxResources(plan["gpu_uuid"])
    baseline = observer.snapshot()
    models = {plan["candidate"]: exported["model"], "eagle_q4": plan["control"]["model"]}
    for model in models.values():
        files.check(model)
    audit = json.loads(files.check(exported["audit"]).read_text())
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
    from w1a1_eagle.nine_model_pipeline import validate_cuda_dispatch

    for repetition, diagnostic in [(rep, False) for rep in range(protocol["repetitions"])] + [
        (0, True)
    ]:
        for cell in cell_order(plan["candidate"], repetition):
            authorization()
            require(not stop_path.exists(), "endpoint STOP requested")
            stage = directory / ("diagnostic" if diagnostic else f"rep-{repetition:02d}") / cell
            stage.mkdir(parents=True)
            bits = int(plan["candidate"].split("_a")[1]) if cell == plan["candidate"] else None
            environment = native_environment(
                "eagle" if cell != "target_only" else cell, bits, plan.get("environment")
            )
            environment["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
            trace = stage / "rounds.jsonl"
            if diagnostic:
                environment["W1AX_ROUND_TRACE_JSONL"] = str(trace)
                if bits is not None:
                    environment["GGML_W1AX_ADMISSION_TRACE"] = "1"
            port = protocol["port"]
            require(available_port("127.0.0.1", port), "endpoint native port occupied")
            command = native_command(
                plan, protocol, cell, models.get(cell), port, diagnostic=diagnostic
            )
            proc, identity, telemetry = None, None, None
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
                    if diagnostic:
                        telemetry = DiagnosticMemorySampler(
                            observer,
                            os.getpid(),
                            interval_seconds=protocol.get("memory_sample_seconds", 1),
                            max_samples=protocol.get("memory_max_samples_per_cell", 3600),
                        ).start()
                    wait_ready(
                        proc,
                        f"http://127.0.0.1:{port}",
                        min(
                            protocol["startup_wall_seconds"],
                            protocol["evaluation_wall_seconds"] - (time.monotonic() - started),
                        ),
                    )
                    for warmup, ordinal, prompt in [
                        (True, i, prompts[i % len(prompts)])
                        for i in range(protocol["warmups_per_cell"])
                    ] + [(False, i, prompt) for i, prompt in enumerate(prompts)]:
                        authorization()
                        require(not stop_path.exists(), "endpoint STOP requested")
                        remaining = protocol["evaluation_wall_seconds"] - (
                            time.monotonic() - started
                        )
                        require(remaining > 0, "endpoint evaluation wall cap exhausted")
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
                            "repetition": repetition,
                            "prompt_id": prompt["id"],
                            "output_tokens": result["completion_tokens"],
                            "latency_s": result["request_wall_s"],
                            "generated_token_ids": result["generated_token_ids"],
                            "speculative": result["speculative"],
                            "raw_result": pin(destination / "measurement.json"),
                        }
                        if diagnostic and cell != "target_only":
                            summary = round_summary(
                                rows(trace)[before:], protocol["draft_lengths"]["eagle"]
                            )
                            require(summary["rounds"] > 0, "native diagnostic round trace absent")
                            row["round_summary"] = summary
                        (diagnostics if diagnostic else records).append(row)
                        atomic_json(
                            directory / "progress.json",
                            {"clean_records": records, "diagnostic_records": diagnostics},
                        )
                finally:
                    try:
                        if telemetry is not None:
                            atomic_json(stage / "sampled-memory.json", telemetry.stop())
                    finally:
                        if proc is not None:
                            stop_owned_server(proc)
            observer.require_released([proc.pid], [identity])
            observed = observer.snapshot()
            resource_gate(observed, baseline, plan["resource_policy"])
            atomic_json(stage / "resource-return.json", observed)
            if cell == plan["candidate"] and diagnostic:
                text = (stage / "server.log").read_text(errors="replace")
                marker = (
                    "CUDA packed W1A8 INT8 dispatch"
                    if bits == 8
                    else "CUDA packed W1A1 XOR/POPCOUNT dispatch"
                )
                require(
                    "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)" in text
                    and marker in text
                    and "dense fallback" not in text.lower(),
                    "actual all-nine native loader/dispatch proof absent",
                )
                if diagnostic:
                    actual = validate_cuda_dispatch(text, audit, activation_bits=bits)
                    require(
                        actual["selected_projection_count"] == 9,
                        "all-nine actual CUDA execution required",
                    )
                    atomic_json(stage / "actual-dispatch.json", actual)
    report = aggregate(plan, records, diagnostics, baseline)
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
    return {
        "schema": "nine_model_lane_endpoint_evaluation_v1",
        "artifact_kind": "production",
        "campaign_complete": False,
        "report": pin(directory / "report.json"),
        "cells": [plan["candidate"], "eagle_q4", "target_only"],
        "owned_release": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--availability", type=Path)
    parser.add_argument("--supervisor-state", type=Path)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stage", choices=("export", "evaluate"))
    parser.add_argument("--lease-sha256")
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--continuation-sha256")
    args = parser.parse_args()
    plan, files, policy, protocol = validate_plan(args.plan, args.plan_sha256)
    if not args.start:
        print(
            json.dumps(
                {
                    "status": "PENDING_committed_endpoint_fresh_lease_and_release",
                    "gpu_queried": False,
                    "training_changed": False,
                    "campaign_complete": False,
                }
            )
        )
        return
    require(
        args.run_dir and args.availability and args.supervisor_state,
        "detached endpoint run/lease/supervisor required",
    )
    if args.stage:
        require(
            args.lease_sha256 and sha256(args.availability) == args.lease_sha256,
            "inherited endpoint startup lease changed",
        )
        parent = json.loads(args.supervisor_state.read_text())
        require(
            parent.get("status") == "running" and parent.get("pid") == os.getppid(),
            "endpoint stage must belong to live supervised controller",
        )
        lease = json.loads(args.availability.read_text())
        require(
            lease.get("schema") == "nine_model_gpu_lease_v1"
            and lease.get("bundle_sha256") == args.plan_sha256
            and lease.get("sole_owner") is True
            and lease.get("pause_requested") is False,
            "inherited endpoint lease provenance differs",
        )
    else:
        lease = require_available(args.availability, args.plan_sha256)
    require(lease["gpu_uuid"] == plan["gpu_uuid"], "endpoint lease physical GPU differs")

    def authorization():
        control = json.loads(Path(plan["gpu_control_path"]).read_text())
        require(
            control.get("rtx5080", {}).get("pause_requested") is False
            and json.loads(args.availability.read_text()) == lease,
            "endpoint paused or ownership changed",
        )
        require(not (args.run_dir / "STOP").exists(), "endpoint STOP requested")
        for record in plan["source"].values():
            files.check(record)

    authorization()
    args.run_dir = args.run_dir.resolve()
    if args.stage:
        endpoint = training_endpoint(plan, files)
        require(endpoint is not None, "trainer/controller still live; endpoint forbidden")
        require(
            args.continuation and args.continuation_sha256,
            "fresh typed post-terminal continuation required",
        )
        verify_continuation(
            plan,
            args.plan_sha256,
            {"path": str(args.availability.resolve()), "sha256": args.lease_sha256},
            {"path": str(args.continuation.resolve()), "sha256": args.continuation_sha256},
            args.stage,
            files,
        )
        observer = LinuxResources(plan["gpu_uuid"])
        groups, identities = owned_training_processes(plan)
        released = observer.require_released(groups, identities)
        current = observer.snapshot()
        require(
            released.get("other_context_pids") == [] and current.get("dxg_holders") == [],
            "foreign CUDA/DXG holders forbid stage dispatch",
        )
        resource_gate(current, endpoint["state"]["resource_baseline"], plan["resource_policy"])
        if args.stage == "export":
            export_endpoint(
                plan,
                files,
                endpoint,
                args.run_dir / "export",
                wall_seconds=policy["export_wall_seconds"],
            )
        else:
            exported = json.loads((args.run_dir / "export/receipt.json").read_text())
            require(
                exported.get("training_receipt") == endpoint["train_receipt"]
                and exported.get("frozen_lane") == plan["frozen_lane"],
                "endpoint export ancestry differs",
            )
            evaluate_endpoint(
                plan,
                files,
                protocol,
                exported,
                args.run_dir / "evaluation",
                args.run_dir / "STOP",
                authorization,
            )
        return
    supervisor = json.loads(args.supervisor_state.read_text())
    require(
        supervisor.get("pid") == os.getpid()
        and supervisor.get("supervisor_pid") == os.getppid()
        and supervisor.get("status") == "running"
        and supervisor.get("stop_grace_seconds", 0) >= 90
        and os.environ.get("TMUX"),
        "detached remote_job endpoint supervision required",
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.run_dir / "state.json"
    started = time.monotonic()
    observer = runner = baseline = None
    try:
        with stop_signals():
            while True:
                authorization()
                endpoint = training_endpoint(plan, files)
                if endpoint is not None:
                    break
                require(
                    time.monotonic() - started < policy["wait_wall_seconds"],
                    "bounded endpoint wait exhausted; healthy trainer untouched",
                )
                atomic_json(
                    state_path,
                    {
                        "status": "waiting_for_natural_endpoint",
                        "training_changed": False,
                        "gpu_queried": False,
                        "campaign_complete": False,
                    },
                )
                time.sleep(
                    min(
                        policy["poll_seconds"],
                        policy["wait_wall_seconds"] - (time.monotonic() - started),
                    )
                )
            lock = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("a+") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owner.seek(0)
                owner.truncate()
                json.dump(process_identity(os.getpid()), owner)
                owner.flush()
                os.fsync(owner.fileno())
                observer = LinuxResources(plan["gpu_uuid"])
                groups, identities = owned_training_processes(plan)
                observer.require_released(groups, identities)
                baseline = observer.snapshot()
                resource_gate(
                    baseline, endpoint["state"]["resource_baseline"], plan["resource_policy"]
                )
                runner = SubprocessRunner(
                    ROOT, plan.get("environment"), authorization=authorization
                )
                common = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--plan",
                    str(args.plan),
                    "--plan-sha256",
                    args.plan_sha256,
                    "--run-dir",
                    str(args.run_dir),
                    "--availability",
                    str(args.availability),
                    "--supervisor-state",
                    str(args.supervisor_state),
                    "--start",
                    "--lease-sha256",
                    sha256(args.availability),
                ]
                for stage, limit in (
                    ("export", policy["export_wall_seconds"]),
                    ("evaluate", policy["evaluation_wall_seconds"]),
                ):
                    authorization()
                    endpoint = training_endpoint(plan, files)
                    require(endpoint is not None, "matching natural endpoint changed")
                    continuation = issue_continuation(
                        plan,
                        args.plan_sha256,
                        pin(args.availability),
                        endpoint,
                        args.run_dir,
                        stage,
                        observer,
                        groups + runner.process_groups,
                        identities + runner.process_identities,
                        baseline,
                    )
                    atomic_json(
                        state_path,
                        {
                            "status": stage,
                            "training_receipt": endpoint["train_receipt"],
                            "campaign_complete": False,
                        },
                    )
                    runner.run(
                        [
                            *common,
                            "--stage",
                            stage,
                            "--continuation",
                            continuation["path"],
                            "--continuation-sha256",
                            continuation["sha256"],
                        ],
                        directory=args.run_dir / "attempts" / stage,
                        stop_path=args.run_dir / "STOP",
                        wall_seconds=limit,
                    )
                    observer.require_released(runner.process_groups, runner.process_identities)
                    resource_gate(observer.snapshot(), baseline, plan["resource_policy"])
                atomic_json(
                    state_path,
                    {
                        "status": "endpoint_complete",
                        "report": pin(args.run_dir / "evaluation/report.json"),
                        "campaign_complete": False,
                        "remaining_candidates": plan["remaining_candidates"],
                    },
                )
    except BaseException as error:
        release = {"status": "PENDING"}
        if observer is not None and runner is not None:
            try:
                observer.require_released(runner.process_groups, runner.process_identities)
                resource_gate(observer.snapshot(), baseline, plan["resource_policy"])
                release = {"status": "PASS"}
            except BaseException as cleanup:
                release = {"status": "FAILED", "reason": str(cleanup)}
        atomic_json(
            state_path,
            {
                "status": "failed",
                "failure": {"type": type(error).__name__, "reason": str(error)},
                "owned_release": release,
                "training_changed": False,
                "campaign_complete": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
