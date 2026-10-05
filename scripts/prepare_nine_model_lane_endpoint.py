#!/usr/bin/env python3
"""Inspect/freeze additive EAGLE endpoint automation; no model/GPU or training action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_EAGLE_Q4_SHA256 = "2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280"
ORIGINAL_DEVELOPMENT_PROMPTS_SHA256 = (
    "131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081"
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_nine_model_lane import validate_lane  # noqa: E402

from w1a1_eagle.nine_model_pipeline import Files, atomic_json, require, sha256  # noqa: E402

REQUIRED = (
    "lane",
    "training_run_dir",
    "training_supervisor_state",
    "controller_identity",
    "supervisor_identity",
    "control",
    "protocol",
    "prompts",
    "development_admission",
    "policy",
    "qa_ledger",
)


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def source_inventory():
    # Endpoint runs in a separate frozen checkout. Original training source
    # remains independently pinned by the original immutable lane.
    paths = [
        path
        for folder in (ROOT / "scripts", ROOT / "src/w1a1_eagle")
        for path in folder.rglob("*.py")
    ]
    return {str(path.relative_to(ROOT)): pin(path) for path in sorted(paths)}


def validate_identity(record):
    require(
        isinstance(record, dict)
        and set(record) == {"pid", "start_ticks", "boot_id"}
        and type(record["pid"]) is int
        and record["pid"] > 0
        and type(record["start_ticks"]) is int
        and record["start_ticks"] >= 0
        and isinstance(record["boot_id"], str)
        and bool(record["boot_id"]),
        "exact controller/supervisor kernel identity required",
    )


def validate_protocol(protocol, policy):
    require(
        protocol.get("schema") == "nine_model_native_protocol_v1"
        and protocol.get("split") == "development",
        "only explicitly admitted unsealed development endpoint supported",
    )
    require(
        type(protocol.get("repetitions")) is int
        and protocol["repetitions"] >= 5
        and type(protocol.get("warmups_per_cell")) is int
        and protocol["warmups_per_cell"] >= 2,
        "five paired repetitions/two warmups required",
    )
    for key in (
        "context_tokens",
        "batch_tokens",
        "microbatch_tokens",
        "max_output_tokens",
        "startup_wall_seconds",
        "port",
    ):
        require(
            type(protocol.get(key)) is int and protocol[key] > 0,
            "positive protocol limit required: " + key,
        )
    require(
        type(protocol.get("seed")) is int
        and type(protocol.get("draft_lengths", {}).get("eagle")) is int
        and protocol["draft_lengths"]["eagle"] > 0,
        "explicit deterministic seed/EAGLE draft length required",
    )
    require(
        protocol.get("diagnostic_pass_separate_from_timing") is True,
        "diagnostics must remain outside clean timing",
    )
    require(
        protocol.get("evaluation_wall_seconds") == policy["evaluation_wall_seconds"],
        "selected evaluation wall cap differs",
    )
    require(
        protocol.get("selection_status") == "SELECTED_operational_under_overnight_delegation",
        "proposed protocol is not evaluation authorization",
    )


def prepare(descriptor, output, *, inspect_draft=False):
    require(
        descriptor.get("schema") == "nine_model_lane_endpoint_inputs_v1",
        "endpoint descriptor schema differs",
    )
    missing = [name for name in REQUIRED if not descriptor.get(name)]
    if inspect_draft:
        return {
            "schema": "nine_model_lane_endpoint_inspection_v1",
            "status": "PENDING" if missing else "RESOLVED_INPUTS_NOT_VALIDATED",
            "missing": missing,
            "production_ready": False,
            "gpu_queried": False,
            "training_changed": False,
        }
    require(not missing, "endpoint inputs PENDING: " + "; ".join(missing))
    files = Files()
    lane_path = files.check(descriptor["lane"])
    lane, _, admission = validate_lane(lane_path, descriptor["lane"]["sha256"])
    require(
        lane["candidate"] in {"eagle_a8", "eagle_a1"},
        "additive endpoint currently supports EAGLE only",
    )
    for key in ("controller_identity", "supervisor_identity"):
        validate_identity(descriptor[key])
    require(
        descriptor["controller_identity"]["boot_id"]
        == descriptor["supervisor_identity"]["boot_id"],
        "training process boot identities differ",
    )
    run = Path(descriptor["training_run_dir"])
    supervisor = Path(descriptor["training_supervisor_state"])
    require(
        run.is_absolute()
        and run == run.resolve()
        and supervisor.is_absolute()
        and supervisor == supervisor.resolve(),
        "canonical training state paths required",
    )
    control = descriptor["control"]
    require(
        control.get("frozen_original") is True
        and control.get("family") == "eagle"
        and control.get("precision") == "Q4_0",
        "genuine immutable original EAGLE Q4_0 control required",
    )
    files.check(control["model"])
    require(
        control["model"]["sha256"] == ORIGINAL_EAGLE_Q4_SHA256,
        "original immutable Q4 EAGLE hash differs",
    )
    provenance = json.loads(files.check(control["provenance"]).read_text())
    require(
        provenance.get("schema") == "nine_model_original_q4_reference_v1"
        and provenance.get("frozen_original") is True
        and provenance.get("family") == "eagle"
        and provenance.get("precision") == "Q4_0"
        and provenance.get("model") == control["model"],
        "original Q4 ancestry provenance differs",
    )
    for record in provenance.get("evidence", []):
        files.check(record)
    require(provenance.get("evidence"), "original Q4 producer/historical evidence absent")
    candidate = admission["candidates"][lane["candidate"]]
    initial_audit = json.loads(files.check(candidate["export"]).read_text())
    base = initial_audit["base_gguf"]
    require(
        base["sha256"] == candidate["source_bindings"]["common_source_sha256"]["base_draft_gguf"],
        "original base differs from frozen training corpus",
    )
    files.check(base)
    runtime = {
        key[len("artifact:") :]: record
        for key, record in admission["source"].items()
        if key.startswith("artifact:")
        and (key == "artifact:binary" or key.startswith("artifact:runtime_library_"))
    }
    require("binary" in runtime, "original native server runtime absent")
    for record in runtime.values():
        files.check(record)
    policy = json.loads(files.check(descriptor["policy"]).read_text())
    require(
        policy.get("schema") == "nine_model_lane_endpoint_policy_v1"
        and policy.get("authorization", {}).get("kind") == "human_delegated_operational_settings",
        "explicit delegated endpoint policy required",
    )
    files.check(policy["authorization"]["record"])
    for key in (
        "wait_wall_seconds",
        "export_wall_seconds",
        "evaluation_wall_seconds",
        "poll_seconds",
    ):
        value = policy.get(key)
        require(
            type(value) in (int, float) and math.isfinite(value) and value > 0,
            "explicit finite endpoint bound required: " + key,
        )
    require(policy["poll_seconds"] <= 1800, "endpoint polling interval exceeds monitor cadence")
    protocol = json.loads(files.check(descriptor["protocol"]).read_text())
    validate_protocol(protocol, policy)
    files.check(descriptor["prompts"])
    require(
        descriptor["prompts"]["sha256"] == ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
        "only prior authenticated development prompt bytes supported",
    )
    development = json.loads(files.check(descriptor["development_admission"]).read_text())
    require(
        development.get("schema") == "nine_model_lane_development_admission_v1"
        and development.get("split") == "development"
        and development.get("training_disjoint") is True
        and development.get("prompts") == descriptor["prompts"]
        and development.get("frozen_training_source") == candidate["source_bindings"]
        and development.get("evidence"),
        "authenticated development/corpus-disjoint prompt evidence required",
    )
    for record in development["evidence"]:
        files.check(record)
    source = source_inventory()
    for name in (
        "scripts/benchmark_native_eagle.py",
        "scripts/benchmark_dspark_screen.py",
    ):
        require(
            source[name]["sha256"] == lane["source"][name]["sha256"],
            "endpoint imported serializer/native producer differs from frozen training source: "
            + name,
        )
    qa = json.loads(files.check(descriptor["qa_ledger"]).read_text())
    require(
        qa.get("schema") == "nine_model_lane_endpoint_qa_v1"
        and qa.get("status") == "PASS"
        and qa.get("lane") == descriptor["lane"]
        and qa.get("source_files") == {name: record["sha256"] for name, record in source.items()}
        and qa.get("protocol") == descriptor["protocol"]
        and qa.get("prompts") == descriptor["prompts"]
        and qa.get("control_model") == control["model"],
        "independent current endpoint source/protocol/control QA required",
    )
    plan = {
        "schema": "nine_model_lane_endpoint_plan_v1",
        "artifact_kind": "production",
        "campaign_complete": False,
        "candidate": lane["candidate"],
        "frozen_lane": descriptor["lane"],
        "source": source,
        "training_run_dir": str(run),
        "training_supervisor_state": str(supervisor),
        "controller_identity": descriptor["controller_identity"],
        "supervisor_identity": descriptor["supervisor_identity"],
        "base_model": base,
        "serializer": lane["source"]["scripts/export_recurrent_binary.py"],
        "serializer_python": admission["python"],
        "serializer_python_invocation": admission["python_invocation"],
        "target": admission["target"],
        "runtime": runtime,
        "control": control,
        "protocol": descriptor["protocol"],
        "prompts": descriptor["prompts"],
        "development_admission": descriptor["development_admission"],
        "policy": descriptor["policy"],
        "qa_ledger": descriptor["qa_ledger"],
        "gpu_uuid": lane["gpu_uuid"],
        "gpu_control_path": lane["gpu_control_path"],
        "resource_policy": lane["resource_policy"],
        "environment": lane.get("environment", {}),
        "remaining_candidates": lane["remaining_candidates"],
        "target_policy": lane["target_policy"],
        "training_source": candidate["source_bindings"],
        "training_source_sha256": hashlib.sha256(
            json.dumps(candidate["source_bindings"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "training_limits": json.loads(Path(lane["budget"]["path"]).read_text())["candidates"][
            lane["candidate"]
        ]["training_limits"],
        "execution_status": "PENDING_committed_endpoint_fresh_lease_and_release",
    }
    require(not output.exists(), "preserve endpoint plan")
    atomic_json(output, plan)
    return {
        "plan": pin(output),
        "production_ready": False,
        "gpu_queried": False,
        "training_changed": False,
        "execution_status": plan["execution_status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-draft", action="store_true")
    args = parser.parse_args()
    require(args.inspect_draft or args.output is not None, "frozen output required")
    print(
        json.dumps(
            prepare(
                json.loads(args.inputs.read_text()), args.output, inspect_draft=args.inspect_draft
            )
        )
    )


if __name__ == "__main__":
    main()
