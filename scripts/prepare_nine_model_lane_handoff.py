#!/usr/bin/env python3
"""Freeze/inspect an EAGLE A8 to direct A1 metadata handoff; never execute it.

Only JSON, JSONL and pinned Python sources are read. Model/checkpoint/raw capture
bytes and live kernel state remain obligations of the existing runtime contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

# Import the existing stdlib-only contract without the package's eager model imports.
_contract_spec = importlib.util.spec_from_file_location(
    "handoff_pipeline_contract", ROOT / "src/w1a1_eagle/nine_model_pipeline.py"
)
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
Files, require, require_available, sha256 = (
    _contract.Files,
    _contract.require,
    _contract.require_available,
    _contract.sha256,
)

PACKET_KEYS = (
    "commands",
    "initial_request",
    "config",
    "resolved_inputs",
    "budget",
    "source_joins",
    "generation_replay_link",
    "replay_receipts",
)
OUTPUTS = (
    "initial_receipt",
    "export_audit",
    "independent_qa",
    "production_inputs",
    "admission_plan",
    "lane",
    "train_receipt",
    "endpoint_inputs",
    "endpoint_plan",
    "endpoint_result",
)
LIVE_GATES = [
    "exact supervisor/controller boot and birth identities absent in live kernel",
    "all owned process groups/CUDA contexts absent; no foreign CUDA or DXG holders",
    "shared ~/.config/binary-eagle-decoding/rtx5080-campaign.lock held with nonblocking flock",
    "adjacent fresh same-boot SM120 physical GPU and hardware/resource observation",
    "fresh nine_model_gpu_lease_v1 for actual stage bundle and same GPU (maximum 300 seconds)",
    "live host rtx5080.pause_requested=false and unchanged sole ownership",
    "detached remote_job supervision; stop/release/resource-return checked after each GPU stage",
]


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def metadata(files, locator):
    require(
        Path(locator["path"]).suffix in {".json", ".jsonl", ".py", ".md", ".log"},
        "metadata/source only; model/checkpoint/raw payload reads forbidden",
    )
    return files.check(locator)


def read(files, locator):
    return json.loads(metadata(files, locator).read_text())


def lines(files, locator):
    return [
        json.loads(line)
        for line in metadata(files, locator).read_text().splitlines()
        if line.strip()
    ]


def sources(files, root, inventory, revision):
    require(root.is_absolute() and root == root.resolve(), "canonical pinned checkout required")
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "exact original Git revision required",
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    require(head == revision, "source checkout HEAD differs from pinned original revision")
    tree = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-rz", revision, "--", "scripts", "src/w1a1_eagle"],
        check=True,
        capture_output=True,
    ).stdout
    blobs = {}
    for row in tree.split(b"\0"):
        if row:
            header, name = row.split(b"\t", 1)
            mode, kind, digest = header.split()
            if name.endswith(b".py"):
                require(
                    kind == b"blob" and mode in {b"100644", b"100755"},
                    "tracked Python source must be regular",
                )
                blobs[name.decode()] = digest.decode()
    require(set(blobs) == set(inventory), "Python inventory differs from pinned Git revision")
    actual = {
        str(p.relative_to(root))
        for folder in (root / "scripts", root / "src/w1a1_eagle")
        for p in folder.rglob("*.py")
    }
    require(
        actual and set(inventory) == actual, "complete original Python source inventory required"
    )
    for name, locator in inventory.items():
        require(locator["path"] == str(root / name), "source outside pinned original checkout")
        path = metadata(files, locator)
        data = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(digest == blobs[name], "dependency bytes differ from pinned Git revision")


def endpoint_snapshot(files, plan, snapshot):
    """Check optional saved metadata, without interpreting it as live release proof."""
    if snapshot is None:
        return "PENDING_natural_endpoint_metadata"
    required = {"supervisor", "lane_state", "train_receipt", "endpoint_state"}
    require(set(snapshot) == required, "complete saved endpoint metadata required")
    supervisor, state, receipt, result = [
        read(files, snapshot[k])
        for k in ("supervisor", "lane_state", "train_receipt", "endpoint_state")
    ]
    require(
        snapshot["supervisor"]["path"] == plan["training_supervisor_state"]
        and snapshot["lane_state"]["path"] == str(Path(plan["training_run_dir"]) / "state.json"),
        "saved states outside original parent",
    )
    require(
        supervisor.get("pid") == plan["controller_identity"]["pid"]
        and supervisor.get("supervisor_pid") == plan["supervisor_identity"]["pid"]
        and supervisor.get("status") == "finished"
        and supervisor.get("exit_code") == 0
        and supervisor.get("received_signal") is None,
        "natural exact supervisor terminal absent",
    )
    require(
        state.get("schema") == "nine_model_lane_state_v1"
        and state.get("status") == "training_complete"
        and state.get("candidate") == "eagle_a8"
        and state.get("bundle_sha256") == plan["frozen_lane"]["sha256"]
        and state.get("train_receipt") == snapshot["train_receipt"]
        and state.get("owned_release", {}).get("owned_process_groups_absent") is True,
        "original successful lane/release metadata absent",
    )
    receipt_path = Path(snapshot["train_receipt"]["path"])
    require(
        receipt_path.name == "train-receipt.json"
        and Path(plan["training_run_dir"]) / "attempts" in receipt_path.parents
        and state.get("attempt") == str(receipt_path.parent),
        "training receipt outside original parent run",
    )
    require(
        result["report"]["path"]
        == str(Path(snapshot["endpoint_state"]["path"]).parent / "evaluation/report.json"),
        "endpoint report outside original endpoint run",
    )
    lane = read(files, plan["frozen_lane"])
    counters = receipt.get("counters", {})
    require(
        receipt.get("schema") == "nine_model_stage_receipt_v1"
        and receipt.get("stage") == "eagle_a8/train"
        and receipt.get("status") == "PASS"
        and receipt.get("artifact_kind") == "production"
        and receipt.get("committed") is True
        and receipt.get("completion_reason") == "approved_budget_complete"
        and receipt.get("bundle_sha256") == plan["frozen_lane"]["sha256"]
        and receipt.get("config_sha256") == lane["config"]["sha256"]
        and type(counters.get("step")) is int
        and counters["step"] > 0
        and type(counters.get("elapsed_seconds")) in (int, float)
        and math.isfinite(counters["elapsed_seconds"])
        and counters["elapsed_seconds"] >= plan["training_limits"]["max_seconds"],
        "positive committed full natural budget endpoint absent",
    )
    checkpoint = receipt.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    require(
        set(checkpoint) == {"path", "sha256"}
        and checkpoint_path.is_absolute()
        and checkpoint_path == checkpoint_path.resolve()
        and Path(plan["training_run_dir"]) / "training/checkpoints" in checkpoint_path.parents
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("sha256", "")),
        "committed checkpoint locator outside original trainer",
    )
    outer = read(files, pin(checkpoint_path.parent / "manifest.json"))
    require(
        outer.get("schema") == "continuous_joint_w1ax_v1"
        and outer.get("optimizer_rng_cursor_exact") is True
        and outer.get("source_sha256") == plan["training_source_sha256"]
        and outer.get("sha256") == checkpoint["sha256"]
        and all(
            type(counters.get(key)) is int
            and counters[key] >= 0
            and type(outer.get(key)) is int
            and outer[key] == counters[key]
            for key in ("step", "epoch", "cursor")
        )
        and set(outer.get("exports", {})) == {"A8"},
        "committed exact optimizer/RNG/cursor/source checkpoint metadata differs",
    )
    # Deliberately do not open resume.pt or joint.npz. Existing training_endpoint
    # must check the actual payload/export bytes at the adjacent runtime gate.
    require(
        receipt.get("hardware", {}).get("gpu_uuid") == plan["gpu_uuid"]
        and receipt["hardware"].get("compute_capability") == [12, 0],
        "endpoint hardware differs",
    )
    require(
        result.get("status") == "endpoint_complete" and result.get("campaign_complete") is False,
        "successful upstream endpoint result absent",
    )
    report = read(files, result["report"])
    require(
        report.get("candidate") == "eagle_a8"
        and report.get("artifact_kind") == "production"
        and report.get("schema") == "nine_model_lane_matched_report_v1"
        and report.get("frozen_lane") == plan["frozen_lane"]
        and report.get("trained_export", {}).get("training_receipt") == snapshot["train_receipt"],
        "upstream endpoint report candidate differs",
    )
    return "SAVED_METADATA_CHECKED_fresh_live_verification_required"


def build(descriptor):
    require(
        descriptor.get("schema") == "nine_model_lane_handoff_inputs_v1",
        "handoff descriptor differs",
    )
    files = Files()
    packet = descriptor.get("packet", {})
    require(set(packet) == set(PACKET_KEYS), "complete immutable A1 packet metadata required")
    require(
        set(descriptor.get("outputs", {})) == set(OUTPUTS),
        "complete pending output inventory required",
    )
    for record in descriptor["outputs"].values():
        path = Path(record.get("path", ""))
        require(
            set(record) == {"path", "status"}
            and record["status"] == "PENDING"
            and path.is_absolute()
            and path == path.resolve()
            and not path.exists(),
            "unresolved output must stay PENDING and unpublished; "
            "refreeze actual runtime contracts separately",
        )
    upstream = read(files, descriptor["upstream_endpoint_plan"])
    require(
        upstream.get("schema") == "nine_model_lane_endpoint_plan_v1"
        and upstream.get("candidate") == "eagle_a8"
        and upstream.get("artifact_kind") == "production"
        and upstream.get("campaign_complete") is False,
        "original production A8 endpoint required",
    )
    for key in ("controller_identity", "supervisor_identity"):
        identity = upstream[key]
        require(
            set(identity) == {"pid", "start_ticks", "boot_id"}
            and type(identity["pid"]) is int
            and identity["pid"] > 0
            and type(identity["start_ticks"]) is int
            and identity["start_ticks"] >= 0
            and isinstance(identity["boot_id"], str)
            and identity["boot_id"],
            "exact kernel identity required",
        )
    require(
        upstream["controller_identity"]["boot_id"] == upstream["supervisor_identity"]["boot_id"],
        "upstream boot identities differ",
    )
    for locator in upstream["source"].values():
        metadata(files, locator)
    root = Path(descriptor["source_checkout"])
    sources(files, root, descriptor["source"], descriptor.get("source_revision"))
    commands, request, spec, resolved, budget, joins, link = [
        read(files, packet[k]) for k in PACKET_KEYS[:-1]
    ]
    require(
        commands.get("schema") == "eagle_lane_source_commands_v1"
        and commands.get("execution_authorized_by_this_file") is False,
        "original non-authorizing commands required",
    )
    for name, locator in commands["source"].items():
        require(
            descriptor["source"].get("scripts/" + name) == locator,
            "command source differs from original checkout",
        )
    require(
        request.get("schema") == "eagle_lane_initial_preparation_v1"
        and request.get("optimizer_updates") == 0
        and request["config"] == packet["config"]
        and request["source"] == commands["source"]["train_nine_model_qat.py"],
        "initial request/current source mismatch; use exact original pinned checkout",
    )
    calibration = read(files, request["initializer_report"])
    require(
        spec.get("candidate") == "eagle_a1"
        and spec.get("family") == "eagle"
        and spec.get("initialization") == calibration.get("initializer")
        and spec["initialization"]["activation_bits"] == 1
        and set(resolved["candidates"]) == {"eagle_a1"}
        and resolved["candidates"]["eagle_a1"]["config"] == packet["config"]
        and resolved["candidates"]["eagle_a1"]["profile"] == "direct_a1",
        "direct A1 calibration/config differs",
    )
    require(
        calibration.get("schema") == "eagle_production_fusion_initializer_v1"
        and calibration.get("authenticated_full_source") == upstream["training_source"]
        and upstream["target"] == joins["runtime"]["inputs"]["target"]
        and upstream["base_model"] == joins["runtime"]["base_model"],
        "original A8 target/base/full TRAIN calibration ancestry differs",
    )
    continuous = read(files, spec["eagle_config"])
    require(continuous["training"]["activation_bits"] == [1], "continuous A1 precision differs")
    require(
        resolved["budget"] == packet["budget"]
        and budget.get("schema") == "nine_model_selected_budget_v1"
        and budget.get("human_selected") is False
        and set(budget["candidates"]) == {"eagle_a1"}
        and budget.get("authorization", {}).get("kind") == "human_delegated_operational_settings",
        "standing delegation must remain operational, not human-selected policy",
    )
    metadata(files, budget["authorization"]["record"])
    policy = read(files, descriptor["policy"])
    require(
        policy.get("schema") == "nine_model_lane_endpoint_policy_v1"
        and policy.get("authorization", {}).get("kind") == budget["authorization"]["kind"]
        and policy["authorization"]["record"]["sha256"]
        == budget["authorization"]["record"]["sha256"],
        "existing delegated policy provenance differs",
    )
    metadata(files, policy["authorization"]["record"])
    require(
        all(
            type(policy.get(key)) in (int, float) and math.isfinite(policy[key]) and policy[key] > 0
            for key in (
                "wait_wall_seconds",
                "export_wall_seconds",
                "evaluation_wall_seconds",
                "poll_seconds",
            )
        )
        and policy["poll_seconds"] <= 1800,
        "existing finite bounded endpoint policy required",
    )
    limits = budget["candidates"]["eagle_a1"]["training_limits"]
    require(
        all(continuous["training"][key] == value for key, value in limits.items()),
        "selected trainer budget differs",
    )
    directory = Path(packet["commands"]["path"]).parent
    for key, filename in (
        ("initial_request", "initial-preparation-request.json"),
        ("source_joins", "golden-source-joins.json"),
        ("generation_replay_link", "generation-replay-link.json"),
        ("config", "configs/eagle_a1.json"),
        ("resolved_inputs", "configs/resolved-inputs.json"),
        ("budget", "budget.json"),
    ):
        require(packet[key]["path"] == str(directory / filename), "packet path joins differ")
    require(
        descriptor["outputs"]["initial_receipt"]["path"]
        == str(directory / "initial-prepare-receipt.json")
        and descriptor["outputs"]["export_audit"]["path"]
        == str(directory / "initial-export-audit.json")
        and descriptor["outputs"]["production_inputs"]["path"]
        == str(directory / "production-inputs.json"),
        "known pending packet output paths differ",
    )
    initial = directory / "initial-prepare"
    checkpoint = initial / "checkpoints/step-000000000000-e000000-r000000000000/A1"
    expected_prepare = [
        commands["initial_prepare"][0],
        str(root / "scripts/train_nine_model_qat.py"),
        "--config",
        packet["config"]["path"],
        "--run-dir",
        str(initial),
        "--bundle-sha256",
        packet["initial_request"]["sha256"],
        "--stage-name",
        "eagle_a1/initial-prepare",
        "--completion-output",
        str(directory / "initial-prepare-receipt.json"),
        "--allow-cuda",
        "--prepare-only",
    ]
    expected_export = [
        commands["initial_export"][0],
        str(root / "scripts/export_recurrent_binary.py"),
        "--base",
        joins["runtime"]["base_model"]["path"],
        "--checkpoint",
        str(checkpoint / "joint.npz"),
        "--manifest",
        str(checkpoint / "joint.json"),
        "--output",
        str(directory / "initial-calibrated-a1.gguf"),
        "--audit",
        str(directory / "initial-export-audit.json"),
    ]
    require(
        commands["initial_prepare"] == expected_prepare
        and commands["initial_export"] == expected_export,
        "original initial command argv changed",
    )
    require(
        joins.get("initializer_report") == request["initializer_report"],
        "calibration source join differs",
    )
    require(link["source_joins"] == packet["source_joins"], "historical link packet join differs")
    generated, requests, replay = (
        lines(files, link["generation_receipts"]),
        lines(files, link["replay_requests"]),
        lines(files, packet["replay_receipts"]),
    )
    require(
        requests == link["requests"] and len(generated) == len(joins["cases"]) == 3,
        "original generated parents absent",
    )
    native = link["native"]
    require(
        native["binary"] == joins["runtime"]["inputs"]["teacher_binary"]
        and native["target"] == joins["runtime"]["inputs"]["target"]
        and native["source_revision"] == joins["runtime"]["native_source_revision"],
        "historical native producer differs",
    )
    metadata(files, native["client_source"])
    require(len(replay) == len(requests) == 3, "three original replay receipts required")
    require(
        {case["ancestry"]["domain"] for case in joins["cases"]} == {"prose", "code", "reasoning"}
        and len({case["ancestry"]["prompt_id"] for case in joins["cases"]}) == 3,
        "three distinct original TRAIN cases required",
    )
    for parent, replay_request, case, receipt in zip(
        generated, requests, joins["cases"], replay, strict=True
    ):
        ancestry = dict(case["ancestry"], prompt_length=parent["prompt_length"])
        require(
            parent["chain_ancestry"] == ancestry == replay_request["chain_ancestry"]
            and parent["tokens"] == replay_request["tokens"]
            and parent["decode_history"] == replay_request["decode_history"]
            and parent.get("prompt_source_sha256") == ancestry["prompt_sha256"]
            and parent.get("generation", {}).get("mode") == "native_target_greedy",
            "forged generation ancestry",
        )
        require(
            all(
                receipt.get(key) == replay_request[key]
                for key in ("tokens", "tap_ids", "logits_mode", "chain_ancestry", "decode_history")
            ),
            "replay differs from generated parent",
        )
        offset = 0
        for history in parent["decode_history"]:
            require(
                history.get("offset") == offset
                and type(history.get("count")) is int
                and 1 <= history["count"] <= 256
                and history.get("phase") in {"prefill", "target_only_greedy"}
                and history.get("kv_reused_from_same_chain") is (offset > 0),
                "noncontiguous native generation history",
            )
            offset += history["count"]
        require(offset == len(parent["tokens"]), "generation history token length differs")
        for row in (parent, receipt):
            require(
                row.get("client_source_sha256") == native["client_source"]["sha256"]
                and row.get("producer_binary_sha256") == native["binary"]["sha256"]
                and row.get("producer_source_revision") == native["source_revision"]
                and row.get("target_sha256") == native["target"]["sha256"],
                "forged historical producer",
            )
    snapshot = endpoint_snapshot(files, upstream, descriptor.get("upstream_snapshot"))
    stages = []
    for name, gpu, argv, prerequisite in (
        ("a8_natural_endpoint", False, None, []),
        ("initial_prepare", True, expected_prepare, ["a8_natural_endpoint"]),
        ("initial_export", False, expected_export, ["initial_prepare"]),
        ("independent_qa", False, None, ["initial_export"]),
        (
            "full_bind",
            False,
            [
                expected_prepare[0],
                str(root / "scripts/prepare_eagle_lane_packet.py"),
                "bind",
                "--packet",
                str(directory),
                "--generation-link-sha256",
                packet["generation_replay_link"]["sha256"],
                "--receipts",
                packet["replay_receipts"]["path"],
                "--initial-model",
                str(directory / "initial-calibrated-a1.gguf"),
                "--export-audit",
                descriptor["outputs"]["export_audit"]["path"],
                "--qa-ledger",
                descriptor["outputs"]["independent_qa"]["path"],
            ],
            ["independent_qa"],
        ),
        ("admission_plan", False, None, ["full_bind"]),
        ("lane_bundle", False, None, ["admission_plan"]),
        ("sm120_admission", True, None, ["lane_bundle"]),
        ("long_qat", True, None, ["sm120_admission"]),
        ("endpoint_plan", False, None, ["long_qat"]),
        ("endpoint", True, None, ["endpoint_plan"]),
    ):
        stages.append(
            {
                "name": name,
                "status": "PENDING",
                "depends_on": prerequisite,
                "gpu_stage": gpu,
                "argv": argv,
                "live_gates": LIVE_GATES if gpu else [],
                "execution_authorized_by_this_file": False,
            }
        )
    return {
        "schema": "nine_model_lane_handoff_plan_v1",
        "planner_source": {
            "generator": pin(Path(__file__)),
            "pipeline_contract": pin(ROOT / "src/w1a1_eagle/nine_model_pipeline.py"),
        },
        "candidate": "eagle_a1",
        "inputs": descriptor,
        "status": "PENDING",
        "campaign_complete": False,
        "production_ready": False,
        "execution_authorized_by_this_file": False,
        "gpu_queried": False,
        "training_changed": False,
        "fresh_live_verification_required": True,
        "upstream_saved_metadata": snapshot,
        "runtime_validators_required": [
            "run_nine_model_lane_endpoint.validate_plan/training_endpoint",
            "prepare_eagle_lane_packet.initial_export_join/bind",
            "prepare_nine_model_lane.build_lane",
            "run_nine_model_lane (combined admission then training)",
            "run_nine_model_lane_endpoint",
        ],
        "historical_replay": "reuse original receipts; no generation or replay stage",
        "raw_capture_and_checkpoint_bytes": (
            "NOT_CHECKED: existing runtime validators remain mandatory"
        ),
        "authorization": budget["authorization"],
        "stages": stages,
        "missing_runtime_pieces": list(OUTPUTS)
        + ["fresh per-stage leases/live release", "staged sequencer not implemented"],
    }


def publish(output, plan):
    """Exclusive creation: concurrent publication cannot replace previous history."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic exclusive publication; cannot replace a path.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", type=Path)
    group.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--availability", type=Path, help="optional saved lease check; grants no execution"
    )
    args = parser.parse_args()
    if args.plan:
        require(
            args.plan_sha256 and sha256(args.plan) == args.plan_sha256,
            "immutable handoff plan changed",
        )
        previous = json.loads(args.plan.read_text())
        plan = build(previous["inputs"])
        require(plan == previous, "handoff contract or dependencies changed")
    else:
        require(args.output is not None, "new frozen output required")
        plan = build(json.loads(args.inputs.read_text()))
        publish(args.output, plan)
    if args.availability:
        require(args.plan and args.plan_sha256, "lease inspection requires frozen plan")
        require_available(args.availability, args.plan_sha256)
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "status",
                    "production_ready",
                    "execution_authorized_by_this_file",
                    "fresh_live_verification_required",
                    "missing_runtime_pieces",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
