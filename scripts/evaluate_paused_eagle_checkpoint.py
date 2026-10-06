#!/usr/bin/env python3
"""Evaluate a released paused EAGLE A8 publication without completing its budget.

Default is read-only inspection. --start requires fresh remote_job supervision,
availability and the shared GPU lock. The existing frozen CPU serializer and
native endpoint helper produce two independent development comparisons. No
training state, old receipt, plan, watcher or optimizer is changed.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Resolve the frozen helper checkout before importing any of its modules.
# The adapter lives in a later checkout and must not silently import its helpers.
_bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_bootstrap.add_argument("--helper-checkout", type=Path, default=ROOT)
_helper_args, _ = _bootstrap.parse_known_args()
HELPER_ROOT = _helper_args.helper_checkout.resolve()
sys.path[:0] = [str(HELPER_ROOT / "scripts"), str(HELPER_ROOT / "src")]
import check_qat_optimization_readiness as shapes  # noqa: E402
import run_nine_model_lane_endpoint as endpoint  # noqa: E402

from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    LinuxResources,
    SubprocessRunner,
    atomic_json,
    identity_active,
    process_identity,
    require,
    require_available,
    resource_gate,
    sha256,
    stop_signals,
)

INITIAL_SHA256 = "b2f6dddcf2eaad503a52523ed6e02061c388c3c92aa60241e212aab49611aee8"
SOURCE_NAMES = dict(
    zip(
        (
            "fc",
            "blk.0.attn_q",
            "blk.0.attn_k",
            "blk.0.attn_v",
            "blk.0.attn_output",
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            "output",
        ),
        shapes.PINNED_SHAPES,
        strict=True,
    )
)


def read(path):
    return json.loads(Path(path).read_text())


def checked(files, path, expected):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), "canonical nonsymlink input required")
    record = {"path": str(path.resolve()), "sha256": expected}
    files.check(record)
    return record


def validate_joint(joint, npz, base):
    require(
        joint.get("checkpoint_sha256") == npz["sha256"]
        and joint.get("base_gguf_sha256") == base["sha256"]
        and joint.get("activation_bits") == 8
        and joint.get("scale_layout") == "row",
        "joint checkpoint/base/A8 row contract differs",
    )
    expected = {
        base_name: {"checkpoint_name": name + ".weight", "shape": shapes.PINNED_SHAPES[name]}
        for base_name, name in SOURCE_NAMES.items()
    }
    require(joint.get("projections") == expected, "exact all-nine projection names/shapes required")


def development_authority(plan, files, *, loader=None):
    """Authenticate file-level split authority for the original bare-row schema."""
    require(
        plan["prompts"]["sha256"] == endpoint.ORIGINAL_DEVELOPMENT_PROMPTS_SHA256,
        "only exact original development prompt bytes supported",
    )
    path = files.check(plan["prompts"])
    admission = read(files.check(plan["development_admission"]))
    require(
        admission.get("schema") == "nine_model_lane_development_admission_v1"
        and admission.get("split") == "development"
        and admission.get("training_disjoint") is True
        and admission.get("prompts") == plan["prompts"]
        and admission.get("frozen_training_source") == plan["training_source"]
        and isinstance(admission.get("evidence"), list)
        and admission["evidence"],
        "authenticated original development/disjoint TRAIN authority required",
    )
    for record in admission["evidence"]:
        files.check(record)
    rows = (endpoint.load_opaque_prompts if loader is None else loader)(path)
    require(
        len(rows) == 24
        and Counter(row.get("domain") for row in rows) == {"code": 8, "prose": 8, "reasoning": 8},
        "original24 development domain inventory differs",
    )
    require(
        all("split" not in row or row["split"] == "development" for row in rows),
        "explicit prompt split conflicts with admitted development authority",
    )
    authority = {
        "schema": "paused_eagle_file_level_development_authority_v1",
        "prompts": plan["prompts"],
        "admission": plan["development_admission"],
        "evidence": admission["evidence"],
        "frozen_training_source_sha256": plan["training_source_sha256"],
        "row_count": len(rows),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "missing_row_split_count": sum("split" not in row for row in rows),
        "adaptation": (
            "missing row split inherited in memory from authenticated "
            "file-level development admission"
        ),
        "original_prompt_bytes_changed": False,
    }
    return authority, rows


@contextmanager
def admitted_development_rows(plan, files):
    """Scope metadata adaptation to one authenticated locator; restore always."""
    original = endpoint.load_opaque_prompts

    def load(path):
        require(
            Path(path).resolve() == Path(plan["prompts"]["path"]),
            "prompt adaptation locator differs",
        )
        _, rows = development_authority(plan, files, loader=original)
        adapted = copy.deepcopy(rows)
        for row in adapted:
            row.setdefault("split", "development")
        return adapted

    endpoint.load_opaque_prompts = load
    try:
        yield
    finally:
        endpoint.load_opaque_prompts = original


def validate_export(plan, files, checkpoint, model, audit_record):
    files.check(model)
    audit = read(files.check(audit_record))
    require(
        audit.get("serialization_audit_passed") is True
        and audit.get("base_gguf") == plan["base_model"]
        and audit.get("checkpoint") == checkpoint["records"]["joint.npz"]
        and audit.get("checkpoint_manifest") == checkpoint["records"]["joint.json"]
        and audit.get("output") == model
        and audit.get("activation_bits") == 8
        and set(audit.get("projections", {})) == set(SOURCE_NAMES),
        "trained serialization must join exact paused publication",
    )
    return {
        "model": model,
        "audit": audit_record,
        "checkpoint_kind": "paused_trained_snapshot",
        "checkpoint": checkpoint["records"]["resume"],
    }


def publication(plan, files, directory, resume_sha, *, initial=False):
    directory = Path(directory).resolve()
    outer = read(directory / "manifest.json")
    require(
        outer.get("schema") == "continuous_joint_w1ax_v1"
        and outer.get("optimizer_rng_cursor_exact") is True
        and outer.get("source_sha256") == plan["training_source_sha256"]
        and set(outer.get("exports", {})) == {"A8"},
        "exact original single-lane publication/source required",
    )
    require(
        all(type(outer.get(k)) is int and outer[k] >= 0 for k in ("step", "epoch", "cursor"))
        and (
            (outer["step"] == outer["epoch"] == outer["cursor"] == 0)
            if initial
            else outer["step"] > 0
        ),
        "publication counters differ from declared snapshot kind",
    )
    require(set(outer["exports"]["A8"]) == {"joint.npz", "joint.json"}, "joint inventory differs")
    require(outer.get("sha256") == resume_sha, "resume hash differs from publication")
    records = {"publication": endpoint.pin(directory / "manifest.json")}
    records["resume"] = checked(files, directory / "resume.pt", resume_sha)
    for name in ("joint.npz", "joint.json"):
        records[name] = checked(files, directory / "A8" / name, outer["exports"]["A8"][name])
    validate_joint(read(records["joint.json"]["path"]), records["joint.npz"], plan["base_model"])
    return {"outer": outer, "records": records}


def inspect(args):
    require(
        Path(args.helper_checkout).resolve() == HELPER_ROOT,
        "helper checkout differs from imported runtime",
    )
    require(endpoint.ROOT.resolve() == HELPER_ROOT, "native helper imported from another checkout")
    plan, files, policy, protocol = endpoint.validate_plan(args.plan, args.plan_sha256)
    require(plan["candidate"] == "eagle_a8", "paused adapter supports EAGLE A8 only")
    for name in (
        "scripts/run_nine_model_lane_endpoint.py",
        "scripts/benchmark_native_eagle.py",
        "scripts/benchmark_dspark_screen.py",
        "src/w1a1_eagle/nine_model_pipeline.py",
    ):
        require(
            sha256(HELPER_ROOT / name) == plan["source"][name]["sha256"],
            "imported frozen helper differs: " + name,
        )
    require(
        protocol["repetitions"] == 5
        and protocol["warmups_per_cell"] == 2
        and protocol["max_output_tokens"] == 128
        and protocol["split"] == "development",
        "original five-repetition/two-warmup development protocol required",
    )
    require(
        policy["export_wall_seconds"] <= 600 and policy["evaluation_wall_seconds"] <= 1200,
        "bounded export600/evaluation1200 caps required",
    )
    authority, _ = development_authority(plan, files)
    checkpoint = Path(args.checkpoint_dir).resolve()
    training = Path(plan["training_run_dir"]) / "training"
    require(
        checkpoint.parent == training / "checkpoints", "checkpoint outside original training run"
    )
    selected = publication(plan, files, checkpoint, args.expected_resume_sha)
    selected["development_prompt_authority"] = authority
    outer = selected["outer"]
    require(
        checkpoint.name
        == f"step-{outer['step']:012d}-e{outer['epoch']:06d}-r{outer['cursor']:012d}",
        "checkpoint directory/counters differ",
    )
    status = read(training / "status.json")
    budget = read(training / "budget-used.json")
    require(
        status.get("status") == "stopped"
        and status.get("step") == outer["step"]
        and status.get("checkpoint") == {**selected["records"]["resume"], "step": outer["step"]}
        and budget.get("schema") == "continuous_training_budget_v1"
        and budget.get("source_sha256") == plan["training_source_sha256"]
        and budget.get("active_attempt") is None
        and budget.get("max_seconds") == plan["training_limits"]["max_seconds"]
        and type(budget.get("training_seconds")) in (int, float)
        and math.isfinite(budget["training_seconds"])
        and budget["training_seconds"] > 0
        and status.get("training_elapsed_seconds") == budget["training_seconds"],
        "stopped checkpoint and settled original accounting must agree",
    )
    selected["records"].update(
        status=endpoint.pin(training / "status.json"),
        budget=endpoint.pin(training / "budget-used.json"),
    )
    require(
        args.initial_model_sha256 == INITIAL_SHA256,
        "exact original calibrated zero-update GGUF required",
    )
    model = checked(files, args.initial_model, args.initial_model_sha256)
    audit_record = checked(files, args.initial_audit, args.initial_audit_sha256)
    audit = read(audit_record["path"])
    require(
        audit.get("serialization_audit_passed") is True
        and audit.get("output") == model
        and audit.get("base_gguf") == plan["base_model"]
        and audit.get("activation_bits") == 8
        and set(audit.get("projections", {})) == set(SOURCE_NAMES),
        "initial serialization ancestry differs",
    )
    npz, joint = audit["checkpoint"], audit["checkpoint_manifest"]
    files.check(npz)
    files.check(joint)
    zero_dir = Path(npz["path"]).parent.parent
    zero = publication(
        plan, files, zero_dir, read(zero_dir / "manifest.json")["sha256"], initial=True
    )
    require(
        npz == zero["records"]["joint.npz"] and joint == zero["records"]["joint.json"],
        "initial audit/publication join differs",
    )
    initial = {
        "model": model,
        "audit": audit_record,
        "checkpoint_kind": "calibrated_zero_update",
        "optimizer_updates": 0,
        "publication": zero,
    }
    require(
        Path(args.control_path).resolve() == Path(plan["gpu_control_path"]).resolve(),
        "pause control path differs",
    )
    existing = [
        getattr(args, name, None)
        for name in (
            "existing_export",
            "existing_export_sha256",
            "existing_export_audit",
            "existing_export_audit_sha256",
        )
    ]
    require(
        not any(existing) or all(existing),
        "existing export requires model/audit paths and SHA256s together",
    )
    if all(existing):
        selected["existing_export"] = validate_export(
            plan,
            files,
            selected,
            checked(files, existing[0], existing[1]),
            checked(files, existing[2], existing[3]),
        )
    return plan, files, policy, protocol, selected, initial, budget


def require_unpaused(path):
    require(read(path).get("rtx5080", {}).get("pause_requested") is False, "RTX5080 remains paused")


def supervised(args):
    state = read(args.supervisor_state)
    require(
        state.get("pid") == os.getpid()
        and state.get("supervisor_pid") == os.getppid()
        and state.get("status") == "running"
        and state.get("stop_grace_seconds", 0) >= 90
        and bool(os.environ.get("TMUX")),
        "fresh detached remote_job controller with90s stop grace required",
    )
    return process_identity(os.getpid()), process_identity(os.getppid())


@contextmanager
def owner_lock(path, owner):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield stream
        finally:
            # Leave exact retired owner identity for audit; closing releases flock.
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def wall_cap(seconds):
    require(
        signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
        "existing deadline timer cannot be replaced",
    )
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_sig, _frame):
        raise TimeoutError("paused comparison stage wall cap exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def summarize(report):
    result = {}
    for cell, value in report["cells"].items():
        tokens, seconds = value["output_tokens"], value["request_seconds"]
        require(
            type(tokens) is int and tokens > 0 and seconds > 0 and math.isfinite(seconds),
            "invalid measured throughput denominator",
        )
        item = {
            "output_tokens": tokens,
            "request_seconds": seconds,
            "request_tokens_per_second": tokens / seconds,
        }
        if cell != "target_only":
            counts = value["native_counts"]
            require(
                all(
                    type(counts.get(k)) is int and counts[k] >= 0
                    for k in ("accepted", "proposed", "rounds")
                )
                and counts["proposed"] > 0
                and counts["rounds"] > 0
                and counts["accepted"] <= counts["proposed"],
                "invalid native acceptance denominators",
            )
            item.update(
                native_counts=counts,
                acceptance_rate=counts["accepted"] / counts["proposed"],
                accepted_per_round=counts["accepted"] / counts["rounds"],
            )
        result[cell] = item
    require(
        set(result) == {"eagle_a8", "eagle_q4", "target_only"},
        "independent matched comparison cells differ",
    )
    result["eagle_a8"]["request_speed_vs_q4"] = (
        result["eagle_a8"]["request_tokens_per_second"]
        / result["eagle_q4"]["request_tokens_per_second"]
    )
    result["eagle_a8"]["request_speed_vs_target_only"] = (
        result["eagle_a8"]["request_tokens_per_second"]
        / result["target_only"]["request_tokens_per_second"]
    )
    return result


def run(args):
    plan, files, policy, protocol, checkpoint, initial, budget = inspect(args)
    if not args.start:
        return {
            "status": "INSPECTED_NOT_EXECUTED",
            "campaign_complete": False,
            "gpu_queried": False,
            "checkpoint": checkpoint,
            "initial": initial,
            "accounted_training_seconds": budget["training_seconds"],
        }
    require(
        args.availability and args.supervisor_state, "fresh availability/supervisor paths required"
    )
    require_unpaused(args.control_path)
    owner, supervisor = supervised(args)
    lease = require_available(args.availability, args.plan_sha256)
    require(lease["gpu_uuid"] == plan["gpu_uuid"], "fresh availability GPU differs")
    require(
        owner not in (plan["controller_identity"], plan["supervisor_identity"]),
        "retired endpoint handles forbidden",
    )
    output = Path(args.output_root).resolve()
    require(
        not output.exists() and not output.is_relative_to(Path(plan["training_run_dir"])),
        "new output outside original training run required",
    )
    lock_path = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    with owner_lock(lock_path, owner), stop_signals():
        output.mkdir(parents=True)
        observer = LinuxResources(plan["gpu_uuid"])
        runner = SubprocessRunner(
            HELPER_ROOT, {**plan.get("environment", {}), "CUDA_VISIBLE_DEVICES": ""}
        )
        groups, identities = endpoint.owned_training_processes(plan)
        identities += [plan["controller_identity"], plan["supervisor_identity"]]
        baseline = None
        records = [
            *checkpoint["records"].values(),
            *initial["publication"]["records"].values(),
            initial["model"],
            initial["audit"],
            plan["prompts"],
            plan["development_admission"],
            *checkpoint["development_prompt_authority"]["evidence"],
        ]
        release = {"status": "PENDING"}

        def authorization():
            require_unpaused(args.control_path)
            require(not (output / "STOP").exists(), "paused comparison STOP requested")
            require(read(args.availability) == lease, "fresh availability changed")
            require(
                identity_active(owner) and identity_active(supervisor),
                "fresh controller/supervisor identity changed",
            )
            current = read(args.supervisor_state)
            require(
                current.get("pid") == owner["pid"]
                and current.get("supervisor_pid") == supervisor["pid"]
                and current.get("status") == "running",
                "remote_job supervision changed",
            )
            require(read(lock_path) == owner, "shared GPU lock owner differs")
            require(
                not any(identity_active(i) for i in identities),
                "original training/producer still active",
            )
            for record in [*plan["source"].values(), *plan["runtime"].values(), *records]:
                files.check(record)

        def released():
            evaluated_groups, evaluated_identities = [], []
            for process_file in output.glob("*/**/process.json"):
                process = read(process_file)
                identity = process.get("kernel_identity")
                if identity is not None:
                    require(
                        process.get("pid") == process.get("pgid") == identity["pid"],
                        "owned evaluation process/group identity differs",
                    )
                    evaluated_groups.append(process["pgid"])
                    evaluated_identities.append(identity)
            proof = observer.require_released(
                groups + runner.process_groups + evaluated_groups,
                identities + runner.process_identities + evaluated_identities,
            )
            current = observer.snapshot()
            require(
                proof.get("other_context_pids") == [] and current.get("dxg_holders") == [],
                "foreign CUDA/DXG context forbids comparison",
            )
            resource_gate(
                current, current if baseline is None else baseline, plan["resource_policy"]
            )
            return current

        def cell_authorization():
            authorization()
            # Endpoint calls before each server and before each HTTP request.
            # Return-to-baseline checks apply between servers, not while loaded.
            if endpoint.available_port("127.0.0.1", protocol["port"]):
                released()

        try:
            authorization()
            baseline = released()
            atomic_json(
                output / "input-ancestry.json",
                {
                    "plan": endpoint.pin(args.plan),
                    "checkpoint": checkpoint,
                    "initial": initial,
                    "budget": budget,
                    "fresh_owner": owner,
                    "fresh_supervisor": supervisor,
                    "availability": endpoint.pin(args.availability),
                    "original_training_budget_complete": False,
                },
            )
            if "existing_export" in checkpoint:
                trained = checkpoint["existing_export"]
                atomic_json(output / "existing-export-reuse.json", trained)
            else:
                export_dir = output / "export"
                export_dir.mkdir()
                command = [
                    plan["serializer_python_invocation"],
                    plan["serializer"]["path"],
                    "--base",
                    plan["base_model"]["path"],
                    "--checkpoint",
                    checkpoint["records"]["joint.npz"]["path"],
                    "--manifest",
                    checkpoint["records"]["joint.json"]["path"],
                    "--output",
                    str(export_dir / "trained.gguf"),
                    "--audit",
                    str(export_dir / "export-audit.json"),
                ]
                runner.authorization = authorization
                runner.run(
                    command,
                    directory=output / "export-process",
                    stop_path=output / "STOP",
                    wall_seconds=policy["export_wall_seconds"],
                )
                released()
                trained = validate_export(
                    plan,
                    files,
                    checkpoint,
                    endpoint.pin(export_dir / "trained.gguf"),
                    endpoint.pin(export_dir / "export-audit.json"),
                )
            records.extend((trained["model"], trained["audit"]))
            comparisons = {}
            for label, exported in (
                ("paused_trained_snapshot", trained),
                ("calibrated_zero_update", initial),
            ):
                cell_authorization()
                with (
                    wall_cap(policy["evaluation_wall_seconds"]),
                    admitted_development_rows(plan, files),
                ):
                    receipt = endpoint.evaluate_endpoint(
                        plan,
                        files,
                        protocol,
                        exported,
                        output / label,
                        output / "STOP",
                        cell_authorization,
                    )
                released()
                report = read(receipt["report"]["path"])
                comparisons[label] = {
                    "comparison_kind": label,
                    "independent_comparison": True,
                    "helper_report": receipt["report"],
                    "evaluated_model": exported["model"],
                    "development_prompt_authority": checkpoint["development_prompt_authority"],
                    "initial_optimizer_updates": 0 if label == "calibrated_zero_update" else None,
                    "cells": summarize(report),
                }
                atomic_json(output / f"{label}-comparison.json", comparisons[label])
            release = {"status": "PASS", "resources": released()}
            result = {
                "schema": "paused_eagle_checkpoint_comparison_v1",
                "status": "complete",
                "campaign_complete": False,
                "original_training_budget_complete": False,
                "accounted_training_seconds": budget["training_seconds"],
                "original_max_training_seconds": budget["max_seconds"],
                "comparisons": comparisons,
                "comparison_scope": (
                    "two independent paired development comparisons; "
                    "no pooled or four-cell interleaved claim"
                ),
                "owned_release": release,
            }
            atomic_json(output / "result.json", result)
            return result
        except BaseException as error:
            try:
                release = {"status": "PASS", "resources": released()}
            except BaseException as cleanup:
                release = {"status": "FAILED", "reason": str(cleanup)}
            atomic_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "campaign_complete": False,
                    "original_training_budget_complete": False,
                    "checkpoint": checkpoint,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "owned_release": release,
                },
            )
            raise


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (
        "helper-checkout",
        "plan",
        "checkpoint-dir",
        "output-root",
        "control-path",
        "initial-model",
        "initial-audit",
    ):
        cli.add_argument("--" + name, type=Path, required=True)
    for name in (
        "plan-sha256",
        "expected-resume-sha",
        "initial-model-sha256",
        "initial-audit-sha256",
    ):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--existing-export", type=Path)
    cli.add_argument("--existing-export-sha256")
    cli.add_argument("--existing-export-audit", type=Path)
    cli.add_argument("--existing-export-audit-sha256")
    cli.add_argument("--availability", type=Path)
    cli.add_argument("--supervisor-state", type=Path)
    cli.add_argument("--start", action="store_true")
    return cli


def main():
    print(json.dumps(run(parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
