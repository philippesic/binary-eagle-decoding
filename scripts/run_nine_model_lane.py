#!/usr/bin/env python3
"""Admit and train one frozen production lane under detached remote_job supervision."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from w1a1_eagle.nine_model_admission import Admission, validate_plan  # noqa: E402
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    GATES,
    Files,
    LinuxResources,
    SubprocessRunner,
    atomic_json,
    deferred_termination,
    require,
    require_available,
    resource_gate,
    sha256,
    stop_signals,
)


def validate_lane(path, expected_sha256):
    require(sha256(path) == expected_sha256, "frozen lane SHA256 differs")
    lane = json.loads(path.read_text())
    require(
        lane.get("schema") == "nine_model_training_lane_v1"
        and lane.get("artifact_kind") == "production"
        and lane.get("preparation_complete") is True
        and lane.get("campaign_complete") is False,
        "production lane preparation required",
    )
    name = lane.get("candidate")
    require(name in CANDIDATES, "unknown candidate")
    require(
        set(lane["remaining_candidates"]) == set(CANDIDATES) - {name},
        "unfinished candidate inventory differs",
    )
    require(
        lane["target_policy"] == {"immutable": True, "weights": "f16", "kv": "f16"},
        "target/F16 KV policy differs",
    )
    files = Files()
    for record in lane["source"].values():
        files.check(record)
    for name in ("config", "admission_plan", "qa_ledger", "budget"):
        files.check(lane[name])
    budget = json.loads(Path(lane["budget"]["path"]).read_text())
    require(
        budget.get("human_selected") is False and budget["authorization"] == lane["authorization"],
        "delegated operational provenance differs",
    )
    require(
        lane["authorization"].get("kind") == "human_delegated_operational_settings",
        "delegation provenance required",
    )
    files.check(lane["authorization"]["record"])
    plan, _ = validate_plan(Path(lane["admission_plan"]["path"]))
    require(
        plan["schema"] == "nine_model_lane_sm120_plan_v1"
        and set(plan["candidates"]) == {lane["candidate"]},
        "staged admission scope differs",
    )
    require(
        plan["candidates"][lane["candidate"]]["config"] == lane["config"],
        "admission config differs",
    )
    require(
        all(lane["source"].get(name) == record for name, record in plan["source"].items()),
        "lane source inventory differs",
    )
    require(
        lane["training_wall_seconds"] == budget["candidates"][lane["candidate"]]["wall_seconds"],
        "lane operational wall cap differs",
    )
    if lane.get("timed_evaluation_plan"):
        files.check(lane["timed_evaluation_plan"])
    return lane, files, plan


def committed_training_checkpoint(run_dir, candidate):
    training = Path(run_dir) / "training"
    return training / (
        "latest.json" if candidate.startswith("eagle_") else "checkpoints/latest.json"
    )


def run_training_lifecycle(
    lane,
    lane_locator,
    run_dir,
    attempt,
    admission_path,
    supervisor_path,
    state,
    state_path,
    files,
    runner,
    resources,
    release,
    authorization,
    *,
    resume=False,
    evaluator=None,
):
    """A child must exit and release before export/eval, and eval before resume.

    Durable pending_training survives controller crashes; retries evaluate that
    same committed checkpoint in a new attempt, never retrain a paid boundary.
    """
    if evaluator is None:
        from evaluate_nine_model_timed_checkpoint import evaluate_checkpoint

        evaluator = evaluate_checkpoint
    timed = lane.get("timed_evaluation_plan")
    if timed:
        from evaluate_nine_model_timed_checkpoint import validate_plan as validate_timed_plan

        admission_plan, _ = validate_plan(Path(lane["admission_plan"]["path"]))
        validate_timed_plan(timed, lane, admission_plan, files)
    receipt_to_resume = state.get("evaluation_receipt")
    while True:
        authorization()
        pending = state.get("pending_training")
        if timed and resume and pending is None:
            from evaluate_nine_model_timed_checkpoint import recover_boundary_receipt

            release()
            pending = recover_boundary_receipt(
                lane, lane_locator, run_dir, attempt, admission_path, state, files
            )
        if pending:
            receipt = files.check(pending)
        else:
            stage = Path(attempt) / "training-attempts" / uuid.uuid4().hex
            receipt = stage / "train-receipt.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/train_nine_model_qat.py"),
                "--config",
                lane["config"]["path"],
                "--run-dir",
                str(Path(run_dir) / "training"),
                "--bundle-sha256",
                lane_locator["sha256"],
                "--stage-name",
                lane["candidate"] + "/train",
                "--completion-output",
                str(receipt),
                "--allow-cuda",
                "--admission",
                str(admission_path),
            ]
            if resume and committed_training_checkpoint(run_dir, lane["candidate"]).is_file():
                command.append("--resume")
                if receipt_to_resume:
                    files.check(receipt_to_resume)
                    command += [
                        "--evaluation-receipt",
                        receipt_to_resume["path"],
                        "--evaluation-receipt-sha256",
                        receipt_to_resume["sha256"],
                    ]
            state.update(status="training")
            atomic_json(state_path, state)
            runner.run(
                command,
                directory=stage / "train",
                stop_path=Path(run_dir) / "STOP",
                wall_seconds=lane["training_wall_seconds"],
            )
        endpoint = json.loads(receipt.read_text())
        reasons = (
            {"approved_budget_complete", "timed_evaluation_boundary"}
            if timed
            else {"approved_budget_complete"}
        )
        require(
            endpoint.get("schema") == "nine_model_stage_receipt_v1"
            and endpoint.get("stage") == lane["candidate"] + "/train"
            and endpoint.get("artifact_kind") == "production"
            and endpoint.get("status") == "PASS"
            and endpoint.get("committed") is True
            and endpoint.get("completion_reason") in reasons
            and endpoint.get("bundle_sha256") == lane_locator["sha256"]
            and endpoint.get("config_sha256") == lane["config"]["sha256"]
            and endpoint.get("hardware", {}).get("gpu_uuid") == lane["gpu_uuid"]
            and endpoint["hardware"].get("compute_capability") == [12, 0]
            and type(endpoint.get("counters", {}).get("step")) is int
            and endpoint["counters"]["step"] > 0,
            "positive-update committed exact lane endpoint missing",
        )
        files.check(endpoint["checkpoint"])
        training = {"path": str(receipt), "sha256": sha256(receipt)}
        state.update(
            train_receipt=training,
            pending_training=training,
            status="awaiting_evaluation" if timed else "training_complete",
        )
        atomic_json(state_path, state)
        release()
        atomic_json(state_path, state)
        if not timed:
            state.update(pending_training=None, evaluation_status="PENDING")
            return
        evaluation_attempt = Path(attempt) / "evaluations" / uuid.uuid4().hex
        authorization()
        evaluated = evaluator(
            lane_locator,
            {"path": str(state_path), "sha256": sha256(state_path)},
            {"path": str(supervisor_path), "sha256": sha256(supervisor_path)},
            training,
            timed,
            evaluation_attempt,
            files=files,
            observer=resources,
            runner=runner,
            release=release,
            authorization=authorization,
            stop_path=Path(run_dir) / "STOP",
        )
        authorization()
        release()
        result = json.loads(files.check(evaluated).read_text())
        request_locator = endpoint["timed_evaluation_request"]
        request = json.loads(files.check(request_locator).read_text())
        require(
            result.get("schema") == "nine_model_timed_evaluation_receipt_v1"
            and result.get("status") == "PASS"
            and result.get("completed") is True
            and result.get("request") == request_locator
            and result.get("checkpoint") == endpoint["checkpoint"]
            and result.get("candidate") == lane["candidate"]
            and result.get("bundle_sha256") == lane_locator["sha256"]
            and result.get("config_sha256") == lane["config"]["sha256"]
            and result.get("protocol") == request["protocol"]
            and result.get("owned_release", {}).get("owned_process_groups_absent") is True
            and result["owned_release"].get("owned_cuda_pids_absent") is True,
            "authenticated exact-checkpoint evaluation receipt missing",
        )
        files.check(result["evaluation"])
        final = endpoint["completion_reason"] == "approved_budget_complete"
        require(
            request.get("final_training_complete") is final, "training completion request differs"
        )
        state.setdefault("evaluations", []).append(evaluated)
        state.update(
            pending_training=None,
            evaluation_receipt=evaluated,
            evaluation_status="PASS",
            status="training_complete" if final else "awaiting_evaluation",
        )
        atomic_json(state_path, state)
        if final:
            return
        resume, receipt_to_resume = True, evaluated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lane", "run-dir", "availability", "supervisor-state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--lane-sha256", required=True)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    lane, files, plan = validate_lane(args.lane, args.lane_sha256)
    if lane.get("timed_evaluation_plan"):
        from evaluate_nine_model_timed_checkpoint import validate_plan as validate_timed_plan

        validate_timed_plan(lane["timed_evaluation_plan"], lane, plan, files)
    if not args.start:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gpu_queried": False,
                    "training_started": False,
                    "candidate": lane["candidate"],
                    "campaign_complete": False,
                }
            )
        )
        return
    lease = require_available(args.availability, args.lane_sha256)

    def authorization():
        control = json.loads(Path(lane["gpu_control_path"]).read_text())
        if control.get("rtx5080", {}).get("pause_requested") is not False:
            raise InterruptedError("RTX5080 paused or control absent")
        require(
            json.loads(args.availability.read_text()) == lease, "availability ownership changed"
        )
        if (args.run_dir / "STOP").exists():
            raise InterruptedError("lane STOP requested")

    authorization()
    require(lease["gpu_uuid"] == lane["gpu_uuid"], "lease physical GPU differs")
    supervisor = json.loads(args.supervisor_state.read_text())
    require(
        supervisor.get("pid") == os.getpid()
        and supervisor.get("supervisor_pid") == os.getppid()
        and supervisor.get("status") == "running"
        and supervisor.get("stop_grace_seconds", 0) >= 90
        and os.environ.get("TMUX"),
        "detached Linux tmux remote_job supervision with 90-second grace required",
    )
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    state_path = args.run_dir / "state.json"
    state = {
        "schema": "nine_model_lane_state_v1",
        "bundle_sha256": args.lane_sha256,
        "candidate": lane["candidate"],
        "campaign_complete": False,
    }
    if args.resume:
        previous = json.loads(state_path.read_text())
        require(
            previous["bundle_sha256"] == args.lane_sha256
            and previous["candidate"] == lane["candidate"],
            "resume frozen lane differs",
        )
        require(
            previous.get("status") != "training_complete",
            "successful endpoint cannot restart training",
        )
        state["previous_attempt"] = previous
        for key in ("pending_training", "evaluation_receipt", "evaluations"):
            if key in previous:
                state[key] = previous[key]
    lock_path = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resources = LinuxResources(lane["gpu_uuid"])
    runner = SubprocessRunner(ROOT, lane.get("environment"), authorization=authorization)
    baseline = None

    def release():
        proof = resources.require_released(runner.process_groups, runner.process_identities)
        observed = resources.snapshot()
        resource_gate(observed, baseline, lane["resource_policy"])
        require(
            proof.get("other_context_pids") == [] and observed.get("dxg_holders") == [],
            "foreign GPU contexts forbid lane continuation",
        )
        state.update(owned_release=proof, resource_return=observed)
        return proof

    with lock_path.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with stop_signals():
                baseline = resources.snapshot()
                resource_gate(baseline, baseline, lane["resource_policy"])
                state.update(status="admitting", resource_baseline=baseline)
                atomic_json(state_path, state)
                if args.resume:
                    for process_path in args.run_dir.glob("**/process.json"):
                        process = json.loads(process_path.read_text())
                        require(
                            process.get("kernel_identity") is not None,
                            "prior process identity absent",
                        )
                        runner.process_groups.append(process["pgid"])
                        runner.process_identities.append(process["kernel_identity"])
                    for lineage_path in args.run_dir.glob("**/process-lineage.json"):
                        runner.process_identities.extend(
                            json.loads(lineage_path.read_text())["kernel_identities"]
                        )
                    release()
                attempt = args.run_dir / "attempts" / uuid.uuid4().hex
                admission = Admission(
                    plan, files, runner, resources, args.lane_sha256, attempt / "admission"
                ).execute(attempt / "admission-receipt.json")
                require(
                    admission["gates"] == dict.fromkeys(sorted(GATES), "PASS"),
                    "selected lane gates incomplete",
                )
                selected = admission["candidate_admissions"][lane["candidate"]]
                admission_path = files.check(selected)
                release()
                state.update(status="training", admission=selected, attempt=str(attempt))
                atomic_json(state_path, state)
                run_training_lifecycle(
                    lane,
                    {"path": str(args.lane), "sha256": args.lane_sha256},
                    args.run_dir,
                    attempt,
                    admission_path,
                    args.supervisor_state,
                    state,
                    state_path,
                    files,
                    runner,
                    resources,
                    release,
                    authorization,
                    resume=args.resume,
                )
        except BaseException as error:
            with deferred_termination():
                state.update(
                    status="stopped" if isinstance(error, InterruptedError) else "failed",
                    failure={"type": type(error).__name__, "reason": str(error)},
                    checkpoints_deleted=False,
                )
                if baseline is not None:
                    try:
                        release()
                    except BaseException as cleanup:
                        state["cleanup_failure"] = str(cleanup)
            raise
        finally:
            atomic_json(state_path, state)
    print(json.dumps(state))


if __name__ == "__main__":
    main()
