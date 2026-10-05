#!/usr/bin/env python3
"""Freeze/inspect or supervise a separately pinned independent-endpoint comparison.

Preparation opens no prompt messages and starts no process. Execution is only a
remote_job controller plus a typed child continuation under the shared GPU lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_nine_model_lane_endpoint import require_held_owner_lock  # noqa: E402

from w1a1_eagle.nine_model_endpoint_collection import (  # noqa: E402
    evaluation_view,
    validate_collection,
)
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    CELLS,
    Files,
    LinuxResources,
    SubprocessRunner,
    atomic_json,
    cleanup_descendants,
    descendant_identities,
    identity_active,
    process_identity,
    require,
    require_available,
    resource_gate,
    sha256,
    stop_signals,
)

SCHEMA = "nine_model_collection_runtime_plan_v1"
LOCK = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def source_pins():
    # New dispatcher AND evaluator plus local transitive Python code. Historical
    # evaluation-source pins remain original and are validated independently.
    return {
        "artifact:python": pin(sys.executable),
        **{
            str(p.relative_to(ROOT)): pin(p)
            for directory in ("scripts", "src/w1a1_eagle")
            for p in sorted((ROOT / directory).rglob("*.py"))
        },
    }


def read(files, locator):
    return json.loads(files.check(locator).read_text())


def contains(value, identity):
    if value == identity:
        return True
    if isinstance(value, dict):
        return any(contains(v, identity) for v in value.values())
    if isinstance(value, list):
        return any(contains(v, identity) for v in value)
    return False


def upstream_records(context, owners):
    """Import original process.json/lineage and original controller birth proof.

    Birth proof must be an existing hashed producer/monitor record containing the
    exact identities; terminal supervisor PID values alone cannot prove birth.
    """
    files = context["files"]
    require(set(owners) == set(CANDIDATES), "six original upstream owners required")
    output = {}
    for name, selected in context["collection"]["candidates"].items():
        owner = owners[name]
        require(
            set(owner) == {"controller", "supervisor", "identity_evidence"},
            "exact original upstream owner proof required",
        )
        terminal = read(files, selected["supervisor_state"])
        state = read(files, selected["lane_state"])
        evidence = read(files, owner["identity_evidence"])
        require(
            terminal["pid"] == owner["controller"]["pid"]
            and terminal["supervisor_pid"] == owner["supervisor"]["pid"],
            "upstream owner differs from original supervisor",
        )
        for identity in (owner["controller"], owner["supervisor"]):
            require(
                set(identity) == {"pid", "start_ticks", "boot_id"}
                and type(identity["pid"]) is int
                and identity["pid"] > 0
                and type(identity["start_ticks"]) is int
                and identity["start_ticks"] > 0
                and identity["boot_id"] == state["resource_baseline"]["boot_id"]
                and contains(evidence, identity),
                "original upstream birth proof absent",
            )
        run = Path(selected["lane_state"]["path"]).parent
        processes = [
            pin(p)
            for pattern in ("**/process.json", "**/process-lineage.json")
            for p in sorted(run.glob(pattern))
        ]
        require(processes, "original upstream process census absent")
        require(
            type(terminal.get("pgid")) is int
            and terminal["pgid"] == terminal["pid"]
            and type(terminal.get("supervisor_pgid")) is int
            and terminal["supervisor_pgid"] == owner["supervisor"]["pid"],
            "original upstream controller/supervisor process groups absent",
        )
        output[name] = {
            **owner,
            "process_records": processes,
            "controller_pgid": terminal["pgid"],
            "supervisor_pgid": terminal["supervisor_pgid"],
        }
    return output


def additional_upstream_records(context, jobs):
    """Separate supervised export/watcher jobs are not lane-training descendants."""
    require(isinstance(jobs, dict) and jobs, "explicit upstream export/watcher jobs required")
    files = context["files"]
    result = {}
    for name, job in jobs.items():
        require(
            set(job)
            == {"run_dir", "supervisor_state", "controller", "supervisor", "identity_evidence"},
            "exact additional upstream job fields required",
        )
        run = Path(job["run_dir"])
        require(
            run.is_absolute() and run.resolve() == run and run.is_dir(),
            "canonical original upstream producer run required",
        )
        terminal = read(files, job["supervisor_state"])
        evidence = read(files, job["identity_evidence"])
        require(
            terminal.get("status") == "finished"
            and type(terminal.get("exit_code")) is int
            and terminal.get("exit_code") == 0
            and terminal.get("received_signal") is None
            and terminal.get("pid") == job["controller"]["pid"]
            and terminal.get("supervisor_pid") == job["supervisor"]["pid"]
            and terminal.get("pgid") == job["controller"]["pid"]
            and terminal.get("supervisor_pgid") == job["supervisor"]["pid"]
            and job["run_dir"] in terminal.get("command", []),
            "original extra upstream natural supervised endpoint differs",
        )
        for identity in (job["controller"], job["supervisor"]):
            require(
                set(identity) == {"pid", "start_ticks", "boot_id"}
                and type(identity["pid"]) is int
                and identity["pid"] > 0
                and type(identity["start_ticks"]) is int
                and identity["start_ticks"] > 0
                and contains(evidence, identity),
                "original extra upstream birth proof absent",
            )
        records = [
            pin(p)
            for pattern in ("**/process.json", "**/process-lineage.json")
            for p in sorted(run.glob(pattern))
        ]
        require(records, "original extra upstream process census absent")
        result[name] = {
            **job,
            "process_records": records,
            "controller_pgid": terminal["pgid"],
            "supervisor_pgid": terminal["supervisor_pgid"],
        }
    require(
        all(
            any(
                Path(candidate["export_receipt"]["path"]).is_relative_to(Path(job["run_dir"]))
                for job in jobs.values()
            )
            for candidate in context["collection"]["candidates"].values()
        ),
        "every original export producer must be covered by upstream supervised release",
    )
    return result


def binding(context):
    lanes = [context["contexts"][name]["lane"] for name in CANDIDATES]
    origin = context["source"]["origin"]
    require(
        all(lane["gpu_control_path"] == lanes[0]["gpu_control_path"] for lane in lanes),
        "one authenticated live host control required",
    )
    # Evaluation settings belong to the authentic evaluation source. Independent
    # historical training checkouts need not have identical library paths.
    policy = origin["resource_policy"]
    require(
        all(
            policy[f"{kind}_floor_bytes"] >= lane["resource_policy"][f"{kind}_floor_bytes"]
            and policy[f"{kind}_return_tolerance_bytes"]
            <= lane["resource_policy"][f"{kind}_return_tolerance_bytes"]
            for lane in lanes
            for kind in ("gpu", "host")
        ),
        "evaluation source weakens an original lane resource/release constraint",
    )
    protocol = context["protocol"]
    require(
        set(protocol["draft_lengths"]) == {"eagle", "dspark", "dflash"}
        and type(protocol["evaluation_wall_seconds"]) in (int, float)
        and 0 < protocol["evaluation_wall_seconds"] < float("inf"),
        "authenticated full-nine bounded native protocol required",
    )
    return {
        "resource_policy": policy,
        "environment": origin["environment"],
        "gpu_control_path": lanes[0]["gpu_control_path"],
    }


def freeze(collection, owners, jobs):
    files = Files()
    context = validate_collection(read(files, collection), files=files)
    return {
        "schema": SCHEMA,
        "collection": collection,
        "source": source_pins(),
        "upstream": upstream_records(context, owners),
        "upstream_jobs": additional_upstream_records(context, jobs),
        **binding(context),
        "gpu_uuid": context["collection"]["hardware_stage"]["gpu_uuid"],
        "scope": list((*CELLS, "target_only")),
        "execution_allowed": False,
    }


def validate_plan(path, expected):
    files = Files()
    plan = read(files, {"path": str(path.resolve()), "sha256": expected})
    require(
        set(plan)
        == {
            "schema",
            "collection",
            "source",
            "upstream",
            "upstream_jobs",
            "resource_policy",
            "environment",
            "gpu_control_path",
            "gpu_uuid",
            "scope",
            "execution_allowed",
        }
        and plan["schema"] == SCHEMA
        and plan["execution_allowed"] is False,
        "immutable collection runtime plan differs",
    )
    current = source_pins()
    require(plan["source"] == current, "NEW dispatcher/evaluator source pins differ")
    for locator in plan["source"].values():
        files.check(locator)
    context = validate_collection(read(files, plan["collection"]), files=files)
    require(
        plan["scope"] == list((*CELLS, "target_only"))
        and plan["gpu_uuid"] == context["collection"]["hardware_stage"]["gpu_uuid"]
        and all(plan[k] == v for k, v in binding(context).items()),
        "collection runtime scope/device/operational controls differ",
    )
    owners = {
        name: {
            k: v
            for k, v in value.items()
            if k not in {"process_records", "controller_pgid", "supervisor_pgid"}
        }
        for name, value in plan["upstream"].items()
    }
    require(
        upstream_records(context, owners) == plan["upstream"],
        "original upstream process census changed",
    )
    jobs = {
        name: {
            key: value
            for key, value in job.items()
            if key not in {"process_records", "controller_pgid", "supervisor_pgid"}
        }
        for name, job in plan["upstream_jobs"].items()
    }
    require(
        additional_upstream_records(context, jobs) == plan["upstream_jobs"],
        "additional upstream export/watcher process census changed",
    )
    return plan, context, files


def upstream_processes(plan, files):
    groups, identities = [], []
    for owner in [*plan["upstream"].values(), *plan["upstream_jobs"].values()]:
        identities += [owner["controller"], owner["supervisor"]]
        groups += [owner["controller_pgid"], owner["supervisor_pgid"]]
        for locator in owner["process_records"]:
            record = read(files, locator)
            if "kernel_identities" in record:
                identities.extend(record["kernel_identities"])
            else:
                identity = record.get("kernel_identity")
                require(
                    identity is not None
                    and identity["pid"] == record["pid"]
                    and record["pgid"] == record["pid"],
                    "upstream process kernel identity absent",
                )
                identities.append(identity)
                groups.append(record["pgid"])
    return groups, identities


def require_idle(observer, groups, identities, baseline, plan):
    release = observer.require_released(groups, identities)
    snapshot = observer.snapshot()
    require(
        release.get("owned_process_groups_absent") is True
        and release.get("owned_cuda_pids_absent") is True
        and release.get("other_context_pids") == []
        and snapshot.get("dxg_holders") == [],
        "owned or foreign CUDA/process/context release incomplete",
    )
    require(
        snapshot["gpu_uuid"] == plan["gpu_uuid"]
        and snapshot["compute_capability"] == [12, 0]
        and snapshot["boot_id"] == baseline["boot_id"],
        "current boot/SM120/device differs",
    )
    resource_gate(snapshot, baseline, plan["resource_policy"])
    return {"actual_release": release, "resources": snapshot}


def require_remote_job(record, supervisor_identity):
    """Validate the live supervisor kernel process, not a self-asserted JSON PID."""
    pid = supervisor_identity["pid"]
    cwd = (Path("/proc") / str(pid) / "cwd").resolve()
    argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
    script = (ROOT / "scripts/remote_job.py").resolve()
    require(
        any(
            (Path(token) if Path(token).is_absolute() else cwd / token).resolve() == script
            for token in argv
            if token
        ),
        "live parent is not pinned remote_job supervisor",
    )
    require(
        record.get("supervisor_pgid") == os.getpgid(pid)
        and record.get("supervisor_sid") == os.getsid(pid),
        "remote_job detached session/process group changed",
    )


class Guard:
    def __init__(
        self,
        plan,
        context,
        files,
        observer,
        baseline,
        run_dir,
        lease,
        lease_record,
        *,
        lock=LOCK,
        owner=None,
        supervisor_state=None,
        runtime_plan=None,
    ):
        self.plan, self.context, self.files = plan, context, files
        self.observer, self.baseline, self.run_dir = observer, baseline, run_dir
        self.lease, self.lease_record, self.lock = lease, lease_record, lock
        self.owner = owner or process_identity(os.getppid())
        self.groups, self.identities = upstream_processes(plan, files)
        self.owned = []
        self.supervisor_state = supervisor_state
        self.runtime_plan = runtime_plan
        self.remote_supervisor = None
        if supervisor_state is not None:
            record = json.loads(supervisor_state.read_text())
            self.remote_supervisor = process_identity(record["supervisor_pid"])
            require_remote_job(record, self.remote_supervisor)

    def authorize(self):
        require(not (self.run_dir / "STOP").exists(), "collection STOP requested")
        control = json.loads(Path(self.plan["gpu_control_path"]).read_text())
        require(
            control.get("rtx5080", {}).get("pause_requested") is False,
            "live host pause/control forbids collection",
        )
        require(
            read(self.files, self.lease) == self.lease_record, "collection startup lease changed"
        )
        require(
            identity_active(self.owner) and process_identity(self.owner["pid"]) == self.owner,
            "collection lock/controller owner is not live",
        )
        require_held_owner_lock(self.lock, self.owner)
        if self.supervisor_state is not None:
            supervisor = json.loads(self.supervisor_state.read_text())
            require(
                supervisor.get("status") == "running"
                and supervisor.get("pid") == self.owner["pid"]
                and supervisor.get("supervisor_pid") == self.remote_supervisor["pid"]
                and identity_active(self.remote_supervisor)
                and process_identity(self.remote_supervisor["pid"]) == self.remote_supervisor,
                "live detached remote_job controller supervision changed",
            )
            require_remote_job(supervisor, self.remote_supervisor)
        for locator in self.plan["source"].values():
            self.files.check(locator)
        # Recheck original checkpoint/export/model/control/receipt immutable bytes
        # without reloading six serialized checkpoint tensors every request.
        self.files.check(self.plan["collection"])
        for selected in self.context["collection"]["candidates"].values():
            for locator in selected.values():
                self.files.check(locator)
        for context in self.context["contexts"].values():
            for key in (
                "checkpoint",
                "checkpoint_sidecar",
                "serializer_source",
                "serializer_python",
            ):
                if key in context:
                    self.files.check(context[key])
            for locator in context["lane"]["source"].values():
                self.files.check(locator)
            for locator in context["exported"].values():
                if isinstance(locator, dict) and set(locator) == {"path", "sha256"}:
                    self.files.check(locator)
        source = self.context["source"]
        for locator in (
            source["target"],
            source["protocol"],
            *source["runtime"].values(),
            *source["source"].values(),
        ):
            self.files.check(locator)
        _, models = evaluation_view(self.context)
        for locator in models.values():
            self.files.check(locator)
        for control in self.context["collection"]["controls"].values():
            provenance = read(self.files, control["provenance"])
            for locator in provenance["evidence"]:
                self.files.check(locator)
        for locator in self.plan["source"].values():
            self.files.check(locator)
        # Every hashed artifact validated during the one-time heavy joins retains
        # its original inode/stat/content identity for the entire run.
        for path, digest in list(self.files.identities):
            self.files.check({"path": path, "sha256": digest})
        upstream_processes(self.plan, self.files)

    def verify_censuses(self):
        runs = [
            (
                Path(self.context["collection"]["candidates"][name]["lane_state"]["path"]).parent,
                owner["process_records"],
            )
            for name, owner in self.plan["upstream"].items()
        ]
        runs += [
            (Path(job["run_dir"]), job["process_records"])
            for job in self.plan["upstream_jobs"].values()
        ]
        for run, records in runs:
            paths = {
                str(p.resolve())
                for pattern in ("**/process.json", "**/process-lineage.json")
                for p in run.glob(pattern)
            }
            require(
                paths == {record["path"] for record in records},
                "upstream process census changed during collection",
            )

    def before_cell(self, directory=None):
        self.authorize()
        self.verify_censuses()
        proof = require_idle(self.observer, self.groups, self.identities, self.baseline, self.plan)
        require(
            proof["resources"]["boot_id"] == self.owner["boot_id"],
            "cell continuation owner boot differs",
        )
        now = time.time()
        continuation = {
            "schema": "nine_model_collection_cell_continuation_v1",
            "stage": "native_cell",
            "collection": self.plan["collection"],
            "new_runtime_source": self.plan["source"],
            "runtime_plan": self.runtime_plan,
            "evaluation_source": self.context["collection"]["evaluation_source"],
            "protocol": self.context["source"]["protocol"],
            "new_human_announcement": False,
            "owner": self.owner,
            "startup_lease": self.lease,
            "shared_lock": str(self.lock),
            "issued_unix": now,
            "expires_unix": now + 300,
            **proof,
        }
        self.cell_continuation = continuation
        if directory is not None:
            atomic_json(directory / "cell-continuation.json", continuation)
        return continuation

    def admit_cell_launch(self):
        self.authorize()
        record = self.cell_continuation
        require(
            record["issued_unix"] <= time.time() < record["expires_unix"]
            and record["expires_unix"] <= record["issued_unix"] + 300,
            "native cell continuation stale",
        )

    def register(self, proc):
        identity = process_identity(proc.pid)
        self.groups.append(proc.pid)
        self.identities.append(identity)
        self.owned.append(identity)
        self.discover(proc)

    def discover(self, proc):
        for identity in descendant_identities(proc.pid):
            if identity not in self.identities:
                self.identities.append(identity)
                self.owned.append(identity)

    def after_cell(self, directory):
        cleanup_descendants(self.owned)
        proof = require_idle(self.observer, self.groups, self.identities, self.baseline, self.plan)
        atomic_json(directory / "actual-release.json", proof)
        self.authorize()


def issue_continuation(plan, plan_hash, lease, directory, observer, baseline, groups, identities):
    now = time.time()
    proof = require_idle(observer, groups, identities, baseline, plan)
    owner = process_identity(os.getpid())
    require_held_owner_lock(LOCK, owner)
    result = {
        "schema": "nine_model_collection_continuation_v1",
        "stage": "evaluate",
        "runtime_plan_sha256": plan_hash,
        "collection": plan["collection"],
        "startup_lease": lease,
        "issued_unix": now,
        "expires_unix": now + 300,
        "owner": owner,
        "shared_lock": str(LOCK),
        **proof,
    }
    path = directory / "evaluation-continuation.json"
    require(not path.exists(), "preserve collection continuation")
    atomic_json(path, result)
    return pin(path)


def verify_continuation(plan, plan_hash, lease, locator, files, *, now=None, parent_pid=None):
    record = read(files, locator)
    current = time.time() if now is None else now
    require(
        record.get("schema") == "nine_model_collection_continuation_v1"
        and record.get("stage") == "evaluate"
        and record.get("runtime_plan_sha256") == plan_hash
        and record.get("collection") == plan["collection"]
        and record.get("startup_lease") == lease,
        "typed collection continuation provenance differs",
    )
    require(
        type(record.get("issued_unix")) in (int, float)
        and type(record.get("expires_unix")) in (int, float)
        and record["issued_unix"]
        <= current
        < record["expires_unix"]
        <= record["issued_unix"] + 300,
        "collection stage continuation stale",
    )
    owner = record["owner"]
    require(
        owner["pid"] == (os.getppid() if parent_pid is None else parent_pid)
        and identity_active(owner)
        and process_identity(owner["pid"]) == owner,
        "collection continuation live parent differs",
    )
    require(Path(record["shared_lock"]) == LOCK, "collection shared GPU lock differs")
    require_held_owner_lock(LOCK, owner)
    require(
        record["resources"]["gpu_uuid"] == plan["gpu_uuid"]
        and record["resources"]["compute_capability"] == [12, 0]
        and record["resources"]["boot_id"] == owner["boot_id"]
        and record["resources"].get("dxg_holders") == []
        and record["actual_release"].get("owned_process_groups_absent") is True
        and record["actual_release"].get("owned_cuda_pids_absent") is True
        and record["actual_release"].get("other_context_pids") == [],
        "continuation boot/device/upstream release differs",
    )
    return record


def evaluate_stage(args, plan, context, files):
    import evaluate_nine_model_native as evaluator

    lease = pin(args.availability)
    record = verify_continuation(plan, args.plan_sha256, lease, pin(args.continuation), files)
    # Fresh stage continuation carries standing authority; the original human
    # startup lease can expire during a bounded evaluation, as in endpoint flow.
    startup = read(files, lease)
    require(
        startup.get("schema") == "nine_model_gpu_lease_v1"
        and startup.get("host") == "rtx5080"
        and startup.get("user_announced_available") is True
        and startup["bundle_sha256"] == args.plan_sha256
        and startup["gpu_uuid"] == plan["gpu_uuid"]
        and startup["sole_owner"] is True
        and startup["pause_requested"] is False,
        "collection inherited startup lease differs",
    )
    parent = json.loads(args.supervisor_state.read_text())
    require(
        parent.get("status") == "running" and parent.get("pid") == os.getppid(),
        "collection stage must belong to supervised live controller",
    )
    observer = LinuxResources(plan["gpu_uuid"])
    guard = Guard(
        plan,
        context,
        files,
        observer,
        record["resources"],
        args.run_dir,
        lease,
        startup,
        owner=record["owner"],
        supervisor_state=args.supervisor_state,
        runtime_plan=pin(args.plan),
    )
    guard.before_cell()  # Before opening any authorized opaque prompt bytes.
    view, models = evaluation_view(context)
    view.update(gpu_uuid=plan["gpu_uuid"], environment=plan["environment"])
    # Marker contracts come from the authenticated full-nine evaluation source;
    # model/training ancestry still comes exclusively from independent endpoints.
    view["candidates"] = context["source"]["origin"]["candidates"]
    require(
        all(view["candidates"][name].get("native_markers") for name in CANDIDATES),
        "authenticated full-nine native marker declarations absent",
    )
    view["inputs"].update(
        protocol=context["source"]["protocol"], prompts=context["source"]["prompts"]
    )
    inputs = {
        "target": context["source"]["target"],
        "export_endpoints": {
            name: value["export_receipt"]
            for name, value in context["collection"]["candidates"].items()
        },
    }
    coverage = {
        name: {
            "selected_projection_inventory": sorted(
                read(files, context["exports"][name]["audit"])["projections"]
            ),
            "original_serializer_audit": context["exports"][name]["audit"],
        }
        for name in CANDIDATES
    }
    coverage.update(
        {
            family + "_q4": control["deployment_coverage"]
            for family, control in context["collection"]["controls"].items()
        }
    )
    ancestry = {
        "collection": plan["collection"],
        "runtime_plan": pin(args.plan),
        "original_candidates": context["collection"]["candidates"],
        "original_controls": context["collection"]["controls"],
        "evaluation_source": context["collection"]["evaluation_source"],
        "new_runtime_source": plan["source"],
    }
    evaluator.evaluate(
        args,
        view,
        files,
        inputs,
        context["protocol"],
        coverage,
        models,
        observer,
        guard=guard,
        ancestry=ancestry,
    )


def run(args):
    outer_started = time.monotonic()
    plan, context, files = validate_plan(args.plan, args.plan_sha256)
    if not args.start:
        return {
            "status": "VALIDATED_SOURCE_RUNTIME_PENDING",
            "execution_allowed": False,
            "gpu_queried": False,
            "collection": plan["collection"],
        }
    require(
        args.run_dir and args.availability and args.supervisor_state,
        "collection detached controller/lease/run directory required",
    )
    if args.stage:
        require(
            args.continuation
            and args.continuation_sha256
            and sha256(args.continuation) == args.continuation_sha256,
            "collection fresh stage continuation required",
        )
        return evaluate_stage(args, plan, context, files)
    lease = require_available(args.availability, args.plan_sha256)
    require(lease["gpu_uuid"] == plan["gpu_uuid"], "collection lease physical GPU differs")
    supervisor = json.loads(args.supervisor_state.read_text())
    require(
        supervisor.get("status") == "running"
        and supervisor.get("pid") == os.getpid()
        and supervisor.get("supervisor_pid") == os.getppid()
        and supervisor.get("stop_grace_seconds", 0) >= 90
        and os.environ.get("TMUX")
        and identity_active(process_identity(os.getppid())),
        "detached remote_job collection supervision required",
    )
    require_remote_job(supervisor, process_identity(os.getppid()))
    require(
        json.loads(Path(plan["gpu_control_path"]).read_text())
        .get("rtx5080", {})
        .get("pause_requested")
        is False,
        "live host pause/control forbids collection",
    )
    require(not (args.run_dir / "STOP").exists(), "collection STOP requested")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock, stop_signals():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = process_identity(os.getpid())
        lock.seek(0)
        lock.truncate()
        json.dump(owner, lock)
        lock.flush()
        os.fsync(lock.fileno())
        observer = LinuxResources(plan["gpu_uuid"])
        baseline = observer.snapshot()
        require(baseline["boot_id"] == owner["boot_id"], "collection controller boot differs")
        guard = Guard(
            plan,
            context,
            files,
            observer,
            baseline,
            args.run_dir,
            pin(args.availability),
            lease,
            owner=owner,
            supervisor_state=args.supervisor_state,
            runtime_plan=pin(args.plan),
        )
        guard.before_cell()
        runner = SubprocessRunner(ROOT, plan["environment"], authorization=guard.authorize)
        continuation = issue_continuation(
            plan,
            args.plan_sha256,
            pin(args.availability),
            args.run_dir,
            observer,
            baseline,
            guard.groups,
            guard.identities,
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(args.plan),
            "--plan-sha256",
            args.plan_sha256,
            "--start",
            "--stage",
            "evaluate",
            "--run-dir",
            str(args.run_dir),
            "--availability",
            str(args.availability),
            "--supervisor-state",
            str(args.supervisor_state),
            "--continuation",
            continuation["path"],
            "--continuation-sha256",
            continuation["sha256"],
            "--completion-output",
            str(args.run_dir / "evaluation-receipt.json"),
        ]
        try:
            remaining = context["protocol"]["evaluation_wall_seconds"] - (
                time.monotonic() - outer_started
            )
            require(remaining > 0, "collection outer allowance exhausted by preflight")
            runner.run(
                command,
                directory=args.run_dir / "attempt",
                stop_path=args.run_dir / "STOP",
                wall_seconds=remaining,
            )
        finally:
            proof = require_idle(
                observer,
                guard.groups + runner.process_groups,
                guard.identities + runner.process_identities,
                baseline,
                plan,
            )
            atomic_json(args.run_dir / "controller-release.json", proof)
        receipt = json.loads((args.run_dir / "evaluation-receipt.json").read_text())
        require(
            receipt.get("schema") == "nine_model_collection_evaluation_receipt_v1"
            and receipt.get("status") == "PASS"
            and receipt.get("artifact_kind") == "production"
            and receipt.get("native") is True
            and receipt.get("cells") == list(CELLS)
            and receipt.get("target_only_diagnostic") is True
            and receipt.get("collection_ancestry", {}).get("collection") == plan["collection"],
            "collection evaluation completion ancestry differs",
        )
        files.check(receipt["report"])
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "run-dir",
        "availability",
        "supervisor-state",
        "continuation",
        "completion-output",
        "freeze-output",
        "collection",
        "upstream-owners",
        "upstream-jobs",
    ):
        parser.add_argument("--" + name, type=Path)
    for name in ("plan-sha256", "continuation-sha256", "collection-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stage", choices=["evaluate"])
    args = parser.parse_args()
    if args.freeze_output:
        require(
            not args.start
            and args.collection
            and args.collection_sha256
            and args.upstream_owners
            and args.upstream_jobs,
            "freeze requires original collection and upstream owner locators",
        )
        require(not args.freeze_output.exists(), "preserve immutable runtime plan")
        result = freeze(
            {"path": str(args.collection.resolve()), "sha256": args.collection_sha256},
            json.loads(args.upstream_owners.read_text()),
            json.loads(args.upstream_jobs.read_text()),
        )
        atomic_json(args.freeze_output, result)
    else:
        require(args.plan and args.plan_sha256, "frozen runtime plan required")
        result = run(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
