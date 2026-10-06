"""Crash-safe cumulative trainer time; unresolved attempts never refund work."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

TIMED_TRAINING_TIME_POLICY = {
    "charged": "active optimizer loop including periodic checkpoint overhead",
    "excluded": ["startup", "reconstruction", "boundary publication", "export", "evaluation"],
    "overshoot": "actual completed-update seconds retained; allocation threshold unchanged",
}


def validate_evaluation_milestones_seconds(values, maximum):
    """Validate cumulative thresholds independently of optimizer step counts."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("nonempty timed evaluation milestones required")
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("timed evaluation requires a finite trainer allocation")
    previous = 0
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= previous:
            raise ValueError("timed evaluation milestones must be finite and strictly increasing")
        previous = value
    if values[-1] != maximum:
        raise ValueError("last timed evaluation milestone must equal trainer allocation")
    return tuple(values)


def checked_locator(record):
    """Authenticate a caller-selected local artifact without opening any models."""
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("exact path/SHA256 locator required")
    path = Path(record["path"])
    if not path.is_file() or path.is_symlink():
        raise ValueError("timed evaluation artifact absent or symlinked")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("timed evaluation artifact SHA256 differs")
    return path


def artifact_locator(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


class TimedEvaluation:
    """Durable per-lane evaluation gate; the controller owns GPU lifecycle.

    A successful trainer exit publishes a checkpoint-bound request. Only an
    explicitly supplied, hash-authenticated result can unlock another update.
    Evaluation and reconstruction occur while the training ledger is inactive.
    """

    def __init__(
        self,
        directory,
        milestones,
        maximum,
        *,
        candidate,
        bundle_sha256,
        config_sha256,
        protocol,
        atomic_write,
    ):
        self.directory = Path(directory)
        self.path = self.directory / "timed-evaluation-state.json"
        self.write = atomic_write
        self.resume_checkpoint = None
        self.milestones = validate_evaluation_milestones_seconds(milestones, maximum)
        checked_locator(protocol)
        self.contract = {
            "candidate": candidate,
            "bundle_sha256": bundle_sha256,
            "config_sha256": config_sha256,
            "protocol": protocol,
            "milestones_seconds": list(self.milestones),
            "max_seconds": maximum,
        }
        self.state = {
            "schema": "nine_model_timed_evaluation_state_v1",
            "contract": self.contract,
            "completed": [],
            "pending": None,
        }
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if (
                self.state.get("schema") != "nine_model_timed_evaluation_state_v1"
                or self.state.get("contract") != self.contract
                or not isinstance(self.state.get("completed"), list)
                or len(self.state["completed"]) > len(self.milestones)
            ):
                raise ValueError("timed evaluation resume contract differs")
            for index, entry in enumerate(self.state["completed"]):
                request = self._request(entry["request"])
                if request["milestone_index"] != index:
                    raise ValueError("timed evaluation completed order differs")
                self._result(entry["receipt"], entry["request"], request)
            if self.state.get("pending") is not None:
                request = self._request(self.state["pending"])
                if request["milestone_index"] != len(self.state["completed"]):
                    raise ValueError("timed evaluation pending order differs")

    def _request(self, locator):
        request = json.loads(checked_locator(locator).read_text())
        index = request.get("milestone_index")
        if (
            request.get("schema") != "nine_model_timed_evaluation_request_v1"
            or type(index) is not int
            or not 0 <= index < len(self.milestones)
            or request.get("milestone_seconds") != self.milestones[index]
            or request.get("allocation_seconds") != self.contract["max_seconds"]
            or request.get("training_time_policy") != TIMED_TRAINING_TIME_POLICY
            or any(
                request.get(key) != self.contract[key]
                for key in ("candidate", "bundle_sha256", "config_sha256", "protocol")
            )
            or type(request.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(request["elapsed_seconds"])
            or request["elapsed_seconds"] < self.milestones[index]
            or request.get("final_training_complete") != (index == len(self.milestones) - 1)
            or type(request.get("counters", {}).get("step")) is not int
            or request["counters"]["step"] <= 0
        ):
            raise ValueError("timed evaluation request contract differs")
        checked_locator(request["checkpoint"])
        ledger = json.loads(checked_locator(request["budget_ledger"]).read_text())
        if (
            ledger.get("schema") != "continuous_training_budget_v1"
            or ledger.get("max_seconds") != self.contract["max_seconds"]
            or ledger.get("active_attempt") is not None
            or ledger.get("training_seconds") != request["elapsed_seconds"]
        ):
            raise ValueError("timed evaluation requires exact settled trainer accounting")
        return request

    def _result(self, locator, request_locator, request):
        result = json.loads(checked_locator(locator).read_text())
        if (
            result.get("schema") != "nine_model_timed_evaluation_receipt_v1"
            or result.get("status") != "PASS"
            or result.get("completed") is not True
            or result.get("request") != request_locator
            or any(
                result.get(key) != request[key]
                for key in ("candidate", "bundle_sha256", "config_sha256", "checkpoint", "protocol")
            )
            or result.get("owned_release", {}).get("owned_process_groups_absent") is not True
            or result.get("owned_release", {}).get("owned_cuda_pids_absent") is not True
        ):
            raise ValueError("timed evaluation receipt/checkpoint/protocol/release differs")
        checked_locator(result["evaluation"])
        return result

    def authorize_resume(self, receipt=None):
        pending = self.state["pending"]
        if pending is None:
            if receipt is not None:
                # Publication of authorization may precede a controller crash.
                if not self.state["completed"] or receipt != self.state["completed"][-1]["receipt"]:
                    raise ValueError("unexpected or replayed timed evaluation receipt")
                checked_locator(receipt)
            return
        if receipt is None:
            raise ValueError("pending timed evaluation requires explicit authenticated receipt")
        request = self._request(pending)
        self._result(receipt, pending, request)
        updated = {
            **self.state,
            "completed": [*self.state["completed"], {"request": pending, "receipt": receipt}],
            "pending": None,
        }
        self.write(self.path, updated)
        self.state = updated
        self.resume_checkpoint = request["checkpoint"]

    def require_checkpoint(self, checkpoint, counters=None):
        if self.resume_checkpoint is not None and checkpoint != self.resume_checkpoint:
            raise ValueError("evaluation receipt does not authorize restored checkpoint")
        if self.state["completed"]:
            request = self._request(self.state["completed"][-1]["request"])
            floor = request["counters"]
            if (
                not isinstance(counters, dict)
                or type(counters.get("step")) is not int
                or counters["step"] < floor["step"]
            ):
                raise ValueError("restored checkpoint rolls back evaluated trainer progress")
            if counters["step"] == floor["step"]:
                for key in ("epoch", "cursor", "block_index"):
                    if (
                        key in floor
                        and type(floor[key]) is int
                        and (type(counters.get(key)) is not int or counters[key] < floor[key])
                    ):
                        raise ValueError("restored cursor rolls back evaluated trainer progress")

    def due(self, elapsed):
        if self.state["pending"] is not None:
            raise ValueError("pending timed evaluation forbids optimizer updates")
        index = len(self.state["completed"])
        return index < len(self.milestones) and elapsed >= self.milestones[index]

    def publish(self, checkpoint, elapsed, ledger, counters):
        if not self.due(elapsed):
            raise ValueError("timed evaluation milestone is not due")
        if (
            not isinstance(counters, dict)
            or type(counters.get("step")) is not int
            or counters["step"] <= 0
        ):
            raise ValueError("timed evaluation requires positive committed optimizer progress")
        index = len(self.state["completed"])
        checked_locator(checkpoint)
        checked_locator(ledger)
        request = {
            "schema": "nine_model_timed_evaluation_request_v1",
            **{
                key: self.contract[key]
                for key in ("candidate", "bundle_sha256", "config_sha256", "protocol")
            },
            "milestone_index": index,
            "milestone_seconds": self.milestones[index],
            "elapsed_seconds": elapsed,
            "checkpoint": checkpoint,
            "counters": counters,
            "allocation_seconds": self.contract["max_seconds"],
            "training_time_policy": TIMED_TRAINING_TIME_POLICY,
            "budget_ledger": ledger,
            "final_training_complete": index == len(self.milestones) - 1,
        }
        directory = self.directory / "timed-evaluation" / f"milestone-{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        if request_path.exists():
            # Recover the completed request rename before the state publication.
            existing = artifact_locator(request_path)
            original = self._request(existing)
            if original["checkpoint"] != checkpoint or original["milestone_index"] != index:
                raise ValueError("timed evaluation request publication differs; recovery required")
            self.state["pending"] = existing
            self.write(self.path, self.state)
            return existing
        # The ledger is mutable across attempts. Preserve its settled bytes as
        # evidence rather than pinning a file the next invocation will rewrite.
        ledger_path = directory / f"budget-used-{time.time_ns()}.json"
        self.write(ledger_path, json.loads(checked_locator(ledger).read_text()))
        request["budget_ledger"] = artifact_locator(ledger_path)
        self.write(request_path, request)
        self.state["pending"] = artifact_locator(request_path)
        self.write(self.path, self.state)
        return self.state["pending"]


def boot_identity() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.is_file():
        return path.read_text().strip()
    # CPU fixtures have no Linux boot identity. Cross-process recovery falls
    # back to charging the full reservation, as on a changed remote boot.
    return f"{platform.node()}:{os.getpid()}"


def process_birth(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class TrainingBudget:
    """Write intent before work, then settle actual monotonic elapsed time.

    A killed attempt is charged through recovery time on the same Linux boot,
    including any uncertain downtime. After reboot its whole remaining reserved
    budget is consumed. Normal exits exclude standalone evaluation/startup.
    The run/GPU locks are still required; PID birth checks catch unsafe overlap.
    """

    def __init__(self, path: Path, source: str, maximum: float | None, atomic_write):
        self.path, self.source, self.maximum = Path(path), source, maximum
        self.write = atomic_write
        self.used = 0.0
        self.active = None

    def load(self, checkpoint_elapsed: float) -> float:
        self.used = checkpoint_elapsed
        if not self.path.exists():
            return self.used
        value = json.loads(self.path.read_text())
        if (
            value.get("schema") != "continuous_training_budget_v1"
            or value.get("source_sha256") != self.source
            or value.get("max_seconds") != self.maximum
        ):
            raise ValueError("persisted training budget contract differs")
        used = value.get("training_seconds")
        if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
            raise ValueError("persisted training budget is invalid")
        self.used = used
        active = value.get("active_attempt")
        if active is not None:
            if not isinstance(active, dict):
                raise ValueError("persisted training attempt is invalid")
            start, reserved = active.get("started_monotonic"), active.get("reserved_seconds")
            if (
                type(start) not in (int, float)
                or not math.isfinite(start)
                or start < 0
                or (
                    reserved is not None
                    and (
                        type(reserved) not in (int, float)
                        or not math.isfinite(reserved)
                        or reserved < 0
                    )
                )
            ):
                raise ValueError("persisted training attempt timing is invalid")
            same_boot = active.get("boot_id") == boot_identity()
            if same_boot and active.get("process_birth") is not None:
                if process_birth(active["pid"]) == active["process_birth"]:
                    raise ValueError("previous trainer budget owner is still alive")
            if same_boot:
                charged = max(0.0, time.monotonic() - start)
                if reserved is not None:
                    charged = min(charged, reserved)
            elif reserved is not None:
                charged = reserved
            else:
                raise ValueError("unbounded crashed training requires recoverable boot clock")
            self.used = max(checkpoint_elapsed, self.used + charged)
            self._persist(
                recovered_attempt={
                    **active,
                    "charged_seconds": charged,
                    "recovery_unix": time.time(),
                }
            )
        self.used = max(checkpoint_elapsed, self.used)
        return self.used

    def _persist(self, **extra):
        self.write(
            self.path,
            {
                "schema": "continuous_training_budget_v1",
                "source_sha256": self.source,
                "max_seconds": self.maximum,
                "training_seconds": self.used,
                "active_attempt": self.active,
                **extra,
            },
        )

    def begin(self, checkpoint_elapsed: float):
        self.load(checkpoint_elapsed)
        self.active = {
            "boot_id": boot_identity(),
            "pid": os.getpid(),
            "process_birth": process_birth(os.getpid()),
            "started_monotonic": time.monotonic(),
            "started_unix": time.time(),
            "reserved_seconds": None
            if self.maximum is None
            else max(0.0, self.maximum - self.used),
        }
        self._persist()

    def elapsed(self) -> float:
        if self.active is None:
            return self.used
        elapsed = max(0.0, time.monotonic() - self.active["started_monotonic"])
        # A whole optimizer transaction can cross the cap. Preserve measured
        # paid work rather than refunding that overage on a normal exit. The
        # reservation bounds only conservative recovery of unresolved attempts.
        return self.used + elapsed

    def finish(self) -> float:
        self.used = self.elapsed()
        self.active = None
        self._persist()
        return self.used
