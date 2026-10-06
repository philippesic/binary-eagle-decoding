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
            or not validate_evaluation_continuation(result)
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


def _research_require(condition, message):
    if not condition:
        raise ValueError(message)


def _research_json(locator):
    return json.loads(checked_locator(locator).read_text())


def _research_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact_identity(value):
    """Allow authenticated relocated aliases of identical bytes, never new models."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            checked_locator(value)
            return {"sha256": value["sha256"]}
        return {key: _artifact_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_artifact_identity(item) for item in value]
    return value


def research_bindings(plan):
    return {
        "candidate": plan["candidate"],
        "target": plan["target"],
        "target_policy": plan["target_policy"],
        "runtime": plan["runtime"],
        "protocol": plan["protocol"],
        "prompts": plan["prompts"],
        "initial_model": plan["initial"]["model"],
        "q4_model": plan["control"]["model"],
        "gpu_uuid": plan["gpu_uuid"],
    }


def checked_measurement_rows(bindings, records, diagnostics):
    """Require complete actual raw token/count/finish joins and paired coverage."""
    protocol = _research_json(bindings["protocol"])
    prompts = [
        json.loads(line)
        for line in checked_locator(bindings["prompts"]).read_text().splitlines()
        if line.strip()
    ]
    identifiers = {row["id"] for row in prompts}
    _research_require(
        len(identifiers) == len(prompts) == 24 and protocol["repetitions"] == 5,
        "research policy requires actual complete 24-prompt five-repetition suite",
    )
    cells = {bindings["candidate"], "initial", "eagle_q4", "target_only"}
    result = []
    for rows, expected_reps, kind in (
        (records, set(range(5)), "clean"),
        (diagnostics, {0}, "diagnostic"),
    ):
        expected = {
            (cell, rep, prompt) for cell in cells for rep in expected_reps for prompt in identifiers
        }
        actual = {(r["cell"], r["repetition"], r["prompt_id"]) for r in rows}
        _research_require(
            actual == expected and len(rows) == len(expected),
            "incomplete/duplicate research measurement coverage",
        )
        for row in rows:
            raw = _research_json(row["raw_result"])
            tokens = row["generated_token_ids"]
            _research_require(
                isinstance(tokens, list)
                and tokens
                and all(type(t) is int and t >= 0 for t in tokens)
                and tokens == raw.get("generated_token_ids")
                and type(row["output_tokens"]) is int
                and row["output_tokens"] > 0
                and row["output_tokens"] == raw.get("completion_tokens")
                and type(row["latency_s"]) in (int, float)
                and math.isfinite(row["latency_s"])
                and row["latency_s"] > 0
                and row["latency_s"] == raw.get("request_wall_s")
                and raw.get("finish_reason") in {"stop", "length"},
                "raw token/count/finish/timing evidence missing or changed",
            )
            if row["cell"] != "target_only":
                counters = row.get("speculative")
                _research_require(
                    counters == raw.get("speculative")
                    and isinstance(counters, dict)
                    and all(
                        type(counters.get(k)) is int and counters[k] >= 0
                        for k in ("proposed", "accepted", "rounds")
                    )
                    and counters["proposed"] > 0
                    and counters["rounds"] > 0
                    and counters["accepted"] <= counters["proposed"],
                    "raw native speculative counters missing or changed",
                )
                if kind == "diagnostic":
                    _research_require(
                        row.get("round_summary", {}).get("rounds", 0) > 0,
                        "native diagnostic rounds missing",
                    )
            result.append({**row, "measurement_kind": kind, "finish_reason": raw["finish_reason"]})
    return result


def _full_output(row):
    return {
        "token_ids": row["generated_token_ids"],
        "finish_reason": row["finish_reason"],
        "output_tokens": row["output_tokens"],
    }


def _first_divergence(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right))
    )


def baseline_cases(bindings, records, diagnostics):
    rows = checked_measurement_rows(bindings, records, diagnostics)
    table = {}
    for row in rows:
        key = (row["cell"], row["prompt_id"])
        output = _full_output(row)
        _research_require(
            table.setdefault(key, output) == output,
            "baseline control/candidate output varies across repetitions/diagnostics",
        )
    cases = {}
    for prompt in {r["prompt_id"] for r in rows}:
        target, q4 = table[("target_only", prompt)], table[("eagle_q4", prompt)]
        _research_require(
            table[(bindings["candidate"], prompt)] == table[("initial", prompt)],
            "zero candidate and calibrated initial outputs differ",
        )
        if q4 != target:
            _research_require(
                q4["token_ids"] != target["token_ids"],
                "finish/count-only failure is not a known numeric token branch",
            )
            _research_require(
                table[("initial", prompt)] in (target, q4),
                "zero initial differs from both authoritative baseline branches",
            )
            cases[prompt] = {
                "target": target,
                "q4": q4,
                "first_divergence_index": _first_divergence(target["token_ids"], q4["token_ids"]),
            }
        else:
            _research_require(
                table[("initial", prompt)] == target,
                "new zero-initial correctness failure outside shared Q4 case",
            )
    _research_require(cases, "known shared target/Q4 failure absent; use strict path")
    return cases


def baseline_control_outputs(bindings, records, diagnostics):
    rows = checked_measurement_rows(bindings, records, diagnostics)
    result = {}
    for row in rows:
        if row["cell"] in {"target_only", "eagle_q4"}:
            pair = result.setdefault(row["prompt_id"], {})
            output = _full_output(row)
            _research_require(
                pair.setdefault(row["cell"], output) == output,
                "baseline controls vary across repetitions/diagnostics",
            )
    return result


def validate_collection_evidence(bindings, evidence_locator, progress_locator):
    """Preserve actual failed controller status separately from complete collection."""
    evidence = _research_json(evidence_locator)
    _research_require(
        evidence.get("schema") == "native_zero_measurement_evidence_v1"
        and evidence.get("progress") == progress_locator
        and _artifact_identity(evidence.get("bindings")) == _artifact_identity(bindings)
        and evidence.get("hardware", {}).get("compute_capability") == [12, 0]
        and evidence["hardware"].get("gpu_uuid") == bindings["gpu_uuid"],
        "actual zero source/hardware evidence differs",
    )
    failed = _research_json(evidence["failed_status"])
    _research_require(
        failed.get("status") in {"failed", "FAILED"}, "original strict failed status required"
    )
    zero = _research_json(evidence["zero_preparation"])
    _research_require(
        zero.get("schema") == "nine_model_preparation_v1"
        and zero.get("status") == "PASS"
        and zero.get("optimizer_updates") == 0
        and zero.get("checkpoint", {}).get("cursor", {}).get("step") == 0,
        "actual zero-update preparation receipt required",
    )
    models = evidence["model_ancestry"]
    _research_require(
        models[bindings["candidate"]]["sha256"]
        == models["initial"]["sha256"]
        == bindings["initial_model"]["sha256"]
        and models["eagle_q4"]["sha256"] == bindings["q4_model"]["sha256"],
        "actual zero/control model bytes differ",
    )
    for model in models.values():
        checked_locator(model)
    resources = evidence.get("resource_returns", [])
    dispatch = evidence.get("dispatch_proofs", [])
    _research_require(
        len(resources) == 24
        and len(dispatch) == 2
        and len({_research_digest(r) for r in resources}) == 24
        and len({_research_digest(r) for r in dispatch}) == 2,
        "complete per-cell/repetition resource and both model dispatch proofs required",
    )
    for locator in resources:
        result = _research_json(locator)
        release = result.get("release", result.get("actual_release", {}))
        snapshot = result.get("resources", {})
        _research_require(
            release.get("owned_process_groups_absent") is True
            and release.get("owned_cuda_pids_absent") is True
            and release.get("other_context_pids") == []
            and snapshot.get("dxg_holders") == []
            and snapshot.get("gpu_uuid") == bindings["gpu_uuid"],
            "actual collection resource release failed",
        )
    for locator in dispatch:
        result = _research_json(locator)
        _research_require(
            result.get("schema") == "nine_model_observed_w1_dispatch_v1"
            and result.get("status") == "PASS"
            and result.get("activation_bits") == 8
            and len(set(result.get("packed_names", []))) == 16,
            "actual sixteen-projection A8 native dispatch proof missing",
        )
    return evidence


def validate_research_policy(locator, bindings):
    policy = _research_json(locator)
    _research_require(
        policy.get("schema") == "native_research_continuation_policy_v1"
        and policy.get("scope") == "research_only_no_deployment_admission"
        and policy.get("strict_quality_status") == "FAILED"
        and _artifact_identity(policy.get("bindings")) == _artifact_identity(bindings),
        "research policy source/model/runtime/protocol differs",
    )
    _research_require(
        bindings["target_policy"] == {"immutable": True, "weights": "f16", "kv": "f16"},
        "immutable F16 target/KV required",
    )
    for name, record in policy["controller_sources"].items():
        _research_require(
            name
            in {
                "scripts/run_nine_model_lane_endpoint.py",
                "scripts/evaluate_nine_model_timed_checkpoint.py",
                "scripts/run_nine_model_lane.py",
                "src/w1a1_eagle/continuous_budget.py",
            },
            "unknown research consumer source",
        )
        _research_require(
            artifact_locator(Path(__file__).resolve().parents[2] / name)["sha256"]
            == record["sha256"],
            "research consumer deployed source changed",
        )
        checked_locator(record)
    _research_require(
        len(policy["controller_sources"]) == 4, "complete active consumer source inventory required"
    )
    review = _research_json(policy["review"])
    body = {key: value for key, value in policy.items() if key != "review"}
    _research_require(
        review.get("schema") == "native_research_operational_review_v1"
        and review.get("status") == "REVIEWED"
        and review.get("policy_body_sha256") == _research_digest(body)
        and review.get("new_scientific_approval_claimed") is False,
        "explicit source-bound operational review required",
    )
    checked_locator(review["approved_plan"])
    checked_locator(review["standing_start_authorization"])
    historical = _research_json(review["historical_failure"])
    baseline = _research_json(policy["baseline_report"])
    _research_require(
        baseline.get("schema") == "native_zero_research_measurement_v1"
        and baseline.get("measurement_status") == "MEASUREMENT_COMPLETE"
        and baseline.get("strict_quality_status") == "FAILED"
        and _artifact_identity(baseline["bindings"]) == _artifact_identity(bindings),
        "genuine zero baseline differs",
    )
    progress = _research_json(baseline["progress"])
    validate_collection_evidence(bindings, baseline["collection_evidence"], baseline["progress"])
    cases = baseline_cases(bindings, progress["clean_records"], progress["diagnostic_records"])
    _research_require(
        baseline.get("control_outputs")
        == baseline_control_outputs(
            bindings, progress["clean_records"], progress["diagnostic_records"]
        ),
        "baseline complete control outputs changed",
    )
    _research_require(
        cases == baseline["allowed_cases"] == policy["allowed_cases"]
        and historical.get("allowed_cases") == cases
        and historical.get("strict_quality_status") == "FAILED",
        "reviewed original shared target/Q4 failure cases/full IDs differ",
    )
    return policy


def research_continuation_decision(locator, bindings, records, diagnostics):
    policy = validate_research_policy(locator, bindings)
    cases = policy["allowed_cases"]
    baseline = _research_json(policy["baseline_report"])
    controls = baseline["control_outputs"]
    rows = checked_measurement_rows(bindings, records, diagnostics)
    table = {(r["measurement_kind"], r["cell"], r["repetition"], r["prompt_id"]): r for r in rows}
    mismatches = []
    for row in rows:
        prompt = row["prompt_id"]
        target = table[(row["measurement_kind"], "target_only", row["repetition"], prompt)]
        output, target_output = _full_output(row), _full_output(target)
        if row["cell"] in {"target_only", "eagle_q4"}:
            _research_require(
                output == controls[prompt][row["cell"]], "baseline complete control output changed"
            )
        if output != target_output:
            mismatches.append(
                {
                    "prompt_id": prompt,
                    "cell": row["cell"],
                    "repetition": row["repetition"],
                    "measurement_kind": row["measurement_kind"],
                }
            )
        if prompt in cases:
            case = cases[prompt]
            if row["cell"] in {"target_only", "eagle_q4"}:
                _research_require(
                    output == case["target" if row["cell"] == "target_only" else "q4"],
                    "known control branch changed",
                )
            else:
                _research_require(
                    output in (case["target"], case["q4"]),
                    "candidate/initial output differs from both reviewed complete branches",
                )
        else:
            _research_require(
                output == target_output, "new strict correctness failure outside reviewed prompt"
            )
    return {
        "measurement_status": "MEASUREMENT_COMPLETE",
        "strict_quality_status": "FAILED" if mismatches else "PASS",
        "continuation_status": "RESEARCH_CONTINUATION_ALLOWED",
        "research_only": True,
        "research_continuation_policy": locator,
        "research_bindings": bindings,
        "strict_mismatches": mismatches,
    }


def validate_evaluation_continuation(result):
    """Actively reauthenticate a research receipt before any exact-state restore."""
    if result.get("status") == "PASS" and result.get("research_continuation_policy") is None:
        _research_require(
            result.get("strict_quality_status", "PASS") == "PASS",
            "strict FAILED cannot be renamed PASS",
        )
        return True
    _research_require(
        result.get("status") == "FAILED"
        and result.get("completed") is True
        and result.get("measurement_status") == "MEASUREMENT_COMPLETE"
        and result.get("strict_quality_status") == "FAILED"
        and result.get("continuation_status") == "RESEARCH_CONTINUATION_ALLOWED",
        "strict failed evaluation cannot resume",
    )
    report = _research_json(result["evaluation"])
    plan = _research_json(result["evaluation_plan"])
    _research_require(
        plan.get("research_continuation_policy") == result["research_continuation_policy"]
        and _artifact_identity(research_bindings(plan))
        == _artifact_identity(result["research_bindings"])
        and result["candidate"] == plan["candidate"]
        and result["protocol"] == plan["protocol"],
        "exact evaluation plan/request research policy bindings differ",
    )
    _research_require(
        report["research_continuation_policy"] == result["research_continuation_policy"]
        and report["research_bindings"] == result["research_bindings"],
        "research report/receipt policy joins differ",
    )
    raw = _research_json(report["measurements"])
    decision = research_continuation_decision(
        result["research_continuation_policy"],
        result["research_bindings"],
        raw["clean_records"],
        raw["diagnostic_records"],
    )
    _research_require(
        all(
            result.get(k) == decision[k]
            for k in ("measurement_status", "strict_quality_status", "continuation_status")
        ),
        "research continuation decision changed",
    )
    return True
