"""CPU planning boundary for bounded native student-trajectory refresh.

No model loading or teacher inference happens here. Hashes establish immutable
ancestry; the capture auditor/provider remains responsible for numerical and
training eligibility. Teacher features have an execution-contract scope because
native same-prefix features can vary with batching/backend policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

PLAN_SCHEMA = "w1ax_trajectory_refresh_plan_v1"
INDEX_SCHEMA = "w1ax_refresh_teacher_index_v1"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_hash(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def verify_file(record: dict) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("file record requires path and sha256")
    path = Path(record["path"])
    if not path.is_absolute() or sha256(path) != checked_hash(record["sha256"]):
        raise ValueError("file path or SHA256 mismatch")
    return path


def file_record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def prefix(value: object, vocab: int, cap: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= cap
        or any(type(token) is not int or not 0 <= token < vocab for token in value)
    ):
        raise ValueError("prefix must contain bounded absolute target token IDs")
    return tuple(value)


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("capacity must be a positive integer")
    return value


def _fraction(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gate threshold must be a finite fraction")
    return float(value)


def validate_policy(policy: dict) -> tuple[set[str], dict]:
    if policy.get("schema") != "w1ax_refresh_policy_v1" or policy.get("split") != "train":
        raise ValueError("refresh policy must use the training split; final is sealed")
    sample = policy.get("train_sample")
    if not isinstance(sample, list) or not sample:
        raise ValueError("freeze a nonempty training sample before rollout")
    ids = [row.get("prompt_id") for row in sample]
    if (
        len(set(ids)) != len(ids)
        or any(not isinstance(item, str) or not item for item in ids)
        or any(row.get("split") != "train" for row in sample)
        or {row.get("domain") for row in sample} != {"prose", "code", "reasoning"}
    ):
        raise ValueError("sample must have unique train prompts in all three domains")
    checked_hash(policy.get("train_prompts_sha256"))
    checked_hash(policy.get("development_sample_sha256"))
    caps = policy.get("caps", {})
    for name in (
        "max_rounds",
        "max_student_rows",
        "max_new_label_rows",
        "max_new_feature_rows",
        "max_prefix_tokens",
    ):
        _positive(caps.get(name))
    thresholds = policy.get("learning_curve", {})
    _positive(thresholds.get("min_completed_steps"))
    for name in (
        "min_relative_ce_improvement",
        "max_acceptance_regression",
        "min_changed_prefix_fraction",
    ):
        _fraction(thresholds.get(name))
    return set(ids), caps


def learning_gate(
    policy: dict, metrics: dict | None, checkpoint: str, changed_fraction: float, round_index: int
) -> dict:
    caps, thresholds = policy["caps"], policy["learning_curve"]
    reasons = []
    stop = round_index >= caps["max_rounds"]
    if stop:
        reasons.append("refresh_round_cap")
    if metrics is None:
        return {"status": "stop" if stop else "pending", "reasons": reasons + ["missing_curve"]}
    if (
        metrics.get("schema") != "w1ax_refresh_learning_curve_v1"
        or metrics.get("split") != "development"
        or metrics.get("development_sample_sha256") != policy["development_sample_sha256"]
        or metrics.get("checkpoint_sha256") != checkpoint
    ):
        raise ValueError("learning curve needs frozen development and current checkpoint ancestry")
    previous, current = metrics.get("previous", {}), metrics.get("current", {})
    for item in (previous, current):
        checked_hash(item.get("checkpoint_sha256"))
        _positive(item.get("completed_steps"))
        for field in ("hard_ce", "native_acceptance"):
            value = item.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return {"status": "stop", "reasons": ["nonfinite_or_invalid_curve"]}
        _fraction(item["native_acceptance"])
    if current["checkpoint_sha256"] != checkpoint:
        raise ValueError("current curve differs from selected student checkpoint")
    if current["completed_steps"] <= previous["completed_steps"]:
        raise ValueError("learning curve checkpoints must advance completed steps")
    if current["native_acceptance"] < (
        previous["native_acceptance"] - thresholds["max_acceptance_regression"]
    ):
        stop = True
        reasons.append("native_acceptance_regression")
    if current["completed_steps"] < thresholds["min_completed_steps"]:
        reasons.append("insufficient_completed_steps")
    improvement = (previous["hard_ce"] - current["hard_ce"]) / max(previous["hard_ce"], 1e-12)
    if improvement < thresholds["min_relative_ce_improvement"]:
        reasons.append("insufficient_ce_improvement")
    if changed_fraction < thresholds["min_changed_prefix_fraction"]:
        reasons.append("insufficient_student_prefix_change")
    return {
        "status": "stop" if stop else "hold" if reasons else "pass",
        "reasons": reasons,
        "relative_ce_improvement": improvement,
    }


def build_plan(spec: dict) -> dict:
    if spec.get("schema") != "w1ax_refresh_request_v1":
        raise ValueError("unsupported refresh request")
    inputs = spec.get("inputs", {})
    required = {
        "policy",
        "student_rows",
        "checkpoint",
        "student_export",
        "teacher_manifest",
        "train_prompts",
    }
    if set(inputs) not in (required, required | {"learning_curve"}):
        raise ValueError("unexpected refresh input inventory")
    paths = {name: verify_file(record) for name, record in inputs.items()}
    policy = json.loads(paths["policy"].read_text())
    allowed, caps = validate_policy(policy)
    if inputs["train_prompts"]["sha256"] != policy["train_prompts_sha256"]:
        raise ValueError("training prompt file differs from frozen policy")
    training_ids = [row.get("id") for row in read_jsonl(paths["train_prompts"])]
    if (
        any(not isinstance(item, str) or not item for item in training_ids)
        or len(set(training_ids)) != len(training_ids)
        or not allowed.issubset(set(training_ids))
    ):
        raise ValueError("frozen sample includes prompts outside actual training freeze")
    student_identity = spec.get("student", {})
    if student_identity != {
        "checkpoint_sha256": inputs["checkpoint"]["sha256"],
        "export_sha256": inputs["student_export"]["sha256"],
    }:
        raise ValueError("student checkpoint/export ancestry mismatch")
    contract = spec.get("native_teacher_contract")
    if not isinstance(contract, dict) or set(contract) != {
        "target_gguf_sha256",
        "tokenizer_sha256",
        "absolute_d2t_sha256",
        "native_revision",
        "execution_policy_sha256",
        "target_vocab_size",
    }:
        raise ValueError("native teacher execution contract inventory differs")
    for key in (
        "target_gguf_sha256",
        "tokenizer_sha256",
        "absolute_d2t_sha256",
        "execution_policy_sha256",
    ):
        checked_hash(contract[key])
    if (
        not isinstance(contract["native_revision"], str)
        or re.fullmatch("[0-9a-f]{40}", contract["native_revision"]) is None
    ):
        raise ValueError("native revision must be a pinned full commit SHA")
    vocab = _positive(contract["target_vocab_size"])
    teacher = json.loads(paths["teacher_manifest"].read_text())
    if (
        teacher.get("schema") != INDEX_SCHEMA
        or teacher.get("split") != "train"
        or teacher.get("native_teacher_contract") != contract
        or teacher.get("train_prompts_sha256") != policy["train_prompts_sha256"]
    ):
        raise ValueError("teacher index changed split, prompt freeze or execution contract")
    index_path = verify_file(teacher["index"])
    artifacts = teacher.get("artifacts", {})
    if not artifacts:
        raise ValueError("teacher index needs re-auditable source artifacts")
    for record in artifacts.values():
        verify_file(record)
    lookup = {}
    for row in read_jsonl(index_path):
        tokens = prefix(row.get("prefix_token_ids"), vocab, caps["max_prefix_tokens"])
        if row.get("split") != "train":
            raise ValueError("teacher index may not include development or sealed final rows")
        kind = row.get("kind")
        if (
            kind not in ("label", "feature")
            or row.get("artifact") not in artifacts
            or type(row.get("row")) is not int
            or row["row"] < 0
            or not isinstance(row.get("capture_id"), str)
            or not row["capture_id"]
            or not isinstance(row.get("prompt_id"), str)
            or not row["prompt_id"]
        ):
            raise ValueError("teacher index row missing native artifact ancestry")
        if kind == "label" and (
            type(row.get("next_target_id")) is not int or not 0 <= row["next_target_id"] < vocab
        ):
            raise ValueError("native label must be an absolute target ID")
        key = (row["prompt_id"], kind, tokens)
        if key in lookup:
            raise ValueError("ambiguous same-prefix teacher rows must be resolved by capture audit")
        lookup[key] = row
    rows = read_jsonl(paths["student_rows"])
    if not 1 <= len(rows) <= caps["max_student_rows"]:
        raise ValueError("student trajectory row cap exceeded or empty")
    row_ids, prompts, requirements, student_rows = set(), set(), {}, []
    for row in rows:
        if (
            row.get("split") != "train"
            or row.get("prompt_id") not in allowed
            or row.get("checkpoint_sha256") != student_identity["checkpoint_sha256"]
            or row.get("export_sha256") != student_identity["export_sha256"]
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in row_ids
        ):
            raise ValueError("student row changed frozen train sample or checkpoint ancestry")
        row_ids.add(row["id"])
        prompts.add(row["prompt_id"])
        tokens = prefix(row.get("prefix_token_ids"), vocab, caps["max_prefix_tokens"])
        root = prefix(row.get("feature_prefix_token_ids"), vocab, caps["max_prefix_tokens"])
        if tokens[: len(root)] != root:
            raise ValueError("accepted feature root is not an ancestor of label prefix")
        label_key = (row["prompt_id"], "label", tokens)
        keys = [label_key]
        keys.extend((row["prompt_id"], "feature", root[:n]) for n in range(1, len(root) + 1))
        for key in keys:
            requirements.setdefault(key, set()).add(row["id"])
        student_rows.append((row["id"], label_key))
    if prompts != allowed:
        raise ValueError("student trajectories must cover exactly the frozen train sample")
    reused, missing = [], []
    for key, consumers in sorted(requirements.items()):
        entry = {
            "prompt_id": key[0],
            "kind": key[1],
            "prefix_token_ids": list(key[2]),
            "prefix_sha256": digest(list(key[2])),
            "student_row_ids": sorted(consumers),
        }
        if key in lookup:
            reused.append({**entry, "teacher": lookup[key]})
        else:
            missing.append(entry)
    new_labels = sum(row["kind"] == "label" for row in missing)
    new_features = len(missing) - new_labels
    changed = sum(key not in lookup for _, key in student_rows) / len(student_rows)
    curve = json.loads(paths["learning_curve"].read_text()) if "learning_curve" in paths else None
    round_index = spec.get("refresh_round")
    if type(round_index) is not int or round_index < 0:
        raise ValueError("refresh round must be a nonnegative integer")
    gate = learning_gate(policy, curve, student_identity["checkpoint_sha256"], changed, round_index)
    within_cap = (
        new_labels <= caps["max_new_label_rows"] and new_features <= caps["max_new_feature_rows"]
    )
    queue_ready = gate["status"] == "pass" and within_cap and bool(missing)
    return {
        "schema": PLAN_SCHEMA,
        "split": "train",
        "request": spec,
        "request_sha256": digest(spec),
        "frozen_sample_sha256": digest(policy["train_sample"]),
        "teacher_index_sha256": teacher["index"]["sha256"],
        "native_teacher_contract_sha256": digest(contract),
        "reused": reused,
        "capture_requests": missing,
        "counts": {
            "student_rows": len(rows),
            "reused_requirements": len(reused),
            "new_label_rows": new_labels,
            "new_feature_rows": new_features,
            "changed_label_prefix_fraction": changed,
        },
        "learning_gate": gate,
        "within_refresh_cap": within_cap,
        "capture_queue_ready": queue_ready,
        "training_eligible": False,
        "readiness": "missing_native_teacher_capture"
        if missing
        else "ancestry_complete_provider_gate_required",
        "limits": [
            "planning does not authorize capture or training",
            "provider must re-audit labels/features",
            "student hidden states/cache must be rebuilt from the current checkpoint",
        ],
    }


def audit_plan(path: Path) -> dict:
    saved = json.loads(path.read_text())
    expected = build_plan(saved["request"])
    if saved != expected:
        raise ValueError("refresh plan differs from recomputed source ancestry or gate")
    return {
        "schema": "w1ax_refresh_plan_audit_v1",
        "status": "pass",
        "plan_sha256": sha256(path),
        "counts": expected["counts"],
        "training_eligible": False,
        "capture_queue_ready": expected["capture_queue_ready"],
    }
