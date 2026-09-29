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
    if not required <= set(inputs) or set(inputs) - required - {"learning_curve", "native_bridge"}:
        raise ValueError("unexpected refresh input inventory")
    paths = {name: verify_file(record) for name, record in inputs.items()}
    if "native_bridge" in paths:
        audit_native_bridge(paths["native_bridge"], paths["student_rows"])
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


def audit_native_source(spec: dict) -> dict:
    """Bridge canonical native own-history head/round captures on CPU.

    This binds declared execution identity and verifies metadata/payload ancestry.
    It never treats diagnostic state bytes or copied head labels as training data.
    """
    import array
    import sys

    if spec.get("schema") != "w1ax_native_refresh_source_v1":
        raise ValueError("unsupported native refresh source")
    files = spec.get("files", {})
    required = {
        "cell_manifest",
        "heads",
        "rounds",
        "states",
        "task_map",
        "checkpoint_manifest",
        "export_report",
        "execution_binding",
        "binary",
        "capture_prompts",
    }
    if set(files) != required:
        raise ValueError("native source requires canonical head/state/round capture inventory")
    paths = {name: verify_file(record) for name, record in files.items()}
    request = spec.get("request", {})
    base = request.get("inputs", {})
    if request.get("schema") != "w1ax_refresh_request_v1" or set(base) not in (
        {"policy", "checkpoint", "student_export", "teacher_manifest", "train_prompts"},
        {
            "policy",
            "checkpoint",
            "student_export",
            "teacher_manifest",
            "train_prompts",
            "learning_curve",
        },
    ):
        raise ValueError("bridge requires a refresh request before normalized student rows")
    base_paths = {name: verify_file(record) for name, record in base.items()}
    policy = json.loads(base_paths["policy"].read_text())
    allowed, caps = validate_policy(policy)
    train_ids = [row.get("id") for row in read_jsonl(base_paths["train_prompts"])]
    if (
        base["train_prompts"]["sha256"] != policy["train_prompts_sha256"]
        or len(set(train_ids)) != len(train_ids)
        or not allowed.issubset(set(train_ids))
    ):
        raise ValueError("native sample differs from actual frozen training prompts")
    identity = {
        "checkpoint_sha256": base["checkpoint"]["sha256"],
        "export_sha256": base["student_export"]["sha256"],
    }
    if request.get("student") != identity:
        raise ValueError("native selected student identity differs from checkpoint/export")
    checkpoint = json.loads(paths["checkpoint_manifest"].read_text())
    exported = json.loads(paths["export_report"].read_text())
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("checkpoint_sha256") != identity["checkpoint_sha256"]
        or exported.get("schema_version") != 1
        or exported.get("serialization_audit_passed") is not True
        or exported.get("checkpoint", {}).get("sha256") != identity["checkpoint_sha256"]
        or exported.get("output", {}).get("sha256") != identity["export_sha256"]
        or exported.get("checkpoint_manifest", {}).get("sha256")
        != files["checkpoint_manifest"]["sha256"]
        or exported.get("scale_layout") != checkpoint.get("scale_layout")
        or exported.get("activation_bits") != checkpoint.get("activation_bits")
    ):
        raise ValueError("native export report does not bind selected checkpoint contract")
    cell = json.loads(paths["cell_manifest"].read_text())
    if cell.get("schema") != "binary_head_capture_cell_v1" or cell.get("complete") is not True:
        raise ValueError(
            "native cell must be complete; benchmark summary alone lacks prefix ancestry"
        )
    contract = request.get("native_teacher_contract", {})
    vocab = _positive(contract.get("target_vocab_size"))
    if (
        cell.get("draft_sha256") != identity["export_sha256"]
        or cell.get("target_sha256") != contract.get("target_gguf_sha256")
        or cell.get("binary_sha256") != files["binary"]["sha256"]
    ):
        raise ValueError("native cell ran a different student, target or executable")
    binding = json.loads(paths["execution_binding"].read_text())
    cache_policy = {
        "student_state_source": "current_checkpoint_rebuild",
        "kv_storage_dtype": "f16",
        "position_policy": "absolute_prefix_contiguous",
        "attention_mask": "causal_exact_prefix",
    }
    if (
        binding.get("schema") != "w1ax_native_refresh_execution_binding_v1"
        or binding.get("cell_manifest_sha256") != files["cell_manifest"]["sha256"]
        or binding.get("native_teacher_contract_sha256") != digest(contract)
        or binding.get("binary_sha256") != files["binary"]["sha256"]
        or binding.get("native_revision") != contract.get("native_revision")
        or binding.get("command_sha256") != digest(cell.get("command"))
        or binding.get("environment_sha256") != digest(cell.get("env"))
        or binding.get("cache_policy") != cache_policy
        or binding.get("state_byteorder") not in ("little", "big")
    ):
        raise ValueError("native execution/cache binding differs from captured cell")
    evidence = binding.get("evidence")
    if not isinstance(evidence, dict) or not {"native_revision", "execution_policy"} <= set(
        evidence
    ):
        raise ValueError("native execution binding requires source and execution-policy evidence")
    evidence_paths = {name: verify_file(record) for name, record in evidence.items()}
    if evidence_paths["native_revision"].read_text().strip() != contract[
        "native_revision"
    ] or evidence["execution_policy"]["sha256"] != contract.get("execution_policy_sha256"):
        raise ValueError("native revision or execution-policy evidence differs")
    for name in ("heads", "rounds", "states"):
        record = cell.get("files", {}).get(paths[name].name, {})
        if (
            record.get("sha256") != files[name]["sha256"]
            or record.get("bytes") != paths[name].stat().st_size
        ):
            raise ValueError("canonical native trace does not match cell file hashes")
    mapping = json.loads(paths["task_map"].read_text())
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(task, str) or not task.isdecimal() for task in mapping)
        or set(mapping.values()) != allowed
        or len(mapping) != len(allowed)
    ):
        raise ValueError("native task ownership must cover exactly the frozen training sample")
    capture_prompts = read_jsonl(paths["capture_prompts"])
    capture_ids = [row.get("id") for row in capture_prompts]
    captured_prompt_map = {row.get("id"): row for row in capture_prompts}
    frozen_prompt_map = {row.get("id"): row for row in read_jsonl(base_paths["train_prompts"])}
    cell_mapping = cell.get("task_prompt_ids", {})
    if (
        cell.get("prompts_sha256") != files["capture_prompts"]["sha256"]
        or cell.get("prompt_count") != len(capture_prompts)
        or cell.get("ordered_prompt_ids") != capture_ids
        or len(set(capture_ids)) != len(capture_ids)
        or set(cell_mapping) != set(mapping)
        or set(cell_mapping.values()) != set(capture_ids)
    ):
        raise ValueError("native capture prompts or source task map differ from cell")
    for task, prompt_id in mapping.items():
        captured = captured_prompt_map[cell_mapping[task]]
        frozen = frozen_prompt_map[prompt_id]
        if (
            not isinstance(frozen.get("messages"), list)
            or not frozen["messages"]
            or captured.get("messages") != frozen["messages"]
        ):
            raise ValueError("native captured prompt content differs from frozen training sample")
    if "EAGLE_FORCE_ROUNDS_JSONL" in cell.get("env", {}):
        raise ValueError("forced round execution is ineligible for student trajectory refresh")
    heads, rounds = read_jsonl(paths["heads"]), read_jsonl(paths["rounds"])
    if not 1 <= len(heads) <= caps["max_student_rows"]:
        raise ValueError("native head row cap exceeded or empty")
    requests = cell.get("requests")
    if not isinstance(requests, list) or len(requests) != len(mapping):
        raise ValueError("native requests have ambiguous task ownership")
    head_cursor = round_cursor = 0
    seen_tasks = set()
    for item in requests:
        task = item.get("task_id")
        if task not in mapping or task in seen_tasks or item.get("id") != cell_mapping[task]:
            raise ValueError("native request task/prompt ownership differs")
        seen_tasks.add(task)
        for key, trace, cursor in (
            ("capture_rows", heads, head_cursor),
            ("forced_round_rows", rounds, round_cursor),
        ):
            span = item.get(key)
            if (
                not isinstance(span, list)
                or len(span) != 2
                or span[0] != cursor
                or type(span[1]) is not int
                or not cursor < span[1] <= len(trace)
                or any(str(row.get("task_id")) != task for row in trace[cursor : span[1]])
            ):
                raise ValueError(
                    "native request row range is incomplete or belongs to another task"
                )
        head_cursor, round_cursor = item["capture_rows"][1], item["forced_round_rows"][1]
    if head_cursor != len(heads) or round_cursor != len(rounds):
        raise ValueError("unclaimed native trace rows cannot be silently dropped")
    recorded, last, gaps = {}, {}, []
    for row in rounds:
        task, index = str(row.get("task_id")), row.get("round_index")
        if (
            row.get("schema") != "eagle_forced_round_v1"
            or task not in mapping
            or type(index) is not int
            or index != len(last.get(task, []))
        ):
            raise ValueError("native proposal rounds must be canonical and contiguous from zero")
        root = prefix(row.get("prefix_token_ids"), vocab, caps["max_prefix_tokens"])
        seed, accepted = row.get("seed_token_id"), row.get("accepted_drafts")
        draft, verifier = row.get("draft_token_ids"), row.get("verifier_token_ids")
        if (
            type(seed) is not int
            or not 0 <= seed < vocab
            or type(row.get("pos0")) is not int
            or row.get("pos0") != len(root)
            or not isinstance(draft, list)
            or not 1 <= len(draft) <= 5
            or any(type(token) is not int or not 0 <= token < vocab for token in draft)
            or type(accepted) is not int
            or not 0 <= accepted <= len(draft)
            or not isinstance(verifier, list)
            or len(verifier) != accepted + 1
            or any(type(token) is not int or not 0 <= token < vocab for token in verifier)
            or verifier[:accepted] != draft[:accepted]
        ):
            raise ValueError("native round token/position/acceptance ancestry differs")
        previous = last.setdefault(task, [])
        if previous:
            old = previous[-1]
            expected_root = (
                old["prefix_token_ids"] + [old["seed_token_id"]] + old["verifier_token_ids"][:-1]
            )
            if list(root) != expected_root or seed != old["verifier_token_ids"][-1]:
                gaps.append(
                    {
                        "task_id": task,
                        "round_index": index,
                        "gate": "native_feature_disposition_continuity_audit_required",
                    }
                )
        previous.append(row)
        recorded[(task, index)] = row
    normalized, seen_heads, width = [], set(), _positive(heads[0].get("state_dim"))
    for state_row, row in enumerate(heads):
        task, index, depth = str(row.get("task_id")), row.get("round_index"), row.get("depth")
        record = recorded.get((task, index))
        if (
            row.get("schema") != "eagle_head_state_v1"
            or type(row.get("state_row")) is not int
            or row.get("state_row") != state_row
            or type(row.get("state_dim")) is not int
            or row.get("state_dim") != width
            or row.get("state_dtype") != "float32_native_endian"
            or row.get("state_boundary") != "native_output_norm_f32_before_head_operand_conversion"
            or row.get("forced") is not False
            or row.get("finite") is not True
            or row.get("valid") is not True
            or row.get("alignment_valid") is not True
            or row.get("is_bonus") is not False
            or record is None
            or type(index) is not int
            or type(depth) is not int
            or not 0 <= depth < len(record["draft_token_ids"])
        ):
            raise ValueError("native head state/own-history contract differs")
        key = (task, index, depth)
        if key in seen_heads:
            raise ValueError("duplicate native head round/depth")
        seen_heads.add(key)
        tokens = prefix(row.get("prefix_token_ids"), vocab, caps["max_prefix_tokens"])
        expected = (
            record["prefix_token_ids"]
            + [record["seed_token_id"]]
            + record["draft_token_ids"][:depth]
        )
        parent = len(record["prefix_token_ids"]) - 1
        if (
            any(
                type(row.get(field)) is not int
                for field in ("parent_position", "input_position", "label_position", "verifier_row")
            )
            or list(tokens) != expected
            or row.get("parent_position") != parent
            or row.get("input_position") != parent + depth + 1
            or row.get("label_position") != parent + depth + 2
            or row.get("verifier_row") != depth
            or row.get("input_token_id") != tokens[-1]
            or row.get("proposed_token_id") != record["draft_token_ids"][depth]
        ):
            raise ValueError("native head prefix or absolute position differs from proposal round")
        normalized.append(
            {
                "id": f"native:{task}:{index}:{depth}",
                "prompt_id": mapping[task],
                "split": "train",
                **identity,
                "prefix_token_ids": list(tokens),
                "feature_prefix_token_ids": record["prefix_token_ids"],
                "native_head_row": state_row,
                "native_round_index": index,
                "native_task_id": task,
                "native_depth": depth,
            }
        )
    expected_heads = {
        (task, index, depth)
        for (task, index), row in recorded.items()
        for depth in range(len(row["draft_token_ids"]))
    }
    if seen_heads != expected_heads:
        raise ValueError("native capture omits a computed proposal depth")
    if paths["states"].stat().st_size != len(heads) * width * 4:
        raise ValueError("native F32 state payload does not match head row inventory")
    with paths["states"].open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            values = array.array("f")
            values.frombytes(block)
            if binding["state_byteorder"] != sys.byteorder:
                values.byteswap()
            if not all(math.isfinite(value) for value in values):
                raise ValueError("native state payload contains nonfinite values")
    return {
        "schema": "w1ax_native_refresh_bridge_v1",
        "source": spec,
        "source_sha256": digest(spec),
        "student_rows": normalized,
        "student_rows_sha256": digest(normalized),
        "counts": {
            "heads": len(heads),
            "rounds": len(rounds),
            "prompts": len(mapping),
            "state_dim": width,
        },
        "continuity_gaps": gaps,
        "training_eligible": False,
        "unresolved_gates": [
            "native_response_and_cross_round_continuity_audit",
            "changed_prefix_native_teacher_capture",
            "provider_label_feature_payload_and_readiness_audit",
        ],
        "limits": [
            "execution binding declares source/build policy; this is no numeric gate",
            "native state bytes and head labels are diagnostics only and never reused",
        ],
    }


def audit_native_bridge(path: Path, student_rows_path: Path) -> dict:
    bridge = json.loads(path.read_text())
    if bridge != audit_native_source(bridge["source"]):
        raise ValueError("native bridge differs from recomputed source ancestry")
    if read_jsonl(student_rows_path) != bridge["student_rows"]:
        raise ValueError("normalized student rows differ from native trace ancestry")
    return bridge
