#!/usr/bin/env python3
"""Prepare/re-audit CPU refresh schedules; import existing audited v1 captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.trajectory_refresh import (  # noqa: E402
    INDEX_SCHEMA,
    audit_native_source,
    audit_plan,
    build_plan,
    digest,
    file_record,
    read_jsonl,
    sha256,
)


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def index_v1(args) -> dict:
    # Lazy import: plan/audit use only the standard library and never load Torch.
    from audit_recurrent_binary_capture import load_audited_capture

    if args.output.exists():
        raise ValueError("teacher index output must be new")
    manifest = json.loads(args.capture_manifest.read_text())
    contract = json.loads(args.native_contract.read_text())
    capture = load_audited_capture(
        args.capture_manifest,
        args.prompts,
        args.prompts_sha256,
        expected_prompt_count=args.prompt_count,
    )
    if contract.get("target_vocab_size") != manifest["target_vocab_size"]:
        raise ValueError("native teacher vocabulary differs from audited bundle")
    cell_path = args.capture_manifest.parent / "source_cell_manifest.json"
    if sha256(cell_path) != manifest["source_report_sha256"]["cell_manifest"]:
        raise ValueError("native source cell manifest changed")
    cell = json.loads(cell_path.read_text())
    if contract.get("target_gguf_sha256") != cell.get("target_sha256"):
        raise ValueError("native teacher target differs from audited bundle")
    # Prefix variation across captures is never arbitrarily merged. One bundle
    # has passed feature-ledger uniqueness and native label ancestry checks.
    capture_id = sha256(args.capture_manifest)
    rows = []
    for (prompt, tokens), row in sorted(capture.feature_lookup.items()):
        rows.append(
            {
                "kind": "feature",
                "prompt_id": prompt,
                "split": "train",
                "prefix_token_ids": list(tokens),
                "artifact": "features",
                "row": row,
                "capture_id": capture_id,
            }
        )
    native_rows = read_jsonl(args.capture_manifest.parent / manifest["rows"]["path"])
    labels = {}
    for index, row in enumerate(native_rows):
        if row["valid"] is True:
            key = (row["prompt_id"], tuple(row["prefix_token_ids"]))
            existing = labels.get(key)
            if existing is not None:
                if existing["next_target_id"] != row["verifier_token_id"]:
                    raise ValueError("audited same-prefix labels disagree; resolve before refresh")
                existing["equivalent_source_rows"].append(index)
                continue
            labels[key] = {
                "kind": "label",
                "prompt_id": row["prompt_id"],
                "split": "train",
                "prefix_token_ids": row["prefix_token_ids"],
                "artifact": "rows",
                "row": index,
                "capture_id": capture_id,
                "next_target_id": row["verifier_token_id"],
                "equivalent_source_rows": [index],
            }
    rows.extend(labels.values())
    args.output.mkdir(parents=True)
    index_path = args.output / "rows.jsonl"
    index_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit_path = args.output / "native-audit.json"
    write_new(audit_path, capture.report)
    artifacts = {
        "capture_manifest": file_record(args.capture_manifest),
        "prompts": file_record(args.prompts),
        "audit": file_record(audit_path),
        "native_contract": file_record(args.native_contract),
        "source_cell": file_record(cell_path),
    }
    for name in (
        "features",
        "feature_rows",
        "rows",
        "anchors",
        "offsets",
        "t2d",
        "target_logits",
        "target_logits_rows",
        "shard_manifest",
        "internal_continuity",
    ):
        record = manifest.get(name)
        if isinstance(record, dict) and "path" in record:
            artifacts[name] = file_record(args.capture_manifest.parent / record["path"])
    for name, filename in (
        ("rows", "rows_preparation.json"),
        ("features", "features_preparation.json"),
    ):
        expected = manifest["source_report_sha256"].get(name)
        if expected is not None:
            path = args.capture_manifest.parent / filename
            if sha256(path) != expected:
                raise ValueError("native preparation source report changed")
            artifacts[name + "_preparation"] = file_record(path)
    result = {
        "schema": INDEX_SCHEMA,
        "source_schema": "recurrent_binary_capture_v1",
        "split": "train",
        "train_prompts_sha256": args.prompts_sha256,
        "native_teacher_contract": contract,
        "index": file_record(index_path),
        "artifacts": artifacts,
        "training_eligible": False,
    }
    write_new(args.output / "manifest.json", result)
    return result


def bridge_native(args) -> dict:
    """Materialize source-bound refresh and pending capture/provider artifacts."""
    if args.output.exists():
        raise ValueError("native bridge output must be new")
    source = json.loads(args.source.read_text())
    bridge = audit_native_source(source)
    args.output.mkdir(parents=True)
    rows_path = args.output / "student-rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in bridge["student_rows"])
    )
    bridge_path = args.output / "native-bridge.json"
    write_new(bridge_path, bridge)
    request = json.loads(json.dumps(source["request"]))
    request["inputs"] = {
        **request["inputs"],
        "student_rows": file_record(rows_path),
        "native_bridge": file_record(bridge_path),
    }
    write_new(args.output / "refresh-request.json", request)
    plan = build_plan(request)
    plan_path = args.output / "refresh-plan.json"
    write_new(plan_path, plan)
    write_new(args.output / "refresh-plan-audit.json", audit_plan(plan_path))
    # Capture work is grouped only by exact prompt/prefix, never by text, row
    # position or a candidate-D proposal. These are HTTP preparation templates.
    capture = {}
    for row in plan["capture_requests"]:
        key = (row["prompt_id"], tuple(row["prefix_token_ids"]))
        entry = capture.setdefault(
            key,
            {
                "prompt_id": row["prompt_id"],
                "prefix_token_ids": row["prefix_token_ids"],
                "prefix_sha256": row["prefix_sha256"],
                "required_outputs": [],
                "student_row_ids": [],
                "request_template": {
                    "endpoint": "/completion",
                    "body": {
                        "prompt": row["prefix_token_ids"],
                        "n_predict": 1,
                        "temperature": 0,
                        "seed": 42,
                        "cache_prompt": False,
                        "return_tokens": True,
                    },
                },
            },
        )
        entry["required_outputs"].append(row["kind"])
        entry["student_row_ids"] = sorted(set(entry["student_row_ids"] + row["student_row_ids"]))
    write_new(
        args.output / "capture-inputs.json",
        {
            "schema": "w1ax_refresh_native_capture_preparation_v1",
            "split": "train",
            "refresh_plan": file_record(plan_path),
            "requests": list(capture.values()),
            "native_teacher_contract": request["native_teacher_contract"],
            "capture_queue_ready": plan["capture_queue_ready"],
            "execution_authorized": False,
            "training_eligible": False,
            "unresolved_gates": [
                "explicit_capture_budget_and_sole_gpu_owner_schedule",
                "native_token_array_request_and_teacher_feature_instrumentation_validation",
                "frozen_verifier_sampler_options_must_bind_request_template",
                "exact_prefix_label_and_feature_storage_audit",
            ],
        },
    )
    references = {
        row["kind"] + ":" + row["prompt_id"] + ":" + row["prefix_sha256"]: {
            "status": "captured",
            "teacher": row["teacher"],
        }
        for row in plan["reused"]
    }
    references.update(
        {
            row["kind"] + ":" + row["prompt_id"] + ":" + row["prefix_sha256"]: {
                "status": "missing_native_capture"
            }
            for row in plan["capture_requests"]
        }
    )
    rounds = {}
    for row in bridge["student_rows"]:
        key = (row["prompt_id"], row["native_round_index"])
        group = rounds.setdefault(
            key,
            {
                "prompt_id": row["prompt_id"],
                "round_index": row["native_round_index"],
                "feature_prefix_token_ids": row["feature_prefix_token_ids"],
                "cache_policy": "rebuild_all_positions_with_current_checkpoint_f16_kv",
                "student_row_ids": [],
                "labels": [],
                "features": [
                    references["feature:" + row["prompt_id"] + ":" + digest(tokens)]
                    for tokens in [
                        row["feature_prefix_token_ids"][:n]
                        for n in range(1, len(row["feature_prefix_token_ids"]) + 1)
                    ]
                ],
            },
        )
        group["student_row_ids"].append(row["id"])
        group["labels"].append(
            {
                "student_row_id": row["id"],
                "depth": row["native_depth"],
                "prefix_token_ids": row["prefix_token_ids"],
                **references["label:" + row["prompt_id"] + ":" + digest(row["prefix_token_ids"])],
            }
        )
    write_new(
        args.output / "provider-preparation.json",
        {
            "schema": "w1ax_refresh_provider_preparation_v1",
            "split": "train",
            "refresh_plan": file_record(plan_path),
            "student": request["student"],
            "native_bridge": file_record(bridge_path),
            "rounds": list(rounds.values()),
            "training_eligible": False,
            "loader_factory": None,
            "unresolved_gates": bridge["unresolved_gates"],
        },
    )
    return {
        "counts": plan["counts"],
        "learning_gate": plan["learning_gate"],
        "continuity_gaps": bridge["continuity_gaps"],
        "training_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    index = commands.add_parser("index-v1")
    index.add_argument("--capture-manifest", type=Path, required=True)
    index.add_argument("--prompts", type=Path, required=True)
    index.add_argument("--prompts-sha256", required=True)
    index.add_argument("--prompt-count", type=int, required=True)
    index.add_argument("--native-contract", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    bridge = commands.add_parser("bridge-native")
    bridge.add_argument("--source", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(json.loads(args.request.read_text()))
        write_new(args.output, result)
        print(
            json.dumps(
                {
                    "counts": result["counts"],
                    "learning_gate": result["learning_gate"],
                    "capture_queue_ready": result["capture_queue_ready"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "audit":
        result = audit_plan(args.plan)
        write_new(args.output, result)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "bridge-native":
        print(json.dumps(bridge_native(args), sort_keys=True))
    else:
        result = index_v1(args)
        print(json.dumps({"schema": result["schema"], "index": result["index"]}, sort_keys=True))


if __name__ == "__main__":
    main()
