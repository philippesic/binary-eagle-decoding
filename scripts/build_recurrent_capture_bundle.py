#!/usr/bin/env python3
"""Join native recurrent preparer outputs into a CPU-audited preparation bundle.

The builder accepts only a complete, cell-owned train capture. It copies every
file needed by the recurrent capture audit into a new directory (or explicitly
hardlinks the retained raw logit stream), verifies both preparers' source hashes and ownership claims, and runs that final audit before
publishing the directory. The current preparers cannot certify pinned model
identity or live drafter cache parity, so this output is always preparation-only.
No model or accelerator is run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from audit_recurrent_binary_capture import (
    TRAIN_PROMPTS,
    TRAIN_PROMPTS_SHA256,
    audit_capture,
    read_jsonl,
    resolve_prompt_expectation,
    sha256,
    validate_shard_manifest,
)
from prepare_recurrent_native_rows import MAX_DEPTH
from run_binary_head_capture import (
    DRAFT_D_D2T_SHA256,
    DRAFT_D_SHA256,
    TARGET_F16_SHA256,
)

FILES = {
    "rows": ("rows.jsonl", "rows.jsonl"),
    "anchors": ("anchors.jsonl", "anchors.jsonl"),
    "offsets": ("offsets.npy", "offsets.npy"),
    "t2d": ("t2d.npy", "t2d.npy"),
    "features": ("features.npy", "features.npy"),
    "feature_rows": ("feature_rows.jsonl", "feature_rows.jsonl"),
}


def _json_object(path: Path, schema: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path.name} has wrong schema")
    return value


def _hash_matches(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256(path)
    if expected != actual:
        raise ValueError(f"{label} SHA256 mismatch")
    return actual


def _same_source(left: object, right: object, label: str) -> None:
    if not isinstance(left, str) or not left or left != right:
        raise ValueError(f"cross-source {label} SHA256 mismatch")


def _verify_cell(cell: dict, prompt_hash: str, prompt_ids: set[str]) -> None:
    if cell.get("complete") is not True or cell.get("prompts_sha256") != prompt_hash:
        raise ValueError("cell manifest is incomplete or has wrong prompt hash")
    task_map = cell.get("task_prompt_ids")
    requests = cell.get("requests")
    if (
        not isinstance(task_map, dict)
        or not task_map
        or not isinstance(requests, list)
        or len(requests) != len(task_map)
        or len(set(task_map.values())) != len(task_map)
        or any(
            not isinstance(task, str) or not task.isdecimal() or prompt not in prompt_ids
            for task, prompt in task_map.items()
        )
    ):
        raise ValueError("cell manifest cannot prove task ownership")
    seen = set()
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("cell request is not an object")
        task = request.get("task_id")
        if task in seen or task_map.get(task) != request.get("id"):
            raise ValueError("cell request/task ownership mismatch")
        seen.add(task)
    if seen != set(task_map):
        raise ValueError("cell manifest omits a mapped task")


def _verify_raw_source_hashes(cell: dict, hashes: dict[str, object]) -> None:
    files = cell.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("cell manifest lacks raw source file records")
    recorded = {
        record.get("sha256")
        for record in files.values()
        if isinstance(record, dict)
        and type(record.get("bytes")) is int
        and record["bytes"] > 0
        and isinstance(record.get("sha256"), str)
    }
    for label, value in hashes.items():
        if not isinstance(value, str) or value not in recorded:
            raise ValueError(f"cell manifest lacks matching {label} source hash")


def _read_prompt_ids(path: Path, expected_count: int) -> set[str]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row.get("id") for row in records if isinstance(row, dict)]
    if (
        len(records) != expected_count
        or len(ids) != expected_count
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != expected_count
    ):
        raise ValueError("frozen train prompt IDs are missing or duplicated")
    return set(ids)


def build_bundle(
    rows_dir: Path,
    features_dir: Path,
    target_logits: Path,
    cell_manifest: Path,
    train_prompts: Path,
    output_dir: Path,
    *,
    continuity_report: Path | None = None,
    shard_manifest_path: Path | None = None,
    expected_prompt_hash: str = TRAIN_PROMPTS_SHA256,
    expected_prompt_count: int = TRAIN_PROMPTS,
    expected_target_hash: str = TARGET_F16_SHA256,
    expected_draft_hash: str = DRAFT_D_SHA256,
    expected_map_raw_hash: str = DRAFT_D_D2T_SHA256,
    raw_logits_storage: str = "copy",
) -> dict:
    """Create an audited non-trainable bundle; raw defaults to an owned copy.

    Explicit hardlink mode shares the retained source inode, requiring the same
    filesystem. It is for transient v2 preparation only and never retires raw.
    """
    if raw_logits_storage not in {"copy", "hardlink"}:
        raise ValueError("raw_logits_storage must be copy or hardlink")
    rows_dir = Path(rows_dir)
    features_dir = Path(features_dir)
    target_logits = Path(target_logits)
    cell_manifest = Path(cell_manifest)
    train_prompts = Path(train_prompts)
    output_dir = Path(output_dir)
    continuity_report = Path(continuity_report) if continuity_report is not None else None
    shard_manifest_path = Path(shard_manifest_path) if shard_manifest_path is not None else None
    if output_dir.exists():
        raise ValueError("bundle output directory must be new")
    prompt_hash = _hash_matches(train_prompts, expected_prompt_hash, "frozen train prompts")
    prompt_ids = _read_prompt_ids(train_prompts, expected_prompt_count)
    if shard_manifest_path is not None:
        ordered_ids = [row["id"] for row in read_jsonl(train_prompts)]
        validate_shard_manifest(
            shard_manifest_path,
            train_prompts,
            prompt_hash,
            expected_prompt_count,
            ordered_ids,
        )
    rows_report = _json_object(
        rows_dir / "preparation.json", "recurrent_native_rows_preparation_v1"
    )
    feature_report = _json_object(
        features_dir / "report.json", "recurrent_native_feature_selection_v1"
    )
    cell = _json_object(cell_manifest, "binary_head_capture_cell_v1")
    _verify_cell(cell, prompt_hash, prompt_ids)
    if cell.get("shard_manifest_sha256") is not None and shard_manifest_path is None:
        raise ValueError("sharded cell requires its frozen shard manifest")
    if shard_manifest_path is not None and (
        cell.get("shard_manifest_sha256") != sha256(shard_manifest_path)
        or cell.get("prompt_count") != expected_prompt_count
        or cell.get("ordered_prompt_ids") != ordered_ids
    ):
        raise ValueError("cell capture differs from frozen shard manifest")
    if (
        cell.get("target_sha256") != expected_target_hash
        or cell.get("draft_sha256") != expected_draft_hash
        or rows_report.get("absolute_map_raw_sha256") != expected_map_raw_hash
    ):
        raise ValueError("capture does not have pinned target, draft and D map identity")
    cell_hash = sha256(cell_manifest)
    if (
        rows_report.get("split") != "train"
        or rows_report.get("prompts_sha256") != prompt_hash
        or feature_report.get("training_prompts_sha256") != prompt_hash
        or feature_report.get("execution_device") != "cpu"
    ):
        raise ValueError("preparer reports disagree with frozen train split or CPU execution")
    if shard_manifest_path is not None and (
        rows_report.get("prompt_count") != expected_prompt_count
        or feature_report.get("training_prompt_count") != expected_prompt_count
    ):
        raise ValueError("preparer reports disagree with frozen shard prompt count")
    if rows_report.get("request_prompt_ownership") != "cell_manifest_request_ranges_verified":
        raise ValueError("native row task ownership is unverified")
    if (
        feature_report.get("task_prompt_ownership")
        != "cell_manifest_request_and_file_hashes_verified"
    ):
        raise ValueError("native feature task ownership is unverified")
    row_sources = rows_report.get("source_sha256")
    feature_sources = feature_report.get("sources")
    row_outputs = rows_report.get("output_sha256")
    if not all(isinstance(item, dict) for item in (row_sources, feature_sources, row_outputs)):
        raise ValueError("preparer source/output hashes are missing")
    _same_source(row_sources.get("cell_manifest"), cell_hash, "cell manifest")
    _same_source(feature_sources.get("cell_manifest_sha256"), cell_hash, "cell manifest")
    _same_source(row_sources.get("prompts"), prompt_hash, "prompts")
    _same_source(row_sources.get("task_map"), feature_sources.get("task_map_sha256"), "task map")
    _same_source(row_outputs.get("anchors.jsonl"), feature_sources.get("anchors_sha256"), "anchors")
    _hash_matches(target_logits, row_sources.get("raw_target_logits"), "target logits")
    _verify_raw_source_hashes(
        cell,
        {
            "heads": row_sources.get("heads"),
            "rounds": row_sources.get("rounds"),
            "target logits": row_sources.get("raw_target_logits"),
            "target feature events": feature_sources.get("metadata_sha256"),
            "target feature values": feature_sources.get("values_sha256"),
        },
    )
    if continuity_report is not None:
        continuity = _json_object(continuity_report, "recurrent_internal_continuity_v1")
        sources = continuity.get("source_sha256")
        if (
            continuity.get("status") != "internal_accepted_prefix_continuity_verified"
            or continuity.get("execution_device") != "cpu"
            or continuity.get("rounds") != rows_report.get("rounds")
            or not isinstance(sources, dict)
        ):
            raise ValueError("continuity report has unverified status or wrong round count")
        _same_source(sources.get("rounds"), row_sources.get("rounds"), "continuity rounds")
        _same_source(
            sources.get("feature_events"),
            feature_sources.get("metadata_sha256"),
            "continuity feature events",
        )
        _same_source(sources.get("task_map"), row_sources.get("task_map"), "continuity task map")
    if (
        type(rows_report.get("target_vocab_size")) is not int
        or rows_report["target_vocab_size"] < 1
    ):
        raise ValueError("target vocabulary is missing")
    if type(rows_report.get("draft_vocab_size")) is not int or rows_report["draft_vocab_size"] < 1:
        raise ValueError("draft vocabulary is missing")
    if (
        type(rows_report.get("raw_target_logit_rows")) is not int
        or rows_report["raw_target_logit_rows"] < 1
    ):
        raise ValueError("raw target logits are missing")
    if (
        type(feature_report.get("selected_feature_rows")) is not int
        or feature_report["selected_feature_rows"] < 1
    ):
        raise ValueError("accepted-prefix feature rows are missing")
    for field, (source_name, _) in FILES.items():
        directory = rows_dir if field in {"rows", "anchors", "offsets", "t2d"} else features_dir
        source = directory / source_name
        expected = (
            row_outputs.get(source_name)
            if field in {"rows", "anchors", "offsets", "t2d"}
            else feature_report.get(
                "features_sha256" if field == "features" else "feature_rows_sha256"
            )
        )
        _hash_matches(source, expected, field)
    if target_logits.stat().st_size == 0:
        raise ValueError("raw target logits payload is empty")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        manifest = {
            "schema": "recurrent_binary_capture_v1",
            "split": "train",
            "prompts_sha256": prompt_hash,
            "prompt_count": expected_prompt_count,
            "target_vocab_size": rows_report["target_vocab_size"],
            "draft_vocab_size": rows_report["draft_vocab_size"],
            "max_depth": MAX_DEPTH,
            "raw_target_logits_retained": True,
            "raw_retirement_allowed": False,
            "training_eligible": False,
            "readiness": "preparation_only",
            "pinned_source_artifact_hashes_verified": True,
            "unverified_gates": [
                "native_model_execution_identity",
                "native_target_feature_numeric_parity",
                "full_drafter_mask_position_and_kv_parity",
                (
                    "initial_sample_terminal_emission_and_request_completeness"
                    if continuity_report is not None
                    else "cross_round_acceptance_ancestry"
                ),
            ],
            "source_report_sha256": {
                "rows": sha256(rows_dir / "preparation.json"),
                "features": sha256(features_dir / "report.json"),
                "cell_manifest": cell_hash,
            },
        }
        for field, (source_name, dest_name) in FILES.items():
            directory = rows_dir if field in {"rows", "anchors", "offsets", "t2d"} else features_dir
            shutil.copyfile(directory / source_name, stage / dest_name)
            manifest[field] = {"path": dest_name, "sha256": sha256(stage / dest_name)}
        for field, source, dest_name in (
            ("target_logits", target_logits, "target_logits.f32"),
            ("prompts", train_prompts, "train_prompts.jsonl"),
            ("cell_manifest", cell_manifest, "source_cell_manifest.json"),
            ("rows_report", rows_dir / "preparation.json", "rows_preparation.json"),
            ("features_report", features_dir / "report.json", "features_preparation.json"),
        ):
            if field == "target_logits" and raw_logits_storage == "hardlink":
                # os.link must fail on a cross-device boundary. Never fall back
                # to duplicating the raw payload for this explicit policy.
                os.link(source, stage / dest_name)
                source_stat, linked_stat = source.stat(), (stage / dest_name).stat()
                if (source_stat.st_dev, source_stat.st_ino) != (
                    linked_stat.st_dev, linked_stat.st_ino
                ):
                    raise ValueError("raw hardlink does not share the retained source inode")
                manifest["raw_target_logits_storage"] = "hardlink_shared_retained_source_inode"
            else:
                shutil.copyfile(source, stage / dest_name)
            if field == "target_logits":
                manifest[field] = {"path": dest_name, "sha256": sha256(stage / dest_name)}
        if shard_manifest_path is not None:
            shutil.copyfile(shard_manifest_path, stage / "source_shard_manifest.json")
            manifest["shard_manifest"] = {
                "path": "source_shard_manifest.json",
                "sha256": sha256(stage / "source_shard_manifest.json"),
            }
        if continuity_report is not None:
            shutil.copyfile(continuity_report, stage / "internal_continuity.json")
            manifest["internal_continuity"] = {
                "path": "internal_continuity.json",
                "sha256": sha256(stage / "internal_continuity.json"),
                "status": "internal_accepted_prefix_continuity_verified",
            }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        audit = audit_capture(
            stage / "manifest.json",
            stage / "train_prompts.jsonl",
            expected_prompt_hash,
            expected_prompt_count=expected_prompt_count,
        )
        if audit["raw_target_logit_rows"] != rows_report["raw_target_logit_rows"]:
            raise ValueError("audited target-logit row count disagrees with row preparer")
        if audit["feature_ledger"]["rows"] != feature_report["selected_feature_rows"]:
            raise ValueError("audited feature row count disagrees with feature preparer")
        audited_trace_counts = dict(audit["counts"])
        audited_trace_counts.pop("logit_rows", None)
        if audited_trace_counts != rows_report["trace_counts"]:
            raise ValueError("audited trace counts disagree with row preparer")
        (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.rename(stage, output_dir)
        return {"manifest": output_dir / "manifest.json", "audit": audit}
    except BaseException:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--target-logits", type=Path, required=True)
    parser.add_argument("--cell-manifest", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--expected-prompt-count", type=int)
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument("--continuity-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_hash, expected_count = resolve_prompt_expectation(
        args.expected_prompt_sha256, args.expected_prompt_count
    )
    result = build_bundle(
        args.rows_dir,
        args.features_dir,
        args.target_logits,
        args.cell_manifest,
        args.train_prompts,
        args.output_dir,
        continuity_report=args.continuity_report,
        shard_manifest_path=args.shard_manifest,
        expected_prompt_hash=expected_hash,
        expected_prompt_count=expected_count,
    )
    print(json.dumps({"manifest": str(result["manifest"]), "training_eligible": False}))


if __name__ == "__main__":
    main()
