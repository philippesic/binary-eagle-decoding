#!/usr/bin/env python3
"""Plan bounded W1Ax native-capture shards from a frozen training split.

CPU-only metadata work. The plan does not run capture or delete raw logits.
``observe`` verifies an instrumented native cell against one shard's hard
row/byte caps after capture, before any bundle/teacher preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from prepare_w1a_data import canonical, sha256

DEFAULT_TARGET_VOCAB = 151_936
DEFAULT_MAX_PROMPTS = 96
DEFAULT_MAX_ROWS = 16_384
DEFAULT_MAX_RAW_BYTES = 12 * 1024**3
DEFAULT_BASE_ROWS = 512
DEFAULT_ROWS_PER_WORD = 1.0


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def _load_training(manifest_path: Path, split: str) -> tuple[dict, list[dict]]:
    if split not in {"train_small", "train_large"}:
        raise ValueError("only frozen training splits may be sharded")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "w1a_data_manifest_v1":
        raise ValueError("wrong parent data manifest schema")
    spec = manifest.get("files", {}).get(split)
    if not isinstance(spec, dict):
        raise ValueError("training split missing from parent manifest")
    prompts = manifest_path.parent / spec["prompts"]
    index = manifest_path.parent / spec["index"]
    if (sha256(prompts) != spec["prompts_sha256"]
            or sha256(index) != spec["index_sha256"]):
        raise ValueError("frozen training prompt/index hash mismatch")
    if prompts.name.lower().find("final") >= 0 or index.name.lower().find("final") >= 0:
        raise ValueError("reserved final path prohibited")
    prompt_lines = [line for line in prompts.read_bytes().splitlines(keepends=True) if line.strip()]
    index_rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if len(prompt_lines) != spec["prompts_count"] or len(index_rows) != len(prompt_lines):
        raise ValueError("training prompt/index count mismatch")
    rows = []
    seen = set()
    for position, (line, meta) in enumerate(zip(prompt_lines, index_rows, strict=True)):
        prompt = json.loads(line)
        rid = prompt.get("id")
        if (not isinstance(rid, str) or not rid or "final" in rid.lower()
                or rid in seen or meta.get("id") != rid
                or not isinstance(prompt.get("messages"), list)
                or hashlib.sha256(canonical(prompt["messages"])).hexdigest()
                != meta.get("content_sha256")
                or not isinstance(meta.get("word_count"), int)
                or meta["word_count"] < 0
                or not isinstance(meta.get("group"), str)
                or not meta["group"]):
            raise ValueError(f"invalid frozen training row at position {position}")
        seen.add(rid)
        rows.append({"id": rid, "position": position, "line": line,
                     "group": meta["group"], "topic": meta.get("topic"),
                     "content_sha256": meta["content_sha256"],
                     "word_count": meta["word_count"]})
    source = {"manifest_sha256": sha256(manifest_path), "split": split,
              "prompts_filename": spec["prompts"],
              "prompts_sha256": spec["prompts_sha256"],
              "index_filename": spec["index"],
              "index_sha256": spec["index_sha256"], "count": len(rows)}
    return source, rows


def _families(rows: list[dict]) -> list[list[dict]]:
    """Keep declared conversation/topic connected components within one shard."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lookup = {}
    for i, row in enumerate(rows):
        for key in (row["group"], row["topic"]):
            if key is None:
                continue
            if key in lookup:
                parent[find(i)] = find(lookup[key])
            else:
                lookup[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return sorted(groups.values(), key=lambda group: min(row["position"] for row in group))


def _estimated_rows(row: dict, base: int, rows_per_word: float) -> int:
    return max(base, math.ceil(row["word_count"] * rows_per_word))


def plan_shards(
    manifest_path: Path, output: Path, *, split: str = "train_small",
    target_vocab: int = DEFAULT_TARGET_VOCAB,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
    base_rows_per_prompt: int = DEFAULT_BASE_ROWS,
    rows_per_word: float = DEFAULT_ROWS_PER_WORD,
) -> dict:
    if output.exists():
        raise ValueError("output must be new; shard plans are immutable")
    if (target_vocab <= 0 or not 1 <= max_prompts <= 96 or max_rows <= 0
            or max_raw_bytes <= 0 or base_rows_per_prompt <= 0
            or not math.isfinite(rows_per_word) or rows_per_word < 0):
        raise ValueError("invalid shard capacity or estimator")
    source, rows = _load_training(manifest_path, split)
    bytes_per_row = 4 * target_vocab
    if bytes_per_row > max_raw_bytes:
        raise ValueError("one F32 target-logit row exceeds raw-byte cap")
    capacity = min(max_rows, max_raw_bytes // bytes_per_row)
    if capacity < 1:
        raise ValueError("raw-byte cap admits no target-logit row")
    families = _families(rows)
    batches, current, estimated = [], [], 0
    for family in families:
        family_rows = sum(
            _estimated_rows(row, base_rows_per_prompt, rows_per_word) for row in family
        )
        if len(family) > max_prompts or family_rows > capacity:
            raise ValueError("one conversation/topic family exceeds prompt or raw-row cap")
        if current and (len(current) + len(family) > max_prompts
                        or estimated + family_rows > capacity):
            batches.append((current, estimated))
            current, estimated = [], 0
        current.extend(family)
        estimated += family_rows
    if current:
        batches.append((current, estimated))
    if not batches:
        raise ValueError("training split has no prompts")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shards = []
        for ordinal, (batch, estimate) in enumerate(batches):
            directory = stage / f"shard-{ordinal:04d}"
            directory.mkdir()
            prompt_path = directory / "train_prompts.jsonl"
            # Preserve parent source lines exactly, ordered by original train position.
            ordered = sorted(batch, key=lambda row: row["position"])
            prompt_path.write_bytes(b"".join(row["line"] for row in ordered))
            shard = {
                "schema": "w1ax_capture_shard_v1", "ordinal": ordinal,
                "parent": source, "prompt_count": len(ordered),
                "prompt_ids": [row["id"] for row in ordered],
                "source_positions": [row["position"] for row in ordered],
                "content_sha256": [row["content_sha256"] for row in ordered],
                "prompts_path": prompt_path.name, "prompts_sha256": sha256(prompt_path),
                "estimated_verifier_logit_rows": estimate,
                "estimated_raw_logit_bytes": estimate * bytes_per_row,
                "caps": {"max_prompts": max_prompts, "max_verifier_logit_rows": max_rows,
                         "max_raw_logit_bytes": max_raw_bytes},
                "target_vocab_size": target_vocab, "bytes_per_raw_logit_row": bytes_per_row,
                "estimator": {"base_rows_per_prompt": base_rows_per_prompt,
                              "rows_per_word": rows_per_word,
                              "basis": "planning_only; observed capture must pass hard caps"},
                "raw_retirement_allowed": False,
            }
            manifest = directory / "manifest.json"
            _write_json(manifest, shard)
            shards.append({"ordinal": ordinal, "manifest": f"shard-{ordinal:04d}/manifest.json",
                           "manifest_sha256": sha256(manifest),
                           "prompts_sha256": shard["prompts_sha256"],
                           "prompt_ids": shard["prompt_ids"],
                           "prompt_ids_sha256": hashlib.sha256(
                               canonical(shard["prompt_ids"])
                           ).hexdigest(),
                           "prompt_count": len(ordered),
                           "estimated_verifier_logit_rows": estimate,
                           "estimated_raw_logit_bytes": estimate * bytes_per_row})
        coverage = sorted((row["position"], row["id"]) for batch, _ in batches for row in batch)
        if coverage != [(row["position"], row["id"]) for row in rows]:
            raise ValueError("shards omit, duplicate or reorder frozen training coverage")
        plan = {"schema": "w1ax_capture_shard_plan_v1", "parent": source,
                "target_vocab_size": target_vocab, "bytes_per_raw_logit_row": bytes_per_row,
                "caps": {"max_prompts": max_prompts, "max_verifier_logit_rows": max_rows,
                         "max_raw_logit_bytes": max_raw_bytes},
                "estimator": {"base_rows_per_prompt": base_rows_per_prompt,
                              "rows_per_word": rows_per_word},
                "shard_count": len(shards), "total_prompts": len(rows),
                "parent_prompt_ids": [row["id"] for row in rows],
                "parent_prompt_ids_sha256": hashlib.sha256(
                    canonical([row["id"] for row in rows])
                ).hexdigest(),
                "total_estimated_verifier_logit_rows": sum(s["estimated_verifier_logit_rows"]
                                                          for s in shards),
                "total_estimated_raw_logit_bytes": sum(s["estimated_raw_logit_bytes"]
                                                      for s in shards),
                "coverage_sha256": hashlib.sha256(canonical(coverage)).hexdigest(),
                "shards": shards,
                "capture_status": "blocked_until_shard_capture_schema_and_audits_extend",
                "required_extensions": [
                    "native_runner_accepts_parent_bound_shard_manifest_and_variable_train_count",
                    "row_and_feature_preparers_accept_shard_hash_and_count_at_cli",
                    "bundle_builder_and_audit_accept_shard_hash_and_count_at_cli",
                    "audited_rows_to_compact_teacher_mapping_pins_exact_prefix_and_label",
                    "raw_retirement_evidence_schema_and_reaudit_before_any_deletion",
                ],
                "raw_retirement_allowed": False}
        _write_json(stage / "plan.json", plan)
        os.rename(stage, output)
        return plan
    except BaseException:
        shutil.rmtree(stage)
        raise


def observe_cell(shard_manifest: Path, cell_manifest: Path, output: Path) -> dict:
    """Verify one completed native cell against the planned hard caps.

    This is a pre-bundle metadata/hash guard. It does not certify training
    eligibility and never retires or modifies the raw logits.
    """
    if output.exists():
        raise ValueError("observation output must be new")
    shard = json.loads(shard_manifest.read_text())
    cell = json.loads(cell_manifest.read_text())
    if shard.get("schema") != "w1ax_capture_shard_v1":
        raise ValueError("wrong shard schema")
    prompt_path = shard_manifest.parent / shard["prompts_path"]
    if sha256(prompt_path) != shard["prompts_sha256"]:
        raise ValueError("shard prompt SHA256 mismatch")
    if (cell.get("schema") != "binary_head_capture_cell_v1"
            or cell.get("complete") is not True
            or cell.get("prompts_sha256") != shard["prompts_sha256"]):
        raise ValueError("native cell is incomplete or belongs to another shard")
    task_map = cell.get("task_prompt_ids")
    requests = cell.get("requests")
    if (not isinstance(task_map, dict) or not isinstance(requests, list)
            or len(requests) != shard["prompt_count"]
            or sorted(task_map.values()) != sorted(shard["prompt_ids"])
            or [request.get("id") for request in requests] != shard["prompt_ids"]
            or any(task_map.get(request.get("task_id")) != request.get("id")
                   for request in requests)):
        raise ValueError("native cell request/task ownership differs from shard")
    raw_path = cell_manifest.parent / "heads.target_logits.f32"
    record = cell.get("files", {}).get(raw_path.name)
    if (not isinstance(record, dict) or not raw_path.is_file()
            or record.get("bytes") != raw_path.stat().st_size
            or record.get("sha256") != sha256(raw_path)):
        raise ValueError("native raw logits file/hash differs from cell manifest")
    size = raw_path.stat().st_size
    row_bytes = shard["bytes_per_raw_logit_row"]
    if size == 0 or size % row_bytes:
        raise ValueError("native raw logits have partial or zero rows")
    rows = size // row_bytes
    if (rows > shard["caps"]["max_verifier_logit_rows"]
            or size > shard["caps"]["max_raw_logit_bytes"]
            or shard["prompt_count"] > shard["caps"]["max_prompts"]):
        raise ValueError("observed native raw logits exceed shard hard cap")
    ranges = [request.get("target_logit_rows") for request in requests]
    if (any(not isinstance(pair, list) or len(pair) != 2
            or any(type(value) is not int for value in pair) for pair in ranges)
            or ranges != sorted(ranges, key=lambda pair: pair[0])):
        raise ValueError("native cell lacks ordered request logit ranges")
    cursor = 0
    for start, stop in ranges:
        if start != cursor or stop <= start:
            raise ValueError("native cell logit ranges have a gap or overlap")
        cursor = stop
    if cursor != rows:
        raise ValueError("native cell logit ranges do not cover raw payload")
    report = {"schema": "w1ax_capture_shard_observation_v1",
              "shard_manifest_sha256": sha256(shard_manifest),
              "cell_manifest_sha256": sha256(cell_manifest),
              "raw_target_logits_sha256": record["sha256"],
              "observed_verifier_logit_rows": rows, "observed_raw_logit_bytes": size,
              "caps": shard["caps"], "status": "pre_bundle_hash_and_cap_verified",
              "raw_retirement_allowed": False}
    _write_json(output, report)
    return report


def emit_teacher_rows(shard_manifest: Path, bundle_manifest: Path, output: Path) -> dict:
    """Map audited bundle target-logit rows to their exact proposal prefixes.

    The bundle's independent audit is still required before this conversion.
    This adapter checks source hashes and ancestry fields; it does not promote a
    preparation-only bundle into a training-eligible one.
    """
    if output.exists():
        raise ValueError("teacher row output must be new")
    shard = json.loads(shard_manifest.read_text())
    bundle = json.loads(bundle_manifest.read_text())
    if (shard.get("schema") != "w1ax_capture_shard_v1"
            or bundle.get("schema") != "recurrent_binary_capture_v1"
            or bundle.get("prompts_sha256") != shard["prompts_sha256"]
            or bundle.get("target_vocab_size") != shard["target_vocab_size"]):
        raise ValueError("bundle does not belong to this capture shard")
    root = bundle_manifest.parent.resolve()

    def owned_record(name: str) -> Path:
        record = bundle.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"bundle missing {name} record")
        path = (root / record["path"]).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or sha256(path) != record.get("sha256")):
            raise ValueError(f"bundle {name} file/hash mismatch")
        return path

    rows_path = owned_record("rows")
    logits_path = owned_record("target_logits")
    bundle_hash = sha256(bundle_manifest)
    audit_path = root / "audit.json"
    if not audit_path.is_file():
        raise ValueError("bundle audit report is missing")
    audit = json.loads(audit_path.read_text())
    sources = audit.get("source_sha256")
    if (audit.get("schema") != "recurrent_binary_capture_audit_v1"
            or audit.get("capture_manifest_sha256") != bundle_hash
            or audit.get("training_prompts_sha256") != shard["prompts_sha256"]
            or not isinstance(sources, dict)
            or sources.get("rows") != sha256(rows_path)
            or sources.get("target_logits") != sha256(logits_path)):
        raise ValueError("bundle audit report does not bind row/logit sources")
    raw_bytes = logits_path.stat().st_size
    row_bytes = shard["bytes_per_raw_logit_row"]
    if (raw_bytes == 0 or raw_bytes % row_bytes
            or raw_bytes > shard["caps"]["max_raw_logit_bytes"]):
        raise ValueError("bundle raw logits are partial or exceed shard byte cap")
    count = raw_bytes // row_bytes
    if count > shard["caps"]["max_verifier_logit_rows"]:
        raise ValueError("bundle raw logits exceed shard row cap")
    if audit.get("raw_target_logit_rows") != count:
        raise ValueError("bundle audit raw-logit row count mismatch")
    allowed = set(shard["prompt_ids"])
    mapped = {}
    with rows_path.open() as stream:
        for state_row, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            number = row.get("target_logits_row")
            if number is None:
                continue
            prefix = row.get("prefix_token_ids")
            label = row.get("verifier_token_id")
            if (type(number) is not int or not 0 <= number < count or number in mapped
                    or row.get("prompt_id") not in allowed
                    or row.get("split") != "train"
                    or row.get("valid") is not True
                    or row.get("alignment_valid") is not True
                    or row.get("forced") is not False
                    or row.get("target_logits_source")
                    != "raw_target_verifier_at_exact_proposal_prefix"
                    or row.get("label_source")
                    != "cloned_native_verifier_sampler_at_actual_proposal_prefix"
                    or not isinstance(prefix, list) or not prefix
                    or any(type(token) is not int or token < 0
                           or token >= shard["target_vocab_size"] for token in prefix)
                    or type(label) is not int or not 0 <= label < shard["target_vocab_size"]):
                raise ValueError("bundle target-logit row lacks exact prefix/label ancestry")
            mapped[number] = {"id": f"{bundle_hash[:16]}:{number}",
                              "prompt_id": row["prompt_id"],
                              "capture_id": bundle_hash,
                              "logits_row": number,
                              "prefix_token_ids": prefix,
                              "next_target_id": label,
                              "source_state_row": state_row}
    if set(mapped) != set(range(count)):
        raise ValueError("bundle target logits have missing row/prefix joins")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for number in range(count):
            stream.write(canonical(mapped[number]) + b"\n")
    return {"rows": count, "rows_sha256": sha256(output),
            "bundle_manifest_sha256": bundle_hash,
            "bundle_audit_sha256": sha256(audit_path),
            "target_logits_sha256": sha256(logits_path),
            "training_eligible": bundle.get("training_eligible") is True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--parent-manifest", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--split", choices=("train_small", "train_large"),
                          default="train_small")
    planning.add_argument("--target-vocab", type=int, default=DEFAULT_TARGET_VOCAB)
    planning.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    planning.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    planning.add_argument("--max-raw-bytes", type=int, default=DEFAULT_MAX_RAW_BYTES)
    planning.add_argument("--base-rows-per-prompt", type=int, default=DEFAULT_BASE_ROWS)
    planning.add_argument("--rows-per-word", type=float, default=DEFAULT_ROWS_PER_WORD)
    observed = sub.add_parser("observe")
    observed.add_argument("--shard-manifest", type=Path, required=True)
    observed.add_argument("--cell-manifest", type=Path, required=True)
    observed.add_argument("--output", type=Path, required=True)
    teacher = sub.add_parser("teacher-rows")
    teacher.add_argument("--shard-manifest", type=Path, required=True)
    teacher.add_argument("--bundle-manifest", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_shards(
            args.parent_manifest, args.output, split=args.split,
            target_vocab=args.target_vocab, max_prompts=args.max_prompts,
            max_rows=args.max_rows, max_raw_bytes=args.max_raw_bytes,
            base_rows_per_prompt=args.base_rows_per_prompt,
            rows_per_word=args.rows_per_word,
        )
        print(json.dumps({"plan": str(args.output / "plan.json"),
                          "sha256": sha256(args.output / "plan.json"),
                          "shards": result["shard_count"],
                          "estimated_raw_bytes": result["total_estimated_raw_logit_bytes"]}))
    elif args.command == "observe":
        result = observe_cell(args.shard_manifest, args.cell_manifest, args.output)
        print(json.dumps(result, sort_keys=True))
    else:
        result = emit_teacher_rows(args.shard_manifest, args.bundle_manifest, args.output)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
