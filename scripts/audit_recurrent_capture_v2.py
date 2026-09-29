#!/usr/bin/env python3
"""Re-audit label-only recurrent captures without reading raw vocabulary logits.

This proves exact-prefix cloned-sampler label provenance and native response
continuity from retained metadata. It does not recompute the native sampler or
raw target probabilities. V1 audit/provider behavior is deliberately unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from audit_recurrent_binary_capture import (
    AuditedCapture,
    audit_feature_ledger,
    read_jsonl,
    sha256,
    validate_shard_manifest,
)
from audit_recurrent_response import audit_response
from prepare_recurrent_native_rows import prepare

from w1a1_eagle.recurrent_trace import RoundAnchor, validate_recurrent_trace

SCHEMA = "recurrent_binary_capture_v2"
POLICY = {
    "schema": "recurrent_capture_storage_policy_v2",
    "teacher_storage": "cloned_sampler_labels_only",
    "objective": "hard_ce",
    "probabilities": "absent_cannot_recompute_raw_probabilities_or_mapped_mass",
    "label_contract": "cloned_native_verifier_sampler_at_actual_proposal_prefix",
    "prefix_contract": "teacher_forced_exact_captured_prefix_only",
    "raw_target_logits_retained_in_bundle": False,
    "raw_retirement_allowed": False,
    "source_raw_artifacts": "preserve_v1_unchanged",
}
PREPARED_FIELDS = ("rows", "anchors", "offsets", "t2d", "features", "feature_rows")
RAW_JOIN_FIELDS = ("target_logits_row", "target_logits_dim", "target_logits_source")


def _file(directory: Path, record: object, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"invalid {label} file record")
    name = record["path"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("v2 paths must be basenames beside the manifest")
    path = directory / name
    if path.is_symlink() or not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"{label} file missing or SHA256 mismatch")
    return path


def _without_raw_join(row: dict) -> dict:
    return {**row, **{field: None for field in RAW_JOIN_FIELDS}}


def audit_capture_v2(
    manifest_path: Path, *, expected_prompt_sha256: str, expected_prompt_count: int
) -> dict:
    """Independently check immutable inputs and replay every metadata join."""
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if not isinstance(m, dict) or m.get("schema") != SCHEMA or m.get("split") != "train":
        raise ValueError("unsupported train capture v2 schema")
    if (
        m.get("storage_policy") != POLICY
        or m.get("training_eligible") is not False
        or m.get("readiness") != "preparation_only"
        or m.get("raw_retirement_allowed") is not False
        or "target_logits" in m
    ):
        raise ValueError("v2 storage/eligibility policy differs; raw retirement is forbidden")
    if (
        type(expected_prompt_count) is not int
        or expected_prompt_count < 1
        or m.get("prompts_sha256") != expected_prompt_sha256
        or m.get("prompt_count") != expected_prompt_count
    ):
        raise ValueError("v2 frozen prompt expectation differs")
    directory = manifest_path.parent
    fields = (
        *PREPARED_FIELDS,
        "prompts",
        "source_v1_manifest",
        "source_v1_audit",
        "source_cell",
        "source_capture",
        "source_rows_report",
        "source_features_report",
        "native_heads",
        "native_rounds",
        "native_round_trace",
        "native_feature_events",
        "task_map",
        "absolute_d2t",
    )
    files = {field: _file(directory, m.get(field), field) for field in fields}
    origin = json.loads(files["source_v1_manifest"].read_text())
    original_audit = json.loads(files["source_v1_audit"].read_text())
    if (
        origin.get("schema") != "recurrent_binary_capture_v1"
        or original_audit.get("schema") != "recurrent_binary_capture_audit_v1"
        or original_audit.get("capture_manifest_sha256") != sha256(files["source_v1_manifest"])
        or original_audit.get("training_prompts_sha256") != expected_prompt_sha256
        or original_audit.get("training_prompt_count") != expected_prompt_count
    ):
        raise ValueError("v2 original v1 manifest/audit ancestry differs")
    for field in (*PREPARED_FIELDS, "shard_manifest"):
        if field in origin:
            path = files[field] if field in files else _file(directory, m.get(field), field)
            files[field] = path
            if sha256(path) != origin[field]["sha256"]:
                raise ValueError(f"v2 changed original {field}")
        elif field in m:
            raise ValueError(f"v2 unexpected {field}")
    if (
        origin.get("prompts_sha256") != expected_prompt_sha256
        or origin.get("split") != "train"
        or any(
            m.get(field) != origin.get(field)
            for field in ("target_vocab_size", "draft_vocab_size", "max_depth", "unverified_gates")
        )
        or sha256(files["prompts"]) != expected_prompt_sha256
    ):
        raise ValueError("v2 prompt/geometry/readiness origin differs")
    prompts = read_jsonl(files["prompts"])
    ids = [row.get("id") for row in prompts]
    if (
        len(ids) != expected_prompt_count
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != expected_prompt_count
    ):
        raise ValueError("v2 frozen training prompt IDs differ")
    shard = None
    if "shard_manifest" in files:
        # The original shard basename describes the original prompt filename.
        shard = json.loads(files["shard_manifest"].read_text())
        if shard.get("prompts_path") != files["prompts"].name:
            raise ValueError("v2 shard prompt filename differs")
        shard = validate_shard_manifest(
            files["shard_manifest"],
            files["prompts"],
            expected_prompt_sha256,
            expected_prompt_count,
            ids,
        )

    cell = json.loads(files["source_cell"].read_text())
    capture = json.loads(files["source_capture"].read_text())
    rp = json.loads(files["source_rows_report"].read_text())
    fp = json.loads(files["source_features_report"].read_text())
    source_hashes = origin.get("source_report_sha256", {})
    for key, field in (
        ("cell_manifest", "source_cell"),
        ("rows", "source_rows_report"),
        ("features", "source_features_report"),
    ):
        if source_hashes.get(key) != sha256(files[field]):
            raise ValueError("v2 source reports differ from original bundle")
    row_sources, feature_sources = rp.get("source_sha256", {}), fp.get("sources", {})
    for key, field in (
        ("heads", "native_heads"),
        ("rounds", "native_rounds"),
        ("task_map", "task_map"),
    ):
        if row_sources.get(key) != sha256(files[field]):
            raise ValueError(f"v2 native {key} ancestry differs")
    absolute = np.load(files["absolute_d2t"], allow_pickle=False)
    if (
        absolute.ndim != 1
        or absolute.dtype.kind not in "iu"
        or hashlib.sha256(np.asarray(absolute, dtype="<i8").tobytes()).hexdigest()
        != rp.get("absolute_map_raw_sha256")
    ):
        raise ValueError("v2 absolute d2t raw identity differs")
    omitted = m.get("omitted_raw_logits")
    if (
        not isinstance(omitted, dict)
        or type(omitted.get("rows")) is not int
        or omitted["rows"] < 1
        or type(omitted.get("bytes")) is not int
        or omitted["bytes"] != omitted["rows"] * m["target_vocab_size"] * 4
        or omitted.get("disposition") != "not_copied_source_v1_preserved"
    ):
        raise ValueError("v2 omitted raw-logit record differs")
    if shard is not None:
        caps = shard.get("caps", {})
        if (
            shard.get("target_vocab_size") != m["target_vocab_size"]
            or shard.get("bytes_per_raw_logit_row") != m["target_vocab_size"] * 4
            or any(
                type(caps.get(k)) is not int
                for k in ("max_prompts", "max_verifier_logit_rows", "max_raw_logit_bytes")
            )
            or expected_prompt_count > caps["max_prompts"]
            or omitted["rows"] > caps["max_verifier_logit_rows"]
            or omitted["bytes"] > caps["max_raw_logit_bytes"]
        ):
            raise ValueError("v2 historical raw capture exceeds frozen shard caps")
    if (
        feature_sources.get("metadata_sha256") != sha256(files["native_feature_events"])
        or feature_sources.get("task_map_sha256") != sha256(files["task_map"])
        or row_sources.get("raw_target_logits") != origin.get("target_logits", {}).get("sha256")
        or origin.get("target_logits", {}).get("sha256")
        != m.get("omitted_raw_logits", {}).get("sha256")
        or original_audit.get("raw_target_logit_rows")
        != m.get("omitted_raw_logits", {}).get("rows")
    ):
        raise ValueError("v2 feature/raw-logit ancestry differs")
    if (
        capture.get("schema") != "recurrent_binary_native_capture_v1"
        or capture.get("complete") is not True
        or capture.get("train_prompts_sha256") != expected_prompt_sha256
        or capture.get("train_prompt_count") != expected_prompt_count
        or cell.get("complete") is not True
        or cell.get("prompts_sha256") != expected_prompt_sha256
    ):
        raise ValueError("v2 source capture is incomplete or has wrong train ownership")
    # Source file records still describe original basenames, preserved on copy.
    for field in ("native_heads", "native_rounds", "native_feature_events"):
        path = files[field]
        record = cell.get("files", {}).get(path.name, {})
        if record.get("sha256") != sha256(path) or record.get("bytes") != path.stat().st_size:
            raise ValueError(f"v2 cell {field} source hash/size differs")
    trace_record = cell.get("files", {}).get(files["native_round_trace"].name)
    if trace_record is not None and (
        trace_record.get("sha256") != sha256(files["native_round_trace"])
        or trace_record.get("bytes") != files["native_round_trace"].stat().st_size
    ):
        raise ValueError("v2 cell round trace hash/size differs")
    if (
        capture.get("cell_manifest_sha256") != sha256(files["source_cell"])
        or capture.get("task_prompt_ids_sha256") != sha256(files["task_map"])
        or capture.get("target_vocab_size") != m["target_vocab_size"]
        or capture.get("d2t_raw_sha256") != rp.get("absolute_map_raw_sha256")
        or capture.get("d2t_sha256") != row_sources.get("absolute_d2t")
        or capture.get("target_sha256") != cell.get("target_sha256")
        or capture.get("draft_sha256") != cell.get("draft_sha256")
        or capture.get("requests") != cell.get("requests")
    ):
        raise ValueError("v2 capture manifest native source identity differs")
    task_map = json.loads(files["task_map"].read_text())
    if (
        not isinstance(task_map, dict)
        or len(task_map) != expected_prompt_count
        or set(task_map.values()) != set(ids)
        or cell.get("task_prompt_ids") != task_map
    ):
        raise ValueError("v2 task map must own every frozen training prompt exactly once")

    # Re-run the independent native preparer with only its raw-logit requirement
    # removed. Derivation is local temporary data, never a rewritten v1 source.
    with tempfile.TemporaryDirectory(prefix="recurrent-v2-audit-") as scratch:
        scratch = Path(scratch)
        heads = scratch / files["native_heads"].name
        heads.write_text(
            "".join(
                json.dumps(_without_raw_join(r)) + "\n" for r in read_jsonl(files["native_heads"])
            )
        )
        derived_cell = json.loads(json.dumps(cell))
        derived_cell["files"][heads.name] = {"sha256": sha256(heads), "bytes": heads.stat().st_size}
        derived_cell_path = scratch / "cell.json"
        derived_cell_path.write_text(json.dumps(derived_cell))
        derived = prepare(
            heads_path=heads,
            rounds_path=files["native_rounds"],
            task_map_path=files["task_map"],
            prompts_path=files["prompts"],
            absolute_map_path=files["absolute_d2t"],
            cell_manifest_path=derived_cell_path,
            offset_path=files["offsets"],
            t2d_path=files["t2d"],
            target_vocab_size=m["target_vocab_size"],
            expected_prompt_hash=expected_prompt_sha256,
            expected_prompt_count=expected_prompt_count,
        )
    rows, anchors, _, _, _ = derived
    saved_rows, saved_anchors = read_jsonl(files["rows"]), read_jsonl(files["anchors"])
    if rows != [_without_raw_join(r) for r in saved_rows] or anchors != saved_anchors:
        raise ValueError("v2 sampled labels/prefixes differ from native metadata derivation")
    # Preserve and validate historical raw joins as lineage, never as payloads.
    heads = read_jsonl(files["native_heads"])
    raw_indices = []
    for row, head in zip(saved_rows, heads, strict=True):
        if any(row.get(f) != head.get(f) for f in RAW_JOIN_FIELDS):
            raise ValueError("v2 historical raw-logit joins differ")
        index = row.get("target_logits_row")
        if index is not None:
            if (
                type(index) is not int
                or index < 0
                or row.get("target_logits_dim") != m["target_vocab_size"]
                or row.get("target_logits_source") != "raw_target_verifier_at_exact_proposal_prefix"
            ):
                raise ValueError("v2 historical raw-logit row/source differs")
            raw_indices.append(index)
    if sorted(raw_indices) != list(range(m["omitted_raw_logits"]["rows"])):
        raise ValueError("v2 historical raw-logit row joins are incomplete")
    trace = validate_recurrent_trace(
        saved_rows,
        [RoundAnchor(**a) for a in anchors],
        offsets=np.load(files["offsets"], allow_pickle=False),
        target_vocab_size=m["target_vocab_size"],
        draft_vocab_size=m["draft_vocab_size"],
        max_depth=m["max_depth"],
        allowed_prompt_ids=set(ids),
        split="train",
    )
    ledger = audit_feature_ledger(
        files["features"],
        files["feature_rows"],
        [RoundAnchor(**a) for a in anchors],
        set(ids),
        m["target_vocab_size"],
    )
    requests = cell.get("requests")
    records = m.get("requests")
    if (
        not isinstance(requests, list)
        or not isinstance(records, list)
        or len(records) != len(requests)
    ):
        raise ValueError("v2 request evidence is incomplete")
    reports = []
    request_files = []
    for request, record in zip(requests, records, strict=True):
        if (
            not isinstance(record, dict)
            or not isinstance(request, dict)
            or record.get("task_id") != request.get("task_id")
            or record.get("id") != request.get("id")
        ):
            raise ValueError("v2 request ownership differs")
        paths = {
            key: _file(directory, record.get(key), key) for key in ("prompt", "request", "response")
        }
        request_files.extend(paths.values())
        for key, path in paths.items():
            if sha256(path) != request.get(f"{key}_sha256"):
                raise ValueError(f"v2 request {key} differs from native cell")
        response = json.loads(paths["response"].read_text())
        if response.get("__verbose", {}).get("tokens") != request.get("generated_token_ids"):
            raise ValueError("v2 cell generated IDs differ from raw response")
        reports.append(
            audit_response(
                files["native_rounds"],
                files["native_round_trace"],
                files["native_feature_events"],
                files["task_map"],
                paths["request"],
                paths["response"],
                record["task_id"],
            )
        )
    claimed = {p.name for p in (*files.values(), *request_files)} | {manifest_path.name}
    # The generated audit is descriptive output; all other files must be pinned
    # inputs. A stray/renamed raw logit copy must not hide outside the manifest.
    if (directory / "audit.json").exists():
        audit_output = directory / "audit.json"
        if audit_output.is_symlink() or not audit_output.is_file():
            raise ValueError("v2 audit output must be a regular JSON file")
        try:
            saved_audit = json.loads(audit_output.read_text())
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("v2 audit output must be metadata JSON") from error
        if (
            not isinstance(saved_audit, dict)
            or saved_audit.get("schema") != "recurrent_binary_capture_audit_v2"
        ):
            raise ValueError("v2 audit output must have the v2 audit schema")
        claimed.add("audit.json")
    if {p.name for p in directory.iterdir()} != claimed:
        raise ValueError("v2 directory contains unclaimed files; raw copies are forbidden")
    return {
        "schema": "recurrent_binary_capture_audit_v2",
        "execution_device": "cpu",
        "capture_manifest_sha256": sha256(manifest_path),
        "source_v1_manifest_sha256": sha256(files["source_v1_manifest"]),
        "source_sha256": {k: sha256(p) for k, p in files.items()},
        "training_prompts_sha256": expected_prompt_sha256,
        "training_prompt_count": expected_prompt_count,
        "counts": dict(trace.counts),
        "per_depth": trace.per_depth,
        "feature_ledger": ledger,
        "label_contract": POLICY["label_contract"],
        "response_requests": len(reports),
        "response_tokens": sum(r["response_tokens"] for r in reports),
        "raw_target_logits_in_bundle": False,
        "historical_raw_target_logit_rows": m["omitted_raw_logits"]["rows"],
        "mapped_probability_mass_mean_on_sampled_logit_rows": None,
        "raw_probability_recomputation": "unavailable_label_only_storage",
        "round_trace_source_binding": (
            "original_cell_hash"
            if trace_record is not None
            else "conversion_hash_and_canonical_emission_recheck"
        ),
        "initial_seed_sampler_parity": "unverified",
        "terminal_sampler_parity": "unverified_where_observed",
        "real_model_feature_and_kv_parity": "unverified",
        "training_eligible": False,
    }


def load_audited_capture_v2(
    manifest_path: Path, *, expected_prompt_sha256: str, expected_prompt_count: int
) -> AuditedCapture:
    """Explicit adapter for a future hard-CE provider; does not grant eligibility."""
    report = audit_capture_v2(
        manifest_path,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    )
    m = json.loads(Path(manifest_path).read_text())
    directory = Path(manifest_path).parent
    anchors = {
        (r["prompt_id"], r["round_index"]): RoundAnchor(**r)
        for r in read_jsonl(directory / m["anchors"]["path"])
    }
    rows = defaultdict(list)
    for row in read_jsonl(directory / m["rows"]["path"]):
        rows[(row["prompt_id"], row["round_index"])].append(row)
    feature_lookup = {
        (r["prompt_id"], tuple(r["prefix_token_ids"])): r["feature_row"]
        for r in read_jsonl(directory / m["feature_rows"]["path"])
    }
    return AuditedCapture(
        report,
        anchors,
        {k: tuple(v) for k, v in rows.items()},
        feature_lookup,
        np.load(directory / m["features"]["path"], mmap_mode="r", allow_pickle=False),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-prompt-sha256", required=True)
    p.add_argument("--expected-prompt-count", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("audit output must be new")
    result = audit_capture_v2(
        a.manifest,
        expected_prompt_sha256=a.expected_prompt_sha256,
        expected_prompt_count=a.expected_prompt_count,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
