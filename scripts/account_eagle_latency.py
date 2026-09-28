#!/usr/bin/env python3
"""Account archived EAGLE CPU-wall round and draft-stage traces.

The input is a round-analysis.json index produced by analyze_binary_rescue_rounds.
This is an accounting report, not a CUDA kernel profile or a speedup estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

STAGE_NAMES = (
    "draft", "target_decode_sync", "process", "check", "checkpoint",
    "kv_repair", "accept_hook",
)
PROCESS_PARTS = (
    "process_feature_copy_us", "process_encoder_us", "process_batch_build_us",
    "process_draft_decode_us",
)


def checked_file(entry: dict) -> Path:
    path = Path(entry["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def round_partition(row: dict) -> dict[str, int]:
    """Partition one round by interval union; overlapping stages stay unassigned."""
    start, end = row["round_start_us"], row["round_end_us"]
    if row["round_us"] != end - start or end <= start:
        raise ValueError("invalid round interval")
    spans = {}
    for name in STAGE_NAMES:
        a, b = row["spans_us"].get(name, [0, 0])
        if a == b:
            continue
        a, b = max(a, start), min(b, end)
        if b > a:
            spans[name] = (a, b)
    points = sorted({start, end, *(p for a, b in spans.values() for p in (a, b))})
    result = dict.fromkeys((*STAGE_NAMES, "shared_overlap", "unassigned"), 0)
    for a, b in zip(points, points[1:]):
        active = [name for name, (left, right) in spans.items() if left <= a and b <= right]
        label = active[0] if len(active) == 1 else "shared_overlap" if active else "unassigned"
        result[label] += b - a
    if sum(result.values()) != row["round_us"]:
        raise ValueError("round partition does not reconcile")
    return result


def checked_stage(row: dict) -> dict[str, int]:
    if row["schema"] != "eagle_draft_stage_v1" or row["clock"] != "ggml_time_us_cpu_wall":
        raise ValueError("unsupported draft stage trace")
    cursor = row["start_us"]
    totals: dict[str, int] = defaultdict(int)
    for span in row["spans"]:
        if span["start_us"] != cursor or span["end_us"] < cursor:
            raise ValueError("noncontiguous draft stages")
        duration = span["end_us"] - cursor
        if span["duration_us"] != duration:
            raise ValueError("draft stage duration mismatch")
        totals[span["stage"]] += duration
        cursor = span["end_us"]
    if (cursor != row["end_us"] or sum(totals.values()) != row["total_us"]
            or row["partition_us"] != row["total_us"] or row["unassigned_us"] != 0
            or totals != row["stage_totals_us"]):
        raise ValueError("draft stage partition mismatch")
    return totals


def analyze(index_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    if index.get("schema") != "binary_rescue_round_cpu_v1":
        raise ValueError("unsupported round analysis index")
    grouped: dict[str, dict] = defaultdict(lambda: {
        "rounds": 0, "replay_rounds": 0, "round_us": 0, "proposed": 0,
        "accepted": 0, "emitted": 0, "round_partition_us": defaultdict(int),
        "process_parts_us": defaultdict(int), "process_unassigned_us": 0,
        "draft_calls": 0, "draft_call_us": 0, "draft_stage_us": defaultdict(int),
        "draft_stage_calls_unmatched": 0, "matched_proposed": 0,
    })
    windows: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for item in index["round_files"]:
        variant = item["variant"]
        out = grouped[variant]
        for row in json.loads(checked_file(item).read_text()):
            if row["schema"] != "w1ax_eagle_round_v1" or row["clock"] != "ggml_time_us_cpu_wall":
                raise ValueError("unsupported round trace")
            out["rounds"] += 1
            out["replay_rounds"] += bool(row["replay"])
            out["round_us"] += row["round_us"]
            if not row["replay"]:
                out["proposed"] += row["n_proposed"]
                out["accepted"] += row["n_accepted"]
                out["emitted"] += row["n_emitted"]
            for name, us in round_partition(row).items():
                out["round_partition_us"][name] += us
            process_parts = sum(row.get(name, 0) for name in PROCESS_PARTS)
            process_us = row["process_us"]
            if process_parts > process_us:
                raise ValueError("process parts exceed parent span")
            for name in PROCESS_PARTS:
                out["process_parts_us"][name] += row.get(name, 0)
            out["process_unassigned_us"] += process_us - process_parts
            windows[variant].append((row["round_start_us"], row["round_end_us"], row["n_proposed"]))

    for item in index["draft_stage_files"]:
        variant = item["variant"]
        out = grouped[variant]
        intervals = windows[variant]
        for row in jsonl(checked_file(item)):
            totals = checked_stage(row)
            matches = [(a, b, n) for a, b, n in intervals
                       if a <= row["start_us"] and row["end_us"] <= b]
            if not matches:
                out["draft_stage_calls_unmatched"] += 1
                continue
            if len(matches) != 1:
                raise ValueError("draft call belongs to multiple round windows")
            out["draft_calls"] += 1
            out["draft_call_us"] += row["total_us"]
            out["matched_proposed"] += matches[0][2]
            for name, us in totals.items():
                out["draft_stage_us"][name] += us

    result = {}
    for variant, raw in sorted(grouped.items()):
        result[variant] = {key: dict(value) if isinstance(value, defaultdict) else value
                           for key, value in raw.items()}
        row = result[variant]
        if sum(row["round_partition_us"].values()) != row["round_us"]:
            raise ValueError("pooled round partition does not reconcile")
        if sum(row["draft_stage_us"].values()) != row["draft_call_us"]:
            raise ValueError("pooled draft partition does not reconcile")
        row["draft_call_us_per_matched_proposed"] = (
            row["draft_call_us"] / row["matched_proposed"] if row["matched_proposed"] else None
        )
        row["round_draft_us_per_proposed"] = (
            row["round_partition_us"].get("draft", 0) / row["proposed"] if row["proposed"] else None
        )
        row["round_draft_minus_traced_calls_us"] = (
            row["round_partition_us"].get("draft", 0) - row["draft_call_us"]
        )
    return {
        "schema": "eagle_latency_accounting_v1",
        "source_index": str(index_path.resolve()),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "clock": "ggml_time_us_cpu_wall",
        "scope": (
            "sum of archived measured round rows; shared batched spans can appear "
            "in multiple rows"
        ),
        "variants": result,
        "unavailable": {
            "projection_pack_dot_output": "no per-projection GPU event or launch attribution",
            "decoder_projection_attention_norm_state": (
                "decode call includes asynchronous work and synchronization"
            ),
            "full_request_prefill_decode": (
                "round files exclude prefill, HTTP, leading seed and inter-round gaps"
            ),
        },
        "rules": [
            "Draft stages nest inside draft calls and rounds; never add them to round totals.",
            "round_draft_minus_traced_calls_us retains call-boundary and unmatched-call time; "
            "it is not an optimization opportunity.",
            "Process parts nest inside process; process_unassigned_us preserves unmeasured work.",
            "Round partition uses disjoint interval union; shared_overlap and unassigned "
            "are retained.",
            "CPU waits may include GPU work. Do not combine with unrelated Nsight kernel sums "
            "or operator medians.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
