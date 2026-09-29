#!/usr/bin/env python3
"""Audit intrusive CUDA event spans without summing overlapping GPU scopes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PREFIXES = {
    "CUDA_EAGLE_EVENT ": "cuda_eagle_event_v1",
    "CUDA_EAGLE_GRAPH_NODE ": "cuda_eagle_graph_node_v1",
    "CUDA_EAGLE_GRAPH_INVENTORY ": "cuda_eagle_graph_inventory_v1",
}


def union_length(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative interval")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def parse(path: Path) -> tuple[list[dict], list[dict], bool]:
    events, inventories = [], []
    truncated = False
    seen = set()
    for line in path.read_text().splitlines():
        for prefix, schema in PREFIXES.items():
            if prefix not in line:
                continue
            row = json.loads(line.split(prefix, 1)[1])
            if row.get("schema") != schema:
                raise ValueError("unsupported CUDA event schema")
            row["source"] = str(path.resolve())
            if schema != "cuda_eagle_event_v1":
                if schema == "cuda_eagle_graph_node_v1" and row.get("cuda_ms") is not None:
                    raise ValueError("CUDA graph inventory must have null timing")
                inventories.append(row)
                if schema == "cuda_eagle_graph_inventory_v1":
                    if not 0 <= row["emitted_nodes"] <= row["total_nodes"]:
                        raise ValueError("invalid CUDA graph inventory counts")
                    truncated |= row["emitted_nodes"] < row["total_nodes"]
                break
            if row.get("kind") == "truncation":
                truncated = True
                break
            if row["id"] in seen:
                raise ValueError("duplicate event ID within one log")
            seen.add(row["id"])
            if row["host_end_us"] < row["host_begin_us"]:
                raise ValueError("negative host event interval")
            values = [row.get(key) for key in ("gpu_begin_ms", "gpu_end_ms", "cuda_ms")]
            if all(value is None for value in values):
                if not row.get("captured_inventory_only"):
                    raise ValueError("missing GPU timing without inventory marker")
            elif any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("partial or nonfinite GPU timing")
            elif values[1] < values[0] or abs(values[2] - (values[1] - values[0])) > 1e-4:
                raise ValueError("inconsistent GPU event interval")
            events.append(row)
            break
    return events, inventories, truncated


def category(row: dict) -> str:
    kind, op, name = row.get("kind", ""), row.get("op", ""), row.get("tensor", "").lower()
    if kind.startswith("transfer_"):
        return "buffer_transfers"
    if kind == "activation_pack_scales" or op == "W1AX_PACK":
        return "activation_pack_scales"
    if kind.startswith("dot_output"):
        return "dot_output_with_A16_cast" if "A16" in kind else "dot_output"
    if kind != "node":
        return "unassigned"
    if "NORM" in op or "norm" in name:
        return "norms"
    if "FLASH_ATTN" in op or name.startswith("kq") or "soft_max" in name:
        return "attention"
    if op == "SET_ROWS" and "cache_" in name:
        return "cache_writes"
    if op in ("MUL_MAT", "W1A1_MUL_MAT"):
        return "projection_combined_unsplit"
    return "unassigned"


def frame_accounting(rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in rows}
    missing_parents = sorted(
        {row["parent"] for row in rows if row["parent"] and row["parent"] not in by_id}
    )
    roots = [row for row in rows if row["id"] == row["frame"]]
    if len(roots) != 1:
        return {
            "gpu_span_ms": None,
            "reason": "missing or ambiguous frame root",
            "missing_parent_ids": missing_parents,
        }
    root = roots[0]
    if root.get("cuda_ms") is None:
        return {"gpu_span_ms": None, "reason": "frame is inventory only"}
    start, stop = root["gpu_begin_ms"], root["gpu_end_ms"]
    timed = [row for row in rows if row.get("cuda_ms") is not None]
    boundaries = {start, stop}
    clipped = []
    outside = 0
    for row in timed:
        a, b = row["gpu_begin_ms"], row["gpu_end_ms"]
        outside += int(a < start - 1e-4 or b > stop + 1e-4)
        if b < start or a > stop:
            continue
        a, b = max(start, a), min(stop, b)
        depth, parent, visited = 0, row["parent"], set()
        while parent in by_id:
            if parent in visited:
                raise ValueError("cycle in event parents")
            visited.add(parent)
            depth += 1
            parent = by_id[parent]["parent"]
        clipped.append((a, b, depth, row))
        boundaries.update((a, b))
    totals = defaultdict(float)
    overlaps = 0
    points = sorted(boundaries)
    for a, b in zip(points, points[1:]):
        active = [value for value in clipped if value[0] <= (a + b) / 2 < value[1]]
        if not active:
            totals["unassigned"] += b - a
            continue
        deepest = max(value[2] for value in active)
        selected = [value for value in active if value[2] == deepest]
        names = {category(value[3]) for value in selected}
        # Distinct simultaneous spans stay explicit; their durations are not added.
        if len(selected) > 1:
            overlaps += 1
            name = "overlapping_spans"
        else:
            name = next(iter(names))
        totals[name] += b - a
    return {
        "gpu_span_ms": stop - start,
        "missing_parent_ids": missing_parents,
        "components_ms": dict(totals),
        "unassigned_ms": totals.get("unassigned", 0.0),
        "overlap_segments": overlaps,
        "outside_root_intervals": outside,
        "inventory_only_records": sum(row.get("cuda_ms") is None for row in rows),
        "host_span_us": root["host_end_us"] - root["host_begin_us"],
        "stage": root.get("stage", "unassigned"),
        "graph_enabled": root.get("graph_enabled"),
        "graph_capture": root.get("graph_capture"),
    }


def request_accounting(events: list[dict], manifest: dict) -> list[dict]:
    output = []
    for request in manifest.get("records", []):
        start = request.get("client_request_begin_monotonic_s")
        stop = request.get("client_request_end_monotonic_s")
        clock = request.get("client_clock_implementation", "")
        if start is None or stop is None or "CLOCK_MONOTONIC" not in clock:
            output.append(
                {
                    "prompt_id": request.get("prompt_id"),
                    "host_unassigned_us": None,
                    "reason": "missing compatible same-host monotonic request boundaries",
                }
            )
            continue
        start, stop = start * 1e6, stop * 1e6
        cell = Path(request["directory"]).parent.name
        matched = [
            row
            for row in events
            if Path(row["source"]).parent.name == cell
            and row.get("host_clock") == "CLOCK_MONOTONIC"
            and row["host_end_us"] > start
            and row["host_begin_us"] < stop
        ]
        intervals = [
            (max(start, row["host_begin_us"]), min(stop, row["host_end_us"])) for row in matched
        ]
        assigned = union_length(intervals)
        output.append(
            {
                "prompt_id": request.get("prompt_id"),
                "variant": request.get("variant"),
                "repetition": request.get("repetition"),
                "warmup": request.get("warmup"),
                "host_request_us": stop - start,
                "instrumented_host_union_us": assigned,
                "host_unassigned_us": stop - start - assigned,
                "matched_event_records": len(matched),
                "scope": "same-host client interval; inclusive event union, not GPU busy time",
            }
        )
    return output


def analyze(paths: list[Path], requests: dict | None = None) -> dict:
    frames = defaultdict(list)
    all_events, inventories = [], []
    truncated = False
    for path in paths:
        events, entries, cut = parse(path)
        all_events.extend(events)
        inventories.extend(entries)
        truncated |= cut
        for row in events:
            frames[(row["source"], row["device"], row["context"], row["frame"])].append(row)
    result = {
        "schema": "eagle_gpu_event_analysis_v1",
        "event_records": len(all_events),
        "cuda_graph_inventory_records": len(inventories),
        "trace_truncated": truncated,
        "orphan_inventory_records": sum(
            (row["source"], row["device"], row["context"], row["frame"]) not in frames
            for row in inventories
        ),
        "frames": [
            {
                "source": key[0],
                "device": key[1],
                "context": key[2],
                "frame": key[3],
                **frame_accounting(rows),
            }
            for key, rows in frames.items()
        ],
        "limitations": [
            "Intrusive synchronized event spans include stream idle and dispatch; "
            "no throughput claim.",
            "Frame GPU clocks have separate origins; totals across frames/streams are not added.",
            "Captured graph node inventory has no timing; missing attribution remains unassigned.",
            "Ordinary projections remain combined where pack/dot stages are not instrumented.",
            "Buffer-transfer annotations are hints; request joins require the same host and run.",
        ],
    }
    if requests is not None:
        result["requests"] = request_accounting(all_events, requests)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--requests", type=Path, help="same-run benchmark manifest on same host")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    requests = json.loads(args.requests.read_text()) if args.requests else None
    result = analyze(args.logs, requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"event_records": result["event_records"], "trace_truncated": result["trace_truncated"]}
        )
    )


if __name__ == "__main__":
    main()
