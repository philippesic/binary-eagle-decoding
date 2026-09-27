#!/usr/bin/env python3
"""Analyze a single bounded rescue-study Nsight Systems SQLite export (read only).

Usage: python scripts/analyze_binary_rescue_kernels.py trace.sqlite --variant D --output kernels.json
Variant is a caller annotation, not a process/model identity inferred from kernels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

import analyze_w1ax_cuda_trace as base


GROUP128 = "w1a16_group128_signadd"
A16_CATEGORIES = {"binary_a16_row_fused", "binary_a16_group128_fused"}
STANDARD_CATEGORIES = {"standard_q8_1_operand_conversion", "standard_quantized_matmul"}
GRAPH_API = re.compile(r"^(?:cuda|cu)(?:StreamBeginCapture|StreamEndCapture|GraphLaunch|GraphInstantiate\w*|GraphExecUpdate|GraphUpload|GraphDestroy|GraphExecDestroy)(?:_v\d+)?$")


def classify(row: dict) -> str:
    names = row["demangled_name"], row["short_name"]
    if base.matches_symbol(GROUP128, *names):
        return "binary_a16_group128_fused"
    category = base.classify(*names)
    return {
        "w1a16_signadd_rescale_fused": "binary_a16_row_fused",
        "anchor_packing_quantization": "standard_q8_1_operand_conversion",
        "anchor_quantized_mulmat": "standard_quantized_matmul",
        "anchor_float_mulmat": "standard_float_matvec",
    }.get(category, category)


def summaries(rows: list[dict]) -> dict:
    return {**base.distribution([row["duration_ns"] for row in rows]), **base.interval_summary(rows)}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def graph_evidence(path: Path, kernels: list[dict], global_pid: int | None = None) -> dict:
    """Retain available graph tables and API calls without inventing graph children."""
    result = {
        "api_tables": [], "graph_tables": [],
        "kernel_scope": "exact_global_pid" if global_pid is not None else "full_export",
        "kernel_rows_with_nonzero_graph_node_id": sum(row.get("graphNodeId") not in (None, 0) for row in kernels),
        "kernel_rows_with_zero_graph_node_id": sum(row.get("graphNodeId") == 0 for row in kernels),
        "kernel_rows_with_missing_graph_node_id": sum(row.get("graphNodeId") is None for row in kernels),
        "notes": [
            "API calls are host API observations, not GPU launch durations or successful captures unless the recorded return value is zero.",
            "Runtime and driver API tables can describe the same action; their counts and CPU durations must not be added together or to GPU times.",
            "Graph table row counts describe the exported schema, not necessarily launches, instantiated graphs, captures, or unique nodes.",
            "Missing tables/counters mean unavailable in this export, not zero real activity. Graph node ID zero is reported separately as an unspecified identifier.",
        ],
    }
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = sorted(row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        for table in tables:
            if "GRAPH" in table.upper():
                quoted = quote_identifier(table)
                columns = [row[1] for row in db.execute(f"PRAGMA table_info({quoted})")]
                filtered = global_pid is not None and "globalPid" in columns
                query = f"SELECT COUNT(*) FROM {quoted}" + (" WHERE globalPid=?" if filtered else "")
                result["graph_tables"].append({
                    "table": table, "columns": columns,
                    "scope": "exact_global_pid" if filtered else "unfiltered_full_export",
                    "row_count": db.execute(query, (global_pid,) if filtered else ()).fetchone()[0],
                })
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            info = {"table": table, "present": table in tables, "calls": [], "scope": "full_export"}
            result["api_tables"].append(info)
            if table not in tables:
                continue
            if "StringIds" not in tables:
                info["unavailable_reason"] = "StringIds table is absent; API names cannot be resolved"
                continue
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if global_pid is not None:
                if "globalPid" not in columns:
                    info.update(scope="omitted_without_globalPid", unavailable_reason="No exact globalPid column; globalTid is not decoded or joined implicitly")
                    continue
                info["scope"] = "exact_global_pid"
            if not {"start", "end", "nameId"} <= columns:
                info["unavailable_reason"] = "requires start, end, and nameId columns"
                continue
            calls = defaultdict(list)
            query = f"SELECT a.*, s.value AS resolved_name FROM {table} a LEFT JOIN StringIds s ON a.nameId=s.id"
            if global_pid is not None:
                query += " WHERE a.globalPid=?"
            for source in db.execute(query, (global_pid,) if global_pid is not None else ()):
                row = dict(source)
                name = row["resolved_name"]
                if not name or not GRAPH_API.fullmatch(name):
                    continue
                if not isinstance(row["start"], int) or not isinstance(row["end"], int) or row["end"] < row["start"]:
                    raise ValueError(f"invalid graph API timestamp in {table}")
                calls[name].append(row)
            for name, rows in sorted(calls.items()):
                known = [row["returnValue"] for row in rows if row.get("returnValue") is not None]
                info["calls"].append({
                    "api": name, "count": len(rows),
                    "cpu_api_duration": base.distribution([row["end"] - row["start"] for row in rows]),
                    "success_count": sum(value == 0 for value in known),
                    "failure_count": sum(value != 0 for value in known),
                    "unknown_return_count": len(rows) - len(known),
                })
    return result


def analyze(path: Path, variant: str | None = None, global_pid: int | None = None) -> dict:
    if global_pid is not None and (type(global_pid) is not int or not -(2**63) <= global_pid < 2**63):
        raise ValueError("global_pid must be a signed 64-bit integer from SQLite")
    schema = {"identity_columns": []}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        has_kernels = bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (base.KERNEL_TABLE,)).fetchone())
    rows = base.read_kernels(path, schema=schema) if has_kernels else []
    for row in rows:
        row["category"] = classify(row)
    optional, presence = base.read_optional_activities(path)
    selection = {"global_pid": global_pid, "identity_rule": "Exact raw SQLite globalPid; no OS PID decoding or bit interpretation", "activities": []}
    if global_pid is not None:
        present = [kind for kind, info in presence.items() if info["present"]]
        missing = [kind for kind in present if "globalPid" not in presence[kind]["identity_columns"]]
        if not present or missing:
            raise ValueError("globalPid filtering unavailable for GPU activity table(s): " + ", ".join(missing or ["none present"]))
    for kind in presence:
        values = [row for row in rows + optional if row["activity_kind"] == kind]
        selected = values if global_pid is None else [row for row in values if row.get("globalPid") == global_pid]
        selection["activities"].append({"activity_kind": kind, "export_row_count": len(values), "selected_row_count": len(selected),
                                        "null_or_missing_global_pid_count": sum(row.get("globalPid") is None for row in values)})
    if global_pid is not None:
        rows = [row for row in rows if row.get("globalPid") == global_pid]
        optional = [row for row in optional if row.get("globalPid") == global_pid]
    categories = defaultdict(list)
    partitions = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
        partitions[base.launch_identity(row)].append(row)
    binary = [row for row in rows if row["category"] in A16_CATEGORIES]
    standard = [row for row in rows if row["category"] in STANDARD_CATEGORIES]
    # Fixed study model: the 32,000-row output head is the only selected projection
    # whose ceil(M/128) grid is 250. This is model-specific inference, not attribution.
    head = [row for row in binary if tuple(row[f"grid{axis}"] for axis in "XZ") == (250, 1)
            and tuple(row[f"block{axis}"] for axis in "XYZ") == (128, 1, 1)]
    return {
        "schema": "binary_rescue_kernels_v1", "duration_unit": "ns",
        "variant_annotation": variant, "input": base.file_provenance(path),
        "analyzer": base.file_provenance(Path(__file__)),
        "shared_analyzer": base.file_provenance(Path(base.__file__)),
        "scope": ("Exact selected globalPid GPU activity" if global_pid is not None else "Whole supplied export") + "; setup, warmups and measurements are pooled unless collection was externally delimited.",
        "selection": selection,
        "notes": [
            "CUDA kernel durations and interval unions are distinct from client latency and CPU wall spans; never add overlapping CPU/GPU spans.",
            "sum_ns adds kernel durations; union_ns covers intervals with at least one observed kernel. Neither is a hardware utilization counter.",
            "Combined GPU activity includes aggregate GRAPH_TRACE intervals that may contain gaps and nested kernels; its union is observed timeline coverage, not kernel busy time.",
            "Standard Q8_1 conversion plus quantized-matmul totals include all observed models/projections, including the verifier. Kernel symbols alone cannot isolate the rescued Q8_0 weight subset from Q4_0 or other standard quantized weights.",
            "Packing-inclusive standard totals include observed Q8_1 conversion and MMVQ/MMQ/fixup kernels without assuming one-to-one pack/matmul pairs; cached/shared packing and host overhead are not inferred.",
            "Custom A16 row/group128 kernels fuse activation F16 rounding, sign-add, and scales in one kernel; no separate packing duration is added.",
            "Float matvec is recognized, but unclassified cuBLAS/tensor-core kernels remain visible by symbol; absence of a recognized float family does not prove no dense-head work.",
            "Grid/block shapes are actual launch dimensions, not logical matrix dimensions. Only the explicitly labeled frozen-model head inference maps a shape to a projection.",
            "Variant is caller-supplied annotation. PID/context/stream partitions are preserved; no request, round, prompt, or phase join is invented.",
            "Nsight collection overhead and proposal/output equivalence require separate matched checks before transferring conclusions to production timings.",
        ],
        "kernel_availability": {"table_present": has_kernels, "observed_row_count": len(rows),
                                "note": "No individual kernel cost is inferred from aggregate graph intervals."},
        "identity_columns": schema["identity_columns"],
        "overall_kernels": summaries(rows),
        "categories": [{"category": key, **summaries(values)} for key, values in sorted(categories.items())],
        "packing_inclusive_standard_quantized": summaries(standard),
        "binary_a16_fused": summaries(binary),
        "binary_head_shape_inference": {
            "status": "inferred" if head else "not_observed" if has_kernels else "unavailable",
            "assumption": "Frozen one-block EAGLE model, 32,000-row head unique among nine selected projections; kernel gridX=ceil(M/128)=250, block=(128,1,1), gridZ=1.",
            "limitation": "Grid alone permits M in [31873,32000]; uniqueness requires the stated frozen model. Not a runtime tensor identity or proof of logical K.",
            **summaries(head), "kernels": base.kernel_summaries(head),
        },
        "kernels": base.kernel_summaries(rows),
        "identity_partitions": [
            {**dict(identity), **summaries(values)} for identity, values in sorted(partitions.items(), key=lambda item: repr(item[0]))
        ],
        "gpu_activities": base.gpu_activity_summary(rows + optional, presence),
        "graph_evidence": graph_evidence(path, rows, global_pid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", help="Caller annotation only, e.g. Q4_0, D, C, rescue_down, dense_head")
    parser.add_argument("--global-pid", type=int, help="Exact raw SQLite globalPid; never decoded as an OS PID")
    args = parser.parse_args()
    if args.output and (args.output.resolve() == args.sqlite.resolve() or
                        (args.output.exists() and args.sqlite.exists() and args.output.samefile(args.sqlite))):
        parser.error("output must not overwrite the input database")
    try:
        result = json.dumps(analyze(args.sqlite, args.variant, args.global_pid), indent=2) + "\n"
        if args.output:
            args.output.write_text(result)
        else:
            print(result, end="")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
