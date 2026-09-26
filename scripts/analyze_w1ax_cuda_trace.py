#!/usr/bin/env python3
"""Read-only pooled kernel timing analysis of an Nsight Systems SQLite export.

No NVTX/capture attribution is inferred. All durations are in nanoseconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
REQUIRED_COLUMNS = {
    "start", "end", "deviceId", "streamId", "gridX", "gridY", "gridZ",
    "blockX", "blockY", "blockZ",
}
# Exact function boundaries avoid treating a validator or a similarly named
# unknown function as production work. These are demangled/short CUDA names.
CATEGORIES = (
    ("w1ax_validation", r"w1ax_validate_integer_dots"),
    ("w1_packing_quantization", r"w1a1_pack_activations|w1ax_quantize"),
    ("w1_dot_rescale_fused", r"w1a1_xor_popc|w1ax_integer_dot"),
    ("w1a16_signadd_rescale_fused", r"w1a16_signadd"),
    ("anchor_packing_quantization", r"quantize_q8_1|quantize_mmq_q8_1"),
    ("anchor_quantized_mulmat", r"mul_mat_vec_q|mul_mat_vec_q_moe|mul_mat_q|mul_mat_q_stream_k_fixup"),
    ("anchor_float_mulmat", r"mul_mat_vec_f"),
)


def classify(*names: str | None) -> str:
    for category, symbols in CATEGORIES:
        if any(re.search(r"(?<![\w])(?:" + symbols + r")(?=[<(\s]|$)", name)
               for name in names if name):
            return category
    return "unclassified"


def distribution(durations: list[int]) -> dict:
    xs = sorted(durations)
    if not xs:
        return {"count": 0, "sum_ns": 0, "median_ns": None, "p95_ns": None}
    pos = (len(xs) - 1) * 0.95
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return {
        "count": len(xs), "sum_ns": sum(xs), "median_ns": median(xs),
        "p95_ns": xs[lo] + (xs[hi] - xs[lo]) * (pos - lo),
    }


def interval_summary(rows: list[dict]) -> dict:
    """Union plus excess summed duration (multiplicity-weighted overlap)."""
    intervals = sorted((row["start"], row["end"]) for row in rows)
    union = 0
    left = right = None
    for start, end in intervals:
        if right is None or start > right:
            if right is not None:
                union += right - left
            left, right = start, end
        else:
            right = max(right, end)
    if right is not None:
        union += right - left
    total = sum(end - start for start, end in intervals)
    span = max(end for _, end in intervals) - intervals[0][0] if intervals else 0
    return {
        "count": len(rows), "sum_ns": total, "union_ns": union,
        "overlap_excess_ns": total - union, "span_ns": span,
        "no_kernel_ns_within_span": span - union,
    }


def read_kernels(path: Path) -> list[dict]:
    # URI mode=ro both prevents creation of missing files and disallows writes.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {KERNEL_TABLE, "StringIds"} - tables
        if missing:
            raise ValueError("missing required SQLite table(s): " + ", ".join(sorted(missing)))
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({KERNEL_TABLE})")}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError("missing kernel column(s): " + ", ".join(sorted(missing)))
        names = [name for name in ("shortName", "demangledName") if name in columns]
        if not names:
            raise ValueError("kernel table requires shortName or demangledName")
        string_columns = {row[1] for row in db.execute("PRAGMA table_info(StringIds)")}
        if not {"id", "value"} <= string_columns:
            raise ValueError("StringIds requires id and value columns")
        selections = ["k.*"]
        joins = []
        for name in names:
            selections.append(f"s_{name}.value AS resolved_{name}")
            joins.append(f"LEFT JOIN StringIds s_{name} ON k.{name} = s_{name}.id")
        query = f"SELECT {', '.join(selections)} FROM {KERNEL_TABLE} k {' '.join(joins)} ORDER BY k.start, k.end"
        result = []
        for row_number, source in enumerate(db.execute(query), 1):
            row = dict(source)
            for name in REQUIRED_COLUMNS:
                if not isinstance(row[name], int):
                    raise ValueError(f"kernel row {row_number}: {name} must be an integer")
            if row["end"] < row["start"]:
                raise ValueError(f"kernel row {row_number}: end precedes start")
            if any(row[name] <= 0 for name in REQUIRED_COLUMNS if name.startswith(("grid", "block"))):
                raise ValueError(f"kernel row {row_number}: launch dimensions must be positive")
            short = row.get("resolved_shortName")
            demangled = row.get("resolved_demangledName")
            # Keep unresolved IDs visible instead of dropping unmatched rows.
            row["symbol"] = demangled or short or "unresolved:" + ",".join(
                f"{name}={row[name]}" for name in names
            )
            row["short_name"] = short
            row["demangled_name"] = demangled
            row["category"] = classify(demangled, short)
            row["duration_ns"] = row["end"] - row["start"]
            result.append(row)
        return result


def analyze(path: Path, act_bits: int | None = None) -> dict:
    if act_bits is not None and act_bits not in (1, 4, 8, 16):
        raise ValueError("act_bits must be 1, 4, 8, or 16")
    rows = read_kernels(path)
    kernels = defaultdict(list)
    categories = defaultdict(list)
    devices = defaultdict(list)
    for row in rows:
        kernels[row["symbol"]].append(row)
        categories[row["category"]].append(row["duration_ns"])
        devices[row["deviceId"]].append(row)
    per_kernel = []
    for symbol, launches in sorted(kernels.items()):
        shapes = defaultdict(list)
        for row in launches:
            key = (row["deviceId"], row["streamId"],
                   *(row[f"grid{axis}"] for axis in "XYZ"),
                   *(row[f"block{axis}"] for axis in "XYZ"))
            shapes[key].append(row["duration_ns"])
        per_kernel.append({
            "symbol": symbol, "short_name": launches[0]["short_name"],
            "demangled_name": launches[0]["demangled_name"],
            "category": launches[0]["category"],
            **distribution([row["duration_ns"] for row in launches]),
            "launch_configurations": [
                {"device_id": key[0], "stream_id": key[1], "grid": list(key[2:5]),
                 "block": list(key[5:8]), **distribution(values)}
                for key, values in sorted(shapes.items())
            ],
        })
    return {
        "schema": "w1ax_cuda_trace_v1", "input": str(path.resolve()),
        "act_bits_annotation": act_bits, "duration_unit": "ns",
        "notes": [
            "Traced correctness, warmup, and sample launches are pooled; no per-capture, layer, or timing-phase attribution is inferred.",
            "Dot kernels include rescaling. Dot-only timing is an already-packed diagnostic, not full operator cost; packing/quantization is separate.",
            "A16 signadd includes the FP16 cast and weight rescaling inside the kernel.",
            "Profiler overhead is measured elsewhere; these are traced CUDA kernel durations, not uninstrumented operator latency.",
            "act_bits is caller-supplied metadata; W1A4 and W1A8 share symbols and cannot be distinguished from names alone.",
            "Union is time with at least one kernel active. Overlap excess is sum minus union, weighted by excess concurrency; gaps are not a measurement of CPU overhead.",
            "Overall union spans all devices on the export timeline; per-device unions are also reported. Only kernel activities are included.",
            "p95 uses linear interpolation at index 0.95*(count-1). Unknown symbols are retained as unclassified.",
        ],
        "overall": interval_summary(rows),
        "devices": [{"device_id": device, **interval_summary(launches)}
                    for device, launches in sorted(devices.items())],
        "categories": [{"category": category, **distribution(values)}
                       for category, values in sorted(categories.items())],
        "kernels": per_kernel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--act-bits", type=int, choices=(1, 4, 8, 16))
    parser.add_argument("--output", type=Path, help="JSON report (default: stdout)")
    args = parser.parse_args()
    if args.output and (args.output.resolve() == args.database.resolve()
                        or (args.output.exists() and args.database.exists()
                            and args.output.samefile(args.database))):
        parser.error("output must not overwrite the input database")
    try:
        report = analyze(args.database, args.act_bits)
    except (sqlite3.Error, ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
