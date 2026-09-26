#!/usr/bin/env python3
"""Read-only kernel and GPU activity analysis of an Nsight Systems SQLite export.

No NVTX/capture attribution is inferred. All durations are in nanoseconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
ACTIVITY_TABLES = {kind: f"CUPTI_ACTIVITY_KIND_{kind}" for kind in ("KERNEL", "GRAPH_TRACE", "MEMCPY", "MEMSET")}
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
PACK_DOT_SYMBOLS = {
    "w1a1_pack_activations": "w1a1_xor_popc",
    "w1ax_quantize": "w1ax_integer_dot",
}
OPTIONAL_IDENTITY_COLUMNS = {
    "globalPid": "global_pid", "contextId": "context_id", "greenContextId": "green_context_id",
}


def launch_identity(row: dict) -> tuple:
    """Preserve every available process/context identifier in grouping keys."""
    return (("device_id", row["deviceId"]),
            *((output, row[column]) for column, output in OPTIONAL_IDENTITY_COLUMNS.items() if column in row),
            ("stream_id", row["streamId"]))


def launch_group_sort(item: tuple) -> tuple:
    key = item[0]
    # Optional SQLite IDs may be NULL; retain them without comparing None/int.
    return (tuple((name, value is not None, value or 0) for name, value in key[0]), *key[1:])


def matches_symbol(symbols: str, *names: str | None) -> bool:
    return any(re.search(r"(?<![\w])(?:" + symbols + r")(?=[<(\s]|$)", name)
               for name in names if name)


def classify(*names: str | None) -> str:
    for category, symbols in CATEGORIES:
        if matches_symbol(symbols, *names):
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


def paired_kernel_summary(rows: list[dict]) -> dict:
    """Pair adjacent known pack/dot launches within each process/context stream.

    Launch adjacency and equal token grids are evidence for a candidate pair,
    not proof of a shared tensor, capture, or layer. No launch is reused.
    """
    streams = defaultdict(list)
    pack_count = dot_count = 0
    for row in rows:
        names = (row["demangled_name"], row["short_name"])
        pack_count += any(matches_symbol(symbol, *names) for symbol in PACK_DOT_SYMBOLS)
        dot_count += any(matches_symbol(symbol, *names) for symbol in PACK_DOT_SYMBOLS.values())
        streams[launch_identity(row)].append(row)
    grouped = defaultdict(list)
    all_pairs = []
    for identity, launches in streams.items():
        # NULL process/context values do not prove identity. Missing columns
        # retain legacy behavior, with an explicit report-level limitation.
        if any(name in ("global_pid", "context_id") and value is None for name, value in identity):
            continue
        launches = sorted(launches, key=lambda row: (row["start"], row["end"]))
        for pack, dot in zip(launches, launches[1:]):
            pair_symbols = next(((packing, compute) for packing, compute in PACK_DOT_SYMBOLS.items()
                                 if matches_symbol(packing, pack["demangled_name"], pack["short_name"])
                                 and matches_symbol(compute, dot["demangled_name"], dot["short_name"])), None)
            if pair_symbols is None or pack["gridX"] != dot["gridY"] or pack["end"] > dot["start"]:
                continue
            pack_grid = tuple(pack[f"grid{axis}"] for axis in "XYZ")
            dot_grid = tuple(dot[f"grid{axis}"] for axis in "XYZ")
            pack_block = tuple(pack[f"block{axis}"] for axis in "XYZ")
            dot_block = tuple(dot[f"block{axis}"] for axis in "XYZ")
            key = (identity, *pair_symbols, pack_grid, pack_block, dot_grid, dot_block)
            values = {
                "pack": pack["duration_ns"], "dot": dot["duration_ns"],
                "kernel_sum": pack["duration_ns"] + dot["duration_ns"],
                "elapsed": dot["end"] - pack["start"],
                "gap": dot["start"] - pack["end"],
            }
            grouped[key].append(values)
            all_pairs.append(values)

    def timing(values: list[dict]) -> dict:
        return {name: distribution([row[name] for row in values])
                for name in ("pack", "dot", "kernel_sum", "elapsed", "gap")}

    count = len(all_pairs)
    return {
        "pair_count": count,
        "eligible_pack_count": pack_count, "eligible_dot_count": dot_count,
        "unpaired_pack_count": pack_count - count,
        "unpaired_dot_count": dot_count - count,
        "unpaired_kernel_count": pack_count + dot_count - 2 * count,
        "timing": timing(all_pairs),
        "launch_configurations": [
            {**dict(key[0]), "pack_symbol": key[1], "dot_symbol": key[2],
             "pack_grid": list(key[3]), "pack_block": list(key[4]),
             "dot_grid": list(key[5]), "dot_block": list(key[6]),
             "pair_count": len(values), "timing": timing(values)}
            for key, values in sorted(grouped.items(), key=launch_group_sort)
        ],
    }


def read_kernels(path: Path, *, schema: dict | None = None) -> list[dict]:
    # URI mode=ro both prevents creation of missing files and disallows writes.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {KERNEL_TABLE, "StringIds"} - tables
        if missing:
            raise ValueError("missing required SQLite table(s): " + ", ".join(sorted(missing)))
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({KERNEL_TABLE})")}
        if schema is not None:
            schema["identity_columns"] = [name for name in OPTIONAL_IDENTITY_COLUMNS if name in columns]
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
            for name in OPTIONAL_IDENTITY_COLUMNS:
                if row.get(name) is not None and not isinstance(row[name], int):
                    raise ValueError(f"kernel row {row_number}: {name} must be an integer or NULL")
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
            row["activity_kind"] = "KERNEL"
            result.append(row)
        return result


def read_optional_activities(path: Path) -> tuple[list[dict], dict]:
    """Read aggregate graph and transfer intervals without inferring children."""
    result = []
    presence = {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for kind, table in ACTIVITY_TABLES.items():
            present = table in tables
            presence[kind] = {"table": table, "present": present}
            if not present:
                continue
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            presence[kind]["identity_columns"] = [name for name in OPTIONAL_IDENTITY_COLUMNS if name in columns]
            if kind == "KERNEL":
                continue
            required = {"start", "end", "deviceId", "streamId"}
            missing = required - columns
            if missing:
                raise ValueError(f"missing {table} column(s): {', '.join(sorted(missing))}")
            for row_number, source in enumerate(db.execute(f"SELECT * FROM {table} ORDER BY start, end"), 1):
                row = dict(source)
                for name in required:
                    if not isinstance(row[name], int):
                        raise ValueError(f"{table} row {row_number}: {name} must be an integer")
                for name in OPTIONAL_IDENTITY_COLUMNS:
                    if row.get(name) is not None and not isinstance(row[name], int):
                        raise ValueError(f"{table} row {row_number}: {name} must be an integer or NULL")
                if row["end"] < row["start"]:
                    raise ValueError(f"{table} row {row_number}: end precedes start")
                if row.get("bytes") is not None and (not isinstance(row["bytes"], int) or row["bytes"] < 0):
                    raise ValueError(f"{table} row {row_number}: bytes must be a nonnegative integer or NULL")
                row["activity_kind"] = kind
                row["duration_ns"] = row["end"] - row["start"]
                result.append(row)
    return result, presence


def gpu_intervals(rows: list[dict]) -> dict:
    summary = interval_summary(rows)
    summary["no_activity_ns_within_span"] = summary.pop("no_kernel_ns_within_span")
    return summary


def gpu_activity_summary(rows: list[dict], presence: dict) -> dict:
    by_kind = defaultdict(list)
    by_identity = defaultdict(list)
    by_device = defaultdict(list)
    for row in rows:
        kind = row["activity_kind"]
        by_kind[kind].append(row)
        by_identity[launch_identity(row), kind].append(row)
        by_device[row["deviceId"]].append(row)

    def details(activities: list[dict]) -> dict:
        metadata = ("graphId", "graphExecId", "copyKind", "srcKind", "dstKind", "srcDeviceId",
                    "srcContextId", "dstDeviceId", "dstContextId", "memKind", "value")
        return {
            **distribution([row["duration_ns"] for row in activities]), **gpu_intervals(activities),
            "bytes_sum": sum(row.get("bytes") or 0 for row in activities),
            "bytes_record_count": sum(row.get("bytes") is not None for row in activities),
            "metadata_values": {
                name: sorted({row[name] for row in activities if row.get(name) is not None})
                for name in metadata if any(name in row for row in activities)
            },
        }

    return {
        "scope": "Collected KERNEL, GRAPH_TRACE, MEMCPY, and MEMSET GPU intervals only; no request/round/stage attribution.",
        "notes": [
            "GRAPH_TRACE rows are aggregate graph replay intervals; no kernel breakdown is inferred for their contents.",
            "Combined sum_ns may count graph intervals and nested kernels/transfers more than once, as well as concurrent work. Use union_ns for observed covered timeline duration, not a naive additive GPU cost.",
            "Combined union_ns is observed activity-span coverage, not physical GPU busy time. It includes internal gaps covered by aggregate graph intervals and is not a hardware utilization or kernel execution counter.",
            "Missing optional tables mean not present in this export, not proof that the GPU performed no such work.",
        ],
        "combined": gpu_intervals(rows),
        "devices": [{"device_id": device, **gpu_intervals(activities)} for device, activities in sorted(by_device.items())],
        "by_activity_kind": [{"activity_kind": kind, **presence[kind], **details(by_kind[kind])} for kind in ACTIVITY_TABLES],
        "identity_partitions": [
            {**dict(key[0]), "activity_kind": key[1], **details(activities)}
            for key, activities in sorted(by_identity.items(), key=launch_group_sort)
        ],
        "graph_granularity": {
            "aggregate_graph_interval_count": len(by_kind["GRAPH_TRACE"]),
            "kernel_rows_with_graph_node_id": sum(row.get("graphNodeId") is not None for row in by_kind["KERNEL"]),
            "kernel_rows_without_graph_node_id": sum(row.get("graphNodeId") is None for row in by_kind["KERNEL"]),
        },
    }


def kernel_summaries(rows: list[dict]) -> list[dict]:
    kernels = defaultdict(list)
    for row in rows:
        kernels[row["symbol"]].append(row)
    per_kernel = []
    for symbol, launches in sorted(kernels.items()):
        shapes = defaultdict(list)
        for row in launches:
            key = (launch_identity(row), tuple(row[f"grid{axis}"] for axis in "XYZ"),
                   tuple(row[f"block{axis}"] for axis in "XYZ"))
            shapes[key].append(row["duration_ns"])
        per_kernel.append({
            "symbol": symbol, "short_name": launches[0]["short_name"],
            "demangled_name": launches[0]["demangled_name"],
            "category": launches[0]["category"],
            **distribution([row["duration_ns"] for row in launches]),
            "launch_configurations": [
                {**dict(key[0]), "grid": list(key[1]), "block": list(key[2]), **distribution(values)}
                for key, values in sorted(shapes.items(), key=launch_group_sort)
            ],
        })
    return per_kernel


def file_provenance(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def benchmark_association(rows: list[dict], database: Path, run: Path, presence: dict) -> dict:
    """Strict, bounded 5x8 server schedule inference; never emit partial labels."""
    result = {
        "status": "unavailable", "label": "CHRONOLOGICAL SCHEDULE INFERENCE",
        "scope": "All collected process GPU activity kinds pooled across startup, warmups, and measurements; no exact request, round, or layer attribution.",
        "limitations": [
            "The runner did not record raw PIDs. Chronology and signatures support inferred labels, not a proven PID-to-run join.",
            "Nonoverlap is checked for observed first-to-last GPU activity spans including graph replays and transfers, not OS process lifetimes or untraced work.",
            "A4 and A8 share kernel symbols; their distinction depends on the schedule and available server mode logs.",
        ],
        "sources": {},
    }
    try:
        loaded = {}
        for name in ("manifest", "report", "records"):
            path = run / f"{name}.json"
            data = path.read_bytes()
            result["sources"][name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
            loaded[name] = json.loads(data)
        result["sources"]["database"] = file_provenance(database)
        result["sources"]["analyzer"] = file_provenance(Path(__file__))
        manifest, report, records = (loaded[name] for name in ("manifest", "report", "records"))
        if not isinstance(manifest, dict) or not isinstance(report, dict) or not isinstance(records, list):
            raise ValueError("manifest/report must be objects and records must be a list")
        if report.get("status") != "complete":
            raise ValueError("benchmark report is not complete")
        variants = {"target_only", "ordinary_eagle", "draft_q4_0", "draft_q8_0",
                    "draft_w1a16", "draft_w1a8", "draft_w1a4", "draft_w1a1"}
        orders = manifest["orders"]
        if (not isinstance(orders, list) or len(orders) != 5
                or any(not isinstance(order, list) or len(order) != 8 or set(order) != variants for order in orders)):
            raise ValueError("association requires the bounded five-repetition, eight-variant schedule")
        schedule = [(rep, variant) for rep, order in enumerate(orders) for variant in order]
        if set(manifest["variants"]) != variants or set(report["variants"]) != variants:
            raise ValueError("manifest/report variants disagree with schedule")
        if not isinstance(manifest["commands"], dict) or not isinstance(manifest["policy"], dict):
            raise ValueError("manifest commands and policy must be objects")
        if any(not isinstance(manifest["commands"].get(variant), list) or not manifest["commands"][variant]
               for variant in variants):
            raise ValueError("missing server command for a scheduled variant")
        if manifest["policy"].get("warmup_requests") != 2 or manifest["policy"].get("repetitions") != 5:
            raise ValueError("policy must record two warmups and five repetitions")
        prompts = manifest["prompt_ids"]
        if len(prompts) != 3 or len(set(prompts)) != 3:
            raise ValueError("association requires three distinct measured prompts")
        expected = {(rep, variant, prompt) for rep, variant in schedule for prompt in prompts}
        observed = [(row["repetition"], row["variant"], row["prompt_id"]) for row in records]
        if len(observed) != 120 or set(observed) != expected or report["records"] != len(observed):
            raise ValueError("completed records do not exactly cover every scheduled repetition/variant/prompt")
        if not rows or any(row.get("globalPid") is None or row.get("contextId") is None for row in rows):
            raise ValueError("complete globalPid and contextId identities are required")
        pids = defaultdict(list)
        for row in rows:
            pids[row["globalPid"]].append(row)
        if len(pids) != len(schedule):
            raise ValueError(f"PID group count {len(pids)} does not match {len(schedule)} scheduled servers")
        ordered = sorted(pids.items(), key=lambda item: min(row["start"] for row in item[1]))
        previous_end = None
        for _, launches in ordered:
            start, end = min(row["start"] for row in launches), max(row["end"] for row in launches)
            if previous_end is not None and start <= previous_end:
                raise ValueError("PID GPU activity spans interleave, overlap, or have ambiguous touching boundaries")
            previous_end = end
        signatures = {
            "draft_w1a16": {"w1a16_signadd"},
            "draft_w1a8": {"w1ax_quantize", "w1ax_integer_dot"},
            "draft_w1a4": {"w1ax_quantize", "w1ax_integer_dot"},
            "draft_w1a1": {"w1a1_pack_activations", "w1a1_xor_popc"},
        }
        markers = {
            "draft_w1a16": "CUDA packed W1A16 FP16 SIGNADD dispatch",
            "draft_w1a8": "CUDA packed W1A8 INT8 dispatch",
            "draft_w1a4": "CUDA packed W1A4 BITSERIAL dispatch",
            "draft_w1a1": "CUDA packed W1A1 XOR/POPCOUNT dispatch",
        }
        known = set().union(*signatures.values())
        mappings = []
        for (rep, variant), (pid, launches) in zip(schedule, ordered):
            kernels = [row for row in launches if row["activity_kind"] == "KERNEL"]
            seen = {symbol for symbol in known if any(matches_symbol(symbol, row["demangled_name"], row["short_name"])
                                                     for row in kernels)}
            if seen != signatures.get(variant, set()):
                raise ValueError(f"W1 kernel signature mismatch at rep-{rep:02d}/{variant}: {sorted(seen)}")
            log = run / f"rep-{rep:02d}" / variant / "server.log"
            log_evidence = {"status": "unavailable", "reason": "server.log absent"}
            if log.exists():
                data = log.read_bytes()
                content = data.decode(errors="replace")
                result["sources"][f"server_log:rep-{rep:02d}/{variant}"] = {
                    "path": str(log.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                }
                seen_modes = set(re.findall(r"EAGLE3 W1Ax activation bits: (\d+)\b", content))
                expected_modes = {variant.removeprefix("draft_w1a")} if variant in markers else set()
                seen_markers = {name for name, marker in markers.items() if marker in content}
                expected_markers = {variant} if variant in markers else set()
                if (seen_markers != expected_markers or seen_modes != expected_modes
                        or "CUDA packed W1A4 CONVENTIONAL dispatch" in content):
                    raise ValueError(f"server mode marker mismatch at rep-{rep:02d}/{variant}")
                log_evidence = {"status": "consistent", "mode_bits": sorted(seen_modes), "dispatch_variants": sorted(seen_markers)}
            mappings.append({
                "global_pid": pid, "repetition": rep, "variant": variant,
                "first_activity_start_ns": min(row["start"] for row in launches),
                "last_activity_end_ns": max(row["end"] for row in launches),
                "first_kernel_start_ns": min((row["start"] for row in kernels), default=None),
                "last_kernel_end_ns": max((row["end"] for row in kernels), default=None),
                "observed_w1_symbols": sorted(seen), "server_log_evidence": log_evidence,
                "kernel_durations": interval_summary(kernels), "kernels": kernel_summaries(kernels),
                "packing_inclusive_pairs": paired_kernel_summary(kernels),
                "gpu_activities": gpu_activity_summary(launches, presence),
            })
        result.update(status="available", process_count=len(mappings), measured_record_count=len(records),
                      warmups_per_process=2, mappings=mappings)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


def analyze(path: Path, act_bits: int | None = None, benchmark_run: Path | None = None) -> dict:
    if act_bits is not None and act_bits not in (1, 4, 8, 16):
        raise ValueError("act_bits must be 1, 4, 8, or 16")
    schema = {}
    rows = read_kernels(path, schema=schema)
    optional_rows, presence = read_optional_activities(path)
    all_activities = rows + optional_rows
    missing_identity = [name for name in ("globalPid", "contextId") if name not in schema["identity_columns"]]
    null_identity_rows = sum(any(name in row and row[name] is None for name in ("globalPid", "contextId"))
                             for row in rows)
    categories = defaultdict(list)
    devices = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row["duration_ns"])
        devices[row["deviceId"]].append(row)
    result = {
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
            "Packing-inclusive pairs require adjacent matching pack/quantize then dot kernels on the same device, process, CUDA context, green context (when available), and stream, pack.gridX == dot.gridY, and pack.end <= dot.start. Kernels in other identity partitions may intervene.",
            "Missing globalPid/contextId columns permit legacy single-process pairing but cannot establish cross-process/context isolation. NULL values in present process/context columns are never paired; identity_attribution records these limits.",
            "Pair kernel_sum includes packing and fused dot/rescaling; elapsed also includes the observed inter-kernel gap. This kernel-only view excludes host allocation and is not full operator latency. Pair duration sums are not a union across pairs or streams.",
            "Pairing is a launch-adjacency heuristic, with no per-capture or layer attribution. Unpaired counts cover eligible known pack and dot kernels only; A16 signadd is a single kernel and is not eligible.",
            "p95 uses linear interpolation at index 0.95*(count-1). Unknown symbols are retained as unclassified.",
        ],
        "overall": interval_summary(rows),
        "identity_attribution": {
            "available_columns": schema["identity_columns"],
            "missing_process_context_columns": missing_identity,
            "null_process_context_row_count": null_identity_rows,
            "limited": bool(missing_identity or null_identity_rows),
        },
        "devices": [{"device_id": device, **interval_summary(launches)}
                    for device, launches in sorted(devices.items())],
        "categories": [{"category": category, **distribution(values)}
                       for category, values in sorted(categories.items())],
        "kernels": kernel_summaries(rows),
        "packing_inclusive_pairs": paired_kernel_summary(rows),
        "gpu_activities": gpu_activity_summary(all_activities, presence),
    }
    if benchmark_run is not None:
        result["benchmark_association"] = benchmark_association(all_activities, path, benchmark_run, presence)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--act-bits", type=int, choices=(1, 4, 8, 16))
    parser.add_argument("--output", type=Path, help="JSON report (default: stdout)")
    parser.add_argument("--benchmark-run", type=Path, help="Completed 5x8 server run directory for conservative chronological schedule inference")
    args = parser.parse_args()
    if args.output and (args.output.resolve() == args.database.resolve()
                        or (args.output.exists() and args.database.exists()
                            and args.output.samefile(args.database))):
        parser.error("output must not overwrite the input database")
    try:
        report = analyze(args.database, args.act_bits, args.benchmark_run)
    except (sqlite3.Error, ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
