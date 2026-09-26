#!/usr/bin/env python3
"""Offline summaries for W1Ax round traces and identical-input operator replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

TRACE_SCHEMA = "w1ax_eagle_round_v1"
STAGES = (
    "draft_us",
    "checkpoint_us",
    "target_decode_sync_us",
    "process_us",
    "check_us",
    "kv_repair_us",
    "accept_hook_us",
    "residual_us",
)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(row)
    return rows


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality = [r for r in rows if r.get("status") != "checkpoint_replay"]
    accepted_values = [r.get("n_accepted") for r in quality]
    proposed_values = [r.get("n_proposed") for r in quality]
    emitted_values = [r.get("n_emitted") for r in quality]
    accepted_ok = all(_nonnegative_number(v) for v in accepted_values)
    proposed_ok = all(_nonnegative_number(v) for v in proposed_values)
    emitted_ok = all(_nonnegative_number(v) for v in emitted_values)
    accepted = sum(accepted_values) if accepted_ok else None
    proposed = sum(proposed_values) if proposed_ok else None
    emitted = sum(emitted_values) if emitted_ok else None
    lengths: list[int] = []
    for row in quality:
        token_ids = row.get("proposed_token_ids")
        n = row.get("n_proposed")
        if isinstance(token_ids, list):
            lengths.append(len(token_ids))
        elif _nonnegative_number(n):
            lengths.append(int(n))

    rounds_by_depth: dict[str, dict[str, int | float | None]] = {}
    max_depth = max(lengths, default=0)
    for depth in range(1, max_depth + 1):
        eligible = 0
        accepted_at_depth = 0
        inferable = True
        for row in quality:
            tokens = row.get("proposed_token_ids")
            n_proposed = len(tokens) if isinstance(tokens, list) else row.get("n_proposed")
            n_accepted = row.get("n_accepted")
            if not _nonnegative_number(n_proposed) or not _nonnegative_number(n_accepted):
                inferable = False
                continue
            if n_proposed >= depth:
                eligible += 1
                if n_accepted >= depth:
                    accepted_at_depth += 1
        rounds_by_depth[str(depth)] = {
            "eligible_rounds": eligible if inferable else None,
            "accepted_at_depth": accepted_at_depth if inferable else None,
            "conditional_acceptance": accepted_at_depth / eligible if inferable and eligible else None,
        }

    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    round_values: list[float] = []
    begin_values: list[float] = []
    for row in rows:
        if row.get("schema") != TRACE_SCHEMA:
            continue
        if _nonnegative_number(row.get("round_us")):
            round_values.append(row["round_us"])
        if _nonnegative_number(row.get("begin_us")):
            begin_values.append(row["begin_us"])
        for stage in STAGES:
            if _nonnegative_number(row.get(stage)):
                stage_values[stage].append(row[stage])

    return {
        "trace_rows": len(rows),
        "quality_rounds": len(quality),
        "excluded_checkpoint_replay_rows": len(rows) - len(quality),
        "accepted": accepted,
        "proposed": proposed,
        "emitted": emitted,
        "rounds": len(quality),
        "accepted_per_proposed": accepted / proposed if accepted is not None and proposed else None,
        "accepted_per_round": accepted / len(quality) if accepted is not None and quality else None,
        "actual_proposal_lengths": distribution([float(v) for v in lengths]),
        "conditional_acceptance_by_depth": rounds_by_depth,
        "cpu_wall_us": {
            "round": distribution(round_values),
            "begin_outside_round": distribution(begin_values),
            "stages": {stage: distribution(values) for stage, values in stage_values.items()},
            "note": "CPU wall spans; named spans may overlap. residual_us is the server-reported unattributed remainder. No CUDA event data is present.",
        },
    }


def _sample_summary(samples: Any, path: Path, line: int) -> list[float]:
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}:{line}: samples_us must be a non-empty array")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError(f"{path}:{line}: samples_us must contain finite non-negative numbers")
    return [float(v) for v in samples]


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[tuple[int, list[float]]]] = defaultdict(list)
    normalized = []
    for i, row in enumerate(rows, 1):
        for field in ("K", "M", "N", "name", "group", "replay_bits"):
            if field not in row:
                raise ValueError(f"operator replay row {i} missing {field}")
        bits = row["replay_bits"]
        if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
            raise ValueError(f"operator replay row {i}: replay_bits must be a positive integer")
        samples = _sample_summary(row.get("samples_us"), Path("operator replay JSONL"), i)
        key = (row["K"], row["M"], row["N"], row["name"], row["group"])
        if any(existing_bits == bits for existing_bits, _ in grouped[key]):
            raise ValueError(f"operator replay row {i}: duplicate precision row for layer/shape key {key} ({bits} bits)")
        grouped[key].append((bits, samples))
        normalized.append((key, bits, samples))

    by_layer_shape = []
    for key in sorted(grouped, key=lambda x: tuple(str(part) for part in x)):
        entries = grouped[key]
        precision = {}
        for bits, samples in sorted(entries, key=lambda x: x[0]):
            precision[str(bits)] = {
                "samples": len(samples),
                "median_us": median(samples),
                "distribution_us": distribution(samples),
            }
        ratios = {}
        baselines = {bits: samples for bits, samples in entries if bits == 16}
        baseline = baselines[16] if len(baselines) == 1 else None
        for bits, samples in sorted(entries, key=lambda x: x[0]):
            if bits == 16:
                continue
            if baseline is None or len(baseline) != len(samples):
                ratios[str(bits)] = {"paired_precision_ratio_vs_16": None, "reason": "missing unique 16-bit baseline or unequal paired sample count"}
                continue
            paired = [low / high for low, high in zip(samples, baseline) if high > 0]
            ratios[str(bits)] = {
                "paired_precision_ratio_vs_16": median(paired) if len(paired) == len(samples) else None,
                "paired_sample_ratios": paired,
            }
        by_layer_shape.append({
            "K": key[0], "M": key[1], "N": key[2], "name": key[3], "group": key[4],
            "precision": precision, "paired_ratios": ratios,
        })
    return {
        "rows": len(rows),
        "by_layer_shape": by_layer_shape,
        "pairing": "Rows are paired by K/M/N/name/group and sample index; replay_bits=16 is the ratio baseline. This assumes input identity from the operator replay producer; no tensor hashes are present in this schema.",
    }


def analyze_run(run_dir: Path, replay_path: Path | None = None) -> dict[str, Any]:
    records_path = run_dir / "records.json"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = json.loads(records_path.read_text())
    if not isinstance(records, list):
        raise ValueError("records.json must contain an array")
    variants = sorted({row.get("variant") for row in records if isinstance(row, dict) and row.get("variant")})
    variant_results = {}
    for variant in variants:
        trace_rows = []
        paths = sorted(run_dir.glob(f"rep-*/{variant}/round-trace.jsonl"))
        for path in paths:
            trace_rows.extend(read_jsonl(path))
        bad = [r for r in trace_rows if r.get("schema") != TRACE_SCHEMA]
        if bad:
            raise ValueError(f"unexpected trace schema in {variant}: {bad[0].get('schema')}")
        summary = summarize_variant(trace_rows)
        summary["round_trace_files"] = len(paths)
        summary["request_mapping"] = "Unavailable: trace rows have server-internal task_id but no benchmark request_id; manifest order mapping is not enough to prove attribution. Aggregates remain per variant."
        variant_results[variant] = summary
    report: dict[str, Any] = {
        "source": str(run_dir),
        "records": len(records),
        "variants": variant_results,
        "limitations": [
            "Round-trace data is CPU wall timing and does not include CUDA event timings.",
            "Trace schema does not carry benchmark request IDs; per-request round association is unavailable.",
            "Benchmark records may contain aggregate quality values; round quality below is computed from trace rows and excludes checkpoint_replay status.",
        ],
    }
    if replay_path is not None:
        report["operator_replay"] = summarize_replay(read_jsonl(replay_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="benchmark run directory containing records.json and rep-*/variant/round-trace.jsonl")
    parser.add_argument("--operator-replay", type=Path, help="JSONL rows with K/M/N/name/group/replay_bits/samples_us")
    parser.add_argument("--output", type=Path, help="write report JSON here; defaults to stdout")
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.operator_replay)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
