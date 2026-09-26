#!/usr/bin/env python3
"""Offline summaries for W1Ax round traces and identical-input operator replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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
TRACE_SPAN_STAGES = (
    "draft", "checkpoint", "target_decode_sync", "process", "check", "kv_repair", "accept_hook",
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


def _trace_span_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    union_values: list[float] = []
    overlap_values: list[float] = []
    outside_values: list[float] = []
    unclamped_residual_values: list[float] = []
    union_residual_values: list[float] = []
    begin_values: list[float] = []
    details = []
    totals = {"invalid_spans": 0, "out_of_bounds_spans": 0, "overlap_rows": 0, "nested_pairs": 0, "clamped_runtime_residual_rows": 0, "runtime_residual_mismatch_rows": 0}

    for row in rows:
        if row.get("schema") != TRACE_SCHEMA:
            continue
        start, end = row.get("round_start_us"), row.get("round_end_us")
        problems = []
        if not _nonnegative_number(start) or not _nonnegative_number(end) or end < start:
            details.append({"task_id": row.get("task_id"), "round_index": row.get("round_index"), "status": "invalid_round_bounds", "flags": ["invalid_round_bounds"]})
            totals["invalid_spans"] += 1
            continue
        duration = end - start
        span_map = row.get("spans_us")
        if not isinstance(span_map, dict):
            details.append({"task_id": row.get("task_id"), "round_index": row.get("round_index"), "status": "missing_spans", "flags": ["missing_spans"]})
            continue

        # begin is a separate pre-round operation by contract and is measured, not attributed to round_us.
        begin_interval = span_map.get("begin")
        if isinstance(begin_interval, list) and len(begin_interval) == 2 and all(_nonnegative_number(v) for v in begin_interval) and begin_interval[1] >= begin_interval[0]:
            begin_values.append(float(begin_interval[1] - begin_interval[0]))

        intervals = []
        outside_by_stage = {}
        invalid_stages = []
        for stage in TRACE_SPAN_STAGES:
            interval = span_map.get(stage)
            if interval is None or interval == [0, 0]:
                continue
            if not isinstance(interval, list) or len(interval) != 2 or not all(_nonnegative_number(v) for v in interval):
                invalid_stages.append(stage)
                continue
            span_start, span_end = interval
            if span_end < span_start:
                invalid_stages.append(stage)
                continue
            if span_start == 0 and span_end == 0:
                continue
            clipped_start = max(float(start), float(span_start))
            clipped_end = min(float(end), float(span_end))
            in_round_duration = max(0.0, clipped_end - clipped_start)
            outside = (span_end - span_start) - in_round_duration
            if outside > 0:
                outside_by_stage[stage] = outside
            if clipped_end > clipped_start:
                intervals.append((clipped_start, clipped_end, stage))

        sorted_intervals = sorted(intervals)
        attributed_duration = sum(right - left for left, right, _ in sorted_intervals)
        merged = []
        nested_pairs = []
        for i, (left, right, stage) in enumerate(sorted_intervals):
            for other_left, other_right, other_stage in sorted_intervals[i + 1:]:
                if other_left >= right:
                    break
                if other_left >= left and other_right <= right:
                    nested_pairs.append([stage, other_stage])
                elif left >= other_left and right <= other_right:
                    nested_pairs.append([other_stage, stage])
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        union_duration = sum(right - left for left, right in merged)
        overlap = attributed_duration - union_duration
        outside_total = sum(outside_by_stage.values())
        unclamped = duration - attributed_duration
        union_residual = duration - union_duration
        runtime_residual = row.get("residual_us")
        if invalid_stages:
            problems.append("invalid_spans")
            totals["invalid_spans"] += len(invalid_stages)
        if outside_by_stage:
            problems.append("out_of_bounds_spans")
            totals["out_of_bounds_spans"] += len(outside_by_stage)
        if overlap > 0:
            problems.append("overlapping_spans")
            totals["overlap_rows"] += 1
        if nested_pairs:
            problems.append("nested_spans")
            totals["nested_pairs"] += len(nested_pairs)
        if unclamped < 0 and _nonnegative_number(runtime_residual) and runtime_residual == 0:
            problems.append("runtime_residual_was_clamped")
            totals["clamped_runtime_residual_rows"] += 1
        expected_runtime_residual = max(0.0, duration - attributed_duration)
        if _nonnegative_number(runtime_residual) and abs(runtime_residual - expected_runtime_residual) > 1.0:
            problems.append("runtime_residual_mismatch")
            totals["runtime_residual_mismatch_rows"] += 1
        runtime_round = row.get("round_us")
        round_delta = runtime_round - duration if _nonnegative_number(runtime_round) else None
        if round_delta is not None and abs(round_delta) > 1.0:
            problems.append("runtime_round_duration_mismatch")
        union_values.append(union_duration)
        overlap_values.append(overlap)
        outside_values.append(outside_total)
        unclamped_residual_values.append(unclamped)
        union_residual_values.append(union_residual)
        details.append({
            "task_id": row.get("task_id"), "round_index": row.get("round_index"),
            "round_duration_us": duration, "attributed_union_us": union_duration,
            "runtime_round_us": runtime_round, "runtime_round_duration_delta_us": round_delta,
            "overlap_us": overlap, "outside_round_us": outside_total,
            "outside_by_stage_us": outside_by_stage,
            "unclamped_residual_us": unclamped, "union_unattributed_us": union_residual,
            "nested_stage_pairs": nested_pairs, "invalid_stages": invalid_stages,
            "runtime_residual_us": runtime_residual, "flags": problems,
        })

    return {
        "rows_with_round_bounds": len(details),
        "totals": totals,
        "distributions_us": {
            "attributed_union": distribution(union_values),
            "overlap": distribution(overlap_values),
            "outside_round": distribution(outside_values),
            "unclamped_residual": distribution(unclamped_residual_values),
            "union_unattributed": distribution(union_residual_values),
            "begin_outside_round": distribution(begin_values),
        },
        "per_round": details,
        "definitions": "Union uses clipped non-begin spans inside [round_start_us, round_end_us]. overlap_us is summed in-round span duration minus interval union. unclamped_residual_us is round duration minus summed in-round span duration and can be negative. begin is reported separately and excluded from round accounting.",
    }


def summarize_variant(
    rows: list[dict[str, Any]],
    trace_scope: str = "All provided trace rows; may include warmup task groups because request mapping has not been applied.",
) -> dict[str, Any]:
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
        "quality_scope": trace_scope,
        "trace_span_accounting": _trace_span_accounting(rows),
        "cpu_wall_us": {
            "round": distribution(round_values),
            "begin_outside_round": distribution(begin_values),
            "stages": {stage: distribution(values) for stage, values in stage_values.items()},
            "note": f"CPU wall spans across {trace_scope.lower()} Named spans may overlap. residual_us is the server-reported remainder and is separately audited in trace_span_accounting. No CUDA event data is present.",
        },
    }


def _sample_summary(samples: Any, path: Path, line: int) -> list[float]:
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}:{line}: samples_us must be a non-empty array")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError(f"{path}:{line}: samples_us must contain finite non-negative numbers")
    return [float(v) for v in samples]


ACTIVATION_RATE_FIELDS = ("source_zero_rate", "code_zero_rate", "clip_rate", "saturation_rate")
ACTIVATION_ERROR_FIELDS = ("mae", "rmse", "max_abs_error")


def _validate_activation_diagnostics(row: dict[str, Any], bits: int, row_number: int) -> dict[str, Any] | None:
    if "activation" not in row:
        return None  # Older replay logs predate per-activation diagnostics.
    activation = row["activation"]
    if not isinstance(activation, dict):
        raise ValueError(f"operator replay row {row_number}: activation must be an object or absent")
    for field in ("elements", *ACTIVATION_RATE_FIELDS, *ACTIVATION_ERROR_FIELDS):
        if field not in activation:
            raise ValueError(f"operator replay row {row_number}: activation missing {field}")
    elements = activation["elements"]
    k, n = row.get("K"), row.get("N")
    if (
        not isinstance(elements, int) or isinstance(elements, bool) or elements <= 0
        or not isinstance(k, int) or isinstance(k, bool) or k <= 0
        or not isinstance(n, int) or isinstance(n, bool) or n <= 0
        or elements != k * n
    ):
        raise ValueError(f"operator replay row {row_number}: activation.elements must equal positive K*N")
    values = {"elements": elements}
    nullable = {
        "source_zero_rate": False,
        "code_zero_rate": bits == 1,
        "clip_rate": bits == 1,
        "saturation_rate": bits in (1, 16),
    }
    for field in ACTIVATION_RATE_FIELDS:
        value = activation[field]
        if value is None:
            if not nullable[field]:
                raise ValueError(f"operator replay row {row_number}: activation.{field} unexpectedly null for {bits} bits")
            values[field] = None
        elif not _nonnegative_number(value) or value > 1:
            raise ValueError(f"operator replay row {row_number}: activation.{field} must be finite in [0, 1]")
        else:
            values[field] = float(value)
    for field in ACTIVATION_ERROR_FIELDS:
        value = activation[field]
        if not _nonnegative_number(value):
            raise ValueError(f"operator replay row {row_number}: activation.{field} must be finite and non-negative")
        values[field] = float(value)
    return values


def _aggregate_activation_rows(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    present = [row for row in rows if row is not None]
    result: dict[str, Any] = {
        "captures": len(rows),
        "captures_with_activation": len(present),
        "captures_without_activation": len(rows) - len(present),
        "activation_elements": sum(row["elements"] for row in present),
        "weighting": "Capture-weighted by activation.elements (K*N); selected correlated captures, not live-invocation weighted.",
        "metrics": {},
    }
    for field in (*ACTIVATION_RATE_FIELDS, *ACTIVATION_ERROR_FIELDS):
        available = [row for row in present if row[field] is not None]
        elements = sum(row["elements"] for row in available)
        unavailable_present = len(present) - len(available)
        metric = {
            "captures_with_value": len(available),
            "captures_without_value": len(rows) - len(available),
            "not_applicable_captures": unavailable_present,
            "missing_activation_captures": len(rows) - len(present),
            "element_count": elements,
        }
        if not available:
            metric["value"] = None
        elif field == "max_abs_error":
            metric["value"] = max(row[field] for row in available)
        elif field == "rmse":
            metric["value"] = math.hypot(*[
                row[field] * math.sqrt(row["elements"] / elements) for row in available
            ])
        else:
            metric["value"] = math.fsum(row[field] * (row["elements"] / elements) for row in available)
        if metric["value"] is not None and not math.isfinite(metric["value"]):
            raise ValueError(f"activation aggregate {field} is non-finite")
        result["metrics"][field] = metric
    return result


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    captures: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(dict)
    shapes: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    paired_by_shape: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    capture_shapes: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    anchor_captures: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(dict)
    anchor_shapes: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    anchor_capture_shapes: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    w1ax_anchor_ratios: dict[tuple[Any, ...], dict[tuple[int, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    anchor_groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    operator_rows = []
    activation_shape_rows: dict[tuple[Any, ...], dict[int, list[dict[str, Any] | None]]] = defaultdict(lambda: defaultdict(list))
    activation_layer_rows: dict[tuple[Any, ...], dict[int, list[dict[str, Any] | None]]] = defaultdict(lambda: defaultdict(list))
    activation_layer_shapes: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    anchor_rows = []
    head_rows = []
    fp16_cast_rows = []
    fp16_cast_head_rows = []
    for i, row in enumerate(rows, 1):
        record_type = row.get("record_type", "operator_replay")
        if record_type == "head_comparison":
            head_rows.append(row)
            continue
        if record_type == "fp16_cast_control":
            fp16_cast_rows.append(row)
            continue
        if record_type == "fp16_cast_head_comparison":
            fp16_cast_head_rows.append(row)
            continue
        if record_type == "anchor_operator_replay":
            for field in ("capture", "sequence", "K", "M", "N", "name", "anchor_format"):
                if field not in row:
                    raise ValueError(f"anchor operator replay row {i} missing {field}")
            anchor_format = row["anchor_format"]
            if anchor_format not in ("fp16", "q8_0", "q4_0"):
                raise ValueError(f"anchor operator replay row {i}: unknown anchor_format {anchor_format!r}")
            samples = _sample_summary(row.get("samples_us"), Path("operator replay JSONL"), i)
            shape_key = (row["K"], row["M"], row["N"], row["name"])
            capture_key = (
                row.get("capture", "<legacy>"), row.get("sequence", 0), row["name"],
                row["K"], row["M"], row["N"],
            )
            if anchor_format in anchor_captures[capture_key]:
                raise ValueError(f"anchor operator replay row {i}: duplicate anchor_format for capture identity {capture_key} ({anchor_format})")
            anchor_captures[capture_key][anchor_format] = samples
            anchor_shapes[shape_key][anchor_format].extend(samples)
            anchor_capture_shapes[shape_key].add(capture_key)
            if isinstance(row.get("group"), str):
                anchor_groups[shape_key].add(row["group"])
            anchor_rows.append((shape_key, capture_key, anchor_format, samples))
            continue
        if record_type != "operator_replay":
            continue
        for field in ("K", "M", "N", "name", "group", "replay_bits"):
            if field not in row:
                raise ValueError(f"operator replay row {i} missing {field}")
        bits = row["replay_bits"]
        if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
            raise ValueError(f"operator replay row {i}: replay_bits must be a positive integer")
        samples = _sample_summary(row.get("samples_us"), Path("operator replay JSONL"), i)
        shape_key = (row["K"], row["M"], row["N"], row["name"], row["group"])
        activation_diagnostics = _validate_activation_diagnostics(row, bits, i)
        capture_key = (
            row.get("capture", "<legacy>"), row.get("sequence", 0), row["name"],
            row["K"], row["M"], row["N"],
        )
        if bits in captures[capture_key]:
            raise ValueError(f"operator replay row {i}: duplicate precision row for capture identity {capture_key} ({bits} bits)")
        captures[capture_key][bits] = samples
        shapes[shape_key][bits].extend(samples)
        capture_shapes[shape_key].add(capture_key)
        activation_shape_rows[shape_key][bits].append(activation_diagnostics)
        layer_key = (row["name"], row["group"])
        activation_layer_rows[layer_key][bits].append(activation_diagnostics)
        activation_layer_shapes[layer_key].add(shape_key)
        operator_rows.append((shape_key, capture_key, bits, samples))

    for shape_key, capture_key, bits, samples in operator_rows:
        for anchor_format, baseline in anchor_captures.get(capture_key, {}).items():
            if len(baseline) != len(samples):
                continue
            ratios = [candidate / reference for candidate, reference in zip(samples, baseline) if reference > 0]
            if len(ratios) == len(samples):
                base_shape = shape_key[:4]
                w1ax_anchor_ratios[base_shape][(bits, anchor_format)].extend(ratios)
        if bits == 16:
            continue
        baseline = captures[capture_key].get(16)
        if baseline is None or len(baseline) != len(samples):
            continue
        ratios = [candidate / reference for candidate, reference in zip(samples, baseline) if reference > 0]
        if len(ratios) == len(samples):
            paired_by_shape[shape_key][bits].extend(ratios)

    by_layer_shape = []
    for key in sorted(shapes, key=lambda x: tuple(str(part) for part in x)):
        entries = shapes[key]
        precision = {}
        for bits, samples in sorted(entries.items()):
            precision[str(bits)] = {
                "samples": len(samples),
                "median_us": median(samples),
                "distribution_us": distribution(samples),
                "activation": _aggregate_activation_rows(activation_shape_rows[key][bits]),
            }
        ratios = {}
        for bits in sorted(entries):
            if bits == 16:
                continue
            paired = paired_by_shape[key].get(bits, [])
            ratios[str(bits)] = {
                "paired_precision_ratio_vs_16": median(paired) if paired else None,
                "paired_sample_ratios": paired,
                "paired_samples": len(paired),
                "reason": None if paired else "no capture has a matching 16-bit baseline and sample count",
            }
        by_layer_shape.append({
            "K": key[0], "M": key[1], "N": key[2], "name": key[3], "group": key[4],
            "captures": len(capture_shapes[key]), "precision": precision, "paired_ratios": ratios,
        })
    activation_by_layer = []
    for key in sorted(activation_layer_rows, key=lambda x: tuple(str(part) for part in x)):
        activation_by_layer.append({
            "name": key[0], "group": key[1], "shapes": len(activation_layer_shapes[key]),
            "by_replay_bits": {
                str(bits): _aggregate_activation_rows(activation_layer_rows[key][bits])
                for bits in sorted(activation_layer_rows[key])
            },
        })
    anchor_by_layer_shape = []
    for key in sorted(anchor_shapes, key=lambda x: tuple(str(part) for part in x)):
        anchor_by_layer_shape.append({
            "K": key[0], "M": key[1], "N": key[2], "name": key[3],
            "group": next(iter(anchor_groups[key])) if len(anchor_groups[key]) == 1 else None,
            "captures": len(anchor_capture_shapes[key]),
            "formats": {
                anchor_format: {
                    "samples": len(samples),
                    "median_us": median(samples),
                    "distribution_us": distribution(samples),
                }
                for anchor_format, samples in sorted(anchor_shapes[key].items())
            },
        })
    w1ax_vs_anchors = []
    cross_shapes = sorted(
        {entry[0][:4] for entry in operator_rows} & set(anchor_shapes),
        key=lambda x: tuple(str(part) for part in x),
    )
    for key in cross_shapes:
        bits_present = sorted({entry[2] for entry in operator_rows if entry[0][:4] == key})
        w1ax_vs_anchors.append({
            "K": key[0], "M": key[1], "N": key[2], "name": key[3],
            "w1ax_vs_anchor": {
                f"w1a{bits}/{anchor_format}": {
                    "paired_precision_ratio": (
                        median(w1ax_anchor_ratios[key].get((bits, anchor_format), []))
                        if w1ax_anchor_ratios[key].get((bits, anchor_format)) else None
                    ),
                    "paired_samples": len(w1ax_anchor_ratios[key].get((bits, anchor_format), [])),
                    "paired_sample_ratios": w1ax_anchor_ratios[key].get((bits, anchor_format), []),
                    "ratio_definition": "W1Ax sample_us / anchor sample_us; lower is faster",
                }
                for bits in bits_present
                for anchor_format in ("fp16", "q8_0", "q4_0")
                if anchor_format in anchor_shapes[key]
            },
        })
    return {
        "rows": len(operator_rows),
        "by_layer_shape": by_layer_shape,
        "pairing": "Rows are paired within capture/sequence/name/K/M/N and sample index; replay_bits=16 is the ratio baseline. Shape summaries pool repeated captures. The schema has no tensor hashes, so capture identity is trusted from the producer.",
        "activation_diagnostics": {
            "by_layer": activation_by_layer,
            "weighting": "Per-row activation.elements equals K*N. Rates and MAE are weighted by elements; RMSE is sqrt(weighted mean squared RMSE); max_abs_error is the maximum across captures. These summarize selected correlated capture rows, not live-invocation weighted distributions.",
            "producer_denominator": "The replay producer divides source/code/clip/saturation counts, MAE numerator, and squared-error numerator by K*N. On FP16 cast overflow, it increments clip_rate and skips that element's error numerator while retaining K*N as the MAE/RMSE denominator.",
            "legacy_missing_activation": "Rows without an activation object remain in capture counts and are explicitly counted as captures_without_activation; no diagnostic values are imputed.",
        },
        "anchor_rows": len(anchor_rows),
        "anchor_by_layer_shape": anchor_by_layer_shape,
        "w1ax_vs_anchors": w1ax_vs_anchors,
        "head_comparison": _summarize_head_comparisons(head_rows),
        "fp16_cast_control": _summarize_fp16_cast_controls(fp16_cast_rows),
        "fp16_cast_head_comparison": _summarize_fp16_cast_heads(fp16_cast_head_rows),
    }


def _summarize_head_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bits: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        bits = row.get("candidate_bits")
        overlap = row.get("topk_set_overlap")
        if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
            raise ValueError(f"head comparison row {index}: candidate_bits must be a positive integer")
        if not isinstance(overlap, dict):
            raise ValueError(f"head comparison row {index}: topk_set_overlap must be an object")
        by_bits[bits].append(row)
    result = {}
    for bits, selected in sorted(by_bits.items()):
        ks = sorted({str(k) for row in selected for k in row["topk_set_overlap"]}, key=int)
        overlap_summary = {}
        for k in ks:
            values = [
                row["topk_set_overlap"][k] / int(k)
                for row in selected
                if _nonnegative_number(row["topk_set_overlap"].get(k))
            ]
            overlap_summary[k] = distribution(values)
        result[str(bits)] = {
            "comparisons": len(selected),
            "captures": len({(row.get("capture"), row.get("sequence")) for row in selected}),
            "top1_agreement_rate": (
                sum(row["top1_agree"] is True for row in selected) / len(selected)
                if all(isinstance(row.get("top1_agree"), bool) for row in selected) else None
            ),
            "topk_overlap_fraction": overlap_summary,
            "reference_top1_margin": distribution([
                float(row["reference_top1_margin"]) for row in selected
                if _nonnegative_number(row.get("reference_top1_margin"))
            ]),
            "candidate_top1_margin": distribution([
                float(row["candidate_top1_margin"]) for row in selected
                if _nonnegative_number(row.get("candidate_top1_margin"))
            ]),
        }
    return {"comparisons": len(rows), "by_candidate_bits": result}


def _summarize_fp16_cast_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        for field in ("K", "M", "N", "name", "mean_abs_output_difference", "max_abs_output_difference"):
            if field not in row:
                raise ValueError(f"fp16 cast control row {index} missing {field}")
        key = (row["K"], row["M"], row["N"], row["name"], row.get("group"))
        groups[key].append(row)
    return {
        "records": len(rows),
        "by_layer_shape": [
            {
                "K": key[0], "M": key[1], "N": key[2], "name": key[3], "group": key[4],
                "captures": len({(row.get("capture"), row.get("sequence")) for row in selected}),
                "mean_abs_output_difference": distribution([
                    float(row["mean_abs_output_difference"]) for row in selected
                    if _nonnegative_number(row.get("mean_abs_output_difference"))
                ]),
                "max_abs_output_difference": distribution([
                    float(row["max_abs_output_difference"]) for row in selected
                    if _nonnegative_number(row.get("max_abs_output_difference"))
                ]),
            }
            for key, selected in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))
        ],
        "scope": "Diagnostic numerical comparison only; not a serving acceptance or throughput metric.",
    }


def _summarize_fp16_cast_heads(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("top1_agree"), bool) or not _nonnegative_number(row.get("top5_set_overlap")):
            raise ValueError(f"fp16 cast head comparison row {index} has invalid top-k fields")
        if row["top5_set_overlap"] > 5:
            raise ValueError(f"fp16 cast head comparison row {index}: top5_set_overlap exceeds 5")
    return {
        "comparisons": len(rows),
        "captures": len({(row.get("capture"), row.get("sequence")) for row in rows}),
        "top1_agreement_rate": sum(row["top1_agree"] for row in rows) / len(rows) if rows else None,
        "top5_overlap_fraction": distribution([row["top5_set_overlap"] / 5 for row in rows]),
        "reference_top1_margin": distribution([
            float(row["reference_top1_margin"]) for row in rows
            if _nonnegative_number(row.get("reference_top1_margin"))
        ]),
        "cast_top1_margin": distribution([
            float(row["cast_top1_margin"]) for row in rows
            if _nonnegative_number(row.get("cast_top1_margin"))
        ]),
        "reference_top5_cutoff_margin": distribution([
            float(row["reference_top5_cutoff_margin"]) for row in rows
            if _nonnegative_number(row.get("reference_top5_cutoff_margin"))
        ]),
        "cast_top5_cutoff_margin": distribution([
            float(row["cast_top5_cutoff_margin"]) for row in rows
            if _nonnegative_number(row.get("cast_top5_cutoff_margin"))
        ]),
        "scope": "Diagnostic numerical comparison only; not a serving acceptance or throughput metric.",
    }


def _single_server_parallelism(manifest: dict[str, Any], variant: str) -> int:
    command = manifest.get("commands", {}).get(variant)
    if not isinstance(command, list) or any(not isinstance(arg, str) for arg in command):
        raise ValueError(f"cannot validate task mapping for {variant}: manifest command is missing")
    values = []
    for index, arg in enumerate(command):
        if arg in ("--parallel", "-np"):
            if index + 1 >= len(command):
                raise ValueError(f"cannot validate task mapping for {variant}: {arg} has no value")
            values.append(command[index + 1])
        elif arg.startswith("--parallel="):
            values.append(arg.split("=", 1)[1])
    if not values or any(value != "1" for value in values):
        raise ValueError(f"cannot validate task mapping for {variant}: server concurrency is not explicitly one")
    return 1


def map_measured_trace_rows(
    trace_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    repetition: int,
    variant: str,
    warmup_requests: int,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map serial task groups to measured requests, proving each match by raw token IDs."""
    _single_server_parallelism(manifest, variant)
    if not isinstance(warmup_requests, int) or isinstance(warmup_requests, bool) or warmup_requests < 0:
        raise ValueError("manifest policy warmup_requests must be a non-negative integer")
    selected = [r for r in records if r.get("repetition") == repetition and r.get("variant") == variant]
    indexed = {}
    for record in selected:
        request_index = record.get("server_request_index")
        if not isinstance(request_index, int) or isinstance(request_index, bool):
            raise ValueError(f"{variant}/rep-{repetition:02d}: measured record lacks server_request_index")
        if request_index in indexed:
            raise ValueError(f"{variant}/rep-{repetition:02d}: duplicate server_request_index {request_index}")
        indexed[request_index] = record
    expected_indices = list(range(warmup_requests, warmup_requests + len(selected)))
    if sorted(indexed) != expected_indices:
        raise ValueError(
            f"{variant}/rep-{repetition:02d}: measured server_request_index values do not match "
            f"warmup offset {warmup_requests} and {len(selected)} records"
        )
    measured = [indexed[index] for index in expected_indices]
    prompt_ids = manifest.get("prompt_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != len(measured):
        raise ValueError(f"{variant}/rep-{repetition:02d}: manifest prompt_ids do not match measured record count")
    for offset, (record, prompt_id) in enumerate(zip(measured, prompt_ids)):
        if record.get("prompt_id") != prompt_id:
            raise ValueError(
                f"{variant}/rep-{repetition:02d}: record order at server_request_index "
                f"{warmup_requests + offset} disagrees with manifest prompt_ids"
            )

    if not trace_rows:
        raise ValueError(f"{variant}/rep-{repetition:02d}: round trace is empty; task mapping cannot be validated")
    blocks: list[list[dict[str, Any]]] = []
    seen_task_ids = set()
    last_start = None
    last_task_id = None
    previous_round_end = None
    for row in trace_rows:
        if row.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"{variant}/rep-{repetition:02d}: unexpected trace schema {row.get('schema')!r}")
        task_id = row.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError(f"{variant}/rep-{repetition:02d}: invalid task_id in round trace")
        start = row.get("round_start_us")
        end = row.get("round_end_us")
        if not _nonnegative_number(start) or not _nonnegative_number(end) or end <= start:
            raise ValueError(f"{variant}/rep-{repetition:02d}: task {task_id} lacks a valid round_start_us")
        if last_start is not None and start <= last_start:
            raise ValueError(f"{variant}/rep-{repetition:02d}: trace rows are not chronological")
        if previous_round_end is not None and start < previous_round_end:
            raise ValueError(f"{variant}/rep-{repetition:02d}: trace round intervals overlap or interleave")
        last_start = start
        previous_round_end = end
        if task_id != last_task_id:
            if task_id in seen_task_ids:
                raise ValueError(f"{variant}/rep-{repetition:02d}: task groups are interleaved")
            blocks.append([])
            seen_task_ids.add(task_id)
            last_task_id = task_id
        blocks[-1].append(row)

    expected_task_count = warmup_requests + len(measured)
    if len(blocks) != expected_task_count:
        raise ValueError(
            f"{variant}/rep-{repetition:02d}: found {len(blocks)} chronological task groups; "
            f"expected {warmup_requests} warmups + {len(measured)} measured requests"
        )

    mapping = []
    measured_rows = []
    prefix_omissions = set()
    response_generated_tokens_total = 0
    trace_emitted_tokens_total = 0
    untraced_leading_tokens_total = 0
    for task_position, block in enumerate(blocks):
        task_id = block[0]["task_id"]
        parent_ids = {row.get("parent_task_id") for row in block}
        if len(parent_ids) != 1:
            raise ValueError(f"{variant}/rep-{repetition:02d}: task {task_id} changes parent_task_id")
        round_indices = [row.get("round_index") for row in block]
        if round_indices != list(range(len(block))):
            raise ValueError(f"{variant}/rep-{repetition:02d}: task {task_id} round_index is not contiguous from zero")
        emitted = []
        for row in block:
            token_ids = row.get("emitted_token_ids")
            if not isinstance(token_ids, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids):
                raise ValueError(f"{variant}/rep-{repetition:02d}: task {task_id} has invalid emitted_token_ids")
            if row.get("n_emitted") != len(token_ids):
                raise ValueError(f"{variant}/rep-{repetition:02d}: task {task_id} n_emitted disagrees with emitted_token_ids")
            if row.get("status") != "checkpoint_replay":
                emitted.extend(token_ids)

        if task_position < warmup_requests:
            mapping.append({"task_id": task_id, "request_kind": "warmup", "server_request_index": task_position})
            continue
        request_index = expected_indices[task_position - warmup_requests]
        record = indexed[request_index]
        expected_ids = record.get("generated_token_ids")
        if not isinstance(expected_ids, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in expected_ids):
            raise ValueError(f"{variant}/rep-{repetition:02d}: record at server_request_index {request_index} lacks raw generated token IDs")
        if emitted != expected_ids:
            if len(expected_ids) > 0 and emitted == expected_ids[1:]:
                prefix_omissions.add(1)
                alignment = "one_untraced_leading_token"
                omitted_count = 1
            else:
                raise ValueError(
                    f"{variant}/rep-{repetition:02d}: task {task_id} emitted IDs do not match the full "
                    f"record or its one-token-leading-omission form for "
                    f"{record.get('request_id', request_index)} at server_request_index {request_index}"
                )
        else:
            prefix_omissions.add(0)
            omitted_count = 0
            alignment = "full_sequence"
        response_generated_tokens_total += len(expected_ids)
        trace_emitted_tokens_total += len(emitted)
        untraced_leading_tokens_total += omitted_count
        measured_rows.extend({**row, "request_id": record.get("request_id"), "prompt_id": record.get("prompt_id")} for row in block)
        mapping.append({
            "task_id": task_id, "request_kind": "measured", "server_request_index": request_index,
            "request_id": record.get("request_id"), "prompt_id": record.get("prompt_id"),
            "emitted_token_ids_match": True,
            "trace_emission_alignment": alignment,
            "response_generated_token_count": len(expected_ids),
            "trace_emitted_token_count": len(emitted),
            "untraced_leading_token_count": omitted_count,
        })
    if len(prefix_omissions) > 1:
        raise ValueError(f"{variant}/rep-{repetition:02d}: inconsistent leading-token omission across measured task groups")
    return measured_rows, {
        "status": "validated",
        "method": "serial chronological task groups aligned to runner server_request_index; measured matches require equality of concatenated non-checkpoint-replay emitted_token_ids to records.generated_token_ids, allowing either exact equality or a uniform one-token leading omission",
        "warmup_tasks_excluded": warmup_requests,
        "measured_requests_mapped": len(measured),
        "untraced_leading_tokens_per_measured_request": next(iter(prefix_omissions)) if prefix_omissions else None,
        "response_generated_tokens_total": response_generated_tokens_total,
        "trace_emitted_tokens_total": trace_emitted_tokens_total,
        "untraced_leading_tokens_total": untraced_leading_tokens_total,
        "task_mappings": mapping,
    }


def _counter_field(record: dict[str, Any], field: str) -> int | None:
    value = (record.get("speculative") or {}).get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def summarize_counter_reconciliation(
    measured_trace_rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare mapped trace counters with request-local metrics without assuming equality."""
    trace_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured_trace_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("mapped trace row lacks request_id for counter reconciliation")
        trace_by_request[request_id].append(row)

    per_request = []
    statuses = Counter()
    trace_totals = {
        "accepted": 0, "accepted_actually_emitted": 0, "emitted": 0,
        "proposed_quality": 0, "proposed_attempts": 0,
        "quality_trace_rows": 0, "verified_rounds": 0,
        "proposal_positive_rounds": 0, "proposal_zero_rounds": 0,
        "accepted_not_emitted": 0,
    }
    native_values: dict[str, list[int]] = {key: [] for key in ("accepted", "proposed", "rounds")}
    native_missing = {key: 0 for key in native_values}
    response_token_counts: list[int] = []
    response_tokens_missing = 0
    completion_token_counts: list[int] = []
    completion_tokens_missing = 0
    for record in sorted(records, key=lambda r: (r.get("repetition", -1), r.get("server_request_index", -1))):
        request_id = record.get("request_id")
        rows = trace_by_request.get(request_id, [])
        if not rows:
            raise ValueError(f"mapped request {request_id!r} has no trace rows for counter reconciliation")
        row_statuses = Counter(row.get("status") for row in rows)
        statuses.update(row_statuses)
        quality = [row for row in rows if row.get("status") != "checkpoint_replay"]
        accepted = sum(row.get("n_accepted", 0) for row in quality)
        emitted = sum(row.get("n_emitted", 0) for row in quality)
        accepted_emitted = sum(min(row.get("n_accepted", 0), row.get("n_emitted", 0)) for row in quality)
        proposed_quality = sum(row.get("n_proposed", 0) for row in quality)
        proposed_attempts = sum(row.get("n_proposed", 0) for row in rows)
        verified_rounds = sum(row.get("status") not in ("checkpoint_replay", "no_proposal") for row in rows)
        positive = sum(row.get("n_proposed", 0) > 0 for row in quality)
        zero = sum(row.get("n_proposed", 0) == 0 for row in quality)
        accepted_after_stop = sum(max(0, row.get("n_accepted", 0) - row.get("n_emitted", 0)) for row in quality)
        native = {field: _counter_field(record, field) for field in native_values}
        for field, value in native.items():
            if value is None:
                native_missing[field] += 1
            else:
                native_values[field].append(value)
        generated_ids = record.get("generated_token_ids")
        if isinstance(generated_ids, list) and all(isinstance(token, int) and not isinstance(token, bool) for token in generated_ids):
            response_token_counts.append(len(generated_ids))
            response_count = len(generated_ids)
        else:
            response_tokens_missing += 1
            response_count = None
        completion_count = record.get("completion_tokens")
        if isinstance(completion_count, int) and not isinstance(completion_count, bool) and completion_count >= 0:
            completion_token_counts.append(completion_count)
        else:
            completion_tokens_missing += 1
            completion_count = None
        differences = {
            "accepted_quality_trace_minus_api_native": accepted - native["accepted"] if native["accepted"] is not None else None,
            "proposed_quality_trace_minus_api_native": proposed_quality - native["proposed"] if native["proposed"] is not None else None,
            "proposed_all_attempt_trace_minus_api_native": proposed_attempts - native["proposed"] if native["proposed"] is not None else None,
            "quality_trace_rows_minus_api_native_rounds": len(quality) - native["rounds"] if native["rounds"] is not None else None,
            "verified_trace_rounds_minus_api_native_rounds": verified_rounds - native["rounds"] if native["rounds"] is not None else None,
            "accepted_actually_emitted_minus_api_native": accepted_emitted - native["accepted"] if native["accepted"] is not None else None,
            "trace_emitted_minus_response_generated_token_ids": emitted - response_count if response_count is not None else None,
            "trace_emitted_minus_api_completion_tokens": emitted - completion_count if completion_count is not None else None,
        }
        per_request.append({
            "repetition": record.get("repetition"), "variant": record.get("variant"),
            "prompt_id": record.get("prompt_id"), "request_id": request_id,
            "response_generated_token_ids": response_count,
            "api_completion_tokens": completion_count,
            "status_counts": dict(row_statuses),
            "trace": {
                "accepted_quality": accepted, "accepted_actually_emitted": accepted_emitted,
                "accepted_quality_not_emitted": accepted_after_stop,
                "emitted": emitted, "proposed_quality": proposed_quality,
                "proposed_all_attempts": proposed_attempts,
                "quality_trace_rows": len(quality), "verified_rounds": verified_rounds,
                "proposal_positive_rounds": positive, "proposal_zero_rounds": zero,
                "accepted_per_proposed_quality": _safe_ratio(accepted, proposed_quality),
                "accepted_per_verified_round": _safe_ratio(accepted, verified_rounds),
            },
            "api_native": native,
            "differences": differences,
        })
        trace_totals["accepted"] += accepted
        trace_totals["accepted_actually_emitted"] += accepted_emitted
        trace_totals["emitted"] += emitted
        trace_totals["proposed_quality"] += proposed_quality
        trace_totals["proposed_attempts"] += proposed_attempts
        trace_totals["quality_trace_rows"] += len(quality)
        trace_totals["verified_rounds"] += verified_rounds
        trace_totals["proposal_positive_rounds"] += positive
        trace_totals["proposal_zero_rounds"] += zero
        trace_totals["accepted_not_emitted"] += accepted_after_stop

    native_totals = {
        key: sum(values) if native_missing[key] == 0 and len(per_request) else None
        for key, values in native_values.items()
    }
    response_tokens_total = sum(response_token_counts) if response_tokens_missing == 0 else None
    completion_tokens_total = sum(completion_token_counts) if completion_tokens_missing == 0 else None
    differences = {
        "accepted_quality_trace_minus_api_native": trace_totals["accepted"] - native_totals["accepted"] if native_totals["accepted"] is not None else None,
        "proposed_quality_trace_minus_api_native": trace_totals["proposed_quality"] - native_totals["proposed"] if native_totals["proposed"] is not None else None,
        "proposed_all_attempt_trace_minus_api_native": trace_totals["proposed_attempts"] - native_totals["proposed"] if native_totals["proposed"] is not None else None,
        "quality_trace_rows_minus_api_native_rounds": trace_totals["quality_trace_rows"] - native_totals["rounds"] if native_totals["rounds"] is not None else None,
        "verified_trace_rounds_minus_api_native_rounds": trace_totals["verified_rounds"] - native_totals["rounds"] if native_totals["rounds"] is not None else None,
        "accepted_actually_emitted_minus_api_native": trace_totals["accepted_actually_emitted"] - native_totals["accepted"] if native_totals["accepted"] is not None else None,
        "trace_emitted_minus_response_generated_token_ids": trace_totals["emitted"] - response_tokens_total if response_tokens_total is not None else None,
        "trace_emitted_minus_api_completion_tokens": trace_totals["emitted"] - completion_tokens_total if completion_tokens_total is not None else None,
    }
    trace_totals["accepted_per_proposed_quality"] = _safe_ratio(trace_totals["accepted"], trace_totals["proposed_quality"])
    trace_totals["accepted_per_verified_round"] = _safe_ratio(trace_totals["accepted"], trace_totals["verified_rounds"])
    trace_totals["accepted_per_quality_trace_row"] = _safe_ratio(trace_totals["accepted"], trace_totals["quality_trace_rows"])
    native_rates = {
        "accepted_per_proposed": _safe_ratio(native_totals["accepted"], native_totals["proposed"]) if all(native_totals[k] is not None for k in native_totals) else None,
        "accepted_per_round": _safe_ratio(native_totals["accepted"], native_totals["rounds"]) if all(native_totals[k] is not None for k in native_totals) else None,
    }
    return {
        "requests": len(per_request),
        "per_request": per_request,
        "pooled": {
            "trace": trace_totals,
            "api_native": {"totals": native_totals, "missing_request_counts": native_missing, "rates": native_rates},
            "api_output": {
                "generated_token_ids_total": response_tokens_total,
                "generated_token_ids_missing_requests": response_tokens_missing,
                "completion_tokens_total": completion_tokens_total,
                "completion_tokens_missing_requests": completion_tokens_missing,
            },
            "status_counts": dict(statuses),
            "differences": differences,
        },
        "semantics": {
            "accepted": "Trace accepted sums n_accepted from non-checkpoint_replay rows. Runtime increments n_draft_accepted after final verifier acceptance and before emitting accepted tokens; accepted_actually_emitted counts the accepted prefix bounded by each row's n_emitted.",
            "proposed": "Runtime increments n_draft_tokens by draft.size() before verification. The API native counter can include a draft attempt later discarded by checkpoint restoration; proposed_all_attempts includes every trace row, while proposed_quality excludes checkpoint_replay rows.",
            "rounds": "Runtime increments n_draft_verif_steps after a successful final verification. `complete` trace rows count verified rounds; `no_proposal` rows are round events without a verifier step. Checkpoint replay rows are excluded from quality counts.",
            "comparison": "Differences are reported, not constrained to zero. Per-request API counters are metrics deltas; trace counters are reconstructed from mapped task rows.",
        },
    }


def analyze_run(run_dir: Path, replay_path: Path | None = None) -> dict[str, Any]:
    records_path = run_dir / "records.json"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    records = json.loads(records_path.read_text())
    if not isinstance(records, list):
        raise ValueError("records.json must contain an array")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    trace_enabled = bool(manifest and manifest.get("round_trace_enabled") is True)
    policy = manifest.get("policy") if isinstance(manifest, dict) else None
    warmups = policy.get("warmup_requests") if isinstance(policy, dict) else None
    variants = sorted({row.get("variant") for row in records if isinstance(row, dict) and row.get("variant")})
    variant_results = {}
    for variant in variants:
        trace_rows = []
        paths = sorted(run_dir.glob(f"rep-*/{variant}/round-trace.jsonl"))
        mapping_reports = []
        repetitions = sorted({row.get("repetition") for row in records if row.get("variant") == variant})
        for repetition in repetitions:
            if not isinstance(repetition, int) or isinstance(repetition, bool):
                raise ValueError(f"{variant}: invalid repetition value in records")
            path = run_dir / f"rep-{repetition:02d}" / variant / "round-trace.jsonl"
            if not trace_enabled:
                if path.is_file() and read_jsonl(path):
                    raise ValueError(f"{variant}/rep-{repetition:02d}: trace rows exist but manifest does not enable round tracing")
                continue
            raw_rows = read_jsonl(path) if path.is_file() else []
            if variant == "target_only":
                if raw_rows:
                    raise ValueError("target_only unexpectedly has speculative round trace rows")
                mapping_reports.append({"repetition": repetition, "status": "not_applicable_target_only"})
                continue
            if manifest is None:
                raise ValueError("manifest.json is required to validate chronological trace task mapping")
            if warmups is None:
                raise ValueError("manifest policy.warmup_requests is required to exclude warmup trace groups")
            measured_rows, mapping = map_measured_trace_rows(
                raw_rows, records, repetition, variant, warmups, manifest
            )
            trace_rows.extend(measured_rows)
            mapping_reports.append({"repetition": repetition, **mapping})
        mapped_offsets = {
            item.get("untraced_leading_tokens_per_measured_request")
            for item in mapping_reports
            if item.get("status") == "validated"
        }
        if len(mapped_offsets) > 1:
            raise ValueError(f"{variant}: trace-emission prefix coverage differs across repetitions")
        if mapped_offsets == {1}:
            emission_scope = "trace emissions omit one leading output token per measured request; this pre-round seed is outside round-level emitted-token counts"
        elif mapped_offsets == {0}:
            emission_scope = "trace emissions exactly cover measured request output token IDs"
        else:
            emission_scope = "no validated trace-emission mapping is available"
        summary = summarize_variant(
            trace_rows,
            trace_scope=(
                "validated measured-request task groups only; warmup task groups were mapped and excluded; "
                f"checkpoint_replay rows are excluded from quality totals; {emission_scope}"
                if trace_enabled and variant != "target_only"
                else "no mapped speculative rows; mapping unavailable or not applicable"
            ),
        )
        summary["round_trace_files"] = len(paths)
        summary["request_mapping"] = {
            "status": "validated" if trace_enabled and variant != "target_only" else (
                "not_applicable_target_only" if variant == "target_only" else "unavailable_trace_disabled"
            ),
            "warmup_requests_per_server": warmups if trace_enabled else None,
            "by_repetition": mapping_reports,
        }
        if trace_enabled and variant != "target_only":
            variant_records = [record for record in records if record.get("variant") == variant]
            summary["counter_reconciliation"] = summarize_counter_reconciliation(trace_rows, variant_records)
        else:
            summary["counter_reconciliation"] = {"status": "not_applicable_or_unavailable"}
        variant_results[variant] = summary
    report: dict[str, Any] = {
        "source": str(run_dir),
        "records": len(records),
        "variants": variant_results,
        "limitations": [
            "Round-trace data is CPU wall timing and does not include CUDA event timings.",
            "Runtime trace rows do not carry benchmark request IDs; per-request mapping is established from explicit single-server concurrency, chronological task groups, runner server_request_index ordering, and exact emitted token ID equality.",
            "Benchmark records may contain aggregate quality values; round quality below is computed from trace rows and excludes checkpoint_replay status.",
        ],
    }
    if replay_path is not None:
        report["operator_replay"] = summarize_replay(read_jsonl(replay_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", help="benchmark run directory containing records.json and rep-*/variant/round-trace.jsonl")
    parser.add_argument("--operator-replay", type=Path, help="JSONL rows with K/M/N/name/group/replay_bits/samples_us")
    parser.add_argument("--replay-only", type=Path, help="analyze operator/head replay JSONL without a benchmark run directory")
    parser.add_argument("--output", type=Path, help="write report JSON here; defaults to stdout")
    args = parser.parse_args()
    if args.replay_only is not None:
        if args.run_dir is not None or args.operator_replay is not None:
            parser.error("--replay-only cannot be combined with run_dir or --operator-replay")
        result = {
            "source": str(args.replay_only),
            "operator_replay": summarize_replay(read_jsonl(args.replay_only)),
        }
    else:
        if args.run_dir is None:
            parser.error("run_dir is required unless --replay-only is used")
        result = analyze_run(args.run_dir, args.operator_replay)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
