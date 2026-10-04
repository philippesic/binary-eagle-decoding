"""Matched native measurements; family Q4 is the primary denominator."""

from __future__ import annotations

import math
from collections import defaultdict

from .nine_model_pipeline import CELLS, require


def aggregate(measurements, *, fixture=False):
    require(
        measurements.get("schema") == "nine_model_native_measurements_v1"
        and measurements.get("native") is True
        and measurements.get("instrumentation_in_clean_timing") is False,
        "actual clean native measurements required",
    )
    require(measurements.get("compute_capability") == [12, 0], "SM120 campaign results required")
    kind = "fixture" if fixture else "production"
    require(
        measurements.get("artifact_kind") == kind,
        "fixture measurements cannot grant production result readiness",
    )
    required_cells = {*CELLS, "target_only"}
    records = measurements["records"]
    require({r["cell"] for r in records} == required_cells, "nine cells and target-only required")
    groups = defaultdict(list)
    pairs = defaultdict(set)
    for row in records:
        key = (row["repetition"], row["prompt_id"])
        require(key not in pairs[row["cell"]], "duplicate paired measurement")
        pairs[row["cell"]].add(key)
        require(
            type(row["output_tokens"]) is int and row["output_tokens"] > 0,
            "positive exact output count required",
        )
        require(
            type(row["latency_s"]) in (float, int)
            and math.isfinite(row["latency_s"])
            and row["latency_s"] > 0,
            "invalid request latency",
        )
        groups[row["cell"]].append(row)
    require(len({frozenset(v) for v in pairs.values()}) == 1, "unmatched prompt/repetition IDs")
    repeats = {r["repetition"] for r in records}
    require(len(repeats) >= 5, "at least five balanced repetitions required")
    tables = {}
    for cell, rows in groups.items():
        tokens = sum(r["output_tokens"] for r in rows)
        duration = sum(r["latency_s"] for r in rows)
        tables[cell] = {
            "output_tokens": tokens,
            "request_seconds": duration,
            "request_tokens_per_second": tokens / duration,
            "mean_request_latency_s": duration / len(rows),
            "requests": len(rows),
        }
    for cell, rows in groups.items():
        if cell == "target_only":
            continue
        counters = [r.get("speculative") for r in rows]
        valid = all(
            isinstance(c, dict)
            and all(type(c.get(k)) is int and c[k] >= 0 for k in ("proposed", "accepted", "rounds"))
            and c["rounds"] > 0
            and c["accepted"] <= c["proposed"]
            for c in counters
        )
        require(fixture or valid, "actual native acceptance counters missing")
        if valid:
            sums = {
                key: sum(c[key] for c in counters) for key in ("proposed", "accepted", "rounds")
            }
            tables[cell].update(
                native_counts=sums,
                accepted_per_round=sums["accepted"] / sums["rounds"],
                emitted_tokens_per_round=tables[cell]["output_tokens"] / sums["rounds"],
                proposal_acceptance=sums["accepted"] / sums["proposed"]
                if sums["proposed"]
                else None,
            )
    for cell in CELLS:
        family = cell.split("_")[0]
        rate = tables[cell]["request_tokens_per_second"]
        tables[cell]["speed_vs_family_q4"] = (
            rate / tables[family + "_q4"]["request_tokens_per_second"]
        )
        tables[cell]["speed_vs_eagle_q4"] = rate / tables["eagle_q4"]["request_tokens_per_second"]
    return {
        "schema": "nine_model_matched_report_v1",
        "artifact_kind": kind,
        "native": True,
        "hardware": measurements["hardware"],
        "compute_capability": [12, 0],
        "protocol": measurements["protocol"],
        "target": measurements["target"],
        "model_ancestry": measurements["model_ancestry"],
        "cells": tables,
        "timing_repetitions": len(repeats),
        "matched_requests_per_cell": len(next(iter(pairs.values()))),
    }
