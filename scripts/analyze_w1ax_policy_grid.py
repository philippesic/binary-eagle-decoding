#!/usr/bin/env python3
"""Validate and summarize the frozen W1Ax QAT-development policy grid offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

DEPTHS = (1, 2, 3, 5)
FLOORS = (0.0, 0.1, 0.3)
VARIANTS = (
    "target_only",
    "ordinary_eagle",
    "draft_q8_0",
    "draft_q4_0",
    "draft_w1a16",
    "draft_w1a8",
    "draft_w1a4",
    "draft_w1a1",
)
W1AX = VARIANTS[4:]
ANCHORS = {"fp16_eagle": "ordinary_eagle", "q4_0_eagle": "draft_q4_0"}
# The suite generator hashes config inputs under their TOML table names. The
# benchmark manifest hashes resolved models under their runtime variant names.
MODEL_SOURCE_KEYS = {
    "target": "model:target",
    "ordinary_draft": "model:ordinary_draft",
    "draft_q8_0": "model:weight_only_variants:q8_0",
    "draft_q4_0": "model:weight_only_variants:q4_0",
    **{variant: "model:w1ax:draft" for variant in W1AX},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def cell_key(depth: int, floor: float) -> str:
    return f"d{depth}-pmin-{floor:.1f}"


def pooled(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    tokens = sum(row["completion_tokens"] for row in selected)
    request_s = sum(row["request_wall_s"] for row in selected)
    decode_s = sum(row["server_predicted_ms"] for row in selected) / 1000
    counters = {
        name: [row["speculative"].get(name) for row in selected]
        for name in ("accepted", "proposed", "rounds")
    }
    totals = {
        name: sum(values) if all(number(v) for v in values) else None
        for name, values in counters.items()
    }
    if variant != "target_only":
        check(
            all(value is not None for value in totals.values()),
            f"{variant}: missing speculative counters",
        )
        check(totals["accepted"] <= totals["proposed"], f"{variant}: accepted exceeds proposed")
        check(
            totals["rounds"] > 0 and totals["proposed"] > 0,
            f"{variant}: no speculative rounds/proposals",
        )
    rounds = totals["rounds"]
    return {
        "requests": len(selected),
        "completion_tokens": tokens,
        "request_wall_s": request_s,
        "decode_server_s": decode_s,
        "request_tokens_per_s": tokens / request_s,
        "decode_tokens_per_s": tokens / decode_s,
        "speculative": totals,
        "acceptance_rate": totals["accepted"] / totals["proposed"]
        if totals["accepted"] is not None and totals["proposed"]
        else None,
        "mean_proposal_length": totals["proposed"] / rounds if rounds else None,
        "mean_accepted_length": totals["accepted"] / rounds
        if rounds and totals["accepted"] is not None
        else None,
    }


def validate_records(records: Any, prompt_ids: list[str], key: str) -> list[dict[str, Any]]:
    check(isinstance(records, list), f"{key}: records.json must be an array")
    expected = {
        (rep, prompt_id, variant)
        for rep in range(5)
        for prompt_id in prompt_ids
        for variant in VARIANTS
    }
    seen: set[tuple[int, str, str]] = set()
    for row in records:
        check(isinstance(row, dict), f"{key}: record must be an object")
        identity = (row.get("repetition"), row.get("prompt_id"), row.get("variant"))
        check(
            identity in expected and identity not in seen,
            f"{key}: duplicate or unexpected record {identity}",
        )
        seen.add(identity)
        check(
            row.get("request_id") == f"rep-{identity[0]:02d}/{identity[2]}/{identity[1]}",
            f"{key}: bad request ID {identity}",
        )
        check(row.get("policy_mode") == "policy_diagnostic", f"{key}: wrong policy mode")
        check(
            type(row.get("max_draft_tokens")) is int and number(row.get("min_draft_probability")),
            f"{key}: missing record policy",
        )
        ids = row.get("generated_token_ids")
        check(
            isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids),
            f"{key}: missing raw generated token IDs {identity}",
        )
        ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        check(
            row.get("generated_token_ids_sha256") == ids_hash,
            f"{key}: raw token ID hash mismatch {identity}",
        )
        check(
            type(row.get("completion_tokens")) is int and row["completion_tokens"] > 0,
            f"{key}: missing completion tokens {identity}",
        )
        check(
            number(row.get("request_wall_s"), positive=True)
            and number(row.get("server_predicted_ms"), positive=True),
            f"{key}: invalid wall/decode time {identity}",
        )
        check(
            isinstance(row.get("speculative"), dict),
            f"{key}: missing speculative counters {identity}",
        )
    check(seen == expected, f"{key}: incomplete paired records ({len(seen)}/{len(expected)})")
    return records


def analyze(suite_path: Path, results_root: Path) -> dict[str, Any]:
    suite_path = suite_path.resolve()
    suite = read_object(suite_path)
    progress = read_object(suite_path.parent / "progress.json")
    check(suite.get("schema") == "w1ax_diagnostic_suite_v1", "invalid suite schema")
    check(progress.get("schema") == "w1ax_diagnostic_progress_v1", "invalid progress schema")
    check(
        Path(progress.get("suite_path", "")).resolve() == suite_path,
        "progress belongs to a different suite path",
    )
    prompt = suite.get("prompt_sets", {}).get("qat_development", {})
    prompt_ids = prompt.get("ids")
    check(
        isinstance(prompt_ids, list)
        and len(prompt_ids) == 24
        and len(set(prompt_ids)) == 24
        and all(isinstance(x, str) and x for x in prompt_ids),
        "suite requires 24 distinct development prompt IDs",
    )
    check(
        prompt.get("count") == 24 and isinstance(prompt.get("sha256"), str),
        "suite development prompt count/hash invalid",
    )
    check(
        all("final" not in x.lower() for x in prompt_ids),
        "reserved final prompt ID in development suite",
    )
    configs = suite.get("configs", [])
    check(isinstance(configs, list), "suite configs missing")
    policy_entries = [entry for entry in configs if entry.get("prompt_set") == "qat_development"]
    check(len(policy_entries) == 12, "expected all 12 policy configs")
    expected_cells = {(depth, floor) for depth in DEPTHS for floor in FLOORS}
    observed_cells = [
        (entry.get("draft_depth"), entry.get("min_draft_probability")) for entry in policy_entries
    ]
    check(
        set(observed_cells) == expected_cells and len(set(observed_cells)) == 12,
        "missing or duplicate policy grid cell",
    )
    input_files = suite.get("input_files", {})
    check(isinstance(input_files, dict), "suite input hashes missing")
    model_hashes = suite.get("model_server_hashes", {})
    check(isinstance(model_hashes, dict), "suite model hashes missing")
    for model, source_key in MODEL_SOURCE_KEYS.items():
        source = input_files.get(source_key, {})
        check(
            isinstance(source, dict)
            and source.get("sha256") == model_hashes.get(source_key)
            and isinstance(source.get("sha256"), str),
            f"suite model hash missing: {model}",
        )
    check(
        input_files.get("prompt:qat_development", {}).get("sha256") == prompt["sha256"],
        "suite prompt hash disagreement",
    )
    check(
        input_files.get("server_binary", {}).get("sha256") == model_hashes.get("server_binary"),
        "suite server hash disagreement",
    )
    check(
        suite.get("primary_config_sha256") == input_files.get("primary_config", {}).get("sha256"),
        "suite primary config hash disagreement",
    )

    cells: dict[str, Any] = {}
    for entry in policy_entries:
        depth, floor = entry["draft_depth"], entry["min_draft_probability"]
        key = cell_key(depth, floor)
        name = entry.get("name")
        check(isinstance(name, str) and name.endswith(".toml"), f"{key}: config name invalid")
        check(
            entry.get("prompt_count") == 24
            and entry.get("prompt_file_sha256") == prompt["sha256"]
            and entry.get("w1ax_policy_diagnostic") is True,
            f"{key}: config prompt/policy disagreement",
        )
        state = progress.get("configs", {}).get(name, {})
        check(
            state.get("status") == "succeeded"
            and state.get("config_sha256") == entry.get("sha256"),
            f"{key}: cell did not succeed with frozen config",
        )
        attempts = state.get("attempts", [])
        successes = [
            item
            for item in attempts
            if item.get("status") == "succeeded" and item.get("exit_code") == 0
        ]
        check(
            len(successes) == 1 and successes[0].get("config_sha256") == entry["sha256"],
            f"{key}: successful attempt is missing or ambiguous",
        )
        run_id = successes[0].get("run_id")
        check(
            isinstance(run_id, str) and run_id and Path(run_id).name == run_id,
            f"{key}: invalid run ID",
        )
        run = results_root.resolve() / run_id
        manifest = read_object(run / "manifest.json")
        report = read_object(run / "report.json")
        check(report.get("status") == "complete", f"{key}: incomplete benchmark report")
        check(
            report.get("variants") == list(VARIANTS) and manifest.get("variants") == list(VARIANTS),
            f"{key}: expected eight benchmark variants",
        )
        check(
            manifest.get("config_sha256") == entry["sha256"]
            and sha256(run / "config.toml") == entry["sha256"],
            f"{key}: benchmark config hash changed",
        )
        check(
            manifest.get("prompt_ids") == prompt_ids
            and sha256(run / "prompts.jsonl") == prompt["sha256"]
            and manifest.get("files", {}).get("prompt_file", {}).get("sha256") == prompt["sha256"],
            f"{key}: development prompt IDs/content changed",
        )
        files = manifest.get("files", {})
        expected_hashes = {
            model: model_hashes[source_key] for model, source_key in MODEL_SOURCE_KEYS.items()
        }
        for model, expected_hash in expected_hashes.items():
            check(
                files.get(model, {}).get("sha256") == expected_hash,
                f"{key}: model hash changed: {model}",
            )
        for variant in ("draft_q8_0", "draft_q4_0", *W1AX):
            check(
                manifest.get("variant_specs", {}).get(variant, {}).get("draft_model_sha256")
                == expected_hashes[variant],
                f"{key}: variant spec hash changed: {variant}",
            )
        check(
            files.get("binary", {}).get("sha256") == model_hashes["server_binary"],
            f"{key}: server binary hash changed",
        )
        policy = {
            "mode": "policy_diagnostic",
            "prompt_set": "qat_development",
            "max_draft_tokens": depth,
            "min_draft_probability": floor,
            "warmup_requests": 2,
            "repetitions": 5,
        }
        check(
            manifest.get("policy") == policy and report.get("policy") == policy,
            f"{key}: policy settings changed",
        )
        dispatch = report.get("w1ax_dispatch_confirmed_by_variant", {})
        check(
            all(dispatch.get(variant) is True for variant in W1AX),
            f"{key}: incomplete W1Ax dispatch",
        )
        for rep in range(5):
            for variant in W1AX:
                evidence = read_object(run / f"rep-{rep:02d}" / variant / "dispatch-evidence.json")
                check(
                    evidence.get("cuda_w1ax_dispatch_confirmed") is True,
                    f"{key}: dispatch evidence missing for rep {rep}/{variant}",
                )
        records = validate_records(json.loads((run / "records.json").read_text()), prompt_ids, key)
        check(report.get("records") == len(records), f"{key}: record count disagreement")
        for row in records:
            variant = row["variant"]
            model = "ordinary_draft" if variant == "ordinary_eagle" else variant
            expected_hash = None if variant == "target_only" else expected_hashes[model]
            check(
                row.get("draft_model_sha256") == expected_hash,
                f"{key}: record draft hash changed: {variant}",
            )
        check(
            all(
                row["max_draft_tokens"] == depth and row["min_draft_probability"] == floor
                for row in records
            ),
            f"{key}: record policy changed",
        )
        by_variant = {variant: pooled(records, variant) for variant in VARIANTS}
        for variant, summary in by_variant.items():
            summary["vs_anchors"] = {
                anchor_name: {
                    metric: summary[f"{metric}_tokens_per_s"]
                    / by_variant[anchor][f"{metric}_tokens_per_s"]
                    for metric in ("decode", "request")
                }
                for anchor_name, anchor in ANCHORS.items()
            }
        cells[key] = {
            "draft_depth": depth,
            "min_draft_probability": floor,
            "run_id": run_id,
            "run_dir": str(run),
            "config_sha256": entry["sha256"],
            "manifest_sha256": sha256(run / "manifest.json"),
            "report_sha256": sha256(run / "report.json"),
            "records_sha256": sha256(run / "records.json"),
            "variants": by_variant,
        }

    ordered = sorted(
        cells, key=lambda key: (cells[key]["draft_depth"], cells[key]["min_draft_probability"])
    )
    best: dict[str, Any] = {}
    for variant in VARIANTS:
        winner = max(
            ordered,
            key=lambda key: (
                cells[key]["variants"][variant]["decode_tokens_per_s"],
                -cells[key]["draft_depth"],
                -cells[key]["min_draft_probability"],
            ),
        )
        best[variant] = {
            "cell": winner,
            "criterion": "max pooled decode tokens/s on development prompts",
            "decode_tokens_per_s": cells[winner]["variants"][variant]["decode_tokens_per_s"],
        }
    return {
        "schema": "w1ax_policy_grid_analysis_v1",
        "source_suite": str(suite_path),
        "suite_sha256": sha256(suite_path),
        "prompt_set": "qat_development",
        "prompt_count": 24,
        "prompt_ids": prompt_ids,
        "prompt_file_sha256": prompt["sha256"],
        "repetitions": 5,
        "variants": list(VARIANTS),
        "cells": {key: cells[key] for key in ordered},
        "best_development_policy_by_variant": best,
        "fixed_primary_policy": {
            "draft_depth": 5,
            "min_draft_probability": 0.0,
            "cell": cell_key(5, 0.0),
        },
        "interpretation": (
            "Five repetitions repeat timing on the same 24 development prompts; they are not "
            "120 independent prompts. Policy selection is exploratory on the development set. "
            "No QAT-final prompt was read and no final-set performance claim is made. "
            "Decode rate pools completion tokens/server predicted decode seconds; request rate "
            "pools completion tokens/full request wall seconds."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="frozen suite.json")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="directory containing benchmark run directories",
    )
    parser.add_argument("--output", type=Path, help="JSON report path (default beside suite.json)")
    args = parser.parse_args()
    output = args.output or args.suite.parent / "policy-grid-report.json"
    try:
        result = analyze(args.suite, args.results_root)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        print(f"W1Ax policy grid analysis failed: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
