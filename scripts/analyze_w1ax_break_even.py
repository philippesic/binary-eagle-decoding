#!/usr/bin/env python3
"""Round-only, frozen-trajectory W1Ax counterfactuals against paired primary rates.

This arithmetic is not an uninstrumented performance bound. Never subtract trace
spans from primary serving times: the two clocks and execution scopes differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

# Also support importlib-based tests without requiring scripts to be a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_w1ax_diagnostics import (  # noqa: E402
    TRACE_SCHEMA, TRACE_SPAN_STAGES, _trace_span_accounting,
    map_measured_trace_rows, read_jsonl,
)

ANCHORS = {"fp16_eagle": "ordinary_eagle", "q4_0_eagle": "draft_q4_0"}
LIMITATIONS = [
    "Counterfactuals use instrumented CPU-wall round spans only; they are not uninstrumented bounds or measured speedups.",
    "Round costs omit begin, the untraced leading seed (when present), inter-round work, prefill, and other outside-round work.",
    "Removing exclusive draft spans freezes proposals, emissions, acceptance, verification, scheduling, and all remaining costs; a real optimization need not preserve them.",
    "Acceptance thresholds and maximum rates hold proposal count, non-draft emissions, round count, and costs fixed. Raising acceptance changes trajectories, stop truncation, and costs in a real run.",
    "CPU spans are not CUDA-event timings. Draft time overlapping any other named stage is retained; process/cache-catch-up work is not removed.",
    "Primary rates use all completion tokens and server_predicted_ms, including the leading seed in the token numerator; trace round-only rates have a different scope.",
    "Primary timing is never reduced by trace timing. Primary ratios are paired-workload observations without uncertainty estimates or a lossless-speedup claim.",
    "Source metadata is preserved; hardware identity and unchanged numerical kernels across runtime revisions require the experiment's external provenance review.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def count(value: Any) -> bool:
    return type(value) is int and value >= 0


def union_length(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(right - left for left, right in merged)


def summarize_counterfactual(rows: list[dict[str, Any]], anchor_rates: dict[str, float]) -> dict[str, Any]:
    """Validate mapped rows and remove only draft intervals exclusive of other stages."""
    require(bool(rows), "no measured trace rows")
    accounting = _trace_span_accounting(rows)
    require(len(accounting["per_round"]) == len(rows), "unexpected trace schema")
    total_cost = removable = draft_union = overlap_draft = 0.0
    emitted = accepted_emitted = proposed = accepted_reported = quality_rounds = 0
    checkpoint_rows = 0
    per_round = []
    for index, (row, spans) in enumerate(zip(rows, accounting["per_round"])):
        label = f"task {row.get('task_id')} round {row.get('round_index')}"
        require(row.get("schema") == TRACE_SCHEMA, f"{label}: unexpected trace schema")
        require(row.get("clock") == "ggml_time_us_cpu_wall", f"{label}: unsupported clock")
        require(row.get("status") in ("complete", "no_proposal", "checkpoint_replay"), f"{label}: unsupported round status")
        require(type(row.get("replay")) is bool, f"{label}: missing replay flag")
        # The existing analyzer reports malformed/out-of-round spans without
        # failing. Counterfactual arithmetic requires rejecting those rows.
        allowed_flags = {"overlapping_spans", "nested_spans", "runtime_residual_was_clamped"}
        require(not (set(spans["flags"]) - allowed_flags), f"{label}: invalid span accounting: {spans['flags']}")
        start, end = row["round_start_us"], row["round_end_us"]
        cost = row.get("round_us")
        require(number(cost) and cost > 0 and cost == end - start, f"{label}: round_us disagrees with bounds")
        span_map = row.get("spans_us")
        require(isinstance(span_map, dict), f"{label}: missing spans_us")
        for stage in (*TRACE_SPAN_STAGES, "begin"):
            interval = span_map.get(stage)
            require(isinstance(interval, list) and len(interval) == 2 and all(number(v) for v in interval) and interval[1] >= interval[0], f"{label}: missing/invalid {stage} span")
            duration = interval[1] - interval[0]
            require(number(row.get(f"{stage}_us")) and row[f"{stage}_us"] == duration, f"{label}: {stage}_us disagrees with span")
            if interval != [0, 0]:
                if stage == "begin":
                    require(interval[1] <= start, f"{label}: begin overlaps round")
                else:
                    require(start <= interval[0] <= interval[1] <= end, f"{label}: {stage} outside round bounds")
        require(number(row.get("residual_us")) and row["residual_us"] == max(0, spans["unclamped_residual_us"]), f"{label}: residual_us disagrees with unclamped accounting")
        draft_left, draft_right = span_map["draft"]
        intersections = []
        for stage in TRACE_SPAN_STAGES:
            if stage == "draft":
                continue
            left, right = span_map[stage]
            left, right = max(left, draft_left), min(right, draft_right)
            if right > left:
                intersections.append((left, right))
        overlap = union_length(intersections)
        draft_duration = draft_right - draft_left
        exclusive = draft_duration - overlap
        require(0 <= exclusive <= cost, f"{label}: invalid removable draft duration")
        total_cost += cost
        removable += exclusive
        draft_union += draft_duration
        overlap_draft += overlap
        tokens, proposals = row.get("emitted_token_ids"), row.get("proposed_token_ids")
        for name, values in (("emitted", tokens), ("proposed", proposals)):
            require(isinstance(values, list) and all(count(v) for v in values), f"{label}: invalid {name} token IDs")
            require(count(row.get(f"n_{name}")) and row[f"n_{name}"] == len(values), f"{label}: n_{name} disagrees with IDs")
        accepted = row.get("n_accepted")
        require(count(accepted) and accepted <= len(proposals), f"{label}: invalid n_accepted")
        a = 0
        if row["status"] == "checkpoint_replay":
            checkpoint_rows += 1
            require(not tokens, f"{label}: checkpoint replay unexpectedly emits tokens")
        else:
            require(not row["replay"], f"{label}: replay quality emission semantics unsupported")
            if row["status"] == "no_proposal":
                require(not proposals and accepted == 0, f"{label}: no_proposal row has proposal/acceptance counts")
            require(0 < len(tokens) <= accepted + 1, f"{label}: invalid emission count versus acceptance")
            a = min(accepted, len(tokens))
            require(tokens[:a] == proposals[:a], f"{label}: emitted accepted prefix disagrees with proposed IDs")
            if len(tokens) <= accepted:
                # Stop processing can truncate accepted IDs, but only on the
                # terminal round of this request (repetitions restart task IDs).
                following = rows[index + 1] if index + 1 < len(rows) else None
                require(following is None or (following.get('repetition'), following.get('task_id')) != (row.get('repetition'), row.get('task_id')), f"{label}: truncated emission before request ended")
            emitted += len(tokens)
            accepted_emitted += a
            accepted_reported += accepted
            proposed += len(proposals)
            quality_rounds += 1
        per_round.append({
            "repetition": row.get("repetition"), "task_id": row.get("task_id"), "round_index": row.get("round_index"),
            "status": row["status"], "round_us": cost,
            "draft_span_union_us": draft_duration, "draft_overlap_other_union_us": overlap,
            "removable_exclusive_draft_us": exclusive,
            "unclamped_residual_us": spans["unclamped_residual_us"],
            "union_unattributed_us": spans["union_unattributed_us"],
            "emitted": len(tokens), "accepted_drafts_actually_emitted": a,
        })
    require(quality_rounds > 0, "no quality rounds")
    remainder = total_cost - removable
    require(remainder > 0, "zero/nonpositive residual round cost after draft removal")
    bonus = emitted - accepted_emitted
    scenarios = {}
    for name, cost_us in (("observed_round_cost", total_cost), ("exclusive_draft_removed", remainder)):
        cost_s = cost_us / 1_000_000
        thresholds = {}
        for anchor, rate in anchor_rates.items():
            require(number(rate) and rate > 0, f"invalid primary anchor rate: {anchor}")
            a_be = rate * cost_s - bonus
            thresholds[anchor] = {
                "primary_decode_tokens_per_s": rate,
                "accepted_drafts_required_raw": a_be,
                "accepted_per_proposed_required_raw": a_be / proposed if proposed else None,
                "accepted_per_quality_round_required_raw": a_be / quality_rounds,
                "exceeds_fixed_proposal_count": a_be > proposed,
            }
        scenarios[name] = {
            "fixed_round_cost_us": cost_us,
            "frozen_emission_rate_tokens_per_s": emitted / cost_s,
            "maximum_fixed_trajectory_rate_tokens_per_s": (proposed + bonus) / cost_s,
            "acceptance_thresholds": thresholds,
        }
    return {
        "counts": {"trace_rows": len(rows), "N_quality_rounds": quality_rounds,
                   "checkpoint_replay_cost_rows": checkpoint_rows, "E_emitted_ids": emitted,
                   "A_accepted_drafts_actually_emitted": accepted_emitted,
                   "accepted_drafts_reported_before_stop": accepted_reported,
                   "accepted_drafts_not_emitted_at_stop": accepted_reported - accepted_emitted,
                   "P_proposed_ids_quality_rounds": proposed, "B_non_draft_emissions": bonus},
        "timing_us": {"C_all_rounds": total_cost, "D_removable_exclusive_draft": removable,
                      "draft_span_union": draft_union, "draft_overlap_other_union": overlap_draft,
                      "C_minus_D": remainder,
                      "unclamped_residual_sum": sum(r["unclamped_residual_us"] for r in per_round)},
        "scenarios": scenarios, "span_accounting": accounting, "per_round": per_round,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    records = json.loads((path / "records.json").read_text())
    require(isinstance(manifest, dict), f"{path}: manifest must be an object")
    require(isinstance(records, list) and bool(records) and all(isinstance(r, dict) for r in records), f"{path}: records must be nonempty object array")
    return manifest, records


def indexed_records(records: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    result = {}
    for row in records:
        require(count(row.get("repetition")) and isinstance(row.get("prompt_id"), str) and isinstance(row.get("variant"), str), "invalid pairing key")
        key = (row["repetition"], row["prompt_id"], row["variant"])
        require(key not in result, f"duplicate request pairing key: {key}")
        result[key] = row
    return result


def primary_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for variant in sorted({r["variant"] for r in records}):
        rows = [r for r in records if r["variant"] == variant]
        require(all(count(r.get("completion_tokens")) and r["completion_tokens"] > 0 and number(r.get("server_predicted_ms")) and r["server_predicted_ms"] > 0 for r in rows), f"{variant}: invalid primary completion tokens or server_predicted_ms")
        require(all(isinstance(r.get("generated_token_ids"), list) and all(count(token) for token in r["generated_token_ids"]) and len(r["generated_token_ids"]) == r["completion_tokens"] for r in rows), f"{variant}: primary completion_tokens disagree with raw generated token IDs or IDs are missing")
        tokens = sum(r["completion_tokens"] for r in rows)
        seconds = sum(r["server_predicted_ms"] for r in rows) / 1000
        result[variant] = {"requests": len(rows), "completion_tokens": tokens,
                           "decode_server_s": seconds, "decode_tokens_per_s": tokens / seconds}
    require(all(v in result for v in ANCHORS.values()), "primary requires FP16 EAGLE and Q4_0 EAGLE anchors")
    for values in result.values():
        values["decode_rate_ratios"] = {a: values["decode_tokens_per_s"] / result[v]["decode_tokens_per_s"] for a, v in ANCHORS.items()}
    return result


def analyze_runs(trace_dir: Path, primary_dir: Path) -> dict[str, Any]:
    trace_manifest, trace_records = load_run(trace_dir)
    primary_manifest, primary_records = load_run(primary_dir)
    require(trace_manifest.get("round_trace_enabled") is True, "trace manifest must enable round tracing")
    require(primary_manifest.get("round_trace_enabled") is False, "primary manifest must explicitly disable round tracing")
    require(all(not r.get("round_trace_path") for r in primary_records), "primary records contain round instrumentation")
    trace_index, primary_index = indexed_records(trace_records), indexed_records(primary_records)
    require(trace_index.keys() == primary_index.keys(), "trace/primary request pairing sets differ")
    for field in ("prompt_ids", "request_options", "precision", "policy"):
        require(field in trace_manifest and field in primary_manifest and trace_manifest[field] == primary_manifest[field], f"trace/primary {field} mismatch or missing")
    # All nonbinary inputs include the prompt source and resolved model files.
    trace_files, primary_files = trace_manifest.get("files"), primary_manifest.get("files")
    require(isinstance(trace_files, dict) and isinstance(primary_files, dict), "missing input file hashes")
    input_names = set(trace_files) - {"binary"}
    require(input_names == set(primary_files) - {"binary"} and {"target", "ordinary_draft", "draft_q4_0"} <= input_names, "trace/primary input file sets differ or lack anchors")
    for name in input_names:
        digest = trace_files[name].get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and digest == primary_files[name].get("sha256"), f"trace/primary input hash mismatch: {name}")
    for key, row in trace_index.items():
        other = primary_index[key]
        for field in ("policy_mode", "max_draft_tokens", "min_draft_probability", "draft_model_sha256", "w1ax_activation_bits_selector", "w1a1_mma_selector", "w8a8_mma_selector", "w4a4_mma_selector"):
            require(row.get(field) == other.get(field), f"trace/primary {field} mismatch for {key}")
    rates = primary_rates(primary_records)
    # Every variant must cover the same repetition/prompt pairs as each anchor.
    expected_pairs = {(rep, prompt) for rep, prompt, variant in primary_index if variant == ANCHORS["fp16_eagle"]}
    for variant in rates:
        require({(rep, prompt) for rep, prompt, v in primary_index if v == variant} == expected_pairs, f"{variant}: primary pairing differs from FP16 anchor")
    anchors = {a: rates[v]["decode_tokens_per_s"] for a, v in ANCHORS.items()}
    sources = []
    for role, path, manifest in (("trace", trace_dir, trace_manifest), ("primary", primary_dir, primary_manifest)):
        sources.append({"role": role, "path": str(path),
                        "sha256": {name: sha256(path / name) for name in ("manifest.json", "records.json")},
                        "provenance": {k: manifest.get(k) for k in ("project_commit", "llama_checkout_commit", "llama_checkout_diff_sha256", "precision", "variant_specs", "initial_gpu_snapshot", "environment_manifest")}})
    results = {}
    trace_hashes = {}
    for variant in sorted(rates):
        if variant == "target_only":
            continue
        rows, mappings = [], []
        for repetition in sorted({r["repetition"] for r in trace_records if r["variant"] == variant}):
            path = trace_dir / f"rep-{repetition:02d}" / variant / "round-trace.jsonl"
            measured, mapping = map_measured_trace_rows(read_jsonl(path), trace_records, repetition, variant, trace_manifest["policy"].get("warmup_requests"), trace_manifest)
            rows.extend({**r, "repetition": repetition} for r in measured)
            mappings.append({"repetition": repetition, **mapping})
            trace_hashes[str(path.relative_to(trace_dir))] = sha256(path)
        offsets = {m["untraced_leading_tokens_per_measured_request"] for m in mappings}
        require(len(offsets) == 1, f"{variant}: leading-token omission differs across repetitions")
        results[variant] = {**summarize_counterfactual(rows, anchors), "request_mapping": mappings,
                            "trace_response_tokens": sum(m["response_generated_tokens_total"] for m in mappings),
                            "untraced_leading_tokens": sum(m["untraced_leading_tokens_total"] for m in mappings)}
    return {
        "schema": "w1ax_frozen_trajectory_break_even_v1", "sources": sources,
        "analysis_source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("analyze_w1ax_break_even.py", "analyze_w1ax_diagnostics.py")},
        "trace_file_sha256": trace_hashes,
        "pairing": {"status": "validated_exact_request_set_and_inputs", "paired_requests": len(primary_index),
                    "trace_vs_primary_equal_output_sequences": sum(trace_index[k].get("generated_token_ids") == primary_index[k].get("generated_token_ids") for k in primary_index)},
        "measured_primary": {"scope": "Uninstrumented primary records only; pooled completion_tokens / (sum(server_predicted_ms)/1000).", "variants": rates},
        "trace_counterfactuals": results,
        "definitions": {
            "C": "sum(round_us), including checkpoint-replay cost rows",
            "D": "sum per-round draft span duration minus its intersection with the union of other named stage spans",
            "E": "actual emitted IDs on quality rounds; excludes any untraced seed",
            "A": "min(n_accepted, emitted ID count), after validating the emitted accepted prefix equals the proposed prefix; replay quality rows unsupported",
            "P": "actual proposed ID count on quality rounds, excluding checkpoint_replay",
            "N": "quality rounds, excluding checkpoint_replay", "B": "E - A (non-draft emissions)",
            "zero_draft_span_removal_rate": "E / ((C-D)/1e6), with only exclusive draft spans removed",
            "acceptance_threshold": "A_BE = R_anchor * (C_star/1e6) - B; C_star is C or C-D; raw A_BE/P is not clamped",
            "maximum_fixed_trajectory_rate": "(P+B) / (C_star/1e6); hypothetical full acceptance with fixed costs and B",
        },
        "limitations": LIMITATIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_run", type=Path)
    parser.add_argument("primary_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_runs(args.trace_run, args.primary_run)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
