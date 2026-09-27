#!/usr/bin/env python3
"""Validated CPU round partitions and conditional draft-removal proxies.

Never subtract instrumented spans from timed serving latency. CUDA/kernel
profiling is separate. Optional --round-paths JSON is a list of objects with
variant, prompt_id, repetition, path (relative to that file, or absolute).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_binary_rescue_benchmark import (
    aggregate,
    apply_round_calibration,
    distribution,
    divergence,
)
from analyze_w1ax_break_even import (
    TRACE_SPAN_STAGES,
    count,
    number,
    require,
    summarize_counterfactual,
)
from run_binary_rescue_benchmark import output_digest, round_quality, write

STAGES = {
    "draft": "draft",
    "target_decode_sync": "target_verify_sync",
    "process": "feature_process_cache_catchup",
    "check": "verification_check",
    "checkpoint": "checkpoint",
    "kv_repair": "cache_repair",
    "accept_hook": "accept_hook",
}
NESTED_FIELDS = (
    "draft_seed_decode_us",
    "draft_step_decode_us",
    "draft_sampler_us",
    "process_feature_copy_us",
    "process_encoder_us",
    "process_batch_build_us",
    "process_draft_decode_us",
)
LIMITATIONS = [
    "All stage spans are ggml_time_us CPU wall, not CUDA events or kernel durations.",
    "Disjoint stage exclusives, shared-overlap, and unattributed sum to complete round cost. "
    "Inclusive named-stage and nested scalar durations must not be added to that partition.",
    "Replay rows contribute cost but no quality emissions or proposals. Accepted drafts exclude "
    "bonus target tokens; terminal accepted-but-unemitted IDs are reported separately.",
    "Begin, possible leading seed, prefill, inter-round gaps, HTTP and other outside-round work "
    "are outside traced round cost. Server decode and full client wall have different scopes.",
    "Draft-removal proxies freeze outputs, proposals, acceptance, round count, verification, "
    "scheduling and remaining costs. They are not measured speedups or "
    "hardware-independent bounds.",
    "Comparisons of instrumented round proxies to timed Q4 decode retain a scope mismatch. "
    "No instrumented duration is subtracted from timed serving latency.",
    "Proposal FNV64 digests are noncryptographic equality evidence, not a collision-free proof. "
    "A failed/missing behavior gate withholds the removal proxy.",
]


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_key(row):
    require(isinstance(row.get("variant"), str), "missing variant")
    require(isinstance(row.get("prompt_id"), str), "missing prompt_id")
    require(count(row.get("repetition")), "invalid repetition")
    return row["variant"], row["prompt_id"], row["repetition"]


def measured(manifest):
    require(manifest.get("schema") == "binary_rescue_benchmark_v1", "unsupported benchmark schema")
    require(manifest.get("status") == "complete", "benchmark must be complete")
    rows = [r for r in manifest["records"] if not r.get("warmup")]
    require(bool(rows), "no measured requests")
    keys = [request_key(r) for r in rows]
    require(len(set(keys)) == len(keys), "duplicate measured request key")
    for row in rows:
        ids = row.get("generated_token_ids")
        require(
            isinstance(ids, list) and bool(ids) and all(count(t) for t in ids), "invalid raw IDs"
        )
        require(row.get("completion_tokens") == len(ids), "completion count disagrees with IDs")
    return rows


def match_inputs(instrumented, reference, variants):
    for field in ("policy", "prompt_sha256", "workload", "q4_variant"):
        require(
            field in instrumented and instrumented[field] == reference.get(field),
            f"mismatched/missing {field}",
        )
    for field in ("binary", "target"):
        value = instrumented.get("hashes", {}).get(field)
        require(
            isinstance(value, str)
            and len(value) == 64
            and value == reference.get("hashes", {}).get(field),
            f"mismatched/missing {field} hash",
        )
    for variant in variants:
        a = instrumented["hashes"].get("drafts", {})
        b = reference["hashes"].get("drafts", {})
        require(
            variant in a and variant in b and a[variant] == b[variant],
            f"mismatched/missing draft hash for {variant}",
        )


def valid_digest(row):
    digest = row.get("request_digest")
    if not isinstance(digest, dict):
        return False
    return (
        all(count(digest.get(k)) for k in ("rounds", "no_proposal", "output_tokens"))
        and isinstance(digest.get("proposal"), str)
        and len(digest["proposal"]) == 16
        and all(c in "0123456789abcdef" for c in digest["proposal"])
        and digest.get("output") == output_digest(row["generated_token_ids"])
        and digest["output_tokens"] == len(row["generated_token_ids"])
    )


def behavior_gate(instrumented, reference):
    """Each traced request is checked against every timing repetition of that prompt."""
    index = defaultdict(list)
    for row in reference:
        index[(row["variant"], row["prompt_id"])].append(row)
    comparisons = []
    for row in instrumented:
        matches = index[(row["variant"], row["prompt_id"])]
        pairs = []
        for other in matches:
            digests_valid = valid_digest(row) and valid_digest(other)
            proposals_equal = digests_valid and all(
                row["request_digest"][k] == other["request_digest"][k]
                for k in ("proposal", "output", "rounds", "no_proposal", "output_tokens")
            )
            counters = ("accepted", "proposed", "rounds")
            counters_valid = all(
                count(r.get("speculative", {}).get(k)) for r in (row, other) for k in counters
            )
            pairs.append(
                {
                    "reference_repetition": other["repetition"],
                    "first_divergence": divergence(
                        other["generated_token_ids"], row["generated_token_ids"]
                    ),
                    "proposal_output_digests_match": proposals_equal,
                    "native_counters_match": all(
                        row["speculative"][k] == other["speculative"][k] for k in counters
                    )
                    if counters_valid
                    else None,
                }
            )
        passed = bool(pairs) and all(
            p["first_divergence"] is None
            and p["proposal_output_digests_match"]
            and p["native_counters_match"] is not False
            for p in pairs
        )
        comparisons.append(
            {
                "variant": row["variant"],
                "prompt_id": row["prompt_id"],
                "instrumented_repetition": row["repetition"],
                "passed": passed,
                "pairs": pairs,
            }
        )
    return {
        "passed": bool(comparisons) and all(r["passed"] for r in comparisons),
        "requests": comparisons,
        "rule": "Exact raw IDs and valid equal proposal/output/round/no-proposal digests "
        "against every matched reference repetition; present native counters must agree.",
    }


def partition(row):
    """Called only after strict span validation; no arbitrary overlap attribution."""
    start, end = row["round_start_us"], row["round_end_us"]
    spans = {
        stage: row["spans_us"][stage]
        for stage in TRACE_SPAN_STAGES
        if row["spans_us"][stage][1] > row["spans_us"][stage][0]
    }
    points = sorted({start, end, *(p for span in spans.values() for p in span)})
    result = dict.fromkeys([*STAGES.values(), "shared_stage_overlap", "unattributed"], 0)
    combinations = defaultdict(float)
    for left, right in zip(points, points[1:]):
        active = sorted(stage for stage, (a, b) in spans.items() if a <= left and b >= right)
        duration = right - left
        if not active:
            result["unattributed"] += duration
        elif len(active) == 1:
            result[STAGES[active[0]]] += duration
        else:
            result["shared_stage_overlap"] += duration
            combinations[" + ".join(active)] += duration
    require(
        math.isclose(sum(result.values()), row["round_us"], abs_tol=1e-6),
        "round partition does not close",
    )
    return result, dict(combinations)


def inspect_request(record, rows):
    require(bool(rows), f"no round rows for {request_key(record)}")
    task_ids = {r.get("task_id") for r in rows}
    require(len(task_ids) == 1, "per-request round file mixes task IDs")
    require(
        [r.get("round_index") for r in rows] == list(range(len(rows))),
        "round indices must be contiguous from zero",
    )
    if valid_digest(record):
        require(
            task_ids == {record["request_digest"].get("task_id")},
            "round task ID differs from request digest",
        )
    for previous, current in zip(rows, rows[1:]):
        require(
            previous["round_end_us"] <= current["round_start_us"], "complete round windows overlap"
        )
    # Existing strict validator checks spans/scalars, accepted prefix, terminal
    # censoring, replay/no-proposal semantics and exclusive draft interval union.
    checked = summarize_counterfactual(rows, {})
    emitted = [
        t for row in rows if row["status"] != "checkpoint_replay" for t in row["emitted_token_ids"]
    ]
    actual = record["generated_token_ids"]
    if emitted == actual:
        omitted = 0
    elif emitted == actual[1:]:
        omitted = 1
    else:
        raise ValueError("round emitted IDs differ from response or one-leading-seed suffix")
    parts = []
    combinations = defaultdict(float)
    for row in rows:
        disjoint, shared = partition(row)
        require(
            math.isclose(
                disjoint["draft"],
                checked["per_round"][len(parts)]["removable_exclusive_draft_us"],
                abs_tol=1e-6,
            ),
            "draft partition disagrees with validated exclusive span",
        )
        parts.append(disjoint)
        for stage, duration in shared.items():
            combinations[stage] += duration
    totals = {stage: sum(p[stage] for p in parts) for stage in parts[0]}
    scalar_details = {}
    for field in NESTED_FIELDS:
        values = [r.get(field) for r in rows]
        scalar_details[field] = sum(values) if all(number(v) for v in values) else None
    return {
        "variant": record["variant"],
        "prompt_id": record["prompt_id"],
        "repetition": record["repetition"],
        "task_id": next(iter(task_ids)),
        "counts": checked["counts"],
        "timing_us": checked["timing_us"],
        "disjoint_partition_us": totals,
        "shared_combinations_us": dict(combinations),
        "inclusive_stage_us": {
            stage: sum(r[f"{stage}_us"] for r in rows) for stage in TRACE_SPAN_STAGES
        },
        "nested_scalar_us_not_additive": scalar_details,
        "begin_outside_round_us": sum(r["begin_us"] for r in rows),
        "inter_round_gaps_us": sum(
            b["round_start_us"] - a["round_end_us"] for a, b in zip(rows, rows[1:])
        ),
        "response_output_tokens": len(actual),
        "untraced_leading_seed_tokens": omitted,
        "round_partition_us": parts,
        "quality": round_quality(rows),
    }


DRAFT_STAGES = {
    "seed_prepare",
    "seed_decode_call",
    "seed_sync_retrieve",
    "recurrent_decode_call",
    "recurrent_sync_retrieve",
    "output_sampling",
    "capture_copy",
    "recurrent_input_pack",
    "step_bookkeeping",
    "finalize_bookkeeping",
}
DRAFT_COUNTS = ("seed_rows", "recurrent_rows", "decode_calls", "sampling_calls", "retrieval_calls")


def validate_stage_call(call):
    require(call.get("schema") == "eagle_draft_stage_v1", "unsupported draft-stage schema")
    require(call.get("clock") == "ggml_time_us_cpu_wall", "unsupported draft-stage clock")
    require(call.get("scope") == "draft_invocation_nested_within_round", "unsupported draft scope")
    require(call.get("explicit_sync_before_sampling") is True, "missing explicit sync semantics")
    require(call.get("serialization_included") is False, "unexpected serialization scope")
    require(
        call.get("status") in ("complete", "empty", "seed_decode_error", "recurrent_decode_error"),
        "unknown draft-stage status",
    )
    require(count(call.get("call_index")), "invalid call index")
    require(all(count(call.get(k)) for k in DRAFT_COUNTS), "invalid draft-stage counters")
    start, end = call.get("start_us"), call.get("end_us")
    require(number(start) and number(end) and end >= start, "invalid draft-call bounds")
    require(call.get("total_us") == end - start, "draft-call duration differs from bounds")
    spans = call.get("spans")
    require(isinstance(spans, list) and bool(spans), "missing draft-stage spans")
    sequences = call.get("sequences")
    require(
        isinstance(sequences, list) and len(sequences) <= 1, "requires concurrency-one stage calls"
    )
    totals = defaultdict(float)
    previous = start
    for span in spans:
        stage = span.get("stage")
        require(stage in DRAFT_STAGES, "unknown draft stage")
        a, b = span.get("start_us"), span.get("end_us")
        require(
            number(a) and number(b) and a == previous and a <= b <= end,
            "draft-stage spans overlap, have gaps, or leave call bounds",
        )
        require(span.get("duration_us") == b - a, "draft-stage scalar/span mismatch")
        require(type(span.get("depth")) is int and span["depth"] >= -1, "invalid recurrent depth")
        totals[stage] += b - a
        previous = b
    require(
        previous == end
        and call.get("partition_us") == end - start
        and call.get("unassigned_us") == 0,
        "draft-stage partition does not close",
    )
    reported = call.get("stage_totals_us")
    require(
        isinstance(reported, dict) and set(reported) <= DRAFT_STAGES, "invalid draft-stage totals"
    )
    require(
        all(number(v) for v in reported.values())
        and all(reported.get(k, 0) == totals.get(k, 0) for k in set(reported) | set(totals)),
        "draft-stage totals disagree with spans",
    )
    expected = {
        "decode_calls": sum(
            v["stage"] in ("seed_decode_call", "recurrent_decode_call") for v in spans
        ),
        "retrieval_calls": sum(
            v["stage"] in ("seed_sync_retrieve", "recurrent_sync_retrieve") for v in spans
        ),
        "sampling_calls": sum(v["stage"] == "output_sampling" for v in spans),
    }
    require(
        all(call[k] == value for k, value in expected.items()),
        "draft-stage call counts disagree with spans",
    )


def inspect_draft_stages(rows, calls):
    if calls is None:
        return {"available": False, "reason": "no optional draft-stage file supplied"}
    seen = set()
    for call in calls:
        validate_stage_call(call)
        require(call["call_index"] not in seen, "duplicate draft call index in file")
        seen.add(call["call_index"])
    mapped = []
    totals = defaultdict(float)
    by_depth = defaultdict(lambda: defaultdict(float))
    for row in rows:
        left, right = row["spans_us"]["draft"]
        selected = []
        for call in calls:
            a, b = call["start_us"], call["end_us"]
            intersects = max(a, left) < min(b, right)
            contained = left <= a <= b <= right and right > left
            require(not intersects or contained, "draft-stage call crosses outer draft bounds")
            if contained:
                selected.append(call)
        selected.sort(key=lambda c: (c["start_us"], c["call_index"]))
        require(
            all(a["end_us"] <= b["start_us"] for a, b in zip(selected, selected[1:])),
            "draft calls overlap inside outer draft",
        )
        for call in selected:
            mapped.append(
                {
                    "round_index": row["round_index"],
                    "call_index": call["call_index"],
                    "status": call["status"],
                    "sequences": call["sequences"],
                    "total_us": call["total_us"],
                    **{k: call[k] for k in DRAFT_COUNTS},
                }
            )
            for span in call["spans"]:
                totals[span["stage"]] += span["duration_us"]
                by_depth[str(span["depth"])][span["stage"]] += span["duration_us"]
    require(
        len({c["call_index"] for c in mapped}) == len(mapped), "draft call maps to multiple rounds"
    )
    inclusive = sum(r["draft_us"] for r in rows)
    duration = sum(c["total_us"] for c in mapped)
    require(duration <= inclusive, "nested calls exceed enclosing draft cost")
    return {
        "available": True,
        "matched_calls": len(mapped),
        "draft_rounds_with_nonzero_span": sum(r["draft_us"] > 0 for r in rows),
        "rounds_with_mapped_calls": len({c["round_index"] for c in mapped}),
        "calls": mapped,
        "stage_partition_us": dict(totals),
        "stage_partition_by_depth_us": {k: dict(v) for k, v in by_depth.items()},
        "nested_calls_total_us": duration,
        "outer_draft_remainder_us": inclusive - duration,
        "counts": {k: sum(c[k] for c in mapped) for k in DRAFT_COUNTS},
        "error_calls": sum(c["status"].endswith("_error") for c in mapped),
        "scope": "Nested CPU draft-call partition, already inside outer inclusive draft. "
        "Do not add to outer costs. Sync/retrieve includes backend wait and state getter; "
        "candidate-logit copying remains in output_sampling. File serialization is "
        "outside these calls but inside outer draft. Explicit diagnostic sync can alter timing.",
    }


def sum_dict(items, field):
    keys = set().union(*(item[field].keys() for item in items))
    return {k: sum(item[field].get(k, 0) for item in items) for k in sorted(keys)}


def removal_proxy(counts, timing, q4_rate):
    emitted = counts["E_emitted_ids"]
    cost = timing["C_all_rounds"]
    removable = timing["D_removable_exclusive_draft"]
    remaining = cost - removable
    budget = emitted * 1e6 / q4_rate
    return {
        "fixed_emitted_tokens": emitted,
        "observed_complete_round_cpu_us": cost,
        "removable_exclusive_draft_cpu_us": removable,
        "remaining_complete_round_cpu_us": remaining,
        "observed_instrumented_round_tok_s": emitted * 1e6 / cost,
        "conditional_draft_removed_round_tok_s": emitted * 1e6 / remaining,
        "q4_timed_decode_tok_s": q4_rate,
        "conditional_round_rate_over_q4_timed_decode": (emitted * 1e6 / remaining) / q4_rate,
        "round_budget_at_q4_timed_decode_us": budget,
        "required_round_time_removal_us_raw": cost - budget,
        "required_exclusive_draft_removal_fraction_raw": (cost - budget) / removable
        if removable
        else None,
        "remaining_budget_after_full_draft_removal_us": budget - remaining,
        "scope": "Conditional fixed-trajectory CPU round-only proxy versus a timed full-decode "
        "reference. Not an optimized measurement; no timed latency is reduced.",
    }


def analyze(instrumented, timed, load_rounds, quality=None, timed_analysis=None, load_stages=None):
    trace_records, timed_records = measured(instrumented), measured(timed)
    require(
        instrumented.get("mode") in ("instrumented", "quality"),
        "input must contain instrumented rounds",
    )
    require(timed.get("mode") == "timed", "primary reference must be timed mode")
    variants = {r["variant"] for r in trace_records}
    q4 = timed["q4_variant"]
    require(q4 in {r["variant"] for r in timed_records}, "missing timed Q4 reference")
    match_inputs(instrumented, timed, variants)
    # Rebuild quality counts from the actual round files before certifying the
    # timed digest correction. Never trust a supplied aggregate denominator.
    round_cache = {request_key(r): load_rounds(r) for r in trace_records}
    trace_records = [
        {**r, "quality": round_quality(round_cache[request_key(r)])} for r in trace_records
    ]
    traced_counts = {**instrumented, "records": trace_records}
    timed_subset = {**timed, "records": [r for r in timed_records if r["variant"] in variants]}
    calibrated, certificate = apply_round_calibration(timed_subset, traced_counts)
    timed_records = calibrated["records"] + [
        r for r in timed_records if r["variant"] not in variants
    ]
    quality_records = measured(quality) if quality else None
    if quality:
        match_inputs(instrumented, quality, variants)
    output = {
        "schema": "binary_rescue_round_cpu_v1",
        "q4_variant": q4,
        "policy": instrumented["policy"],
        "workload": instrumented["workload"],
        "limitations": LIMITATIONS,
        "round_denominator_calibration": certificate,
        "variants": {},
    }
    used_stage_calls = set()
    stage_files = {}
    for variant in sorted(variants, key=lambda name: (name != q4, name)):
        records = [r for r in trace_records if r["variant"] == variant]
        matched_timed = [r for r in timed_records if r["variant"] == variant]
        prompt_ids = {r["prompt_id"] for r in records}
        require(
            matched_timed and prompt_ids <= {r["prompt_id"] for r in matched_timed},
            "missing timed candidate prompts",
        )
        q4_records = [
            r for r in timed_records if r["variant"] == q4 and r["prompt_id"] in prompt_ids
        ]
        expected = {
            (r["prompt_id"], r["repetition"]) for r in matched_timed if r["prompt_id"] in prompt_ids
        }
        require(
            expected == {(r["prompt_id"], r["repetition"]) for r in q4_records},
            "Q4 timing pairing set differs",
        )
        primary = aggregate([r for r in matched_timed if r["prompt_id"] in prompt_ids])
        q4_primary = aggregate(q4_records)
        require(
            number(q4_primary["server_decode_tok_s"]) and q4_primary["server_decode_tok_s"] > 0,
            "Q4 decode timing missing/invalid",
        )
        if timed_analysis:
            require(
                timed_analysis.get("mode") == "timed"
                and timed_analysis.get("policy") == timed["policy"],
                "timed analysis mismatch",
            )
            supplied = timed_analysis["variants"][variant]
            full = aggregate(matched_timed)
            for field in ("requests", "completion_tokens", "client_wall_s", "server_decode_s"):
                require(
                    supplied[field] == full[field],
                    f"timed analysis disagrees with raw manifest: {variant}/{field}",
                )
        gate = behavior_gate(records, timed_records)
        quality_gate = behavior_gate(records, quality_records) if quality_records else None
        requests, all_rounds = [], []
        for record in records:
            rows = round_cache[request_key(record)]
            if not rows:
                require(
                    instrumented["hashes"]["drafts"][variant] is None,
                    "speculative variant has no round trace",
                )
                continue
            item = inspect_request(record, rows)
            calls = load_stages(record) if load_stages else None
            item["draft_stages"] = inspect_draft_stages(rows, calls)
            if calls is not None:
                file_key = (variant, record["repetition"])
                stage_files[file_key] = {c["call_index"] for c in calls}
                for call in item["draft_stages"]["calls"]:
                    call_key = (*file_key, call["call_index"])
                    require(call_key not in used_stage_calls, "draft call reused across requests")
                    used_stage_calls.add(call_key)
            requests.append(item)
            all_rounds.extend(rows)
        item = {
            "timed_behavior_gate": gate,
            "quality_behavior_gate": quality_gate,
            "paired_timed_candidate": primary,
            "paired_timed_q4": q4_primary,
            "request_count": len(records),
            "prompt_count": len(prompt_ids),
            "requests": requests,
            "round_scope_available": bool(requests),
        }
        if requests:
            counts = sum_dict(requests, "counts")
            timing = sum_dict(requests, "timing_us")
            partition_totals = sum_dict(requests, "disjoint_partition_us")
            require(
                math.isclose(sum(partition_totals.values()), timing["C_all_rounds"], abs_tol=1e-6),
                "aggregate partition does not close",
            )
            item.update(
                counts=counts,
                timing_us=timing,
                disjoint_partition_us=partition_totals,
                shared_combinations_us=sum_dict(requests, "shared_combinations_us"),
                inclusive_stage_us=sum_dict(requests, "inclusive_stage_us"),
                quality=round_quality(all_rounds),
                emitted_per_quality_round=counts["E_emitted_ids"] / counts["N_quality_rounds"],
                emitted_per_cost_row=counts["E_emitted_ids"] / counts["trace_rows"],
                begin_outside_round_us=sum(r["begin_outside_round_us"] for r in requests),
                inter_round_gaps_us=sum(r["inter_round_gaps_us"] for r in requests),
                untraced_leading_seed_tokens=sum(
                    r["untraced_leading_seed_tokens"] for r in requests
                ),
                round_cpu_us=distribution([row["round_us"] for row in all_rounds]),
            )
            for field in NESTED_FIELDS:
                values = [r["nested_scalar_us_not_additive"][field] for r in requests]
                item.setdefault("nested_scalar_us_not_additive", {})[field] = (
                    sum(values) if all(v is not None for v in values) else None
                )
            nested = [r["draft_stages"] for r in requests if r["draft_stages"]["available"]]
            item["draft_stages"] = {
                "requests_with_data": len(nested),
                "requests_without_data": len(requests) - len(nested),
                "matched_calls": sum(r["matched_calls"] for r in nested),
                "error_calls": sum(r["error_calls"] for r in nested),
                "stage_partition_us": sum_dict(nested, "stage_partition_us") if nested else {},
                "counts": sum_dict(nested, "counts") if nested else {},
                "nested_calls_total_us": sum(r["nested_calls_total_us"] for r in nested),
                "outer_draft_remainder_us": sum(r["outer_draft_remainder_us"] for r in nested),
                "additive_with_outer_round": False,
            }
            if (
                gate["passed"]
                and (quality_gate is None or quality_gate["passed"])
                and item["draft_stages"]["error_calls"] == 0
            ):
                item["conditional_fixed_trajectory_proxy"] = removal_proxy(
                    counts, timing, q4_primary["server_decode_tok_s"]
                )
            else:
                item["conditional_fixed_trajectory_proxy"] = None
                item["proxy_withheld_reason"] = (
                    "instrumented/reference behavior gate failed/unavailable or "
                    "draft-stage call reported error"
                )
        output["variants"][variant] = item
    if q4 in output["variants"] and output["variants"][q4].get("round_scope_available"):
        anchor = output["variants"][q4]
        rate = anchor["counts"]["E_emitted_ids"] * 1e6 / anchor["timing_us"]["C_all_rounds"]
        for item in output["variants"].values():
            proxy = item.get("conditional_fixed_trajectory_proxy")
            if proxy and {r["prompt_id"] for r in item["requests"]} == {
                r["prompt_id"] for r in anchor["requests"]
            }:
                proxy["q4_instrumented_round_tok_s"] = rate
                proxy["conditional_round_rate_over_q4_instrumented_round"] = (
                    proxy["conditional_draft_removed_round_tok_s"] / rate
                )
    output["draft_stage_files_coverage"] = [
        {
            "variant": v,
            "repetition": rep,
            "file_calls": len(indices),
            "matched_measured_calls": sum((v, rep, i) in used_stage_calls for i in indices),
            "unmatched_calls": sum((v, rep, i) not in used_stage_calls for i in indices),
            "unmatched_semantics": (
                "Warmups/outside selected measured round windows; not charged to measured rounds"
            ),
        }
        for (v, rep), indices in stage_files.items()
    ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--timed", type=Path, required=True)
    parser.add_argument("--timed-analysis", type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--round-paths", type=Path)
    parser.add_argument(
        "--draft-stages",
        type=Path,
        help="Optional JSON list of variant,repetition,path for per-server stage JSONL files",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        name: json.loads(path.read_text())
        for name in ("instrumented", "timed", "timed_analysis", "quality")
        if (path := getattr(args, name))
    }
    overrides = {}
    if args.round_paths:
        for row in json.loads(args.round_paths.read_text()):
            key = request_key(row)
            require(key not in overrides, "duplicate round-path override")
            path = Path(row["path"])
            overrides[key] = path if path.is_absolute() else args.round_paths.parent / path
    sources = []

    def load_rounds(record):
        path = overrides.get(request_key(record), Path(record["directory"]) / "rounds.json")
        require(
            path.is_file(), f"missing round file {path}; use --round-paths for relocated artifacts"
        )
        sources.append(
            {
                "variant": record["variant"],
                "prompt_id": record["prompt_id"],
                "repetition": record["repetition"],
                "path": str(path),
                "sha256": digest_file(path),
            }
        )
        return json.loads(path.read_text())

    stage_paths, stage_cache = {}, {}
    if args.draft_stages:
        for row in json.loads(args.draft_stages.read_text()):
            key = row["variant"], row["repetition"]
            require(key not in stage_paths, "duplicate draft-stage path override")
            path = Path(row["path"])
            stage_paths[key] = path if path.is_absolute() else args.draft_stages.parent / path

    def load_stages(record):
        key = record["variant"], record["repetition"]
        if key not in stage_paths:
            return None
        if key not in stage_cache:
            stage_cache[key] = [
                json.loads(line)
                for line in stage_paths[key].read_text().splitlines()
                if line.strip()
            ]
        return stage_cache[key]

    result = analyze(
        inputs["instrumented"],
        inputs["timed"],
        load_rounds,
        inputs.get("quality"),
        inputs.get("timed_analysis"),
        load_stages,
    )
    result["draft_stage_files"] = [
        {"variant": v, "repetition": rep, "path": str(path), "sha256": digest_file(path)}
        for (v, rep), path in stage_paths.items()
    ]
    result["sources"] = {
        name: {"path": str(getattr(args, name)), "sha256": digest_file(getattr(args, name))}
        for name in inputs
    }
    result["round_files"] = sources
    result["analysis_sources_sha256"] = {
        name: digest_file(Path(__file__).with_name(name))
        for name in (
            Path(__file__).name,
            "analyze_binary_rescue_benchmark.py",
            "analyze_w1ax_break_even.py",
            "analyze_w1ax_diagnostics.py",
            "run_binary_rescue_benchmark.py",
        )
    }
    write(args.output, result)
    print(f"Wrote CPU round analysis for {len(result['variants'])} variants to {args.output}")


if __name__ == "__main__":
    main()
