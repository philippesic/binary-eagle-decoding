#!/usr/bin/env python3
"""Validate and summarize the two frozen W1Ax context runs without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path

# Support both direct CLI execution and importing this file in offline tests.
try:
    from scripts.analyze_native_benchmark import (
        W1AX_BITS,
        W1AX_MATRIX_VARIANTS,
        paired_bootstrap,
        paired_index,
        percentile,
        pooled_rate,
        sha256,
    )
except ModuleNotFoundError:
    from analyze_native_benchmark import (
        W1AX_BITS,
        W1AX_MATRIX_VARIANTS,
        paired_bootstrap,
        paired_index,
        percentile,
        pooled_rate,
        sha256,
    )

FROZEN_PROMPT_SHA256 = "5653cfe7599e5dd4ae44e057df27b816221f8ee89635056bc0fb24b9f44a21a3"
VARIANTS = W1AX_MATRIX_VARIANTS
BOUNDS = {"short": (1, 256), "medium": (257, 768), "long": (769, 1536)}
CATEGORIES = {"prose", "code", "reasoning"}
ANCHORS = {"fp16_eagle": "ordinary_eagle", "q4_0_eagle": "draft_q4_0"}
OUTPUT_ANCHORS = {**ANCHORS, "target_only": "target_only"}
MODEL_KEYS = ("target", "ordinary_draft", "draft_q8_0", "draft_q4_0", *W1AX_BITS)
POLICY = {
    "mode": "primary_matrix",
    "prompt_set": "context_diagnostic",
    "max_draft_tokens": 5,
    "min_draft_probability": 0.0,
    "warmup_requests": 2,
    "repetitions": 5,
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_number(value, *, positive=False) -> bool:
    return (
        type(value) in (float, int)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def read_json(path: Path):
    return json.loads(path.read_text())


def distribution(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "observations": len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def validate_records(records: list[dict], prompts: list[dict], cap: int) -> None:
    check(isinstance(records, list) and len(records) == 360, f"cap {cap}: expected 360 records")
    index, repetitions, prompt_ids = paired_index(records, VARIANTS)
    check(
        repetitions == list(range(5)) and set(prompt_ids) == {p["id"] for p in prompts},
        f"cap {cap}: expected five repetitions of all nine frozen prompts",
    )
    bins = {p["id"]: p["context_bin"] for p in prompts}
    for (rep, prompt, variant), row in index.items():
        label = f"cap {cap}/{rep}/{prompt}/{variant}"
        check(type(row.get("repetition")) is int, f"{label}: invalid repetition")
        check(
            row.get("request_id") == f"rep-{rep:02d}/{variant}/{prompt}",
            f"{label}: invalid request ID",
        )
        check(
            row.get("policy_mode") == POLICY["mode"]
            and row.get("max_draft_tokens") == 5
            and row.get("min_draft_probability") == 0.0,
            f"{label}: wrong record policy",
        )
        expected_bits = str(W1AX_BITS[variant]) if variant in W1AX_BITS else None
        check(
            row.get("w1ax_activation_bits_selector") == expected_bits,
            f"{label}: wrong W1Ax activation selector",
        )
        ids = row.get("generated_token_ids")
        check(
            isinstance(ids, list) and bool(ids) and all(type(t) is int and t >= 0 for t in ids),
            f"{label}: missing or invalid raw token IDs",
        )
        check(
            row.get("generated_token_ids_sha256")
            == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            f"{label}: raw token ID hash mismatch",
        )
        tokens = row.get("completion_tokens")
        check(
            type(tokens) is int and 0 < tokens <= cap and len(ids) == tokens,
            f"{label}: invalid completion count or raw ID length",
        )
        for field in ("request_wall_s", "server_predicted_ms", "server_prompt_ms"):
            check(
                valid_number(row.get(field), positive=field != "server_prompt_ms"),
                f"{label}: missing or invalid {field}",
            )
        length = row.get("prompt_tokens")  # extract_record: response.usage.prompt_tokens
        lower, upper = BOUNDS[bins[prompt]]
        check(
            type(length) is int and lower <= length <= upper,
            f"{label}: native prompt_tokens outside {bins[prompt]} bin or missing",
        )


def load_run(run: Path, cap: int) -> tuple[dict, list[dict], list[dict]]:
    run = run.resolve()
    manifest, report = read_json(run / "manifest.json"), read_json(run / "report.json")
    check(
        report.get("status") == "complete" and report.get("records") == 360,
        f"cap {cap}: incomplete report or wrong record count",
    )
    check(
        manifest.get("variants") == list(VARIANTS) and report.get("variants") == list(VARIANTS),
        f"cap {cap}: expected eight paths",
    )
    check(
        manifest.get("policy") == POLICY and report.get("policy") == POLICY,
        f"cap {cap}: expected primary D5/p0, two warmups, five repetitions",
    )
    check(
        manifest.get("request_options", {}).get("max_tokens") == cap, f"cap {cap}: wrong output cap"
    )
    config = tomllib.loads((run / "config.toml").read_text())
    evaluation = config.get("evaluation", {})
    check(
        all(
            evaluation.get(k) == v
            for k, v in {
                "max_output_tokens": cap,
                "repetitions": 5,
                "warmup_requests": 2,
                "max_draft_tokens": 5,
                "min_draft_probability": 0.0,
                "prompt_set": "context_diagnostic",
                "w1ax_matrix": True,
                "w1ax_policy_diagnostic": False,
            }.items()
        ),
        f"cap {cap}: config cap/policy disagreement",
    )
    check(
        sha256(run / "config.toml") == manifest.get("config_sha256"),
        f"cap {cap}: config hash disagreement",
    )
    files = manifest.get("files", {})
    binary_hash = files.get("binary", {}).get("sha256")
    check(
        isinstance(binary_hash, str)
        and len(binary_hash) == 64
        and all(c in "0123456789abcdef" for c in binary_hash),
        f"cap {cap}: binary hash missing or invalid",
    )
    check(
        isinstance(manifest.get("precision"), dict) and bool(manifest["precision"]),
        f"cap {cap}: precision metadata missing",
    )
    check(
        sha256(run / "prompts.jsonl") == FROZEN_PROMPT_SHA256
        and files.get("prompt_file", {}).get("sha256") == FROZEN_PROMPT_SHA256,
        f"cap {cap}: frozen prompt SHA changed",
    )
    prompts = [
        json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines() if line
    ]
    check(
        len(prompts) == 9
        and len({p["id"] for p in prompts}) == 9
        and {(p["context_bin"], p["category"]) for p in prompts}
        == {(b, c) for b in BOUNDS for c in CATEGORIES},
        f"cap {cap}: expected three bins by three categories",
    )
    check(
        manifest.get("prompt_ids") == [p["id"] for p in prompts],
        f"cap {cap}: manifest prompt IDs differ",
    )
    check(
        all(report.get("w1ax_dispatch_confirmed_by_variant", {}).get(v) is True for v in W1AX_BITS),
        f"cap {cap}: unconfirmed W1Ax dispatch",
    )
    models = {k: files.get(k, {}).get("sha256") for k in MODEL_KEYS}
    check(
        all(
            isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
            for h in models.values()
        ),
        f"cap {cap}: model hash missing or invalid",
    )
    check(
        len({models[v] for v in W1AX_BITS}) == 1,
        f"cap {cap}: W1Ax paths do not share one packed model",
    )
    for variant in ("draft_q8_0", "draft_q4_0", *W1AX_BITS):
        check(
            manifest.get("variant_specs", {}).get(variant, {}).get("draft_model_sha256")
            == models[variant],
            f"cap {cap}: variant model hash differs: {variant}",
        )
    records = read_json(run / "records.json")
    validate_records(records, prompts, cap)
    for row in records:
        model = "ordinary_draft" if row["variant"] == "ordinary_eagle" else row["variant"]
        expected = None if model == "target_only" else models[model]
        check(
            row.get("draft_model_sha256") == expected,
            f"cap {cap}: record model hash differs: {row['request_id']}",
        )
    metadata = {
        "run_dir": str(run),
        "cap": cap,
        "records": len(records),
        "model_hashes": models,
        "sources": {
            name: {"path": str(run / name), "sha256": sha256(run / name)}
            for name in (
                "manifest.json",
                "report.json",
                "records.json",
                "config.toml",
                "prompts.jsonl",
            )
        },
        "policy": POLICY,
        "request_options": manifest["request_options"],
        "precision": manifest.get("precision"),
        "variant_specs": manifest.get("variant_specs"),
        "initial_gpu_snapshot": manifest.get("initial_gpu_snapshot"),
        "environment_manifest": manifest.get("environment_manifest"),
        "server_binary": files.get("binary"),
        "project_commit": manifest.get("project_commit"),
        "llama_checkout_commit": manifest.get("llama_checkout_commit"),
        "dispatch_validation": (
            "Report all-repetition W1Ax dispatch flags plus per-record selectors."
        ),
    }
    return metadata, records, prompts


def output_comparison(candidate: list[dict], reference: list[dict], *, prefix_only=False) -> dict:
    """Missing data is fatal; observed differences are descriptive and nonfatal."""
    indexes = [
        {(r["repetition"], r["prompt_id"]): r for r in rows} for rows in (candidate, reference)
    ]
    a, b = indexes
    check(
        len(a) == len(candidate) and len(b) == len(reference) and a.keys() == b.keys() and a,
        "output comparison has missing or duplicate pairs",
    )
    mismatches, lengths = [], []
    for rep, pid in sorted(a):
        left, right = a[rep, pid], b[rep, pid]
        x, y = left.get("generated_token_ids"), right.get("generated_token_ids")
        check(
            isinstance(x, list) and isinstance(y, list) and x and y,
            "output comparison missing raw token IDs",
        )
        common = min(len(x), len(y))
        first = next((i for i in range(common) if x[i] != y[i]), None)
        if first is None and not prefix_only and len(x) != len(y):
            first = common
        pair = {
            "repetition": rep,
            "prompt_id": pid,
            "candidate_length": len(x),
            "reference_length": len(y),
            "compared_prefix_length": common,
        }
        lengths.append(pair)
        if first is not None:
            mismatches.append(
                {
                    **pair,
                    "first_difference": {
                        "index": first,
                        "candidate_token_id": x[first] if first < len(x) else None,
                        "reference_token_id": y[first] if first < len(y) else None,
                    },
                    "candidate_request_id": left["request_id"],
                    "reference_request_id": right["request_id"],
                    "candidate_token_ids_sha256": left["generated_token_ids_sha256"],
                    "reference_token_ids_sha256": right["generated_token_ids_sha256"],
                }
            )
    return {
        "scope": "common_prefix_only" if prefix_only else "full_observed_sequences",
        "paired_requests": len(a),
        "matched_requests": len(a) - len(mismatches),
        "mismatched_requests": len(mismatches),
        "unequal_observed_length_requests": sum(
            p["candidate_length"] != p["reference_length"] for p in lengths
        ),
        "mismatches": mismatches,
        "length_scope_by_pair": lengths,
        "first_difference_convention": "Zero-based index; null token ID means the sequence ended.",
        "strict_lossless_speedup_claim": False,
        "interpretation": (
            "Only the shorter observed prefix is compared across caps; length "
            "differences alone are not mismatches. "
            if prefix_only
            else ""
        )
        + "Agreement is limited to these requests. Differences have no assumed cause; "
        "rate ratios remain timing observations, with no universal losslessness claim.",
    }


def summarize(records: list[dict], samples: int, seed: int) -> dict:
    result = {}
    for variant in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        result[variant] = {
            "requests": len(rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "decode_tokens_per_s": pooled_rate(records, variant, "decode"),
            "request_tokens_per_s": pooled_rate(records, variant, "request"),
            "prefill_server_ms": distribution([r["server_prompt_ms"] for r in rows]),
            "http_wall_s": distribution([r["request_wall_s"] for r in rows]),
            "vs_anchors": {
                name: {
                    metric: pooled_rate(records, variant, metric)
                    / pooled_rate(records, anchor, metric)
                    for metric in ("decode", "request")
                }
                for name, anchor in ANCHORS.items()
            },
        }
        if samples and variant in W1AX_BITS:
            intervals = paired_bootstrap(records, samples, seed, candidate=variant)
            result[variant]["paired_bootstrap_vs_anchors"] = {
                name: {metric: intervals[f"{anchor}/{metric}"] for metric in ("decode", "request")}
                for name, anchor in ANCHORS.items()
            }
    return result


def analyze(
    cap32: Path, cap128: Path, *, hf_audit: Path | None = None, samples: int = 0, seed: int = 42
) -> dict:
    check(samples == 0 or samples >= 100, "bootstrap samples must be zero or at least 100")
    loaded = {cap: load_run(run, cap) for cap, run in ((32, cap32), (128, cap128))}
    check(
        loaded[32][0]["model_hashes"] == loaded[128][0]["model_hashes"],
        "model hashes differ between caps",
    )
    binary_hashes = {str(cap): loaded[cap][0]["server_binary"]["sha256"] for cap in (32, 128)}
    check(binary_hashes["32"] == binary_hashes["128"], "server binary hashes differ between caps")
    build_provenance = {"server_binary_sha256": {"status": "match", "by_cap": binary_hashes}}
    for field in ("project_commit", "llama_checkout_commit"):
        values = {str(cap): loaded[cap][0].get(field) for cap in (32, 128)}
        available = all(isinstance(value, str) and value.strip() for value in values.values())
        build_provenance[field] = {
            "status": ("match" if values["32"] == values["128"] else "different")
            if available
            else "unavailable",
            "by_cap": values,
        }
    options = [
        {k: v for k, v in loaded[cap][0]["request_options"].items() if k != "max_tokens"}
        for cap in (32, 128)
    ]
    check(options[0] == options[1], "request options differ between caps beyond output cap")
    check(
        loaded[32][0]["precision"] == loaded[128][0]["precision"], "precision differs between caps"
    )
    hf_rows = {}
    hf_source = None
    if hf_audit is not None:
        audit = read_json(hf_audit)
        check(audit.get("prompts_sha256") == FROZEN_PROMPT_SHA256, "HF audit prompt SHA differs")
        hf_rows = {r["id"]: r for r in audit["rows"]}
        check(
            len(audit["rows"]) == 9 and set(hf_rows) == {p["id"] for p in loaded[32][2]},
            "HF audit prompt IDs differ",
        )
        check(
            all(
                type(r.get("tokenized_prompt_length")) is int and r["tokenized_prompt_length"] > 0
                for r in hf_rows.values()
            ),
            "HF audit lengths missing or invalid",
        )
        hf_source = {"path": str(hf_audit.resolve()), "sha256": sha256(hf_audit), "audit": audit}
    output = {
        "schema": "w1ax_context_matrix_analysis_v1",
        "frozen_prompt_sha256": FROZEN_PROMPT_SHA256,
        "cross_cap_build_provenance": build_provenance,
        "hf_freeze_audit": hf_source,
        "caps": {},
        "cross_cap_outputs": {},
        "definitions": {
            "sampling": (
                "Five repetitions of three prompts (one per category) per bin: 15 "
                "requests per variant/bin/cap."
            ),
            "rates": (
                "Pooled completion tokens / summed time. Decode uses "
                "server_predicted_ms; request uses HTTP client request_wall_s "
                "including prefill."
            ),
            "latencies": (
                "Median/p95/max over 15 request observations per bin, or 45 overall. "
                "p95 uses linear interpolation, not a confidence bound; repeated "
                "prompts are not 15 independent prompts."
            ),
            "prefill": (
                "server_prompt_ms is reported server prompt processing time, not time "
                "to first token."
            ),
            "native_prompt_length": (
                "prompt_tokens from response.usage.prompt_tokens, distinct from HF "
                "apply_chat_template freeze lengths; server_prompt_n retained for "
                "comparison."
            ),
            "bootstrap": (
                "Optional paired resampling of repetition and prompt factors; not a "
                "claim about broader workloads."
            ),
            "claims": (
                "Synthetic context diagnostic, not final-set quality or general "
                "losslessness. Output differences are nonfatal timing observations "
                "with no inferred cause."
            ),
        },
        "bootstrap": {"samples": samples, "seed": seed},
    }
    for cap, (metadata, records, prompts) in loaded.items():
        token_lengths = []
        for prompt in prompts:
            rows = [r for r in records if r["prompt_id"] == prompt["id"]]
            lengths = sorted({r["prompt_tokens"] for r in rows})
            hf_length = hf_rows.get(prompt["id"], {}).get("tokenized_prompt_length")
            token_lengths.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "context_bin": prompt["context_bin"],
                    "bin_bounds": BOUNDS[prompt["context_bin"]],
                    "native_prompt_tokens_observed": lengths,
                    "within_native_bin": True,
                    "native_server_prompt_n_observed": sorted(
                        {
                            r["server_prompt_n"]
                            for r in rows
                            if type(r.get("server_prompt_n")) is int
                        }
                    ),
                    "native_server_prompt_n_missing_or_invalid": sum(
                        type(r.get("server_prompt_n")) is not int for r in rows
                    ),
                    "hf_freeze_prompt_tokens": hf_length,
                    "native_minus_hf_observed": [n - hf_length for n in lengths]
                    if hf_length
                    else None,
                }
            )
        comparisons = {}
        for variant in VARIANTS:
            for name, anchor in OUTPUT_ANCHORS.items():
                if variant == anchor:
                    continue
                comparisons[f"{variant}/vs/{name}"] = {
                    "candidate_source": {"variant": variant, **metadata["sources"]["records.json"]},
                    "reference_source": {"variant": anchor, **metadata["sources"]["records.json"]},
                    **output_comparison(
                        [r for r in records if r["variant"] == variant],
                        [r for r in records if r["variant"] == anchor],
                    ),
                }
        output["caps"][str(cap)] = {
            **metadata,
            "prompt_lengths": token_lengths,
            "overall": summarize(records, samples, seed),
            "by_bin": {
                b: summarize(
                    [
                        r
                        for r in records
                        if r["prompt_id"] in {p["id"] for p in prompts if p["context_bin"] == b}
                    ],
                    samples,
                    seed,
                )
                for b in BOUNDS
            },
            "output_comparisons": comparisons,
        }
    for variant in VARIANTS:
        output["cross_cap_outputs"][variant] = {
            "candidate_source": {
                "cap": 128,
                "variant": variant,
                **loaded[128][0]["sources"]["records.json"],
            },
            "reference_source": {
                "cap": 32,
                "variant": variant,
                **loaded[32][0]["sources"]["records.json"],
            },
            **output_comparison(
                [r for r in loaded[128][1] if r["variant"] == variant],
                [r for r in loaded[32][1] if r["variant"] == variant],
                prefix_only=True,
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap32", type=Path, required=True)
    parser.add_argument("--cap128", type=Path, required=True)
    parser.add_argument("--hf-audit", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(
            args.cap32,
            args.cap128,
            hf_audit=args.hf_audit,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"context analysis failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "records": 720,
                "bootstrap_samples": args.bootstrap_samples,
            }
        )
    )


if __name__ == "__main__":
    main()
