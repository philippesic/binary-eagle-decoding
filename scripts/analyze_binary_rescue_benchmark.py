#!/usr/bin/env python3
"""Aggregate rescue benchmarks with Q4_0 first and paired prompt-cluster uncertainty.

Never treat repetitions as independent quality prompts. Confidence intervals
resample prompt IDs, retaining every repetition and paired variant within each
sample. They describe this selected development workload, not final-set quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from run_binary_rescue_benchmark import round_quality, write


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return {
        "n": len(values),
        "p50": median(values) if values else None,
        "p95": values[math.ceil(0.95 * len(values)) - 1] if values else None,
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }


def aggregate(rows):
    tokens = sum(r["completion_tokens"] for r in rows)
    wall = sum(r["request_wall_s"] for r in rows)
    decode = [r.get("server_predicted_ms") for r in rows]
    prefill = [r.get("server_prompt_ms") for r in rows]
    valid_decode = len(decode) > 0 and all(v is not None and v > 0 for v in decode)
    decode_s = sum(decode) / 1000 if valid_decode else None
    counters = {}
    for key in ("accepted", "proposed", "rounds"):
        values = [r.get("speculative", {}).get(key) for r in rows]
        counters[key] = sum(values) if values and all(v is not None for v in values) else None
    a, p, n = (counters[k] for k in ("accepted", "proposed", "rounds"))
    counters.update(
        {
            "accepted_fraction": a / p if a is not None and p else None,
            "accepted_per_round": a / n if a is not None and n else None,
        }
    )
    return {
        "requests": len(rows),
        "prompt_count": len({r["prompt_id"] for r in rows}),
        "completion_tokens": tokens,
        "client_wall_s": wall,
        "server_decode_s": decode_s,
        "client_request_tok_s": tokens / wall if wall else None,
        "server_decode_tok_s": tokens / decode_s if decode_s else None,
        "server_decode_ms_per_token": decode_s * 1000 / tokens if decode_s and tokens else None,
        "client_request_s": distribution([r["request_wall_s"] for r in rows]),
        "ttft_s": distribution([r.get("ttft_s") for r in rows]),
        "server_prefill_ms": distribution(prefill),
        "inter_token_s": distribution([v for r in rows for v in r.get("inter_token_s", [])]),
        "inter_chunk_s": distribution([v for r in rows for v in r.get("inter_chunk_s", [])]),
        "inter_token_status_counts": dict_counts(
            r.get("inter_token_status", "missing") for r in rows
        ),
        "speculative": counters,
    }


def dict_counts(values):
    result = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def key(row):
    return row["prompt_id"], row["repetition"]


def divergence(reference, candidate):
    for i, (a, b) in enumerate(zip(reference, candidate)):
        if a != b:
            return {"position": i, "reference_id": a, "candidate_id": b, "reason": "token"}
    if len(reference) != len(candidate):
        return {
            "position": min(len(reference), len(candidate)),
            "reason": "length",
            "reference_length": len(reference),
            "candidate_length": len(candidate),
        }
    return None


def throughput_ratio(candidate, reference, field):
    a, b = aggregate(candidate)[field], aggregate(reference)[field]
    return a / b if a is not None and b else None


def paired_comparison(candidate, reference, samples=2000):
    c = {key(r): r for r in candidate}
    b = {key(r): r for r in reference}
    if len(c) != len(candidate) or len(b) != len(reference):
        raise ValueError("duplicate prompt/repetition pairing key")
    shared = sorted(c.keys() & b.keys())
    if not shared:
        raise ValueError("no paired observations")
    pairs = [
        {
            "prompt_id": k[0],
            "repetition": k[1],
            "candidate_wall_s": c[k]["request_wall_s"],
            "reference_wall_s": b[k]["request_wall_s"],
            "first_divergence": divergence(
                b[k]["generated_token_ids"], c[k]["generated_token_ids"]
            ),
        }
        for k in shared
    ]
    prompt_ids = sorted({k[0] for k in shared})
    by_prompt = {p: [k for k in shared if k[0] == p] for p in prompt_ids}
    fields = ("client_request_tok_s", "server_decode_tok_s")
    rng = random.Random(20260927)
    boot = {field: [] for field in fields}
    for _ in range(samples):
        selected = [k for p in rng.choices(prompt_ids, k=len(prompt_ids)) for k in by_prompt[p]]
        cr, br = [c[k] for k in selected], [b[k] for k in selected]
        # Avoid recomputing distributions in the bootstrap inner loop.
        ct, bt = sum(r["completion_tokens"] for r in cr), sum(r["completion_tokens"] for r in br)
        for field, span in ((fields[0], "request_wall_s"), (fields[1], "server_predicted_ms")):
            if all(r.get(span) is not None and r[span] > 0 for r in cr + br):
                boot[field].append(
                    (ct / sum(r[span] for r in cr)) / (bt / sum(r[span] for r in br))
                )
    result = {
        "paired_requests": len(shared),
        "paired_prompts": len(prompt_ids),
        "candidate_unpaired": len(c.keys() - b.keys()),
        "reference_unpaired": len(b.keys() - c.keys()),
        "same_raw_ids": sum(p["first_divergence"] is None for p in pairs),
        "pairs": pairs,
        "uncertainty": (
            "paired prompt-cluster percentile bootstrap; repetitions kept "
            "within prompt; development selection bias remains"
        ),
        "bootstrap_samples": samples,
        "per_prompt": {},
    }
    for field in fields:
        values = sorted(boot[field])
        result[field + "_ratio"] = throughput_ratio(
            [c[k] for k in shared], [b[k] for k in shared], field
        )
        result[field + "_ratio_ci95"] = (
            [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
            if values
            else None
        )
    for prompt, ks in by_prompt.items():
        result["per_prompt"][prompt] = {
            field + "_ratio": throughput_ratio([c[k] for k in ks], [b[k] for k in ks], field)
            for field in fields
        }
    return result


def behavior_compare(primary, diagnostic):
    """Compare each instrumented request against each primary repetition.

    Outputs alone cannot establish proposal parity. Counts are an additional
    gate; exact proposal parity is available only if both retain round traces.
    """
    index = defaultdict(list)
    for row in primary:
        if not row.get("warmup"):
            index[(row["variant"], row["prompt_id"])].append(row)
    results = []
    for row in diagnostic:
        if row.get("warmup"):
            continue
        matches = index.get((row["variant"], row["prompt_id"]), [])
        item = {
            "variant": row["variant"],
            "prompt_id": row["prompt_id"],
            "matched_requests": len(matches),
            "outputs_match": bool(matches)
            and all(r["generated_token_ids"] == row["generated_token_ids"] for r in matches),
            "counts_match": None,
            "proposal_ids_match": None,
        }
        keys = ("accepted", "proposed", "rounds")
        if (
            all(row.get("speculative", {}).get(k) is not None for k in keys)
            and matches
            and all(all(r.get("speculative", {}).get(k) is not None for k in keys) for r in matches)
        ):
            item["counts_match"] = all(
                all(r["speculative"][k] == row["speculative"][k] for k in keys) for r in matches
            )
        digest = row.get("request_digest")
        other_digests = [r.get("request_digest") for r in matches]
        item["proposal_digest_match"] = None
        if digest and matches and all(other_digests):
            item["proposal_digest_match"] = all(
                all(
                    d[k] == digest[k]
                    for k in ("proposal", "output", "rounds", "no_proposal", "output_tokens")
                )
                for d in other_digests
            )

        def signature(r):
            path = Path(r["directory"]) / "rounds.json"
            if not path.is_file():
                return None
            return [
                (v["n_accepted"], v["proposed_token_ids"])
                for v in json.loads(path.read_text())
                if v.get("status") != "checkpoint_replay"
            ]

        sig = signature(row)
        other = [signature(r) for r in matches]
        if sig is not None and matches and all(s is not None for s in other):
            item["proposal_ids_match"] = all(s == sig for s in other)
        results.append(item)
    return {
        "requests": results,
        "outputs_and_counts_verified": bool(results)
        and all(r["outputs_match"] and r["counts_match"] is True for r in results),
        "exact_proposals_verified": bool(results)
        and all(r["proposal_ids_match"] is True for r in results),
        "proposal_digests_verified": bool(results)
        and all(r["proposal_digest_match"] is True for r in results),
        "limitation": (
            "Production timing omits round dumps; FNV64 digests provide compact "
            "trajectory equality evidence with nonzero hash-collision risk. "
            "Outputs and counts alone do not prove proposal parity."
        ),
    }


def analyze(manifest):
    rows = [r for r in manifest["records"] if not r.get("warmup")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    q4 = manifest["q4_variant"]
    if q4 not in grouped:
        raise ValueError("Q4_0 reference missing")
    result = {
        "mode": manifest["mode"],
        "workload": manifest["workload"],
        "run_status": manifest["status"],
        "q4_variant": q4,
        "policy": manifest["policy"],
        "variants": {},
        "limitations": [
            "Concurrency-one streaming benchmark, not saturated serving capacity.",
            "Server decode time and client wall time have distinct boundaries.",
            "SSE batching/network arrival can prevent token-level ITL measurement.",
            "p95 uses nearest rank; sample counts accompany each distribution.",
        ],
    }
    for name in [q4] + [n for n in grouped if n != q4]:
        group = grouped[name]
        item = aggregate(group)
        item["versus_q4"] = paired_comparison(group, grouped[q4])
        if "target_only" in grouped:
            reference = {key(r): r for r in grouped["target_only"]}
            item["versus_target_only_raw_ids"] = [
                {
                    "prompt_id": r["prompt_id"],
                    "repetition": r["repetition"],
                    "first_divergence": divergence(
                        reference[key(r)]["generated_token_ids"], r["generated_token_ids"]
                    ),
                }
                for r in group
                if key(r) in reference
            ]
        # Quality uses one deterministic repetition, never N repeats as new prompts.
        first_rep = min(r["repetition"] for r in group)
        round_rows = []
        for row in group:
            path = Path(row["directory"]) / "rounds.json"
            if row["repetition"] == first_rep and path.is_file():
                round_rows += json.loads(path.read_text())
        if round_rows:
            item["single_pass_quality"] = round_quality(round_rows)
        item["graph_blocks"] = [
            b.get("graph_status", "unverified") for b in manifest["blocks"] if b["variant"] == name
        ]
        result["variants"][name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-instrumented", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = analyze(manifest)
    if args.compare_instrumented:
        other = json.loads(args.compare_instrumented.read_text())
        for field in ("policy", "prompt_sha256", "hashes", "workload"):
            if manifest[field] != other[field]:
                raise ValueError(f"behavior comparison has mismatched {field}")
        result["instrumented_behavior"] = behavior_compare(manifest["records"], other["records"])
    write(args.output, result)
    print(
        "| Variant | Accepted/round | Decode tok/s | Client tok/s | Wall p"
        "50/p95 s | TTFT p50/p95 s |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "unavailable"

    for name, item in result["variants"].items():
        wall, ttft = item["client_request_s"], item["ttft_s"]
        print(
            f"| {name} | {fmt(item['speculative']['accepted_per_round'])} | "
            f"{fmt(item['server_decode_tok_s'])} | {fmt(item['client_request_tok_s'])} | "
            f"{fmt(wall['p50'])}/{fmt(wall['p95'])} | {fmt(ttft['p50'])}/{fmt(ttft['p95'])} |"
        )


if __name__ == "__main__":
    main()
