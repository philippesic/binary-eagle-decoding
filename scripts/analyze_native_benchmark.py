"""Analyze paired native EAGLE records with pooled rates and paired bootstrap intervals."""

import argparse
import hashlib
import json
import random
from pathlib import Path

VARIANTS = ("target_only", "ordinary_eagle", "packed_head_w1a1")
MMA_VARIANT = "packed_head_w1a1_mma"
ALL_VARIANTS = (*VARIANTS, MMA_VARIANT)
GROUP_VARIANTS = (
    "packed_fusion_w1a1",
    "packed_attention_w1a1",
    "packed_ffn_w1a1",
    "packed_head_w1a1",
    "packed_all_w1a1",
)
GROUP_MATRIX_VARIANTS = (*VARIANTS[:2], *GROUP_VARIANTS)
WEIGHT_ONLY_VARIANTS = ("draft_q4_0", "draft_q8_0")
NATIVE_OPERAND_VARIANTS = ("draft_w8a8", "draft_w4a4")
METRICS = ("request", "decode")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_variants(records: list[dict], reported: list[str] | None = None) -> tuple[str, ...]:
    observed = {row["variant"] for row in records}
    weight_only = tuple(variant for variant in WEIGHT_ONLY_VARIANTS if variant in observed)
    native_operand = tuple(variant for variant in NATIVE_OPERAND_VARIANTS if variant in observed)
    core = observed - set(weight_only) - set(native_operand)
    if core == set(GROUP_MATRIX_VARIANTS):
        variants = (*GROUP_MATRIX_VARIANTS, *weight_only, *native_operand)
    elif core == set(ALL_VARIANTS):
        variants = (*ALL_VARIANTS, *weight_only, *native_operand)
    elif core == set(VARIANTS):
        variants = (*VARIANTS, *weight_only, *native_operand)
    else:
        variants = VARIANTS
    if observed != set(variants) or (reported is not None and tuple(reported) != variants):
        raise ValueError("benchmark variants are incomplete or unknown")
    return variants


def paired_index(
    records: list[dict], variants: tuple[str, ...] | None = None
) -> tuple[dict, list[int], list[str]]:
    variants = variants or selected_variants(records)
    index = {}
    for row in records:
        key = (row["repetition"], row["prompt_id"], row["variant"])
        if key in index or row["variant"] not in variants:
            raise ValueError("duplicate or unknown variant in benchmark records")
        index[key] = row
    if not index:
        raise ValueError("benchmark records are empty")
    repetitions = sorted({key[0] for key in index})
    prompts = sorted({key[1] for key in index})
    for repetition in repetitions:
        for prompt in prompts:
            if any((repetition, prompt, variant) not in index for variant in variants):
                raise ValueError(f"incomplete paired records for {repetition}/{prompt}")
    return index, repetitions, prompts


def pooled_rate(rows: list[dict], variant: str, metric: str) -> float | None:
    selected = [row for row in rows if row["variant"] == variant]
    tokens = [row.get("completion_tokens") for row in selected]
    if not selected or any(not isinstance(value, int) or value < 0 for value in tokens):
        return None
    field = "request_wall_s" if metric == "request" else "server_predicted_ms"
    seconds = [row.get(field) for row in selected]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in seconds):
        return None
    total_s = sum(seconds) if metric == "request" else sum(seconds) / 1000
    return sum(tokens) / total_s if total_s > 0 else None


def pooled_speedup(
    rows: list[dict], anchor: str, metric: str, candidate: str = "packed_head_w1a1"
) -> float | None:
    packed = pooled_rate(rows, candidate, metric)
    baseline = pooled_rate(rows, anchor, metric)
    return (
        packed / baseline if packed is not None and baseline is not None and baseline > 0 else None
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_bootstrap(
    records: list[dict],
    samples: int,
    seed: int,
    candidate: str = "packed_head_w1a1",
) -> dict:
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    variants = selected_variants(records)
    if candidate not in variants or candidate == "target_only":
        raise ValueError("candidate variant is unavailable")
    anchors = tuple(variant for variant in variants if variant != candidate)
    index, repetitions, prompts = paired_index(records, variants)
    rng = random.Random(seed)
    draws = {(anchor, metric): [] for anchor in anchors for metric in METRICS}
    unavailable = set()
    for _ in range(samples):
        selected_repetitions = rng.choices(repetitions, k=len(repetitions))
        selected_prompts = rng.choices(prompts, k=len(prompts))
        sample_rows = [
            index[repetition, prompt, variant]
            for repetition in selected_repetitions
            for prompt in selected_prompts
            for variant in variants
        ]
        for key, values in draws.items():
            if key in unavailable:
                continue
            value = pooled_speedup(sample_rows, *key, candidate=candidate)
            if value is None:
                unavailable.add(key)
            else:
                values.append(value)
    intervals = {}
    for (anchor, metric), values in draws.items():
        if (anchor, metric) in unavailable or len(values) != samples:
            intervals[f"{anchor}/{metric}"] = None
            continue
        values.sort()
        intervals[f"{anchor}/{metric}"] = {
            "p2_5": percentile(values, 0.025),
            "median": percentile(values, 0.5),
            "p97_5": percentile(values, 0.975),
        }
    return intervals


def category_summary(records: list[dict], categories: dict[str, str]) -> dict:
    variants = selected_variants(records)
    result = {}
    for category in sorted(set(categories.values())):
        result[category] = {}
        for variant in variants:
            rows = [
                row
                for row in records
                if row["variant"] == variant and categories[row["prompt_id"]] == category
            ]
            counters = [row["speculative"] for row in rows]
            accepted = (
                sum(item["accepted"] for item in counters)
                if all(item["accepted"] is not None for item in counters)
                else None
            )
            rounds = (
                sum(item["rounds"] for item in counters)
                if all(item["rounds"] is not None for item in counters)
                else None
            )
            result[category][variant] = {
                "requests": len(rows),
                "request_tokens_per_s": pooled_rate(rows, variant, "request"),
                "decode_tokens_per_s": pooled_rate(rows, variant, "decode"),
                "accepted_per_round": accepted / rounds
                if accepted is not None and rounds
                else None,
            }
    return result


def analyze(run_dir: Path, samples: int, seed: int) -> dict:
    report_path = run_dir / "report.json"
    records_path = run_dir / "records.json"
    prompts_path = run_dir / "prompts.jsonl"
    report = json.loads(report_path.read_text())
    records = json.loads(records_path.read_text())
    if report.get("status") != "complete" or report.get("records") != len(records):
        raise ValueError("benchmark run is incomplete or records count differs")
    variants = selected_variants(records, report.get("variants"))
    native_operand = tuple(variant for variant in variants if variant in NATIVE_OPERAND_VARIANTS)
    if native_operand:
        dispatch = report.get("native_operand_dispatch_confirmed_by_variant", {})
        if any(dispatch.get(variant) is not True for variant in native_operand):
            raise ValueError("native operand runtime dispatch is unconfirmed")
    _, repetitions, prompts = paired_index(records, variants)
    prompt_rows = [json.loads(line) for line in prompts_path.read_text().splitlines() if line]
    categories = {row["id"]: row["category"] for row in prompt_rows}
    if set(categories) != set(prompts):
        raise ValueError("benchmark prompts differ from paired records")
    packed_speedups = {
        variant: {
            f"{anchor}/{metric}": pooled_speedup(records, anchor, metric, variant)
            for anchor in VARIANTS[:2]
            for metric in METRICS
        }
        for variant in variants
        if variant.startswith(("packed_", "draft_q", "draft_w"))
    }
    portable_speedups = {
        f"{anchor}/{metric}": pooled_speedup(records, anchor, metric)
        for anchor in VARIANTS[:2]
        for metric in METRICS
    }
    result = {
        "source_sha256": {
            "report.json": sha256(report_path),
            "records.json": sha256(records_path),
            "prompts.jsonl": sha256(prompts_path),
        },
        "repetitions": len(repetitions),
        "prompts": len(prompts),
        "variants": list(variants),
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "pooled_packed_speedup": portable_speedups,
        "pooled_variant_speedups": packed_speedups,
        "native_operand_variant_specs": report.get("native_operand_variant_specs", {}),
        "paired_prompt_repetition_bootstrap_95pct": paired_bootstrap(records, samples, seed),
        "category_summary": category_summary(records, categories),
        "interpretation": (
            f"Resample prompt IDs and repetitions with replacement, preserving all {len(variants)} "
            "variants per draw; each speedup is a ratio of pooled token/time rates. "
            "These descriptive intervals do not correct model or hardware systematic bias."
        ),
    }
    result["paired_variant_bootstrap_95pct"] = {
        variant: paired_bootstrap(records, samples, seed, variant)
        for variant in variants
        if variant.startswith(("packed_", "draft_q", "draft_w")) and variant != MMA_VARIANT
    }
    if MMA_VARIANT in variants:
        result["pooled_mma_speedup"] = {
            f"{anchor}/{metric}": pooled_speedup(records, anchor, metric, MMA_VARIANT)
            for anchor in VARIANTS
            for metric in METRICS
        }
        result["paired_mma_bootstrap_95pct"] = paired_bootstrap(records, samples, seed, MMA_VARIANT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    result = analyze(run_dir, args.samples, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
