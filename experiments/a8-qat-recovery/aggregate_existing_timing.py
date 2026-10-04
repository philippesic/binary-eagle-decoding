#!/usr/bin/env python3
"""CPU-only audit of existing a8_native_request_metrics_v1 manifests.

Reads bytes without altering them. Requires exactly five measured repetitions
of 24 prompts per variant; warmups must not appear among measured records.
No GPU, network, project imports, raw token output, or inferential statistics.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

VARIANTS = ("A8", "Q4_0", "target_only")
RATES = ("request_tokens_per_s", "decode_tokens_per_s")
POOLED_FIELDS = ("requests", "completion_tokens", "request_wall_s",
                 "decode_server_s", *RATES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, context):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{context}: expected finite positive number")
    return value


def aggregate(rows):
    tokens = sum(row["completion_tokens"] for row in rows)
    wall = math.fsum(row["request_wall_s"] for row in rows)
    decode_ms = math.fsum(row["server_predicted_ms"] for row in rows)
    return {"requests": len(rows), "completion_tokens": tokens,
            "request_wall_s": wall, "decode_server_s": decode_ms / 1000,
            "request_tokens_per_s": tokens / wall,
            "decode_tokens_per_s": tokens * 1000 / decode_ms}


def ratios(variants):
    return {baseline: {rate: variants["A8"][rate] / variants[baseline][rate]
                       for rate in RATES}
            for baseline in ("Q4_0", "target_only")}


def distribution(values):
    return {"n": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values),
            "range": max(values) - min(values)}


def same_number(actual, reported, context):
    positive(reported, context)
    require(math.isclose(actual, reported, rel_tol=1e-9, abs_tol=1e-9),
            f"{context}: pooled recomputation differs from reported value")


def audit(report, source_path, source_sha256):
    require(isinstance(report, dict), "manifest must be an object")
    require(report.get("schema") == "a8_native_request_metrics_v1", "unsupported schema")
    require(report.get("kind") == "native_request_timing_no_tensor_capture", "unsupported kind")
    require(report.get("complete") is True and report.get("status") == "complete",
            "incomplete manifest")
    require(report.get("metrics_status") == "complete_request_metrics",
            "complete positive decode metrics required")
    require(type(report.get("repetitions")) is int and report["repetitions"] == 5,
            "exactly five repetitions required")
    require(type(report.get("warmup_requests_per_server")) is int
            and report["warmup_requests_per_server"] >= 1, "warmup contract missing")
    prompts = report.get("prompt_ids")
    require(isinstance(prompts, list) and len(prompts) == 24
            and all(isinstance(prompt, str) and prompt for prompt in prompts)
            and len(set(prompts)) == 24, "exactly 24 unique prompt IDs required")
    prompt_set = set(prompts)
    rows = report.get("records")
    require(isinstance(rows, list) and len(rows) == 5 * 24 * 3,
            "exactly 360 measured records required; missing or extra/warmup records")
    by_key = {}
    observed_orders = [[] for _ in range(5)]
    for row in rows:
        require(isinstance(row, dict), "record must be an object")
        rep, variant, prompt = row.get("repetition"), row.get("variant"), row.get("prompt_id")
        require(type(rep) is int and 0 <= rep < 5, "invalid repetition")
        require(isinstance(variant, str) and variant in VARIANTS, "unknown variant")
        require(isinstance(prompt, str) and prompt in prompt_set, "unknown prompt ID")
        require(not row.get("warmup") and not row.get("is_warmup")
                and row.get("phase") != "warmup", "warmup in measured records")
        key = rep, variant, prompt
        require(key not in by_key, f"duplicate measured record: {rep}/{variant}/{prompt}")
        by_key[key] = row
        if variant not in observed_orders[rep]:
            observed_orders[rep].append(variant)
        tokens, ids = row.get("completion_tokens"), row.get("generated_token_ids")
        require(type(tokens) is int and tokens > 0, "positive integer completion count required")
        require(isinstance(ids, list) and len(ids) == tokens
                and all(type(token) is int and token >= 0 for token in ids),
                "completion count must equal real integer token ID length")
        positive(row.get("request_wall_s"), "request_wall_s")
        positive(row.get("server_predicted_ms"), "server_predicted_ms")
    for rep in range(5):
        for variant in VARIANTS:
            require({prompt for r, v, prompt in by_key if r == rep and v == variant} == prompt_set,
                    "repetition/variant prompt set differs")
        for prompt in prompts:
            reference = by_key[rep, "Q4_0", prompt]["generated_token_ids"]
            for variant in ("A8", "target_only"):
                require(by_key[rep, variant, prompt]["generated_token_ids"] == reference,
                        f"token ID parity failed: {rep}/{variant}/{prompt}")
    orders = report.get("orders")
    if orders is not None:
        require(isinstance(orders, list) and len(orders) == 5
                and all(isinstance(order, list) and len(order) == 3
                        and all(isinstance(v, str) for v in order)
                        and set(order) == set(VARIANTS) for order in orders),
                "invalid repetition orders")
        require(orders == observed_orders, "declared order differs from observed record order")
    pooled = {variant: aggregate([row for row in rows if row["variant"] == variant])
              for variant in VARIANTS}
    reported = report.get("variants")
    require(isinstance(reported, dict) and set(reported) == set(VARIANTS),
            "reported variant aggregates missing")
    checked_fields = 0
    for variant in VARIANTS:
        require(isinstance(reported[variant], dict), "reported variant must be an object")
        for field in POOLED_FIELDS:
            value = reported[variant].get(field)
            if field in ("requests", "completion_tokens"):
                require(type(value) is int and value == pooled[variant][field],
                        f"reported {variant}/{field} differs")
            else:
                same_number(pooled[variant][field], value, f"reported {variant}/{field}")
            checked_fields += 1
    pooled_ratios = ratios(pooled)
    for baseline, field in (("Q4_0", "speedup_vs_q4_0"),
                            ("target_only", "speedup_vs_no_speculation")):
        require(isinstance(report.get(field), dict), f"reported {field} missing")
        for rate in RATES:
            same_number(pooled_ratios[baseline][rate], report[field].get(rate), f"{field}/{rate}")
            checked_fields += 1
    repeats = []
    for rep in range(5):
        variants = {variant: aggregate([by_key[rep, variant, prompt] for prompt in prompts])
                    for variant in VARIANTS}
        repeats.append({"repetition": rep,
                        "variant_order": orders[rep] if orders is not None else None,
                        "variants": variants, "a8_ratios": ratios(variants)})
    return {"source": {"path": source_path, "sha256": source_sha256},
            "contract": {"repetitions": 5, "prompts_per_variant_per_repetition": 24,
                         "measured_records": len(rows), "warmups_excluded": True,
                         "matched_token_id_pairs_vs_q4_0": {"A8": 120, "target_only": 120}},
            "repetitions": repeats,
            "pooled": {"variants": pooled, "a8_ratios": pooled_ratios,
                       "reported_aggregate_crosscheck": {"passed": True,
                                                        "numeric_fields_checked": checked_fields,
                                                        "rel_tol": 1e-9, "abs_tol": 1e-9}},
            "repeat_distributions": {
                "variants": {variant: {field: distribution([rep["variants"][variant][field]
                                                             for rep in repeats])
                                       for field in POOLED_FIELDS if field != "requests"}
                             for variant in VARIANTS},
                "a8_ratios": {baseline: {rate: distribution([rep["a8_ratios"][baseline][rate]
                                                             for rep in repeats])
                                         for rate in RATES}
                              for baseline in ("Q4_0", "target_only")}}}


def fixture():
    prompts = [f"prompt-{n:02d}" for n in range(24)]
    orders = [["Q4_0", "A8", "target_only"], ["A8", "target_only", "Q4_0"],
              ["target_only", "Q4_0", "A8"], ["target_only", "A8", "Q4_0"],
              ["Q4_0", "target_only", "A8"]]
    rows = []
    for rep, order in enumerate(orders):
        for variant in order:
            scale = {"A8": 1, "Q4_0": 2, "target_only": 3}[variant] * (1 + rep / 4)
            for n, prompt in enumerate(prompts):
                wall = (1 if n % 2 == 0 else 9) * scale
                rows.append({"repetition": rep, "variant": variant, "prompt_id": prompt,
                             "completion_tokens": 10, "generated_token_ids": list(range(10)),
                             "request_wall_s": wall, "server_predicted_ms": wall * 500})
    variants = {v: aggregate([r for r in rows if r["variant"] == v]) for v in VARIANTS}
    ratio = ratios(variants)
    return {"schema": "a8_native_request_metrics_v1",
            "kind": "native_request_timing_no_tensor_capture", "complete": True,
            "status": "complete", "metrics_status": "complete_request_metrics",
            "repetitions": 5, "warmup_requests_per_server": 1, "orders": orders,
            "prompt_ids": prompts, "records": rows, "variants": variants,
            "speedup_vs_q4_0": ratio["Q4_0"],
            "speedup_vs_no_speculation": ratio["target_only"]}


def self_test():
    report = fixture()
    result = audit(report, "synthetic", "synthetic")
    actual = result["repetitions"][0]["variants"]["A8"]["request_tokens_per_s"]
    mean_rates = statistics.mean(r["completion_tokens"] / r["request_wall_s"]
                                for r in report["records"]
                                if r["variant"] == "A8" and r["repetition"] == 0)
    require(actual == 2 and not math.isclose(actual, mean_rates), "ratio-of-sums test failed")
    require(result["repetitions"][0]["a8_ratios"]["Q4_0"][RATES[0]] == 2,
            "matched ratio test failed")
    bad_cases = {}
    bad_cases["missing_record"] = lambda r: r["records"].pop()
    bad_cases["extra_warmup_duplicate"] = lambda r: r["records"].append(copy.deepcopy(r["records"][0]))
    bad_cases["same_count_duplicate"] = lambda r: r["records"].__setitem__(1, copy.deepcopy(r["records"][0]))
    bad_cases["wrong_token_count"] = lambda r: r["records"][0].__setitem__("completion_tokens", 9)
    bad_cases["token_parity"] = lambda r: r["records"][24]["generated_token_ids"].__setitem__(0, 100)
    bad_cases["nan_wall"] = lambda r: r["records"][0].__setitem__("request_wall_s", float("nan"))
    bad_cases["zero_decode"] = lambda r: r["records"][0].__setitem__("server_predicted_ms", 0)
    bad_cases["bool_count"] = lambda r: r["records"][0].__setitem__("completion_tokens", True)
    bad_cases["incomplete"] = lambda r: r.__setitem__("complete", False)
    bad_cases["incorrect_reported_rate"] = lambda r: r["variants"]["A8"].__setitem__(RATES[0], 99)
    bad_cases["warmup_flag"] = lambda r: r["records"][0].__setitem__("warmup", True)
    for name, mutate in bad_cases.items():
        invalid = copy.deepcopy(report)
        mutate(invalid)
        try:
            audit(invalid, "synthetic", "synthetic")
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid fixture: {name}")
    no_orders = copy.deepcopy(report)
    no_orders.pop("orders")
    require(audit(no_orders, "synthetic", "synthetic")["repetitions"][0]["variant_order"] is None,
            "missing optional orders test failed")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "request-timing.json"
        raw = (json.dumps(report, indent=2) + "\n").encode()
        path.write_bytes(raw)
        loaded = read_manifest(path)
        require(path.read_bytes() == raw and loaded["source"]["sha256"] == hashlib.sha256(raw).hexdigest(),
                "source byte preservation/SHA test failed")
    return {"passed": True, "rejection_cases": list(bad_cases),
            "ratio_of_sums_tokens_per_s": actual, "mean_of_request_rates_tokens_per_s": mean_rates,
            "optional_order_and_source_bytes_checks": True}


def read_manifest(path):
    raw = path.read_bytes()
    return audit(json.loads(raw), str(path.resolve()), hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path,
                        help="existing complete request-timing.json manifests (one or more)")
    parser.add_argument("--output", "-o", type=Path, help="new compact JSON file; default stdout")
    parser.add_argument("--self-test", action="store_true", help="run CPU-only synthetic checks")
    args = parser.parse_args()
    if args.self_test:
        require(not args.manifests and args.output is None, "self-test accepts no manifest/output paths")
        print(json.dumps(self_test(), separators=(",", ":"), allow_nan=False))
        return
    if not args.manifests:
        parser.error("provide at least one existing manifest, or --self-test")
    try:
        require(len({p.resolve() for p in args.manifests}) == len(args.manifests), "duplicate input paths")
        if args.output is not None:
            require(args.output.resolve() not in {p.resolve() for p in args.manifests},
                    "output must not overwrite any input manifest")
            require(not args.output.exists(), "output already exists; choose a new output path")
        audited = []
        for path in args.manifests:
            try:
                audited.append(read_manifest(path))
            except (ValueError, TypeError, KeyError, OSError) as error:
                raise ValueError(f"{path}: {error}") from error
        result = {"schema": "a8_existing_timing_repeat_aggregate_v1",
                  "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "method": {"request_rate": "sum(completion_tokens)/sum(full HTTP request_wall_s)",
                             "decode_rate": "1000*sum(completion_tokens)/sum(server_predicted_ms)",
                             "ratios": "matched same-repetition A8 rate/baseline rate",
                             "repeat_distributions": "descriptive five-repeat min/median/max/range",
                             "uncertainty": "no confidence interval or significance claim"},
                  "manifests": audited}
        encoded = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            # Exclusive creation also protects hardlink aliases to existing source files.
            with args.output.open("x") as stream:
                stream.write(encoded)
    except (ValueError, OSError) as error:
        parser.exit(2, f"audit rejected: {error}\n")


if __name__ == "__main__":
    main()
