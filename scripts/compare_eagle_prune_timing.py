#!/usr/bin/env python3
"""Compare paired EAGLE pruning timing runs with Q4_0 kept primary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analyze_binary_rescue_benchmark import aggregate, key, paired_comparison


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, prune: str) -> tuple[dict, dict, dict[str, list[dict]]]:
    manifest = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    if (
        manifest.get("schema") != "binary_rescue_benchmark_v1"
        or manifest.get("mode") != "timed"
        or manifest.get("status") != "complete"
        or config.get("graph_env", {}).get("GGML_EAGLE_PRUNE_UNUSED_HEAD") != prune
    ):
        raise ValueError(f"{path}: incomplete or wrong prune-mode timed run")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in manifest["records"]:
        if not row["warmup"]:
            grouped[row["variant"]].append(row)
    if not grouped:
        raise ValueError(f"{path}: no measured requests")
    return manifest, config, grouped


def compare(off_path: Path, on_path: Path) -> dict:
    off, off_config, off_groups = load(off_path, "0")
    on, on_config, on_groups = load(on_path, "1")
    off_config["graph_env"].pop("GGML_EAGLE_PRUNE_UNUSED_HEAD")
    on_config["graph_env"].pop("GGML_EAGLE_PRUNE_UNUSED_HEAD")
    if off_config != on_config:
        raise ValueError("paired configurations differ beyond the prune switch")
    for field in ("workload", "policy", "q4_variant", "prompt_sha256", "hashes"):
        if off[field] != on[field]:
            raise ValueError(f"paired run {field} differs")
    if set(off_groups) != set(on_groups) or off["q4_variant"] not in off_groups:
        raise ValueError("paired variant inventory differs or lacks Q4_0")

    variants = {}
    mismatches = []
    for name in [off["q4_variant"], *(v for v in off_groups if v != off["q4_variant"])]:
        off_rows, on_rows = off_groups[name], on_groups[name]
        off_by_key, on_by_key = ({key(row): row for row in rows} for rows in (off_rows, on_rows))
        if (
            len(off_by_key) != len(off_rows)
            or len(on_by_key) != len(on_rows)
            or set(off_by_key) != set(on_by_key)
        ):
            raise ValueError(f"{name}: measured prompt/repetition pairing differs")
        for request_key in sorted(off_by_key):
            left, right = off_by_key[request_key], on_by_key[request_key]
            for field in (
                "generated_token_ids",
                "completion_tokens",
                "finish_reason",
                "speculative",
            ):
                if left.get(field) != right.get(field):
                    mismatches.append({"variant": name, "request": request_key, "field": field})
        paired = paired_comparison(on_rows, off_rows)
        off_total, on_total = aggregate(off_rows), aggregate(on_rows)
        variants[name] = {
            "paired_requests": paired["paired_requests"],
            "paired_prompts": paired["paired_prompts"],
            "same_raw_ids": paired["same_raw_ids"],
            "off": {
                "client_request_tok_s": off_total["client_request_tok_s"],
                "server_decode_tok_s": off_total["server_decode_tok_s"],
                "speculative": off_total["speculative"],
            },
            "on": {
                "client_request_tok_s": on_total["client_request_tok_s"],
                "server_decode_tok_s": on_total["server_decode_tok_s"],
                "speculative": on_total["speculative"],
            },
            "on_over_off_client_ratio": paired["client_request_tok_s_ratio"],
            "on_over_off_client_ci95": paired["client_request_tok_s_ratio_ci95"],
            "on_over_off_decode_ratio": paired["server_decode_tok_s_ratio"],
            "on_over_off_decode_ci95": paired["server_decode_tok_s_ratio_ci95"],
        }

    return {
        "schema": "eagle_prune_timing_comparison_v1",
        "off_manifest_sha256": sha256(off_path),
        "on_manifest_sha256": sha256(on_path),
        "workload": off["workload"],
        "q4_variant": off["q4_variant"],
        "variants": variants,
        "behavior_mismatch_count": len(mismatches),
        "first_behavior_mismatches": mismatches[:20],
        "uncertainty": "paired prompt-cluster bootstrap over 24 reused development prompts",
        "scope": (
            "concurrency-one client request and server decode; "
            "not process() attribution or serving capacity"
        ),
        "external_load": "assess telemetry and idle GPU separately before claiming speed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("comparison output must be new")
    result = compare(args.off, args.on)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "variants": list(result["variants"]),
                "behavior_mismatch_count": result["behavior_mismatch_count"],
            }
        )
    )
    if result["behavior_mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
