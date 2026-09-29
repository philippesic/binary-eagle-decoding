#!/usr/bin/env python3
"""Compare paired EAGLE quality runs without using their timing spans."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROUND_FIELDS = (
    "n_proposed",
    "n_accepted",
    "n_emitted",
    "proposed_token_ids",
    "emitted_token_ids",
    "status",
    "replay",
    "stopped_low_confidence",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, expected_prune: str) -> tuple[dict, dict, dict]:
    manifest = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    if (
        manifest.get("schema") != "binary_rescue_benchmark_v1"
        or manifest.get("mode") != "quality"
        or manifest.get("status") != "complete"
        or config.get("graph_env", {}).get("GGML_EAGLE_PRUNE_UNUSED_HEAD") != expected_prune
    ):
        raise ValueError(f"{path}: incomplete or wrong prune-mode quality run")
    records = {}
    for row in manifest["records"]:
        if row["warmup"]:
            continue
        key = row["variant"], row["repetition"], row["prompt_id"]
        if key in records:
            raise ValueError(f"{path}: duplicate measured request {key}")
        records[key] = row
    if not records:
        raise ValueError(f"{path}: no measured requests")
    return manifest, config, records


def rounds(row: dict) -> list[dict]:
    path = Path(row["directory"]) / "rounds.json"
    if not path.is_file():
        raise ValueError(f"missing per-request round trace: {path}")
    return json.loads(path.read_text())


def compare(off_path: Path, on_path: Path) -> dict:
    off, off_config, off_rows = load(off_path, "0")
    on, on_config, on_rows = load(on_path, "1")
    off_config["graph_env"].pop("GGML_EAGLE_PRUNE_UNUSED_HEAD")
    on_config["graph_env"].pop("GGML_EAGLE_PRUNE_UNUSED_HEAD")
    if off_config != on_config:
        raise ValueError("paired configurations differ beyond the prune switch")
    for field in ("workload", "policy", "q4_variant", "prompt_sha256", "hashes"):
        if off[field] != on[field]:
            raise ValueError(f"paired run {field} differs")
    if set(off_rows) != set(on_rows):
        raise ValueError("paired measured request keys differ")

    counts: dict[str, Counter] = defaultdict(Counter)
    mismatches = []
    for key in sorted(off_rows):
        left, right = off_rows[key], on_rows[key]
        for field in (
            "generated_token_ids",
            "completion_sha256",
            "completion_tokens",
            "finish_reason",
            "quality",
            "speculative",
        ):
            if left.get(field) != right.get(field):
                mismatches.append({"request": key, "field": field})
        left_rounds, right_rounds = rounds(left), rounds(right)
        if len(left_rounds) != len(right_rounds):
            mismatches.append({"request": key, "field": "round_count"})
        else:
            for index, (a, b) in enumerate(zip(left_rounds, right_rounds, strict=True)):
                for field in ROUND_FIELDS:
                    if a.get(field) != b.get(field):
                        mismatches.append({"request": key, "round": index, "field": field})
        variant = key[0]
        counts[variant]["requests"] += 1
        counts[variant]["output_tokens"] += left["completion_tokens"]
        if "quality" in left:
            for field in ("rounds", "proposed", "accepted", "emitted"):
                counts[variant][field] += left["quality"][field]
    result = {
        "schema": "eagle_prune_quality_comparison_v1",
        "off_manifest_sha256": sha256(off_path),
        "on_manifest_sha256": sha256(on_path),
        "matched_requests": len(off_rows),
        "per_variant": {key: dict(value) for key, value in sorted(counts.items())},
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
        "scope": "token, acceptance and round semantics; no speed or cache-byte claim",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("comparison output must be new")
    report = compare(args.off, args.on)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "matched_requests": report["matched_requests"],
                "mismatch_count": report["mismatch_count"],
            }
        )
    )
    if report["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
