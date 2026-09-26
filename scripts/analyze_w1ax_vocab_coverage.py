#!/usr/bin/env python3
"""Audit emitted target-token IDs against the EAGLE draft vocabulary.

Run in the GGUF conversion environment (numpy and gguf-py required). This
post-run audit reads records.json and prompts.jsonl without changing raw data.
It cannot estimate target probability mass or speculative acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_draft_ids(
    values: np.ndarray, target_vocab_size: int, expected_size: int = 32_000
) -> np.ndarray:
    """Validate GGUF's absolute d2t IDs, not checkpoint offset-form d2t."""
    if target_vocab_size < 1 or expected_size < 1:
        raise ValueError("vocabulary sizes must be positive")
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("int64") or values.ndim != 1:
        raise ValueError("GGUF d2t must be a one-dimensional I64 array")
    if values.size != expected_size:
        raise ValueError(f"GGUF d2t has {values.size} IDs; expected {expected_size}")
    if np.any(values < 0) or np.any(values >= target_vocab_size):
        raise ValueError("GGUF d2t contains an out-of-range target ID")
    if np.unique(values).size != values.size:
        raise ValueError("GGUF d2t contains duplicate target IDs")
    return values


def read_draft_ids(path: Path, target_vocab_size: int, expected_size: int) -> np.ndarray:
    from gguf import GGMLQuantizationType, GGUFReader

    matches = [tensor for tensor in GGUFReader(path).tensors if tensor.name == "d2t"]
    if len(matches) != 1 or matches[0].tensor_type != GGMLQuantizationType.I64:
        raise ValueError(f"{path}: expected exactly one I64 d2t tensor")
    return absolute_draft_ids(matches[0].data, target_vocab_size, expected_size).copy()


def prompt_categories(path: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a prompt object")
        prompt_id, category = row.get("id"), row.get("category")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in categories:
            raise ValueError(f"{path}:{line_number}: missing or duplicate prompt ID")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{path}:{line_number}: missing category")
        categories[prompt_id] = category
    if not categories:
        raise ValueError(f"{path}: empty prompt manifest")
    return categories


def token_coverage(ids: list[int], membership: np.ndarray) -> dict[str, Any]:
    if not isinstance(ids, list) or any(
        not isinstance(token, int)
        or isinstance(token, bool)
        or token < 0
        or token >= membership.size
        for token in ids
    ):
        raise ValueError("generated_token_ids must be a list of in-range integer target IDs")
    array = np.asarray(ids, dtype=np.int64)
    covered = int(np.count_nonzero(membership[array]))
    unique = np.unique(array)
    covered_unique = int(np.count_nonzero(membership[unique]))
    return {
        "emitted_ids": len(ids),
        "in_draft_vocab": covered,
        "outside_draft_vocab": len(ids) - covered,
        "fraction_in_draft_vocab": covered / len(ids) if ids else None,
        "distinct_emitted_ids": int(unique.size),
        "distinct_in_draft_vocab": covered_unique,
        "distinct_outside_draft_vocab": int(unique.size) - covered_unique,
    }


def summarize(rows: list[dict[str, Any]], membership: np.ndarray) -> dict[str, Any]:
    all_ids = [token for row in rows for token in row["generated_token_ids"]]
    return {"requests": len(rows), **token_coverage(all_ids, membership)}


def analyze_run(
    run_dir: Path, membership: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.parent if run_dir.name == "records.json" else run_dir
    records_path = run_dir / "records.json"
    prompts_path = run_dir / "prompts.jsonl"
    categories = prompt_categories(prompts_path)
    prompts_hash = sha256(prompts_path)
    rows = json.loads(records_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{records_path}: expected a nonempty array")
    seen: set[tuple[str, int, str]] = set()
    tagged = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{records_path}: row {index} is not an object")
        variant, repetition, prompt_id = (
            row.get(key) for key in ("variant", "repetition", "prompt_id")
        )
        if (
            not isinstance(variant, str)
            or not variant
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition < 0
        ):
            raise ValueError(f"{records_path}: row {index} has invalid variant or repetition")
        if not isinstance(prompt_id, str) or prompt_id not in categories:
            raise ValueError(f"{records_path}: row {index} has unknown prompt ID")
        key = (variant, repetition, prompt_id)
        if key in seen:
            raise ValueError(f"{records_path}: duplicate variant/repetition/prompt record {key}")
        seen.add(key)
        token_coverage(row.get("generated_token_ids"), membership)
        tagged.append(
            {**row, "category": categories[prompt_id], "prompt_key": f"{prompts_hash}:{prompt_id}"}
        )
    if not any(row["variant"] == "target_only" for row in tagged):
        raise ValueError(f"{records_path}: no target_only records found")
    result = {
        "run_dir": str(run_dir.resolve()),
        "source_sha256": {
            "records.json": sha256(records_path),
            "prompts.jsonl": prompts_hash,
            **(
                {"manifest.json": sha256(run_dir / "manifest.json")}
                if (run_dir / "manifest.json").is_file()
                else {}
            ),
        },
        "prompt_manifest": {"prompt_count": len(categories), "categories": categories},
        "record_count": len(rows),
    }
    return result, tagged


def analyze(
    gguf_path: Path,
    run_dirs: list[Path],
    target_vocab_size: int = 151_936,
    expected_draft_size: int = 32_000,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one benchmark run is required")
    draft_ids = read_draft_ids(gguf_path, target_vocab_size, expected_draft_size)
    membership = np.zeros(target_vocab_size, dtype=np.bool_)
    membership[draft_ids] = True
    inputs = []
    rows = []
    seen_runs: set[Path] = set()
    for run_dir in run_dirs:
        canonical_dir = (run_dir.parent if run_dir.name == "records.json" else run_dir).resolve()
        if canonical_dir in seen_runs:
            raise ValueError(f"duplicate run directory: {canonical_dir}")
        seen_runs.add(canonical_dir)
        entry, tagged = analyze_run(run_dir, membership)
        inputs.append(entry)
        rows.extend(tagged)
    target_rows = [row for row in rows if row["variant"] == "target_only"]
    if not target_rows:
        raise ValueError("no target_only records found")
    variants = sorted({row["variant"] for row in rows})
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_prompt[row["prompt_key"]].append(row)
        by_category[row["category"]].append(row)
    variant_summary = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        variant_summary[variant] = {
            "overall": summarize(selected, membership),
            "by_prompt": {
                key: summarize(group, membership)
                for key, group in sorted(_group(selected, "prompt_key").items())
            },
            "by_category": {
                key: summarize(group, membership)
                for key, group in sorted(_group(selected, "category").items())
            },
        }
    return {
        "schema": "w1ax_emitted_id_vocab_coverage_v1",
        "gguf": {
            "path": str(gguf_path.resolve()),
            "sha256": sha256(gguf_path),
            "d2t_tensor": "d2t",
            "d2t_encoding": "absolute_target_ids",
            "draft_vocab_size": len(draft_ids),
        },
        "target_vocab_size": target_vocab_size,
        "inputs": inputs,
        "target_only": {
            "overall": summarize(target_rows, membership),
            "by_prompt": {
                key: summarize(group, membership) for key, group in sorted(by_prompt.items())
            },
            "by_category": {
                key: summarize(group, membership) for key, group in sorted(by_category.items())
            },
        },
        "variants": variant_summary,
        "interpretation": (
            "Coverage counts membership of emitted target token IDs in the draft vocabulary. "
            "Repetitions are repeated generations, not independent prompts. "
            "Target probability mass is unavailable from generated IDs. "
            "Coverage does not measure speculative acceptance; "
            "use native acceptance counters for that."
        ),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gguf", type=Path, help="all-nine W1A1 EAGLE GGUF containing I64 absolute d2t"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="native benchmark directories or records.json paths (with sibling prompts.jsonl)",
    )
    parser.add_argument("--target-vocab-size", type=int, default=151_936)
    parser.add_argument("--expected-draft-size", type=int, default=32_000)
    parser.add_argument("--output", type=Path, help="write the audit JSON here (default: stdout)")
    args = parser.parse_args()
    report = analyze(args.gguf, args.run_dirs, args.target_vocab_size, args.expected_draft_size)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
