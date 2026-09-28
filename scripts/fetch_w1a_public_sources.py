#!/usr/bin/env python3
"""Fetch pinned public source files and prepare CPU-only W1A prompt inputs.

Raw and transformed prompts remain under ignored data/. This script never
downloads or runs a model. Licenses and transformations are recorded in the
catalog, which can be passed to prepare_w1a_data.py for a frozen split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

from prepare_w1a_data import prepare, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "dolly": {
        "uri": ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
                "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"),
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "CC-BY-SA-3.0",
        "upstream_sha256": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        "domain": "prose",
    },
    "gsm8k": {
        "uri": ("https://raw.githubusercontent.com/openai/grade-school-math/"
                "3101c7d5072418e28b9008a6636bde82a006892c/"
                "grade_school_math/data/train.jsonl"),
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "license": "MIT",
        "upstream_sha256": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "domain": "reasoning",
    },
    "mbpp": {
        "uri": ("https://raw.githubusercontent.com/google-research/google-research/"
                "d36068b845da4c2b24927fee2cea1e6ef98dadda/mbpp/mbpp.jsonl"),
        "revision": "d36068b845da4c2b24927fee2cea1e6ef98dadda",
        "license": "CC-BY-4.0",
        "upstream_sha256": "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f",
        "domain": "code",
    },
}
DOLLY_CATEGORIES = {
    "brainstorming", "creative_writing", "general_qa", "open_qa", "summarization"
}


def download(name: str, raw_dir: Path, *, offline: bool) -> Path:
    spec = SOURCES[name]
    path = raw_dir / f"{name}.jsonl"
    if path.exists():
        if sha256(path) != spec["upstream_sha256"]:
            raise ValueError(f"{name}: existing raw file has wrong SHA256")
        return path
    if offline:
        raise ValueError(f"{name}: pinned raw file missing in offline mode")
    raw_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-", dir=raw_dir)
    os.close(descriptor)
    try:
        request = urllib.request.Request(
            spec["uri"], headers={"User-Agent": "binary-eagle-data-prep/1"}
        )
        with urllib.request.urlopen(request, timeout=90) as source, open(temporary, "wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if sha256(Path(temporary)) != spec["upstream_sha256"]:
            raise ValueError(f"{name}: downloaded raw file SHA256 mismatch")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def transformed_rows(name: str, raw: Path):
    with raw.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            if name == "dolly":
                if source["category"] not in DOLLY_CATEGORIES:
                    continue
                instruction = source["instruction"].strip()
                context = source["context"].strip()
                content = instruction + (f"\n\nContext:\n{context}" if context else "")
                topic = (
                    hashlib.sha256(context.casefold().encode()).hexdigest()[:16]
                    if context else None
                )
                row = {"id": f"line-{line_number:06d}", "messages": [
                    {"role": "user", "content": content}],
                    "group_id": f"line-{line_number:06d}", "category": source["category"]}
                if topic:
                    row["topic_id"] = f"context-{topic}"
                yield row
            elif name == "gsm8k":
                content = source["question"].strip()
                yield {"id": f"train-{line_number:06d}", "messages": [
                    {"role": "user", "content": content}],
                    "group_id": f"train-{line_number:06d}"}
            elif name == "mbpp":
                task = source["task_id"]
                tests = "\n".join(source["test_list"])
                content = ("Write a Python function for this task.\n\n"
                           + source["text"].strip()
                           + "\n\nYour code should pass these tests:\n" + tests)
                yield {"id": f"task-{task}", "messages": [
                    {"role": "user", "content": content}],
                    "group_id": f"task-{task}"}


def build_catalog(directory: Path, *, offline: bool) -> dict:
    raw_dir, normalized_dir = directory / "raw", directory / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = directory / "catalog.json"
    if catalog_path.exists() or any(
        (normalized_dir / f"{name}.jsonl").exists() for name in SOURCES
    ):
        raise ValueError("catalog or normalized source exists; use a new directory")
    raw_paths = {name: download(name, raw_dir, offline=offline) for name in SOURCES}
    entries, counts = [], Counter()
    transform_hash = sha256(Path(__file__))
    for name, spec in SOURCES.items():
        normalized = normalized_dir / f"{name}.jsonl"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-", dir=normalized_dir)
        os.close(descriptor)
        try:
            with open(temporary, "wb") as target:
                for row in transformed_rows(name, raw_paths[name]):
                    target.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode() + b"\n")
                    counts[name] += 1
            os.replace(temporary, normalized)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        entries.append({"id": name, "uri": spec["uri"], "revision": spec["revision"],
                        "license": spec["license"], "upstream_sha256": spec["upstream_sha256"],
                        "transform": "fetch_w1a_public_sources_v1",
                        "transform_sha256": transform_hash,
                        "domain": spec["domain"], "path": f"normalized/{name}.jsonl",
                        "sha256": sha256(normalized)})
    catalog = {"schema": "w1a_source_catalog_v1", "sources": entries}
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return {"catalog": catalog_path, "counts": dict(counts),
            "catalog_sha256": sha256(catalog_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=ROOT / "data/w1a-public-sources/pinned-v1")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--freeze-output", type=Path,
                        help="create a new 2k training + 192/192 dev/final split directory")
    parser.add_argument("--seed", type=int, default=1429)
    args = parser.parse_args()
    report = build_catalog(args.directory, offline=args.offline)
    if args.freeze_output:
        manifest = prepare(report["catalog"], args.freeze_output, 2000, 2000,
                           192, 192, args.seed, 64, 24000, 0.88)
        report["freeze_manifest"] = str(args.freeze_output / "manifest.json")
        report["freeze_manifest_sha256"] = sha256(args.freeze_output / "manifest.json")
        report["split_counts"] = {
            name: value["prompts_count"] for name, value in manifest["files"].items()
        }
    print(json.dumps({key: str(value) if isinstance(value, Path) else value
                      for key, value in report.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
