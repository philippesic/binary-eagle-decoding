#!/usr/bin/env python3
"""Bind ordered eligible capture-provider manifests to a frozen shard plan.

The JSONL input has one `{ordinal, provider_manifest}` object per planned
shard. This tool checks plan and child identity, writes a new execution
manifest, and never loads model weights or raw verifier logits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1ax_capture_provider import sha256  # noqa: E402
from w1ax_multishard_provider import SCHEMA, _plan  # noqa: E402


def prepare(plan_path: Path, parent_manifest: Path, children_jsonl: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("multi-shard execution manifest must be new")
    plan_path, parent_manifest = Path(plan_path).resolve(), Path(parent_manifest).resolve()
    plan_hash = sha256(plan_path)
    _, records = _plan(plan_path, parent_manifest, plan_hash)
    with Path(children_jsonl).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(records):
        raise ValueError("one child provider manifest is required per planned shard")
    attachments = []
    for ordinal, (row, record) in enumerate(zip(rows, records)):
        if not isinstance(row, dict) or set(row) != {"ordinal", "provider_manifest"}:
            raise ValueError("children JSONL requires ordinal and provider_manifest")
        if row["ordinal"] != ordinal:
            raise ValueError("children JSONL ordinals must match frozen plan order")
        path = Path(row["provider_manifest"]).resolve()
        child = json.loads(path.read_text())
        if (
            child.get("schema") != "w1ax_native_train_provider_v1"
            or child.get("training_eligible") is not True
            or child.get("prompt_count") != record["prompt_count"]
            or child.get("sha256", {}).get("prompts") != record["prompts_sha256"]
        ):
            raise ValueError("child provider is ineligible or differs from planned prompts")
        attachments.append(
            {
                "ordinal": ordinal,
                "provider_manifest": str(path),
                "provider_manifest_sha256": sha256(path),
            }
        )
    result = {
        "schema": SCHEMA,
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "parent_manifest": str(parent_manifest),
        "parent_manifest_sha256": sha256(parent_manifest),
        "shards": attachments,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--children-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.plan, args.parent_manifest, args.children_jsonl, args.output)
    print(json.dumps({"output": str(args.output), "shards": len(result["shards"])}))


if __name__ == "__main__":
    main()
