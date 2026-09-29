#!/usr/bin/env python3
"""Convert an audited v1 train bundle to independently auditable hard-CE v2.

Requires the original native capture metadata and per-request JSON. Copies no
raw full-vocabulary logits, never modifies or deletes inputs, and never grants
training eligibility. Full-tier capture/storage approval remains separate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from audit_recurrent_binary_capture import audit_capture, sha256
from audit_recurrent_capture_v2 import POLICY, PREPARED_FIELDS, SCHEMA, audit_capture_v2


def convert_capture_v2(
    source_manifest: Path,
    capture_root: Path,
    output: Path,
    *,
    expected_prompt_sha256: str,
    expected_prompt_count: int,
) -> dict:
    source_manifest, capture_root, output = map(Path, (source_manifest, capture_root, output))
    if output.exists():
        raise ValueError("v2 output must be new; inputs are immutable")
    original = json.loads(source_manifest.read_text())
    source = source_manifest.parent
    prompts = source / "train_prompts.jsonl"
    original_audit = audit_capture(
        source_manifest,
        prompts,
        expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    )
    if "target_logits" not in original or original_audit["raw_target_logit_rows"] < 1:
        raise ValueError("conversion requires an audited retained v1 raw-logit source")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        m = {
            "schema": SCHEMA,
            "split": "train",
            "prompts_sha256": expected_prompt_sha256,
            "prompt_count": expected_prompt_count,
            "storage_policy": dict(POLICY),
            "training_eligible": False,
            "readiness": "preparation_only",
            "raw_retirement_allowed": False,
            "unverified_gates": original.get("unverified_gates", []),
            **{k: original[k] for k in ("target_vocab_size", "draft_vocab_size", "max_depth")},
            "omitted_raw_logits": {
                "sha256": original["target_logits"]["sha256"],
                "rows": original_audit["raw_target_logit_rows"],
                "bytes": (source / original["target_logits"]["path"]).stat().st_size,
                "disposition": "not_copied_source_v1_preserved",
            },
        }

        def copy(field: str, path: Path, name: str) -> dict:
            if Path(name).name != name or (stage / name).exists():
                raise ValueError("v2 destination basename is invalid or collides")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing regular input file: {path}")
            shutil.copyfile(path, stage / name)
            record = {"path": name, "sha256": sha256(stage / name)}
            m[field] = record
            return record

        for field in PREPARED_FIELDS:
            copy(field, source / original[field]["path"], f"prepared_{original[field]['path']}")
        copy("prompts", prompts, prompts.name)
        copy("source_v1_manifest", source_manifest, "source_v1_manifest.json")
        audit_path = stage / "source_v1_audit.json"
        audit_path.write_text(json.dumps(original_audit, indent=2, sort_keys=True) + "\n")
        m["source_v1_audit"] = {"path": audit_path.name, "sha256": sha256(audit_path)}
        copy("source_rows_report", source / "rows_preparation.json", "source_rows_preparation.json")
        copy(
            "source_features_report",
            source / "features_preparation.json",
            "source_features_preparation.json",
        )
        cell = capture_root / "d_d"
        copy("source_cell", cell / "manifest.json", "source_cell_manifest.json")
        copy(
            "source_capture", capture_root / "capture-manifest.json", "source_capture_manifest.json"
        )
        for field, path in (
            ("native_heads", cell / "heads.jsonl"),
            ("native_rounds", cell / "forced-rounds.jsonl"),
            ("native_round_trace", cell / "rounds.jsonl"),
            ("native_feature_events", cell / "heads.target_features.jsonl"),
            ("task_map", capture_root / "task_prompt_ids.json"),
        ):
            copy(field, path, path.name)
        offsets = np.load(stage / m["offsets"]["path"], allow_pickle=False)
        absolute = stage / "absolute_d2t.npy"
        np.save(absolute, np.asarray([int(v) + i for i, v in enumerate(offsets)], dtype=np.int64))
        m["absolute_d2t"] = {"path": absolute.name, "sha256": sha256(absolute)}
        if "shard_manifest" in original:
            copy(
                "shard_manifest",
                source / original["shard_manifest"]["path"],
                "source_shard_manifest.json",
            )
        native_cell = json.loads((cell / "manifest.json").read_text())
        m["requests"] = []
        for i, item in enumerate(native_cell["requests"]):
            record = {"task_id": item["task_id"], "id": item["id"]}
            for key, filename in (
                ("prompt", "prompt.json"),
                ("request", "request.json"),
                ("response", "response.json"),
            ):
                temp_field = f"request_{i:05d}_{key}"
                record[key] = copy(
                    temp_field, cell / f"request-{i:03d}" / filename, f"request-{i:05d}-{filename}"
                )
                del m[temp_field]
            m["requests"].append(record)
        manifest = stage / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        audit = audit_capture_v2(
            manifest,
            expected_prompt_sha256=expected_prompt_sha256,
            expected_prompt_count=expected_prompt_count,
        )
        (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.rename(stage, output)
        return {
            "manifest": str(output / "manifest.json"),
            "audit": audit,
            "omitted_raw_bytes": m["omitted_raw_logits"]["bytes"],
        }
    except BaseException:
        shutil.rmtree(stage)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--expected-prompt-sha256", required=True)
    p.add_argument("--expected-prompt-count", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = convert_capture_v2(
        a.source_manifest,
        a.capture_root,
        a.output,
        expected_prompt_sha256=a.expected_prompt_sha256,
        expected_prompt_count=a.expected_prompt_count,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "audit"}))


if __name__ == "__main__":
    main()
