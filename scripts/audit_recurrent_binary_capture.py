#!/usr/bin/env python3
"""CPU-only metadata gate for a future native recurrent training capture.

This validates prompt ownership, file hashes, proposal ancestry, verifier
label provenance, vocabulary offsets and unsupported-label masks. It does not
create a native capture or certify target features and draft K/V cache parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.recurrent_trace import RoundAnchor, validate_recurrent_trace  # noqa: E402

TRAIN_PROMPTS_SHA256 = "80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74"
TRAIN_PROMPTS = 96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: expected nonempty JSON object rows")
    return rows


def _owned_file(directory: Path, value: str) -> Path:
    path = directory / value
    if Path(value).name != value or not path.is_file():
        raise ValueError("capture paths must be existing basenames beside the manifest")
    return path


def audit_capture(manifest_path: Path, prompts_path: Path, expected_prompt_hash: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "recurrent_binary_capture_v1":
        raise ValueError("unsupported recurrent capture manifest")
    if manifest.get("split") != "train" or sha256(prompts_path) != expected_prompt_hash:
        raise ValueError("capture requires the frozen training prompt split")
    if manifest.get("prompts_sha256") != expected_prompt_hash:
        raise ValueError("manifest prompt hash differs from frozen training prompts")
    prompts = read_jsonl(prompts_path)
    prompt_ids = [row.get("id") for row in prompts]
    if (
        len(prompt_ids) != TRAIN_PROMPTS
        or any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids)
        or len(set(prompt_ids)) != TRAIN_PROMPTS
    ):
        raise ValueError("expected 96 unique frozen training prompt IDs")
    files = {}
    for field in ("rows", "anchors", "offsets", "t2d"):
        record = manifest.get(field)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"missing {field} file record")
        path = _owned_file(manifest_path.parent, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"{field} SHA256 mismatch")
        files[field] = path
    rows = read_jsonl(files["rows"])
    anchor_rows = read_jsonl(files["anchors"])
    anchors = [RoundAnchor(**row) for row in anchor_rows]
    offsets = np.load(files["offsets"], allow_pickle=False)
    t2d = np.load(files["t2d"], allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu":
        raise ValueError("native d2t offsets must be a one-dimensional integer array")
    trace = validate_recurrent_trace(
        rows,
        anchors,
        offsets=offsets,
        target_vocab_size=manifest["target_vocab_size"],
        draft_vocab_size=manifest["draft_vocab_size"],
        max_depth=manifest["max_depth"],
        allowed_prompt_ids=set(prompt_ids),
        split="train",
    )
    if t2d.dtype != np.bool_ or t2d.shape != (manifest["target_vocab_size"],):
        raise ValueError("t2d must be a target-vocabulary boolean mask")
    if not np.array_equal(t2d, np.asarray(trace.target_to_draft) >= 0):
        raise ValueError("t2d mask disagrees with offset-form d2t inverse")
    observed_mass = [mass for mass in trace.mapped_probability_mass if mass is not None]
    return {
        "schema": "recurrent_binary_capture_audit_v1",
        "execution_device": "cpu",
        "capture_manifest_sha256": sha256(manifest_path),
        "training_prompts_sha256": expected_prompt_hash,
        "source_sha256": {field: sha256(path) for field, path in files.items()},
        "counts": trace.counts,
        "per_depth": trace.per_depth,
        "mapped_probability_mass_mean_on_sampled_logit_rows": (
            sum(observed_mass) / len(observed_mass) if observed_mass else None
        ),
        "real_model_feature_and_kv_parity": "unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be a new path")
    report = audit_capture(args.manifest, args.prompts, TRAIN_PROMPTS_SHA256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
