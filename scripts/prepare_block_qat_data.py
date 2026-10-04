#!/usr/bin/env python3
"""Authenticate a native TRAIN block manifest and publish bounded preparation evidence.

Input arrays and native producer receipt must already exist. This command does
not make EAGLE's three taps/full-vocabulary omissions valid block data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from w1a1_eagle.block_data import BlockDataset, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-soft-teachers", action="store_true")
    parser.add_argument("--max-teacher-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite preparation history")
    dataset = BlockDataset(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        max_teacher_bytes=args.max_teacher_bytes,
    )
    # Only one block per chain is opened; constructor checks every horizon and
    # finite array in bounded chunks. No eager corpus teacher tensor is created.
    for cid in dataset.chains:
        dataset.load_block(cid, 0, require_teacher=args.require_soft_teachers)
    report = {
        "schema": "block_data_preparation_v1",
        "manifest_sha256": dataset.sha256,
        "producer_receipt_sha256": dataset.manifest["producer"]["receipt"]["sha256"],
        "source_sha256": file_sha256(ROOT / "src/w1a1_eagle/block_data.py"),
        "family": dataset.manifest["family"],
        "layout": dataset.manifest["layout"],
        "storage": dataset.storage_summary(),
        "profiles": {
            "hard_ce_target_prefix": "PASS",
            "exact_soft_teacher_target_prefix": "PASS"
            if all(logits is not None for _, _, logits in dataset._arrays.values())
            else "PENDING: missing full-vocabulary logits",
            "student_prefix_after_divergence": "EXCLUDED: requires fresh exact-prefix recapture",
        },
        "evidence_hardware": "CPU schema/integrity checks; producer hardware separately recorded",
        "fresh_sm120_gate": "PENDING: target-device teacher portability/trajectory check",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
