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

from w1a1_eagle.block_data import BlockDataset, file_sha256, import_capture_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--capture-plan", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--capture-plan-sha256")
    parser.add_argument("--capture-output-dir", type=Path)
    parser.add_argument("--max-capture-bytes", type=int, default=1024**3)
    parser.add_argument("--admission-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-soft-teachers", action="store_true")
    parser.add_argument("--max-teacher-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists() or args.admission_output.exists():
        raise ValueError("refuse to overwrite preparation history")
    if args.capture_plan is not None:
        if args.capture_output_dir is None or args.capture_plan_sha256 is None:
            raise ValueError(
                "capture import requires admitted plan pin and new materialization directory"
            )
        args.manifest = import_capture_plan(
            args.capture_plan,
            expected_sha256=args.capture_plan_sha256,
            output_dir=args.capture_output_dir,
            max_capture_bytes=args.max_capture_bytes,
            admission_output=args.admission_output,
        )
        args.manifest_sha256 = file_sha256(args.manifest)
    import_admission = args.admission_output if args.capture_plan is not None else None
    dataset = BlockDataset(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        max_teacher_bytes=args.max_teacher_bytes,
        admission_path=import_admission,
        admission_sha256=file_sha256(import_admission) if import_admission is not None else None,
    )
    # Only one block per chain is opened; constructor checks every horizon and
    # finite array in bounded chunks. No eager corpus teacher tensor is created.
    for cid in dataset.chains:
        dataset.load_block(cid, 0, require_teacher=args.require_soft_teachers)
    report = {
        "schema": "block_data_preparation_v1",
        "manifest_sha256": dataset.sha256,
        "manifest_path": str(args.manifest.resolve()),
        "completed_admission": {
            "path": str(
                (args.admission_output if import_admission is None else import_admission).resolve()
            ),
            "sha256": dataset.write_admission(args.admission_output)
            if import_admission is None
            else file_sha256(import_admission),
        },
        "producer_receipt_sha256": dataset.manifest["producer"]["receipt"]["sha256"],
        "source_sha256": file_sha256(ROOT / "src/w1a1_eagle/block_data.py"),
        "family": dataset.manifest["family"],
        "layout": dataset.manifest["layout"],
        "storage": dataset.storage_summary(),
        "profiles": {
            "hard_ce_target_prefix": "PASS",
            "exact_soft_teacher_target_prefix": "PASS"
            if all(c["logits"] is not None for c in dataset.chains.values())
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
