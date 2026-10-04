#!/usr/bin/env python3
"""Replay eight authenticated held-out fusion inputs through native CPU A8.

This is a layer arithmetic gate, not a target/drafter trajectory evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-sha256", required=True)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if sha256(a.receipt) != a.receipt_sha256 or sha256(a.candidate) != a.candidate_sha256:
        raise ValueError("external provenance/candidate pin differs")
    receipt = json.loads(a.receipt.read_text())
    manifest = json.loads(a.manifest.read_text())
    operands = a.manifest.parent / manifest["operands"]
    if (
        receipt["manifest_sha256"] != sha256(a.manifest)
        or receipt["operands_sha256"] != sha256(operands)
        or manifest["synthetic"] is not False
    ):
        raise ValueError("real operand archive does not match trusted receipt")
    with np.load(operands, allow_pickle=False) as z:
        raw = z["raw_input"]
    with np.load(a.candidate, allow_pickle=False) as z:
        packed, scale = z["fc.w1a1_packed"], z["fc.scale"]
    if (
        raw.dtype != np.float32
        or raw.shape != (384, 7680)
        or packed.shape != (2560, 240)
        or scale.shape != (2560,)
        or not np.isfinite(raw).all()
        or scale.dtype != np.float32
    ):
        raise ValueError("unexpected frozen real fusion operand geometry")
    validation = [i for i, row in enumerate(manifest["rows"]) if row["split"] == "validation"]
    chosen = [validation[i] for i in np.linspace(0, len(validation) - 1, 8, dtype=int)]
    values = np.ascontiguousarray(raw[chosen], dtype="<f4")
    for i, row in zip(chosen, values, strict=True):
        if hashlib.sha256(row.tobytes()).hexdigest() != manifest["rows"][i]["raw_input_sha256"]:
            raise ValueError("selected held-out row hash differs")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    directory = a.output_dir / "captures"
    directory.mkdir()
    capture = directory / "op-000000000001.bin"
    capture.write_bytes(
        struct.pack("<8sQQQQI128s", b"W1AXACT1", 1, 7680, 2560, 8, 32, b"fc.weight")
        + values.tobytes()
    )
    command = [
        str(a.binary.resolve()),
        "--gguf",
        str(a.gguf.resolve()),
        "--capture-dir",
        str(directory.resolve()),
        "--backend",
        "cpu",
        "--act-bits",
        "8",
        "--check-rows",
        "0",
        "--warmups",
        "0",
        "--samples",
        "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    with (
        (a.output_dir / "native.jsonl").open("x") as out,
        (a.output_dir / "native.stderr").open("x") as err,
    ):
        result = subprocess.run(command, stdout=out, stderr=err, env=env, check=False)
    records = [
        json.loads(line)
        for line in (a.output_dir / "native.jsonl").read_text().splitlines()
        if line.strip()
    ]
    operations = [r for r in records if r.get("record_type") == "operator_replay"]
    passed = (
        result.returncode == 0
        and len(operations) == 1
        and operations[0]["backend_type"] == "cpu"
        and operations[0]["replay_bits"] == 8
        and operations[0]["checked_outputs"] == 20480
        and operations[0]["max_abs_error"] == 0
    )
    report = {
        "gate_passed": passed,
        "synthetic": False,
        "command": command,
        "receipt_sha256": a.receipt_sha256,
        "candidate_sha256": a.candidate_sha256,
        "gguf_sha256": sha256(a.gguf),
        "binary_sha256": sha256(a.binary),
        "capture_sha256": sha256(capture),
        "selected_rows": chosen,
        "selected_row_ids": [manifest["rows"][i]["row_id"] for i in chosen],
        "operator": operations[0] if operations else None,
        "native_acceptance_or_throughput_claim": False,
    }
    (a.output_dir / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"gate_passed": passed, "checked_outputs": 20480}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
