#!/usr/bin/env python3
"""Replay synthetic fusion operands through the existing native CPU A8 operator.

This gate does not invoke a model or establish real-data fitting/acceptance.
The caller supplies a replay binary built with CUDA and Metal disabled.
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


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.operands, allow_pickle=False) as operands:
        inputs = operands["raw_input"]
        weight = operands["reference_weight"]
    if (
        inputs.dtype != np.float32
        or weight.dtype != np.float32
        or inputs.ndim != 2
        or weight.ndim != 2
        or inputs.shape[1] != weight.shape[1]
        or not np.isfinite(inputs).all()
        or not np.isfinite(weight).all()
        or not 0 < len(inputs) <= 512
        or not 0 < weight.shape[0] <= 2560
        or not 0 < inputs.shape[1] <= 7680
    ):
        raise ValueError("expected bounded finite F32 synthetic fusion operands")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    captures = args.output_dir / "captures"
    captures.mkdir()
    capture = captures / "op-000000000001.bin"
    capture.write_bytes(
        struct.pack(
            "<8sQQQQI128s",
            b"W1AXACT1",
            1,
            inputs.shape[1],
            weight.shape[0],
            len(inputs),
            32,
            b"fc.weight",
        )
        + np.ascontiguousarray(inputs, dtype="<f4").tobytes()
    )
    command = [
        str(args.binary.resolve()),
        "--gguf",
        str(args.gguf.resolve()),
        "--capture-dir",
        str(captures.resolve()),
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
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "2"
    environment["GGML_METAL_DISABLE"] = "1"
    with (
        (args.output_dir / "native.jsonl").open("x") as out,
        (args.output_dir / "native.stderr").open("x") as err,
    ):
        completed = subprocess.run(command, stdout=out, stderr=err, env=environment, check=False)
    records = [
        json.loads(line)
        for line in (args.output_dir / "native.jsonl").read_text().splitlines()
        if line.strip()
    ]
    operators = [record for record in records if record.get("record_type") == "operator_replay"]
    passed = (
        completed.returncode == 0
        and len(operators) == 1
        and operators[0].get("backend_type") == "cpu"
        and operators[0].get("replay_bits") == 8
        and operators[0].get("checked_outputs") == inputs.shape[0] * weight.shape[0]
    )
    report = {
        "synthetic_only": True,
        "gate_passed": passed,
        "returncode": completed.returncode,
        "command": command,
        "operands_sha256": sha256(args.operands),
        "gguf_sha256": sha256(args.gguf),
        "binary_sha256": sha256(args.binary),
        "capture_sha256": sha256(capture),
        "native_jsonl_sha256": sha256(args.output_dir / "native.jsonl"),
        "operators": operators,
        "acceptance_or_performance_claim": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"gate_passed": passed, "report": str(args.output_dir / "report.json")}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
