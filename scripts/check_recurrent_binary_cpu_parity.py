#!/usr/bin/env python3
"""Replay archived candidate-D native samples with the CPU scalar reference.

The input JSONL contains outputs from a previously completed native run. This
command reads those files and executes only NumPy scalar arithmetic on CPU.
It does not validate a newly trained export or start model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_binary_rescue import Model  # noqa: E402
from select_w1ax_captures import HEADER, parse_capture, sha256  # noqa: E402

from w1a1_eagle.recurrent_binary import (  # noqa: E402
    GroupedBinaryLinear,
    replay_packed_linear,
)


def audit(draft: Path, selected: Path, native_rows: Path, arithmetic: str = "native_order") -> dict:
    if arithmetic not in ("native_order", "group_matmul"):
        raise ValueError("unknown arithmetic mode")
    model = Model(draft)
    rows = [json.loads(line) for line in native_rows.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("native sample file is empty")
    checks = 0
    exact_matches = 0
    max_abs = 0.0
    max_scaled_abs = 0.0
    names = set()
    capture_hashes = {}
    for row in rows:
        if row.get("record_type") != "operator_replay" or row.get("replay_bits") != 16:
            raise ValueError("expected archived native A16 operator replay")
        capture_name = row["capture"]
        if Path(capture_name).name != capture_name:
            raise ValueError("capture must be a basename under selected directory")
        path = selected / capture_name
        capture = parse_capture(path)
        if capture["K"] != row["K"] or capture["name"] != row["name"].replace(
            ".w1a1_packed", ".weight"
        ):
            raise ValueError("capture and native row identity differ")
        data = np.memmap(
            path, mode="r", dtype="<f4", offset=HEADER.size, shape=(capture["N"], capture["K"])
        )
        base = row["name"].removesuffix(".w1a1_packed")
        if base == row["name"]:
            raise ValueError("expected a packed binary projection")
        tokens, outputs = row["token_indices"], row["row_indices"]
        if not tokens or not outputs or any(t < 0 or t >= capture["N"] for t in tokens):
            raise ValueError("invalid selected token indices")
        packed = model.tensors[base + ".w1a1_packed"].data[outputs]
        scales = model.tensors[base + ".w1a1_scale"].data[outputs]
        if arithmetic == "native_order":
            actual = replay_packed_linear(np.asarray(data[tokens]), packed, scales, capture["K"])
        else:
            layer = GroupedBinaryLinear.from_packed(
                packed, scales, in_features=capture["K"], arithmetic="group_matmul"
            )
            with torch.no_grad():
                actual = layer(torch.from_numpy(np.array(data[tokens], dtype=np.float32))).numpy()
        expected = np.asarray(row["sampled_native_outputs"], dtype=np.float32).reshape(
            len(tokens), len(outputs)
        )
        if arithmetic == "native_order" and not np.array_equal(actual, expected):
            raise ValueError(f"CPU/native sample mismatch: {base}/{capture_name}")
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        exact_matches += int(np.count_nonzero(actual == expected))
        max_abs = max(max_abs, float(delta.max()))
        max_scaled_abs = max(max_scaled_abs, float((delta / (1 + np.abs(expected))).max()))
        checks += actual.size
        names.add(base)
        capture_hashes[capture_name] = sha256(path)
    return {
        "schema": "recurrent_binary_cpu_parity_v1",
        "execution_device": "cpu",
        "arithmetic": arithmetic,
        "reference": "archived native sampled outputs; no current GPU execution",
        "draft_sha256": sha256(draft),
        "native_rows_sha256": sha256(native_rows),
        "capture_sha256": capture_hashes,
        "captures": len(rows),
        "layers": sorted(names),
        "checked_outputs": checks,
        "exact_matches": exact_matches,
        "max_abs_error": max_abs,
        "max_abs_over_one_plus_native": max_scaled_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--native-rows", type=Path, required=True)
    parser.add_argument(
        "--arithmetic", choices=("native_order", "group_matmul"), default="native_order"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new path")
    result = audit(args.draft, args.selected, args.native_rows, args.arithmetic)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"captures": result["captures"], "exact_matches": result["exact_matches"]}))


if __name__ == "__main__":
    main()
