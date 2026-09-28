#!/usr/bin/env python3
"""Replay a sealed CPU FFN product/down from the native SiLU gate."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import sha256  # noqa: E402
from check_recurrent_ffn_from_graph import (  # noqa: E402
    FFN_BASES,
    _array_sha256,
    _cpu_model,
    _metrics,
)
from check_recurrent_ffn_stage_parity import read_first_seed, verify_capture  # noqa: E402
from load_recurrent_binary_init import D_SHA256, load_candidate_d_arrays  # noqa: E402

from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402


def audit(capture: Path, candidate_d: Path, stage_report: Path, silu_report: Path) -> dict:
    manifest, seal = verify_capture(capture, require_stages=True)
    stage = json.loads(stage_report.read_text())
    silu = json.loads(silu_report.read_text())
    if (
        stage.get("schema") != "recurrent_binary_ffn_stage_parity_cpu_v1"
        or silu.get("schema") != "recurrent_binary_silu_arithmetic_cpu_v1"
        or stage["source_sha256"]["new_capture_manifest"] != seal["manifest"]
        or silu["source_sha256"]["capture_manifest"] != seal["manifest"]
        or silu["source_sha256"]["stage_report"] != sha256(stage_report)
        or silu["results"]["ggml_cpu_vector"]["versus_native"]["exact_elements"] != 9728
        or silu["native_revision"] != manifest["native_revision"]
    ):
        raise ValueError("sealed capture, stage report or SiLU replay identity differs")
    arrays, d_audit = load_candidate_d_arrays(candidate_d)
    if d_audit["gguf_sha256"] != D_SHA256:
        raise ValueError("candidate D differs from pinned GGUF")
    first = read_first_seed(capture, candidate_d, require_stages=True)
    native = first["stages"]
    expected_hashes = stage["results"]["native_order"]["native_stage_sha256"]
    if any(_array_sha256(native[key]) != expected_hashes[key] for key in native):
        raise ValueError("captured FFN stage differs from prior sealed report")
    down_packed, down_scales = arrays[FFN_BASES[2]]
    down_linear = GroupedBinaryLinear.from_packed(
        down_packed, down_scales, in_features=9728, arithmetic="native_order"
    )
    with torch.no_grad():
        up = torch.from_numpy(native["up"].copy())
        gate = torch.from_numpy(native["gate"].copy())
        native_silu = torch.from_numpy(native["silu_gate"].copy())
        replay = {}
        for name, activated in (
            ("torch_silu_control", F.silu(gate)),
            ("captured_native_silu", native_silu),
        ):
            product = up * activated
            down = down_linear(product)
            replay[name] = {"mul": product.numpy().copy(), "down": down.numpy().copy()}
    results = {
        name: {
            key: {
                "versus_native": _metrics(value, native[key]),
                "output_sha256": _array_sha256(value),
            }
            for key, value in outputs.items()
        }
        for name, outputs in replay.items()
    }
    control = stage["results"]["native_order"]["versus_native"]
    for key in ("mul", "down"):
        if results["torch_silu_control"][key]["versus_native"] != control[key]:
            raise ValueError(f"Torch SiLU control does not reproduce prior {key} metrics")
    return {
        "schema": "recurrent_binary_ffn_native_silu_cpu_v1",
        "scope": "one sealed reasoning first-seed FFN column; no optimizer or GPU",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "precision": "native F32 up and SiLU; F32 product; A16 grouped ordered binary down",
        "native_revision": manifest["native_revision"],
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "candidate_d": D_SHA256,
            "stage_report": sha256(stage_report),
            "silu_report": sha256(silu_report),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--silu-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    torch.set_num_threads(1)
    result = audit(args.capture_dir, args.candidate_d, args.stage_report, args.silu_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                name: {key: value["versus_native"] for key, value in stages.items()}
                for name, stages in result["results"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
