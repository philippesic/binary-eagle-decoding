#!/usr/bin/env python3
"""Replay one sealed CPU EAGLE SiLU row with the pinned ggml CPU library."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import sha256  # noqa: E402
from check_recurrent_ffn_from_graph import _array_sha256, _cpu_model, _metrics  # noqa: E402
from check_recurrent_ffn_stage_parity import verify_capture  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402


def ggml_silu(lib_path: Path, gate: np.ndarray, *, scalar: bool) -> np.ndarray:
    """Call the built ggml function; n=1 selects its scalar tail path."""
    lib = ctypes.CDLL(str(lib_path.resolve()))
    func = lib.ggml_vec_silu_f32
    pointer = ctypes.POINTER(ctypes.c_float)
    func.argtypes = (ctypes.c_int, pointer, pointer)
    func.restype = None
    src = np.ascontiguousarray(gate, dtype=np.float32)
    dst = np.empty_like(src)
    if scalar:
        for index in range(src.size):
            dst_ptr = dst[index : index + 1].ctypes.data_as(pointer)
            src_ptr = src[index : index + 1].ctypes.data_as(pointer)
            func(1, dst_ptr, src_ptr)
    else:
        func(src.size, dst.ctypes.data_as(pointer), src.ctypes.data_as(pointer))
    return dst


def audit(capture: Path, stage_report: Path, lib: Path, binary: Path, cmake: Path) -> dict:
    if (
        lib.resolve().parent != binary.resolve().parent
        or cmake.resolve().parent != binary.resolve().parent.parent
    ):
        raise ValueError("ggml CPU library, server binary and CMake cache must share one build")
    manifest, seal = verify_capture(capture, require_stages=True)
    prior = json.loads(stage_report.read_text())
    if (
        prior.get("schema") != "recurrent_binary_ffn_stage_parity_cpu_v1"
        or prior["source_sha256"]["new_capture_manifest"] != seal["manifest"]
        or prior["source_sha256"]["native_binary"] != sha256(binary)
        or prior["source_sha256"]["cmake_cache"] != sha256(cmake)
        or manifest.get("native_revision") != prior["native_revisions"]["new"]
    ):
        raise ValueError("stage report, sealed capture or native build identity differs")
    records, values, footer = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    selected = [
        row
        for row in records
        if row.get("group_kind") == "decoder" and row.get("group_execution") == 2
    ]
    native = {}
    for stage, name in (("gate", "ffn_gate-0"), ("silu_gate", "ffn_gate_silu-0")):
        matches = [row for row in selected if row.get("tensor_name") == name]
        if len(matches) != 1 or matches[0].get("dtype") != "f32":
            raise ValueError(f"expected one native F32 {name} tap")
        native[stage] = np.array(_column(matches[0], values, 0), copy=True)
        expected = prior["results"]["native_order"]["native_stage_sha256"][stage]
        if native[stage].shape != (9728,) or _array_sha256(native[stage]) != expected:
            raise ValueError(f"native {name} row differs from sealed stage report")
    if footer.get("status") != "complete":
        raise ValueError("native graph is incomplete")

    gate = native["gate"]
    with torch.no_grad():
        source = torch.from_numpy(gate.copy())
        torch_silu = F.silu(source).numpy().copy()
        torch_sigmoid_mul = (source * torch.sigmoid(source)).numpy().copy()
    with np.errstate(over="ignore"):
        numpy_f32 = gate / (np.float32(1) + np.exp(-gate))
    choices = {
        "ggml_cpu_vector": ggml_silu(lib, gate, scalar=False),
        "ggml_cpu_scalar_tail": ggml_silu(lib, gate, scalar=True),
        "torch_silu": torch_silu,
        "torch_sigmoid_mul": torch_sigmoid_mul,
        "numpy_f32_exp_div": numpy_f32,
    }
    return {
        "schema": "recurrent_binary_silu_arithmetic_cpu_v1",
        "scope": "one sealed reasoning first-seed FFN gate row; elementwise only",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "stage_report": sha256(stage_report),
            "ggml_cpu_library": sha256(lib),
            "native_binary": sha256(binary),
            "cmake_cache": sha256(cmake),
            "gate": _array_sha256(gate),
            "native_silu": _array_sha256(native["silu_gate"]),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
        "results": {
            name: {
                "versus_native": _metrics(value, native["silu_gate"]),
                "output_sha256": _array_sha256(value),
            }
            for name, value in choices.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--ggml-cpu-lib", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    torch.set_num_threads(1)
    report = audit(
        args.capture_dir,
        args.stage_report,
        args.ggml_cpu_lib,
        args.native_binary,
        args.cmake_cache,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["versus_native"] for key, value in report["results"].items()}))


if __name__ == "__main__":
    main()
