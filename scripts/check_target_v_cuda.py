#!/usr/bin/env python3
"""Replay Qwen3 block-0 V matmul on ggml CUDA with F32 and F16-cast input."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))

from check_target_block0_capture import (  # noqa: E402
    HIDDEN,
    LADDER_MANIFEST_SHA256,
    PROMPT_ID,
    TARGET_GGUF_SHA256,
    TOKENS,
    _index,
    sealed_ladder,
    sha256,
)
from check_target_block0_hf import D_SHA256, _metrics, _native_rows  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

V_WIDTH = 1024


def _safe_rows(run: Path, mode: str, name: str, width: int) -> tuple[np.ndarray, dict]:
    report_path = run / "comparison.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "target_block0_native_cuda_capture_v1"
        or report.get("status") != "same_native_block_output"
        or report.get("capture_mode") != mode
        or report.get("prompt_id") != PROMPT_ID
        or report["block_output"]["exact_elements"] != TOKENS * HIDDEN
        or report["source_sha256"]["ladder_manifest"] != LADDER_MANIFEST_SHA256
    ):
        raise ValueError(f"{mode} does not preserve the sealed native target block")
    entries, hashes = _index(run / "block0", mode=mode)
    if any(report["source_sha256"].get(key) != digest for key, digest in hashes.items()):
        raise ValueError(f"{mode} tensor payload differs from capture report")
    return _native_rows(run / "block0", entries, name, width), report


def _v_weight(target: Path, hf_model: Path) -> tuple[np.ndarray, Path]:
    if sha256(target) != TARGET_GGUF_SHA256:
        raise ValueError("target GGUF differs from pinned F16 source")
    reader = GGUFReader(target, mode="r")
    matches = [tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_v.weight"]
    if len(matches) != 1 or matches[0].tensor_type != GGMLQuantizationType.F16:
        raise ValueError("target V weight is not one native F16 GGUF tensor")
    native = np.asarray(matches[0].data, dtype="<f2")
    if native.shape != (V_WIDTH, HIDDEN):
        raise ValueError("target V weight has wrong matrix geometry")
    index = json.loads((hf_model / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.0.self_attn.v_proj.weight"
    shard = hf_model / index[name]
    with safe_open(shard, framework="pt", device="cpu") as file:
        source = file.get_tensor(name).to(torch.float16).numpy()
    if not np.array_equal(native, source):
        raise ValueError("HF and GGUF F16 V weight bytes differ")
    return native.copy(), shard


def _compile_helper(build_bin: Path, binary: Path) -> None:
    source = ROOT / "scripts/native_target_v_cuda.cpp"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I" + str(ROOT / "third_party/llama.cpp/ggml/include"),
        str(source),
        "-L" + str(build_bin),
        "-Wl,-rpath,$ORIGIN",
        "-lggml-base",
        "-lggml",
        "-lggml-cpu",
        "-o",
        str(binary),
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=120)


def probe(
    run_dir: Path,
    ladder: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    norm_run: Path,
    v_run: Path,
    prior_hf_report: Path,
    source_build_run: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start V projection probe through scripts/remote_job.py")
    prefix, _, _ = sealed_ladder(ladder, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen training prompt or candidate D differs")
    native_norm, norm_report = _safe_rows(norm_run, "attn_norm", "attn_norm-0", HIDDEN)
    native_v, v_report = _safe_rows(v_run, "v_only", "Vcur-0", V_WIDTH)
    if (
        norm_report["source_sha256"]["libggml-cuda.so"]
        != v_report["source_sha256"]["libggml-cuda.so"]
    ):
        raise ValueError("safe native norm and V captures use different CUDA builds")
    weight, hf_shard = _v_weight(target, hf_model)
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("projection comparison requires the registered RTX 5080")
    prior = json.loads(prior_hf_report.read_text())
    if (
        prior.get("schema") != "target_block0_hf_cuda_comparison_v1"
        or prior.get("target_rope_layout") != "NeoX half-split, no Q/K row permutation"
        or prior["source_sha256"]["safe_native_reports"]["attn_norm"]
        != sha256(norm_run / "comparison.json")
        or prior["source_sha256"]["safe_native_reports"]["v_only"]
        != sha256(v_run / "comparison.json")
    ):
        raise ValueError("prior HF same-input control differs from safe native captures")
    source_state = json.loads((source_build_run / "state.json").read_text())
    if (
        source_state.get("status") != "finished"
        or not (source_build_run / "build/bin/libggml-cuda.so").is_file()
    ):
        raise ValueError("reused CUDA build is missing or still running")
    build = run_dir / "build"
    build.mkdir()
    shutil.copy2(source_build_run / "build/CMakeCache.txt", build / "CMakeCache.txt")
    shutil.copytree(source_build_run / "build/bin", build / "bin", symlinks=True)
    helper = build / "bin/native-target-v-cuda"
    _compile_helper(build / "bin", helper)
    operands = run_dir / "operands"
    operands.mkdir()
    native_norm.astype("<f4").tofile(operands / "native_norm.f32")
    weight.astype("<f2").tofile(operands / "v_weight.f16")
    for mode in ("f32", "f16cast"):
        subprocess.run([str(helper), str(operands), mode], check=True, timeout=120)
    ggml_f32 = np.fromfile(operands / "ggml_v_f32.f32", dtype="<f4").reshape(TOKENS, V_WIDTH)
    ggml_cast = np.fromfile(operands / "ggml_v_f16cast.f32", dtype="<f4").reshape(TOKENS, V_WIDTH)
    with torch.no_grad():
        input_half = torch.from_numpy(native_norm.astype("<f2")).to(device="cuda")
        weight_half = torch.from_numpy(weight).to(device="cuda")
        torch_output = F.linear(input_half, weight_half).float().cpu().numpy()
        torch.cuda.synchronize()
    if torch_output.shape != native_v.shape or not all(
        np.isfinite(array).all() for array in (ggml_f32, ggml_cast, torch_output)
    ):
        raise ValueError("projection operator returned wrong or nonfinite shape")
    metrics = {
        "ggml_f32_vs_server": _metrics(ggml_f32, native_v),
        "ggml_f16cast_vs_server": _metrics(ggml_cast, native_v),
        "torch_f16_vs_server": _metrics(torch_output, native_v),
        "ggml_cast_vs_ggml_f32": _metrics(ggml_cast, ggml_f32),
        "torch_vs_ggml_cast": _metrics(torch_output, ggml_cast),
    }
    old_torch = prior["same_input_kv_intervention"]["Vcur-0"]
    same_input_reproduced = all(
        metrics["torch_f16_vs_server"][key] == old_torch[key]
        for key in ("exact_f16_cast", "exact_f32", "max_abs", "rms", "position3_relative_row_l2")
    )
    report = {
        "schema": "target_v_cuda_same_input_projection_v1",
        "status": "server_projection_bitwise"
        if metrics["ggml_f32_vs_server"]["exact_f32"] == native_v.size
        else "operator_proxy_differs",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": (
            "F16 GGUF V weight; F32 or explicit F16-cast native norm input; Torch F16 control"
        ),
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "same_input_torch_control_reproduced": same_input_reproduced,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "native_norm_capture": sha256(norm_run / "comparison.json"),
            "native_v_capture": sha256(v_run / "comparison.json"),
            "prior_hf_comparison": sha256(prior_hf_report),
            "v_weight_f16": sha256(operands / "v_weight.f16"),
            "native_norm_f32": sha256(operands / "native_norm.f32"),
            "hf_weight_shard": sha256(hf_shard),
            "helper_source": sha256(ROOT / "scripts/native_target_v_cuda.cpp"),
            "helper_binary": sha256(helper),
            "ggml_cuda_library": sha256(build / "bin/libggml-cuda.so"),
            "cmake_cache": sha256(build / "CMakeCache.txt"),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
    }
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--native-norm-run", type=Path, required=True)
    parser.add_argument("--native-v-run", type=Path, required=True)
    parser.add_argument("--prior-hf-report", type=Path, required=True)
    parser.add_argument("--source-build-run", type=Path, required=True)
    args = parser.parse_args()
    result = probe(
        ROOT / "runs" / args.run_id,
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.hf_model,
        args.native_norm_run,
        args.native_v_run,
        args.prior_hf_report,
        args.source_build_run,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "same_input_torch": result["same_input_torch_control_reproduced"],
            }
        )
    )
    if result["status"] != "server_projection_bitwise":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
