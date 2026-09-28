#!/usr/bin/env python3
"""Replay block-0 Flash Attention, output projection and residual on ggml CUDA."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))

from check_target_block0_capture import (  # noqa: E402
    HIDDEN,
    LADDER_MANIFEST_SHA256,
    PROMPT_ID,
    TARGET_GGUF_SHA256,
    TOKENS,
    sealed_ladder,
    sha256,
)
from check_target_block0_hf import D_SHA256, _metrics  # noqa: E402
from check_target_k_cuda import _safe_rows  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402

HEAD = 128
Q_HEADS = 32
KV_HEADS = 8
Q_WIDTH = HEAD * Q_HEADS
KV_WIDTH = HEAD * KV_HEADS
KV_SLOTS = 256


def _o_weight(target: Path, hf_model: Path) -> tuple[np.ndarray, Path]:
    if sha256(target) != TARGET_GGUF_SHA256:
        raise ValueError("target GGUF differs from pinned F16 source")
    reader = GGUFReader(target, mode="r")
    matches = [tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_output.weight"]
    if len(matches) != 1 or matches[0].tensor_type != GGMLQuantizationType.F16:
        raise ValueError("target output projection is not one native F16 GGUF tensor")
    weight = np.asarray(matches[0].data, dtype="<f2")
    if weight.shape != (HIDDEN, Q_WIDTH):
        raise ValueError("target output projection has wrong geometry")
    index = json.loads((hf_model / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.0.self_attn.o_proj.weight"
    shard = hf_model / index[name]
    with safe_open(shard, framework="pt", device="cpu") as file:
        source = file.get_tensor(name).to(torch.float16).numpy()
    if not np.array_equal(weight, source):
        raise ValueError("HF and GGUF F16 output projection bytes differ")
    return weight.copy(), shard


def _compile_helper(build_bin: Path, binary: Path) -> None:
    source = ROOT / "scripts/native_target_attention_cuda.cpp"
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
    ladder_dir: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    q_run: Path,
    k_run: Path,
    v_run: Path,
    ffn_run: Path,
    source_build_run: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start Flash Attention replay through scripts/remote_job.py")
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen training prompt or candidate D differs")
    stages = {
        "q": (q_run, "q_deferred_k_norm", "Qcur-0", Q_WIDTH),
        "k": (k_run, "k_rope", "Kcur-0", KV_WIDTH),
        "v": (v_run, "v_only", "Vcur-0", KV_WIDTH),
        "ffn_input": (ffn_run, "ffn", "ffn_inp-0", HIDDEN),
    }
    native = {}
    capture_hashes = {}
    cuda_libraries = set()
    for stage, (path, mode, name, width) in stages.items():
        native[stage], report = _safe_rows(path, mode, name, width)
        if report["source_sha256"]["target_gguf"] != TARGET_GGUF_SHA256:
            raise ValueError(f"safe {stage} target GGUF differs")
        cuda_libraries.add(report["source_sha256"]["libggml-cuda.so"])
        capture_hashes[stage] = sha256(path / "comparison.json")
    if len(cuda_libraries) != 1:
        raise ValueError("safe native Q/K/V and residual use different CUDA libraries")
    weight, hf_shard = _o_weight(target, hf_model)
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("attention replay requires the registered RTX 5080")
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
    helper = build / "bin/native-target-attention-cuda"
    _compile_helper(build / "bin", helper)
    operands = run_dir / "operands"
    operands.mkdir()
    native["q"].astype("<f4").tofile(operands / "q_rope.f32")
    k_cache = np.zeros((KV_SLOTS, KV_WIDTH), dtype="<f2")
    v_cache = np.zeros((KV_SLOTS, KV_WIDTH), dtype="<f2")
    k_cache[:TOKENS] = native["k"].astype("<f2")
    v_cache[:TOKENS] = native["v"].astype("<f2")
    k_cache.tofile(operands / "k_cache.f16")
    v_cache.tofile(operands / "v_cache.f16")
    mask = np.full((TOKENS, KV_SLOTS), -np.inf, dtype="<f2")
    for position in range(TOKENS):
        mask[position, : position + 1] = 0
    mask.tofile(operands / "causal_mask.f16")
    weight.tofile(operands / "o_weight.f16")
    np.asarray(ladder[:, 0, :], dtype="<f4").tofile(operands / "layer_input.f32")
    for mode in ("qf32", "qf16cast"):
        subprocess.run([str(helper), str(operands), mode], check=True, timeout=120)
    outputs = {}
    for mode in ("qf32", "qf16cast"):
        values = np.fromfile(operands / f"ggml_attn_{mode}_residual.f32", dtype="<f4")
        if values.size != TOKENS * HIDDEN:
            raise ValueError(f"ggml {mode} attention residual has wrong size")
        outputs[mode] = values.reshape(TOKENS, HIDDEN)
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("ggml attention residual has nonfinite values")
    metrics = {
        "ggml_qf32_vs_server": _metrics(outputs["qf32"], native["ffn_input"]),
        "ggml_qf16cast_vs_server": _metrics(outputs["qf16cast"], native["ffn_input"]),
        "ggml_q_cast_vs_qf32": _metrics(outputs["qf16cast"], outputs["qf32"]),
    }
    report = {
        "schema": "target_attention_cuda_same_input_v1",
        "status": "server_attention_residual_bitwise"
        if metrics["ggml_qf32_vs_server"]["exact_f32"] == TOKENS * HIDDEN
        else "operator_proxy_differs",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": (
            "F32 native post-RoPE Q or explicit F16 cast; F16 padded K/V cache and causal mask; "
            "F32 Flash Attention accumulation; F16 GGUF output weight; F32 residual"
        ),
        "prefill_tokens": TOKENS,
        "kv_slots": KV_SLOTS,
        "prompt_id": PROMPT_ID,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "safe_native_reports": capture_hashes,
            "cuda_library_from_safe_captures": cuda_libraries.pop(),
            "ggml_cuda_library": sha256(build / "bin/libggml-cuda.so"),
            "hf_weight_shard": sha256(hf_shard),
            **{path.name: sha256(path) for path in operands.iterdir() if path.is_file()},
            "helper_source": sha256(ROOT / "scripts/native_target_attention_cuda.cpp"),
            "helper_binary": sha256(helper),
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
    parser.add_argument("--native-q-run", type=Path, required=True)
    parser.add_argument("--native-k-run", type=Path, required=True)
    parser.add_argument("--native-v-run", type=Path, required=True)
    parser.add_argument("--native-ffn-run", type=Path, required=True)
    parser.add_argument("--source-build-run", type=Path, required=True)
    args = parser.parse_args()
    result = probe(
        ROOT / "runs" / args.run_id,
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.hf_model,
        args.native_q_run,
        args.native_k_run,
        args.native_v_run,
        args.native_ffn_run,
        args.source_build_run,
    )
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "server_attention_residual_bitwise":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
