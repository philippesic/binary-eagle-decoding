#!/usr/bin/env python3
"""Replay Qwen3 block-0 Q projection, head norm and RoPE on ggml CUDA."""

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
import transformers
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

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

Q_WIDTH = 4096
HEAD_WIDTH = 128
HEADS = Q_WIDTH // HEAD_WIDTH


def _q_weights(target: Path, hf_model: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    if sha256(target) != TARGET_GGUF_SHA256:
        raise ValueError("target GGUF differs from pinned F16 source")
    reader = GGUFReader(target, mode="r")
    matches = [tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_q.weight"]
    if len(matches) != 1 or matches[0].tensor_type != GGMLQuantizationType.F16:
        raise ValueError("target Q weight is not one native F16 GGUF tensor")
    weight = np.asarray(matches[0].data, dtype="<f2")
    if weight.shape != (Q_WIDTH, HIDDEN):
        raise ValueError("target Q weight has wrong matrix geometry")
    norm_matches = [
        tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_q_norm.weight"
    ]
    if len(norm_matches) != 1 or norm_matches[0].tensor_type != GGMLQuantizationType.F32:
        raise ValueError("target Q norm weight is not one native F32 GGUF tensor")
    norm = np.asarray(norm_matches[0].data, dtype="<f4")
    if norm.shape != (HEAD_WIDTH,):
        raise ValueError("target Q norm weight has wrong geometry")
    index = json.loads((hf_model / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = hf_model / index[name]
    with safe_open(shard, framework="pt", device="cpu") as file:
        source = file.get_tensor(name).to(torch.float16).numpy()
    if not np.array_equal(weight, source):
        raise ValueError("HF and GGUF F16 Q weight bytes differ")
    norm_name = "model.layers.0.self_attn.q_norm.weight"
    norm_shard = hf_model / index[norm_name]
    with safe_open(norm_shard, framework="pt", device="cpu") as file:
        norm_source = file.get_tensor(norm_name).to(torch.float32).numpy()
    if not np.array_equal(norm, norm_source):
        raise ValueError("HF and GGUF Q norm values differ")
    config = json.loads((hf_model / "config.json").read_text())
    if config.get("rope_theta") != 1000000 or config.get("max_position_embeddings") != 40960:
        raise ValueError("HF source has different Qwen3 RoPE parameters")
    return weight.copy(), norm.copy(), shard


def _compile_helper(build_bin: Path, binary: Path) -> None:
    source = ROOT / "scripts/native_target_q_cuda.cpp"
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


def _load(operands: Path, mode: str, stage: str) -> np.ndarray:
    path = operands / f"ggml_q_{mode}_{stage}.f32"
    values = np.fromfile(path, dtype="<f4")
    if values.size != TOKENS * Q_WIDTH:
        raise ValueError(f"Q {mode} {stage} has wrong geometry")
    return values.reshape(TOKENS, Q_WIDTH)


def probe(
    run_dir: Path,
    ladder: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    norm_run: Path,
    q_run: Path,
    deferred_audit: Path,
    source_build_run: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start Q replay through scripts/remote_job.py")
    prefix, _, _ = sealed_ladder(ladder, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen training prompt or candidate D differs")
    native_norm, norm_report = _safe_rows(norm_run, "attn_norm", "attn_norm-0", HIDDEN)
    native_q, q_report = _safe_rows(q_run, "q_deferred_k_norm", "Qcur-0", Q_WIDTH)
    if (
        norm_report["source_sha256"]["libggml-cuda.so"]
        != q_report["source_sha256"]["libggml-cuda.so"]
    ):
        raise ValueError("safe native norm and deferred Q captures use different CUDA builds")
    audit = json.loads(deferred_audit.read_text())
    if (
        audit.get("schema") != "target_q_deferred_audit_v1"
        or audit.get("status") != "safe_q_deferred_capture"
        or audit["source_sha256"]["deferred_report"] != sha256(q_run / "comparison.json")
        or audit["source_sha256"]["deferred_q"] != sha256(q_run / "block0/Qcur-0_0.f32")
    ):
        raise ValueError("deferred Q has not passed safe K/output audit")
    weight, norm_weight, hf_shard = _q_weights(target, hf_model)
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("Q comparison requires the registered RTX 5080")
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
    helper = build / "bin/native-target-q-cuda"
    _compile_helper(build / "bin", helper)
    operands = run_dir / "operands"
    operands.mkdir()
    native_norm.astype("<f4").tofile(operands / "native_norm.f32")
    weight.astype("<f2").tofile(operands / "q_weight.f16")
    norm_weight.astype("<f4").tofile(operands / "q_norm_weight.f32")
    np.arange(TOKENS, dtype="<i4").tofile(operands / "positions.i32")
    for mode in ("f32", "f16cast", "f32retain"):
        subprocess.run([str(helper), str(operands), mode], check=True, timeout=120)
    ggml = {
        mode: {stage: _load(operands, mode, stage) for stage in ("raw", "normed", "rope")}
        for mode in ("f32", "f16cast", "f32retain")
    }
    if not all(np.isfinite(value).all() for stages in ggml.values() for value in stages.values()):
        raise ValueError("Q replay produced nonfinite values")
    retained = ggml["f32retain"]
    norm_l2 = np.linalg.norm(retained["normed"].astype(np.float64), axis=1)
    rope_l2 = np.linalg.norm(retained["rope"].astype(np.float64), axis=1)
    if not np.allclose(norm_l2, rope_l2, rtol=1e-5, atol=1e-5):
        raise ValueError("retained Q intermediate readback fails RoPE norm preservation")
    config = AutoConfig.from_pretrained(hf_model, local_files_only=True)
    if config.rms_norm_eps != 1e-6:
        raise ValueError("HF Qwen3 RMS epsilon differs from pinned ggml graph")
    with torch.no_grad():
        input_half = torch.from_numpy(native_norm.astype("<f2")).to(device="cuda")
        weight_half = torch.from_numpy(weight).to(device="cuda")
        q_raw = F.linear(input_half, weight_half).reshape(1, TOKENS, HEADS, HEAD_WIDTH)
        q_norm_module = Qwen3RMSNorm(HEAD_WIDTH, eps=config.rms_norm_eps).to(
            device="cuda", dtype=torch.float16
        )
        q_norm_module.weight.copy_(torch.from_numpy(norm_weight).to(device="cuda"))
        q_norm = q_norm_module(q_raw).transpose(1, 2)
        rotary = Qwen3RotaryEmbedding(config, device="cuda").to(device="cuda")
        positions = torch.arange(TOKENS, device="cuda", dtype=torch.long)[None, :]
        cos, sin = rotary(q_norm, positions)
        q_rope, _ = apply_rotary_pos_emb(q_norm, q_norm, cos, sin)
        torch_stages = {
            "raw": q_raw.float().cpu().numpy().reshape(TOKENS, Q_WIDTH),
            "normed": q_norm.transpose(1, 2).float().cpu().numpy().reshape(TOKENS, Q_WIDTH),
            "rope": q_rope.transpose(1, 2).float().cpu().numpy().reshape(TOKENS, Q_WIDTH),
        }
        torch.cuda.synchronize()
    if not all(np.isfinite(value).all() for value in torch_stages.values()):
        raise ValueError("Torch Q path produced nonfinite values")
    metrics = {
        "ggml_f32_rope_vs_server": _metrics(ggml["f32"]["rope"], native_q),
        "ggml_f16cast_rope_vs_server": _metrics(ggml["f16cast"]["rope"], native_q),
        "ggml_cast_vs_f32_raw": _metrics(ggml["f16cast"]["raw"], ggml["f32"]["raw"]),
        "ggml_cast_vs_f32_rope": _metrics(ggml["f16cast"]["rope"], ggml["f32"]["rope"]),
        "ggml_retained_raw_vs_exact_raw": _metrics(retained["raw"], ggml["f32"]["raw"]),
        "ggml_retained_rope_vs_server": _metrics(retained["rope"], native_q),
        "torch_f16_vs_ggml_raw": _metrics(torch_stages["raw"], ggml["f32"]["raw"]),
        "torch_f16_vs_ggml_retained_normed": _metrics(torch_stages["normed"], retained["normed"]),
        "torch_f16_vs_ggml_retained_rope": _metrics(torch_stages["rope"], retained["rope"]),
        "torch_f16_rope_vs_server": _metrics(torch_stages["rope"], native_q),
    }
    report = {
        "schema": "target_q_cuda_same_input_projection_v1",
        "status": "server_q_rope_bitwise"
        if metrics["ggml_f32_rope_vs_server"]["exact_f32"] == native_q.size
        else "operator_proxy_differs",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": (
            "F16 GGUF Q matrix, F32 GGUF head norm and RoPE; "
            "F32 or F16-cast native input; Torch CUDA/F16 source Qwen3 control"
        ),
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "native_norm_capture": sha256(norm_run / "comparison.json"),
            "native_q_capture": sha256(q_run / "comparison.json"),
            "deferred_q_audit": sha256(deferred_audit),
            "native_norm_f32": sha256(operands / "native_norm.f32"),
            "q_weight_f16": sha256(operands / "q_weight.f16"),
            "q_norm_weight_f32": sha256(operands / "q_norm_weight.f32"),
            "positions_i32": sha256(operands / "positions.i32"),
            "hf_weight_shard": sha256(hf_shard),
            "hf_config": sha256(hf_model / "config.json"),
            "hf_qwen3_implementation": sha256(Path(sys.modules[Qwen3RMSNorm.__module__].__file__)),
            "helper_source": sha256(ROOT / "scripts/native_target_q_cuda.cpp"),
            "helper_binary": sha256(helper),
            "ggml_cuda_library": sha256(build / "bin/libggml-cuda.so"),
            "cmake_cache": sha256(build / "CMakeCache.txt"),
            "probe": sha256(Path(__file__)),
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
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
    parser.add_argument("--native-q-run", type=Path, required=True)
    parser.add_argument("--deferred-audit-report", type=Path, required=True)
    parser.add_argument("--source-build-run", type=Path, required=True)
    args = parser.parse_args()
    result = probe(
        ROOT / "runs" / args.run_id,
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.hf_model,
        args.native_norm_run,
        args.native_q_run,
        args.deferred_audit_report,
        args.source_build_run,
    )
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "server_q_rope_bitwise":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
