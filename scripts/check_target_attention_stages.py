#!/usr/bin/env python3
"""Separate target block-0 Flash Attention from O projection on identical inputs."""

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
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    eager_attention_forward,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))

from check_target_attention_cuda import (  # noqa: E402
    KV_SLOTS,
    KV_WIDTH,
    Q_WIDTH,
    _compile_helper,
    _o_weight,
)
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


def _native_heads(rows: np.ndarray, heads: int) -> torch.Tensor:
    return (
        torch.from_numpy(rows.astype("<f2"))
        .to(device="cuda")
        .reshape(1, TOKENS, heads, 128)
        .transpose(1, 2)
    )


def _load(operands: Path, mode: str, width: int) -> np.ndarray:
    suffix = ".f32" if mode == "attn_only" else "_residual.f32"
    values = np.fromfile(operands / f"ggml_attn_{mode}{suffix}", dtype="<f4")
    if values.size != TOKENS * width:
        raise ValueError(f"ggml {mode} output has wrong geometry")
    result = values.reshape(TOKENS, width)
    if not np.isfinite(result).all():
        raise ValueError(f"ggml {mode} output has nonfinite values")
    return result


def probe(
    run_dir: Path,
    ladder_dir: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    q_run: Path,
    k_run: Path,
    v_run: Path,
    attn_run: Path,
    ffn_run: Path,
    prior_intervention: Path,
    source_build_run: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start attention stage probe through scripts/remote_job.py")
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen training prefix or candidate D differs")
    stages = {
        "q": (q_run, "q_deferred_k_norm", "Qcur-0", Q_WIDTH),
        "k": (k_run, "k_rope", "Kcur-0", KV_WIDTH),
        "v": (v_run, "v_only", "Vcur-0", KV_WIDTH),
        "attn": (attn_run, "attn_output", "kqv_out-0", Q_WIDTH),
        "ffn": (ffn_run, "ffn", "ffn_inp-0", HIDDEN),
    }
    native = {}
    capture_hashes = {}
    cuda_libraries = set()
    for stage, (path, mode, name, width) in stages.items():
        native[stage], report = _safe_rows(path, mode, name, width)
        if report["source_sha256"]["target_gguf"] != TARGET_GGUF_SHA256:
            raise ValueError(f"safe {stage} GGUF differs")
        cuda_libraries.add(report["source_sha256"]["libggml-cuda.so"])
        capture_hashes[stage] = sha256(path / "comparison.json")
    if len(cuda_libraries) != 1:
        raise ValueError("safe captures use different CUDA libraries")
    weight, hf_shard = _o_weight(target, hf_model)
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("attention stage probe requires the registered RTX 5080")
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
    native["attn"].astype("<f4").tofile(operands / "native_attn.f32")

    config = AutoConfig.from_pretrained(hf_model, local_files_only=True)
    attn_module = Qwen3Attention(config, layer_idx=0).to(device="cuda", dtype=torch.float16).eval()
    with torch.no_grad():
        attn_module.o_proj.weight.copy_(torch.from_numpy(weight).to(device="cuda"))
        q = _native_heads(native["q"], 32)
        k = _native_heads(native["k"], 8)
        v = _native_heads(native["v"], 8)
        causal = torch.zeros((1, 1, TOKENS, TOKENS), dtype=torch.float16, device="cuda")
        future = torch.triu(
            torch.ones((TOKENS, TOKENS), dtype=torch.bool, device="cuda"), diagonal=1
        )
        causal.masked_fill_(future[None, None], torch.finfo(torch.float16).min)
        torch_attn, _ = eager_attention_forward(
            attn_module,
            q,
            k,
            v,
            causal,
            scaling=attn_module.scaling,
            dropout=0.0,
            sliding_window=attn_module.sliding_window,
        )
        torch_attn_rows = torch_attn.reshape(TOKENS, Q_WIDTH).float().cpu().numpy()
        torch_attn_rows.tofile(operands / "torch_attn.f32")
        residual_half = torch.from_numpy(np.asarray(ladder[:, 0, :]).astype("<f2")).to(
            device="cuda"
        )
        native_attn_half = torch.from_numpy(native["attn"].astype("<f2")).to(device="cuda")
        torch_o_native = (
            (attn_module.o_proj(native_attn_half) + residual_half).float().cpu().numpy()
        )
        torch_o_torch = (
            (attn_module.o_proj(torch_attn.reshape(TOKENS, Q_WIDTH)) + residual_half)
            .float()
            .cpu()
            .numpy()
        )
        torch.cuda.synchronize()
    if not all(
        np.isfinite(array).all() for array in (torch_attn_rows, torch_o_native, torch_o_torch)
    ):
        raise ValueError("Torch attention/O stage returned nonfinite values")
    modes = ("qf32", "attn_only", "o_native", "o_f16cast", "o_torch")
    for mode in modes:
        subprocess.run([str(helper), str(operands), mode], check=True, timeout=120)
    ggml = {
        mode: _load(operands, mode, Q_WIDTH if mode == "attn_only" else HIDDEN) for mode in modes
    }
    prior = json.loads(prior_intervention.read_text())
    prior_metrics = prior["metrics"]["native_qkv_torch_eager_vs_native_ffn_input"]
    torch_combined = _metrics(torch_o_torch, native["ffn"])
    control_keys = ("exact_f32", "exact_f16_cast", "max_abs", "position3_relative_row_l2")
    torch_control_reproduced = all(
        torch_combined[key] == prior_metrics[key] for key in control_keys
    )
    metrics = {
        "ggml_full_vs_server_residual": _metrics(ggml["qf32"], native["ffn"]),
        "ggml_attention_only_vs_server": _metrics(ggml["attn_only"], native["attn"]),
        "ggml_o_native_vs_server_residual": _metrics(ggml["o_native"], native["ffn"]),
        "ggml_o_f16cast_vs_server_residual": _metrics(ggml["o_f16cast"], native["ffn"]),
        "ggml_o_torch_vs_server_residual": _metrics(ggml["o_torch"], native["ffn"]),
        "torch_attention_vs_server": _metrics(torch_attn_rows, native["attn"]),
        "torch_o_native_vs_server_residual": _metrics(torch_o_native, native["ffn"]),
        "torch_o_torch_vs_server_residual": torch_combined,
    }
    exact_gate = all(
        metrics[key]["exact_f32"] == metrics[key]["elements"]
        for key in (
            "ggml_full_vs_server_residual",
            "ggml_attention_only_vs_server",
            "ggml_o_native_vs_server_residual",
        )
    )
    report = {
        "schema": "target_attention_cuda_stage_split_v1",
        "status": "exact_native_stages_and_torch_control"
        if exact_gate and torch_control_reproduced
        else "stage_control_differs",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": (
            "native ggml CUDA/F32 Q, F16 K/V and mask, F32 attention/O input, "
            "F16 O weight, F32 residual; "
            "source Torch eager F16 Q/K/V, attention and O/residual"
        ),
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "kv_slots": KV_SLOTS,
        "exact_native_stage_gate": exact_gate,
        "torch_native_qkv_control_reproduced": torch_control_reproduced,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "safe_native_reports": capture_hashes,
            "prior_intervention": sha256(prior_intervention),
            "ggml_cuda_library": sha256(build / "bin/libggml-cuda.so"),
            "cuda_library_from_safe_captures": cuda_libraries.pop(),
            "hf_config": sha256(hf_model / "config.json"),
            "hf_weight_shard": sha256(hf_shard),
            "hf_qwen3_implementation": sha256(
                Path(sys.modules[Qwen3Attention.__module__].__file__)
            ),
            "native_attn_f32": sha256(operands / "native_attn.f32"),
            "torch_attn_f32": sha256(operands / "torch_attn.f32"),
            "helper_source": sha256(ROOT / "scripts/native_target_attention_cuda.cpp"),
            "helper_binary": sha256(helper),
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
    parser.add_argument("--native-q-run", type=Path, required=True)
    parser.add_argument("--native-k-run", type=Path, required=True)
    parser.add_argument("--native-v-run", type=Path, required=True)
    parser.add_argument("--native-attn-run", type=Path, required=True)
    parser.add_argument("--native-ffn-run", type=Path, required=True)
    parser.add_argument("--prior-intervention", type=Path, required=True)
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
        args.native_attn_run,
        args.native_ffn_run,
        args.prior_intervention,
        args.source_build_run,
    )
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "exact_native_stages_and_torch_control":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
