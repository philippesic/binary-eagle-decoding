#!/usr/bin/env python3
"""Intervene on block-0 HF eager attention with output-preserving native Q/K/V."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

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
from transformers import Qwen3Model  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    apply_rotary_pos_emb,
    eager_attention_forward,
)

HEAD = 128
Q_HEADS = 32
KV_HEADS = 8


def _audit_weights(target: Path, layer: torch.nn.Module) -> dict[str, str]:
    if sha256(target) != TARGET_GGUF_SHA256:
        raise ValueError("target GGUF differs from pinned source")
    reader = GGUFReader(target, mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    names = {
        "blk.0.attn_norm.weight": layer.input_layernorm.weight,
        "blk.0.attn_q.weight": layer.self_attn.q_proj.weight,
        "blk.0.attn_k.weight": layer.self_attn.k_proj.weight,
        "blk.0.attn_v.weight": layer.self_attn.v_proj.weight,
        "blk.0.attn_output.weight": layer.self_attn.o_proj.weight,
        "blk.0.attn_q_norm.weight": layer.self_attn.q_norm.weight,
        "blk.0.attn_k_norm.weight": layer.self_attn.k_norm.weight,
    }
    matches = {}
    for name, weight in names.items():
        tensor = tensors[name]
        gguf_type = GGMLQuantizationType.F16 if weight.ndim == 2 else GGMLQuantizationType.F32
        if tensor.tensor_type != gguf_type:
            raise ValueError(f"target {name} has wrong GGUF type")
        source = weight.detach().cpu().numpy()
        native = np.asarray(tensor.data)
        if native.shape != source.shape or not np.array_equal(native, source.astype(native.dtype)):
            raise ValueError(f"target {name} differs from HF source weight")
        matches[name] = gguf_type.name
    return matches


def _qkv(layer: torch.nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    attn = layer.self_attn
    q = attn.q_norm(attn.q_proj(x).reshape(1, TOKENS, Q_HEADS, HEAD)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).reshape(1, TOKENS, KV_HEADS, HEAD)).transpose(1, 2)
    v = attn.v_proj(x).reshape(1, TOKENS, KV_HEADS, HEAD).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _attention_residual(
    layer: torch.nn.Module,
    residual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    attn = layer.self_attn
    output, _ = eager_attention_forward(
        attn,
        q,
        k,
        v,
        mask,
        scaling=attn.scaling,
        dropout=0.0,
        sliding_window=attn.sliding_window,
    )
    projected = attn.o_proj(output.reshape(1, TOKENS, -1).contiguous())
    return residual + projected


def _native_heads(rows: np.ndarray, heads: int) -> torch.Tensor:
    return (
        torch.from_numpy(rows.astype("<f2"))
        .to(device="cuda")
        .reshape(1, TOKENS, heads, HEAD)
        .transpose(1, 2)
    )


def probe(
    run_dir: Path,
    ladder_dir: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    norm_run: Path,
    q_run: Path,
    k_run: Path,
    v_run: Path,
    ffn_run: Path,
    prior_q_report: Path,
    prior_v_report: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start attention intervention through scripts/remote_job.py")
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen prefix or candidate D differs")
    stages = {
        "norm": (norm_run, "attn_norm", "attn_norm-0", HIDDEN),
        "q": (q_run, "q_deferred_k_norm", "Qcur-0", Q_HEADS * HEAD),
        "k": (k_run, "k_rope", "Kcur-0", KV_HEADS * HEAD),
        "v": (v_run, "v_only", "Vcur-0", KV_HEADS * HEAD),
        "ffn_input": (ffn_run, "ffn", "ffn_inp-0", HIDDEN),
    }
    native = {}
    safe_reports = {}
    cuda_libraries = set()
    for stage, (path, mode, name, width) in stages.items():
        native[stage], report = _safe_rows(path, mode, name, width)
        if report["source_sha256"]["target_gguf"] != TARGET_GGUF_SHA256:
            raise ValueError(f"safe {stage} target GGUF differs")
        cuda_libraries.add(report["source_sha256"]["libggml-cuda.so"])
        safe_reports[stage] = sha256(path / "comparison.json")
    if len(cuda_libraries) != 1:
        raise ValueError("safe target captures use different CUDA builds")
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("attention intervention requires the registered RTX 5080")
    model = (
        Qwen3Model.from_pretrained(
            str(hf_model), local_files_only=True, dtype=torch.float16, attn_implementation="eager"
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    layer = model.layers[0]
    source_weights = _audit_weights(target, layer)
    token_tensor = torch.tensor([prefix], dtype=torch.long, device="cuda")
    captured = {}

    def attention_input(_module, args, kwargs):
        value = kwargs.get("hidden_states", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        position = kwargs.get("position_embeddings")
        if value is None or mask is None or position is None:
            raise ValueError("HF attention prehook lacks operands")
        captured["attn_input"] = value.detach().clone()
        captured["mask"] = mask.detach().clone()
        captured["cos"] = position[0].detach().clone()
        captured["sin"] = position[1].detach().clone()

    def ffn_input(_module, args):
        captured["ffn_input"] = args[0].detach().clone()

    hooks = [
        layer.self_attn.register_forward_pre_hook(attention_input, with_kwargs=True),
        layer.post_attention_layernorm.register_forward_pre_hook(ffn_input),
    ]
    try:
        with torch.no_grad():
            residual = model.embed_tokens(token_tensor)
            if not np.array_equal(residual[0].float().cpu().numpy(), np.asarray(ladder[:, 0, :])):
                raise ValueError("HF and native layer-0 input embeddings differ")
            model(input_ids=token_tensor, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {"attn_input", "mask", "cos", "sin", "ffn_input"}:
        raise ValueError("HF block-0 operand capture is incomplete")
    with torch.no_grad():
        hf_norm = layer.input_layernorm(residual)
        if not torch.equal(hf_norm, captured["attn_input"]):
            raise ValueError("HF attention input hook differs from standalone norm")
        q_hf, k_hf, v_hf = _qkv(layer, hf_norm, captured["cos"], captured["sin"])
        hf_replay = _attention_residual(layer, residual, q_hf, k_hf, v_hf, captured["mask"])
        baseline_exact = bool(torch.equal(hf_replay, captured["ffn_input"]))
        same_norm = torch.from_numpy(native["norm"].astype("<f2")).to(device="cuda")[None]
        q_source, k_source, v_source = _qkv(layer, same_norm, captured["cos"], captured["sin"])
        source_residual = _attention_residual(
            layer, residual, q_source, k_source, v_source, captured["mask"]
        )
        q_native = _native_heads(native["q"], Q_HEADS)
        k_native = _native_heads(native["k"], KV_HEADS)
        v_native = _native_heads(native["v"], KV_HEADS)
        native_qkv_residual = _attention_residual(
            layer, residual, q_native, k_native, v_native, captured["mask"]
        )
        torch.cuda.synchronize()

    def rows(tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].float().cpu().numpy().reshape(TOKENS, -1)

    prior_q = json.loads(prior_q_report.read_text())
    prior_v = json.loads(prior_v_report.read_text())
    q_metrics = _metrics(rows(q_source.transpose(1, 2)), native["q"])
    v_metrics = _metrics(rows(v_source.transpose(1, 2)), native["v"])
    q_prior = prior_q["metrics"]["torch_f16_rope_vs_server"]
    v_prior = prior_v["metrics"]["torch_f16_vs_server"]
    controls_reproduced = all(
        measured[key] == prior[key]
        for measured, prior in ((q_metrics, q_prior), (v_metrics, v_prior))
        for key in ("exact_f32", "exact_f16_cast", "max_abs", "position3_relative_row_l2")
    )
    metrics = {
        "hf_manual_vs_hf_forward": _metrics(rows(hf_replay), rows(captured["ffn_input"])),
        "hf_forward_vs_native_ffn_input": _metrics(
            rows(captured["ffn_input"]), native["ffn_input"]
        ),
        "source_qkv_on_native_norm_vs_native_ffn_input": _metrics(
            rows(source_residual), native["ffn_input"]
        ),
        "native_qkv_torch_eager_vs_native_ffn_input": _metrics(
            rows(native_qkv_residual), native["ffn_input"]
        ),
        "same_input_source_q_vs_native": q_metrics,
        "same_input_source_v_vs_native": v_metrics,
    }
    report = {
        "schema": "target_block0_attention_intervention_v1",
        "status": "hf_baseline_and_qv_controls_reproduced"
        if baseline_exact and controls_reproduced
        else "control_mismatch",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": (
            "native Q/K/V F32 cast to F16 for source Qwen3 eager attention "
            "and F16 output projection"
        ),
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "hf_manual_f16_exact": baseline_exact,
        "same_input_qv_controls_reproduced": controls_reproduced,
        "source_weight_identity": source_weights,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "hf_config": sha256(hf_model / "config.json"),
            "hf_qwen3_implementation": sha256(Path(sys.modules[Qwen3Model.__module__].__file__)),
            "prior_q_report": sha256(prior_q_report),
            "prior_v_report": sha256(prior_v_report),
            "ggml_cuda_library": cuda_libraries.pop(),
            "probe": sha256(Path(__file__)),
            "safe_native_reports": safe_reports,
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
    parser.add_argument("--native-k-run", type=Path, required=True)
    parser.add_argument("--native-v-run", type=Path, required=True)
    parser.add_argument("--native-ffn-run", type=Path, required=True)
    parser.add_argument("--prior-q-report", type=Path, required=True)
    parser.add_argument("--prior-v-report", type=Path, required=True)
    args = parser.parse_args()
    result = probe(
        ROOT / "runs" / args.run_id,
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.hf_model,
        args.native_norm_run,
        args.native_q_run,
        args.native_k_run,
        args.native_v_run,
        args.native_ffn_run,
        args.prior_q_report,
        args.prior_v_report,
    )
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "hf_baseline_and_qv_controls_reproduced":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
