#!/usr/bin/env python3
"""Compare safe native block-14 stages with baseline and native-input HF passes."""

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
    _index,
    _tensor,
    sealed_ladder,
    sha256,
)
from check_target_block0_hf import D_SHA256, _metrics  # noqa: E402
from check_target_layer_ladder import forward_with_layer_input  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from transformers import Qwen3Model  # noqa: E402

STAGES = {
    "attn_norm": ("attn_norm-14", HIDDEN),
    "pre_o": ("kqv_out-14", 4096),
    "ffn_input": ("ffn_inp-14", HIDDEN),
    "ffn_norm": ("ffn_norm-14", HIDDEN),
    "ffn_out": ("ffn_out-14", HIDDEN),
    "output": ("l_out-14", HIDDEN),
}


def _native_stages(run: Path) -> tuple[dict[str, np.ndarray], dict]:
    report_path = run / "comparison.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "target_layer14_native_cuda_capture_v1"
        or report.get("status") != "same_native_block_output"
        or report.get("capture_mode") != "block14_stages"
        or report.get("output_layer") != 14
        or report.get("prompt_id") != PROMPT_ID
        or report["block_output"]["exact_elements"] != TOKENS * HIDDEN
        or report["source_sha256"]["ladder_manifest"] != LADDER_MANIFEST_SHA256
        or report["source_sha256"]["target_gguf"] != TARGET_GGUF_SHA256
    ):
        raise ValueError("block-14 stage capture does not preserve sealed output")
    entries, hashes = _index(run / "block0", mode="block14_stages")
    if any(report["source_sha256"].get(name) != digest for name, digest in hashes.items()):
        raise ValueError("block-14 stage tensor payload differs from report")
    arrays = {}
    for stage, (name, width) in STAGES.items():
        if len(entries[name]) != 1:
            raise ValueError(f"block-14 {name} has wrong capture count")
        value = _tensor(run / "block0", entries[name][0]).reshape(TOKENS, width)
        if not np.isfinite(value).all():
            raise ValueError(f"block-14 {name} is nonfinite")
        arrays[stage] = value
    return arrays, report


def _audit_weights(target: Path, layer: torch.nn.Module) -> dict[str, str]:
    reader = GGUFReader(target, mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    names = {
        "blk.14.attn_norm.weight": layer.input_layernorm.weight,
        "blk.14.attn_q.weight": layer.self_attn.q_proj.weight,
        "blk.14.attn_k.weight": layer.self_attn.k_proj.weight,
        "blk.14.attn_v.weight": layer.self_attn.v_proj.weight,
        "blk.14.attn_output.weight": layer.self_attn.o_proj.weight,
        "blk.14.attn_q_norm.weight": layer.self_attn.q_norm.weight,
        "blk.14.attn_k_norm.weight": layer.self_attn.k_norm.weight,
        "blk.14.ffn_norm.weight": layer.post_attention_layernorm.weight,
        "blk.14.ffn_gate.weight": layer.mlp.gate_proj.weight,
        "blk.14.ffn_up.weight": layer.mlp.up_proj.weight,
        "blk.14.ffn_down.weight": layer.mlp.down_proj.weight,
    }
    audited = {}
    for name, weight in names.items():
        tensor = tensors[name]
        expected_type = GGMLQuantizationType.F16 if weight.ndim == 2 else GGMLQuantizationType.F32
        if tensor.tensor_type != expected_type:
            raise ValueError(f"block-14 {name} has wrong GGUF type")
        native = np.asarray(tensor.data)
        source = weight.detach().cpu().numpy()
        if native.shape != source.shape or not np.array_equal(native, source.astype(native.dtype)):
            raise ValueError(f"block-14 {name} differs from HF source")
        audited[name] = expected_type.name
    return audited


def probe(
    run_dir: Path,
    ladder_dir: Path,
    target: Path,
    candidate_d: Path,
    hf_model: Path,
    native_run: Path,
    prior_report: Path,
) -> dict:
    started = time.monotonic()
    if not (run_dir / "state.json").is_file():
        raise ValueError("start block-14 comparison through scripts/remote_job.py")
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    if len(prefix) != TOKENS or sha256(candidate_d) != D_SHA256:
        raise ValueError("frozen prefix or candidate D differs")
    native, capture = _native_stages(native_run)
    if not np.array_equal(native["output"].view("<u4"), np.asarray(ladder[:, 15, :]).view("<u4")):
        raise ValueError("captured block-14 output differs from sealed layer-15 input")
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise ValueError("block-14 stage comparison requires the registered RTX 5080")
    model = (
        Qwen3Model.from_pretrained(
            str(hf_model), local_files_only=True, dtype=torch.float16, attn_implementation="eager"
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    layer = model.layers[14]
    weights = _audit_weights(target, layer)
    tokens = torch.tensor([prefix], dtype=torch.long, device="cuda")
    stage_sink: dict[str, dict[str, np.ndarray]] = {"rows": {}}

    def save(name: str):
        def hook(_module, _inputs, output):
            if name in stage_sink["rows"]:
                raise ValueError(f"HF block-14 {name} hook repeated")
            stage_sink["rows"][name] = (
                output.detach().float().cpu().numpy().reshape(TOKENS, -1).copy()
            )

        return hook

    def save_input(name: str):
        def hook(_module, inputs):
            if name in stage_sink["rows"]:
                raise ValueError(f"HF block-14 {name} prehook repeated")
            stage_sink["rows"][name] = (
                inputs[0].detach().float().cpu().numpy().reshape(TOKENS, -1).copy()
            )

        return hook

    hooks = [
        layer.input_layernorm.register_forward_hook(save("attn_norm")),
        layer.self_attn.o_proj.register_forward_pre_hook(save_input("pre_o")),
        layer.post_attention_layernorm.register_forward_pre_hook(save_input("ffn_input")),
        layer.post_attention_layernorm.register_forward_hook(save("ffn_norm")),
        layer.mlp.register_forward_hook(save("ffn_out")),
    ]
    try:
        with torch.no_grad():
            embedding = model.embed_tokens(tokens)[0].float().cpu().numpy()
            if not np.array_equal(embedding, np.asarray(ladder[:, 0, :])):
                raise ValueError("HF embedding differs from pinned native layer-0 input")
            baseline_hidden = model(
                input_ids=tokens, output_hidden_states=True, use_cache=False
            ).hidden_states
            baseline = stage_sink["rows"]
            baseline["output"] = baseline_hidden[15][0].float().cpu().numpy().copy()
            baseline_input = baseline_hidden[14][0].float().cpu().numpy().copy()
            stage_sink["rows"] = {}
            replacement = torch.from_numpy(np.asarray(ladder[:, 14, :]).astype("<f2")).to(
                device="cuda"
            )[None]
            intervention_hidden = forward_with_layer_input(model, tokens, replacement, 14)
            intervention = stage_sink["rows"]
            intervention["output"] = intervention_hidden[15][0].float().cpu().numpy().copy()
            intervention_input = replacement[0].float().cpu().numpy().copy()
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    if set(baseline) != set(STAGES) or set(intervention) != set(STAGES):
        raise ValueError("HF block-14 stage hook set is incomplete")
    for stage, (_, width) in STAGES.items():
        if baseline[stage].shape != (TOKENS, width) or intervention[stage].shape != (TOKENS, width):
            raise ValueError(f"HF block-14 {stage} has wrong geometry")
    metrics = {
        "baseline_input": _metrics(baseline_input, np.asarray(ladder[:, 14, :])),
        "intervention_cast_input": _metrics(intervention_input, np.asarray(ladder[:, 14, :])),
        "baseline_vs_native": {stage: _metrics(baseline[stage], native[stage]) for stage in STAGES},
        "native_input_vs_native": {
            stage: _metrics(intervention[stage], native[stage]) for stage in STAGES
        },
        "intervention_vs_baseline": {
            stage: _metrics(intervention[stage], baseline[stage]) for stage in STAGES
        },
    }
    prior = json.loads(prior_report.read_text())
    if prior.get("schema") != "target_layer_ladder_cuda_comparison_v3":
        raise ValueError("prior block-14 intervention has wrong schema")
    prior_position3 = prior["block14_input_intervention"]["position3"]
    reproduced = all(
        measured == prior_position3[name]
        for measured, name in (
            (metrics["baseline_input"]["position3_relative_row_l2"], "baseline_input_relative_l2"),
            (
                metrics["intervention_cast_input"]["position3_relative_row_l2"],
                "cast_input_relative_l2",
            ),
            (
                metrics["baseline_vs_native"]["output"]["position3_relative_row_l2"],
                "baseline_output_relative_l2",
            ),
            (
                metrics["native_input_vs_native"]["output"]["position3_relative_row_l2"],
                "intervention_output_relative_l2",
            ),
        )
    )
    report = {
        "schema": "target_block14_native_stage_hf_intervention_v1",
        "status": "prior_block14_intervention_reproduced"
        if reproduced
        else "prior_control_differs",
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": "native ggml F32 taps vs HF CUDA/F16 eager; native input cast to F16",
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "prior_position3_reproduced": reproduced,
        "source_weight_identity": weights,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "target_gguf": TARGET_GGUF_SHA256,
            "candidate_d": D_SHA256,
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "native_block14_capture": sha256(native_run / "comparison.json"),
            "native_cuda_library": capture["source_sha256"]["libggml-cuda.so"],
            "prior_intervention": sha256(prior_report),
            "hf_config": sha256(hf_model / "config.json"),
            "hf_qwen3_implementation": sha256(Path(sys.modules[Qwen3Model.__module__].__file__)),
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
    parser.add_argument("--native-block14-run", type=Path, required=True)
    parser.add_argument("--prior-intervention", type=Path, required=True)
    args = parser.parse_args()
    result = probe(
        ROOT / "runs" / args.run_id,
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.hf_model,
        args.native_block14_run,
        args.prior_intervention,
    )
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "prior_block14_intervention_reproduced":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
