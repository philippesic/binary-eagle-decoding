#!/usr/bin/env python3
"""Compare a validated native CUDA block-0 capture with HF CUDA/F16 stages."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch
from check_recurrent_full_target_features import _source_weights_match_gguf
from check_target_block0_capture import (
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

SAFE_STAGES = {
    "attn_norm-0": ("attn_norm", "attn_norm", HIDDEN),
    "Kcur_normed-0": ("k_norm", "k_norm", 1024),
    "Vcur-0": ("v_only", "v_proj", 1024),
    "ffn_inp-0": ("ffn", "ffn_inp", HIDDEN),
    "ffn_norm-0": ("ffn", "ffn_norm", HIDDEN),
    "ffn_out-0": ("ffn", "ffn_out", HIDDEN),
    "l_out-0": ("ffn", "block_output", HIDDEN),
}
D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"


def _native_rows(root: Path, entries: dict, name: str, width: int) -> np.ndarray:
    selected = entries[name]
    if not 1 <= len(selected) <= 2:
        raise ValueError(f"native {name} has wrong capture count")
    arrays = [_tensor(root, entry).reshape(TOKENS, width) for entry in selected]
    result = arrays[0]
    if any(not np.array_equal(result.view("<u4"), array.view("<u4")) for array in arrays[1:]):
        raise ValueError(f"native {name} repeated taps have different values")
    if not np.isfinite(result).all():
        raise ValueError(f"native {name} has nonfinite values")
    return result


def _metrics(hf: np.ndarray, native: np.ndarray) -> dict:
    if hf.shape != native.shape or hf.dtype != np.float32 or native.dtype != np.float32:
        raise ValueError("block-0 stage comparison requires matching F32 arrays")
    delta = hf.astype(np.float64) - native.astype(np.float64)
    reference = native.astype(np.float64)
    row_l2 = np.linalg.norm(delta, axis=1)
    row_reference = np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
    relative = row_l2 / row_reference
    return {
        "elements": hf.size,
        "exact_f32": int(np.count_nonzero(hf.view("<u4") == native.view("<u4"))),
        "exact_f16_cast": int(
            np.count_nonzero(hf.astype("<f2").view("<u2") == native.astype("<f2").view("<u2"))
        ),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "median_relative_row_l2": float(statistics.median(relative.tolist())),
        "max_relative_row_l2": float(np.max(relative)),
        "position3_relative_row_l2": float(relative[3]),
        "position3_rms": float(np.sqrt(np.mean(delta[3] ** 2))),
    }


def compare(
    ladder_dir: Path,
    target: Path,
    candidate_d: Path,
    safe_runs: dict[str, Path],
    q_run: Path,
    hf_model: Path,
) -> dict:
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    server_log = ladder_dir / "server.log"
    if "rope type             = 2" not in server_log.read_text(errors="replace"):
        raise ValueError("Qwen3 target does not use the pinned NeoX RoPE row order")
    if sha256(candidate_d) != D_SHA256:
        raise ValueError("candidate-D source differs from the frozen target capture")
    if set(safe_runs) != {"attn_norm", "k_norm", "v_only", "ffn"}:
        raise ValueError("four safe block-0 capture modes are required")
    captures = {}
    capture_hashes = {}
    for mode, run in safe_runs.items():
        report_path = run / "comparison.json"
        meta = json.loads(report_path.read_text())
        if (
            meta.get("schema") != "target_block0_native_cuda_capture_v1"
            or meta.get("capture_mode") != mode
            or meta.get("status") != "same_native_block_output"
            or meta.get("prompt_id") != PROMPT_ID
            or meta["block_output"]["exact_elements"] != TOKENS * HIDDEN
            or meta["source_sha256"]["ladder_manifest"] != LADDER_MANIFEST_SHA256
            or meta["source_sha256"]["target_gguf"] != TARGET_GGUF_SHA256
        ):
            raise ValueError(f"native {mode} capture does not preserve the server block output")
        entries, hashes = _index(run / "block0", mode=mode)
        if any(meta["source_sha256"].get(name) != digest for name, digest in hashes.items()):
            raise ValueError(f"native {mode} tensor files differ from validated report")
        captures[mode] = (run / "block0", entries)
        capture_hashes[mode] = sha256(report_path)
    q_report = q_run / "comparison.json"
    q_meta = json.loads(q_report.read_text())
    if (
        q_meta.get("capture_mode") != "q_norm"
        or q_meta.get("status") != "block_output_differs"
        or q_meta.get("prompt_id") != PROMPT_ID
        or q_meta["block_output"]["exact_elements"] != 72329
        or q_meta["source_sha256"]["ladder_manifest"] != LADDER_MANIFEST_SHA256
    ):
        raise ValueError("Q normalization perturbation control changed")
    if not torch.cuda.is_available():
        raise ValueError("HF block-0 comparison requires CUDA")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5080" not in device_name:
        raise ValueError("HF block-0 comparison requires the registered RTX 5080")
    sampled_weights = _source_weights_match_gguf(hf_model, target)
    try:
        import transformers
        from transformers import Qwen3Model
    except ImportError as error:
        raise ValueError("HF target comparison requires pinned transformers") from error
    model = (
        Qwen3Model.from_pretrained(
            str(hf_model), local_files_only=True, dtype=torch.float16, attn_implementation="eager"
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    token_tensor = torch.tensor([prefix], dtype=torch.long, device="cuda")
    hooks = []
    snapshots: dict[str, np.ndarray] = {}

    def save(name: str):
        def callback(_module, _input, output):
            value = output.detach().float().cpu().numpy()
            snapshots[name] = value.reshape(TOKENS, -1).copy()

        return callback

    def save_input(name: str):
        def callback(_module, inputs):
            value = inputs[0].detach().float().cpu().numpy()
            snapshots[name] = value.reshape(TOKENS, -1).copy()

        return callback

    layer = model.layers[0]
    hooks.extend(
        (
            layer.input_layernorm.register_forward_hook(save("attn_norm")),
            layer.self_attn.q_norm.register_forward_hook(save("q_norm")),
            layer.self_attn.k_norm.register_forward_hook(save("k_norm")),
            layer.self_attn.v_proj.register_forward_hook(save("v_proj")),
            layer.post_attention_layernorm.register_forward_pre_hook(save_input("ffn_inp")),
            layer.post_attention_layernorm.register_forward_hook(save("ffn_norm")),
            layer.mlp.register_forward_hook(save("ffn_out")),
        )
    )
    try:
        with torch.no_grad():
            embedding = model.embed_tokens(token_tensor)[0].float().cpu().numpy()
            if not np.array_equal(embedding, np.asarray(ladder[:, 0, :])):
                raise ValueError("HF embeddings differ from native layer-0 input")
            hidden = model(input_ids=token_tensor, output_hidden_states=True, use_cache=False)
            snapshots["block_output"] = hidden.hidden_states[1][0].float().cpu().numpy().copy()
        torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    if set(snapshots) != {
        "attn_norm",
        "q_norm",
        "k_norm",
        "v_proj",
        "ffn_inp",
        "ffn_norm",
        "ffn_out",
        "block_output",
    }:
        raise ValueError("HF block-0 hook set is incomplete")
    measurements = {}
    for native_name, (mode, hf_name, width) in SAFE_STAGES.items():
        hf_value = snapshots[hf_name]
        capture, entries = captures[mode]
        measurements[native_name] = _metrics(
            hf_value, _native_rows(capture, entries, native_name, width)
        )
    norm_capture, norm_entries = captures["attn_norm"]
    native_norm = _native_rows(norm_capture, norm_entries, "attn_norm-0", HIDDEN)
    k_capture, k_entries = captures["k_norm"]
    v_capture, v_entries = captures["v_only"]
    with torch.no_grad():
        same_input = torch.from_numpy(native_norm.astype("<f2")).to(device="cuda")
        k_same = layer.self_attn.k_proj(same_input).reshape(TOKENS, 8, 128)
        k_same = layer.self_attn.k_norm(k_same).float().cpu().numpy().reshape(TOKENS, 1024)
        v_same = layer.self_attn.v_proj(same_input).float().cpu().numpy().reshape(TOKENS, 1024)
        cast_input = same_input.float().cpu().numpy()
    torch.cuda.synchronize()
    same_input_metrics = {
        "native_norm_to_hf_f16_cast": _metrics(cast_input, native_norm),
        "Kcur_normed-0": _metrics(
            k_same, _native_rows(k_capture, k_entries, "Kcur_normed-0", 1024)
        ),
        "Vcur-0": _metrics(v_same, _native_rows(v_capture, v_entries, "Vcur-0", 1024)),
        "k_change_from_baseline": _metrics(k_same, snapshots["k_norm"]),
        "v_change_from_baseline": _metrics(v_same, snapshots["v_proj"]),
    }
    return {
        "schema": "target_block0_hf_cuda_comparison_v1",
        "hardware": {
            "machine": platform.machine(),
            "device": device_name,
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": "native ggml CUDA/F32 graph taps versus HF CUDA/F16 eager",
        "target_rope_layout": "NeoX half-split, no Q/K row permutation",
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "stages": measurements,
        "same_input_kv_intervention": same_input_metrics,
        "q_norm_capture_perturbation": q_meta["block_output"],
        "q_norm_hf_snapshot_available_without_safe_native_tap": True,
        "sampled_source_weight_identity": sampled_weights,
        "source_sha256": {
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "ladder_server_log": sha256(server_log),
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "safe_native_reports": capture_hashes,
            "q_perturbation_report": sha256(q_report),
            "hf_config": sha256(hf_model / "config.json"),
            "comparator": sha256(Path(__file__)),
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--attn-norm-run", type=Path, required=True)
    parser.add_argument("--k-norm-run", type=Path, required=True)
    parser.add_argument("--v-run", type=Path, required=True)
    parser.add_argument("--ffn-run", type=Path, required=True)
    parser.add_argument("--q-run", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be new")
    result = compare(
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        {
            "attn_norm": args.attn_norm_run,
            "k_norm": args.k_norm_run,
            "v_only": args.v_run,
            "ffn": args.ffn_run,
        },
        args.q_run,
        args.hf_model,
    )
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {name: value["position3_relative_row_l2"] for name, value in result["stages"].items()}
        )
    )


if __name__ == "__main__":
    main()
