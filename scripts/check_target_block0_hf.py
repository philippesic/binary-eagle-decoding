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
    TOKENS,
    _index,
    _tensor,
    sealed_ladder,
    sha256,
)

STAGES = {
    "attn_norm-0": ("attn_norm", HIDDEN),
    "Qcur_normed-0": ("q_norm", 4096),
    "Kcur_normed-0": ("k_norm", 1024),
    "Vcur-0": ("v_proj", 1024),
    "ffn_inp-0": ("ffn_inp", HIDDEN),
    "ffn_norm-0": ("ffn_norm", HIDDEN),
    "ffn_out-0": ("ffn_out", HIDDEN),
    "l_out-0": ("block_output", HIDDEN),
}
D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"


def _native_rows(root: Path, entries: dict, name: str, width: int) -> np.ndarray:
    selected = entries[name]
    if len(selected) != 1:
        raise ValueError(f"native {name} is missing or repeated")
    value = _tensor(root, selected[0])
    if value.size != TOKENS * width:
        raise ValueError(f"native {name} has wrong width")
    result = value.reshape(TOKENS, width)
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
    capture: Path,
    native_report: Path,
    hf_model: Path,
) -> dict:
    prefix, ladder, _ = sealed_ladder(ladder_dir, target)
    if sha256(candidate_d) != D_SHA256:
        raise ValueError("candidate-D source differs from the frozen target capture")
    native_meta = json.loads(native_report.read_text())
    if (
        native_meta.get("schema") != "target_block0_native_cuda_capture_v1"
        or native_meta.get("status") != "same_native_block_output"
        or native_meta["block_output"]["exact_elements"] != TOKENS * HIDDEN
    ):
        raise ValueError("native block-0 capture is not a same-run proxy")
    entries, hashes = _index(capture)
    if any(native_meta["source_sha256"].get(name) != digest for name, digest in hashes.items()):
        raise ValueError("native block-0 tensor files differ from validated report")
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
    if set(snapshots) != {stage[0] for stage in STAGES.values()}:
        raise ValueError("HF block-0 hook set is incomplete")
    measurements = {}
    for native_name, (hf_name, width) in STAGES.items():
        hf_value = snapshots[hf_name]
        if native_name.startswith(("Qcur", "Kcur")):
            heads = width // 128
            hf_value = hf_value.reshape(TOKENS, heads, 2, 64)
            hf_value = hf_value.swapaxes(-2, -1).reshape(TOKENS, width).copy()
        measurements[native_name] = _metrics(
            hf_value, _native_rows(capture, entries, native_name, width)
        )
    return {
        "schema": "target_block0_hf_cuda_comparison_v1",
        "hardware": {
            "machine": platform.machine(),
            "device": device_name,
            "compute_capability": torch.cuda.get_device_capability(0),
        },
        "precision": "native ggml CUDA/F32 graph taps versus HF CUDA/F16 eager",
        "prompt_id": native_meta["prompt_id"],
        "prefill_tokens": TOKENS,
        "stages": measurements,
        "sampled_source_weight_identity": sampled_weights,
        "source_sha256": {
            "ladder_manifest": native_meta["source_sha256"]["ladder_manifest"],
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "native_capture_report": sha256(native_report),
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
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be new")
    result = compare(
        args.ladder_dir,
        args.target_gguf,
        args.candidate_d,
        args.capture_dir,
        args.native_report,
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
