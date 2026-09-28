#!/usr/bin/env python3
"""Test whether native tap-2 input explains a sealed target tap-18 outlier.

The diagnostic changes the complete input to independent HF decoder layer 2
for one frozen training prompt. It is not an exact native-backend forward or
a training tolerance decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_recurrent_binary_capture import sha256
from check_recurrent_full_target_features import (
    HIDDEN,
    TAPS,
    WIDTH,
    _source_weights_match_gguf,
    select_prefill_rows,
    validate_first_round_prefixes,
    validate_sources,
)

from w1a1_eagle.frozen_operands import FrozenOperands

PROMPT_ID = "qat-revisit-train-code-data-validation-03"
INTERVENTION_LAYER = 2
OUTLIER_POSITION = 3


def select_frozen_prompt(
    capture_root: Path,
    train_prompts: Path,
    target_gguf: Path,
    candidate_d: Path,
    prompt_id: str = PROMPT_ID,
) -> tuple[dict, dict, list[int], list[int], Path]:
    """Return only a complete owned prefill from the hash-checked 96-task capture."""
    capture, tasks, requests = validate_sources(capture_root, train_prompts)
    if sha256(target_gguf) != capture.get("target_sha256") or sha256(candidate_d) != capture.get(
        "draft_sha256"
    ):
        raise ValueError("source GGUF hashes differ from frozen capture")
    cell_path = Path(capture_root) / capture["cell_manifest_path"]
    events_path = cell_path.parent / "heads.target_features.jsonl"
    selected = select_prefill_rows(events_path, requests)
    validate_first_round_prefixes(cell_path.parent / "forced-rounds.jsonl", requests, selected)
    owners = [request for request in requests if request["id"] == prompt_id]
    if len(owners) != 1:
        raise ValueError("outlier prompt has no unique frozen training owner")
    request = owners[0]
    task = int(request["task_id"])
    if tasks[task] != prompt_id:
        raise ValueError("outlier prompt task owner differs")
    tokens, rows = selected[task]
    if len(tokens) <= OUTLIER_POSITION:
        raise ValueError("outlier position is outside the complete native prefill")
    return capture, request, tokens, rows, cell_path


def forward_with_layer_input(model, tokens: torch.Tensor, replacement: torch.Tensor):
    """Replace exactly one layer-2 input for one no-cache forward and remove hook."""
    calls = 0

    def replace(_module, args):
        nonlocal calls
        calls += 1
        if calls != 1 or not args or args[0].shape != replacement.shape:
            raise ValueError("intervention layer input shape or invocation differs")
        if args[0].device != replacement.device or args[0].dtype != replacement.dtype:
            raise ValueError("intervention tensor device or dtype differs")
        return (replacement, *args[1:])

    handle = model.layers[INTERVENTION_LAYER].register_forward_pre_hook(replace)
    try:
        result = model(input_ids=tokens, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    if calls != 1:
        raise ValueError("intervention layer was not called exactly once")
    return result.hidden_states


def compare_rows(reference: np.ndarray, native: np.ndarray) -> dict:
    """Full row errors, including the designated outlier, in F64 arithmetic."""
    if reference.shape != native.shape or reference.ndim != 2 or len(reference) <= OUTLIER_POSITION:
        raise ValueError("comparison rows do not include the complete outlier prefill")
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(native, dtype=np.float64)
    delta = expected - actual
    if not np.isfinite(delta).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite target-feature comparison")
    relative = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(actual, axis=1), 1e-12)
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    max_abs = np.max(np.abs(delta), axis=1)
    return {
        "row_count": len(relative),
        "relative_l2_mean": float(np.mean(relative)),
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_max": float(np.max(relative)),
        "relative_l2_max_position": int(np.argmax(relative)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(max_abs)),
        "rows": [
            {
                "position": position,
                "relative_l2": float(relative[position]),
                "rms": float(rms[position]),
                "max_abs": float(max_abs[position]),
            }
            for position in range(len(relative))
        ],
        "outlier_position": OUTLIER_POSITION,
        "outlier_relative_l2": float(relative[OUTLIER_POSITION]),
    }


def compare(
    capture_root: Path,
    train_prompts: Path,
    hf_model: Path,
    target_gguf: Path,
    candidate_d: Path,
    *,
    attention_implementation: str = "eager",
) -> dict:
    if attention_implementation not in {"eager", "sdpa"}:
        raise ValueError("attention implementation must be eager or sdpa")
    if not torch.cuda.is_available():
        raise ValueError("tap-2 intervention requires available CUDA")
    capture, request, prefix, row_ids, cell_path = select_frozen_prompt(
        capture_root, train_prompts, target_gguf, candidate_d
    )
    sampled = _source_weights_match_gguf(hf_model, target_gguf)
    try:
        import transformers
        from transformers import Qwen3Model
    except ImportError as error:  # pragma: no cover - remote optional dependency
        raise ValueError("install transformers for the CUDA intervention") from error
    model = (
        Qwen3Model.from_pretrained(
            str(hf_model),
            local_files_only=True,
            dtype=torch.float16,
            attn_implementation=attention_implementation,
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    operands = FrozenOperands(target_gguf, candidate_d)
    values_path = cell_path.parent / "heads.target_features.f32"
    values = np.memmap(values_path, dtype="<f4", mode="r").reshape(-1, WIDTH)
    native = np.asarray(values[row_ids], dtype=np.float64)
    input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
    with torch.no_grad():
        embedding = model.embed_tokens(input_ids)[0].cpu().numpy()
        gguf_embedding = np.stack([operands(token).to(torch.float16).numpy() for token in prefix])
        if not np.array_equal(embedding, gguf_embedding):
            raise ValueError("prompt embedding differs from pinned target GGUF")
        baseline_hidden = model(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        ).hidden_states
        replacement = torch.from_numpy(native[:, :HIDDEN].astype(np.float16)).to("cuda")[None]
        intervention_hidden = forward_with_layer_input(model, input_ids, replacement)
    measurements = {}
    for index, tap in enumerate(TAPS):
        native_tap = native[:, index * HIDDEN : (index + 1) * HIDDEN]
        baseline_tap = baseline_hidden[tap][0].float().cpu().numpy()
        # HF records hidden_states[2] before layer 2 is called, so that entry
        # does not reflect the pre-hook replacement. Measure the actual cast
        # input for tap 2 and the propagated hidden states for later taps.
        intervention_tensor = (
            replacement[0] if tap == INTERVENTION_LAYER else intervention_hidden[tap][0]
        )
        intervention_tap = intervention_tensor.float().cpu().numpy()
        measurements[str(tap)] = {
            "baseline": compare_rows(baseline_tap, native_tap),
            "intervention": compare_rows(intervention_tap, native_tap),
        }
    index_path = hf_model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    return {
        "schema": "target_tap2_intervention_cuda_v1",
        "status": "intervention_measured_causality_and_parity_unproven",
        "prompt_id": request["id"],
        "task_id": int(request["task_id"]),
        "prefill_tokens": len(prefix),
        "feature_rows": row_ids,
        "outlier_position": OUTLIER_POSITION,
        "tap_layers": list(TAPS),
        "intervention": "replace_complete_hf_decoder_layer_2_input_with_native_tap_2_f32_cast_f16",
        "replacement_dtype": "float16_from_native_float32",
        "compute_dtype": "float16_from_f16_rounded_source_weights",
        "execution_device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_implementation": attention_implementation,
        "embedding_rows_exact": True,
        "sampled_ffn_weights_exact": sampled,
        "measurements": measurements,
        "source_sha256": {
            "train_prompts": sha256(train_prompts),
            "capture_manifest": sha256(Path(capture_root) / "capture-manifest.json"),
            "task_map": capture["task_prompt_ids_sha256"],
            "cell_manifest": capture["cell_manifest_sha256"],
            "feature_events": sha256(cell_path.parent / "heads.target_features.jsonl"),
            "feature_values": sha256(values_path),
            "forced_rounds": sha256(cell_path.parent / "forced-rounds.jsonl"),
            "target_gguf": capture["target_sha256"],
            "draft_gguf": capture["draft_sha256"],
            "hf_config": sha256(hf_model / "config.json"),
            "hf_index": sha256(index_path),
            "hf_shards": {name: sha256(hf_model / name) for name in shards},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = compare(
        args.capture_root,
        args.train_prompts,
        args.hf_model,
        args.target_gguf,
        args.candidate_d,
        attention_implementation=args.attention_implementation,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "prompt_id": result["prompt_id"]}))


if __name__ == "__main__":
    main()
