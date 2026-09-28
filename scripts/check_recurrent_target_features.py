#!/usr/bin/env python3
"""Compare native target-feature taps with an independent CPU Qwen3 forward.

The local BF16 source weights are rounded to F16; sampled tensors are checked
against the pinned target GGUF before F32 CPU computation. This is a numeric
diagnostic,
not a proof of exact llama.cpp target parity or an accelerator result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from gguf import GGUFReader  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402

TAPS = (2, 18, 33)
HIDDEN = 2560
WIDTH = len(TAPS) * HIDDEN


def _source_weights_match_gguf(hf_model: Path, target_gguf: Path) -> dict[str, bool]:
    index = json.loads((hf_model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    reader = GGUFReader(target_gguf, mode="r")
    gguf_tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    matches = {}
    for layer in TAPS:
        source_name = f"model.layers.{layer}.mlp.gate_proj.weight"
        target_name = f"blk.{layer}.ffn_gate.weight"
        with safe_open(hf_model / weight_map[source_name], framework="pt", device="cpu") as file:
            source = file.get_tensor(source_name).to(torch.float16).numpy()
        matches[target_name] = bool(np.array_equal(source, gguf_tensors[target_name]))
    if not all(matches.values()):
        raise ValueError("sampled F16-rounded source tensors differ from pinned GGUF")
    return matches


def _capture_metrics(
    capture_dir: Path, prefix: list[int], reference: np.ndarray
) -> tuple[dict, dict[str, str]]:
    metadata_path = capture_dir / "heads.target_features.jsonl"
    values_path = capture_dir / "heads.target_features.f32"
    decoded = [row for row in read_jsonl(metadata_path) if row.get("event") == "decoded_row"]
    prefill = [row for row in decoded if row.get("phase") == "prefill"]
    if len(prefill) != len(prefix) or any(
        row.get("feature_row") != position
        or row.get("position") != position
        or row.get("token_id") != prefix[position]
        or row.get("prefix_token_ids") != prefix[: position + 1]
        or row.get("target_layer_ids") != list(TAPS)
        for position, row in enumerate(prefill)
    ):
        raise ValueError("capture prefill ancestry or target tap order differs")
    values = np.memmap(values_path, dtype="<f4", mode="r")
    if values.size != len(decoded) * WIDTH:
        raise ValueError("native target-feature payload size differs from metadata")
    native = values.reshape(-1, WIDTH)[: len(prefix)].astype(np.float64)
    comparison = {}
    for index, layer in enumerate(TAPS):
        start, end = index * HIDDEN, (index + 1) * HIDDEN
        expected = reference[:, start:end].astype(np.float64)
        actual = native[:, start:end]
        delta = expected - actual
        abs_delta = np.abs(delta)
        native_norm = np.linalg.norm(actual, axis=1)
        relative_row = np.linalg.norm(delta, axis=1) / np.maximum(native_norm, 1e-12)
        row_rms = np.sqrt(np.mean(delta * delta, axis=1))
        comparison[str(layer)] = {
            "max_abs": float(abs_delta.max()),
            "rms": float(np.sqrt(np.mean(delta * delta))),
            "row_zero_rms": float(row_rms[0]),
            "row_zero_native_rms": float(np.sqrt(np.mean(actual[0] * actual[0]))),
            "row_zero_relative_l2": float(relative_row[0]),
            "max_relative_row_l2": float(relative_row.max()),
            "median_relative_row_l2": float(np.median(relative_row)),
            "nonzero_position_rms": float(np.sqrt(np.mean(delta[1:] * delta[1:]))),
        }
    return comparison, {"events": sha256(metadata_path), "values": sha256(values_path)}


def compare(
    hf_model: Path,
    target_gguf: Path,
    candidate_d: Path,
    capture_dirs: list[Path],
    *,
    threads: int = 8,
) -> dict:
    try:
        import transformers
        from transformers import Qwen3Model
    except ImportError as error:  # pragma: no cover - optional local diagnostic dependency
        raise ValueError("install transformers==4.57.1 for this CPU diagnostic") from error
    if not capture_dirs or type(threads) is not int or threads < 1:
        raise ValueError("one or more captures and positive CPU thread count are required")
    torch.set_num_threads(threads)
    first_round = read_jsonl(capture_dirs[0] / "forced-rounds.jsonl")[0]
    prefix = first_round.get("prefix_token_ids")
    if not isinstance(prefix, list) or len(prefix) < 2:
        raise ValueError("first captured native prompt prefix is missing")
    operands = FrozenOperands(target_gguf, candidate_d)
    sampled_weights = _source_weights_match_gguf(hf_model, target_gguf)
    model = Qwen3Model.from_pretrained(
        str(hf_model), local_files_only=True, dtype=torch.float16, attn_implementation="eager"
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        tokens = torch.tensor([prefix], dtype=torch.long, device="cpu")
        embedding = model.embed_tokens(tokens)[0].cpu().numpy()
        gguf_embedding = np.stack([operands(int(token)).float().numpy() for token in prefix])
        if not np.array_equal(embedding, gguf_embedding):
            raise ValueError("F16-rounded HF prompt embedding differs from target GGUF")
        hidden = model(input_ids=tokens, output_hidden_states=True, use_cache=False).hidden_states
        reference = np.concatenate([hidden[layer][0].cpu().numpy() for layer in TAPS], axis=1)
    results = {}
    sources = {}
    for directory in capture_dirs:
        key = directory.name
        rounds = read_jsonl(directory / "forced-rounds.jsonl")
        if rounds[0].get("prefix_token_ids") != prefix:
            raise ValueError("capture prompt prefixes differ")
        results[key], sources[key] = _capture_metrics(directory, prefix, reference)
    config_path = hf_model / "config.json"
    index_path = hf_model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    return {
        "schema": "recurrent_target_feature_hf_cpu_comparison_v1",
        "status": "independent_cpu_feature_drift_measured_parity_unproven",
        "execution_device": "cpu",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "compute_dtype": "float32_from_f16_rounded_source_weights",
        "prompt_tokens": len(prefix),
        "tap_layers": list(TAPS),
        "embedding_rows_exact": True,
        "sampled_ffn_weights_exact": sampled_weights,
        "captures": results,
        "source_sha256": {
            "target_gguf": sha256(target_gguf),
            "draft_gguf": sha256(candidate_d),
            "hf_config": sha256(config_path),
            "hf_index": sha256(index_path),
            "hf_shards": {name: sha256(hf_model / name) for name in shards},
            "native_capture": sources,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = compare(
        args.hf_model,
        args.target_gguf,
        args.candidate_d,
        args.capture_dir,
        threads=args.threads,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "captures": list(report["captures"])}))


if __name__ == "__main__":
    main()
