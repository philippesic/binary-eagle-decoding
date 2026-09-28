#!/usr/bin/env python3
"""Compare an audited native target layer-input ladder with HF CUDA/F16.

The one-prompt ladder is a diagnostic only. Its existing feature taps must
reproduce the sealed 96-request source bitwise before numerical comparison.
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
)
from check_target_tap2_intervention import PROMPT_ID, select_frozen_prompt
from prepare_recurrent_native_features import RAW_BOUNDARY, RAW_SCHEMA, RAW_SOURCE

from w1a1_eagle.frozen_operands import FrozenOperands

SEALED_CAPTURE_SHA256 = "2b2f49861c010214d2e424ad49053c829acdd6c6dafccbad721406390ed888fd"
LADDER_SCHEMA = "eagle_target_layer_ladder_v1"
LADDER_BOUNDARY = "raw_target_layer_input"
LADDER_LAYERS = (*range(19), 33)
LADDER_WIDTH = len(LADDER_LAYERS) * HIDDEN
LADDER_ROW_BYTES = LADDER_WIDTH * 4
OUTLIER_POSITION = 3


def _manifest_file(root: Path, files: dict, name: str) -> Path:
    """Require a regular in-directory file with its manifest SHA and size."""
    record = files.get(name)
    path = root / name
    if (
        not isinstance(record, dict)
        or path.is_symlink()
        or not path.is_file()
        or type(record.get("bytes")) is not int
        or record["bytes"] != path.stat().st_size
        or record.get("sha256") != sha256(path)
    ):
        raise ValueError(f"diagnostic manifest file hash/size mismatch: {name}")
    return path


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"blank row in {path.name}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"nonobject row in {path.name}")
            rows.append(row)
    return rows


def _new_prefill(events: list[dict], prefix: list[int]) -> tuple[list[dict], int]:
    """Join first-run prefill rows by exact native task, position and ancestry."""
    decoded = [event for event in events if event.get("event") == "decoded_row"]
    prefill = [event for event in decoded if event.get("phase") == "prefill"]
    if len(prefill) != len(prefix) or not decoded:
        raise ValueError("new capture lacks the complete frozen prefill")
    tasks = {event.get("task_id") for event in decoded}
    if len(tasks) != 1 or type(next(iter(tasks))) is not int:
        raise ValueError("new capture has multiple or invalid native task owners")
    task = next(iter(tasks))
    for position, event in enumerate(prefill):
        if (
            event.get("schema") != RAW_SCHEMA
            or event.get("feature_row") != position
            or event.get("task_id") != task
            or event.get("position") != position
            or event.get("token_id") != prefix[position]
            or event.get("prefix_token_ids") != prefix[: position + 1]
            or event.get("target_layer_ids") != list(TAPS)
            or event.get("feature_dim") != WIDTH
            or event.get("boundary") != RAW_BOUNDARY
            or event.get("source") != RAW_SOURCE
            or any(
                type(event.get(key)) is not int or event[key] < 0
                for key in ("slot_id", "decode_ordinal", "batch_row_local", "batch_row_global")
            )
        ):
            raise ValueError(f"new feature prefill row {position} differs from frozen ancestry")
    if any(event.get("phase") == "prefill" for event in decoded[len(prefix) :]):
        raise ValueError("new prefill resumes after decoding")
    return prefill, task


def _validate_ladder_rows(
    metadata: list[dict], feature_prefill: list[dict], task: int, size: int
) -> str:
    """Check every raw F32 row offset and its matching native decode identity."""
    if len(metadata) != len(feature_prefill) or size != len(metadata) * LADDER_ROW_BYTES:
        raise ValueError("ladder row count or F32 payload size differs from complete prefill")
    target_sources: set[str] = set()
    for position, (row, event) in enumerate(zip(metadata, feature_prefill, strict=True)):
        target_source = row.get("target_source")
        if not isinstance(target_source, str) or not target_source:
            raise ValueError("ladder target source is missing")
        target_sources.add(target_source)
        if (
            row.get("schema") != LADDER_SCHEMA
            or row.get("row") != position
            or row.get("task_id") != task
            or row.get("position") != position
            or row.get("token_id") != event["token_id"]
            or row.get("layer_ids") != list(LADDER_LAYERS)
            or row.get("hidden") != HIDDEN
            or row.get("byte_offset") != position * LADDER_ROW_BYTES
            or row.get("byte_count") != LADDER_ROW_BYTES
            or row.get("dtype") != "float32_native_endian"
            or row.get("boundary") != LADDER_BOUNDARY
            or row.get("source") != RAW_SOURCE
            or any(
                row.get(key) != event[key]
                for key in ("slot_id", "decode_ordinal", "batch_row_local", "batch_row_global")
            )
        ):
            raise ValueError(f"ladder row {position} does not join the native prefill")
    if len(target_sources) != 1:
        raise ValueError("ladder target source changed during prefill")
    return target_sources.pop()


def validate_capture(
    capture_dir: Path,
    sealed_capture_root: Path,
    train_prompts: Path,
    target_gguf: Path,
    candidate_d: Path,
    binary: Path,
    cmake_cache: Path,
) -> tuple[list[int], np.ndarray, dict]:
    """Prove sealed ownership, new capture integrity and old-tap identity."""
    sealed_root = Path(sealed_capture_root)
    sealed_manifest_path = sealed_root / "capture-manifest.json"
    if sha256(sealed_manifest_path) != SEALED_CAPTURE_SHA256:
        raise ValueError("full96 capture manifest differs from sealed source")
    sealed, request, prefix, rows, sealed_cell = select_frozen_prompt(
        sealed_root, train_prompts, target_gguf, candidate_d
    )
    root = Path(capture_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get("source_sha256")
    files = manifest.get("files")
    if (
        manifest.get("schema") != "recurrent_cuda_native_diagnostic_v1"
        or manifest.get("execution_device") != "cuda"
        or manifest.get("prompt_id") != PROMPT_ID
        or manifest.get("training_eligible") is not False
        or not isinstance(sources, dict)
        or not isinstance(files, dict)
        or sources.get("target") != sealed.get("target_sha256")
        or sources.get("draft") != sealed.get("draft_sha256")
        or sources.get("frozen_train_prompts") != sealed.get("train_prompts_sha256")
        or sources.get("binary") != sha256(binary)
        or sources.get("cmake_cache") != sha256(cmake_cache)
    ):
        raise ValueError("one-prompt diagnostic manifest or source hashes differ")
    ladder_meta = _manifest_file(root, files, "heads.target_layer_ladder.jsonl")
    ladder_values_path = _manifest_file(root, files, "heads.target_layer_ladder.f32")
    feature_meta = _manifest_file(root, files, "heads.target_features.jsonl")
    feature_values_path = _manifest_file(root, files, "heads.target_features.f32")
    rounds_path = _manifest_file(root, files, "forced-rounds.jsonl")
    rounds = _jsonl(rounds_path)
    if not rounds or rounds[0].get("prefix_token_ids") != prefix:
        raise ValueError("new first native round differs from frozen prefill")
    events = _jsonl(feature_meta)
    feature_prefill, task = _new_prefill(events, prefix)
    if rounds[0].get("task_id") != task:
        raise ValueError("new native task does not own the first round")
    ladder_rows = _jsonl(ladder_meta)
    target_source = _validate_ladder_rows(
        ladder_rows, feature_prefill, task, ladder_values_path.stat().st_size
    )
    counts = manifest.get("capture_counts")
    if not isinstance(counts, dict) or counts.get("target_ladder_rows") != len(ladder_rows):
        raise ValueError("new target-ladder count differs from diagnostic manifest")
    if Path(target_source).resolve() != Path(target_gguf).resolve():
        raise ValueError("ladder target source path differs from pinned target GGUF")
    decoded = sum(event.get("event") == "decoded_row" for event in events)
    if feature_values_path.stat().st_size != decoded * WIDTH * 4:
        raise ValueError("new target-feature values file has wrong row count")
    if counts.get("target_feature_rows") != decoded:
        raise ValueError("new target-feature count differs from diagnostic manifest")
    ladder = np.memmap(ladder_values_path, dtype="=f4", mode="r", shape=(len(prefix), LADDER_WIDTH))
    feature = np.memmap(feature_values_path, dtype="=f4", mode="r", shape=(decoded, WIDTH))
    sealed_values = np.memmap(
        sealed_cell.parent / "heads.target_features.f32", dtype="=f4", mode="r"
    ).reshape(-1, WIDTH)
    for index, layer in enumerate(TAPS):
        ladder_tap = ladder[:, LADDER_LAYERS.index(layer) * HIDDEN :]
        ladder_tap = ladder_tap[:, :HIDDEN]
        new_tap = feature[: len(prefix), index * HIDDEN : (index + 1) * HIDDEN]
        old_tap = sealed_values[rows, index * HIDDEN : (index + 1) * HIDDEN]
        if not np.array_equal(ladder_tap.view(np.uint32), new_tap.view(np.uint32)):
            raise ValueError(f"ladder tap {layer} differs bitwise from same-run feature capture")
        if not np.array_equal(new_tap.view(np.uint32), old_tap.view(np.uint32)):
            raise ValueError(f"new tap {layer} differs bitwise from sealed full96 source")
    if not np.isfinite(ladder).all():
        raise ValueError("nonfinite native ladder payload")
    return (
        prefix,
        np.asarray(ladder, dtype=np.float64),
        {
            "prompt_id": request["id"],
            "task_id": task,
            "feature_rows_sealed": rows,
            "target_source": target_source,
            "source_sha256": {
                "train_prompts": sha256(train_prompts),
                "sealed_capture_manifest": sha256(sealed_manifest_path),
                "sealed_cell_manifest": sealed["cell_manifest_sha256"],
                "sealed_feature_values": sha256(sealed_cell.parent / "heads.target_features.f32"),
                "new_manifest": sha256(manifest_path),
                "new_binary": sources["binary"],
                "new_cmake_cache": sources["cmake_cache"],
                "new_ladder_metadata": sha256(ladder_meta),
                "new_ladder_values": sha256(ladder_values_path),
                "new_feature_events": sha256(feature_meta),
                "new_feature_values": sha256(feature_values_path),
                "new_forced_rounds": sha256(rounds_path),
                "target_gguf": sealed["target_sha256"],
                "draft_gguf": sealed["draft_sha256"],
            },
        },
    )


def layer_metrics(reference: np.ndarray, native: np.ndarray) -> dict:
    """Per-position F64 relative L2, RMS and max-absolute layer-input error."""
    if reference.shape != native.shape or reference.ndim != 2 or reference.shape[1] != HIDDEN:
        raise ValueError("layer comparison has wrong hidden shape")
    expected, actual = np.asarray(reference, dtype=np.float64), np.asarray(native, dtype=np.float64)
    delta = expected - actual
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite layer comparison")
    relative = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(actual, axis=1), 1e-12)
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    max_abs = np.max(np.abs(delta), axis=1)
    return {
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_max": float(np.max(relative)),
        "relative_l2_max_position": int(np.argmax(relative)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(max_abs)),
        "positions": [
            {
                "position": index,
                "relative_l2": float(relative[index]),
                "rms": float(rms[index]),
                "max_abs": float(max_abs[index]),
            }
            for index in range(len(relative))
        ],
    }


def first_sharp_growth(layers: dict[str, dict]) -> dict | None:
    """Find the first largest adjacent absolute rise at position 3, with no gate."""
    largest = None
    for previous, current in zip(LADDER_LAYERS, LADDER_LAYERS[1:], strict=False):
        if current != previous + 1:
            continue
        before = layers[str(previous)]["positions"][OUTLIER_POSITION]["relative_l2"]
        after = layers[str(current)]["positions"][OUTLIER_POSITION]["relative_l2"]
        increase = after - before
        if increase > 0 and (largest is None or increase > largest["absolute_increase"]):
            largest = {
                "from_layer": previous,
                "to_layer": current,
                "from_relative_l2": float(before),
                "to_relative_l2": float(after),
                "absolute_increase": float(increase),
                "ratio": float(after / max(before, 1e-12)),
            }
    return largest


def compare(
    capture_dir: Path,
    sealed_capture_root: Path,
    train_prompts: Path,
    hf_model: Path,
    target_gguf: Path,
    candidate_d: Path,
    binary: Path,
    cmake_cache: Path,
) -> dict:
    if not torch.cuda.is_available():
        raise ValueError("target ladder comparison requires available CUDA")
    prefix, native, provenance = validate_capture(
        capture_dir,
        sealed_capture_root,
        train_prompts,
        target_gguf,
        candidate_d,
        binary,
        cmake_cache,
    )
    if len(prefix) <= OUTLIER_POSITION:
        raise ValueError("frozen prefill omits designated outlier position")
    sampled = _source_weights_match_gguf(hf_model, target_gguf)
    try:
        import transformers
        from transformers import Qwen3Model
    except ImportError as error:  # pragma: no cover - remote optional dependency
        raise ValueError("install transformers for CUDA target ladder comparison") from error
    model = (
        Qwen3Model.from_pretrained(
            str(hf_model), local_files_only=True, dtype=torch.float16, attn_implementation="eager"
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    operands = FrozenOperands(target_gguf, candidate_d)
    tokens = torch.tensor([prefix], dtype=torch.long, device="cuda")
    with torch.no_grad():
        embedding = model.embed_tokens(tokens)[0].cpu().numpy()
        gguf_embedding = np.stack([operands(token).to(torch.float16).numpy() for token in prefix])
        if not np.array_equal(embedding, gguf_embedding):
            raise ValueError("prompt embedding differs from pinned target GGUF")
        hidden = model(input_ids=tokens, output_hidden_states=True, use_cache=False).hidden_states
    measurements = {}
    for index, layer in enumerate(LADDER_LAYERS):
        reference = hidden[layer][0].float().cpu().numpy()
        actual = native[:, index * HIDDEN : (index + 1) * HIDDEN]
        measurements[str(layer)] = layer_metrics(reference, actual)
    index_path = hf_model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    provenance["source_sha256"].update(
        hf_config=sha256(hf_model / "config.json"),
        hf_index=sha256(index_path),
        hf_shards={name: sha256(hf_model / name) for name in shards},
    )
    return {
        "schema": "target_layer_ladder_cuda_comparison_v1",
        "status": "independent_f16_layer_inputs_measured_parity_unproven",
        **provenance,
        "prefill_tokens": len(prefix),
        "outlier_position": OUTLIER_POSITION,
        "layer_ids": list(LADDER_LAYERS),
        "execution_device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "compute_dtype": "float16_from_f16_rounded_source_weights",
        "attention_implementation": "eager",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "embedding_rows_exact": True,
        "sampled_ffn_weights_exact": sampled,
        "old_taps_bitwise_equal_to_same_run_and_sealed": True,
        "layers": measurements,
        "first_largest_position3_adjacent_growth": first_sharp_growth(measurements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sealed-capture-root", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = compare(
        args.capture_dir,
        args.sealed_capture_root,
        args.train_prompts,
        args.hf_model,
        args.target_gguf,
        args.candidate_d,
        args.binary,
        args.cmake_cache,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "prompt_id": result["prompt_id"]}))


if __name__ == "__main__":
    main()
