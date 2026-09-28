#!/usr/bin/env python3
"""Compare all frozen training prefill target features with local HF CUDA F16.

This is an independent numerical diagnostic, not exact llama.cpp parity or a
training-eligibility decision. Raw feature metadata is streamed, and F32 values
remain memory mapped. No development or final prompts are accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_recurrent_binary_capture import TRAIN_PROMPTS, TRAIN_PROMPTS_SHA256, sha256  # noqa: E402
from check_recurrent_target_features import (  # noqa: E402
    HIDDEN,
    TAPS,
    WIDTH,
    _source_weights_match_gguf,
)
from prepare_recurrent_native_features import (  # noqa: E402
    RAW_BOUNDARY,
    RAW_SCHEMA,
    RAW_SOURCE,
    _task_map,
    _training_ids,
    _verify_cell_manifest,
)

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402


def _owned_file(root: Path, name: object) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("capture manifest contains an unsafe file path")
    return root / name


def validate_sources(
    capture_root: Path, train_prompts: Path
) -> tuple[dict, dict[int, str], list[dict]]:
    """Verify frozen split, all 96 task owners, and sealed feature source hashes."""
    root = Path(capture_root)
    ids = _training_ids(Path(train_prompts), TRAIN_PROMPTS_SHA256, TRAIN_PROMPTS)
    manifest_path = root / "capture-manifest.json"
    capture = json.loads(manifest_path.read_text())
    if (
        not isinstance(capture, dict)
        or capture.get("schema") != "recurrent_binary_native_capture_v1"
        or capture.get("split") != "train"
        or capture.get("trajectory") != "own_history"
        or capture.get("body") != "D"
        or capture.get("head") != "D"
        or capture.get("complete") is not True
        or capture.get("train_prompts_sha256") != TRAIN_PROMPTS_SHA256
    ):
        raise ValueError("capture manifest is not the complete frozen training capture")
    map_path = _owned_file(root, capture.get("task_prompt_ids_path"))
    cell_path = _owned_file(root, capture.get("cell_manifest_path"))
    if sha256(map_path) != capture.get("task_prompt_ids_sha256"):
        raise ValueError("capture task map SHA256 mismatch")
    if sha256(cell_path) != capture.get("cell_manifest_sha256"):
        raise ValueError("capture cell manifest SHA256 mismatch")
    tasks = _task_map(map_path, ids)
    if len(tasks) != TRAIN_PROMPTS or set(tasks.values()) != ids:
        raise ValueError("capture does not own all 96 frozen training prompts")
    metadata = cell_path.parent / "heads.target_features.jsonl"
    values = cell_path.parent / "heads.target_features.f32"
    _verify_cell_manifest(cell_path, tasks, TRAIN_PROMPTS_SHA256, metadata, values)
    cell = json.loads(cell_path.read_text())
    raw = capture.get("raw_files")
    if not isinstance(raw, dict):
        raise ValueError("capture manifest lacks raw feature records")
    for path in (metadata, values, cell_path.parent / "forced-rounds.jsonl"):
        record = raw.get(path.name)
        cell_record = cell["files"].get(path.name)
        if (
            not isinstance(record, dict)
            or not isinstance(cell_record, dict)
            or _owned_file(root, record.get("path")).resolve() != path.resolve()
            or record.get("bytes") != cell_record["bytes"]
            or record.get("sha256") != cell_record["sha256"]
            or (path.name == "forced-rounds.jsonl" and sha256(path) != cell_record["sha256"])
        ):
            raise ValueError("capture and cell feature hash records differ")
    requests = cell.get("requests")
    if (
        not isinstance(requests, list)
        or len(requests) != TRAIN_PROMPTS
        or capture.get("requests") != requests
    ):
        raise ValueError("capture and cell request records differ")
    feature_cursor = event_cursor = 0
    for request in requests:
        if not isinstance(request, dict) or tasks.get(
            int(request.get("task_id", "-1"))
        ) != request.get("id"):
            raise ValueError("request task/prompt ownership differs from frozen task map")
        for field, cursor in (
            ("target_feature_rows", feature_cursor),
            ("target_feature_event_rows", event_cursor),
        ):
            span = request.get(field)
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or span[0] != cursor
                or span[1] <= cursor
            ):
                raise ValueError(f"request {field} is not contiguous")
        feature_cursor = request["target_feature_rows"][1]
        event_cursor = request["target_feature_event_rows"][1]
    if values.stat().st_size != feature_cursor * WIDTH * 4:
        raise ValueError("raw feature F32 size differs from request ranges")
    return capture, tasks, requests


def select_prefill_rows(
    metadata_path: Path, requests: list[dict]
) -> dict[int, tuple[list[int], list[int]]]:
    """Stream decoded events and select complete ordered prefill for each request."""
    selected: dict[int, tuple[list[int], list[int]]] = {}
    decoded_index = event_index = request_index = 0
    saw_nonprefill: set[int] = set()
    dispositions: set[int] = set()
    batches: dict[int, tuple[int, int]] = {}
    task_slots: dict[int, int] = {}
    with Path(metadata_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("schema") != RAW_SCHEMA:
                raise ValueError("invalid native target-feature event schema")
            while event_index == requests[request_index]["target_feature_event_rows"][1]:
                request_index += 1
                if request_index == len(requests):
                    raise ValueError("native feature metadata exceeds request event ranges")
            request = requests[request_index]
            task = int(request["task_id"])
            if event.get("task_id") != task:
                raise ValueError("feature event belongs to another request task")
            kind = event.get("event")
            if kind == "decoded_row":
                if event.get("feature_row") != decoded_index or not (
                    request["target_feature_rows"][0]
                    <= decoded_index
                    < request["target_feature_rows"][1]
                ):
                    raise ValueError("decoded feature row is missing or outside request")
                if (
                    event.get("target_layer_ids") != list(TAPS)
                    or event.get("feature_dim") != WIDTH
                    or event.get("boundary") != RAW_BOUNDARY
                    or event.get("source") != RAW_SOURCE
                ):
                    raise ValueError("native target-feature tap order or provenance differs")
                ordinal = event.get("decode_ordinal")
                local = event.get("batch_row_local")
                global_row = event.get("batch_row_global")
                slot = event.get("slot_id")
                if (
                    type(ordinal) is not int
                    or ordinal < 0
                    or type(local) is not int
                    or local < 0
                    or type(global_row) is not int
                    or global_row < local
                    or type(slot) is not int
                    or slot < 0
                ):
                    raise ValueError("native decode batch row or slot is invalid")
                next_local, base = batches.get(ordinal, (0, global_row - local))
                if local != next_local or global_row - local != base:
                    raise ValueError("native feature row is outside its decode batch")
                batches[ordinal] = (next_local + 1, base)
                if task in task_slots and task_slots[task] != slot:
                    raise ValueError("native task changed sequence slot during capture")
                task_slots[task] = slot
                phase = event.get("phase")
                if phase == "prefill":
                    if task in saw_nonprefill:
                        raise ValueError("prefill resumes after nonprefill events")
                    tokens, rows = selected.setdefault(task, ([], []))
                    position = len(tokens)
                    token = event.get("token_id")
                    prefix = event.get("prefix_token_ids")
                    if (
                        type(token) is not int
                        or event.get("position") != position
                        or not isinstance(prefix, list)
                        or prefix != tokens + [token]
                    ):
                        raise ValueError("prefill position, token ancestry or row order differs")
                    tokens.append(token)
                    rows.append(decoded_index)
                elif phase in {"target_only", "speculative"}:
                    saw_nonprefill.add(task)
                else:
                    raise ValueError("unknown target-feature phase")
                decoded_index += 1
            elif kind == "disposition":
                if type(event.get("feature_row")) is not int or not (
                    request["target_feature_rows"][0]
                    <= event["feature_row"]
                    < request["target_feature_rows"][1]
                ):
                    raise ValueError("feature disposition row is outside request")
                if event["feature_row"] >= decoded_index or event["feature_row"] in dispositions:
                    raise ValueError("feature disposition is missing or duplicated")
                dispositions.add(event["feature_row"])
            else:
                raise ValueError("unknown native target-feature event")
            event_index += 1
    if (
        event_index != requests[-1]["target_feature_event_rows"][1]
        or decoded_index != requests[-1]["target_feature_rows"][1]
        or dispositions != set(range(decoded_index))
        or len(selected) != len(requests)
        or any(not tokens for tokens, _ in selected.values())
    ):
        raise ValueError("complete prefill rows are missing from native feature stream")
    return selected


def validate_first_round_prefixes(rounds_path: Path, requests: list[dict], selected: dict) -> None:
    """Require every selected prefill to equal its task's complete first round prefix."""
    first = {}
    with Path(rounds_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task_id")
            if type(task) is not int or task not in selected:
                raise ValueError("native first-round task is outside frozen request map")
            first.setdefault(task, row.get("prefix_token_ids"))
    if len(first) != len(requests):
        raise ValueError("native first round is missing for a frozen request")
    for request in requests:
        task = int(request["task_id"])
        if first[task] != selected[task][0]:
            raise ValueError("complete prefill differs from first native round prefix")


def _stats(relative: np.ndarray, delta: np.ndarray) -> dict:
    if relative.size == 0 or not np.isfinite(relative).all() or not np.isfinite(delta).all():
        raise ValueError("nonfinite target-feature comparison")
    return {
        "rows": int(relative.size),
        "relative_l2_mean": float(np.mean(relative)),
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_p90": float(np.quantile(relative, 0.90)),
        "relative_l2_p95": float(np.quantile(relative, 0.95)),
        "relative_l2_p99": float(np.quantile(relative, 0.99)),
        "relative_l2_max": float(np.max(relative)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def compare_full(
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
        raise ValueError("full target-feature diagnostic requires available CUDA")
    capture, tasks, requests = validate_sources(capture_root, train_prompts)
    if sha256(target_gguf) != capture.get("target_sha256") or sha256(candidate_d) != capture.get(
        "draft_sha256"
    ):
        raise ValueError("source GGUF hashes differ from frozen capture")
    cell_path = Path(capture_root) / capture["cell_manifest_path"]
    selected = select_prefill_rows(cell_path.parent / "heads.target_features.jsonl", requests)
    validate_first_round_prefixes(cell_path.parent / "forced-rounds.jsonl", requests, selected)
    sampled = _source_weights_match_gguf(hf_model, target_gguf)
    try:
        import transformers
        from transformers import Qwen3Model
    except ImportError as error:  # pragma: no cover - remote optional dependency
        raise ValueError("install transformers for the CUDA target-feature diagnostic") from error
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
    all_relative: dict[int, list[np.ndarray]] = defaultdict(list)
    all_delta: dict[int, list[np.ndarray]] = defaultdict(list)
    prompts = []
    for request in requests:
        task = int(request["task_id"])
        tokens, row_ids = selected[task]
        token_tensor = torch.tensor([tokens], dtype=torch.long, device="cuda")
        with torch.no_grad():
            embedding = model.embed_tokens(token_tensor)[0].cpu().numpy()
            gguf_embedding = np.stack(
                [operands(token).to(torch.float16).numpy() for token in tokens]
            )
            if not np.array_equal(embedding, gguf_embedding):
                raise ValueError(
                    f"prompt {request['id']} embedding differs from pinned target GGUF"
                )
            hidden = model(
                input_ids=token_tensor, output_hidden_states=True, use_cache=False
            ).hidden_states
            per_tap = {}
            native = np.asarray(values[row_ids], dtype=np.float64)
            for index, layer in enumerate(TAPS):
                start = index * HIDDEN
                actual = native[:, start : start + HIDDEN]
                expected = hidden[layer][0].float().cpu().numpy().astype(np.float64)
                delta = expected - actual
                relative = np.linalg.norm(delta, axis=1) / np.maximum(
                    np.linalg.norm(actual, axis=1), 1e-12
                )
                per_tap[str(layer)] = {
                    **_stats(relative, delta),
                    "max_relative_l2_position": int(np.argmax(relative)),
                }
                all_relative[layer].append(relative)
                all_delta[layer].append(delta)
        prompts.append(
            {
                "prompt_id": tasks[task],
                "task_id": task,
                "prefill_tokens": len(tokens),
                "feature_rows": row_ids,
                "taps": per_tap,
            }
        )
    index_path = hf_model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    return {
        "schema": "recurrent_full_target_feature_hf_cuda_comparison_v1",
        "status": "independent_cuda_f16_prefill_feature_drift_measured_parity_unproven",
        "execution_device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "compute_dtype": "float16_from_f16_rounded_source_weights",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_implementation": attention_implementation,
        "prompt_count": len(prompts),
        "prefill_rows": sum(item["prefill_tokens"] for item in prompts),
        "tap_layers": list(TAPS),
        "embedding_rows_exact": True,
        "sampled_ffn_weights_exact": sampled,
        "aggregate_taps": {
            str(layer): _stats(
                np.concatenate(all_relative[layer]), np.concatenate(all_delta[layer])
            )
            for layer in TAPS
        },
        "prompts": prompts,
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
    result = compare_full(
        args.capture_root,
        args.train_prompts,
        args.hf_model,
        args.target_gguf,
        args.candidate_d,
        attention_implementation=args.attention_implementation,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "prompts": result["prompt_count"],
                "rows": result["prefill_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
