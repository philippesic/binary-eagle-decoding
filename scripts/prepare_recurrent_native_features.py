#!/usr/bin/env python3
"""Select accepted-prefix target feature rows from a native raw capture on CPU.

The native stream contains prefill, retained speculative inputs, and rejected
suffix inputs. This program joins every decoded row to one disposition event,
then exports only raw target features needed by audited training-round anchors.
It never treats a verifier label row as proof that its input feature was kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_recurrent_binary_capture import (  # noqa: E402
    FEATURE_BOUNDARY,
    FEATURE_SOURCE,
    FEATURE_TAPS,
    FEATURE_WIDTH,
    TRAIN_PROMPTS,
    TRAIN_PROMPTS_SHA256,
    read_jsonl,
    sha256,
)

RAW_SCHEMA = "eagle_target_feature_v1"
RAW_BOUNDARY = "raw_target_layer_input_before_eagle_encoder"
RAW_SOURCE = "target_verifier"


def _training_ids(prompts: Path, expected_hash: str, expected_count: int) -> set[str]:
    if sha256(prompts) != expected_hash:
        raise ValueError("training prompt SHA256 differs from frozen split")
    rows = read_jsonl(prompts)
    ids = [row.get("id") for row in rows]
    if (
        len(ids) != expected_count
        or len(set(ids)) != expected_count
        or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise ValueError(f"expected {expected_count} unique frozen training prompt IDs")
    return set(ids)


def _task_map(path: Path, allowed_ids: set[str]) -> dict[int, str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("task prompt map must be a nonempty JSON object")
    result = {}
    for key, value in payload.items():
        if (
            not isinstance(key, str)
            or not key.isdecimal()
            or not isinstance(value, str)
            or value not in allowed_ids
        ):
            raise ValueError("task prompt map contains a nontraining task")
        result[int(key)] = value
    if len(set(result.values())) != len(result):
        raise ValueError("one native task per frozen training prompt is required")
    return result


def _verify_cell_manifest(
    path: Path,
    tasks: dict[int, str],
    expected_prompt_hash: str,
    metadata_path: Path,
    values_path: Path,
) -> None:
    manifest = json.loads(path.read_text())
    expected_tasks = {str(task): prompt for task, prompt in tasks.items()}
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "binary_head_capture_cell_v1"
        or manifest.get("complete") is not True
        or manifest.get("prompts_sha256") != expected_prompt_hash
        or manifest.get("task_prompt_ids") != expected_tasks
    ):
        raise ValueError("capture cell manifest cannot prove request ownership")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != len(tasks):
        raise ValueError("capture cell request count differs from task map")
    observed = {}
    for request in requests:
        task, prompt = request.get("task_id"), request.get("id")
        if not isinstance(task, str) or task in observed or expected_tasks.get(task) != prompt:
            raise ValueError("capture cell request/task ownership mismatch")
        observed[task] = prompt
    if observed != expected_tasks:
        raise ValueError("capture cell omitted a mapped task")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("capture cell lacks raw feature file hashes")
    for source in (metadata_path, values_path):
        record = files.get(source.name)
        if (
            not isinstance(record, dict)
            or record.get("bytes") != source.stat().st_size
            or record.get("sha256") != sha256(source)
        ):
            raise ValueError("capture cell raw feature hash/size mismatch")


def prepare_native_features(
    metadata_path: Path,
    values_path: Path,
    anchors_path: Path,
    task_map_path: Path,
    train_prompts: Path,
    output_dir: Path,
    *,
    cell_manifest_path: Path | None = None,
    expected_prompt_hash: str = TRAIN_PROMPTS_SHA256,
    expected_prompt_count: int = TRAIN_PROMPTS,
    target_vocab_size: int = 151_936,
) -> dict:
    """Verify all native rows/events before writing the accepted ledger."""
    metadata_path, values_path, anchors_path = map(Path, (metadata_path, values_path, anchors_path))
    task_map_path, train_prompts, output_dir = map(Path, (task_map_path, train_prompts, output_dir))
    allowed = _training_ids(train_prompts, expected_prompt_hash, expected_prompt_count)
    tasks = _task_map(task_map_path, allowed)
    if cell_manifest_path is not None:
        cell_manifest_path = Path(cell_manifest_path)
        _verify_cell_manifest(
            cell_manifest_path, tasks, expected_prompt_hash, metadata_path, values_path
        )
    if type(target_vocab_size) is not int or target_vocab_size < 1:
        raise ValueError("target vocabulary size must be positive")
    events = read_jsonl(metadata_path)
    decoded: dict[int, dict] = {}
    dispositions: dict[int, dict] = {}
    decode_identity = set()
    for event in events:
        if event.get("schema") != RAW_SCHEMA:
            raise ValueError("unknown native feature event schema")
        feature_row = event.get("feature_row")
        if type(feature_row) is not int or feature_row < 0:
            raise ValueError("native feature_row must be nonnegative")
        kind = event.get("event")
        if kind == "decoded_row":
            if feature_row in decoded:
                raise ValueError("duplicate decoded feature row")
            task_id, slot = event.get("task_id"), event.get("slot_id")
            position, token = event.get("position"), event.get("token_id")
            prefix = event.get("prefix_token_ids")
            local, global_row = event.get("batch_row_local"), event.get("batch_row_global")
            ordinal = event.get("decode_ordinal")
            phase = event.get("phase")
            if (
                type(task_id) is not int
                or task_id not in tasks
                or type(slot) is not int
                or slot < 0
                or type(position) is not int
                or position < 0
                or type(token) is not int
                or not 0 <= token < target_vocab_size
                or not isinstance(prefix, list)
                or len(prefix) != position + 1
                or prefix[-1] != token
                or any(
                    type(value) is not int or not 0 <= value < target_vocab_size for value in prefix
                )
                or type(local) is not int
                or local < 0
                or type(global_row) is not int
                or global_row < local
                or type(ordinal) is not int
                or ordinal < 0
                or phase not in ("prefill", "target_only", "speculative")
                or event.get("target_layer_ids") != FEATURE_TAPS
                or event.get("feature_dim") != FEATURE_WIDTH
                or event.get("boundary") != RAW_BOUNDARY
                or event.get("source") != RAW_SOURCE
            ):
                raise ValueError("decoded target feature provenance, prefix or shape is invalid")
            identity = task_id, ordinal, local
            if identity in decode_identity:
                raise ValueError("duplicate task/decode/local-row identity")
            decode_identity.add(identity)
            if phase == "speculative":
                if (
                    type(event.get("round_index")) is not int
                    or type(event.get("spec_input_row")) is not int
                ):
                    raise ValueError("speculative feature row lacks round/input index")
            decoded[feature_row] = event
        elif kind == "disposition":
            if feature_row in dispositions:
                raise ValueError("duplicate feature disposition")
            dispositions[feature_row] = event
        else:
            raise ValueError("unknown native target-feature event")
    if set(decoded) != set(range(len(decoded))) or set(dispositions) != set(decoded):
        raise ValueError("native feature rows or dispositions are missing")
    row_bytes = FEATURE_WIDTH * 4
    if values_path.stat().st_size != len(decoded) * row_bytes:
        raise ValueError("raw target-feature F32 byte count differs from decoded rows")
    values = np.memmap(values_path, mode="r", dtype="<f4", shape=(len(decoded), FEATURE_WIDTH))
    if not np.isfinite(values).all():
        raise ValueError("raw target-feature values are nonfinite")
    retained = defaultdict(list)
    rejected = 0
    for index, event in decoded.items():
        status = dispositions[index]
        kept = status.get("retained_input")
        if status.get("task_id") != event["task_id"] or type(kept) is not bool:
            raise ValueError("feature disposition does not match decoded task")
        if event["phase"] == "speculative":
            accepted = status.get("accepted_drafts")
            proposal_input = event["spec_input_row"]
            if (
                type(accepted) is not int
                or accepted < 0
                or proposal_input < 0
                or status.get("round_index") != event["round_index"]
                or kept != (proposal_input <= accepted)
            ):
                raise ValueError("speculative feature retention disagrees with accepted depth")
        elif not kept:
            raise ValueError("prefill/target-only feature input was discarded")
        if kept:
            if status.get("reason") != "accepted_prefix":
                raise ValueError("retained feature needs accepted-prefix disposition")
            prompt = tasks[event["task_id"]]
            retained[(prompt, tuple(event["prefix_token_ids"]))].append(index)
        else:
            if status.get("reason") != "rejected_suffix":
                raise ValueError("rejected feature needs suffix disposition")
            rejected += 1
    anchors = read_jsonl(anchors_path)
    required = set()
    for anchor in anchors:
        prompt = anchor.get("prompt_id")
        prefix = anchor.get("prefix_token_ids")
        if (
            anchor.get("split") != "train"
            or prompt not in allowed
            or not isinstance(prefix, list)
            or not prefix
            or any(type(value) is not int for value in prefix)
        ):
            raise ValueError("round anchor is outside frozen training prefixes")
        for position in range(len(prefix)):
            key = prompt, tuple(prefix[: position + 1])
            if key not in retained:
                raise ValueError("accepted-prefix feature row is missing for round anchor")
            candidates = retained[key]
            if any(
                not np.array_equal(values[candidates[0]], values[other]) for other in candidates[1:]
            ):
                raise ValueError("duplicate accepted-prefix features disagree numerically")
            required.add(key)
    if not required:
        raise ValueError("no accepted-prefix feature rows were selected")
    ordered = sorted(required, key=lambda item: (item[0], len(item[1]), item[1]))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    array_path = output_dir / "features.npy"
    output = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float32, shape=(len(ordered), FEATURE_WIDTH)
    )
    metadata = []
    for output_index, key in enumerate(ordered):
        source_index = retained[key][0]
        source = decoded[source_index]
        output[output_index] = values[source_index]
        metadata.append(
            {
                "feature_row": output_index,
                "prompt_id": key[0],
                "position": len(key[1]) - 1,
                "prefix_token_ids": list(key[1]),
                "tap_ids": FEATURE_TAPS,
                "boundary": FEATURE_BOUNDARY,
                "source": FEATURE_SOURCE,
                "accepted_prefix": True,
                "native_feature_row": source_index,
                "native_decode_ordinal": source["decode_ordinal"],
                "task_id": source["task_id"],
            }
        )
    output.flush()
    del output
    metadata_path_out = output_dir / "feature_rows.jsonl"
    metadata_path_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata)
    )
    report = {
        "schema": "recurrent_native_feature_selection_v1",
        "execution_device": "cpu",
        "training_prompts_sha256": expected_prompt_hash,
        "sources": {
            "metadata_sha256": sha256(metadata_path),
            "values_sha256": sha256(values_path),
            "anchors_sha256": sha256(anchors_path),
            "task_map_sha256": sha256(task_map_path),
            **({"cell_manifest_sha256": sha256(cell_manifest_path)} if cell_manifest_path else {}),
        },
        "decoded_rows": len(decoded),
        "task_prompt_ownership": (
            "cell_manifest_request_and_file_hashes_verified"
            if cell_manifest_path
            else "plain_task_map_unverified_without_request_manifest"
        ),
        "rejected_suffix_rows": rejected,
        "selected_feature_rows": len(ordered),
        "features_sha256": sha256(array_path),
        "feature_rows_sha256": sha256(metadata_path_out),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-jsonl", type=Path, required=True)
    parser.add_argument("--native-f32", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--task-prompts", type=Path, required=True)
    parser.add_argument("--cell-manifest", type=Path)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_native_features(
        args.native_jsonl,
        args.native_f32,
        args.anchors,
        args.task_prompts,
        args.train_prompts,
        args.output_dir,
        cell_manifest_path=args.cell_manifest,
    )
    print(json.dumps({"selected_feature_rows": result["selected_feature_rows"]}))


if __name__ == "__main__":
    main()
