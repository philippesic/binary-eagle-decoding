#!/usr/bin/env python3
"""Prepare strict CPU-only recurrent rows from an unforced native EAGLE capture.

The task map is a plain JSON object mapping native task IDs (as strings) to
frozen train prompt IDs. An optional cell manifest from
``run_binary_head_capture.py`` checks its request ranges against *every* native
head and round row. Without it, prompt ownership cannot be proven from native
rows alone and is explicitly reported as unverified. The D map input contains
absolute target IDs; native inverse files are derived from it. No model runs.

This creates proposal rows and anchors, not a complete training capture: raw
accepted-prefix target features and fixed-prefix numeric parity are still
required. Full draft-head logits are never used as target verifier logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import resolve_prompt_expectation  # noqa: E402

from w1a1_eagle.recurrent_trace import (  # noqa: E402
    LABEL_SOURCE,
    VERIFIER_LOGITS_SOURCE,
    RoundAnchor,
    validate_offset_d2t,
    validate_recurrent_trace,
)

TRAIN_PROMPTS_SHA256 = "80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74"
TRAIN_PROMPTS = 96
MAX_DEPTH = 5
HEAD_SCHEMA = "eagle_head_state_v1"
ROUND_SCHEMA = "eagle_forced_round_v1"  # Schema name also used for unforced recorded rounds.
HEAD_BOUNDARY = "native_output_norm_f32_before_head_operand_conversion"
DRAFT_LOGITS_SOURCE = "draft_head_mapped_target_vocabulary_before_sampler"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    # JSONL records end at newlines; Unicode separators can occur inside strings.
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: expected nonempty JSON object rows")
    return rows


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def token_ids(value: object, name: str, vocabulary: int, *, nonempty: bool = True) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a token list")
    if any(type(token) is not int or token < 0 or token >= vocabulary for token in value):
        raise ValueError(f"{name} contains an invalid target token")
    return value


def _file_record(manifest: dict, name: str, path: Path) -> None:
    record = manifest.get("files", {}).get(path.name)
    if record is not None and (
        record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256(path)
    ):
        raise ValueError(f"cell manifest {name} file hash/size mismatch")


def _task_map(path: Path, prompt_ids: set[str]) -> dict[str, str]:
    joins = json.loads(path.read_text())
    if not isinstance(joins, dict) or not joins:
        raise ValueError("task map must be a nonempty plain JSON object")
    if any(
        not isinstance(task, str)
        or not task.isdecimal()
        or not isinstance(prompt, str)
        or prompt not in prompt_ids
        for task, prompt in joins.items()
    ):
        raise ValueError("task map contains an unknown training prompt")
    if len(set(joins.values())) != len(joins):
        raise ValueError("task map assigns one train prompt to multiple tasks")
    return joins


def _request_ownership(
    manifest: dict, joins: dict[str, str], heads: list[dict], rounds: list[dict]
) -> None:
    if (
        manifest.get("schema") != "binary_head_capture_cell_v1"
        or manifest.get("complete") is not True
    ):
        raise ValueError("source cell manifest is incomplete or has wrong schema")
    requests = manifest.get("requests")
    if manifest.get("task_prompt_ids") != joins or not isinstance(requests, list) or not requests:
        raise ValueError("cell manifest task map differs from plain task map")
    if len(requests) != len(joins):
        raise ValueError("ambiguous task-to-prompt ownership")
    head_cursor = round_cursor = 0
    seen_tasks: set[str] = set()
    for request in requests:
        task = request.get("task_id")
        if not isinstance(task, str) or task not in joins or task in seen_tasks:
            raise ValueError("request task is absent or duplicated in task map")
        seen_tasks.add(task)
        if request.get("id") != joins[task]:
            raise ValueError("request prompt disagrees with task map")
        h_range, r_range = request.get("capture_rows"), request.get("forced_round_rows")
        if (
            not isinstance(h_range, list)
            or len(h_range) != 2
            or h_range[0] != head_cursor
            or not isinstance(r_range, list)
            or len(r_range) != 2
            or r_range[0] != round_cursor
        ):
            raise ValueError("request capture ranges are not contiguous")
        head_end, round_end = h_range[1], r_range[1]
        if (
            type(head_end) is not int
            or type(round_end) is not int
            or head_end <= head_cursor
            or round_end <= round_cursor
            or head_end > len(heads)
            or round_end > len(rounds)
        ):
            raise ValueError("request capture range is empty or out of bounds")
        if any(str(row.get("task_id")) != task for row in heads[head_cursor:head_end]):
            raise ValueError("head row belongs to wrong request task")
        if any(str(row.get("task_id")) != task for row in rounds[round_cursor:round_end]):
            raise ValueError("round row belongs to wrong request task")
        head_cursor, round_cursor = head_end, round_end
    if head_cursor != len(heads) or round_cursor != len(rounds) or seen_tasks != set(joins):
        raise ValueError("unclaimed head/round row or task")


def _maps(
    absolute_map_path: Path, target_vocab_size: int, offset_path: Path | None, t2d_path: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.load(absolute_map_path, allow_pickle=False)
    if absolute.ndim != 1 or absolute.dtype.kind not in "iu" or not len(absolute):
        raise ValueError("D d2t must be a nonempty absolute target-ID vector")
    if len(absolute) > target_vocab_size or np.any(absolute >= target_vocab_size):
        raise ValueError("D d2t contains out-of-range target IDs")
    # Integer array subtraction must not wrap at uint boundaries.
    mapped = [int(value) for value in absolute]
    if len(set(mapped)) != len(mapped):
        raise ValueError("D d2t contains duplicate target IDs")
    offsets = np.asarray([target - draft for draft, target in enumerate(mapped)], dtype=np.int64)
    inverse = validate_offset_d2t(offsets, target_vocab_size, len(mapped))
    t2d = np.asarray([draft >= 0 for draft in inverse], dtype=np.bool_)
    if offset_path is not None:
        supplied = np.load(offset_path, allow_pickle=False)
        if (
            supplied.dtype.kind not in "iu"
            or supplied.shape != offsets.shape
            or not np.array_equal(supplied, offsets)
        ):
            raise ValueError("native offset d2t disagrees with absolute D map")
    if t2d_path is not None:
        supplied = np.load(t2d_path, allow_pickle=False)
        if supplied.dtype != np.bool_ or not np.array_equal(supplied, t2d):
            raise ValueError("native t2d mask disagrees with D map inverse")
    return offsets, t2d


def _rounds(
    rounds: list[dict], joins: dict[str, str], vocabulary: int
) -> dict[tuple[str, int], dict]:
    by_key: dict[tuple[str, int], dict] = {}
    last_index: dict[str, int] = {}
    for record in rounds:
        if record.get("schema") != ROUND_SCHEMA:
            raise ValueError("wrong native round schema")
        task = str(record.get("task_id"))
        if task not in joins:
            raise ValueError("round has no request task")
        index = integer(record.get("round_index"), "round_index")
        if index != last_index.get(task, -1) + 1:
            raise ValueError("request round indices must be contiguous from zero")
        last_index[task] = index
        prefix = token_ids(record.get("prefix_token_ids"), "round prefix", vocabulary)
        seed = integer(record.get("seed_token_id"), "round seed")
        if seed >= vocabulary or record.get("pos0") != len(prefix):
            raise ValueError("round seed or native pos0 disagrees with prefix")
        draft = token_ids(record.get("draft_token_ids"), "round proposals", vocabulary)
        if len(draft) > MAX_DEPTH:
            raise ValueError("round exceeds frozen proposal horizon")
        accepted = integer(record.get("accepted_drafts"), "accepted_drafts")
        verifier = token_ids(record.get("verifier_token_ids"), "round verifier", vocabulary)
        if accepted > len(draft) or len(verifier) != accepted + 1:
            raise ValueError("native round acceptance length mismatch")
        if verifier[:accepted] != draft[:accepted]:
            raise ValueError("accepted verifier tokens disagree with proposals")
        key = (task, index)
        if key in by_key:
            raise ValueError("duplicate native round")
        by_key[key] = record
    return by_key


def _target_logit_rows(heads: list[dict], path: Path | None, vocabulary: int) -> int:
    indexes = [row.get("target_logits_row") for row in heads]
    if path is None:
        if any(index is not None for index in indexes):
            raise ValueError("native target logits row needs its raw target file")
        return 0
    row_bytes = vocabulary * 4
    if row_bytes <= 0 or path.stat().st_size % row_bytes:
        raise ValueError("raw target logit file size differs from target vocabulary")
    count = path.stat().st_size // row_bytes
    if count == 0:
        raise ValueError("supplied raw target logit file has no rows")
    used = [index for index in indexes if index is not None]
    if any(type(index) is not int or not 0 <= index < count for index in used):
        raise ValueError("invalid target logits row")
    if sorted(used) != list(range(count)):
        raise ValueError("target logits file has duplicated or missing joins")
    for row in heads:
        if row.get("target_logits_row") is not None and (
            row.get("target_logits_dim") != vocabulary
            or row.get("target_logits_source") != VERIFIER_LOGITS_SOURCE
        ):
            raise ValueError("target logits dimension/source mismatch")
    return count


def prepare(
    *,
    heads_path: Path,
    rounds_path: Path,
    task_map_path: Path,
    prompts_path: Path,
    absolute_map_path: Path,
    target_vocab_size: int,
    cell_manifest_path: Path | None = None,
    target_logits_path: Path | None = None,
    offset_path: Path | None = None,
    t2d_path: Path | None = None,
    expected_prompt_hash: str = TRAIN_PROMPTS_SHA256,
    expected_prompt_count: int = TRAIN_PROMPTS,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, dict]:
    """Validate all native joins and return output payloads without writing."""
    integer(target_vocab_size, "target_vocab_size", 1)
    if sha256(prompts_path) != expected_prompt_hash:
        raise ValueError("frozen training prompt hash mismatch")
    prompts = read_jsonl(prompts_path)
    prompt_ids = [row.get("id") for row in prompts]
    if (
        len(prompt_ids) != expected_prompt_count
        or any(
            not isinstance(value, str)
            or not value
            or "final" in value.lower()
            or "development" in value.lower()
            for value in prompt_ids
        )
        or len(set(prompt_ids)) != expected_prompt_count
    ):
        raise ValueError("expected unique frozen train-only prompt IDs")
    heads, rounds = read_jsonl(heads_path), read_jsonl(rounds_path)
    joins = _task_map(task_map_path, set(prompt_ids))
    manifest = None
    if cell_manifest_path is not None:
        manifest = json.loads(cell_manifest_path.read_text())
        if manifest.get("prompts_sha256") != expected_prompt_hash:
            raise ValueError("cell manifest prompt hash differs from frozen train split")
        _file_record(manifest, "heads", heads_path)
        _file_record(manifest, "rounds", rounds_path)
        _request_ownership(manifest, joins, heads, rounds)
    offsets, t2d = _maps(absolute_map_path, target_vocab_size, offset_path, t2d_path)
    if target_logits_path is not None and manifest is not None:
        _file_record(manifest, "target logits", target_logits_path)
    logit_count = _target_logit_rows(heads, target_logits_path, target_vocab_size)
    recorded = _rounds(rounds, joins, target_vocab_size)
    rows: list[dict] = []
    anchors: list[dict] = []
    counts: Counter[str] = Counter()
    by_round: dict[tuple[str, int], list[dict]] = defaultdict(list)
    state_width = heads[0].get("state_dim")
    for state_row, native in enumerate(heads):
        if native.get("schema") != HEAD_SCHEMA or native.get("state_row") != state_row:
            raise ValueError("native head row schema or state order mismatch")
        if (
            native.get("state_boundary") != HEAD_BOUNDARY
            or native.get("state_dtype") != "float32_native_endian"
            or type(native.get("state_dim")) is not int
            or native["state_dim"] < 1
            or native["state_dim"] != state_width
        ):
            raise ValueError("native head state boundary/type mismatch")
        if native.get("forced") is not False:
            raise ValueError("forced native head row cannot prepare own-history training")
        if native.get("finite") is not True or native.get("valid") is not True:
            raise ValueError("invalid native masked row cannot be silently dropped")
        if native.get("label_source") != LABEL_SOURCE:
            raise ValueError("native label is not cloned exact-prefix verifier")
        if native.get("full_logits_row") is not None and (
            native.get("full_logits_source") != DRAFT_LOGITS_SOURCE
        ):
            raise ValueError("native full_logits_row is not declared draft-head logits")
        if native.get("is_bonus") is not False or native.get("alignment_valid") is not True:
            raise ValueError("bonus or unaligned native head row")
        task = str(native.get("task_id"))
        if task not in joins:
            raise ValueError("unmapped native head task")
        index = integer(native.get("round_index"), "round_index")
        depth = integer(native.get("depth"), "depth")
        if depth >= MAX_DEPTH:
            raise ValueError("native head depth exceeds frozen proposal horizon")
        key = (task, index)
        record = recorded.get(key)
        if record is None:
            raise ValueError("head row has no matching native round")
        draft = record["draft_token_ids"]
        if depth >= len(draft):
            raise ValueError("head row exceeds recorded round proposals")
        prefix = token_ids(native.get("prefix_token_ids"), "head prefix", target_vocab_size)
        expected = record["prefix_token_ids"] + [record["seed_token_id"]] + draft[:depth]
        if prefix != expected:
            raise ValueError("head prefix has wrong proposal ancestry")
        parent = len(record["prefix_token_ids"]) - 1
        if (
            native.get("parent_position") != parent
            or native.get("input_position") != parent + depth + 1
            or native.get("label_position") != parent + depth + 2
            or native.get("verifier_row") != depth
            or native.get("input_token_id") != prefix[-1]
        ):
            raise ValueError("native head state/verifier positions mismatch")
        label = integer(native.get("verifier_token_id"), "verifier_token_id")
        if label >= target_vocab_size or native.get("proposed_token_id") != draft[depth]:
            raise ValueError("native label or proposal conflicts with round")
        supported = bool(t2d[label])
        if native.get("label_supported") is not supported:
            raise ValueError("native supported label disagrees with D d2t/t2d")
        if native.get("trainable") is not None and native["trainable"] is not supported:
            raise ValueError("native trainable mask disagrees with supported label")
        reached = depth <= record["accepted_drafts"] and depth < len(record["verifier_token_ids"])
        if native.get("verifier_reached") is not reached:
            raise ValueError("live verifier reach disagrees with round acceptance")
        expected_sample = record["verifier_token_ids"][depth] if reached else None
        if native.get("verifier_sampled_token_id") != expected_sample:
            raise ValueError("live verifier sample disagrees with round record")
        if reached and label != expected_sample:
            raise ValueError("cloned native label disagrees with live verifier")
        if (
            native.get("target_logits_row") is not None
            and native.get("target_logits_source") != VERIFIER_LOGITS_SOURCE
        ):
            raise ValueError("draft-head logits cannot substitute raw target logits")
        row = {
            "prompt_id": joins[task],
            "split": "train",
            "round_index": index,
            "depth": depth,
            "parent_position": parent,
            "input_position": parent + depth + 1,
            "label_position": parent + depth + 2,
            "verifier_row": depth,
            "prefix_token_ids": prefix,
            "input_token_id": prefix[-1],
            "alignment_valid": True,
            "is_bonus": False,
            "forced": False,
            "valid": True,
            "verifier_reached": reached,
            "invalid_reason": None,
            "label_source": LABEL_SOURCE,
            "verifier_token_id": label,
            "proposed_token_id": draft[depth],
            "label_supported": supported,
            "native_state_row": state_row,
            "native_task_id": task,
            "full_logits_row": native.get("full_logits_row"),
            "full_logits_source": native.get("full_logits_source"),
            "target_logits_row": native.get("target_logits_row"),
            "target_logits_dim": native.get("target_logits_dim"),
            "target_logits_source": native.get("target_logits_source"),
        }
        rows.append(row)
        by_round[key].append(row)
        counts["supported" if supported else "unsupported"] += 1
        if reached:
            counts["verifier_reached"] += 1
    if set(by_round) != set(recorded):
        raise ValueError("native round missing all proposal head rows")
    for key, record in recorded.items():
        group = by_round[key]
        if len(group) != len(record["draft_token_ids"]) or [r["depth"] for r in group] != list(
            range(len(group))
        ):
            raise ValueError("native round has missing, duplicate or unordered proposal rows")
        anchors.append(
            {
                "prompt_id": joins[key[0]],
                "split": "train",
                "round_index": key[1],
                "prefix_token_ids": record["prefix_token_ids"],
                "seed_token_id": record["seed_token_id"],
            }
        )
    trace = validate_recurrent_trace(
        rows,
        [RoundAnchor(**anchor) for anchor in anchors],
        offsets=offsets,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=len(offsets),
        max_depth=MAX_DEPTH,
        allowed_prompt_ids=set(prompt_ids),
        split="train",
    )
    sources = {
        name: sha256(path)
        for name, path in {
            "heads": heads_path,
            "rounds": rounds_path,
            "task_map": task_map_path,
            "prompts": prompts_path,
            "absolute_d2t": absolute_map_path,
            **({"cell_manifest": cell_manifest_path} if cell_manifest_path else {}),
            **({"raw_target_logits": target_logits_path} if target_logits_path else {}),
            **({"native_offsets": offset_path} if offset_path else {}),
            **({"native_t2d": t2d_path} if t2d_path else {}),
        }.items()
    }
    report = {
        "schema": "recurrent_native_rows_preparation_v1",
        "split": "train",
        "prompts_sha256": expected_prompt_hash,
        "prompt_count": expected_prompt_count,
        "source_sha256": sources,
        "absolute_map_raw_sha256": hashlib.sha256(
            np.asarray(np.load(absolute_map_path, allow_pickle=False), dtype="<i8").tobytes()
        ).hexdigest(),
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": len(offsets),
        "rounds": len(anchors),
        "rows": len(rows),
        "raw_target_logit_rows": logit_count,
        "counts": dict(counts),
        "trace_counts": trace.counts,
        "raw_target_feature_ledger": "unverified_missing",
        "full_drafter_numeric_parity": "unverified",
        "target_model_identity": "unverified_from_capture_metadata",
        "draft_map_identity": "unverified_absolute_map_source_requires_D_GGUF_audit",
        "request_prompt_ownership": (
            "cell_manifest_request_ranges_verified"
            if manifest
            else "unverified_plain_task_map_only"
        ),
        "cross_round_acceptance_ancestry": "unverified_native_rounds_lack_gap_provenance",
    }
    return rows, anchors, offsets, t2d, report


def write_prepared(
    output: Path, result: tuple[list[dict], list[dict], np.ndarray, np.ndarray, dict]
) -> dict:
    if output.exists():
        raise ValueError("output directory must be new")
    rows, anchors, offsets, t2d, report = result
    output.mkdir(parents=True)
    for name, records in (("rows.jsonl", rows), ("anchors.jsonl", anchors)):
        (output / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        )
    np.save(output / "offsets.npy", offsets)
    np.save(output / "t2d.npy", t2d)
    report["output_sha256"] = {
        name: sha256(output / name)
        for name in ("rows.jsonl", "anchors.jsonl", "offsets.npy", "t2d.npy")
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument(
        "--task-map",
        type=Path,
        required=True,
        help="plain JSON object {task_id_string: frozen_train_prompt_id}",
    )
    parser.add_argument("--cell-manifest", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--expected-prompt-count", type=int)
    parser.add_argument("--absolute-d2t", type=Path, required=True)
    parser.add_argument("--target-vocab-size", type=int, required=True)
    parser.add_argument("--target-logits", type=Path)
    parser.add_argument("--native-offsets", type=Path)
    parser.add_argument("--native-t2d", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_hash, expected_count = resolve_prompt_expectation(
        args.expected_prompt_sha256, args.expected_prompt_count
    )
    report = write_prepared(
        args.output,
        prepare(
            heads_path=args.heads,
            rounds_path=args.rounds,
            task_map_path=args.task_map,
            cell_manifest_path=args.cell_manifest,
            prompts_path=args.prompts,
            absolute_map_path=args.absolute_d2t,
            target_vocab_size=args.target_vocab_size,
            target_logits_path=args.target_logits,
            offset_path=args.native_offsets,
            t2d_path=args.native_t2d,
            expected_prompt_hash=expected_hash,
            expected_prompt_count=expected_count,
        ),
    )
    print(json.dumps({"rounds": report["rounds"], "rows": report["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
