#!/usr/bin/env python3
"""Compact captured native target logits into auditable top-k teacher shards.

This is CPU postprocessing only. Probabilities are unconditional on the full
target vocabulary; probability outside the mapped draft vocabulary remains
explicit. Rows retain their exact captured prefix, so they must never be
silently relabeled for changed student prefixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _hash_arg(value: str, name: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def _load_logits(path: Path, fmt: str, vocab: int) -> np.ndarray:
    if fmt == "npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != vocab or values.dtype != np.float32:
            raise ValueError("logits .npy must be F32 [rows,target_vocab]")
        return values
    if fmt == "raw-f32":
        if path.stat().st_size % (4 * vocab):
            raise ValueError("raw logits size is not a whole number of F32 rows")
        return np.memmap(path, dtype="<f4", mode="r",
                         shape=(path.stat().st_size // (4 * vocab), vocab))
    raise ValueError("logits format must be npy or raw-f32")


def _topk(prob: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.argpartition(prob, -k)[-k:]
    ids = ids[np.lexsort((ids, -prob[ids]))]
    return ids.astype(np.int32), prob[ids].astype(np.float32)


def compact(logits_path: Path, logits_format: str, rows_path: Path, d2t_path: Path,
            output: Path, *, target_vocab: int, topk: int, shard_rows: int,
            target_gguf_sha256: str, prompts_sha256: str,
            capture_manifest_sha256: str) -> dict:
    if output.exists():
        raise ValueError("output must be new; teacher shards are immutable")
    for name, value in (("target_gguf_sha256", target_gguf_sha256),
                        ("prompts_sha256", prompts_sha256),
                        ("capture_manifest_sha256", capture_manifest_sha256)):
        _hash_arg(value, name)
    if target_vocab <= 0 or topk <= 0 or shard_rows <= 0:
        raise ValueError("invalid vocab, topk or shard size")
    d2t = np.load(d2t_path, allow_pickle=False)
    if (d2t.ndim != 1 or d2t.dtype.kind not in "iu" or len(d2t) < topk
            or np.any(d2t < 0) or np.any(d2t >= target_vocab)
            or len(np.unique(d2t)) != len(d2t)):
        raise ValueError("d2t must map distinct draft IDs to absolute target IDs")
    logits = _load_logits(logits_path, logits_format, target_vocab)
    reverse = np.full(target_vocab, -1, dtype=np.int32)
    reverse[d2t] = np.arange(len(d2t), dtype=np.int32)
    output.mkdir(parents=True)
    index_path = output / "rows.jsonl"
    shards, buffers, used_rows, seen_ids, count = [], {}, set(), set(), 0
    arrays = ("draft_topk_ids", "draft_topk_probs", "draft_tail_mass",
              "outside_draft_mass", "target_topk_ids", "target_topk_probs",
              "target_tail_mass", "next_target_id", "next_draft_id")
    buffers = {name: [] for name in arrays}

    def flush() -> None:
        nonlocal buffers
        if not buffers["draft_topk_ids"]:
            return
        number = len(shards)
        shard_path = output / f"teacher-{number:05d}.npz"
        np.savez_compressed(
            shard_path, **{key: np.asarray(value) for key, value in buffers.items()}
        )
        shards.append({"path": shard_path.name, "sha256": sha256(shard_path),
                       "first_row": count - len(buffers["draft_topk_ids"]),
                       "rows": len(buffers["draft_topk_ids"])})
        buffers = {name: [] for name in arrays}

    with rows_path.open() as source, index_path.open("wb") as index:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} must be an object")
            rid, prompt, capture = (row.get(key) for key in ("id", "prompt_id", "capture_id"))
            logit_row, prefix = row.get("logits_row"), row.get("prefix_token_ids")
            if (not all(isinstance(value, str) and value for value in (rid, prompt, capture))
                    or not isinstance(logit_row, int) or not 0 <= logit_row < len(logits)
                    or not isinstance(prefix, list) or not prefix
                    or any(not isinstance(token, int) or not 0 <= token < target_vocab
                           for token in prefix)):
                raise ValueError(f"row {line_number} lacks exact prefix/capture ancestry")
            if rid in seen_ids or logit_row in used_rows:
                raise ValueError("duplicate teacher row ID or reused logits row")
            seen_ids.add(rid)
            used_rows.add(logit_row)
            label = row.get("next_target_id", -1)
            if not isinstance(label, int) or not -1 <= label < target_vocab:
                raise ValueError(f"row {line_number} has invalid next target ID")
            values = np.asarray(logits[logit_row], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"row {line_number} contains nonfinite logits")
            probabilities = np.exp(values - values.max())
            probabilities /= probabilities.sum(dtype=np.float64)
            draft_probabilities = probabilities[d2t]
            draft_ids, draft_probs = _topk(draft_probabilities, topk)
            target_ids, target_probs = _topk(probabilities, topk)
            draft_tail = float(draft_probabilities.sum(dtype=np.float64)
                               - draft_probabilities[draft_ids].sum(dtype=np.float64))
            outside = float(1 - draft_probabilities.sum(dtype=np.float64))
            target_tail = float(1 - probabilities[target_ids].sum(dtype=np.float64))
            if (min(draft_tail, outside, target_tail) < -1e-10
                    or abs(float(draft_probs.sum(dtype=np.float64)) + draft_tail + outside - 1)
                    > 1e-6):
                raise ValueError(f"row {line_number} failed probability mass check")
            payload = {"draft_topk_ids": draft_ids,
                       "draft_topk_probs": draft_probs,
                       "draft_tail_mass": np.float32(max(draft_tail, 0)),
                       "outside_draft_mass": np.float32(max(outside, 0)),
                       "target_topk_ids": target_ids,
                       "target_topk_probs": target_probs,
                       "target_tail_mass": np.float32(max(target_tail, 0)),
                       "next_target_id": np.int32(label),
                       "next_draft_id": np.int32(reverse[label] if label >= 0 else -1)}
            for key, value in payload.items():
                buffers[key].append(value)
            prefix_hash = hashlib.sha256(json_bytes(prefix)).hexdigest()
            index.write(json_bytes({"row": count, "id": rid, "prompt_id": prompt,
                                    "capture_id": capture, "logits_row": logit_row,
                                    "prefix_token_ids": prefix,
                                    "prefix_sha256": prefix_hash}) + b"\n")
            count += 1
            if count % shard_rows == 0:
                flush()
    flush()
    if count == 0:
        raise ValueError("no teacher rows")
    manifest = {"schema": "w1a_compact_teacher_v1", "rows": count,
                "target_vocab": target_vocab, "draft_vocab": len(d2t), "topk": topk,
                "probability_contract": (
                    "full_target_softmax_unconditional_f32_topk_with_f32_tail_mass"
                ),
                "tail_contract": (
                    "draft_tail_excludes_draft_topk;outside_draft_is_unmapped_target_mass"
                ),
                "loss_contract": "topk_is_approximation;no_implicit_draft_renormalization",
                "prefix_contract": "teacher_forced_exact_captured_prefix_only",
                "target_gguf_sha256": target_gguf_sha256,
                "prompts_sha256": prompts_sha256,
                "capture_manifest_sha256": capture_manifest_sha256,
                "logits_sha256": sha256(logits_path), "rows_sha256": sha256(rows_path),
                "d2t_sha256": sha256(d2t_path), "index": "rows.jsonl",
                "index_sha256": sha256(index_path), "shards": shards}
    (output / "manifest.json").write_bytes(json_bytes(manifest) + b"\n")
    return manifest


def iter_verified_shards(
    directory: Path, *, expected_prompts_sha256: str,
    expected_target_gguf_sha256: str, expected_d2t_sha256: str,
    expected_prefixes: dict[str, list[int]] | None = None,
):
    """Yield (row metadata, arrays) after hash, mass and optional prefix checks.

    A trainer must additionally bind each yielded row to its current sample ID
    and prefix. Supply expected_prefixes for a strict exact-prefix comparison.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "w1a_compact_teacher_v1":
        raise ValueError("wrong compact teacher schema")
    for key, expected in (("prompts_sha256", expected_prompts_sha256),
                          ("target_gguf_sha256", expected_target_gguf_sha256),
                          ("d2t_sha256", expected_d2t_sha256)):
        if manifest.get(key) != expected:
            raise ValueError(f"teacher {key} mismatch")
    index_path = directory / manifest["index"]
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("teacher row index SHA256 mismatch")
    cursor = 0
    with index_path.open() as index:
        for shard in manifest["shards"]:
            path = directory / shard["path"]
            if sha256(path) != shard["sha256"] or shard["first_row"] != cursor:
                raise ValueError("teacher shard hash or row range mismatch")
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            count = shard["rows"]
            rows = []
            for i in range(cursor, cursor + count):
                line = index.readline()
                if not line:
                    raise ValueError("teacher index omits rows")
                row = json.loads(line)
                if (row.get("row") != i or not isinstance(row.get("id"), str)
                        or hashlib.sha256(json_bytes(row.get("prefix_token_ids"))).hexdigest()
                        != row.get("prefix_sha256")):
                    raise ValueError("teacher prefix ancestry mismatch")
                if (expected_prefixes is not None
                        and expected_prefixes.get(row["id"]) != row["prefix_token_ids"]):
                    raise ValueError(f"teacher prefix does not match current sample: {row['id']}")
                rows.append(row)
            if (any(len(value) != count for value in arrays.values())
                    or arrays["draft_topk_ids"].shape != (count, manifest["topk"])
                    or arrays["draft_topk_probs"].shape != (count, manifest["topk"])):
                raise ValueError("teacher shard shape mismatch")
            mass = (arrays["draft_topk_probs"].sum(axis=1, dtype=np.float64)
                    + arrays["draft_tail_mass"] + arrays["outside_draft_mass"])
            if (not np.isfinite(mass).all() or not np.allclose(mass, 1, atol=1e-5)
                    or np.any(arrays["draft_topk_probs"] < 0)
                    or np.any(arrays["draft_tail_mass"] < 0)
                    or np.any(arrays["outside_draft_mass"] < 0)):
                raise ValueError("teacher probability mass mismatch")
            yield rows, arrays
            cursor += count
        if cursor != manifest["rows"] or index.readline():
            raise ValueError("teacher shard/index row count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--logits-format", choices=("npy", "raw-f32"), required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--d2t", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-vocab", type=int, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--shard-rows", type=int, default=4096)
    parser.add_argument("--target-gguf-sha256", required=True)
    parser.add_argument("--prompts-sha256", required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    args = parser.parse_args()
    result = compact(args.logits, args.logits_format, args.rows, args.d2t, args.output,
                     target_vocab=args.target_vocab, topk=args.topk,
                     shard_rows=args.shard_rows, target_gguf_sha256=args.target_gguf_sha256,
                     prompts_sha256=args.prompts_sha256,
                     capture_manifest_sha256=args.capture_manifest_sha256)
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "sha256": sha256(args.output / "manifest.json"),
                      "rows": result["rows"], "shards": len(result["shards"])}))


if __name__ == "__main__":
    main()
