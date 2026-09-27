#!/usr/bin/env python3
"""Audit native own-history D captures and select the frozen train-only head data.

The JSON capture manifest identifies split=train, body=D, trajectory=own_history,
train_prompts_sha256, d2t_path (absolute target IDs), target_vocab_size and captures.
Each capture has states_path (.npy or raw little-endian .f32), rows_path (.jsonl),
state_dim (required for .f32), and optional task_prompt_ids mapping server task IDs
as strings to training prompt IDs. Native row schema is eagle_head_state_v1.
No development/final file is opened. Invalid alignment stops preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCHEMA = "frozen_d_head_train_v1"
ROWS_PER_BUCKET = 17
TRAIN_PROMPTS = 96
DEPTHS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_d2t(d2t: np.ndarray, target_vocab_size: int) -> np.ndarray:
    if d2t.ndim != 1 or d2t.dtype.kind not in "iu" or not len(d2t):
        raise ValueError("d2t must be a nonempty integer vector of absolute target IDs")
    if target_vocab_size <= 0 or np.any(d2t < 0) or np.any(d2t >= target_vocab_size):
        raise ValueError("d2t contains out-of-range target IDs")
    if len(np.unique(d2t)) != len(d2t):
        raise ValueError("duplicate d2t target IDs")
    reverse = np.full(target_vocab_size, -1, dtype=np.int64)
    reverse[d2t] = np.arange(len(d2t))
    return reverse


def bucket_centers(count: int, cap: int = ROWS_PER_BUCKET) -> list[int]:
    take = min(count, cap)
    return [((2 * i + 1) * count) // (2 * take) for i in range(take)]


def audit_rows(rows: list[dict], states: np.ndarray, prompt_ids: set[str],
               d2t: np.ndarray, target_vocab_size: int) -> tuple[dict, dict]:
    reverse = validate_d2t(d2t, target_vocab_size)
    if states.ndim != 2 or len(states) != len(rows) or states.dtype != np.float32:
        raise ValueError("states must be an F32 matrix with exactly one row per metadata record")
    if not np.isfinite(states).all():
        raise ValueError("nonfinite native head states")
    valid = np.zeros(len(rows), dtype=bool)
    supported = np.zeros(len(rows), dtype=bool)
    labels = np.full(len(rows), -1, dtype=np.int64)
    selected = np.zeros(len(rows), dtype=bool)
    buckets = defaultdict(list)
    seen = set()
    denominator = defaultdict(Counter)
    for index, row in enumerate(rows):
        prompt_id, depth = row.get("prompt_id"), row.get("depth")
        if prompt_id not in prompt_ids or row.get("split", "train") != "train" or row.get("forced", False):
            raise ValueError("capture contains unknown/nontraining prompt")
        if not isinstance(depth, int) or not 0 <= depth < DEPTHS:
            raise ValueError("capture depth outside frozen D5 protocol")
        parent = row.get("parent_position")
        if not isinstance(parent, int) or parent < 0 or row.get("round_index", -1) < 0:
            raise ValueError("invalid parent position or round index")
        if (row.get("alignment_valid") is not True
                or row.get("input_position") != parent + depth + 1
                or row.get("label_position") != parent + depth + 2
                or row.get("verifier_row") != depth):
            raise ValueError("invalid state/verifier next-token alignment")
        prefix = row.get("prefix_token_ids")
        if (not isinstance(prefix, list) or len(prefix) != row["label_position"]
                or not prefix or prefix[-1] != row.get("input_token_id")
                or any(not isinstance(t, int) or not 0 <= t < target_vocab_size for t in prefix)):
            raise ValueError("invalid actual parent/prefix/position alignment")
        key = (prompt_id, row["round_index"], depth)
        if key in seen:
            raise ValueError("duplicate prompt/round/depth state")
        seen.add(key)
        bucket = (prompt_id, depth)
        denominator[bucket]["total"] += 1
        if row.get("is_bonus") is not False:
            raise ValueError("bonus verifier rows cannot be training states")
        if row.get("label_source") != "cloned_native_verifier_sampler_at_actual_proposal_prefix":
            raise ValueError("label is not the true native verifier sampler at the actual prefix")
        token = row.get("verifier_token_id")
        # Censored states have no verified next-token label, and remain in masks.
        if row.get("valid") is False:
            if token is not None:
                raise ValueError("invalid/censored row unexpectedly has a verifier label")
            denominator[bucket]["censored"] += 1
            continue
        if row.get("valid") is not True or not isinstance(token, int) or not 0 <= token < target_vocab_size:
            raise ValueError("missing or out-of-range true verifier token")
        valid[index] = True
        labels[index] = reverse[token]
        supported[index] = labels[index] >= 0
        if row.get("label_supported") is not bool(supported[index]):
            raise ValueError("native label_supported differs from audited d2t mapping")
        denominator[bucket]["valid"] += 1
        denominator[bucket]["supported" if supported[index] else "unsupported"] += 1
        if supported[index]:
            buckets[bucket].append(index)
    for bucket, indices in buckets.items():
        indices.sort(key=lambda i: (rows[i]["round_index"], rows[i]["parent_position"], i))
        chosen = [indices[i] for i in bucket_centers(len(indices))]
        selected[chosen] = True
        denominator[bucket]["selected"] = len(chosen)
    if not selected.any():
        raise ValueError("no valid supported training states")
    coverage = [{"prompt_id": p, "depth": d, **{key: denominator[(p, d)][key]
                 for key in ("total", "valid", "supported", "unsupported", "censored", "selected")}}
                for p in sorted(prompt_ids) for d in range(DEPTHS)]
    arrays = {"states": states[selected], "labels": labels[selected], "selected_indices": np.flatnonzero(selected),
              "valid_mask": valid, "supported_mask": supported, "selected_mask": selected,
              "all_labels": labels, "d2t": np.asarray(d2t, dtype=np.int64)}
    report = {"rows": len(rows), "selected_rows": int(selected.sum()), "valid_rows": int(valid.sum()),
              "unsupported_rows": int((valid & ~supported).sum()), "censored_rows": int((~valid).sum()),
              "per_prompt_depth": coverage, "selection": "up to17 ordered bucket centers per prompt/depth"}
    return arrays, report


def prepare(manifest_path: Path, train_prompts: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(k) != v for k, v in {"split": "train", "body": "D", "trajectory": "own_history"}.items()):
        raise ValueError("only own-history frozen D training captures are allowed")
    if manifest.get("train_prompts_sha256") != sha256(train_prompts):
        raise ValueError("training prompt hash mismatch")
    prompts = read_jsonl(train_prompts)
    ids = {row["id"] for row in prompts}
    if len(prompts) != TRAIN_PROMPTS or len(ids) != TRAIN_PROMPTS:
        raise ValueError("expected exactly96 unique frozen training prompts")
    base = manifest_path.parent
    d2t_path = resolve(base, manifest["d2t_path"])
    d2t = np.load(d2t_path, allow_pickle=False)
    rows, chunks, sources = [], [], []
    for capture in manifest["captures"]:
        state_path = resolve(base, capture["states_path"])
        rows_path = resolve(base, capture["rows_path"])
        metadata = read_jsonl(rows_path)
        state = (np.load(state_path, allow_pickle=False) if state_path.suffix == ".npy" else
                 np.fromfile(state_path, dtype="<f4").reshape(-1, capture["state_dim"]))
        if len(state) != len(metadata):
            raise ValueError("native capture row count mismatch")
        for index, record in enumerate(metadata):
            if record.get("schema") != "eagle_head_state_v1" or record.get("state_row") != index:
                raise ValueError("invalid native state schema/order")
            if record.get("state_dim") != state.shape[1]:
                raise ValueError("native state_dim mismatch")
            record = dict(record)
            if "prompt_id" not in record:
                record["prompt_id"] = capture.get("task_prompt_ids", {}).get(str(record["task_id"]))
            record["source_capture"] = len(sources)
            rows.append(record)
        chunks.append(state)
        sources.append({"states_path": str(state_path), "states_sha256": sha256(state_path),
                        "rows_path": str(rows_path), "rows_sha256": sha256(rows_path)})
    if not chunks:
        raise ValueError("capture manifest is empty")
    arrays, report = audit_rows(rows, np.concatenate(chunks), ids, d2t, manifest["target_vocab_size"])
    report.update({"schema": SCHEMA, "split": "train", "body": "D", "trajectory": "own_history",
                   "train_prompts_sha256": sha256(train_prompts), "capture_manifest_sha256": sha256(manifest_path),
                   "d2t_sha256": sha256(d2t_path), "target_vocab_size": manifest["target_vocab_size"],
                   "training_prompts": TRAIN_PROMPTS, "sources": sources})
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(output_dir / "dataset.npz", **arrays)
    (output_dir / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    report["dataset_sha256"] = sha256(output_dir / "dataset.npz")
    report["rows_sha256"] = sha256(output_dir / "rows.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def prepare_parity_arrays(manifest_path: Path, output_dir: Path) -> dict:
    """Gather bounded native full-logit rows into draft-vocabulary parity arrays.

    Input manifest uses the same captures interface, with logits_path per capture.
    This only prepares evidence; it cannot grant zero-export/behavior approval.
    """
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    d2t_path = resolve(base, manifest["d2t_path"])
    d2t = np.load(d2t_path, allow_pickle=False)
    target_vocab = manifest["target_vocab_size"]
    validate_d2t(d2t, target_vocab)
    xs, ys, provenance, sources = [], [], [], []
    for capture in manifest["captures"]:
        states_path = resolve(base, capture["states_path"])
        rows_path = resolve(base, capture["rows_path"])
        logits_path = resolve(base, capture["logits_path"])
        rows = read_jsonl(rows_path)
        states = (np.load(states_path, allow_pickle=False) if states_path.suffix == ".npy" else
                  np.fromfile(states_path, dtype="<f4").reshape(-1, capture["state_dim"]))
        native = np.fromfile(logits_path, dtype="<f4").reshape(-1, target_vocab)
        if len(states) != len(rows):
            raise ValueError("parity state/metadata row count mismatch")
        seen = set()
        for index, row in enumerate(rows):
            full_row = row.get("full_logits_row")
            if full_row is None:
                continue
            if (not isinstance(full_row, int) or not 0 <= full_row < len(native) or full_row in seen
                    or row.get("full_logits_dim") != target_vocab or row.get("state_row") != index
                    or not row.get("full_logits_boundary")):
                raise ValueError("invalid full native logits row mapping")
            seen.add(full_row)
            values = native[full_row, d2t]
            if not np.isfinite(values).all() or not np.isfinite(states[index]).all():
                raise ValueError("nonfinite supported native parity values")
            xs.append(states[index])
            ys.append(values)
            provenance.append({"capture": len(sources), "state_row": index, "full_logits_row": full_row,
                               "depth": row["depth"], "boundary": row["full_logits_boundary"]})
        if len(seen) != len(native):
            raise ValueError("unreferenced full native logits rows")
        sources.append({"states_path": str(states_path), "states_sha256": sha256(states_path),
                        "rows_path": str(rows_path), "rows_sha256": sha256(rows_path),
                        "logits_path": str(logits_path), "logits_sha256": sha256(logits_path)})
    if not xs:
        raise ValueError("no full native logits were captured")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "parity-states.npy", np.stack(xs).astype(np.float32), allow_pickle=False)
    np.save(output_dir / "parity-logits.npy", np.stack(ys).astype(np.float32), allow_pickle=False)
    report = {"states_path": "parity-states.npy", "states_sha256": sha256(output_dir / "parity-states.npy"),
              "logits_path": "parity-logits.npy", "logits_sha256": sha256(output_dir / "parity-logits.npy"),
              "rows": len(xs), "draft_vocab": len(d2t), "d2t_sha256": sha256(d2t_path),
              "sources": sources, "provenance": provenance,
              "capture_manifest_sha256": sha256(manifest_path)}
    (output_dir / "parity-artifacts.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path)
    parser.add_argument("--parity-only", action="store_true", help="prepare bounded full native logits evidence only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.parity_only:
        report = prepare_parity_arrays(args.capture_manifest, args.output_dir)
    else:
        if args.train_prompts is None:
            parser.error("--train-prompts is required unless --parity-only")
        report = prepare(args.capture_manifest, args.train_prompts, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
