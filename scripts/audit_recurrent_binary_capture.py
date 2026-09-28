#!/usr/bin/env python3
"""CPU-only metadata gate for a future native recurrent training capture.

This validates prompt ownership, file hashes, proposal ancestry, verifier
label provenance, vocabulary offsets and unsupported-label masks. It does not
create a native capture or certify target-feature values and draft K/V parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.recurrent_trace import (  # noqa: E402
    VERIFIER_LOGITS_SOURCE,
    RoundAnchor,
    validate_recurrent_trace,
)

TRAIN_PROMPTS_SHA256 = "80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74"
TRAIN_PROMPTS = 96
FEATURE_WIDTH = 7680
FEATURE_TAPS = [2, 18, 33]
FEATURE_BOUNDARY = "native_target_block_inputs_concat_before_draft_fc"
FEATURE_SOURCE = "native_target_features_on_accepted_prefix"


def resolve_prompt_expectation(
    expected_hash: str | None, expected_count: int | None
) -> tuple[str, int]:
    """Keep legacy 96 defaults; custom frozen splits require an explicit pair."""
    if expected_hash is None and expected_count is None:
        return TRAIN_PROMPTS_SHA256, TRAIN_PROMPTS
    if expected_hash is None or expected_count is None:
        raise ValueError("custom prompts require both expected SHA256 and count")
    if (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
        or type(expected_count) is not int
        or expected_count < 1
    ):
        raise ValueError("expected prompt SHA256/count is invalid")
    return expected_hash, expected_count


def validate_shard_manifest(
    shard_path: Path, prompts_path: Path, expected_hash: str, expected_count: int,
    prompt_ids: list[str],
) -> dict:
    """Bind a frozen train shard to its exact ordered prompts and parent split."""
    shard = json.loads(shard_path.read_text())
    if (
        not isinstance(shard, dict)
        or shard.get("schema") != "w1ax_capture_shard_v1"
        or shard.get("prompts_path") != prompts_path.name
        or shard.get("prompts_sha256") != expected_hash
        or shard.get("prompt_count") != expected_count
        or shard.get("prompt_ids") != prompt_ids
        or not isinstance(shard.get("parent"), dict)
        or shard["parent"].get("split") not in ("train_small", "train_large")
    ):
        raise ValueError("shard manifest does not match frozen training prompts")
    parent = shard["parent"]
    if (
        any(
            not isinstance(parent.get(key), str)
            or len(parent[key]) != 64
            or any(character not in "0123456789abcdef" for character in parent[key])
            for key in ("manifest_sha256", "prompts_sha256", "index_sha256")
        )
        or type(parent.get("count")) is not int
        or parent["count"] < expected_count
    ):
        raise ValueError("shard parent freeze record is incomplete")
    if shard_path.resolve().parent / shard["prompts_path"] != prompts_path.resolve():
        raise ValueError("shard prompt path must identify the supplied JSONL")
    return shard


@dataclass(frozen=True)
class CapturedRound:
    anchor: RoundAnchor
    rows: tuple[dict, ...]
    prefix_token_ids: tuple[int, ...]
    raw_target_features: torch.Tensor
    feature_positions: tuple[int, ...]


@dataclass
class AuditedCapture:
    report: dict
    anchors: dict[tuple[str, int], RoundAnchor]
    rows: dict[tuple[str, int], tuple[dict, ...]]
    feature_lookup: dict[tuple[str, tuple[int, ...]], int]
    features: np.ndarray

    def round_inputs(self, prompt_id: str, round_index: int) -> CapturedRound:
        """Return one validated accepted-prefix bundle for CPU replay."""
        key = (prompt_id, round_index)
        if key not in self.anchors:
            raise KeyError(key)
        anchor = self.anchors[key]
        accepted = tuple(anchor.prefix_token_ids)
        indices = [
            self.feature_lookup[(prompt_id, accepted[: position + 1])]
            for position in range(len(accepted))
        ]
        raw = torch.from_numpy(np.array(self.features[indices], dtype=np.float32, copy=True))
        return CapturedRound(
            anchor,
            tuple(dict(row) for row in self.rows[key]),
            (*accepted, anchor.seed_token_id),
            raw,
            tuple(range(len(accepted))),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: expected nonempty JSON object rows")
    return rows


def _owned_file(directory: Path, value: str) -> Path:
    path = directory / value
    if Path(value).name != value or not path.is_file():
        raise ValueError("capture paths must be existing basenames beside the manifest")
    return path


def audit_feature_ledger(
    feature_path: Path,
    feature_rows_path: Path,
    anchors: list[RoundAnchor],
    allowed_prompt_ids: set[str],
    target_vocab_size: int,
) -> dict:
    metadata = read_jsonl(feature_rows_path)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.dtype != np.float32 or features.shape != (len(metadata), FEATURE_WIDTH):
        raise ValueError("raw target features must be F32 [feature rows, 7680]")
    if not np.isfinite(features).all():
        raise ValueError("raw target features contain nonfinite values")
    lookup = {}
    for index, row in enumerate(metadata):
        prompt = row.get("prompt_id")
        position = row.get("position")
        prefix = row.get("prefix_token_ids")
        if (
            row.get("feature_row") != index
            or prompt not in allowed_prompt_ids
            or type(position) is not int
            or position < 0
            or not isinstance(prefix, list)
            or len(prefix) != position + 1
            or any(type(token) is not int or not 0 <= token < target_vocab_size for token in prefix)
            or row.get("tap_ids") != FEATURE_TAPS
            or row.get("boundary") != FEATURE_BOUNDARY
            or row.get("source") != FEATURE_SOURCE
            or row.get("accepted_prefix") is not True
        ):
            raise ValueError("feature row has invalid accepted-prefix provenance or position")
        key = (prompt, tuple(prefix))
        if key in lookup:
            raise ValueError("duplicate raw target feature prefix")
        lookup[key] = index
    used = set()
    for anchor in anchors:
        for position in range(len(anchor.prefix_token_ids)):
            key = (anchor.prompt_id, tuple(anchor.prefix_token_ids[: position + 1]))
            if key not in lookup:
                raise ValueError("accepted-prefix target feature row is missing")
            used.add(lookup[key])
    if used != set(range(len(metadata))):
        raise ValueError("feature ledger contains rows outside audited accepted prefixes")
    return {
        "rows": len(metadata),
        "width": FEATURE_WIDTH,
        "tap_ids": FEATURE_TAPS,
        "boundary": FEATURE_BOUNDARY,
        "source": FEATURE_SOURCE,
    }


def validate_raw_target_logit_rows(rows: list[dict], path: Path, target_vocab_size: int) -> int:
    """Join native logit row indices without materializing vocabulary vectors."""
    row_bytes = target_vocab_size * 4
    size = path.stat().st_size
    if target_vocab_size < 1 or size < row_bytes or size % row_bytes:
        raise ValueError("raw target logits payload size is incompatible with target vocabulary")
    count = size // row_bytes
    used = set()
    for row in rows:
        index = row.get("target_logits_row")
        if row.get("verifier_logits") is not None:
            raise ValueError("inline verifier logits are not a native target-logit join")
        if index is None:
            continue
        if (
            type(index) is not int
            or not 0 <= index < count
            or index in used
            or row.get("target_logits_dim") != target_vocab_size
            or row.get("target_logits_source") != VERIFIER_LOGITS_SOURCE
            or row.get("valid") is not True
        ):
            raise ValueError("raw target logits row, source or valid mask mismatch")
        used.add(index)
    if used != set(range(count)):
        raise ValueError("raw target logits contain unjoined or duplicated rows")
    return count


def streamed_mapped_probability_mass(
    path: Path, count: int, target_vocab_size: int, target_to_draft: tuple[int, ...]
) -> float:
    """Compute diagnostic mapped mass in bounded F64 batches over raw F32 rows."""
    values = np.memmap(path, mode="r", dtype="<f4", shape=(count, target_vocab_size))
    mapped_ids = np.flatnonzero(np.asarray(target_to_draft) >= 0)
    total = 0.0
    for start in range(0, count, 16):
        batch = np.asarray(values[start : start + 16], dtype=np.float64)
        if np.isnan(batch).any() or np.isposinf(batch).any():
            raise ValueError("invalid verifier logits")
        maxima = batch.max(axis=1)
        if not np.isfinite(maxima).all():
            raise ValueError("verifier logits have zero probability mass")
        batch -= maxima[:, None]
        np.exp(batch, out=batch)
        masses = batch[:, mapped_ids].sum(axis=1) / batch.sum(axis=1)
        total += float(masses.sum())
    return total / count


def audit_capture(
    manifest_path: Path,
    prompts_path: Path,
    expected_prompt_hash: str,
    *,
    expected_prompt_count: int = TRAIN_PROMPTS,
    shard_manifest_path: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "recurrent_binary_capture_v1":
        raise ValueError("unsupported recurrent capture manifest")
    if manifest.get("split") != "train" or sha256(prompts_path) != expected_prompt_hash:
        raise ValueError("capture requires the frozen training prompt split")
    if manifest.get("prompts_sha256") != expected_prompt_hash:
        raise ValueError("manifest prompt hash differs from frozen training prompts")
    if "prompt_count" in manifest and manifest["prompt_count"] != expected_prompt_count:
        raise ValueError("manifest prompt count differs from frozen training prompts")
    prompts = read_jsonl(prompts_path)
    prompt_ids = [row.get("id") for row in prompts]
    if (
        len(prompt_ids) != expected_prompt_count
        or any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids)
        or len(set(prompt_ids)) != expected_prompt_count
    ):
        raise ValueError(f"expected {expected_prompt_count} unique frozen training prompt IDs")
    shard_record = manifest.get("shard_manifest")
    if shard_record is not None:
        if not isinstance(shard_record, dict) or set(shard_record) != {"path", "sha256"}:
            raise ValueError("invalid shard manifest file record")
        embedded = _owned_file(manifest_path.parent, shard_record["path"])
        if sha256(embedded) != shard_record["sha256"]:
            raise ValueError("shard manifest SHA256 mismatch")
        if shard_manifest_path is not None and sha256(shard_manifest_path) != sha256(embedded):
            raise ValueError("supplied shard manifest differs from bundle")
        shard_manifest_path = embedded
    shard = None
    if shard_manifest_path is not None:
        shard = validate_shard_manifest(
            shard_manifest_path, prompts_path, expected_prompt_hash,
            expected_prompt_count, prompt_ids,
        )
        if (
            shard.get("target_vocab_size") != manifest.get("target_vocab_size")
            or shard.get("bytes_per_raw_logit_row") != manifest["target_vocab_size"] * 4
        ):
            raise ValueError("shard target vocabulary differs from captured bundle")
    files = {}
    for field in ("rows", "anchors", "offsets", "t2d", "features", "feature_rows"):
        record = manifest.get(field)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"missing {field} file record")
        path = _owned_file(manifest_path.parent, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"{field} SHA256 mismatch")
        files[field] = path
    rows = read_jsonl(files["rows"])
    target_logit_rows = 0
    target_logit_path = None
    if "target_logits" in manifest:
        record = manifest["target_logits"]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("invalid target_logits file record")
        path = _owned_file(manifest_path.parent, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError("target_logits SHA256 mismatch")
        files["target_logits"] = path
        target_logit_rows = validate_raw_target_logit_rows(
            rows, path, manifest["target_vocab_size"]
        )
        target_logit_path = path
        if shard is not None:
            caps = shard.get("caps")
            if (
                not isinstance(caps, dict)
                or type(caps.get("max_prompts")) is not int
                or expected_prompt_count > caps["max_prompts"]
                or type(caps.get("max_verifier_logit_rows")) is not int
                or type(caps.get("max_raw_logit_bytes")) is not int
                or target_logit_rows > caps["max_verifier_logit_rows"]
                or path.stat().st_size > caps["max_raw_logit_bytes"]
            ):
                raise ValueError("raw target logits exceed frozen shard cap")
    elif any(
        row.get("target_logits_row") is not None or row.get("verifier_logits") is not None
        for row in rows
    ):
        raise ValueError("target logits require the native raw target-logit file")
    if shard is not None and target_logit_path is None:
        raise ValueError("sharded capture requires retained raw target logits")
    anchor_rows = read_jsonl(files["anchors"])
    anchors = [RoundAnchor(**row) for row in anchor_rows]
    offsets = np.load(files["offsets"], allow_pickle=False)
    t2d = np.load(files["t2d"], allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu":
        raise ValueError("native d2t offsets must be a one-dimensional integer array")
    trace = validate_recurrent_trace(
        rows,
        anchors,
        offsets=offsets,
        target_vocab_size=manifest["target_vocab_size"],
        draft_vocab_size=manifest["draft_vocab_size"],
        max_depth=manifest["max_depth"],
        allowed_prompt_ids=set(prompt_ids),
        split="train",
    )
    if t2d.dtype != np.bool_ or t2d.shape != (manifest["target_vocab_size"],):
        raise ValueError("t2d must be a target-vocabulary boolean mask")
    if not np.array_equal(t2d, np.asarray(trace.target_to_draft) >= 0):
        raise ValueError("t2d mask disagrees with offset-form d2t inverse")
    feature_ledger = audit_feature_ledger(
        files["features"],
        files["feature_rows"],
        anchors,
        set(prompt_ids),
        manifest["target_vocab_size"],
    )
    mapped_mass_mean = (
        streamed_mapped_probability_mass(
            target_logit_path,
            target_logit_rows,
            manifest["target_vocab_size"],
            trace.target_to_draft,
        )
        if target_logit_path is not None
        else None
    )
    counts = dict(trace.counts)
    counts["logit_rows"] = target_logit_rows
    return {
        "schema": "recurrent_binary_capture_audit_v1",
        "execution_device": "cpu",
        "capture_manifest_sha256": sha256(manifest_path),
        "training_prompts_sha256": expected_prompt_hash,
        "training_prompt_count": expected_prompt_count,
        "shard_manifest_sha256": (
            sha256(shard_manifest_path) if shard_manifest_path is not None else None
        ),
        "source_sha256": {field: sha256(path) for field, path in files.items()},
        "counts": counts,
        "per_depth": trace.per_depth,
        "feature_ledger": feature_ledger,
        "raw_target_logit_rows": target_logit_rows,
        "mapped_probability_mass_mean_on_sampled_logit_rows": mapped_mass_mean,
        "real_model_feature_and_kv_parity": "unverified",
    }


def load_audited_capture(
    manifest_path: Path,
    prompts_path: Path,
    expected_prompt_hash: str,
    *,
    expected_prompt_count: int = TRAIN_PROMPTS,
) -> AuditedCapture:
    """Audit once, then expose CPU round bundles for prefix reconstruction."""
    report = audit_capture(
        manifest_path,
        prompts_path,
        expected_prompt_hash,
        expected_prompt_count=expected_prompt_count,
    )
    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent
    anchors = {
        (item["prompt_id"], item["round_index"]): RoundAnchor(**item)
        for item in read_jsonl(directory / manifest["anchors"]["path"])
    }
    rows = defaultdict(list)
    for row in read_jsonl(directory / manifest["rows"]["path"]):
        rows[(row["prompt_id"], row["round_index"])].append(row)
    for group in rows.values():
        group.sort(key=lambda row: row["depth"])
    feature_lookup = {
        (item["prompt_id"], tuple(item["prefix_token_ids"])): item["feature_row"]
        for item in read_jsonl(directory / manifest["feature_rows"]["path"])
    }
    features = np.load(directory / manifest["features"]["path"], mmap_mode="r", allow_pickle=False)
    return AuditedCapture(
        report,
        anchors,
        {key: tuple(group) for key, group in rows.items()},
        feature_lookup,
        features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--expected-prompt-count", type=int)
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be a new path")
    expected_hash, expected_count = resolve_prompt_expectation(
        args.expected_prompt_sha256, args.expected_prompt_count
    )
    report = audit_capture(
        args.manifest, args.prompts, expected_hash,
        expected_prompt_count=expected_count, shard_manifest_path=args.shard_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
