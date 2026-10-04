#!/usr/bin/env python3
"""Bounded CPU-only direct binary fusion fitting and supported fusion-only GGUF export.

Input package: NPZ {raw_input:F32[N,K], reference_weight:F32[M,K],
raw_join_ids:Unicode[N]}; a v1 manifest with source hashes and prompt-disjoint
TRAIN/validation row joins. No capture, model execution or accelerator access.
Real data requires a separately SHA-pinned acquisition receipt proving TRAIN
membership and raw-feature ancestry, plus exact frozen model files. Output
reconstruction is a surrogate and makes no native acceptance/throughput claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_CONTRACT = (
    "raw_f32_absmax_f32_reciprocal_nearest_even_clip127_dot_row_scale_token_scale_f32"
)
TEACHER_CONTRACT = "raw_f32_original_bf16_promoted_f32_blas"
SOURCE_KEYS = {"frozen_weights_sha256", "base_gguf_sha256", "capture_manifest_sha256"}
ROW_KEYS = {
    "row_id",
    "prompt_id",
    "prompt_sha256",
    "depth",
    "position",
    "split",
    "raw_input_sha256",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


FROZEN_SOURCE_WEIGHTS_SHA256 = "58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e"
FROZEN_BASE_GGUF_SHA256 = "c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1"
REAL_PRODUCER_CONTRACT = (
    "native_target_block_inputs_concat_before_draft_fc_f32_taps_2_18_33_no_upstream_cast_or_norm"
)
REAL_FUSION_SHAPE = (2560, 7680)
REAL_PROMPT_COUNTS = {"train": 8, "validation": 4}
REAL_ROWS_PER_PROMPT = 32
RECEIPT_CHECKS = {
    "train_membership",
    "prompt_disjoint",
    "raw_boundary_verified",
    "capture_eligible",
    "source_hashes_verified",
}


def _receipt_file(record, directory, label, override=None):
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or type(record["path"]) is not str
        or not record["path"]
        or not is_hash(record["sha256"])
    ):
        raise ValueError(f"{label}: exact path and SHA256 required")
    recorded = (directory / record["path"]).resolve()
    actual = Path(override).resolve() if override is not None else recorded
    if not actual.is_file() or sha256(actual) != record["sha256"]:
        raise ValueError(f"{label}: local source file hash mismatch")
    return actual


def verify_real_receipt(
    receipt_path,
    receipt_hash,
    manifest_path,
    manifest,
    x,
    w,
    *,
    source_weights_path=None,
    base_gguf_path=None,
):
    """Recheck a trusted acquisition receipt; a manifest cannot self-authorize fitting.

    The receipt SHA must be pinned separately by the CPU run owner after the data
    owner authenticates the read-only transfer. Capture/producer hashes are remote
    evidence bound by that receipt, not a claim that this CPU importer rereads them.
    Local inventory, input package, original BF16 weights and F16 base are rehashed.
    """
    if receipt_path is None or not is_hash(receipt_hash):
        raise ValueError("real fitting unavailable without independently pinned provenance receipt")
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file() or sha256(receipt_path) != receipt_hash:
        raise ValueError("provenance receipt SHA256 differs from external pin")
    r = json.loads(receipt_path.read_text())
    keys = {
        "schema",
        "manifest_sha256",
        "operands_sha256",
        "source_weights",
        "base_gguf",
        "producer",
        "train_inventory",
        "selected_prompts",
        "rows",
        "checks",
    }
    if (
        not isinstance(r, dict)
        or set(r) != keys
        or r["schema"] != "fusion_binary_train_provenance_v1"
    ):
        raise ValueError("real provenance receipt schema differs from v1")
    if (
        r["manifest_sha256"] != sha256(manifest_path)
        or r["operands_sha256"] != manifest["operands_sha256"]
    ):
        raise ValueError("receipt does not bind this manifest and operand archive")
    if (
        not isinstance(r["checks"], dict)
        or set(r["checks"]) != RECEIPT_CHECKS
        or any(v is not True for v in r["checks"].values())
    ):
        raise ValueError("receipt does not certify every required TRAIN/raw-boundary check")
    source = _receipt_file(
        r["source_weights"], receipt_path.parent, "frozen weights", source_weights_path
    )
    _receipt_file(r["base_gguf"], receipt_path.parent, "frozen base GGUF", base_gguf_path)
    if (
        r["source_weights"]["sha256"] != FROZEN_SOURCE_WEIGHTS_SHA256
        or r["base_gguf"]["sha256"] != FROZEN_BASE_GGUF_SHA256
        or manifest["source"]["frozen_weights_sha256"] != FROZEN_SOURCE_WEIGHTS_SHA256
        or manifest["source"]["base_gguf_sha256"] != FROZEN_BASE_GGUF_SHA256
    ):
        raise ValueError("real source files differ from frozen original BF16/F16 models")
    if w.shape != REAL_FUSION_SHAPE or x.shape[1] != REAL_FUSION_SHAPE[1]:
        raise ValueError("real fusion operands differ from frozen projection shape")
    import torch
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as original:
        original_weight = original.get_tensor("fc.weight")
        if (
            original_weight.dtype != torch.bfloat16
            or tuple(original_weight.shape) != REAL_FUSION_SHAPE
            or not np.array_equal(original_weight.float().numpy(), w)
        ):
            raise ValueError("reference_weight differs from original frozen BF16 fc.weight")
    del original_weight
    producer = r["producer"]
    producer_keys = {
        "contract",
        "source_files",
        "native_revision",
        "capture_manifest_sha256",
        "readiness_sha256",
    }
    if not isinstance(producer, dict) or set(producer) != producer_keys:
        raise ValueError("producer identity and capture/readiness hashes required")
    if (
        producer["contract"] != REAL_PRODUCER_CONTRACT
        or type(producer["native_revision"]) is not str
        or len(producer["native_revision"]) != 40
        or any(c not in "0123456789abcdef" for c in producer["native_revision"])
        or not is_hash(producer["capture_manifest_sha256"])
        or not is_hash(producer["readiness_sha256"])
        or producer["capture_manifest_sha256"] != manifest["source"]["capture_manifest_sha256"]
        or not isinstance(producer["source_files"], dict)
        or not producer["source_files"]
        or any(
            type(k) is not str or not k or not is_hash(v)
            for k, v in producer["source_files"].items()
        )
    ):
        raise ValueError("producer/capture identities differ from authenticated source evidence")
    for source_path, digest in producer["source_files"].items():
        local_source = (receipt_path.parent / source_path).resolve()
        if not local_source.is_file() or sha256(local_source) != digest:
            raise ValueError("producer source file missing locally or SHA256 differs")
    runtime_sources = [
        (receipt_path.parent / name).resolve()
        for name in producer["source_files"]
        if Path(name).name == "native-runtime-manifest.json"
    ]
    if len(runtime_sources) != 1:
        raise ValueError("one captured native runtime manifest must bind producer revision")
    runtime = json.loads(runtime_sources[0].read_text())
    if not isinstance(runtime, dict) or runtime.get("native_commit") != producer["native_revision"]:
        raise ValueError("producer native revision differs from captured runtime manifest")
    inventory_path = _receipt_file(r["train_inventory"], receipt_path.parent, "TRAIN inventory")
    inventory = json.loads(inventory_path.read_text())
    inventory_keys = {
        "schema",
        "source_train_prompts_sha256",
        "source_train_index_sha256",
        "prompts",
    }
    if (
        not isinstance(inventory, dict)
        or set(inventory) != inventory_keys
        or inventory["schema"] != "fusion_binary_train_inventory_v1"
        or any(
            not is_hash(inventory[k])
            for k in ("source_train_prompts_sha256", "source_train_index_sha256")
        )
        or any(
            inventory[k] not in producer["source_files"].values()
            for k in ("source_train_prompts_sha256", "source_train_index_sha256")
        )
        or not isinstance(inventory["prompts"], list)
        or not inventory["prompts"]
    ):
        raise ValueError("TRAIN inventory must bind original local prompt and index source hashes")
    eligible = {}
    for item in inventory["prompts"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"prompt_id", "prompt_sha256", "group_id", "source_split"}
            or type(item["prompt_id"]) is not str
            or not item["prompt_id"]
            or not is_hash(item["prompt_sha256"])
            or type(item["group_id"]) is not str
            or not item["group_id"]
            or item["source_split"] != "train"
            or item["prompt_id"] in eligible
        ):
            raise ValueError(
                "inventory contains duplicate, non-TRAIN or incomplete prompt identity"
            )
        eligible[item["prompt_id"]] = item
    prompt_sources = [
        (receipt_path.parent / name).resolve()
        for name, digest in producer["source_files"].items()
        if digest == inventory["source_train_prompts_sha256"]
    ]
    original_prompts = [
        json.loads(line) for line in prompt_sources[0].read_text().splitlines() if line.strip()
    ]
    original_by_id = {item.get("id"): item for item in original_prompts if isinstance(item, dict)}
    if len(original_by_id) != len(original_prompts) or None in original_by_id:
        raise ValueError("original TRAIN prompt source has missing or duplicate identities")
    for item in inventory["prompts"]:
        original_prompt = original_by_id.get(item["prompt_id"])
        if original_prompt is None or not isinstance(original_prompt.get("messages"), list):
            raise ValueError("inventory prompt missing original TRAIN messages")
        content = json.dumps(
            original_prompt["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(content).hexdigest() != item["prompt_sha256"]:
            raise ValueError("TRAIN prompt content differs from inventory identity")
    index_sources = [
        (receipt_path.parent / name).resolve()
        for name, digest in producer["source_files"].items()
        if digest == inventory["source_train_index_sha256"]
    ]
    index_rows = [
        json.loads(line) for line in index_sources[0].read_text().splitlines() if line.strip()
    ]
    original_index = {item.get("id"): item for item in index_rows if isinstance(item, dict)}
    if len(original_index) != len(index_rows) or None in original_index:
        raise ValueError("original TRAIN index has missing or duplicate identities")
    for item in inventory["prompts"]:
        index_row = original_index.get(item["prompt_id"])
        if (
            index_row is None
            or index_row.get("group") != item["group_id"]
            or index_row.get("content_sha256") != item["prompt_sha256"]
        ):
            raise ValueError("inventory group/content/membership differs from original TRAIN index")
    selected = r["selected_prompts"]
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected TRAIN prompts required")
    selected_map = {}
    for prompt in selected:
        if (
            not isinstance(prompt, dict)
            or set(prompt) != {"prompt_id", "prompt_sha256", "group_id", "split"}
            or type(prompt["prompt_id"]) is not str
            or not prompt["prompt_id"]
            or not is_hash(prompt["prompt_sha256"])
            or type(prompt["group_id"]) is not str
            or not prompt["group_id"]
            or prompt["split"] not in REAL_PROMPT_COUNTS
            or prompt["prompt_id"] in selected_map
        ):
            raise ValueError("selected prompt identity/group/split is invalid")
        allowed = eligible.get(prompt["prompt_id"])
        if allowed is None or any(prompt[k] != allowed[k] for k in ("prompt_sha256", "group_id")):
            raise ValueError("selected prompt absent from authenticated TRAIN inventory")
        selected_map[prompt["prompt_id"]] = prompt
    groups = [
        {p["group_id"] for p in selected if p["split"] == split} for split in REAL_PROMPT_COUNTS
    ]
    if groups[0] & groups[1] or any(
        sum(p["split"] == split for p in selected) != count
        for split, count in REAL_PROMPT_COUNTS.items()
    ):
        raise ValueError("real fit/validation requires 8/4 distinct TRAIN prompt groups")
    rows = manifest["rows"]
    counts = {pid: 0 for pid in selected_map}
    for row in rows:
        selected_prompt = selected_map.get(row["prompt_id"])
        if selected_prompt is None or any(
            row[k] != selected_prompt[k] for k in ("prompt_sha256", "split")
        ):
            raise ValueError("manifest prompt split/content differs from trusted receipt")
        counts[row["prompt_id"]] += 1
    if any(count != REAL_ROWS_PER_PROMPT for count in counts.values()):
        raise ValueError("real package requires exactly 32 raw rows per selected prompt")
    if not isinstance(r["rows"], list) or len(r["rows"]) != len(rows):
        raise ValueError("receipt must bind every selected raw feature row")
    for row, evidence in zip(rows, r["rows"], strict=True):
        if (
            not isinstance(evidence, dict)
            or set(evidence)
            != {
                "row_id",
                "raw_input_sha256",
                "feature_row",
                "feature_metadata",
                "prompt_token_ids",
                "anchor",
                "native_events",
            }
            or evidence["row_id"] != row["row_id"]
            or evidence["raw_input_sha256"] != row["raw_input_sha256"]
            or type(evidence["feature_row"]) is not int
            or evidence["feature_row"] < 0
            or not isinstance(evidence["feature_metadata"], dict)
            or not evidence["feature_metadata"]
            or not isinstance(evidence["anchor"], dict)
            or not evidence["anchor"]
            or not isinstance(evidence["prompt_token_ids"], list)
            or not evidence["prompt_token_ids"]
            or any(
                type(token) is not int or not 0 <= token < 151936
                for token in evidence["prompt_token_ids"]
            )
        ):
            raise ValueError("raw feature row/anchor/token ancestry is incomplete or differs")
        metadata, anchor, events = (
            evidence[k] for k in ("feature_metadata", "anchor", "native_events")
        )
        prefix = metadata.get("prefix_token_ids")
        anchor_prefix = anchor.get("prefix_token_ids")
        if (
            metadata.get("feature_row") != evidence["feature_row"]
            or metadata.get("prompt_id") != row["prompt_id"]
            or metadata.get("position") != row["position"]
            or row["depth"] != 0
            or metadata.get("tap_ids") != [2, 18, 33]
            or metadata.get("boundary") != "native_target_block_inputs_concat_before_draft_fc"
            or metadata.get("source") != "native_target_features_on_accepted_prefix"
            or metadata.get("accepted_prefix") is not True
            or evidence["prompt_token_ids"] != prefix
            or not isinstance(prefix, list)
            or len(prefix) != row["position"] + 1
            or any(type(token) is not int or not 0 <= token < 151936 for token in prefix)
            or not isinstance(anchor_prefix, list)
            or anchor_prefix[: len(prefix)] != prefix
            or any(type(token) is not int or not 0 <= token < 151936 for token in anchor_prefix)
            or anchor.get("prompt_id") != row["prompt_id"]
            or anchor.get("split") != "train"
            or type(anchor.get("round_index")) is not int
            or anchor["round_index"] < 0
            or type(anchor.get("seed_token_id")) is not int
            or not 0 <= anchor["seed_token_id"] < 151936
            or anchor_prefix[: len(evidence["prompt_token_ids"])] != evidence["prompt_token_ids"]
            or not isinstance(events, dict)
            or set(events) != {"decoded_row", "disposition"}
        ):
            raise ValueError(
                "raw feature prefix, tap ordering, TRAIN anchor or native event join differs"
            )
        decoded, disposition = events["decoded_row"], events["disposition"]
        if (
            not isinstance(decoded, dict)
            or not isinstance(disposition, dict)
            or decoded.get("schema") != "eagle_target_feature_v1"
            or decoded.get("event") != "decoded_row"
            or disposition.get("schema") != "eagle_target_feature_v1"
            or disposition.get("event") != "disposition"
            or type(metadata.get("native_feature_row")) is not int
            or metadata["native_feature_row"] < 0
            or decoded.get("feature_row") != metadata["native_feature_row"]
            or disposition.get("feature_row") != metadata["native_feature_row"]
            or decoded.get("decode_ordinal") != metadata.get("native_decode_ordinal")
            or type(metadata.get("native_decode_ordinal")) is not int
            or metadata["native_decode_ordinal"] < 0
            or type(metadata.get("task_id")) is not int
            or metadata["task_id"] < 0
            or decoded.get("task_id") != metadata["task_id"]
            or disposition.get("task_id") != metadata["task_id"]
            or decoded.get("position") != row["position"]
            or decoded.get("prefix_token_ids") != prefix
            or decoded.get("target_layer_ids") != [2, 18, 33]
            or decoded.get("feature_dim") != REAL_FUSION_SHAPE[1]
            or decoded.get("feature_dtype") != "float32_native_endian"
            or decoded.get("boundary") != "raw_target_layer_input_before_eagle_encoder"
            or decoded.get("source") != "target_verifier"
            or disposition.get("retained_input") is not True
        ):
            raise ValueError(
                "decoded/disposition native row did not retain the joined target feature"
            )
        phase = decoded.get("phase")
        if phase not in ("prefill", "target_only", "speculative"):
            raise ValueError("unknown native target-feature decode phase")
        if phase == "speculative":
            accepted, proposal = disposition.get("accepted_drafts"), decoded.get("spec_input_row")
            if (
                type(accepted) is not int
                or accepted < 0
                or type(proposal) is not int
                or proposal < 0
                or proposal > accepted
                or decoded.get("round_index") != disposition.get("round_index")
            ):
                raise ValueError("native speculative feature was not in the accepted prefix")
    return {
        "provenance_receipt_sha256": receipt_hash,
        "provenance_schema": r["schema"],
        "prompt_token_ids_contract": "exact_native_feature_prefix_including_prefill_rows",
        "train_inventory_sha256": r["train_inventory"]["sha256"],
        "producer": producer,
    }


def load_config(path):
    value = json.loads(Path(path).read_text())
    default = json.loads((ROOT / "configs/fusion_binary_discrete_a8.json").read_text())
    if (
        not isinstance(value, dict)
        or set(value) != set(default)
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("configuration fields differ from v1")
    for key in ("activation_bits", "activation_contract", "teacher_contract", "scale_layout"):
        if value[key] != default[key]:
            raise ValueError(f"unsupported contract {key}")
    limits = {
        "alternating_passes": 2,
        "scans_per_pass": 2,
        "max_flips_per_row": 32,
        "max_proposals_per_row_per_scan": 64,
        "row_batch": 64,
        "max_samples": 3072,
        "max_workspace_bytes": 8 * 1024**3,
        "max_cpu_seconds": 43200,
        "scale_neighbor_steps": 16,
    }
    for key, cap in limits.items():
        if type(value[key]) is not int or not 1 <= value[key] <= cap:
            raise ValueError(f"{key} must be positive integer <= {cap}")
    for key in ("improvement_relative_margin", "improvement_absolute_margin"):
        if type(value[key]) not in (int, float) or not np.isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"invalid {key}")
    return value


def load_operands(
    path,
    config,
    *,
    source_weights_path=None,
    base_gguf_path=None,
    provenance_receipt_path=None,
    provenance_receipt_sha256=None,
):
    path = Path(path)
    m = json.loads(path.read_text())
    keys = {
        "schema_version",
        "projection",
        "raw_input_stage",
        "source_data_split",
        "eligibility",
        "synthetic",
        "source",
        "operands",
        "operands_sha256",
        "rows",
    }
    if (
        not isinstance(m, dict)
        or set(m) != keys
        or type(m["schema_version"]) is not int
        or m["schema_version"] != 1
    ):
        raise ValueError("operand manifest differs from v1")
    if (m["projection"], m["raw_input_stage"], m["source_data_split"], m["eligibility"]) != (
        "fc",
        "pre_activation_quantization",
        "train",
        "training_allowed",
    ):
        raise ValueError("only explicitly train-eligible raw fusion inputs are allowed")
    if (
        type(m["synthetic"]) is not bool
        or not isinstance(m["source"], dict)
        or set(m["source"]) != SOURCE_KEYS
    ):
        raise ValueError("synthetic flag and exact source identities required")
    if not all(is_hash(v) for v in m["source"].values()) or not is_hash(m["operands_sha256"]):
        raise ValueError("source/archive hashes must be SHA256")
    if type(m["operands"]) is not str:
        raise ValueError("operand archive path required")
    p = (path.parent / m["operands"]).resolve()
    if sha256(p) != m["operands_sha256"]:
        raise ValueError("operand archive hash mismatch")
    # Zip headers checked before decompression; avoid silently exceeding memory budget.
    import zipfile

    with zipfile.ZipFile(p) as z:
        if sum(info.file_size for info in z.infolist()) > config["max_workspace_bytes"] // 4:
            raise ValueError("operand archive exceeds bounded memory budget")
    with np.load(p, allow_pickle=False) as z:
        if len(z.files) != 3 or set(z.files) != {"raw_input", "reference_weight", "raw_join_ids"}:
            raise ValueError("archive requires raw input, original weights and join IDs")
        arrays = {key: z[key].copy() for key in z.files}
    x, w, ids = (arrays[k] for k in ("raw_input", "reference_weight", "raw_join_ids"))
    for name, a in (("raw_input", x), ("reference_weight", w)):
        if a.ndim != 2 or min(a.shape) < 1 or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a finite nonempty F32 matrix")
    if x.shape[1] != w.shape[1] or len(x) > config["max_samples"] or x.shape[1] * 127 >= 2**24:
        raise ValueError("input widths, sample budget or exact-integer-dot bound violated")
    # Conservative allocation estimate includes duplicate casts, outputs, correlation block,
    # original/current/control signs, the archive and row metadata. No KxK Gram is allocated.
    estimated = (
        16 * x.nbytes
        + 8 * w.nbytes
        + 24 * len(x) * len(w)
        + 16 * config["row_batch"] * w.shape[1]
        + len(w) * (config["max_flips_per_row"] * 1024 + 2048)
    )
    if estimated > config["max_workspace_bytes"]:
        raise ValueError("estimated fitter workspace exceeds budget")
    rows = m["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(x)
        or ids.shape != (len(x),)
        or ids.dtype.kind != "U"
    ):
        raise ValueError("one Unicode join ID and manifest row per input required")
    owners, row_ids, coordinates = {}, set(), set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("row metadata requires exact prompt/hash/split/coordinate fields")
        if any(type(row[k]) is not str or not row[k] for k in ("row_id", "prompt_id")):
            raise ValueError("row and prompt IDs must be nonempty strings")
        if any(type(row[k]) is not int or row[k] < 0 for k in ("depth", "position")):
            raise ValueError("depth/position must be nonnegative integers")
        if row["split"] not in ("train", "validation") or not is_hash(row["prompt_sha256"]):
            raise ValueError("only train/validation rows with prompt content identity allowed")
        coordinate = (row["prompt_id"], row["depth"], row["position"])
        if row["row_id"] in row_ids or coordinate in coordinates or str(ids[i]) != row["row_id"]:
            raise ValueError("duplicate or mismatched raw input join")
        row_ids.add(row["row_id"])
        coordinates.add(coordinate)
        owner = (row["prompt_sha256"], row["split"])
        if owners.setdefault(row["prompt_id"], owner) != owner:
            raise ValueError("prompt alias crosses content identity or split")
        if row["raw_input_sha256"] != array_hash(x[i].astype("<f4")):
            raise ValueError("raw input row hash mismatch")
    train = np.array([r["split"] == "train" for r in rows])
    hashes = [
        {r["prompt_sha256"] for r in rows if r["split"] == split}
        for split in ("train", "validation")
    ]
    if not all(hashes) or hashes[0] & hashes[1]:
        raise ValueError("fit and validation require disjoint prompt contents")
    provenance = {}
    if not m["synthetic"]:
        provenance = verify_real_receipt(
            provenance_receipt_path,
            provenance_receipt_sha256,
            path,
            m,
            x,
            w,
            source_weights_path=source_weights_path,
            base_gguf_path=base_gguf_path,
        )
    return (
        x,
        w,
        train,
        m,
        {
            "manifest_sha256": sha256(path),
            "operands_sha256": sha256(p),
            "estimated_workspace_bytes": estimated,
            **provenance,
        },
    )


def quantize_a8(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("raw A8 operand must be finite 2D F32")
    limit = np.max(np.abs(raw), axis=1)
    beta = np.divide(limit, np.float32(127), dtype=np.float32)
    inv = np.divide(np.float32(127), limit, out=np.zeros_like(limit), where=limit > 0)
    if not np.isfinite(inv).all():
        raise ValueError(
            "native fixed A8 reciprocal overflows on tiny operand: no arithmetic substitution"
        )
    normalized = np.multiply(raw, inv[:, None], dtype=np.float32)
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int16)
    return codes, beta


def integer_dots(codes, signs):
    # Every product/partial sum is an exact F32 integer below 2**24, including BLAS.
    if codes.shape[1] * 127 >= 2**24:
        raise ValueError("integer signed sum cannot be exactly represented in F32")
    return (codes.astype(np.float32) @ signs.T.astype(np.float32)).astype(np.int32)


def native_output(dots, scales, beta):
    return np.multiply(
        np.multiply(
            dots.astype(np.float32), np.asarray(scales, dtype=np.float32), dtype=np.float32
        ),
        np.asarray(beta, dtype=np.float32),
        dtype=np.float32,
    )


def sse(prediction, teacher):
    r = prediction.astype(np.float64) - teacher.astype(np.float64)
    return float(np.sum(r * r, dtype=np.float64))


def margin(loss, config):
    return max(config["improvement_absolute_margin"], config["improvement_relative_margin"] * loss)


def solve_scale(dots, beta, teacher, previous, config):
    """Exact unregularized continuous row LS, then finite exported-point descent.

    Zero design returns +0. Exported F32 products can perturb this continuous
    optimum: check adjacent scales and previous scale, descend up to 16 ULPs.
    A material unresolved numeric gain fails closed, rather than crediting signs.
    """
    z = dots.astype(np.float64) * beta.astype(np.float64)
    denom = float(z @ z)
    optimum = max(0.0, float(z @ teacher.astype(np.float64)) / denom) if denom else 0.0
    if not np.isfinite(optimum) or optimum > np.finfo(np.float32).max:
        raise ValueError("nonfinite scale optimum")
    center = np.float32(optimum)
    candidates = [center, np.float32(previous), np.float32(0)]
    best = min(candidates, key=lambda a: sse(native_output(dots, a, beta), teacher))
    loss = sse(native_output(dots, best, beta), teacher)
    steps = 0
    for steps in range(config["scale_neighbor_steps"]):
        neighbors = [np.nextafter(best, np.float32(0)), np.nextafter(best, np.float32(np.inf))]
        for a in neighbors:
            if np.isfinite(a) and a >= 0:
                candidate_loss = sse(native_output(dots, a, beta), teacher)
                if candidate_loss < loss:
                    best, loss = a, candidate_loss
        # Compare with old nextafter neighborhood; if no descent, finite local convergence.
        fresh = [np.nextafter(best, np.float32(0)), np.nextafter(best, np.float32(np.inf))]
        gain = max(
            [loss - sse(native_output(dots, a, beta), teacher) for a in fresh if np.isfinite(a)]
            + [0]
        )
        if gain <= margin(loss, config):
            break
    else:
        if gain > margin(loss, config):
            raise ValueError("scale exported-neighbor gate did not converge within budget")
    return np.float32(best), {
        "continuous_optimum": optimum,
        "exported_scale_f32": float(best),
        "continuous_derivative": float(2 * z @ (z * optimum - teacher)),
        "continuous_kkt_relative": float(
            abs(min(0.0, float(z @ (z * optimum - teacher))))
            if optimum == 0
            else abs(float(z @ (z * optimum - teacher)))
        )
        / max(float(np.linalg.norm(z) * np.linalg.norm(teacher)), np.finfo(np.float64).tiny),
        "finite_sse": loss,
        "neighbor_steps": steps + 1,
        "neighbor_gain": gain,
    }


def fit(codes, beta, teacher, weight, config):
    """Train-only row-independent finite-objective descent, at most four scans."""
    started = time.monotonic()
    if (
        codes.ndim != 2
        or beta.shape != (len(codes),)
        or weight.ndim != 2
        or teacher.shape != (len(codes), len(weight))
        or weight.shape[1] != codes.shape[1]
        or not np.isfinite(teacher).all()
        or not np.isfinite(weight).all()
        or not np.isfinite(beta).all()
        or np.any(beta < 0)
        or codes.dtype.kind not in "iu"
        or np.any(np.abs(codes.astype(np.int64)) > 127)
    ):
        raise ValueError("invalid finite fitting operands/shapes/codes")
    signs = np.where(weight < 0, -1, 1).astype(np.int8)  # +/-0 are positive
    # Match existing legacy initializer's CPU F32 mean, including row chunking.
    import torch

    initializer_scales = np.empty(len(weight), dtype=np.float32)
    for first in range(0, len(weight), 128):
        initializer_scales[first : first + 128] = (
            torch.from_numpy(weight[first : first + 128]).abs().mean(1).numpy()
        )

    def check_budget():
        if time.monotonic() - started > config["max_cpu_seconds"]:
            raise TimeoutError("bounded CPU fitting budget exhausted")

    initial_signs = signs.copy()
    scales, controls = initializer_scales.copy(), []
    dots = integer_dots(codes, signs)
    for row in range(len(weight)):
        check_budget()
        scales[row], detail = solve_scale(dots[:, row], beta, teacher[:, row], scales[row], config)
        controls.append(detail)
    control_scales = scales.copy()
    flip_counts = np.zeros(len(weight), dtype=np.int32)
    events, scan_summary, scale_updates = [], [], []
    x = codes.astype(np.float64) * beta[:, None].astype(np.float64)
    diagonal = np.sum(x * x, axis=0)
    for pass_id in range(config["alternating_passes"]):
        for scan_id in range(config["scans_per_pass"]):
            accepted = proposals = 0
            for first in range(0, len(weight), config["row_batch"]):
                check_budget()
                stop = min(first + config["row_batch"], len(weight))
                residual = (
                    native_output(dots[:, first:stop], scales[first:stop], beta[:, None]).astype(
                        np.float64
                    )
                    - teacher[:, first:stop]
                )
                correlations = residual.T @ x
                for local, row in enumerate(range(first, stop)):
                    if flip_counts[row] >= config["max_flips_per_row"] or scales[row] == 0:
                        continue
                    delta = -2 * float(scales[row]) * signs[row].astype(np.float64)
                    estimate = 2 * delta * correlations[local] + delta * delta * diagonal
                    indices = np.flatnonzero(estimate < 0)
                    indices = indices[np.argsort(estimate[indices], kind="stable")][
                        : config["max_proposals_per_row_per_scan"]
                    ]
                    loss = sse(native_output(dots[:, row], scales[row], beta), teacher[:, row])
                    for col in indices:
                        if flip_counts[row] >= config["max_flips_per_row"]:
                            break
                        proposals += 1
                        # Re-evaluate stale-ranked proposals against updated exact integer dots.
                        trial_dots = dots[:, row] - 2 * int(signs[row, col]) * codes[:, col].astype(
                            np.int32
                        )
                        trial_loss = sse(
                            native_output(trial_dots, scales[row], beta), teacher[:, row]
                        )
                        if trial_loss < loss - margin(loss, config):
                            events.append(
                                {
                                    "pass": pass_id,
                                    "scan": scan_id,
                                    "row": row,
                                    "column": int(col),
                                    "scale_f32": float(scales[row]),
                                    "before_sse": loss,
                                    "after_sse": trial_loss,
                                }
                            )
                            dots[:, row] = trial_dots
                            signs[row, col] *= -1
                            flip_counts[row] += 1
                            accepted += 1
                            loss = trial_loss
            scan_summary.append(
                {"pass": pass_id, "scan": scan_id, "proposals": proposals, "accepted": accepted}
            )
        for row in range(len(weight)):
            check_budget()
            before = sse(native_output(dots[:, row], scales[row], beta), teacher[:, row])
            scales[row], solved = solve_scale(
                dots[:, row], beta, teacher[:, row], scales[row], config
            )
            scale_updates.append(
                {
                    "pass": pass_id,
                    "row": row,
                    "before_sse": before,
                    "after_sse": solved["finite_sse"],
                    "scale_f32": float(scales[row]),
                }
            )
    # Recompute dots rather than trusting incremental updates before candidate freeze.
    if not np.array_equal(integer_dots(codes, signs), dots):
        raise ValueError("incremental sign operands differ from frozen reconstruction")
    final_loss = sse(native_output(dots, scales, beta[:, None]), teacher)
    control_loss = sse(
        native_output(integer_dots(codes, initial_signs), control_scales, beta[:, None]), teacher
    )
    if final_loss > control_loss + margin(control_loss, config):
        raise ValueError("frozen candidate worsens actual finite training objective")
    return (
        signs,
        scales,
        {
            "initializer_signs": initial_signs,
            "initializer_scales": initializer_scales,
            "scale_only_scales": control_scales,
            "control_solver": controls,
            "flip_events": events,
            "scans": scan_summary,
            "scale_updates": scale_updates,
            "flip_counts": flip_counts.tolist(),
            "fit_seconds": time.monotonic() - started,
        },
    )


def diagnostics(prediction, teacher):
    p, y = prediction.astype(np.float64), teacher.astype(np.float64)
    error, energy = sse(p, y), float(np.sum(y * y))
    norms = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
    valid = norms > 0
    cosine = np.sum(p * y, axis=1)[valid] / norms[valid]
    return {
        "sse": error,
        "mse": error / p.size,
        "relative_squared_error": error / energy if energy else None,
        "mean_row_cosine": float(np.mean(cosine)) if len(cosine) else None,
        "zero_norm_rows": int((~valid).sum()),
        "coordinate_sign_agreement": float(np.mean((p >= 0) == (y >= 0))),
        "max_coordinate_agreement": float(np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1))),
        "decision_scope": (
            "fusion output coordinate sign/argmax only; not vocabulary logits or acceptance"
        ),
    }


def pack_signs(signs):
    if signs.ndim != 2 or not np.all((signs == 1) | (signs == -1)):
        raise ValueError("sign matrix must contain only +/-1")
    positive = signs > 0
    if signs.shape[1] % 32:
        positive = np.pad(positive, ((0, 0), (0, (-signs.shape[1]) % 32)))
    return np.ascontiguousarray(np.packbits(positive, axis=1, bitorder="little")).view("<i4")


def unpack_signs(packed, width):
    bits = np.unpackbits(np.ascontiguousarray(packed).view(np.uint8), axis=1, bitorder="little")
    if np.any(bits[:, width:]):
        raise ValueError("nonzero packed tail bits")
    return np.where(bits[:, :width], 1, -1).astype(np.int8)


def export_candidate(base_path, output_path, signs, scales, *, expected_base_sha256):
    """Existing native v2 fusion-only binary representation; preserve other operands."""
    candidates = [ROOT / "third_party/llama.cpp/gguf-py"]
    if os.environ.get("EAGLE_GGUF_PY"):
        candidates.insert(0, Path(os.environ["EAGLE_GGUF_PY"]))
    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            break
    from gguf import GGMLQuantizationType as Type
    from gguf import GGUFReader, GGUFWriter

    base_path, output_path = Path(base_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("candidate export destination already exists")
    if sha256(base_path) != expected_base_sha256:
        raise ValueError("base GGUF differs from source manifest")
    if (
        scales.shape != (len(signs),)
        or scales.dtype != np.float32
        or not np.isfinite(scales).all()
        or np.any(scales < 0)
        or np.any(np.signbit(scales) & (scales == 0))
    ):
        raise ValueError("export requires finite nonnegative row F32 scales")
    reader = GGUFReader(base_path)
    prefix = "eagle3.w1a1."
    if reader.byte_order != "I" or reader.fields["general.architecture"].contents() != "eagle3":
        raise ValueError("base must be little-endian eagle3 GGUF")
    if any(
        k.startswith((prefix, "eagle3.fusion_correction.", "eagle3.affine_weights."))
        for k in reader.fields
    ):
        raise ValueError("export requires original base without learned/binary metadata")
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or "fc.weight" not in tensors:
        raise ValueError("duplicate tensors or missing fusion source")
    if (
        tensors["fc.weight"].data.shape != signs.shape
        or tensors["fc.weight"].tensor_type != Type.F16
    ):
        raise ValueError("fusion source must have original F16 shape")
    packed = pack_signs(signs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fusion-discrete-export-", dir=output_path.parent
    ) as tmp:
        temp = Path(tmp) / "candidate.gguf"
        writer = GGUFWriter(temp, "eagle3")
        for key, field in reader.fields.items():
            if not key.startswith("GGUF.") and key != "general.architecture":
                writer.add_key_value(
                    key,
                    field.contents(),
                    field.types[0],
                    field.types[-1] if len(field.types) > 1 else None,
                )
        writer.add_uint32(prefix + "version", 2)
        writer.add_uint32(prefix + "scale_group_size", 0)
        writer.add_uint32(prefix + "activation_bits", 8)
        writer.add_array(prefix + "groups", ["fusion"])
        writer.add_array(prefix + "tensors", ["fc.weight"])
        writer.add_string(prefix + "bit_order", "little")
        writer.add_string(prefix + "sign_rule", "nonnegative_is_one")
        writer.add_string(prefix + "scale_rule", "f32_nonnegative_least_squares")
        writer.add_string(prefix + "arithmetic", "f32")
        writer.add_uint32(prefix + "tensor.fc_weight.logical_k", signs.shape[1])
        writer.add_string(prefix + "tensor.fc_weight.packed", "fc.w1a1_packed")
        writer.add_string(prefix + "tensor.fc_weight.scale", "fc.w1a1_scale")
        for tensor in reader.tensors:
            if tensor.name != "fc.weight":
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.add_tensor("fc.w1a1_packed", packed, raw_dtype=Type.I32)
        writer.add_tensor("fc.w1a1_scale", scales, raw_dtype=Type.F32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reread = GGUFReader(temp)
        actual = {t.name: t for t in reread.tensors}
        if set(actual) != (set(tensors) - {"fc.weight"}) | {"fc.w1a1_packed", "fc.w1a1_scale"}:
            raise ValueError("export tensor inventory mismatch")
        for name, source in tensors.items():
            if name != "fc.weight":
                copy = actual[name]
                if (
                    source.tensor_type != copy.tensor_type
                    or not np.array_equal(source.shape, copy.shape)
                    or array_hash(source.data) != array_hash(copy.data)
                ):
                    raise ValueError(f"nonfusion operand changed: {name}")
        for name, source in reader.fields.items():
            if not name.startswith("GGUF.") and name != "general.architecture":
                if source.contents() != reread.fields[name].contents():
                    raise ValueError(f"original metadata changed: {name}")
        if (
            actual["fc.w1a1_packed"].tensor_type != Type.I32
            or actual["fc.w1a1_scale"].tensor_type != Type.F32
        ):
            raise ValueError("export signs/scales storage precision mismatch")
        if not np.array_equal(actual["fc.w1a1_packed"].data, packed) or not np.array_equal(
            actual["fc.w1a1_scale"].data, scales
        ):
            raise ValueError("export/reload changes signs or row scales")
        if not np.array_equal(unpack_signs(actual["fc.w1a1_packed"].data, signs.shape[1]), signs):
            raise ValueError("packed sign reconstruction mismatch")
        os.replace(temp, output_path)
    return {
        "sha256": sha256(output_path),
        "base_gguf_sha256": expected_base_sha256,
        "unchanged_nonfusion_tensors": len(tensors) - 1,
        "packed_sha256": array_hash(packed),
        "scale_sha256": array_hash(scales),
        "fusion_only": True,
        "native_validation": "deferred",
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError("fit output directory already exists; preserve previous artifacts")
    config = load_config(args.config)
    x, w, train, manifest, identities = load_operands(
        args.manifest,
        config,
        source_weights_path=args.source_weights,
        base_gguf_path=args.base_gguf,
        provenance_receipt_path=getattr(args, "provenance_receipt", None),
        provenance_receipt_sha256=getattr(args, "provenance_receipt_sha256", None),
    )
    # Only training operands enter fit; validation is quantized/evaluated after freeze.
    codes, beta = quantize_a8(x[train])
    teacher = x[train] @ w.T  # Frozen F32 BLAS reconstruction teacher, no model execution.
    signs, scales, detail = fit(codes, beta, teacher, w, config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate = args.output_dir / "fusion_candidate.npz"
    np.savez(
        candidate,
        **{
            "fc.latent": signs.astype(np.float32),
            "fc.scale": scales,
            "fc.w1a1_packed": pack_signs(signs),
        },
    )
    frozen_hash = sha256(candidate)
    control_path = args.output_dir / "control_scales.npz"
    np.savez(
        control_path,
        initializer_scale=detail["initializer_scales"],
        scale_only_scale=detail["scale_only_scales"],
    )
    frozen_control_hash = sha256(control_path)
    with np.load(control_path, allow_pickle=False) as control:
        if not np.array_equal(
            control["initializer_scale"], detail["initializer_scales"]
        ) or not np.array_equal(control["scale_only_scale"], detail["scale_only_scales"]):
            raise ValueError("matched control scales changed on reload")
    # Verify reload before any held-out result is visible.
    with np.load(candidate, allow_pickle=False) as z:
        if (
            not np.array_equal(z["fc.latent"], signs)
            or not np.array_equal(z["fc.scale"], scales)
            or not np.array_equal(unpack_signs(z["fc.w1a1_packed"], w.shape[1]), signs)
        ):
            raise ValueError("candidate checkpoint operands changed on reload")
    metrics, operand_hashes = {}, {}
    for split, selection in (("train", train), ("validation", ~train)):
        q, b = quantize_a8(x[selection])
        y = x[selection] @ w.T
        if not np.isfinite(y).all():
            raise ValueError("frozen teacher output overflowed F32")
        operand_hashes[split] = {
            "codes_sha256": array_hash(q),
            "token_scale_sha256": array_hash(b),
            "teacher_sha256": array_hash(y),
        }
        metrics[split] = {
            name: diagnostics(native_output(integer_dots(q, s), a, b[:, None]), y)
            for name, s, a in (
                (
                    "original_binary_initializer",
                    detail["initializer_signs"],
                    detail["initializer_scales"],
                ),
                ("converged_scale_only", detail["initializer_signs"], detail["scale_only_scales"]),
                ("sign_and_scale", signs, scales),
            )
        }
    if sha256(candidate) != frozen_hash or sha256(control_path) != frozen_control_hash:
        raise ValueError("frozen checkpoint/control scales changed during validation")
    export = None
    if args.base_gguf:
        export = export_candidate(
            args.base_gguf,
            args.output_dir / "fusion_candidate.gguf",
            signs,
            scales,
            expected_base_sha256=manifest["source"]["base_gguf_sha256"],
        )
    detail.pop("initializer_signs")
    detail.pop("initializer_scales")
    detail.pop("scale_only_scales")
    report = {
        "schema_version": 1,
        "synthetic": manifest["synthetic"],
        "source": manifest["source"],
        **identities,
        "script_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "config": config,
        "raw_input_sha256": array_hash(x),
        "reference_weight_sha256": array_hash(w),
        "train_rows": int(train.sum()),
        "validation_rows": int((~train).sum()),
        "train_prompt_hashes": sorted(
            {r["prompt_sha256"] for r in manifest["rows"] if r["split"] == "train"}
        ),
        "validation_prompt_hashes": sorted(
            {r["prompt_sha256"] for r in manifest["rows"] if r["split"] == "validation"}
        ),
        "frozen_candidate_sha256": frozen_hash,
        "control_scales_sha256": frozen_control_hash,
        "export": export,
        "metrics": metrics,
        "operand_hashes": operand_hashes,
        "fit": detail,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": "CPU",
            "numpy": np.__version__,
        },
        "limitations": (
            "F32-BLAS frozen-layer reconstruction teacher; native A8 exported candidate "
            "arithmetic. Fusion-coordinate decisions are not vocabulary logits. "
            "No native acceptance or GPU throughput evaluated."
        ),
    }
    (args.output_dir / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/fusion_binary_discrete_a8.json")
    p.add_argument(
        "--source-weights",
        type=Path,
        help=("Exact original BF16 safetensors file for receipt-verified real data"),
    )
    p.add_argument(
        "--provenance-receipt", type=Path, help="Authenticated real-data acquisition receipt"
    )
    p.add_argument(
        "--provenance-receipt-sha256", help="Receipt hash independently pinned by run owner"
    )
    p.add_argument("--base-gguf", type=Path, help="Original F16 GGUF for fusion-only export")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "synthetic": report["synthetic"],
                "metrics": report["metrics"],
                "candidate_sha256": report["frozen_candidate_sha256"],
                "export": report["export"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
