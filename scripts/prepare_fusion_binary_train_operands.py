#!/usr/bin/env python3
"""Acquire saved TRAIN operands read-only, then authenticate a bounded local package.

Remote mode writes only a ZIP to stdout. Run it via tmux MCP SSH stdin; it never
opens a model, starts a capture, imports Torch, or writes to the remote filesystem.
Local mode builds original BF16 reconstruction operands and a separately pinned
provenance receipt. The receipt is evidence, not a generic TRAIN permission flag.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

CONTRACT = (
    "native_target_block_inputs_concat_before_draft_fc_f32_taps_2_18_33_no_upstream_cast_or_norm"
)
READY_SHA = "bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498"
WEIGHT_SHA = "58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e"
BASE_SHA = "c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1"
BOUNDARY = "native_target_block_inputs_concat_before_draft_fc"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stable_stat(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def records(raw):
    return [json.loads(line) for line in raw.decode().split("\n") if line.strip()]


def message_hash(prompt):
    return digest(
        json.dumps(
            prompt["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def acquire(config, output):
    """Authenticate one immutable completed shard and stream selected rows only."""
    root = Path(config["remote_root"])
    saved = {}
    hashes = {}
    total_reads = 0

    def read(name, path, expected=None):
        nonlocal total_reads
        p = Path(path)
        before = p.stat()
        total_reads += before.st_size
        require(total_reads <= config["max_remote_read_bytes"], "remote read cap exceeded")
        raw = p.read_bytes()
        require(stable_stat(p.stat()) == stable_stat(before), "source changed during read")
        h = digest(raw)
        require(expected is None or h == expected, "source hash differs: " + str(p))
        saved[name] = raw
        hashes[str(p)] = h
        return raw

    binding = json.loads(read("provider-binding.json", root / config["binding_relative_path"]))
    require(
        binding["prepared_ready_sha256"] == config["prepared_ready_sha256"] == READY_SHA,
        "completed receipt pin differs",
    )
    prepared = Path(binding["prepared_run_dir"])
    ready = json.loads(
        read("preparation-ready.json", prepared / "preparation-ready.json", READY_SHA)
    )
    require(
        ready["schema"] == "continuous_w1ax_preparation_ready_v1"
        and ready["preparation_complete"] is True
        and ready["optimization_started"] is False
        and ready["step"] == 0,
        "not completed frozen preparation",
    )
    source = ready["teacher_coverage"]["source"]
    index = json.loads(
        read(
            "train-providers.json",
            prepared / "stages/train-providers.json",
            source["execution_manifest_sha256"],
        )
    )
    require(
        index["split"] == "train" and index["training_eligible"] is True,
        "full TRAIN admission absent",
    )
    record = index["shards"][config["shard_ordinal"]]
    require(
        record["provider_manifest_sha256"]
        == source["shards"][config["shard_ordinal"]]["provider_manifest_sha256"],
        "completed shard binding differs",
    )
    provider = json.loads(
        read("provider.json", record["provider_manifest"], record["provider_manifest_sha256"])
    )
    require(
        provider["split"] == "train" and provider["training_eligible"] is True,
        "provider is not TRAIN eligible",
    )
    paths, pins = provider["paths"], provider["sha256"]
    capture_path = Path(paths["capture_manifest"])
    capture = json.loads(read("capture-manifest.json", capture_path, pins["capture_manifest"]))
    require(
        capture["split"] == "train"
        and capture["training_eligible"] is False
        and capture["readiness"] == "preparation_only",
        "source eligibility must remain original preparation-only",
    )
    readiness = json.loads(
        read(
            "readiness.json",
            provider["continuous_readiness"]["path"],
            provider["continuous_readiness"]["sha256"],
        )
    )
    require(
        pins["capture_manifest"] in readiness["teacher_capture_manifest_sha256"]
        and readiness["native_binary_sha256"] == capture["binary_sha256"],
        "readiness does not admit capture/runtime",
    )
    require(
        readiness["common_source_sha256"] == source["common_source_sha256"]
        and all(pins[k] == v for k, v in source["common_source_sha256"].items()),
        "frozen common ancestry differs",
    )
    audit = json.loads(
        read(
            "audit-receipt.json",
            provider["native_label_audit_receipt"]["path"],
            provider["native_label_audit_receipt"]["sha256"],
        )
    )
    require(
        audit["full_semantic_audit"] is True
        and audit["binding"]["capture_manifest"]["sha256"] == pins["capture_manifest"]
        and audit["binding"]["split"] == "train",
        "historical semantic audit absent",
    )
    prompts = records(read("train-prompts.jsonl", paths["prompts"], pins["prompts"]))
    stages = json.loads(
        read(
            "retained-stages.json",
            binding["stages_manifest"]["path"],
            binding["stages_manifest"]["sha256"],
        )
    )
    plan = [x for x in stages["captures"] if x["split"] == "train"][config["shard_ordinal"]]
    require(plan["prompts_sha256"] == pins["prompts"], "TRAIN plan membership differs")
    corpus_path = Path(stages["corpus_manifest"]["path"])
    corpus = json.loads(
        read("corpus-manifest.json", corpus_path, stages["corpus_manifest"]["sha256"])
    )
    corpus_shard = next(
        x
        for x in corpus["files"]["train"]["shards"]
        if x["prompts_sha256"] == plan["source_prompts_sha256"]
    )
    full_prompts = records(
        read(
            "source-train-prompts.jsonl",
            corpus_path.parent / corpus_shard["prompts"],
            corpus_shard["prompts_sha256"],
        )
    )
    train_index = records(
        read(
            "source-train-index.jsonl",
            corpus_path.parent / corpus_shard["index"],
            corpus_shard["index_sha256"],
        )
    )
    require(
        corpus_shard["index_sha256"] == plan["source_index_sha256"]
        and [full_prompts[i] for i in plan["source_positions"]] == prompts,
        "prompt bytes differ from source TRAIN inventory",
    )
    directory = capture_path.parent
    ledger_raw = read(
        "feature-rows.jsonl",
        directory / capture["feature_rows"]["path"],
        capture["feature_rows"]["sha256"],
    )
    ledger = records(ledger_raw)
    anchors = records(
        read("anchors.jsonl", directory / capture["anchors"]["path"], capture["anchors"]["sha256"])
    )
    cell_record = capture["files"]["source_cell"]
    cell = json.loads(
        read("source-cell.json", directory / cell_record["path"], cell_record["sha256"])
    )
    require(
        cell["complete"] is True
        and cell["target_sha256"] == capture["target_sha256"] == pins["target_gguf"]
        and cell["draft_sha256"] == capture["draft_sha256"] == pins["candidate_d_gguf"],
        "producer target/draft identity differs",
    )
    fpath = directory / capture["features"]["path"]
    before = fpath.stat()
    total_reads += before.st_size
    require(total_reads <= config["max_remote_read_bytes"], "feature hash exceeds read cap")
    require(
        sha256(fpath)
        == capture["features"]["sha256"]
        == audit["binding"]["owned_files"][fpath.name],
        "raw feature hash differs",
    )
    require(stable_stat(fpath.stat()) == stable_stat(before), "features changed while hashing")
    hashes[str(fpath)] = capture["features"]["sha256"]
    features = np.load(fpath, mmap_mode="r", allow_pickle=False)
    require(
        features.dtype == np.float32 and features.shape == (len(ledger), 7680),
        "unexpected raw fusion shape/cast",
    )
    selected, groups, contents = [], set(), set()
    for prompt in prompts:
        ix = plan["source_positions"][prompts.index(prompt)]
        original_index = train_index[ix]
        group = original_index.get("group_id", original_index.get("group"))
        require(isinstance(group, str) and group, "TRAIN group identity field absent")
        content = message_hash(prompt)
        rows = [r for r in ledger if r["prompt_id"] == prompt["id"]]
        if len(rows) < config["rows_per_prompt"] or group in groups or content in contents:
            continue
        groups.add(group)
        contents.add(content)
        chosen = [
            rows[int(i)]
            for i in np.linspace(0, len(rows) - 1, config["rows_per_prompt"], dtype=int)
        ]
        selected.append((prompt, group, content, chosen))
        if len(selected) == config["fit_prompts"] + config["validation_prompts"]:
            break
    require(
        len(selected) == config["fit_prompts"] + config["validation_prompts"],
        "insufficient independent TRAIN prompts",
    )
    raw, row_evidence, prompt_evidence = [], [], []
    requests = {r["id"]: r for r in capture["requests"]}
    selected_native = {r["native_feature_row"] for _, _, _, rows in selected for r in rows}
    raw_events = {}
    raw_metadata_path = directory / "heads.target_features.jsonl"
    require(
        raw_metadata_path.name in audit["binding"]["owned_files"], "native event lineage absent"
    )
    # Stream and hash the saved event ledger once; preserve only selected events.
    h = hashlib.sha256()
    with raw_metadata_path.open("rb") as f:
        for line in f:
            total_reads += len(line)
            require(total_reads <= config["max_remote_read_bytes"], "event read cap exceeded")
            h.update(line)
            item = json.loads(line)
            if item["feature_row"] in selected_native:
                raw_events.setdefault(item["feature_row"], {})[item["event"]] = item
    require(
        h.hexdigest() == audit["binding"]["owned_files"][raw_metadata_path.name],
        "native event hash differs",
    )
    hashes[str(raw_metadata_path)] = h.hexdigest()
    with (directory / "heads.target_features.f32").open("rb") as native:
        for ordinal, (prompt, group, content, chosen) in enumerate(selected):
            request = requests[prompt["id"]]
            req = json.loads(
                read(
                    f"request-{ordinal:02d}.json",
                    directory / request["request"]["path"],
                    request["request"]["sha256"],
                )
            )
            original_prompt = json.loads(
                read(
                    f"request-prompt-{ordinal:02d}.json",
                    directory / request["prompt"]["path"],
                    request["prompt"]["sha256"],
                )
            )
            prompt_evidence.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_sha256": content,
                    "group_id": group,
                    "split": "train" if ordinal < config["fit_prompts"] else "validation",
                    "source_split": "train",
                    "prompt": prompt,
                    "request": req,
                    "request_prompt": original_prompt,
                }
            )
            for r in chosen:
                require(
                    r["accepted_prefix"] is True
                    and r["boundary"] == BOUNDARY
                    and r["tap_ids"] == [2, 18, 33]
                    and r["source"] == "native_target_features_on_accepted_prefix"
                    and len(r["prefix_token_ids"]) == r["position"] + 1,
                    "raw boundary/prefix differs",
                )
                anchor = next(
                    a
                    for a in anchors
                    if a["prompt_id"] == r["prompt_id"]
                    and a["prefix_token_ids"][: r["position"] + 1] == r["prefix_token_ids"]
                )
                events = raw_events[r["native_feature_row"]]
                decoded, disposition = events["decoded_row"], events["disposition"]
                require(
                    decoded["boundary"] == "raw_target_layer_input_before_eagle_encoder"
                    and decoded["target_layer_ids"] == [2, 18, 33]
                    and decoded["feature_dim"] == 7680
                    and decoded["prefix_token_ids"] == r["prefix_token_ids"]
                    and decoded["task_id"] == r["task_id"]
                    and disposition["retained_input"] is True
                    and disposition["reason"] == "accepted_prefix",
                    "actual native raw/accepted event differs",
                )
                x = np.array(features[r["feature_row"]], dtype="<f4", copy=True)
                native.seek(r["native_feature_row"] * 7680 * 4)
                original = native.read(7680 * 4)
                total_reads += len(original)
                require(
                    original == x.tobytes() and np.isfinite(x).all(),
                    "selected original native bytes differ",
                )
                raw.append(x)
                row_evidence.append(
                    {
                        "row_id": f"{r['prompt_id']}:feature:{r['feature_row']}",
                        "raw_input_sha256": digest(x.tobytes()),
                        "feature_row": r["feature_row"],
                        "feature_metadata": r,
                        "prompt_token_ids": r["prefix_token_ids"],
                        "anchor": anchor,
                        "native_events": events,
                    }
                )
    raw = np.stack(raw)
    require(
        raw.nbytes <= config["max_selected_bytes"]
        and total_reads <= config["max_remote_read_bytes"],
        "package cap exceeded",
    )
    buffer = io.BytesIO()
    np.save(buffer, raw, allow_pickle=False)
    saved["raw-input.npy"] = buffer.getvalue()
    source_dir = Path(binding["source_checkout"])
    for name in (
        "scripts/prepare_recurrent_native_features.py",
        "scripts/audit_recurrent_binary_capture.py",
        "src/w1a1_eagle/native_step.py",
        "scripts/w1ax_capture_provider.py",
    ):
        read("producer/" + name.replace("/", "__"), source_dir / name)
    saved["acquisition.json"] = json_bytes(
        {
            "schema": "fusion_binary_saved_train_acquisition_v1",
            "source_hashes": hashes,
            "remote_read_bytes": total_reads,
            "selected_prompts": prompt_evidence,
            "rows": row_evidence,
            "producer_contract": CONTRACT,
            "capture_manifest_sha256": pins["capture_manifest"],
            "readiness_sha256": provider["continuous_readiness"]["sha256"],
        }
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw_bytes in saved.items():
            z.writestr(name, raw_bytes)


def validate_row_evidence(x, evidence):
    """Verify exact native operand, token, accepted-disposition and cache joins."""
    meta, anchor = evidence["feature_metadata"], evidence["anchor"]
    decoded = evidence["native_events"]["decoded_row"]
    disposition = evidence["native_events"]["disposition"]
    prefix = meta["prefix_token_ids"]
    require(
        meta["accepted_prefix"] is True
        and meta["boundary"] == BOUNDARY
        and meta["tap_ids"] == [2, 18, 33]
        and meta["source"] == "native_target_features_on_accepted_prefix"
        and len(prefix) == meta["position"] + 1
        and evidence["prompt_token_ids"] == prefix
        and evidence["feature_row"] == meta["feature_row"],
        "raw boundary/coordinate/token join differs",
    )
    require(
        anchor["split"] == "train"
        and anchor["prompt_id"] == meta["prompt_id"]
        and anchor["prefix_token_ids"][: len(prefix)] == prefix,
        "raw accepted-prefix/cache anchor differs",
    )
    require(
        decoded["feature_row"] == disposition["feature_row"] == meta["native_feature_row"]
        and decoded["task_id"] == disposition["task_id"] == meta["task_id"]
        and decoded["decode_ordinal"] == meta["native_decode_ordinal"]
        and decoded["prefix_token_ids"] == prefix
        and decoded["position"] == meta["position"]
        and decoded["boundary"] == "raw_target_layer_input_before_eagle_encoder"
        and decoded["source"] == "target_verifier"
        and decoded["feature_dtype"] == "float32_native_endian"
        and decoded["target_layer_ids"] == [2, 18, 33]
        and decoded["feature_dim"] == 7680
        and disposition["retained_input"] is True
        and disposition["reason"] == "accepted_prefix",
        "actual producer/native retained-input identity differs",
    )
    if decoded["phase"] == "speculative":
        require(
            decoded["round_index"] == disposition["round_index"]
            and decoded["spec_input_row"] <= disposition["accepted_drafts"],
            "speculative raw row is outside accepted cache",
        )
    else:
        require(decoded["phase"] in {"prefill", "target_only"}, "invalid raw native phase")
    require(
        x.dtype == np.float32
        and x.shape == (7680,)
        and np.isfinite(x).all()
        and digest(x.astype("<f4").tobytes()) == evidence["raw_input_sha256"],
        "raw native F32 operand bytes differ",
    )


def read_bf16_fusion(path):
    """Read only the original fc.weight from a hash-checked safetensors file."""
    with Path(path).open("rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        require(0 < header_size < 16 * 1024**2, "invalid safetensors header")
        header = json.loads(f.read(header_size))
        info = header["fc.weight"]
        require(
            info["dtype"] == "BF16" and info["shape"] == [2560, 7680],
            "original fusion tensor differs",
        )
        start, end = info["data_offsets"]
        require(end - start == 2560 * 7680 * 2, "BF16 fusion byte size differs")
        f.seek(8 + header_size + start)
        raw = f.read(end - start)
        require(len(raw) == end - start, "truncated original fusion")
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").reshape(2560, 7680)


def prepare(bundle, bundle_sha, weights, base, output_dir, runtime_manifest):
    """Verify the externally pinned acquisition before producing fitter inputs."""
    require(sha256(bundle) == bundle_sha, "saved acquisition package SHA differs")
    require(
        sha256(weights) == WEIGHT_SHA and sha256(base) == BASE_SHA,
        "original frozen model hash differs",
    )
    require(
        sha256(runtime_manifest)
        == "a199cfdabd81b5ba7414509e31007eab338c125b881a9276a78696cd9208fd5e",
        "captured native runtime manifest differs",
    )
    runtime = json.loads(Path(runtime_manifest).read_text())
    require(
        runtime["native_commit"] == "b4e366d4f0a30cac07f14d51c54c5b1329b3f485",
        "captured native revision differs",
    )
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "output must be a new owned path")
    with zipfile.ZipFile(bundle) as z:
        infos = z.infolist()
        require(
            sum(i.file_size for i in infos) <= 64 * 1024**2
            and len({i.filename for i in infos}) == len(infos),
            "duplicate/oversized acquisition archive",
        )
        require(
            all(
                not Path(i.filename).is_absolute() and ".." not in Path(i.filename).parts
                for i in infos
            ),
            "unsafe package path",
        )
        files = {i.filename: z.read(i) for i in infos}
    native_runtime_raw = Path(runtime_manifest).read_bytes()
    acquisition = json.loads(files["acquisition.json"])
    require(
        acquisition["schema"] == "fusion_binary_saved_train_acquisition_v1"
        and acquisition["producer_contract"] == CONTRACT,
        "producer contract differs",
    )
    require(
        acquisition["remote_read_bytes"] <= 512 * 1024**2, "acquisition exceeded CPU read budget"
    )
    for name, raw in files.items():
        if name not in {"acquisition.json", "raw-input.npy"}:
            require(
                digest(raw) in acquisition["source_hashes"].values(),
                "source file lacks authenticated byte identity",
            )
    raw_input = np.load(io.BytesIO(files["raw-input.npy"]), allow_pickle=False)
    require(
        raw_input.shape == (384, 7680)
        and raw_input.dtype == np.float32
        and np.isfinite(raw_input).all(),
        "bounded actual operand contract differs",
    )
    selected = acquisition["selected_prompts"]
    require(
        len(selected) == 12
        and [x["split"] for x in selected] == ["train"] * 8 + ["validation"] * 4,
        "frozen fit/validation prompt budget differs",
    )
    prompt_lookup = {p["id"]: p for p in records(files["source-train-prompts.jsonl"])}
    index_lookup = {p["id"]: p for p in records(files["source-train-index.jsonl"])}
    require(
        len({p["group_id"] for p in selected}) == 12
        and len({p["prompt_sha256"] for p in selected}) == 12,
        "prompt/group leakage",
    )
    for p in selected:
        require(
            p["source_split"] == "train"
            and p["prompt_id"] in prompt_lookup
            and message_hash(prompt_lookup[p["prompt_id"]]) == p["prompt_sha256"]
            and index_lookup[p["prompt_id"]]["group"] == p["group_id"],
            "TRAIN inventory/content/group differs",
        )
        require(
            p["request_prompt"] == prompt_lookup[p["prompt_id"]]
            and p["request"]["messages"] == p["request_prompt"]["messages"]
            and p["request"]["cache_prompt"] is False,
            "native request ancestry differs",
        )
    owners = {p["prompt_id"]: p for p in selected}
    ledger = {r["feature_row"]: r for r in records(files["feature-rows.jsonl"])}
    anchor_ledger = records(files["anchors.jsonl"])
    rows = []
    for x, evidence in zip(raw_input, acquisition["rows"], strict=True):
        validate_row_evidence(x, evidence)
        meta, anchor = evidence["feature_metadata"], evidence["anchor"]
        require(
            meta == ledger[evidence["feature_row"]]
            and anchor in anchor_ledger
            and meta["accepted_prefix"] is True
            and meta["boundary"] == BOUNDARY
            and meta["tap_ids"] == [2, 18, 33],
            "selected raw row/anchor differs from original ledger",
        )
        require(
            anchor["prompt_id"] == meta["prompt_id"]
            and anchor["split"] == "train"
            and anchor["prefix_token_ids"][: meta["position"] + 1] == meta["prefix_token_ids"],
            "selected prefix/cache context differs",
        )
        require(
            digest(x.astype("<f4").tobytes()) == evidence["raw_input_sha256"],
            "selected raw byte hash differs",
        )
        p = owners[meta["prompt_id"]]
        rows.append(
            {
                "row_id": evidence["row_id"],
                "prompt_id": meta["prompt_id"],
                "prompt_sha256": p["prompt_sha256"],
                "depth": 0,
                "position": meta["position"],
                "split": p["split"],
                "raw_input_sha256": evidence["raw_input_sha256"],
            }
        )
    # Native graph defaults must match the authenticated producer boundary.
    from gguf import GGUFReader

    reader = GGUFReader(str(base))
    norm = reader.fields.get("eagle3.norm_before_fc")
    if norm is not None:
        require(
            not bool(norm.parts[norm.data[0]][0]), "base requires upstream fusion normalization"
        )
    del reader
    reference = read_bf16_fusion(weights)
    output_dir.mkdir(parents=True)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    source_files = {}
    for name, raw in files.items():
        if name in {"acquisition.json", "raw-input.npy"}:
            continue
        p = source_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        source_files[str(p)] = digest(raw)
    native_manifest_path = source_dir / "native-runtime-manifest.json"
    native_manifest_path.write_bytes(native_runtime_raw)
    source_files[str(native_manifest_path)] = digest(native_runtime_raw)
    inventory = {
        "schema": "fusion_binary_train_inventory_v1",
        "source_train_prompts_sha256": digest(files["source-train-prompts.jsonl"]),
        "source_train_index_sha256": digest(files["source-train-index.jsonl"]),
        "prompts": [
            {
                "prompt_id": p["id"],
                "prompt_sha256": message_hash(p),
                "group_id": index_lookup[p["id"]]["group"],
                "source_split": "train",
            }
            for p in prompt_lookup.values()
        ],
    }
    inventory_path = output_dir / "train-inventory.json"
    inventory_path.write_bytes(json_bytes(inventory))
    operands = output_dir / "fusion-operands.npz"
    np.savez(
        operands,
        raw_input=raw_input,
        reference_weight=reference,
        raw_join_ids=np.asarray([r["row_id"] for r in rows]),
    )
    manifest = {
        "schema_version": 1,
        "projection": "fc",
        "raw_input_stage": "pre_activation_quantization",
        "source_data_split": "train",
        "eligibility": "training_allowed",
        "synthetic": False,
        "source": {
            "frozen_weights_sha256": WEIGHT_SHA,
            "base_gguf_sha256": BASE_SHA,
            "capture_manifest_sha256": acquisition["capture_manifest_sha256"],
        },
        "operands": operands.name,
        "operands_sha256": sha256(operands),
        "rows": rows,
    }
    manifest_path = output_dir / "fusion-manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    receipt = {
        "schema": "fusion_binary_train_provenance_v1",
        "manifest_sha256": sha256(manifest_path),
        "operands_sha256": sha256(operands),
        "source_weights": {"path": str(Path(weights).resolve()), "sha256": WEIGHT_SHA},
        "base_gguf": {"path": str(Path(base).resolve()), "sha256": BASE_SHA},
        "producer": {
            "contract": CONTRACT,
            "source_files": source_files,
            "native_revision": "b4e366d4f0a30cac07f14d51c54c5b1329b3f485",
            "capture_manifest_sha256": acquisition["capture_manifest_sha256"],
            "readiness_sha256": acquisition["readiness_sha256"],
        },
        "train_inventory": {"path": str(inventory_path), "sha256": sha256(inventory_path)},
        "selected_prompts": [
            {k: p[k] for k in ("prompt_id", "prompt_sha256", "group_id", "split")} for p in selected
        ],
        "rows": acquisition["rows"],
        "checks": {
            "train_membership": True,
            "prompt_disjoint": True,
            "raw_boundary_verified": True,
            "capture_eligible": True,
            "source_hashes_verified": True,
        },
    }
    receipt_path = output_dir / "provenance-receipt.json"
    receipt_path.write_bytes(json_bytes(receipt))
    (output_dir / "acquisition.json").write_bytes(files["acquisition.json"])
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "operands_sha256": sha256(operands),
        "archive_sha256": bundle_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--remote-stdout", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--source-weights", type=Path)
    parser.add_argument("--base-gguf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    if args.remote_stdout:
        acquire(json.loads(args.config.read_text()), sys.stdout.buffer)
    else:
        require(
            all(
                (
                    args.bundle,
                    args.bundle_sha256,
                    args.source_weights,
                    args.base_gguf,
                    args.output_dir,
                    args.runtime_manifest,
                )
            ),
            "all local package/model/output bindings required",
        )
        print(
            json.dumps(
                prepare(
                    args.bundle,
                    args.bundle_sha256,
                    args.source_weights,
                    args.base_gguf,
                    args.output_dir,
                    args.runtime_manifest,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
