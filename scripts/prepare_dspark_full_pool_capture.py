#!/usr/bin/env python3
"""Freeze full original TRAIN role/group membership and <=8 GiB capture plans.

Default is CPU preparation only. Capture/native/model admission is still pending.
Plans use original ecff F16 target generation plus exact indexed replay, preserving
both receipts and complete five-tap F32 contexts without prompt cropping.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_nine_model_train_data as capture
from prepare_block_production_capture_plans import BINARY_RELATIVE, BINARY_SHA, NATIVE_REVISION, TARGET_SHA

GIB = 1024**3
VARIANT = "native_target_generated_plus_indexed_replay_v1"
SELECTOR_REL = "results/dspark-next-run-plan-20261006/full-train-singleton-calibration-selector.proposed.v2.json"
CORPUS_REL = "results/nine-model-qat-preparation/development-pilot-source-20261004/corpus-manifest.json"
TARGET_PINS_REL = "results/nine-model-qat-preparation/development-pilot-source-20261004/target-source-pins.json"


def write(path, value):
    pin = capture.write_json(path, value)
    return {"path": Path(pin["path"]).name, "sha256": pin["sha256"]}


def interleave_groups(selection, *, calibration):
    """Seeded domain/role interleaving of whole groups; chronology is downstream."""
    components = {}
    for row in selection:
        if (row["split"] != "train") == calibration:
            components.setdefault(row["group_id"], []).append(row)
    buckets = {(role, domain): [] for role in capture.SPLITS for domain in capture.DOMAINS}
    for group in components.values():
        roles = {r["split"] for r in group}
        domains = {r["domain"] for r in group}
        if len(roles) != 1 or len(domains) != 1:
            raise ValueError("cross-role/domain canonical component cannot be silently split")
        buckets[(group[0]["split"], group[0]["domain"])].append(group)
    rng = random.Random(8101)
    for key, groups in buckets.items():
        groups.sort(key=lambda rows: rows[0]["group_id"])
        rng.shuffle(groups)
    order = []
    while any(buckets.values()):
        for role in capture.SPLITS:
            for domain in capture.DOMAINS:
                groups = buckets[(role, domain)]
                if groups:
                    order.append(groups.pop())
    return order


def make_chunks(groups, *, source_bytes, first_goldens=False, shard_bytes=8*GIB):
    # Charge both original generation and exact replay full-context payloads.
    per_chain = 2 * 4096 * 5 * 2560 * 4 + 513 * 151936 * 4
    overhead_per_chain = 1024**2 + 2 * (128 + 4096 * 8)
    golden = 3 * 4 * (4096 * 8 * 2560 + 2 * 151936) + 6 * 1024**2
    chunks, current = [], []
    used = source_bytes + 32*1024**2 + (golden if first_goldens else 0)
    for group in groups:
        amount = len(group) * (per_chain + overhead_per_chain)
        if used + amount > shard_bytes:
            if not current:
                raise MemoryError("whole canonical group exceeds immutable 8 GiB chunk bound")
            chunks.append(current)
            current, used = [], source_bytes + 32*1024**2
        current.extend(group)
        used += amount
    if current:
        chunks.append(current)
    if first_goldens and set(r["domain"] for r in chunks[0]) != set(capture.DOMAINS):
        raise ValueError("initial calibration chunk must cover three golden domains")
    return chunks


def prepare(*, source_root, checkout_root, asset_root, native_artifact_root, output_root,
            selector=None, corpus=None, corpus_variant=VARIANT):
    if corpus_variant != VARIANT:
        raise ValueError("fresh corpus must be explicitly frozen as indexed-replay variant")
    source_root = Path(source_root).resolve()
    roots = [Path(p) for p in (checkout_root, asset_root, native_artifact_root)]
    if any(not p.is_absolute() for p in roots):
        raise ValueError("checkout, asset and native artifact roots must be absolute")
    checkout_root, asset_root, native_artifact_root = roots
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError("refuse to overwrite immutable preparation")
    selector = Path(selector) if selector else source_root / SELECTOR_REL
    corpus = Path(corpus) if corpus else source_root / CORPUS_REL
    selector = capture.pinned({"path": str(selector), "sha256": capture.APPROVED_DSPARK_SELECTOR_SHA256}, Path.cwd())
    selected = json.loads(selector.read_text())
    corpus = capture.pinned({"path": str(corpus), "sha256": selected["source_manifest"]["sha256"]}, Path.cwd())
    original = json.loads(corpus.read_text())
    output.mkdir(parents=True)
    shutil.copyfile(selector, output / "role-selector.json")
    shutil.copyfile(corpus, output / "original-corpus.json")
    selector_pin = {"path": "role-selector.json", "sha256": capture.file_sha256(selector)}
    original_pin = {"path": "original-corpus.json", "sha256": capture.file_sha256(corpus)}
    contracts, corpora = {}, {}
    source_files = []
    for location in ("local", "remote"):
        shards = []
        for ordinal, shard in enumerate(original["files"]["train"]["shards"]):
            mapped = copy.deepcopy(shard)
            for kind in ("prompts", "index"):
                relative = Path("data/continuous-w1ax/freeze-002") / Path(shard[kind]).name
                local = capture.pinned({"path": str(source_root / relative), "sha256": shard[kind + "_sha256"]}, Path.cwd())
                mapped[kind] = str(local if location == "local" else asset_root / relative)
                if location == "local":
                    source_files.append({"shard": ordinal, "kind": kind, "path": relative.as_posix(),
                                         "sha256": shard[kind + "_sha256"], "bytes": local.stat().st_size})
            shards.append(mapped)
        corpora[location] = write(output / f"{location}-corpus.json", {
            "schema": "dspark_full_pool_transport_corpus_v1", "original_corpus": original_pin,
            "files": {"train": {"shards": shards}}})
        contracts[location] = write(output / f"{location}-contract.json", {
            "schema": "dspark_full_pool_source_contract_v1", "selector": selector_pin,
            "original_corpus": original_pin, "source_corpus": corpora[location]})
    _, verified = capture.validate_full_pool_contract(contracts["local"], output, corpora["local"])
    target_pins = json.loads((source_root / TARGET_PINS_REL).read_text())
    if target_pins["target_sha256"] != TARGET_SHA:
        raise ValueError("original F16 target differs")
    client = ROOT / "scripts/capture_block_qat_teacher.py"
    client_sha = capture.file_sha256(client)
    runtime = {"schema": "nine_model_train_capture_runtime_v1", "binary_sha256": BINARY_SHA,
               "target_sha256": TARGET_SHA, "native_source_revision": NATIVE_REVISION,
               "teacher_client_sha256": client_sha,
               "tokenizer_metadata_sha256": target_pins["tokenizer_metadata_sha256"],
               "chat_template_sha256": target_pins["chat_template_sha256"]}
    runtime_pin = write(output / "runtime.json", runtime)
    variant_pin = write(output / "generation-variant.json", {
        "schema": "dspark_fresh_native_generation_variant_v1", "corpus_variant": corpus_variant,
        "kind": "fresh_greedy", "max_new_tokens": 512, "natural_eog": True,
        "generation_temperature": 0.0, "thinking": False,
        "context_capacity": 4096, "original_prompt_text_preserved": True,
        "target_sha256": TARGET_SHA, "teacher_client_sha256": client_sha,
        "native_binary_sha256": BINARY_SHA, "native_revision": NATIVE_REVISION,
        "tokenizer_metadata_sha256": runtime["tokenizer_metadata_sha256"],
        "chat_template_sha256": runtime["chat_template_sha256"],
        "existing_committed_continuation_ancestry": "NOT_PROVEN_FRESH_VARIANT_SELECTED",
        "actual_tokens_EOG_teacher_anchors": "PENDING_NATIVE_CAPTURE"})
    source_bytes = verified["source_bytes"] + 16*1024**2
    groups_by_phase = [("calibration", interleave_groups(selected["selection"], calibration=True)),
                       ("train", interleave_groups(selected["selection"], calibration=False))]
    chunks = []
    for phase, groups in groups_by_phase:
        for phase_index, rows in enumerate(make_chunks(groups, source_bytes=source_bytes,
                                                       first_goldens=phase == "calibration")):
            number = len(chunks)
            goldens = phase == "calibration" and phase_index == 0
            caps = {"max_requests": 2*len(rows) + (6 if goldens else 0),
                    "max_tokens_per_chain": 4096, "max_prompt_tokens": 3584,
                    "max_request_bytes": GIB, "max_shard_bytes": 8*GIB,
                    "max_total_bytes": 8*GIB, "max_source_bytes": 128*1024**2,
                    "max_source_row_bytes": 1024**2, "max_host_rss_bytes": 12*GIB,
                    "min_host_available_bytes": 4*GIB, "min_free_disk_bytes": 10*GIB,
                    "request_timeout_seconds": 600, "total_timeout_seconds": (2*len(rows)+6)*600,
                    "max_eagle_golden_tokens": 4096}
            choice = [{"shard": r["shard"], "row": r["row"], "prompt_id": r["prompt_id"],
                       "prompt_sha256": r["content_sha256"], "domain": r["domain"], "split": r["split"]}
                      for r in rows]
            plans = {}
            for location in ("local", "remote"):
                native = {"binary": {"path": str(native_artifact_root / BINARY_RELATIVE), "sha256": BINARY_SHA},
                          "target": {"path": str(native_artifact_root / "models/gguf/Qwen3-4B-f16.gguf"), "sha256": TARGET_SHA},
                          "source_revision": NATIVE_REVISION,
                          "client_source": {"path": str(client if location == "local" else checkout_root / "scripts/capture_block_qat_teacher.py"), "sha256": client_sha},
                          "tokenizer_metadata_sha256": runtime["tokenizer_metadata_sha256"],
                          "chat_template_sha256": runtime["chat_template_sha256"],
                          "gpu_layers": 999, "expected_compute_capability": [12, 0]}
                plans[location] = write(output / f"chunk-{number:04d}-{location}-plan.json", {
                    "schema": "nine_model_train_capture_plan_v1", "corpus": corpora[location],
                    "full_pool_contract": contracts[location], "runtime": runtime_pin, "native": native,
                    "vocab_size": 151936, "target_width": 2560, "mask_token_id": 151669,
                    "input_mode": "native_chat", "objective": "exact_soft", "teacher_logits_layout": "indexed",
                    "label_policy": capture.PARTIAL_LABEL_POLICY, "capture_goldens": goldens,
                    "generation": {"max_new_tokens": 512, "algorithm": "greedy"}, "caps": caps,
                    "selection": choice})
            chunks.append({"chunk_id": number, "phase": phase, "phase_index": phase_index,
                           "plans": plans, "prompt_ids": [r["prompt_id"] for r in rows],
                           "group_ids": sorted({r["group_id"] for r in rows}), "capture_goldens": goldens,
                           "chain_ids": [f"source-{r['shard']:02d}-{r['row']:04d}" for r in rows],
                           "counts": {s: dict(Counter(r["domain"] for r in rows if r["split"] == s)) for s in capture.SPLITS},
                           "max_capture_bytes": 8*GIB, "actual_native_anchors": "PENDING_FIRST_CAPTURE"})
    _, records, cost = capture.prepare_plan(output / chunks[0]["plans"]["local"]["path"], chunks[0]["plans"]["local"]["sha256"])
    write(output / "initial-calibration-cost.json", cost)
    # Genuine default producer CPU preparation for all 144 calibration records.
    calibration_reports = []
    for chunk in chunks:
        if chunk["phase"] != "calibration":
            continue
        plan_pin = chunk["plans"]["local"]
        report = capture.run_capture(output / plan_pin["path"], plan_pin["sha256"],
                                     output / f"calibration-{chunk['phase_index']:04d}-cpu-plan")
        calibration_reports.append(report)
    write(output / "calibration-default-cpu-summary.json", {
        "schema": "dspark_calibration_cpu_preparation_v1", "status": "SOURCE_VALIDATED_NATIVE_CAPTURE_PENDING",
        "prompts": sum(r["selected_prompt_count"] for r in calibration_reports),
        "native_requests": sum(r["max_native_requests"] for r in calibration_reports),
        "golden_requests": 6, "roles": {"calibration_fit": 96, "calibration_validation": 48},
        "chunks": len(calibration_reports), "capture_executed": False,
        "actual_tensor_admission": "PENDING", "teacher_client_sha256": client_sha})
    selector_by_id = {r["prompt_id"]: r for r in selected["selection"]}
    manifest = {"schema": "dspark_full_pool_capture_schedule_v1", "status": "SOURCE_PREPARED_NATIVE_CAPTURE_PENDING",
                "production_data_status": "PENDING_NATIVE_CAPTURE", "model_admission": "PENDING",
                "corpus_variant": corpus_variant, "generation_variant": {"kind": "fresh_greedy", "receipt_sha256": variant_pin["sha256"], "corpus_variant": corpus_variant}, "generation_variant_receipt": variant_pin, "existing_committed_token_reuse": "NOT_PROVEN_THIS_VARIANT_IS_FRESH",
                "seed": 8101, "role_selector": selector_pin, "source_contracts": contracts,
                "counts": verified["counts"], "canonical_components": verified["canonical_components"],
                "multi_record_components": verified["multi_record_components"],
                "multi_record_prompts": verified["multi_record_prompts"], "source_files": source_files,
                "native_artifact_root": str(native_artifact_root), "checkout_root": str(checkout_root),
                "chunk_bytes": 8*GIB, "max_resident_chunks": 3,
                "label_policy": capture.PARTIAL_LABEL_POLICY, "computed_slots": 7,
                "anchor_stride": 7, "teacher_vocab": 151936, "teacher_dtype": "float32",
                "context_capacity": 4096, "no_prompt_crop": True,
                "over_capacity_policy": "FAIL_REVIEW_CAPACITY_NEVER_TRUNCATE",
                "selection": selected["selection"], "chunks": chunks,
                "chains": {f"source-{r['shard']:02d}-{r['row']:04d}": {
                    "chain_id": f"source-{r['shard']:02d}-{r['row']:04d}",
                    "prompt_id": r["prompt_id"], "prompt_sha256": r["content_sha256"],
                    "domain": r["domain"], "split": r["split"], "group_id": r["group_id"],
                    "source_shard": r["shard"], "source_row": r["row"], "input_tokens": r["input_tokens"],
                    # Full per-chain raw capture plus even share of publication/source/golden headroom.
                    "capture_bytes_bound": 2*4096*5*2560*4 + 513*151936*4 + 1024**2 + 2*(128+4096*8)
                        + (source_bytes + 32*1024**2 + (3*4*(4096*8*2560+2*151936)+6*1024**2 if c["capture_goldens"] else 0)
                           + len(c["prompt_ids"]) - 1)//len(c["prompt_ids"])
                } for c in chunks for pid in c["prompt_ids"] for r in [selector_by_id[pid]]},
                "source_modules": {str(p.relative_to(ROOT)): capture.file_sha256(p) for p in (
                    ROOT / "src/w1a1_eagle/block_data.py", ROOT / "scripts/capture_nine_model_train_data.py",
                    client, ROOT / "scripts/replay_dspark_shard.py", Path(__file__))},
                "boundaries": ["CPU source preparation only; actual capture/import/model/native admission pending",
                               "Actual emitted anchors, EOS, tensor hashes and consumed counts sealed on capture",
                               "Full 144 calibration roles precede all 9856 optimizer TRAIN prompts",
                               "Controller owns physical cycling/eviction/archive and logical training cursor",
                               "No original prompt deleted or cropped; multi-record TRAIN components retained"]}
    return capture.write_json(output / "full-pool-schedule.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--native-artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-variant", required=True, choices=[VARIANT])
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
