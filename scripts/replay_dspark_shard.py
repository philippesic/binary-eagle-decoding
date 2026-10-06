#!/usr/bin/env python3
"""Reconstruct exact chain tensor bytes; preserve historical and fresh receipts."""
from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from pathlib import Path

import capture_nine_model_train_data as capture
from capture_block_qat_teacher import NativeTeacher
from w1a1_eagle.block_data import BlockDataset, TAPS


def physical_bytes(root):
    seen, total = set(), 0
    for p in Path(root).rglob("*"):
        if p.is_file():
            stat = p.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def replay(request_path, output_path, *, teacher_factory=None, device_query=None,
           rss_query=None, available_query=None):
    request_path, output_path = Path(request_path).resolve(), Path(output_path).resolve()
    request = json.loads(request_path.read_text())
    global_path = capture.pinned(request["plan"], request_path.parent)
    plan = json.loads(global_path.read_text())
    if (request.get("schema") != "block_shard_restore_request_v1"
        or capture.identity(plan) != request["plan_sha256"]
        or request["generation_variant"] != plan["generation_variant"]
        or request["capture_bytes_bound"] != 8*1024**3):
        raise ValueError("replay logical plan/generation/storage identity differs")
    metadata_path = capture.pinned(request["original_metadata"], request_path.parent)
    metadata = json.loads(metadata_path.read_text())
    directory = Path(request["directory"]).resolve()
    shard = request["shard"]
    if (not Path(request["directory"]).is_absolute()
        or metadata.get("schema") != "block_shard_original_metadata_v1"
        or metadata["plan_sha256"] != request["plan_sha256"] or metadata["shard"] != shard
        or metadata["identity"] != request["original_identity"]
        or Path(metadata["original_directory"]).resolve() != directory
        or directory.exists()):
        raise ValueError("replay requires original metadata and same fresh absolute physical directory")
    producer_pin = {"path": str(Path(__file__).resolve()), "sha256": capture.file_sha256(Path(__file__))}
    if not request.get("replay_probe", False):
        admission_path = capture.pinned(request["replay_admission"], request_path.parent)
        admitted = json.loads(admission_path.read_text())
        if (admitted.get("schema") != "block_shard_replay_admission_v1"
            or admitted.get("status") != "PASS" or admitted["plan_sha256"] != request["plan_sha256"]
            or admitted["source"] != plan["source"] or admitted["producer"]["sha256"] != producer_pin["sha256"]
            or set(admitted["domains"]) != set(capture.DOMAINS)):
            raise ValueError("actual same-source three-domain replay admission required")
    paths = {}
    for relative, pin in metadata["files"].items():
        dest = (directory / relative).resolve()
        if not dest.is_relative_to(directory):
            raise ValueError("metadata path escapes physical shard")
        paths[dest] = capture.pinned(pin, metadata_path.parent)
    old_manifest_path = directory / metadata["manifest_relative_path"]
    if old_manifest_path not in paths:
        raise ValueError("original manifest bytes absent from metastore")
    old_manifest = json.loads(paths[old_manifest_path].read_text())
    if set(c["chain_id"] for c in old_manifest["chains"]) != set(plan["shards"][shard]):
        raise ValueError("original replay chain membership differs")
    capture_plan_path = capture.pinned(plan["capture_plans"][shard]["remote"], global_path.parent)
    capture_plan = json.loads(capture_plan_path.read_text())
    native, caps = capture_plan["native"], capture_plan["caps"]
    binary = capture.pinned(native["binary"], capture_plan_path.parent)
    target = capture.pinned(native["target"], capture_plan_path.parent)
    actual_client = Path(inspect.getfile(NativeTeacher)).resolve()
    if teacher_factory is None and capture.file_sha256(actual_client) != native["client_source"]["sha256"]:
        raise ValueError("tensor replay teacher client differs from frozen generation variant")
    device = (device_query or capture.cuda_device)()
    if device["compute_capability"] != native["expected_compute_capability"]:
        raise ValueError("tensor replay actual CUDA device differs")
    available_query = available_query or capture.host_available_bytes
    rss_query = rss_query or capture.rss_bytes
    directory.mkdir(parents=True)
    (directory / "replay-proof").mkdir()
    start = time.monotonic()
    teacher, fresh_pins, joins = None, {}, {}

    def budget():
        if time.monotonic() - start > caps["total_timeout_seconds"]:
            raise TimeoutError("tensor replay total wall cap reached")
        measured_rss = rss_query()
        if teacher is not None and hasattr(teacher, "process") and teacher.process.poll() is None:
            measured_rss += int(Path(f"/proc/{teacher.process.pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if available_query() < caps["min_host_available_bytes"] or measured_rss > caps["max_host_rss_bytes"]:
            raise MemoryError("tensor replay host resource gate reached")
        if shutil.disk_usage(directory).free < caps["min_free_disk_bytes"]:
            raise MemoryError("tensor replay free disk floor reached")
        if physical_bytes(directory) > request["capture_bytes_bound"]:
            raise MemoryError("tensor replay physical publication exceeds 8 GiB")

    def link_exact(source, original):
        dest = Path(original["path"]).resolve()
        if not dest.is_relative_to(directory) or capture.file_sha256(source) != original["sha256"]:
            raise ValueError("actual newly computed tensor differs from original bytes/path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if capture.file_sha256(dest) != original["sha256"]:
                raise ValueError("replay historical destination collision")
        else:
            os.link(source, dest)
        budget()

    try:
        budget()
        factory = teacher_factory or NativeTeacher
        teacher = factory(binary, target, directory / "replay-proof/raw", target_sha256=native["target"]["sha256"],
                          max_tokens=caps["max_tokens_per_chain"], gpu_layers=native["gpu_layers"],
                          producer_source_revision=native["source_revision"], timeout_seconds=caps["request_timeout_seconds"])
        for chain in old_manifest["chains"]:
            budget()
            cid = chain["chain_id"]
            old_receipt = json.loads(paths[Path(chain["native_receipt"]["path"])].read_text())
            expected = request["original_identity"][cid]
            if (expected["anchors"] != chain["anchors"] or expected["prompt_length"] != chain["prompt_length"]
                or expected["logits_indices"] != chain.get("logits_indices")
                or any(expected["artifacts"][k] != chain[k]["sha256"] for k in expected["artifacts"])):
                raise ValueError("retained original manifest differs from sealed tensor identity")
            history = old_receipt["decode_history"]
            fresh = teacher.capture_prefix(old_receipt["tokens"], TAPS, logits_mode="indexed",
                    logits_indices=chain["logits_indices"], chain_ancestry=old_receipt["chain_ancestry"],
                    decode_history=history)
            capture.validate_replay(fresh, old_receipt["tokens"], TAPS, native, device)
            if fresh.get("decode_history") != history or fresh["logits_indices"] != chain["logits_indices"]:
                raise ValueError("actual replay exact decode partitions/teacher map differ")
            fresh_pins[cid] = capture.write_json(directory / "replay-proof" / f"{cid}-actual-receipt.json", fresh)
            link_exact(Path(fresh["files"]["features"]["path"]), chain["features"])
            link_exact(Path(fresh["files"]["logits"]["path"]), chain["logits"])
            if "generation_source" in old_receipt:
                source = json.loads(paths[Path(old_receipt["generation_source"]["path"])].read_text())
                if source["tokens"] != fresh["tokens"] or source["decode_history"] != history:
                    raise ValueError("historical generation and actual replay prefix differ")
                link_exact(Path(fresh["files"]["features"]["path"]), source["files"]["features"])
            token_source = paths[Path(chain["tokens"]["path"])]
            if capture.file_sha256(token_source) != expected["artifacts"]["tokens"]:
                raise ValueError("retained original token array differs")
            import numpy as np
            if np.load(token_source, allow_pickle=False).tolist() != fresh["tokens"]:
                raise ValueError("retained token array differs from actual replay input")
            joins[cid] = {"features_sha256": fresh["files"]["features"]["sha256"],
                          "logits_sha256": fresh["files"]["logits"]["sha256"],
                          "tokens_sha256": expected["artifacts"]["tokens"]}
        # Only after every actual tensor passed restore byte-identical historical
        # JSON/token metadata. These receipts describe the original capture; the
        # independently retained fresh receipts describe this reconstruction.
        for dest, source in paths.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if capture.file_sha256(dest) != capture.file_sha256(source):
                    raise ValueError("historical metadata destination collision")
            else:
                shutil.copyfile(source, dest)
        ds = BlockDataset(old_manifest_path, expected_sha256=capture.file_sha256(old_manifest_path), audit_only=True,
                          budget_check=budget)
        admission_path = directory / "replay-proof/completed-admission.json"
        try:
            ds.write_admission(admission_path)
        finally:
            ds.close()
        join_pin = capture.write_json(directory / "replay-proof/reconstruction-join.json", {
            "schema": "block_shard_reconstruction_join_v1", "status": "PASS",
            "plan_sha256": request["plan_sha256"], "shard": shard,
            "original_metadata": request["original_metadata"], "original_identity": request["original_identity"],
            "fresh_receipts": fresh_pins, "tensor_identity_equal": True,
            "historical_receipt_bytes_restored": True, "measured_tensor_joins": joins,
            "producer": producer_pin, "physical_bytes": physical_bytes(directory)})
        budget()
        return capture.write_json(output_path, {
            "schema": "block_shard_publication_v1", "status": "PASS",
            "request": {"path": str(request_path), "sha256": capture.file_sha256(request_path)},
            "plan_sha256": request["plan_sha256"], "shard": shard, "directory": str(directory),
            "manifest": {"path": str(old_manifest_path), "sha256": capture.file_sha256(old_manifest_path)},
            "admission": {"path": str(admission_path), "sha256": capture.file_sha256(admission_path)},
            "reconstruction_join": join_pin})
    finally:
        if teacher is not None:
            teacher.close()
            if not teacher.closed or (hasattr(teacher, "process") and teacher.process.poll() is None):
                raise RuntimeError("owned tensor replay producer remains live after close")
