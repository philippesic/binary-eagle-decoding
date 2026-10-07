#!/usr/bin/env python3
"""Plan and materialize bounded target-only original TRAIN capture.

A reviewed SHA-pinned plan selects explicit original corpus shard/row positions.
The default command writes inventory/cost evidence only. --execute is for the
sole coordinated CUDA operator, within remote_job supervision; it never SSHs,
loads a drafter, updates an optimizer, or chooses corpus coverage implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from w1a1_eagle.block_data import (  # noqa: E402
    BLOCK_LOGITS_SELECTION,
    PARTIAL_LABEL_POLICY,
    DOMAINS,
    GENERATED_PREFIX_CONTRACT,
    identity,
    REPLAY_PREFIX_CONTRACT,
    SPLITS,
    TAPS,
    block_teacher_indices,
    file_sha256,
    generated_block_anchors,
    import_capture_plan,
    joined_generation_source,
    validate_logits_indices,
    validate_native_generation,
)

EAGLE_TAPS = (2, 18, 33)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def pinned(record, base, *, max_bytes=None):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("artifact requires exact path/SHA256")
    path = (Path(base) / record["path"]).resolve()
    if max_bytes is not None and path.is_file() and path.stat().st_size > max_bytes:
        raise MemoryError("original source inventory exceeds source cap")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError("artifact missing or differs from SHA256 pin")
    return path


def content_hash(value):
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def selected_jsonl_rows(path, positions, max_row_bytes):
    """Stream only selected original ordinals; never expand a complete JSON shard."""
    selected = {}
    ordinal = 0
    with Path(path).open("rb") as stream:
        while len(selected) < len(positions):
            raw = stream.readline(max_row_bytes + 1)
            if not raw:
                break
            if len(raw) > max_row_bytes:
                raise MemoryError("original TRAIN source row exceeds explicit row storage cap")
            if not raw.strip():
                continue
            if ordinal in positions:
                selected[ordinal] = json.loads(raw)
            ordinal += 1
    if set(selected) != positions:
        raise ValueError("original TRAIN row ordinal invalid or prompt/index selection incomplete")
    return selected



APPROVED_DSPARK_SELECTOR_SHA256 = "6de06976383246d4b512558fa45cf67780546ae11f121ed31f376e81ee1d7ea5"


def canonical_groups(records):
    """Original transitive group OR topic components; never content-deduplicate."""
    parents = list(range(len(records)))
    lookup = {}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(records):
        for key in (row["source_group"], row["source_topic"]):
            if key is None:
                continue
            if key in lookup:
                parents[find(i)] = find(lookup[key])
            else:
                lookup[key] = i
    components = {}
    for i, row in enumerate(records):
        components.setdefault(find(i), []).append(row)
    for rows in components.values():
        group = "original_group_component:" + content_hash(sorted(r["prompt_id"] for r in rows))
        for row in rows:
            row["group_id"] = group
    return components


def validate_full_pool_contract(pin, base, corpus_pin):
    """Authenticate ALL original TRAIN bodies/indices and approved role/group joins.

    Reading TRAIN only is intentional: no development or sealed bodies are opened.
    This CPU validation is source preparation, never tensor/native admission.
    """
    path = pinned(pin, base)
    contract = json.loads(path.read_text())
    if (set(contract) != {"schema", "selector", "original_corpus", "source_corpus"}
        or contract["schema"] != "dspark_full_pool_source_contract_v1"
        or contract["selector"]["sha256"] != APPROVED_DSPARK_SELECTOR_SHA256
        or contract["source_corpus"] != corpus_pin):
        raise ValueError("full-pool source/selector contract differs")
    selector = json.loads(pinned(contract["selector"], path.parent).read_text())
    original = json.loads(pinned(contract["original_corpus"], path.parent).read_text())
    source_path = pinned(corpus_pin, base)
    source = json.loads(source_path.read_text())
    if (selector["seed"] != 8101
        or selector["source_manifest"]["sha256"] != contract["original_corpus"]["sha256"]
        or len(original["files"]["train"]["shards"]) != 10):
        raise ValueError("approved ten-shard corpus ancestry differs")
    shards = source["files"]["train"]["shards"]
    if len(shards) != 10 or len(selector["source_indexes"]) != 10:
        raise ValueError("all ten original TRAIN shards required")
    records = []
    source_bytes = 0
    source_files = []
    for shard_id, shard in enumerate(shards):
        expected = original["files"]["train"]["shards"][shard_id]
        if {k: v for k, v in shard.items() if k not in ("prompts", "index")} != {
            k: v for k, v in expected.items() if k not in ("prompts", "index")
        } or selector["source_indexes"][shard_id]["sha256"] != shard["index_sha256"]:
            raise ValueError("source shard/index pin differs from approved original")
        files = {k: pinned({"path": shard[k], "sha256": shard[k + "_sha256"]}, source_path.parent)
                 for k in ("prompts", "index")}
        source_bytes += sum(p.stat().st_size for p in files.values())
        source_files.extend({"path": str(p), "sha256": shard[k + "_sha256"]} for k, p in files.items())
        positions = set(range(expected["prompts_count"]))
        rows = {k: selected_jsonl_rows(p, positions, 1024**2) for k, p in files.items()}
        # Also reject extra nonblank records: selector must cover the exact pool.
        for source_file in files.values():
            with source_file.open("rb") as stream:
                if sum(bool(line.strip()) for line in stream) != len(positions):
                    raise ValueError("source shard record count differs")
        for ordinal in sorted(positions):
            prompt, index = rows["prompts"][ordinal], rows["index"][ordinal]
            if (prompt.get("id") != index.get("id") or prompt.get("domain") != index.get("domain")
                or content_hash(prompt.get("messages")) != index.get("content_sha256")
                or any(r.get(k, "TRAIN") not in ("train", "TRAIN")
                       for r in (prompt, index) for k in ("split", "source_split", "role"))):
                raise ValueError("full-pool original TRAIN body/index content/domain join differs")
            if not isinstance(index.get("group_id", index.get("group")), str):
                raise ValueError("original source group required")
            records.append({"shard": shard_id, "row": ordinal, "prompt_id": index["id"],
                            "content_sha256": index["content_sha256"], "domain": index["domain"],
                            "source_group": index.get("group_id", index.get("group")),
                            "source_topic": index.get("topic"), "category": index.get("category"),
                            "input_tokens": index["input_tokens"]})
    components = canonical_groups(records)
    choices = selector["selection"]
    if len(records) != 10000 or len(choices) != 10000 or len({r["prompt_id"] for r in records}) != 10000:
        raise ValueError("full-pool must retain all 10000 original prompts exactly once")
    counts = {s: {d: 0 for d in DOMAINS} for s in SPLITS}
    role_maps = {k: {} for k in ("prompt_id", "group_id", "content_sha256")}
    by_location = {}
    for row, selected in zip(records, choices):
        if any(selected.get(k) != v for k, v in row.items()) or selected["split"] not in SPLITS:
            raise ValueError("selector exact source/group/topic/uncropped-length join differs")
        role = selected["split"]
        for key, lookup in role_maps.items():
            value = row[key]
            if value in lookup and lookup[value] != role:
                raise ValueError("canonical group/topic/content cross-role leakage")
            lookup[value] = role
        counts[role][row["domain"]] += 1
        by_location[(row["shard"], row["row"])] = selected
    if counts != {"train": {"prose": 3286, "code": 3285, "reasoning": 3285},
                  "calibration_fit": dict.fromkeys(DOMAINS, 32),
                  "calibration_validation": dict.fromkeys(DOMAINS, 16)}:
        raise ValueError("approved full-pool role/domain counts differ")
    repeated = [rows for rows in components.values() if len(rows) > 1]
    if any(by_location[(r["shard"], r["row"])]["split"] != "train" for rows in repeated for r in rows):
        raise ValueError("all multi-record source components must remain in optimizer TRAIN")
    return by_location, {"counts": counts, "source_bytes": source_bytes,
                         "source_files": source_files, "canonical_components": len(components),
                         "multi_record_components": len(repeated),
                         "multi_record_prompts": sum(map(len, repeated))}


def prepare_plan(plan_path, expected_sha256, *, execution_profile="production_CUDA"):
    """Authenticate explicit TRAIN source selection and report conservative costs."""
    path = pinned({"path": str(Path(plan_path).resolve()), "sha256": expected_sha256}, Path.cwd())
    plan = json.loads(path.read_text())
    fields = {
        "schema",
        "corpus",
        "runtime",
        "native",
        "vocab_size",
        "target_width",
        "mask_token_id",
        "input_mode",
        "objective",
        "generation",
        "caps",
        "selection",
    }
    cpu = execution_profile == "development_CPU"
    if execution_profile not in ("production_CUDA", "development_CPU"):
        raise ValueError("unsupported execution profile")
    if cpu:
        fields |= {"execution_profile", "cpu_build"}
    if (
        not fields <= set(plan) or set(plan) - fields - {"teacher_logits_layout", "full_pool_contract", "label_policy", "capture_goldens"}
        or plan["schema"] != "nine_model_train_capture_plan_v1"
        or (cpu and plan["execution_profile"] != "development_CPU")
    ):
        raise ValueError("unsupported explicit TRAIN capture plan")
    partial = plan.get("label_policy") == PARTIAL_LABEL_POLICY
    if "label_policy" in plan and not partial:
        raise ValueError("unsupported partial-label policy")
    if partial and plan.get("teacher_logits_layout", "dense") != "indexed":
        raise ValueError("partial capture requires explicit indexed replay")
    capture_goldens = plan.get("capture_goldens", True)
    if type(capture_goldens) is not bool:
        raise ValueError("capture_goldens must be boolean")
    golden_requests = 6 if capture_goldens else 0
    if plan["input_mode"] not in ("raw_text", "native_chat"):
        raise ValueError("explicit raw_text or native_chat input mode required")
    if plan["objective"] not in ("hard_ce", "exact_soft"):
        raise ValueError("only hard_ce or bounded full-vocabulary exact_soft supported")
    layout = plan.get("teacher_logits_layout", "dense")
    if layout not in ("dense", "indexed") or (
        layout == "indexed" and plan["objective"] != "exact_soft"
    ):
        raise ValueError("indexed teacher layout requires exact_soft objective")
    vocab = positive(plan["vocab_size"], "vocab_size")
    width = positive(plan["target_width"], "target_width")
    if type(plan["mask_token_id"]) is not int or not 0 <= plan["mask_token_id"] < vocab:
        raise ValueError("mask token outside full vocabulary")
    generation = plan["generation"]
    if set(generation) != {"max_new_tokens", "algorithm"} or generation["algorithm"] != "greedy":
        raise ValueError("native target-only greedy generation required")
    positive(generation["max_new_tokens"], "max_new_tokens")
    if generation["max_new_tokens"] < 8 and not partial:
        raise ValueError("seven-slot author profile requires at least eight generated tokens")
    caps = plan["caps"]
    cap_fields = {
        "max_requests",
        "max_tokens_per_chain",
        "max_prompt_tokens",
        "max_request_bytes",
        "max_shard_bytes",
        "max_total_bytes",
        "max_source_bytes",
        "max_source_row_bytes",
        "max_host_rss_bytes",
        "min_host_available_bytes",
        "min_free_disk_bytes",
        "request_timeout_seconds",
        "total_timeout_seconds",
        "max_eagle_golden_tokens",
    }
    if cpu:
        cap_fields.add("min_host_available_live_bytes")
    if set(caps) != cap_fields:
        raise ValueError("explicit request/source/storage/host/time bounds required")
    for key, value in caps.items():
        positive(value, key)
    if cpu and (
        caps["max_tokens_per_chain"] > 544
        or caps["max_prompt_tokens"] > 512
        or generation["max_new_tokens"] > 32
        or caps["max_requests"] > 15
        or caps["max_total_bytes"] > 8 * 1024**3
        or caps["max_host_rss_bytes"] > 12 * 1024**3
        or caps["min_host_available_bytes"] < 12 * 1024**3
        or caps["min_host_available_live_bytes"] < 4 * 1024**3
        or caps["total_timeout_seconds"] > 1800
    ):
        raise ValueError("development_CPU exceeds approved pilot caps or weakens memory floors")
    if not 1 <= caps["max_eagle_golden_tokens"] <= caps["max_tokens_per_chain"] <= 32768:
        raise ValueError("native chain/golden token bounds invalid")
    if caps["max_prompt_tokens"] + generation["max_new_tokens"] > caps["max_tokens_per_chain"]:
        raise ValueError("prompt token cap plus generation cap exceeds native chain bound")
    if caps["request_timeout_seconds"] > caps["total_timeout_seconds"]:
        raise ValueError("request timeout exceeds total wall cap")
    native = plan["native"]
    if set(native) != {
        "binary",
        "target",
        "source_revision",
        "client_source",
        "tokenizer_metadata_sha256",
        "chat_template_sha256",
        "gpu_layers",
        "expected_compute_capability",
    }:
        raise ValueError("exact native/model/tokenizer/template/device pins required")
    if (
        not isinstance(native["source_revision"], str)
        or len(native["source_revision"]) != 40
        or any(c not in "0123456789abcdef" for c in native["source_revision"])
    ):
        raise ValueError("native source revision required")
    for key in ("tokenizer_metadata_sha256", "chat_template_sha256"):
        if (
            not isinstance(native[key], str)
            or len(native[key]) != 64
            or any(c not in "0123456789abcdef" for c in native[key])
        ):
            raise ValueError("native tokenizer/template SHA256 pins required")
    if cpu:
        if (
            native["gpu_layers"] != 0
            or type(native["gpu_layers"]) is not int
            or native["expected_compute_capability"] is not None
        ):
            raise ValueError("development_CPU requires zero GPU layers and no CUDA capability")
    else:
        if type(native["gpu_layers"]) is not int or not 1 <= native["gpu_layers"] <= 999:
            raise ValueError("actual CUDA target offload required")
        if native["expected_compute_capability"] not in ([7, 5], [12, 0]):
            raise ValueError("explicit supported CUDA device required")
    corpus_path = pinned(plan["corpus"], path.parent, max_bytes=caps["max_source_bytes"])
    runtime_path = pinned(plan["runtime"], path.parent, max_bytes=caps["max_source_bytes"])
    client_path = pinned(native["client_source"], path.parent, max_bytes=caps["max_source_bytes"])
    runtime = json.loads(runtime_path.read_text())
    expected_runtime = {
        "schema": "nine_model_train_capture_runtime_v1",
        "binary_sha256": native["binary"]["sha256"],
        "target_sha256": native["target"]["sha256"],
        "native_source_revision": native["source_revision"],
        "teacher_client_sha256": native["client_source"]["sha256"],
        "tokenizer_metadata_sha256": native["tokenizer_metadata_sha256"],
        "chat_template_sha256": native["chat_template_sha256"],
    }
    if runtime != expected_runtime:
        raise ValueError(
            "runtime schema/native/client/target/tokenizer/template source inventory differs"
        )
    cpu_build = validate_cpu_build(plan["cpu_build"], native, path.parent) if cpu else None
    corpus = json.loads(corpus_path.read_text())
    shards = corpus["files"]["train"]["shards"]
    full_pool = None
    if "full_pool_contract" in plan:
        full_pool, full_pool_summary = validate_full_pool_contract(
            plan["full_pool_contract"], path.parent, plan["corpus"])
    selected = plan["selection"]
    if cpu and len(selected) > 9:
        raise ValueError("development_CPU supports only the bounded nine-chain pilot")
    if not isinstance(selected, list) or not selected or len(selected) * (2 if partial else 1) + golden_requests > caps["max_requests"]:
        raise ValueError("selected capture/golden requests exceed explicit count cap")
    records, loaded, source_bytes = (
        [],
        {},
        corpus_path.stat().st_size + runtime_path.stat().st_size + client_path.stat().st_size,
    )
    seen_ids, seen_content, seen_groups = set(), set(), set()
    counts = {s: {d: 0 for d in DOMAINS} for s in SPLITS}
    for choice in selected:
        if set(choice) != {"shard", "row", "prompt_id", "prompt_sha256", "domain", "split"}:
            raise ValueError(
                "selection requires exact original shard/row/content/domain/split joins"
            )
        shard_id, row_id = choice["shard"], choice["row"]
        if full_pool is not None:
            approved = full_pool.get((shard_id, row_id))
            if approved is None or any(choice[k] != approved.get("content_sha256" if k == "prompt_sha256" else k) for k in choice):
                raise ValueError("chunk selection differs from immutable global role selector")
        if type(shard_id) is not int or not 0 <= shard_id < len(shards):
            raise ValueError("original TRAIN shard ordinal invalid")
        if choice["domain"] not in DOMAINS or choice["split"] not in SPLITS:
            raise ValueError("selection domain or TRAIN-derived role invalid")
        if shard_id not in loaded:
            shard = shards[shard_id]
            paths = {
                name: pinned(
                    {"path": shard[name], "sha256": shard[name + "_sha256"]},
                    corpus_path.parent,
                    max_bytes=caps["max_source_bytes"] - source_bytes,
                )
                for name in ("prompts", "index")
            }
            source_bytes += sum(p.stat().st_size for p in paths.values())
            if source_bytes > caps["max_source_bytes"]:
                raise MemoryError("original source inventory exceeds source cap")
            positions = {c["row"] for c in selected if c["shard"] == shard_id}
            if any(type(i) is not int or i < 0 for i in positions):
                raise ValueError("original TRAIN row ordinal invalid")
            rows = {
                name: selected_jsonl_rows(p, positions, caps["max_source_row_bytes"])
                for name, p in paths.items()
            }
            loaded[shard_id] = (paths, rows)
        paths, rows = loaded[shard_id]
        if type(row_id) is not int or row_id not in rows["prompts"]:
            raise ValueError("original TRAIN row ordinal invalid")
        prompt, index = rows["prompts"][row_id], rows["index"][row_id]
        source_input = prompt.get(
            "messages" if plan["input_mode"] == "native_chat" else "prompt_text"
        )
        if plan["input_mode"] == "native_chat":
            if (
                not isinstance(source_input, list)
                or not source_input
                or any(
                    not isinstance(m, dict)
                    or set(m) != {"role", "content"}
                    or m["role"] not in ("system", "user", "assistant")
                    or not isinstance(m["content"], str)
                    for m in source_input
                )
            ):
                raise ValueError("original native chat messages missing or unsupported")
        elif not isinstance(source_input, str) or not source_input:
            raise ValueError("original raw prompt_text missing; no guessed chat flattening")
        digest = content_hash(source_input)
        if (
            prompt.get("id") != choice["prompt_id"]
            or index.get("id") != choice["prompt_id"]
            or digest != choice["prompt_sha256"]
        ):
            raise ValueError("original TRAIN prompt/index/source-content join differs")
        for row in (prompt, index):
            for key in ("split", "source_split", "role"):
                if key in row and row[key] not in ("train", "TRAIN"):
                    raise ValueError("selected source is not original TRAIN")
            domain = row.get("domain", row.get("category"))
            if domain is not None and domain != choice["domain"]:
                raise ValueError("original TRAIN domain differs")
        group = (full_pool[(shard_id, row_id)]["group_id"] if full_pool is not None
                 else index.get("group_id", index.get("group")))
        if not isinstance(group, str) or not group:
            raise ValueError("original source group identity required")
        if choice["prompt_id"] in seen_ids or (full_pool is None and (digest in seen_content or group in seen_groups)):
            raise ValueError("prompt/content/group overlap among selected TRAIN-derived roles")
        seen_ids.add(choice["prompt_id"])
        seen_content.add(digest)
        seen_groups.add(group)
        counts[choice["split"]][choice["domain"]] += 1
        records.append(
            choice
            | {
                "group_id": group,
                "source_input": source_input,
                "source_prompt": prompt,
                "source_index": index,
                "source_files": {
                    k: {"path": str(v), "sha256": shards[shard_id][k + "_sha256"]}
                    for k, v in paths.items()
                },
            }
        )
    if full_pool is None and any(min(c.values()) < 1 or len(set(c.values())) != 1 for c in counts.values()):
        raise ValueError("each TRAIN-derived role requires all three domains with balanced counts")
    # Hard CE omits block logits; every EAGLE portability golden retains final full logits.
    feature_bytes = caps["max_tokens_per_chain"] * 4 * 5 * width * (2 if partial else 1)
    retained_teacher_rows = (
        (
            len(block_teacher_indices(generated_block_anchors(1, 1 + generation["max_new_tokens"], partial=partial), 1 + generation["max_new_tokens"] if partial else None))
            if layout == "indexed"
            else caps["max_tokens_per_chain"]
        )
        if plan["objective"] == "exact_soft"
        else 0
    )
    # Native generated indexed writes candidate rows then truncates an incomplete horizon.
    # Include that temporary peak in caps; retained rows use the exact existing anchors.
    peak_teacher_rows = (
        generation["max_new_tokens"] + 1 if layout == "indexed" else retained_teacher_rows
    )
    request_bytes = feature_bytes + peak_teacher_rows * vocab * 4
    golden_bytes = 4 * (caps["max_eagle_golden_tokens"] * 8 * width + 2 * vocab)
    # Metadata/headroom is explicit rather than omitted from the retention budget.
    # Seven persisted map copies: three native/client/external receipts and
    # manifest/producer-receipt pairs for both block families. Absolute indices
    # are below 32768; 16 bytes/entry covers every current JSON indentation.
    index_map_metadata_bytes = 7 * (64 + 16 * retained_teacher_rows) if layout == "indexed" else 0
    bound = (
        len(records) * (request_bytes + 65536 + index_map_metadata_bytes)
        + (3 * (golden_bytes + 65536) if capture_goldens else 0)
        + source_bytes
    )
    largest_request = max(request_bytes, 4 * (caps["max_eagle_golden_tokens"] * 5 * width + vocab))
    if largest_request > caps["max_request_bytes"] or largest_request > caps["max_shard_bytes"]:
        raise MemoryError("declared request/shard storage bound cannot hold worst-case capture")
    if bound > caps["max_total_bytes"]:
        raise MemoryError("declared total storage bound cannot hold selected capture plan")
    return (
        plan,
        records,
        {
            "schema": "nine_model_train_capture_cost_v1",
            "status": "PENDING",
            "production_data_status": "PENDING_NATIVE_CAPTURE",
            "optimizer_updates": 0,
            "plan_sha256": expected_sha256,
            "selected_prompt_count": len(records),
            "counts": counts,
            "max_native_requests": len(records) * (2 if partial else 1) + golden_requests,
            "max_chain_rows": len(records) * caps["max_tokens_per_chain"],
            "max_prompt_tokens": caps["max_prompt_tokens"],
            "max_new_tokens": generation["max_new_tokens"],
            "max_capture_bytes": bound,
            "max_block_request_bytes": request_bytes,
            "teacher_logits_layout": layout,
            "max_block_feature_bytes": feature_bytes,
            "max_block_retained_teacher_rows": retained_teacher_rows,
            "max_block_peak_teacher_rows": peak_teacher_rows,
            "max_block_retained_teacher_bytes": retained_teacher_rows * vocab * 4,
            "max_block_index_map_metadata_bytes": index_map_metadata_bytes,
            "max_block_retained_bytes": feature_bytes + retained_teacher_rows * vocab * 4,
            "max_selected_teacher_rows": len(records) * retained_teacher_rows,
            "context_policy": "all_native_features_and_tokens_retained",
            "storage_scope": "selected pinned plan only; no full corpus fit claim",
            "max_paired_golden_bytes": golden_bytes,
            "max_wall_seconds": caps["total_timeout_seconds"],
            "full_vocab_teacher": plan["objective"] == "exact_soft",
            "coverage_selected_by": "external_pinned_plan",
            **({"full_pool_source": full_pool_summary, "label_policy": PARTIAL_LABEL_POLICY} if full_pool is not None else {}),
            "capture_portability": "PENDING fresh device numeric/decision gate",
            "corpus": plan["corpus"],
            "runtime": {"path": str(runtime_path), "sha256": file_sha256(runtime_path)},
            **(
                {
                    "execution_profile": "development_CPU",
                    "cpu_build": cpu_build,
                    "readiness_scope": "development CPU data only; CUDA/SM120 remains PENDING",
                }
                if cpu
                else {}
            ),
        },
    )


def cuda_device():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("one unambiguous visible CUDA device required")
    name, capability, uuid = [v.strip() for v in rows[0]]
    return {
        "name": name,
        "compute_capability": [int(v) for v in capability.split(".")],
        "uuid": uuid,
    }


def rss_bytes():
    # Linux measures actual resident pages; local CPU fixtures cannot admit CUDA.
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def host_available_bytes():
    """Read the actual Linux host availability; missing evidence refuses capture."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, *values = line.split()
        if key == "MemAvailable:":
            if len(values) != 2 or values[1] != "kB" or not values[0].isdigit():
                raise ValueError("Linux MemAvailable must be typed integer kB")
            fields[key] = int(values[0]) * 1024
    if set(fields) != {"MemAvailable:"}:
        raise ValueError("Linux MemAvailable evidence missing")
    return fields["MemAvailable:"]


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


CPU_OFF_OPTIONS = (
    "GGML_CUDA",
    "GGML_METAL",
    "GGML_BLAS",
    "GGML_BACKEND_DL",
    "GGML_VULKAN",
    "GGML_HIP",
    "GGML_SYCL",
    "GGML_OPENCL",
    "GGML_CANN",
    "GGML_MUSA",
    "GGML_RPC",
    "GGML_WEBGPU",
)


def require_cpu_project_libraries(text, *, allow_transitive_system_frameworks=False):
    allowed = {"libllama", "libggml", "libggml-base", "libggml-cpu"}
    stems = {name.split(".")[0] for name in re.findall(r"lib(?:llama|ggml)[\w-]*\.[^\s/]+", text)}
    if stems - allowed or (
        not allow_transitive_system_frameworks and "metal.framework" in text.lower()
    ):
        raise ValueError("CPU runtime inventory includes an unapproved GPU/backend library")


def validate_cpu_build(record, native, base):
    """Pinned CPU-only build and exact resolved dylibs; not Linux map proof."""
    path = pinned(record, base)
    proof = json.loads(path.read_text())
    fields = {
        "schema",
        "binary_sha256",
        "source_revision",
        "cmake_cache",
        "dylibs",
        "otool_links",
        "library_directory",
    }
    if (
        set(proof) != fields
        or proof["schema"] != "nine_model_cpu_teacher_build_v1"
        or proof["binary_sha256"] != native["binary"]["sha256"]
        or proof["source_revision"] != native["source_revision"]
    ):
        raise ValueError("CPU-only build proof source/binary differs")
    cache = pinned(proof["cmake_cache"], path.parent)
    options = {}
    for line in cache.read_text().splitlines():
        if line.startswith("GGML_") and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            options[key.split(":", 1)[0]] = value
    if options.get("GGML_CPU") != "ON" or any(
        options.get(k, "OFF") != "OFF" for k in CPU_OFF_OPTIONS
    ):
        raise ValueError("development_CPU requires CPU ON and GPU/BLAS/dynamic backends OFF")
    # These must be explicit actual cache entries, never inferred defaults.
    if any(options.get(k) != "OFF" for k in CPU_OFF_OPTIONS[:4]):
        raise ValueError("CPU build lacks explicit CUDA/Metal/BLAS/backend-DL OFF evidence")
    library_directory = Path(proof["library_directory"]).resolve()
    dylibs = []
    if not isinstance(proof["dylibs"], list) or not proof["dylibs"]:
        raise ValueError("CPU build requires resolved project dylib pins")
    for library in proof["dylibs"]:
        resolved = pinned(library, path.parent)
        if resolved.parent != library_directory or resolved.name.split(".")[0] not in {
            "libllama",
            "libggml",
            "libggml-base",
            "libggml-cpu",
        }:
            raise ValueError("CPU runtime project dylib directory/name differs")
        dylibs.append({"path": str(resolved), "sha256": library["sha256"]})
    if len(dylibs) != 4 or {Path(p["path"]).name.split(".")[0] for p in dylibs} != {
        "libllama",
        "libggml",
        "libggml-base",
        "libggml-cpu",
    }:
        raise ValueError("CPU runtime must pin each of the four project libraries exactly once")
    links = pinned(proof["otool_links"], path.parent).read_text()
    if any(Path(item["path"]).name.split(".")[0] + "." not in links for item in dylibs):
        raise ValueError("CPU otool dependency inventory lacks pinned project dylib")
    require_cpu_project_libraries(links)
    return {
        "proof": {"path": str(path), "sha256": record["sha256"]},
        "dylibs": dylibs,
        "library_directory": str(library_directory),
        "options": options,
        "scope": "pinned CPU build/link inventory; dyld loaded paths checked at launch",
    }


def mac_available_bytes():
    output = subprocess.check_output(["vm_stat"], text=True, timeout=10)
    match = re.search(r"page size of (\d+) bytes", output)
    if match is None:
        raise ValueError("Mac vm_stat page size unavailable")
    counts = {}
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        value = re.search(r"^" + re.escape(label) + r":\s*(\d+)\.", output, re.MULTILINE)
        if value is None:
            raise ValueError("Mac vm_stat free/reclaimable page count unavailable")
        counts[label] = int(value.group(1))
    return int(match.group(1)) * sum(counts.values())


def mac_rss_bytes(pid=None):
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid or os.getpid())], text=True, timeout=10
    ).strip()
    if not output.isdigit():
        raise ValueError("Mac process RSS unavailable")
    return int(output) * 1024


def mac_cpu_device():
    if platform.system() != "Darwin":
        raise ValueError("development_CPU is a local Mac-only profile")
    name = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=10
    ).strip()
    return {
        "name": name,
        "backend": "CPU",
        "compute_capability": None,
        "hardware_scope": "Mac CPU; no GPU query or compute",
    }


def cpu_loaded_library_proof(teacher, build):
    text = Path(teacher.log.name).read_text(errors="replace")
    require_cpu_project_libraries(text, allow_transitive_system_frameworks=True)
    pid = teacher.process.pid
    loaded = set()
    for line in text.splitlines():
        if "dyld[" + str(pid) + "]" in line:
            for match in re.findall(r"(/[^\r\n]+?\.dylib)", line):
                path = Path(match).resolve()
                if path.name.startswith(("libllama", "libggml")):
                    loaded.add(str(path))
    missing = [p for p in build["dylibs"] if p["path"] not in loaded]
    if loaded - {p["path"] for p in build["dylibs"]}:
        raise ValueError("Mac dyld loaded extra project library paths outside exact CPU pins")
    if missing:
        raise ValueError("Mac dyld log does not prove all pinned project libraries loaded")
    for record in build["dylibs"]:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Mac pinned runtime dylib changed during capture")
    return {
        "scope": "mac_dyld_loaded_path_and_file_hash",
        "execution_scope": "CPU native model execution under pinned CPU-only GGML runtime",
        "observed_system_metal_framework": [
            line
            for line in text.splitlines()
            if "dyld[" + str(pid) + "]" in line
            and "/System/Library/Frameworks/Metal.framework/" in line
        ],
        "actual_dyld_paths_checked": True,
        "actual_linux_mapping_checked": False,
        "libraries": build["dylibs"],
        "producer_pid": pid,
        "dyld_log_sha256_at_check": file_sha256(Path(teacher.log.name)),
    }


def run_capture(
    plan_path,
    expected_sha256,
    output_root,
    *,
    execute=False,
    teacher_factory=None,
    device_query=None,
    rss_query=None,
    available_query=None,
    clock=time.monotonic,
    execution_profile="production_CUDA",
    chain_ids=None,
):
    cpu = execution_profile == "development_CPU"
    injected = teacher_factory is not None
    plan, records, cost = prepare_plan(
        plan_path, expected_sha256, execution_profile=execution_profile
    )
    if chain_ids is not None and (
        not isinstance(chain_ids, dict) or set(chain_ids) != {r["prompt_id"] for r in records}
        or any(not isinstance(cid, str) or not cid for cid in chain_ids.values())
        or len(set(chain_ids.values())) != len(chain_ids)
    ):
        raise ValueError("logical shard exact unique prompt/chain ID map differs")
    device_query = device_query or (mac_cpu_device if cpu else cuda_device)
    rss_query = rss_query or (mac_rss_bytes if cpu else rss_bytes)
    available_query = available_query or (mac_available_bytes if cpu else host_available_bytes)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError("refuse to overwrite capture history")
    output.mkdir(parents=True)
    report = cost | {
        "artifact_kind": "synthetic_fixture"
        if injected
        else "development_CPU"
        if cpu
        else "production",
        "status": "PENDING",
        "failure": None,
        "producer_closed": None,
        "orchestrator_source_sha256": file_sha256(Path(__file__)),
        **(
            {
                "execution_profile": "development_CPU",
                "sm120_readiness": "PENDING",
                "production_data_status": "PENDING_CPU_DEVELOPMENT_CAPTURE",
            }
            if cpu
            else {}
        ),
    }
    write_json(output / "cost.json", cost)
    inventory = {
        "schema": "block_train_inventory_v1",
        "prompts": {
            r["prompt_id"]: {"sha256": r["prompt_sha256"], "domain": r["domain"], "split": "TRAIN"}
            for r in records
        },
    }
    inventory_pin = write_json(output / "train-inventory.json", inventory)
    write_json(
        output / "original-source-joins.json", {"plan_sha256": expected_sha256, "records": records}
    )
    if not execute:
        write_json(output / "capture-report.json", report)
        return report
    teacher = None
    old_handlers = {}
    monitor_stop = threading.Event()
    monitor = None
    monitor_error = []
    saved_environment = {}
    construction_started = False
    start = clock()
    caps, native = plan["caps"], plan["native"]

    report["host_available_floor_bytes"] = caps["min_host_available_bytes"]
    report["host_available_scope"] = (
        "Mac vm_stat (free+inactive+speculative)*page_size; excludes compressed/purgeable"
        if cpu
        else "actual Linux MemAvailable; separate from GPU resource release"
    )

    def budget(*, full_storage=True):
        if clock() - start >= caps["total_timeout_seconds"]:
            raise TimeoutError("capture total wall cap reached")
        available = available_query()
        if type(available) is not int or available < 0:
            raise ValueError("Linux MemAvailable evidence must be typed nonnegative bytes")
        report["last_host_available_bytes"] = available
        report["min_observed_host_available_bytes"] = min(
            report.get("min_observed_host_available_bytes", available), available
        )
        floor = (
            caps["min_host_available_live_bytes"]
            if cpu and construction_started
            else caps["min_host_available_bytes"]
        )
        if available < floor:
            raise MemoryError(
                "capture host available memory below explicit prelaunch/live floor"
                if cpu
                else "capture Linux MemAvailable below explicit host floor"
            )
        measured_rss = rss_query()
        if teacher is not None and hasattr(teacher, "process") and teacher.process.poll() is None:
            if cpu:
                measured_rss += mac_rss_bytes(teacher.process.pid)
            else:
                child_stat = Path(f"/proc/{teacher.process.pid}/statm")
                measured_rss += int(child_stat.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        report["last_owner_and_producer_rss_bytes"] = measured_rss
        report["max_observed_owner_and_producer_rss_bytes"] = max(
            report.get("max_observed_owner_and_producer_rss_bytes", 0), measured_rss
        )
        if measured_rss > caps["max_host_rss_bytes"]:
            raise MemoryError("capture owner plus native producer host RSS cap reached")
        # Retained-tree enumeration belongs at write/chain/family boundaries.
        # Read-only finite slices still check time, available memory, RSS and
        # free disk on every callback, without rescanning unchanged files.
        if full_storage and tree_bytes(output) > caps["max_total_bytes"]:
            raise MemoryError("capture retained storage cap reached")
        if shutil.disk_usage(output).free < caps["min_free_disk_bytes"]:
            raise MemoryError("capture free disk floor reached")

    def stopped(signum, _frame):
        if monitor_error:
            raise monitor_error[0]
        raise InterruptedError(f"capture STOP signal {signum}")

    def watch_cpu():
        last_progress = 0.0
        while not monitor_stop.wait(2):
            try:
                budget()
                if clock() - last_progress >= 15:
                    last_progress = clock()
                    progress = {
                        "schema": "development_cpu_capture_progress_v1",
                        "artifact_kind": report["artifact_kind"],
                        "elapsed_seconds": clock() - start,
                        "completed_prompt_count": report.get("completed_prompt_count", 0),
                        "rss_bytes": report["last_owner_and_producer_rss_bytes"],
                        "host_available_bytes": report["last_host_available_bytes"],
                    }
                    with (output / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
            except Exception as error:
                if monitor_stop.is_set():
                    return
                monitor_error.append(error)
                os.kill(os.getpid(), signal.SIGINT)
                return

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stopped)
        binary = pinned(native["binary"], Path(plan_path).resolve().parent)
        target = pinned(native["target"], Path(plan_path).resolve().parent)
        device = device_query()
        if device["compute_capability"] != native["expected_compute_capability"] or (
            cpu and device.get("backend") != "CPU"
        ):
            raise ValueError("actual CUDA device differs from selected capture device")
        report["producer_device"] = device
        budget()
        if cpu:
            monitor = threading.Thread(
                target=watch_cpu, name="development-cpu-resource-monitor", daemon=True
            )
            monitor.start()
        if teacher_factory is None:
            from capture_block_qat_teacher import NativeTeacher

            teacher_factory = NativeTeacher
            actual_client_path = Path(inspect.getfile(NativeTeacher)).resolve()
            if file_sha256(actual_client_path) != native["client_source"]["sha256"]:
                raise ValueError(
                    "current production NativeTeacher client source differs from plan pin"
                )
            if (
                "max_prompt_tokens"
                not in inspect.signature(NativeTeacher.generate_capture).parameters
            ):
                raise ValueError("native producer lacks prompt token cap before decoding")
            report["actual_teacher_client_source"] = {
                "path": str(actual_client_path),
                "sha256": file_sha256(actual_client_path),
            }
        if cpu:
            for key in (
                "DYLD_LIBRARY_PATH",
                "DYLD_PRINT_LIBRARIES",
                "DYLD_INSERT_LIBRARIES",
                "DYLD_FRAMEWORK_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH",
            ):
                saved_environment[key] = os.environ.get(key)
                os.environ.pop(key, None)
            os.environ["DYLD_LIBRARY_PATH"] = cost["cpu_build"]["library_directory"]
            os.environ["DYLD_PRINT_LIBRARIES"] = "1"
        construction_started = True
        teacher = teacher_factory(
            binary,
            target,
            output / "native",
            target_sha256=native["target"]["sha256"],
            max_tokens=caps["max_tokens_per_chain"],
            gpu_layers=native["gpu_layers"],
            producer_source_revision=native["source_revision"],
            timeout_seconds=min(caps["request_timeout_seconds"], caps["total_timeout_seconds"]),
        )
        chains, goldens, block_goldens = [], [], []
        for r in records:
            budget()
            ancestry = {k: r[k] for k in ("prompt_id", "prompt_sha256", "domain")} | {
                "source_split": "TRAIN"
            }
            kwargs = {
                "messages" if plan["input_mode"] == "native_chat" else "prompt_text": r[
                    "source_input"
                ]
            }
            teacher.timeout = min(
                caps["request_timeout_seconds"], caps["total_timeout_seconds"] - (clock() - start)
            )
            receipt = teacher.generate_capture(
                **kwargs,
                max_new_tokens=plan["generation"]["max_new_tokens"],
                max_prompt_tokens=caps["max_prompt_tokens"],
                template_mode=plan["input_mode"],
                tap_ids=TAPS,
                logits_mode=("indexed" if plan.get("teacher_logits_layout") == "indexed" else "all")
                if plan["objective"] == "exact_soft"
                else "none",
                chain_ancestry=ancestry,
                **({"label_policy": PARTIAL_LABEL_POLICY} if plan.get("label_policy") == PARTIAL_LABEL_POLICY else {}),
            )
            validate_generated(receipt, r, plan, device, execution_profile=execution_profile)
            if cpu and not injected:
                report["cpu_runtime_proof"] = cpu_loaded_library_proof(teacher, cost["cpu_build"])
                report["native_runtime_binding"] = receipt.get("native_runtime_binding")
            if sum(Path(d["path"]).stat().st_size for d in receipt["files"].values()) > min(
                caps["max_request_bytes"], caps["max_shard_bytes"]
            ):
                raise MemoryError("actual native request/shard bytes exceed explicit bound")
            receipt_pin = write_json(output / "receipts" / f"{len(chains):06d}-block.json", receipt)
            generation_receipt = joined_generation_source(receipt) if "generation_source" in receipt else receipt
            tokens, length = receipt["tokens"], generation_receipt["prompt_length"]
            anchors = generated_block_anchors(length, len(tokens), partial=plan.get("label_policy") == PARTIAL_LABEL_POLICY)
            if not anchors:
                raise ValueError("native generated chain has no complete seven-slot author horizon")
            chains.append(
                {k: r[k] for k in ("prompt_id", "prompt_sha256", "domain", "split")}
                | {
                    "chain_id": chain_ids[r["prompt_id"]] if chain_ids is not None else f"chain-{len(chains):06d}",
                    "prompt_length": length,
                    "anchors": anchors,
                    "include_logits": plan["objective"] == "exact_soft",
                    "native_receipt": receipt_pin,
                    **({"label_policy": PARTIAL_LABEL_POLICY} if plan.get("label_policy") == PARTIAL_LABEL_POLICY else {}),
                }
            )
            report["completed_prompt_count"] = len(chains)
            budget()
            if not plan.get("capture_goldens", True) or r["domain"] in {g["domain"] for g in goldens}:
                continue
            if length >= caps["max_eagle_golden_tokens"]:
                raise ValueError(
                    "selected golden prefix cap cannot include original prompt and continuation"
                )
            golden_tokens = tokens[: min(len(tokens), caps["max_eagle_golden_tokens"])]
            golden_ancestry = ancestry | {"prompt_length": length}
            history = prefix_history(receipt.get("decode_history"), len(golden_tokens))
            teacher.timeout = min(
                caps["request_timeout_seconds"], caps["total_timeout_seconds"] - (clock() - start)
            )
            block_golden = teacher.capture_prefix(
                golden_tokens,
                TAPS,
                logits_mode="last",
                chain_ancestry=golden_ancestry,
                decode_history=history,
            )
            validate_replay(
                block_golden,
                golden_tokens,
                TAPS,
                native,
                device,
                execution_profile=execution_profile,
            )
            block_pin = write_json(
                output / "receipts" / f"{len(block_goldens):06d}-block-golden.json", block_golden
            )
            block_goldens.append({"chain_id": chains[-1]["chain_id"], "native_receipt": block_pin})
            budget()
            teacher.timeout = min(
                caps["request_timeout_seconds"], caps["total_timeout_seconds"] - (clock() - start)
            )
            golden = teacher.capture_prefix(
                golden_tokens,
                EAGLE_TAPS,
                logits_mode="last",
                chain_ancestry=golden_ancestry,
                decode_history=history,
            )
            validate_replay(
                golden,
                golden_tokens,
                EAGLE_TAPS,
                native,
                device,
                execution_profile=execution_profile,
            )
            golden_pin = write_json(output / "receipts" / f"{len(goldens):06d}-eagle.json", golden)
            goldens.append(
                {
                    "chain_id": chains[-1]["chain_id"],
                    "prompt_id": r["prompt_id"],
                    "domain": r["domain"],
                    "split": r["split"],
                    "generation_receipt": receipt.get("generation_source", receipt_pin),
                    "native_receipt": golden_pin,
                }
            )
            budget()
        runtime_pin = cost["runtime"]
        manifests = {}
        for family in ("dspark", "dflash"):
            budget()
            block_plan = {
                "schema": "block_capture_plan_v1",
                "family": family,
                "vocab_size": plan["vocab_size"],
                "target_width": plan["target_width"],
                "mask_token_id": plan["mask_token_id"],
                "train_inventory": inventory_pin,
                "runtime": runtime_pin,
                "chains": chains,
            }
            pin = write_json(output / f"{family}-capture-plan.json", block_plan)
            manifest = import_capture_plan(
                pin["path"],
                expected_sha256=pin["sha256"],
                output_dir=output / family,
                max_capture_bytes=caps["max_total_bytes"],
                budget_check=budget,
                audit_budget_check=lambda: budget(full_storage=False),
            )
            budget()
            manifests[family] = {"path": str(manifest), "sha256": file_sha256(manifest)}
        eagle_pin, block_golden_pins = None, {}
        if plan.get("capture_goldens", True):
            eagle_pin = write_json(
                output / "eagle-goldens.json",
                {
                    "schema": "nine_model_cpu_development_goldens_v1"
                    if cpu
                    else "nine_model_train_capture_goldens_v1",
                    "family": "eagle",
                    "tap_ids": list(EAGLE_TAPS),
                    "target_sha256": native["target"]["sha256"],
                    "train_inventory": inventory_pin,
                    "vocab_size": plan["vocab_size"],
                    "target_width": plan["target_width"],
                    "cases": [{k: g[k] for k in ("chain_id", "native_receipt")} for g in goldens],
                },
            )
            from check_block_capture_portability import NativeCaptureGoldens

            if not cpu:
                NativeCaptureGoldens(eagle_pin["path"], expected_sha256=eagle_pin["sha256"])
            block_golden_pins = {}
            for family in ("dspark", "dflash"):
                pin = write_json(
                    output / f"{family}-goldens.json",
                    {
                        "schema": "nine_model_cpu_development_goldens_v1"
                        if cpu
                        else "nine_model_train_capture_goldens_v1",
                        "family": family,
                        "tap_ids": list(TAPS),
                        "target_sha256": native["target"]["sha256"],
                        "train_inventory": inventory_pin,
                        "vocab_size": plan["vocab_size"],
                        "target_width": plan["target_width"],
                        "cases": block_goldens,
                    },
                )
                if not cpu:
                    NativeCaptureGoldens(pin["path"], expected_sha256=pin["sha256"])
                block_golden_pins[family] = pin
            write_json(output / "eagle-generation-joins.json", {"cases": goldens})
        budget()
        report.update(
            status="PASS",
            production_data_status=(
                "CAPTURED"
                if report["artifact_kind"] == "production"
                else "DEVELOPMENT_CPU_ONLY"
                if report["artifact_kind"] == "development_CPU"
                else "SYNTHETIC_ONLY"
            ),
            manifests=manifests,
            eagle_goldens=eagle_pin,
            block_goldens=block_golden_pins,
            original_train_inventory=inventory_pin,
            completed_prompt_count=len(chains),
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="FAIL", failure={"type": type(error).__name__, "message": str(error)})
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=30)
            if monitor.is_alive():
                report.update(
                    status="FAIL",
                    failure={
                        "type": "ResourceError",
                        "message": "CPU resource monitor did not stop",
                    },
                )
        if teacher is not None:
            try:
                teacher.close()
                report["producer_closed"] = teacher.closed
                if hasattr(teacher, "process"):
                    report["producer_pid"] = teacher.process.pid
                    report["producer_returncode"] = teacher.process.poll()
                if not teacher.closed or (
                    hasattr(teacher, "process") and teacher.process.poll() is None
                ):
                    raise RuntimeError("owned native producer remains live after close")
            except Exception as error:
                report.update(
                    status="FAIL",
                    failure={"type": type(error).__name__, "message": f"producer cleanup: {error}"},
                )
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report["elapsed_wall_seconds"] = clock() - start
        report["retained_bytes"] = tree_bytes(output)
        write_json(output / "capture-report.json", report)
    return report


def prefix_history(history, length):
    """Cut native-issued contiguous decode history at an actual token boundary."""
    if not isinstance(history, list) or not history:
        raise ValueError("native-generated decode history missing")
    offset, bounded = 0, []
    for record in history:
        if (
            not isinstance(record, dict)
            or set(record) != {"offset", "count", "phase", "kv_reused_from_same_chain"}
            or record["offset"] != offset
            or type(record["count"]) is not int
            or not 1 <= record["count"] <= 256
            or record["phase"] not in ("prefill", "target_only_greedy")
            or type(record["kv_reused_from_same_chain"]) is not bool
            or record["kv_reused_from_same_chain"] != (offset > 0)
        ):
            raise ValueError("native-issued decode history is not contiguous exact-chain execution")
        if offset < length:
            bounded.append(record | {"count": min(record["count"], length - offset)})
        offset += record["count"]
    if offset < length:
        raise ValueError("native-issued decode history shorter than exact prefix")
    return bounded


def _validate_capture(
    receipt, tokens, taps, native, device, *, execution_profile="production_CUDA"
):
    cpu = execution_profile == "development_CPU"
    buffers = receipt.get("executed_result_buffers", [])
    storage = receipt.get("target_storage_buffers", {})
    hardware = receipt.get("hardware", [])
    execution_ok = (
        bool(buffers)
        and all(isinstance(b, str) and re.fullmatch(r"CPU(?:_Mapped)?", b) for b in buffers)
        and bool(storage)
        and all(re.fullmatch(r"CPU(?:_Mapped)?", b) for b in storage)
        and hardware == [device["name"]]
        and receipt.get("gpu_layers") == 0
        if cpu
        else bool(buffers)
        and all(isinstance(b, str) and re.fullmatch(r"CUDA[0-9]+", b) for b in buffers)
        and any(re.fullmatch(r"CUDA[0-9]+", b) for b in storage)
    )
    if (
        receipt.get("schema") != "block_native_teacher_request_v1"
        or receipt.get("complete") is not True
        or receipt.get("optimizer_updates") != 0
        or receipt.get("tokens") != tokens
        or receipt.get("target_precision") != "F16"
        or receipt.get("kv_type") != "F16"
        or receipt.get("tap_ids") != list(taps)
        or receipt.get("target_sha256") != native["target"]["sha256"]
        or receipt.get("producer_binary_sha256") != native["binary"]["sha256"]
        or receipt.get("producer_source_revision") != native["source_revision"]
        or receipt.get("client_source_sha256") != native["client_source"]["sha256"]
        or not any(device["name"] in str(x) for x in receipt.get("hardware", []))
        or not execution_ok
        or receipt.get("teacher_context_reset_between_requests") is not True
    ):
        raise ValueError("native receipt lacks exact target/prefix/taps/actual CUDA producer proof")
    mode = receipt.get("logits_mode")
    wanted = {"features"} if mode == "none" else {"features", "logits"}
    if mode not in ("none", "last", "all", "indexed") or set(receipt.get("files", {})) != wanted:
        raise ValueError("native feature/full-vocabulary file inventory differs")
    if receipt.get("features_shape", [])[:2] != [len(tokens), len(taps)]:
        raise ValueError("native replay feature shape differs from exact prefix")
    for descriptor in receipt.get("files", {}).values():
        path = Path(descriptor["path"])
        expected_bytes = math.prod(descriptor["shape"]) * 4
        if (
            descriptor.get("dtype") != "float32"
            or path.stat().st_size != expected_bytes
            or file_sha256(path) != descriptor["sha256"]
        ):
            raise ValueError("original native raw capture bytes differ from receipt")


def validate_replay(receipt, tokens, taps, native, device, *, execution_profile="production_CUDA"):
    if (
        receipt.get("prefix_contract") != REPLAY_PREFIX_CONTRACT
        or any(k in receipt for k in ("generation", "prompt", "prompt_source_sha256"))
        or receipt.get("prefix_freshness") != "caller_current_student_prefix"
    ):
        raise ValueError("native replay prefix contract differs")
    _validate_capture(receipt, tokens, taps, native, device, execution_profile=execution_profile)


def validate_generated(receipt, record, plan, device, *, execution_profile="production_CUDA"):
    if plan.get("label_policy") == PARTIAL_LABEL_POLICY:
        source = joined_generation_source(receipt)
        validate_generated(source, record, {k: v for k, v in (plan | {"objective": "hard_ce"}).items()
                                          if k != "label_policy"}, device, execution_profile=execution_profile)
        validate_replay(receipt, source["tokens"], TAPS, plan["native"], device,
                        execution_profile=execution_profile)
        expected = block_teacher_indices(generated_block_anchors(source["prompt_length"], len(source["tokens"]), partial=True), len(source["tokens"]))
        validate_logits_indices(receipt.get("logits_indices"), len(source["tokens"]), expected=expected)
        if receipt.get("logits_mode") != "indexed" or receipt.get("logits_shape") != [len(expected), plan["vocab_size"]]:
            raise ValueError("partial indexed replay does not retain all loss-bearing rows")
        return
    tokens = receipt.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or not 0 <= t < plan["vocab_size"] for t in tokens)
    ):
        raise ValueError("native generated tokens outside exact full vocabulary")
    history = receipt.get("decode_history")
    if sum(r["count"] for r in prefix_history(history, len(tokens))) != len(tokens) or sum(
        r["count"] for r in history
    ) != len(tokens):
        raise ValueError("native-generated decode history differs from complete token chain")
    if receipt.get("prefix_contract") != GENERATED_PREFIX_CONTRACT:
        raise ValueError("native generated prefix contract differs")
    _validate_capture(
        receipt, tokens, TAPS, plan["native"], device, execution_profile=execution_profile
    )
    validate_native_generation(
        receipt,
        prompt_sha256=record["prompt_sha256"],
        prompt_length=receipt.get("prompt_length"),
        token_count=len(tokens),
        tokenizer_metadata_sha256=plan["native"]["tokenizer_metadata_sha256"],
        chat_template_sha256=plan["native"]["chat_template_sha256"],
    )
    mode = (
        ("indexed" if plan.get("teacher_logits_layout") == "indexed" else "all")
        if plan["objective"] == "exact_soft"
        else "none"
    )
    if mode == "indexed":
        expected = block_teacher_indices(
            generated_block_anchors(receipt["prompt_length"], len(tokens))
        )
        validate_logits_indices(receipt.get("logits_indices"), len(tokens), expected=expected)
        if receipt.get("logits_selection") != BLOCK_LOGITS_SELECTION:
            raise ValueError("generated indexed logit selection differs from issued block policy")
    if (
        receipt.get("features_shape") != [len(tokens), 5, plan["target_width"]]
        or receipt.get("logits_mode") != mode
        or receipt.get("logits_shape")
        != [
            len(tokens) if mode == "all" else len(expected) if mode == "indexed" else 0,
            plan["vocab_size"],
        ]
    ):
        raise ValueError("generated native feature/full-vocabulary shape differs")
    # Native helper must bind its actual tokenizer, model template and generation history.
    generation = receipt.get("generation", {})
    if (
        receipt.get("prompt_length", 0) < 1
        or receipt.get("prompt_length", 0) > plan["caps"]["max_prompt_tokens"]
        or receipt.get("prompt", {}).get("max_prompt_tokens") != plan["caps"]["max_prompt_tokens"]
        or len(receipt.get("tokens", [])) > plan["caps"]["max_tokens_per_chain"]
        or generation.get("mode") != "native_target_greedy"
        or generation.get("max_new_tokens") != plan["generation"]["max_new_tokens"]
        or generation.get("generated_tokens") != len(receipt["tokens"]) - receipt["prompt_length"]
        or generation.get("stop_eog") is not True
        or generation.get("termination") not in ("max_new_tokens", "eog")
        or receipt.get("prompt", {}).get("template_mode") != plan["input_mode"]
        or receipt.get("prompt_source_sha256") != record["prompt_sha256"]
        or receipt.get("tokenizer_metadata_sha256") != plan["native"]["tokenizer_metadata_sha256"]
        or (
            plan["input_mode"] == "native_chat"
            and receipt.get("chat_template_sha256") != plan["native"]["chat_template_sha256"]
        )
    ):
        raise ValueError("authentic native tokenizer/template/target generation history differs")
    ancestry = {k: record[k] for k in ("prompt_id", "prompt_sha256", "domain")} | {
        "source_split": "TRAIN",
        "prompt_length": receipt["prompt_length"],
    }
    if receipt.get("chain_ancestry") != ancestry:
        raise ValueError("generated prefix differs from original TRAIN ancestry")



def run_shard_publication(request_path, output_path, **capture_kwargs):
    """First actual capture publication for the bounded logical shard controller.

    Restoring original bytes belongs to verified archive/replay ownership. This
    producer refuses a recapture claim when original_identity is already sealed.
    """
    request_path, output_path = Path(request_path).resolve(), Path(output_path).resolve()
    request = json.loads(request_path.read_text())
    fields = {"schema", "plan", "plan_sha256", "shard", "directory", "original_identity",
              "generation_variant", "capture_bytes_bound"}
    if not fields <= set(request) or set(request) - fields - {"original_metadata", "replay_admission", "replay_probe"} or request["schema"] != "block_shard_restore_request_v1":
        raise ValueError("unsupported logical shard producer request")
    global_path = pinned(request["plan"], request_path.parent)
    global_plan = json.loads(global_path.read_text())
    if (global_plan.get("schema") != "block_logical_shard_plan_v1"
        or identity(global_plan) != request["plan_sha256"]
        or global_plan.get("generation_variant") != request["generation_variant"]
        or request["capture_bytes_bound"] != 8*1024**3):
        raise ValueError("logical shard plan/generation/8 GiB identity differs")
    geometry = global_plan.get("geometry", {})
    family = geometry.get("family")
    if family not in ("dspark", "dflash"):
        raise ValueError("logical shard family must be dspark or dflash")
    if request["original_identity"] is not None:
        if "original_metadata" not in request:
            raise ValueError("original shard identity already sealed: verified archive/replay restoration required")
        metadata_path = pinned(request["original_metadata"], request_path.parent)
        metadata = json.loads(metadata_path.read_text())
        original_manifest_pin = metadata["files"][metadata["manifest_relative_path"]]
        original_manifest = json.loads(
            pinned(original_manifest_pin, metadata_path.parent).read_text()
        )
        if any(original_manifest.get(key) != wanted for key, wanted in geometry.items()):
            raise ValueError("original replay manifest differs from logical shard family/geometry")
        from replay_dspark_shard import replay
        return replay(request_path, output_path, **capture_kwargs)
    directory = Path(request["directory"])
    if not directory.is_absolute() or directory.exists():
        raise ValueError("fresh absolute owned physical shard directory required")
    directory = directory.resolve()
    shard = request["shard"]
    ids = global_plan["shards"][shard]
    mapping = {global_plan["chains"][cid]["prompt_id"]: cid for cid in ids}
    if len(mapping) != len(ids):
        raise ValueError("logical shard duplicate prompt ownership")
    plan_pin = global_plan["capture_plans"][shard]["remote"]
    plan_path = pinned(plan_pin, global_path.parent)
    capture_plan = json.loads(plan_path.read_text())
    if (set(mapping) != {r["prompt_id"] for r in capture_plan["selection"]}
        or capture_plan.get("label_policy") != PARTIAL_LABEL_POLICY
        or capture_plan["caps"]["max_total_bytes"] > request["capture_bytes_bound"]):
        raise ValueError("physical capture selection/partial labels/storage differs from global shard")
    if any(capture_plan.get(key) != geometry[key]
           for key in ("vocab_size", "target_width", "mask_token_id") if key in geometry):
        raise ValueError("native capture geometry differs from logical shard")
    if "taps" in geometry and geometry["taps"] != list(TAPS):
        raise ValueError("logical shard must preserve all five native target taps")
    if "tap_semantics" in geometry and geometry["tap_semantics"] != "native_layer_input_f32":
        raise ValueError("logical shard native target tap semantics differs")
    by_prompt = {global_plan["chains"][cid]["prompt_id"]: global_plan["chains"][cid] for cid in ids}
    for row in capture_plan["selection"]:
        if any(row[k] != by_prompt[row["prompt_id"]][k] for k in ("prompt_sha256", "domain", "split")):
            raise ValueError("physical shard source/role differs from immutable logical plan")
    report = run_capture(plan_path, plan_pin["sha256"], directory, execute=True,
                         chain_ids=mapping, **capture_kwargs)
    if report["status"] != "PASS":
        raise RuntimeError("native shard capture/import failed: " + json.dumps(report["failure"]))
    manifest = report["manifests"][family]
    admission_path = directory / family / "completed-admission.json"
    admission = {"path": str(admission_path), "sha256": file_sha256(admission_path)}
    # No external tensor/receipt may masquerade as this physical publication.
    from w1a1_eagle.block_data import BlockDataset
    ds = BlockDataset(manifest["path"], expected_sha256=manifest["sha256"],
                      admission_path=admission["path"], admission_sha256=admission["sha256"])
    try:
        if any(ds.manifest.get(key) != wanted for key, wanted in geometry.items()):
            raise ValueError("actual native publication differs from logical shard family/geometry")
        if set(ds.chains) != set(ids):
            raise ValueError("actual native chain membership differs from logical shard")
        for pin in ds._fingerprints.values():
            # Global runtime/original TRAIN inventory may live outside, tensors
            # and native generation/replay receipts must be physically owned.
            p = Path(pin["path"])
            if p.suffix in (".npy", ".f32") and not p.is_relative_to(directory):
                raise ValueError("physical shard tensor escapes owned directory")
        for chain in ds.chains.values():
            for kind in ("tokens", "features", "logits", "native_receipt"):
                if chain[kind] is not None and not Path(chain[kind]["path"]).is_relative_to(directory):
                    raise ValueError("physical shard native artifact escapes owned directory")
        for native in ds.native_receipts.values():
            if "generation_source" in native:
                source_pin = native["generation_source"]
                if not Path(source_pin["path"]).is_relative_to(directory):
                    raise ValueError("physical shard original generation receipt escapes directory")
                original = joined_generation_source(native)
                if any(not Path(r["path"]).is_relative_to(directory) for r in original["files"].values()):
                    raise ValueError("physical shard original generation tensor escapes directory")
        seal = {cid: {"artifacts": {k: c[k]["sha256"] for k in ("tokens", "features", "logits", "native_receipt")},
                      "anchors": c["anchors"], "prompt_length": c["prompt_length"],
                      "logits_indices": c.get("logits_indices")} for cid, c in ds.chains.items()}
    finally:
        ds.close()
    if tree_bytes(directory) > request["capture_bytes_bound"]:
        raise MemoryError("actual complete shard publication exceeds immutable 8 GiB bound")
    write_json(directory / "original-tensor-seal.json", {"schema": "block_shard_original_tensor_seal_v1",
                                                       "plan_sha256": request["plan_sha256"], "shard": shard,
                                                       "identity": seal})
    return write_json(output_path, {"schema": "block_shard_publication_v1", "status": "PASS",
                      "request": {"path": str(request_path), "sha256": file_sha256(request_path)},
                      "plan_sha256": request["plan_sha256"], "shard": shard, "directory": str(directory),
                      "manifest": manifest, "admission": admission})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--request", type=Path, help="Pinned logical controller first-capture request")
    parser.add_argument("--output", type=Path, help="Actual controller publication receipt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--development-cpu",
        action="store_true",
        help="Explicit local Mac CPU data development; never CUDA readiness",
    )
    args = parser.parse_args()
    if args.request is not None:
        if args.output is None or args.plan is not None or args.development_cpu:
            parser.error("--request requires --output and the coordinated CUDA producer")
        print(json.dumps(run_shard_publication(args.request, args.output)))
        return 0
    if args.plan is None or args.plan_sha256 is None or args.output_root is None:
        parser.error("--plan, --plan-sha256 and --output-root required without --request")
    report = run_capture(
        args.plan,
        args.plan_sha256,
        args.output_root,
        execute=args.execute,
        execution_profile="development_CPU" if args.development_cpu else "production_CUDA",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "production_data_status": report["production_data_status"],
                "failure": report["failure"],
            }
        )
    )
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
