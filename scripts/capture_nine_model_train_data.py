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
    DOMAINS,
    GENERATED_PREFIX_CONTRACT,
    REPLAY_PREFIX_CONTRACT,
    SPLITS,
    TAPS,
    file_sha256,
    import_capture_plan,
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
        set(plan) != fields
        or plan["schema"] != "nine_model_train_capture_plan_v1"
        or (cpu and plan["execution_profile"] != "development_CPU")
    ):
        raise ValueError("unsupported explicit TRAIN capture plan")
    if plan["input_mode"] not in ("raw_text", "native_chat"):
        raise ValueError("explicit raw_text or native_chat input mode required")
    if plan["objective"] not in ("hard_ce", "exact_soft"):
        raise ValueError("only hard_ce or bounded full-vocabulary exact_soft supported")
    vocab = positive(plan["vocab_size"], "vocab_size")
    width = positive(plan["target_width"], "target_width")
    if type(plan["mask_token_id"]) is not int or not 0 <= plan["mask_token_id"] < vocab:
        raise ValueError("mask token outside full vocabulary")
    generation = plan["generation"]
    if set(generation) != {"max_new_tokens", "algorithm"} or generation["algorithm"] != "greedy":
        raise ValueError("native target-only greedy generation required")
    positive(generation["max_new_tokens"], "max_new_tokens")
    if generation["max_new_tokens"] < 8:
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
    selected = plan["selection"]
    if cpu and len(selected) > 9:
        raise ValueError("development_CPU supports only the bounded nine-chain pilot")
    if not isinstance(selected, list) or not selected or len(selected) + 6 > caps["max_requests"]:
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
        group = index.get("group_id", index.get("group"))
        if not isinstance(group, str) or not group:
            raise ValueError("original source group identity required")
        if choice["prompt_id"] in seen_ids or digest in seen_content or group in seen_groups:
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
    if any(min(c.values()) < 1 or len(set(c.values())) != 1 for c in counts.values()):
        raise ValueError("each TRAIN-derived role requires all three domains with balanced counts")
    # Hard CE omits block logits; every EAGLE portability golden retains final full logits.
    block_row_bytes = 4 * (5 * width + (vocab if plan["objective"] == "exact_soft" else 0))
    request_bytes = caps["max_tokens_per_chain"] * block_row_bytes
    golden_bytes = 4 * (caps["max_eagle_golden_tokens"] * 8 * width + 2 * vocab)
    # Metadata/headroom is explicit rather than omitted from the retention budget.
    bound = len(records) * (request_bytes + 65536) + 3 * (golden_bytes + 65536) + source_bytes
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
            "max_native_requests": len(records) + 6,
            "max_chain_rows": len(records) * caps["max_tokens_per_chain"],
            "max_prompt_tokens": caps["max_prompt_tokens"],
            "max_new_tokens": generation["max_new_tokens"],
            "max_capture_bytes": bound,
            "max_block_request_bytes": request_bytes,
            "max_paired_golden_bytes": golden_bytes,
            "max_wall_seconds": caps["total_timeout_seconds"],
            "full_vocab_teacher": plan["objective"] == "exact_soft",
            "coverage_selected_by": "external_pinned_plan",
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
):
    cpu = execution_profile == "development_CPU"
    injected = teacher_factory is not None
    plan, records, cost = prepare_plan(
        plan_path, expected_sha256, execution_profile=execution_profile
    )
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

    def budget():
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
        if tree_bytes(output) > caps["max_total_bytes"]:
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
                logits_mode="all" if plan["objective"] == "exact_soft" else "none",
                chain_ancestry=ancestry,
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
            tokens, length = receipt["tokens"], receipt["prompt_length"]
            anchors = list(range(length - 1, len(tokens) - 7, 7))
            if not anchors:
                raise ValueError("native generated chain has no complete seven-slot author horizon")
            chains.append(
                {k: r[k] for k in ("prompt_id", "prompt_sha256", "domain", "split")}
                | {
                    "chain_id": f"chain-{len(chains):06d}",
                    "prompt_length": length,
                    "anchors": anchors,
                    "include_logits": plan["objective"] == "exact_soft",
                    "native_receipt": receipt_pin,
                }
            )
            report["completed_prompt_count"] = len(chains)
            budget()
            if r["domain"] in {g["domain"] for g in goldens}:
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
                    "generation_receipt": receipt_pin,
                    "native_receipt": golden_pin,
                }
            )
            budget()
        runtime_pin = cost["runtime"]
        manifests = {}
        for family in ("dspark", "dflash"):
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
            )
            manifests[family] = {"path": str(manifest), "sha256": file_sha256(manifest)}
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
    if mode not in ("none", "last", "all") or set(receipt.get("files", {})) != wanted:
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
    mode = "all" if plan["objective"] == "exact_soft" else "none"
    if (
        receipt.get("features_shape") != [len(tokens), 5, plan["target_width"]]
        or receipt.get("logits_mode") != mode
        or receipt.get("logits_shape") != [len(tokens) if mode == "all" else 0, plan["vocab_size"]]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--development-cpu",
        action="store_true",
        help="Explicit local Mac CPU data development; never CUDA readiness",
    )
    args = parser.parse_args()
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
