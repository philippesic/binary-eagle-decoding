#!/usr/bin/env python3
"""Supervised native head capture/factorial runner; heavy diagnostics, never timing.

Run under scripts/remote_job.py. Factorial records Q4_0 canonical rounds before
replaying q4_q4,d_d,q4_d,d_q4,d_f16. Train captures unchanged D own histories.
Diagnostic permits up to three historical prompts and the same five-arm gate.
Raw captures retain every state; prepare_frozen_d_head_data.py selects at most
17 states per prompt/depth (8,160 total) without opening development data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from analyze_binary_head_factorial import ARMS, analyze, audit_rows
from capture_w1ax_activations import (
    generated_token_ids,
    handle_stop_signal,
    http_json,
    load_prompts,
    read_jsonl,
    sha256,
    stop_server,
    wait_ready,
)
from run_binary_scale_screen import quality

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PROMPTS_SHA256 = "80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74"
TARGET_F16_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
DRAFT_D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"
# SHA256 of the canonical little-endian I64 contents of candidate D's d2t tensor.
DRAFT_D_D2T_SHA256 = "03d2f0e3420175955c14a43a1c1dda360cde26c8a03318ffb1a013bb06d0fe0a"
TARGET_VOCAB_SIZE = 151_936
FEATURE_WIDTH = 7680
DEFAULT_TARGET_LOGITS_LIMIT = 32
DEFAULT_TARGET_FEATURES_LIMIT = 32_768


def _frozen_shard(args, prompts: list[dict]) -> dict | None:
    """Return a new train shard contract, or None for the inherited 96 prompts."""
    shard_path = getattr(args, "shard_manifest", None)
    expected_hash = getattr(args, "expected_prompt_sha256", None)
    expected_count = getattr(args, "expected_prompt_count", None)
    if shard_path is None and expected_hash is None and expected_count is None:
        return None
    if args.mode != "recurrent-train" or any(
        value is None for value in (shard_path, expected_hash, expected_count)
    ):
        raise ValueError("recurrent shard requires manifest, expected SHA256 and count")
    if getattr(args, "prompt_manifest", None) is not None or getattr(args, "prompts_sha256", None):
        raise ValueError("shard and legacy prompt manifest/hash options cannot be combined")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
        or type(expected_count) is not int
        or expected_count < 1
    ):
        raise ValueError("invalid expected shard prompt SHA256/count")
    shard_path = Path(shard_path)
    shard = json.loads(shard_path.read_text())
    ids = [prompt.get("id") for prompt in prompts]
    if (
        not isinstance(shard, dict)
        or shard.get("schema") != "w1ax_capture_shard_v1"
        or shard.get("prompts_path") != args.prompts.name
        or (shard_path.resolve().parent / shard["prompts_path"]) != args.prompts.resolve()
        or shard.get("prompts_sha256") != expected_hash
        or sha256(args.prompts) != expected_hash
        or shard.get("prompt_count") != expected_count
        or len(ids) != expected_count
        or len(set(ids)) != expected_count
        or shard.get("prompt_ids") != ids
        or not isinstance(shard.get("parent"), dict)
        or shard["parent"].get("split") not in ("train_small", "train_large")
    ):
        raise ValueError("shard manifest/hash/count/ordered prompt IDs disagree")
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
    caps = shard.get("caps")
    row_bytes = args.target_vocab_size * 4 if args.target_vocab_size else 0
    if (
        shard.get("target_vocab_size") != args.target_vocab_size
        or shard.get("bytes_per_raw_logit_row") != row_bytes
        or not isinstance(caps, dict)
        or type(caps.get("max_verifier_logit_rows")) is not int
        or caps["max_verifier_logit_rows"] < 1
        or type(caps.get("max_prompts")) is not int
        or expected_count > caps["max_prompts"]
        or type(caps.get("max_raw_logit_bytes")) is not int
        or caps["max_raw_logit_bytes"] < row_bytes
        or args.target_logits_limit != caps["max_verifier_logit_rows"]
    ):
        raise ValueError("shard raw-logit cap/vocabulary differs from capture settings")
    return shard


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_inputs(args, prompts: list[dict], variants: dict) -> None:
    if (getattr(args, "activation_bits", 16) != 16 or getattr(args, "label_only", False)):
        raise ValueError("new arithmetic/label-only capture uses the typed continuous stage API; "
                         "legacy frozen capture CLI cannot change its contract")
    if not 1 <= args.tokens <= 128:
        raise ValueError("capture output token cap must be 1..128")
    if args.mode != "diagnostic" and args.tokens != 128:
        raise ValueError("frozen primary capture requires128 output cap")
    if "final" in str(args.prompts).lower() or any("final" in p["id"].lower() for p in prompts):
        raise ValueError("reserved-final prompts prohibited")
    if sys.byteorder != "little":
        raise ValueError("training preparation currently requires little-endian native captures")
    shard = _frozen_shard(args, prompts)
    if args.mode == "diagnostic":
        if len(prompts) > 3 or any(p["id"].startswith("qat-") for p in prompts):
            raise ValueError("diagnostic requires at most3 historical prompts")
    elif shard is None:
        split, expected = (
            ("train", 96) if args.mode in ("train", "recurrent-train") else ("development", 24)
        )
        if len(prompts) != expected or any(
            not p["id"].startswith(f"qat-revisit-{split}-") for p in prompts
        ):
            raise ValueError(f"expected exactly{expected} frozen {split} prompts")
        if args.prompts_sha256:
            if sha256(args.prompts) != args.prompts_sha256:
                raise ValueError("frozen prompt hash/count mismatch")
        else:
            frozen = args.prompt_manifest or args.prompts.parent / "manifest.json"
            spec = json.loads(frozen.read_text())["splits"][split]
            if spec["sha256"] != sha256(args.prompts) or spec["prompts"] != expected:
                raise ValueError("frozen prompt manifest hash/count mismatch")
    required = {"d_d"} if args.mode in ("train", "recurrent-train") else set(ARMS)
    if set(variants) != required:
        raise ValueError(f"expected variants {sorted(required)}")
    for name, spec in variants.items():
        if (spec.get("body"), spec.get("head")) != ARMS[name] or not spec.get("draft"):
            raise ValueError(f"incorrect body/head identity for {name}")
        for key in spec.get("env", {}):
            if args.mode == "recurrent-train" and key.startswith(
                ("EAGLE_", "W1AX_", "GGML_", "CUDA_", "LLAMA_ARG_")
            ):
                raise ValueError(f"variant cannot override recurrent capture policy {key}")
            if key in ("EAGLE_CAPTURE_FULL_LOGITS", "EAGLE_CAPTURE_FULL_LOGITS_LIMIT"):
                continue
            if key.startswith(("EAGLE_", "W1AX_")) or key in (
                "CUDA_VISIBLE_DEVICES",
                "GGML_CUDA_DISABLE_GRAPHS",
            ):
                raise ValueError(f"variant cannot override capture control {key}")
    if args.mode in ("train", "recurrent-train") and (not args.d2t or not args.target_vocab_size):
        raise ValueError("train requires --d2t and --target-vocab-size")
    if args.mode == "recurrent-train":
        if shard is None and sha256(args.prompts) != TRAIN_PROMPTS_SHA256:
            raise ValueError("recurrent train requires the pinned frozen96 prompt SHA256")
        if (
            sha256(args.target) != TARGET_F16_SHA256
            or sha256(Path(variants["d_d"]["draft"])) != DRAFT_D_SHA256
        ):
            raise ValueError(
                "recurrent train requires the pinned FP16 target and candidate D draft"
            )
        if args.target_vocab_size != TARGET_VOCAB_SIZE:
            raise ValueError("recurrent train target vocabulary size differs from pinned model")
        mapping = np.load(args.d2t, allow_pickle=False)
        if mapping.dtype != np.dtype("<i8") or mapping.shape != (32_000,):
            raise ValueError("recurrent train requires candidate D's absolute I64 d2t map")
        import hashlib

        if hashlib.sha256(mapping.tobytes()).hexdigest() != DRAFT_D_D2T_SHA256:
            raise ValueError("recurrent train d2t map differs from candidate D")
        # These explicit caps cover the frozen 96 x 128-token run even at
        # five proposals per round; files still contain only actual rows.
        for value, name, maximum in (
            (args.target_logits_limit, "target logits", 65_536),
            (args.target_features_limit, "target features", 131_072),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"recurrent {name} limit must be in 1..{maximum}")


def server_command(args, spec):
    command = [
        str(args.binary.resolve()),
        "-m",
        str(args.target.resolve()),
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
        "--fit",
        "off",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--jinja",
        "--metrics",
        "--perf",
        "-lv",
        "4",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-md",
        str(Path(spec["draft"]).resolve()),
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        "5",
        "--spec-draft-p-min",
        "0",
        "--spec-draft-ngl",
        "all",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--no-spec-draft-backend-sampling",
    ]
    if args.mode == "recurrent-train":
        command.extend(
            (
                "--no-context-shift",
                "--no-cache-prompt",
                "--cache-reuse",
                "0",
                "--ctx-checkpoints",
                "0",
            )
        )
    return command


def audit_recurrent_files(
    cell: Path,
    manifest: dict,
    target_vocab_size: int,
    logit_limit: int,
    feature_limit: int,
    *,
    require_full_logits: bool = False,
    label_only: bool = False,
) -> None:
    """Reject incomplete raw streams before marking a recurrent cell complete."""
    heads = read_jsonl(cell / "heads.jsonl")
    events = read_jsonl(cell / "heads.target_features.jsonl")
    feature_path = cell / "heads.target_features.f32"
    logit_path = cell / "heads.target_logits.f32"
    if not heads or not events or not feature_path.is_file() or not logit_path.is_file():
        raise ValueError("missing recurrent target capture files")
    decoded = [event for event in events if event.get("event") == "decoded_row"]
    dispositions = [event for event in events if event.get("event") == "disposition"]
    if (
        len(decoded) < 1
        or len(decoded) > feature_limit
        or len(events) != 2 * len(decoded)
        or feature_path.stat().st_size != len(decoded) * FEATURE_WIDTH * 4
    ):
        raise ValueError("truncated or over-limit target feature capture")
    expected_rows = set(range(len(decoded)))
    if (
        {event.get("feature_row") for event in decoded} != expected_rows
        or {event.get("feature_row") for event in dispositions} != expected_rows
        or len(dispositions) != len(expected_rows)
        or any(str(event.get("task_id")) not in manifest["task_prompt_ids"] for event in events)
    ):
        raise ValueError("ambiguous target feature task/row join")
    indexes = [row.get("target_logits_row") for row in heads]
    count = 0 if label_only else min(len(heads), logit_limit)
    if require_full_logits and len(heads) > logit_limit:
        raise ValueError("shard verifier logits exceed frozen raw-logit cap")
    if indexes != [*range(count), *([None] * (len(heads) - count))]:
        raise ValueError("truncated or misindexed target verifier logits")
    if logit_path.stat().st_size != count * target_vocab_size * 4:
        raise ValueError("truncated target verifier logits file")
    rounds = read_jsonl(cell / "forced-rounds.jsonl")
    head_cursor = round_cursor = event_cursor = feature_cursor = logit_cursor = 0
    for request in manifest["requests"]:
        task = request["task_id"]
        head_start, head_end = request["capture_rows"]
        round_start, round_end = request["forced_round_rows"]
        event_start, event_end = request["target_feature_event_rows"]
        feature_start, feature_end = request["target_feature_rows"]
        logit_start, logit_end = request["target_logit_rows"]
        if (
            head_start != head_cursor
            or round_start != round_cursor
            or event_start != event_cursor
            or feature_start != feature_cursor
            or logit_start != logit_cursor
            or not head_start < head_end <= len(heads)
            or not round_start < round_end <= len(rounds)
            or not event_start < event_end <= len(events)
            or not feature_start < feature_end <= len(decoded)
            or not logit_start <= logit_end <= count
            or any(str(row.get("task_id")) != task for row in heads[head_start:head_end])
            or any(str(row.get("task_id")) != task for row in rounds[round_start:round_end])
            or any(str(event.get("task_id")) != task for event in events[event_start:event_end])
            or {
                event.get("feature_row")
                for event in events[event_start:event_end]
                if event.get("event") == "decoded_row"
            }
            != set(range(feature_start, feature_end))
            or [index for index in indexes[head_start:head_end] if index is not None]
            != list(range(logit_start, logit_end))
        ):
            raise ValueError("ambiguous recurrent request/task row ranges")
        head_cursor, round_cursor, event_cursor = head_end, round_end, event_end
        feature_cursor, logit_cursor = feature_end, logit_end
    if (head_cursor, round_cursor, event_cursor, feature_cursor, logit_cursor) != (
        len(heads),
        len(rounds),
        len(events),
        len(decoded),
        count,
    ):
        raise ValueError("unclaimed recurrent capture rows")



def verify_mapped_runtime(pid: int, runtime: dict) -> dict:
    expected = {str(Path(record["path"]).resolve()): record["sha256"]
                for record in runtime["libraries"]}
    maps = Path(f"/proc/{pid}/maps").read_text()
    loaded = set()
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        raw = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), parts[5])
        if Path(raw).name.startswith(("libllama", "libggml")):
            if raw.endswith(" (deleted)"):
                raise ValueError("native server mapped a deleted/replaced runtime library")
            loaded.add(str(Path(raw).resolve()))
    if not loaded or not any(Path(p).name.startswith("libllama") for p in loaded):
        raise ValueError("native server mapped no declared llama libraries")
    if not any(Path(p).name.startswith("libggml-cuda") for p in loaded):
        raise ValueError("native server mapped no declared CUDA backend library")
    for path in loaded:
        if path not in expected or sha256(Path(path)) != expected[path]:
            raise ValueError("native server mapped an unexpected or changed llama/ggml library")
    return {"source": f"/proc/{pid}/maps", "mapped_libraries":
            [{"path": path, "sha256": expected[path]} for path in sorted(loaded)]}

def run_cell(args, name, spec, prompts, forced=None):
    if args.mode == "recurrent-train" and (name != "d_d" or forced is not None):
        raise ValueError("recurrent train requires unforced candidate D own histories")
    stop_file = getattr(args, "stop_file", None)
    if stop_file is not None and Path(stop_file).exists():
        raise InterruptedError("user STOP requested before native capture")
    cell = args.output.resolve() / name
    cell.mkdir()
    trace = cell / "rounds.jsonl"
    rows_path = cell / "heads.jsonl"
    rounds_path = cell / "forced-rounds.jsonl"
    recurrent = args.mode == "recurrent-train"
    features_metadata = cell / "heads.target_features.jsonl"
    features_values = cell / "heads.target_features.f32"
    target_logits = cell / "heads.target_logits.f32"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GGML_W1", "GGML_EAGLE", "W1AX_", "EAGLE_"))
        and (not recurrent or not k.startswith(("GGML_", "CUDA_", "LLAMA_ARG_")))
    }
    env.update(spec.get("env", {}))
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "EAGLE_CAPTURE_PREFIX": str(cell / "heads"),
            "EAGLE_RECORD_ROUNDS_JSONL": str(rounds_path),
            "EAGLE_STATE_TRACE_JSONL": str(cell / "state.jsonl"),
            "EAGLE_REQUEST_DIGEST": "1",
            "W1AX_ROUND_TRACE_JSONL": str(trace),
        }
    )
    if forced:
        env["EAGLE_FORCE_ROUNDS_JSONL"] = str(forced)
    if name == "d_f16":
        env.setdefault("EAGLE_CAPTURE_FULL_LOGITS", "1")
        env.setdefault("EAGLE_CAPTURE_FULL_LOGITS_LIMIT", "32")
    if recurrent:
        bits = getattr(args, "activation_bits", 16)
        if bits not in (1, 8, 16):
            raise ValueError("capture activation contract must be A1, A8 or frozen A16")
        env.update(
            {
                "GGML_W1AX_ACT_BITS": str(bits),
                "EAGLE_CAPTURE_TARGET_LOGITS": "1",
                "EAGLE_CAPTURE_TARGET_LOGITS_LIMIT": str(args.target_logits_limit),
                "EAGLE_CAPTURE_TARGET_FEATURES": "1",
                "EAGLE_CAPTURE_TARGET_FEATURES_LIMIT": str(args.target_features_limit),
            }
        )
    if recurrent and getattr(args, "label_only", False):
        env.pop("EAGLE_CAPTURE_TARGET_LOGITS", None)
        env.pop("EAGLE_CAPTURE_TARGET_LOGITS_LIMIT", None)
        target_logits.touch()  # Empty compatibility sentinel, never teacher logits.
    cmd = server_command(args, spec)
    manifest = {
        "schema": "binary_head_capture_cell_v1",
        "name": name,
        "spec": spec,
        "command": cmd,
        "env": {
            k: v for k, v in env.items()
            if k.startswith(("GGML_", "EAGLE_", "W1AX_", "CUDA_")) or k == "LD_LIBRARY_PATH"
        },
        "binary_sha256": sha256(args.binary),
        "target_sha256": sha256(args.target),
        "draft_sha256": sha256(Path(spec["draft"])),
        "prompts_sha256": sha256(args.prompts),
        "prompt_count": len(prompts),
        "ordered_prompt_ids": [prompt["id"] for prompt in prompts],
        "shard_manifest_sha256": (
            sha256(Path(args.shard_manifest))
            if getattr(args, "shard_manifest", None) is not None
            else None
        ),
        "task_prompt_ids": {},
        "requests": [],
        "kind": "instrumented_quality_capture_not_timing",
        "complete": False,
    }
    write(cell / "manifest.json", manifest)
    with (cell / "server.log").open("wb") as log:
        proc = None
        guard = getattr(args, "cancellation_guard", None)
        critical = guard.defer if guard is not None else nullcontext
        try:
            with critical():
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True
                )
                manifest["server_pid"] = proc.pid
                manifest["server_pgid"] = os.getpgid(proc.pid)
                manifest["owner_pid"] = os.getpid()
                write(cell / "manifest.json", manifest)
            url = f"http://127.0.0.1:{args.port}"
            deadline = getattr(args, "deadline", None)
            startup_budget = min(120, max(1, deadline - time.monotonic())) if deadline else 300
            wait_ready(url, proc, startup_budget)
            if getattr(args, "native_runtime", None) is not None:
                manifest["native_runtime"] = args.native_runtime
                manifest["mapped_runtime"] = verify_mapped_runtime(proc.pid, args.native_runtime)
                write(cell / "manifest.json", manifest)
            for index, prompt in enumerate(prompts):
                if stop_file is not None and Path(stop_file).exists():
                    raise InterruptedError("user STOP requested during native capture")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("native development aggregate deadline exceeded")
                before = len(read_jsonl(rows_path))
                rounds_before = len(read_jsonl(rounds_path))
                trace_before = len(read_jsonl(trace))
                if recurrent:
                    event_before = len(read_jsonl(features_metadata))
                    feature_before = sum(
                        event.get("event") == "decoded_row"
                        for event in read_jsonl(features_metadata)
                    )
                    logit_before = (
                        target_logits.stat().st_size // (args.target_vocab_size * 4)
                        if target_logits.exists()
                        else 0
                    )
                body = {
                    "messages": prompt["messages"],
                    "max_tokens": args.tokens,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_format": "none",
                    "return_tokens": True,
                    "verbose": True,
                }
                request = cell / f"request-{index:03d}"
                request.mkdir()
                write(request / "prompt.json", prompt)
                write(request / "request.json", body)
                timeout = getattr(args, "request_timeout", 600)
                if deadline is not None:
                    timeout = min(timeout, max(1, deadline - time.monotonic()))
                response = http_json(url + "/v1/chat/completions", body, timeout)
                write(request / "response.json", response)
                ids = generated_token_ids(response)
                if not ids:
                    raise ValueError("missing raw output token IDs")
                rows = read_jsonl(rows_path)[before:]
                rounds = read_jsonl(rounds_path)[rounds_before:]
                traces = read_jsonl(trace)[trace_before:]
                if recurrent:
                    feature_events = read_jsonl(features_metadata)[event_before:]
                    feature_count = sum(
                        event.get("event") == "decoded_row" for event in feature_events
                    )
                    logit_after = (
                        target_logits.stat().st_size // (args.target_vocab_size * 4)
                        if target_logits.exists()
                        else 0
                    )
                    if (
                        not feature_events
                        or not feature_count
                        or not features_values.is_file()
                        or not target_logits.is_file()
                        or logit_after < logit_before
                    ):
                        raise ValueError("missing recurrent target capture for request")
                if not rows or not rounds or not traces:
                    raise ValueError("missing request head states or native rounds")
                task_ids = {str(r["task_id"]) for r in rows}
                if len(task_ids) != 1 or task_ids != {str(r["task_id"]) for r in rounds}:
                    raise ValueError("ambiguous request-to-task join")
                task_id = next(iter(task_ids))
                if task_id in manifest["task_prompt_ids"]:
                    raise ValueError("native task ID reused across requests")
                if recurrent and any(
                    str(event.get("task_id")) != task_id for event in feature_events
                ):
                    raise ValueError("ambiguous target feature request-to-task join")
                manifest["task_prompt_ids"][task_id] = prompt["id"]
                if any(r["forced"] != bool(forced) for r in rows):
                    raise ValueError("wrong native forced/own-history state")
                write(request / "rounds.json", traces)
                item = {
                    "id": prompt["id"],
                    "task_id": task_id,
                    "generated_token_ids": ids,
                    "capture_rows": [before, before + len(rows)],
                    "forced_round_rows": [rounds_before, rounds_before + len(rounds)],
                    "request_sha256": sha256(request / "request.json"),
                    "prompt_sha256": sha256(request / "prompt.json"),
                    "response_sha256": sha256(request / "response.json"),
                    "quality": quality(traces),
                }
                if recurrent:
                    item.update(
                        {
                            "target_feature_event_rows": [
                                event_before,
                                event_before + len(feature_events),
                            ],
                            "target_feature_rows": [feature_before, feature_before + feature_count],
                            "target_logit_rows": [logit_before, logit_after],
                        }
                    )
                manifest["requests"].append(item)
                write(cell / "manifest.json", manifest)
                progress_file = getattr(args, "progress_file", None)
                if progress_file is not None:
                    progress_path = Path(progress_file)
                    status = json.loads(progress_path.read_text())
                    status.update(heartbeat_unix=time.time(),
                                  native_capture_prompt=index + 1,
                                  native_capture_prompts=len(prompts),
                                  native_capture_prompt_id=prompt["id"],
                                  native_server_pid=proc.pid,
                                  native_server_pgid=manifest["server_pgid"],
                                  disk_free_bytes=shutil.disk_usage(args.output).free)
                    temporary = progress_path.with_name(progress_path.name + ".native.tmp")
                    write(temporary, status)
                    os.replace(temporary, progress_path)
                progress = f"{name} {index + 1}/{len(prompts)} {prompt['id']} rows={len(rows)}"
                print(progress, flush=True)
        finally:
            if proc is not None:
                manifest["server_stop"] = stop_server(proc)
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    manifest["server_stop"]["process_group_gone"] = True
                else:
                    manifest["server_stop"]["process_group_gone"] = False
                write(cell / "manifest.json", manifest)
    if (not manifest["server_stop"].get("stopped")
            or not manifest["server_stop"].get("process_group_gone")):
        raise RuntimeError("server process group did not stop")
    for marker in spec.get("required_markers", []):
        if marker not in (cell / "server.log").read_text(errors="replace"):
            raise ValueError(f"missing runtime marker: {marker}")
    manifest["audit"] = audit_rows(
        read_jsonl(rows_path), cell / "heads.f32", manifest["task_prompt_ids"]
    )
    if recurrent:
        audit_recurrent_files(
            cell,
            manifest,
            args.target_vocab_size,
            args.target_logits_limit,
            args.target_features_limit,
            require_full_logits=getattr(args, "shard_manifest", None) is not None,
            label_only=getattr(args, "label_only", False),
        )
    manifest["files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in (
            (
                rows_path,
                cell / "heads.f32",
                rounds_path,
                trace,
                cell / "server.log",
                features_metadata,
                features_values,
                target_logits,
            )
            if recurrent
            else (rows_path, cell / "heads.f32", rounds_path, trace, cell / "server.log")
        )
    }
    manifest["complete"] = True
    write(cell / "manifest.json", manifest)
    return manifest


def run(args):
    if args.mode == "recurrent-train" and "final" in str(args.prompts).lower():
        raise ValueError("reserved-final prompts prohibited")
    prompts = load_prompts(args.prompts)
    variants = json.loads(args.variants.read_text())
    validate_inputs(args, prompts, variants)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "config.json",
        {
            "mode": args.mode,
            "tokens": args.tokens,
            "activation_bits": getattr(args, "activation_bits", 16),
            "label_only": getattr(args, "label_only", False),
            "variants": variants,
            "prompts_sha256": sha256(args.prompts),
            "prompt_count": len(prompts),
            "shard_manifest_sha256": (
                sha256(Path(args.shard_manifest))
                if getattr(args, "shard_manifest", None) is not None
                else None
            ),
        },
    )
    if args.mode in ("train", "recurrent-train"):
        cell = run_cell(args, "d_d", variants["d_d"], prompts)
        if args.mode == "recurrent-train":
            task_map = args.output / "task_prompt_ids.json"
            write(task_map, cell["task_prompt_ids"])
            write(
                args.output / "capture-manifest.json",
                {
                    "schema": "recurrent_binary_native_capture_v1",
                    "split": "train",
                    "trajectory": "own_history",
                    "body": "D",
                    "head": "D",
                    "train_prompts_sha256": sha256(args.prompts),
                    "train_prompt_count": len(prompts),
                    "shard_manifest_sha256": (
                        sha256(Path(args.shard_manifest))
                        if getattr(args, "shard_manifest", None) is not None
                        else None
                    ),
                    "task_prompt_ids_path": task_map.name,
                    "task_prompt_ids_sha256": sha256(task_map),
                    "cell_manifest_path": "d_d/manifest.json",
                    "cell_manifest_sha256": sha256(args.output / "d_d" / "manifest.json"),
                    "binary_sha256": cell["binary_sha256"],
                    "target_sha256": cell["target_sha256"],
                    "draft_sha256": cell["draft_sha256"],
                    "d2t_path": str(args.d2t.resolve()),
                    "d2t_sha256": sha256(args.d2t),
                    "d2t_raw_sha256": DRAFT_D_D2T_SHA256,
                    "target_vocab_size": args.target_vocab_size,
                    "target_logits_limit": args.target_logits_limit,
                    "target_features_limit": args.target_features_limit,
                    "raw_files": {
                        name: {"path": "d_d/" + name, **cell["files"][name]}
                        for name in (
                            "heads.jsonl",
                            "heads.f32",
                            "forced-rounds.jsonl",
                            "heads.target_logits.f32",
                            "heads.target_features.jsonl",
                            "heads.target_features.f32",
                        )
                    },
                    "requests": cell["requests"],
                    "complete": True,
                    "kind": "instrumented_quality_capture_not_timing",
                },
            )
            return
        write(
            args.output / "capture-manifest.json",
            {
                "split": "train",
                "body": "D",
                "trajectory": "own_history",
                "train_prompts_sha256": sha256(args.prompts),
                "prompt_ids": [p["id"] for p in prompts],
                "d2t_path": str(args.d2t.resolve()),
                "d2t_sha256": sha256(args.d2t),
                "target_vocab_size": args.target_vocab_size,
                "selection_budget": (
                    "prepare:17 per prompt/depth, at most8160; raw capture unfiltered"
                ),
                "captures": [
                    {
                        "states_path": "d_d/heads.f32",
                        "rows_path": "d_d/heads.jsonl",
                        "state_dim": cell["audit"]["state_dim"],
                        "task_prompt_ids": cell["task_prompt_ids"],
                    }
                ],
            },
        )
    else:
        run_cell(args, "canonical", variants["q4_q4"], prompts)
        for name in ARMS:
            run_cell(
                args,
                name,
                variants[name],
                prompts,
                args.output / "canonical" / "forced-rounds.jsonl",
            )
        write(args.output / "factorial-analysis.json", analyze(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("factorial", "train", "diagnostic", "recurrent-train"), required=True
    )
    for name in ("binary", "target", "prompts", "variants", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--expected-prompt-count", type=int)
    parser.add_argument(
        "--prompts-sha256", help="frozen split hash, alternative to prompt manifest"
    )
    parser.add_argument("--d2t", type=Path)
    parser.add_argument("--target-vocab-size", type=int)
    parser.add_argument("--activation-bits", type=int, choices=(1, 8, 16), default=16)
    parser.add_argument("--label-only", action="store_true",
                        help="native hard-CE labels without target logit payloads")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--target-logits-limit", type=int, default=DEFAULT_TARGET_LOGITS_LIMIT)
    parser.add_argument("--target-features-limit", type=int, default=DEFAULT_TARGET_FEATURES_LIMIT)
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handle_stop_signal)
    run(args)


if __name__ == "__main__":
    main()
