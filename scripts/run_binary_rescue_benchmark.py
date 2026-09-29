#!/usr/bin/env python3
"""Graph-enabled rescue/head benchmark. Run only under the coordinated GPU owner.

CLI: --config CONFIG --mode quality|timed|instrumented --output NEW_DIRECTORY.
Paths in CONFIG are relative to the invocation directory. See the JSON schema.
Primary: sealed 24 development prompts, D5/p0, 128 tokens, context 2048, F16 KV.
Timed mode has >=5 rotating/reversed blocks, >=2 warmups per server launch.
Quality/instrumented modes collect round traces and are NOT performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import benchmark_native_eagle as base
import benchmark_w1ax_streaming as streaming
from capture_w1ax_activations import handle_stop_signal

ROOT = Path(__file__).resolve().parents[1]
DEV_SHA256 = "a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885"
TARGET_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
PRIMARY = {"tokens": 128, "draft_length": 5, "p_min": 0, "context": 2048}
# Clear ambient experimental switches; variants explicitly own their dispatch.
CLEAR_PREFIXES = ("GGML_W1", "GGML_EAGLE", "W1AX_", "EAGLE_", "GGML_CUDA_")
HEAVY = (
    "CAPTURE",
    "DUMP",
    "STATE_TRACE",
    "ROUND_TRACE",
    "DRAFT_STAGE",
    "MATMUL_AUDIT",
    "PROFILE",
    "EVENTS",
    "REPLAY",
)
write = streaming.write_json


def validate_config(config: dict, diagnostic: bool = False) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if "draft_backend_sampling" in config and type(config["draft_backend_sampling"]) is not bool:
        raise ValueError("draft_backend_sampling must be boolean")
    variants = config.get("variants", {})
    if not variants or config.get("q4_variant", "q4_0") not in variants:
        raise ValueError("variants must include the named Q4_0 primary control")
    for name, spec in variants.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError("unsafe variant name")
        for key, value in spec.get("env", {}).items():
            if not isinstance(value, str) or not isinstance(key, str):
                raise ValueError("environment keys and values must be strings")
            if key == "GGML_CUDA_DISABLE_GRAPHS":
                raise ValueError(
                    "graphs must remain enabled; use a separately labeled external control"
                )
            if any(x in key for x in HEAVY):
                raise ValueError("put diagnostic instrumentation in instrumented_env")
    if config.get("warmups", 2) < 2 or config.get("repetitions", 5) < 5:
        raise ValueError("require >=2 warmups and >=5 repetitions")
    policy = dict(PRIMARY)
    if diagnostic:
        workload = config.get("diagnostic")
        if not workload or not workload.get("label") or not workload.get("prompts_sha256"):
            raise ValueError("diagnostic needs a frozen label and prompt hash")
        policy.update(workload.get("policy", {}))
        if policy["draft_length"] not in (1, 3, 5) or policy["p_min"] != 0:
            raise ValueError("bounded depth diagnostic permits D={1,3,5}, p_min=0 only")
        if policy["tokens"] not in (128, 256) or policy["context"] not in (2048, 4096):
            raise ValueError("bounded diagnostic permits tokens={128,256}, context={2048,4096}")
    elif config.get("policy", PRIMARY) != PRIMARY:
        raise ValueError("primary policy is frozen")
    for section in ("graph_env", "instrumented_env"):
        for key, value in config.get(section, {}).items():
            if not isinstance(value, str):
                raise ValueError("environment values must be strings")
            if key == "GGML_CUDA_DISABLE_GRAPHS":
                raise ValueError("graphs must remain enabled in every mode")
            if section == "graph_env" and any(x in key for x in HEAVY):
                raise ValueError("heavy instrumentation prohibited in graph_env")
    if "EAGLE_STATE_TRACE_JSONL" in config.get("instrumented_env", {}) and not (
        diagnostic and config["diagnostic"].get("no_spec_draft_backend_sampling") is True
    ):
        raise ValueError("EAGLE state trace requires diagnostic backend sampling disabled")
    return policy


def orders(names: list[str], repetitions: int) -> list[list[str]]:
    """Rotated alternating orders; exact position balance needs a multiple of 2N."""
    result = []
    for i in range(repetitions):
        shift = (i // 2) % len(names)
        row = names[shift:] + names[:shift]
        result.append(row if i % 2 == 0 else row[::-1])
    return result


def round_quality(rows: list[dict]) -> dict:
    rows = [row for row in rows if row.get("status") != "checkpoint_replay"]
    for row in rows:
        p, a = row["n_proposed"], row["n_accepted"]
        if not (isinstance(p, int) and isinstance(a, int) and 0 <= a <= p <= 5):
            raise ValueError("invalid draft counts")
        if len(row["proposed_token_ids"]) != p:
            raise ValueError("proposal IDs/count mismatch")
    n = len(rows)
    proposed = sum(r["n_proposed"] for r in rows)
    accepted = sum(r["n_accepted"] for r in rows)
    depth = {}
    for d in range(1, 6):
        eligible = sum(r["n_proposed"] >= d for r in rows)
        survived = sum(r["n_accepted"] >= d for r in rows)
        # Later-token conditional denominator excludes early rejects and short proposals.
        reached = sum(r["n_proposed"] >= d and r["n_accepted"] >= d - 1 for r in rows)
        depth[str(d)] = {
            "eligible": eligible,
            "survived": survived,
            "reached": reached,
            "prefix_survival": survived / eligible if eligible else None,
            "conditional_acceptance": survived / reached if reached else None,
        }
    emitted = sum(r.get("n_emitted", 0) for r in rows)
    return {
        "rounds": n,
        "proposed": proposed,
        "accepted": accepted,
        "emitted": emitted,
        "accepted_fraction": accepted / proposed if proposed else None,
        "accepted_per_round": accepted / n if n else None,
        "emitted_per_round": emitted / n if n else None,
        "zero_accept_rounds": sum(r["n_accepted"] == 0 for r in rows),
        "no_proposal_rounds": sum(r["n_proposed"] == 0 for r in rows),
        "proposal_lengths": dict(sorted(Counter(r["n_proposed"] for r in rows).items())),
        "depth": depth,
    }


class StreamParser:
    """Native chat SSE: per-event tokens, final __verbose.tokens is cumulative.

    Also accepts explicit final generated_token_ids as a cumulative raw-ID field.
    Arrival intervals reflect HTTP/SSE delivery, not internal CUDA token timing.
    """

    def __init__(self):
        self.lines = []
        self.events = []
        self.done = False
        self.text = []
        self.token_ids = []
        self.final_ids = None
        self.arrivals = []
        self.content_arrivals = []
        self.timings = {}
        self.usage = {}
        self.finish_reason = None

    def feed(self, line: bytes, at: float):
        if self.done and line.strip():
            raise ValueError("SSE data after DONE")
        line = line.rstrip(b"\r\n")
        if line.startswith(b"data:"):
            self.lines.append(line[5:].lstrip(b" "))
        elif not line and self.lines:
            payload = b"\n".join(self.lines)
            self.lines = []
            if payload == b"[DONE]":
                self.done = True
                return
            event = json.loads(payload)
            if not isinstance(event, dict) or event.get("error"):
                raise ValueError(f"invalid/error SSE event: {event}")
            self.events.append({"received_s": at, "event": event})
            self.timings.update(event.get("timings") or {})
            self.usage.update(event.get("usage") or {})
            final = event.get("generated_token_ids")
            verbose = event.get("__verbose") or {}
            if final is None:
                final = verbose.get("tokens")
            if final is not None:
                if not isinstance(final, list) or any(type(t) is not int for t in final):
                    raise ValueError("invalid cumulative raw token IDs")
                self.final_ids = final
            else:
                ids = base.generated_token_ids(event)
                if ids:
                    self.token_ids.extend(ids)
                    self.arrivals.append({"received_s": at, "token_count": len(ids)})
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                text = streaming.content_text(delta)
                if text:
                    self.text.append(text)
                    self.content_arrivals.append(at)
                self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def result(self, wall: float) -> dict:
        if not self.done:
            raise ValueError("SSE response ended without DONE")
        ids = self.final_ids if self.final_ids is not None else self.token_ids
        if not ids:
            raise ValueError(
                "stream lacks raw IDs; runtime must expose per-event tokens or "
                "final generated_token_ids"
            )
        if (
            self.final_ids is not None
            and self.token_ids
            and self.token_ids != self.final_ids[: len(self.token_ids)]
        ):
            raise ValueError("stream chunk IDs disagree with final IDs")
        tokens = self.usage.get("completion_tokens", self.timings.get("predicted_n", len(ids)))
        if tokens != len(ids):
            raise ValueError(f"raw IDs/completion count mismatch: {len(ids)} != {tokens}")
        arrival_times = [r["received_s"] for r in self.arrivals]
        # Per-token intervals only when the entire stream has one ID per arrival.
        token_stream = (
            bool(self.arrivals)
            and len(self.token_ids) == len(ids)
            and all(r["token_count"] == 1 for r in self.arrivals)
        )
        times = arrival_times if arrival_times else self.content_arrivals
        if not times:
            raise ValueError("stream has no token or content arrivals")
        intervals = [b - a for a, b in zip(times, times[1:])]
        return {
            "request_wall_s": wall,
            "ttft_s": times[0],
            "ttft_boundary": (
                "before HTTP request to first complete SSE event carrying tokens, "
                "or nonempty content when per-event IDs absent"
            ),
            "wall_boundary": "before HTTP urlopen through HTTP EOF",
            "inter_token_s": intervals if token_stream else [],
            "inter_chunk_s": intervals,
            "token_arrivals": self.arrivals,
            "inter_token_status": "one_token_per_event"
            if token_stream
            else "unavailable; batched or absent event IDs",
            "completion_tokens": len(ids),
            "generated_token_ids": ids,
            "completion_sha256": hashlib.sha256("".join(self.text).encode()).hexdigest(),
            "server_predicted_ms": self.timings.get("predicted_ms"),
            "server_prompt_ms": self.timings.get("prompt_ms"),
            "server_timings": self.timings,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


def stream_request(url: str, body: dict, directory: Path, timeout: float = 600) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    parser = StreamParser()
    raw_lines = []
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("expected SSE response")
        while True:
            line = response.readline()
            at = time.perf_counter() - start
            if not line:
                break
            raw_lines.append(line)
            parser.feed(line, at)
            if at > timeout:
                raise TimeoutError("stream exceeded request timeout")
    end = time.perf_counter()
    wall = end - start
    # Disk persistence stays outside the measured request interval.
    (directory / "response.sse").write_bytes(b"".join(raw_lines))
    write(directory / "events.json", parser.events)
    result = parser.result(wall)
    result.update(
        {
            "client_request_begin_monotonic_s": start,
            "client_request_end_monotonic_s": end,
            "client_clock_implementation": time.get_clock_info("perf_counter").implementation,
        }
    )
    return result


GPU_FIELDS = (
    "timestamp",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
)


class GPUSampler:
    """One owned nvidia-smi process; whole-device sampled peaks are not allocator peaks."""

    def __init__(self, directory):
        self.directory = directory
        self.process = None
        self.loaded_snapshot = None
        self.summary = {}

    def __enter__(self):
        self.out = (self.directory / "gpu-telemetry.csv").open("wb")
        self.err = (self.directory / "gpu-telemetry.stderr.log").open("wb")
        executable = base.executable_path("nvidia-smi")
        self.command = [
            executable or "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_FIELDS),
            "--format=csv,noheader,nounits",
            "--loop=1",
        ]
        if executable:
            try:
                self.process = subprocess.Popen(
                    self.command, stdout=self.out, stderr=self.err, start_new_session=True
                )
            except BaseException:
                self.out.close()
                self.err.close()
                raise
        return self

    def __exit__(self, *_):
        exited = self.process.poll() if self.process else None
        try:
            if self.process:
                base.stop_server(self.process)
        finally:
            self.out.close()
            self.err.close()
        raw = (self.directory / "gpu-telemetry.csv").read_text(errors="replace")
        rows, rejected = streaming.gpu_csv_rows(raw, GPU_FIELDS)
        from statistics import median

        devices = {}
        for uuid in sorted({r["uuid"] for r in rows}):
            device = [r for r in rows if r["uuid"] == uuid]
            fields = {}
            for field in GPU_FIELDS[3:]:
                values = [
                    v for r in device if (v := streaming.numeric_gpu_value(r[field])) is not None
                ]
                fields[field] = {
                    "samples": len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "p50": median(values) if values else None,
                }
            devices[uuid] = {"name": device[0]["name"], "fields": fields}
        self.summary = {
            "status": "unavailable"
            if not self.process
            else "poller_exited_early"
            if exited is not None
            else "complete",
            "command": self.command,
            "sample_count": len(rows),
            "rejected_rows": rejected,
            "devices": devices,
            "loaded_snapshot": self.loaded_snapshot,
            "peak_semantics": streaming.GPU_PEAK_SEMANTICS,
            "timing_note": (
                "One-second whole-device polling spans startup, warmups, timing "
                "and shutdown; matched across variants."
            ),
        }
        write(self.directory / "gpu-telemetry-summary.json", self.summary)
        return False


def output_digest(token_ids):
    value = 14695981039346656037
    for token in token_ids:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def attach_digests(log, records):
    pattern = (
        r"eagle_request_digest task_id=(\d+) proposal=([0-9a-f]{16}) "
        r"output=([0-9a-f]{16}) rounds=(\d+)(?: no_proposal=(\d+))? output_tokens=(\d+)"
    )
    matches = re.findall(pattern, log)
    if len(matches) != len(records):
        raise ValueError(
            f"request digest mapping mismatch: {len(matches)} logs for {len(records)} requests"
        )
    task_ids = set()
    for ordinal, (fields, row) in enumerate(zip(matches, records)):
        task, proposal, output, rounds, no_proposal, tokens = fields
        if (
            task in task_ids
            or int(tokens) != len(row["generated_token_ids"])
            or output != output_digest(row["generated_token_ids"])
        ):
            raise ValueError("request digest raw-output validation failed")
        task_ids.add(task)
        row["request_digest"] = {
            "task_id": int(task),
            "proposal": proposal,
            "output": output,
            "rounds": int(rounds),
            "no_proposal": int(no_proposal) if no_proposal else None,
            "output_tokens": int(tokens),
            "completion_ordinal": ordinal,
            "mapping": (
                "sequential concurrency-one completion order; output FNV and token"
                " count independently verified"
            ),
            "semantics": (
                "FNV64 proposal digest covers full initial prompt, tagged no-proposal events, "
                "round/prefix length, seed, proposal IDs/count and accepted count; "
                "noncryptographic equality evidence"
            ),
        }


def graph_stats(log):
    rows = []
    for line in log.splitlines():
        if "CUDA_GRAPH_STATS " in line:
            payload = line.split("CUDA_GRAPH_STATS ", 1)[1]
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError:
                raise ValueError("malformed CUDA_GRAPH_STATS record")
    counters = (
        "calls",
        "launches",
        "captures",
        "recaptures",
        "direct_disabled",
        "direct_incompatible",
        "direct_warmup",
        "warmup_resets",
        "update_reinstantiations",
        "evictions",
        "w1ax_launches",
        "w1ax_captures",
    )
    for row in rows:
        if any(type(row.get(k)) is not int or row[k] < 0 for k in counters):
            raise ValueError("invalid graph statistics counters")
        if row["calls"] != sum(
            row[k] for k in ("launches", "direct_disabled", "direct_incompatible", "direct_warmup")
        ):
            raise ValueError("graph dispatch counts do not sum to calls")
        if (
            row["recaptures"] > row["captures"]
            or row["captures"] > row["launches"]
            or row["w1ax_launches"] > row["launches"]
        ):
            raise ValueError("inconsistent graph captures/launches")
    totals = {k: sum(r[k] for r in rows) for k in counters}
    return {
        "contexts": rows,
        "totals": totals,
        "verified_launches": bool(rows) and totals["launches"] > 0,
        "custom_w1ax_launches_verified": totals["w1ax_launches"] > 0,
        "scope": (
            "Per-server lifetime includes model setup and warmups; not timed-r"
            "equest-only. Captures include recaptures and launches include cap"
            "ture launches."
        ),
    }


def server_env(config: dict, spec: dict, mode: str, cell: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(CLEAR_PREFIXES)}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": config.get("cuda_visible_devices", "0"),
            "GGML_CUDA_GRAPH_STATS": "1",
            "EAGLE_REQUEST_DIGEST": "1",
        }
    )
    for extra in (spec.get("env", {}), config.get("graph_env", {})):
        env.update({k: v.replace("{output}", str(cell)) for k, v in extra.items()})
    if mode != "timed":
        env["W1AX_ROUND_TRACE_JSONL"] = str(cell / "rounds.jsonl")
    if mode == "instrumented":
        env.update(
            {
                k: v.replace("{output}", str(cell))
                for k, v in config.get("instrumented_env", {}).items()
            }
        )
    return env


def command(config: dict, spec: dict, policy: dict, *, diagnostic: bool = False) -> list[str]:
    cmd = [
        str(Path(config["binary"]).resolve()),
        "-m",
        str(Path(config["target"]).resolve()),
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        str(policy["context"]),
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
        "3",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.get("port", 18090)),
    ]
    if spec.get("draft"):
        cmd += [
            "-md",
            str(Path(spec["draft"]).resolve()),
            "--spec-type",
            "draft-eagle3",
            "--spec-draft-n-max",
            str(policy["draft_length"]),
            "--spec-draft-p-min",
            "0",
            "--spec-draft-ngl",
            "all",
            "--spec-draft-type-k",
            "f16",
            "--spec-draft-type-v",
            "f16",
        ]
        if config.get("draft_backend_sampling", True) is False or (
            diagnostic
            and config.get("diagnostic", {}).get("no_spec_draft_backend_sampling") is True
        ):
            cmd.append("--no-spec-draft-backend-sampling")
    else:
        cmd += ["--spec-type", "none"]
    return cmd


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def run(config: dict, mode: str, output: Path, diagnostic: bool = False) -> None:
    policy = validate_config(config, diagnostic)
    prompt_path = Path(config["diagnostic"]["prompts"] if diagnostic else config["prompts"])
    expected = config["diagnostic"]["prompts_sha256"] if diagnostic else DEV_SHA256
    if "final" in str(prompt_path).lower() or base.sha256(prompt_path) != expected:
        raise ValueError("prompt input does not match frozen non-final workload")
    prompts = base.load_prompts(prompt_path)
    if not diagnostic and len(prompts) != 24:
        raise ValueError("primary benchmark requires 24 prompts")
    if base.sha256(Path(config["target"])) != TARGET_SHA256:
        raise ValueError("target differs from the frozen FP16 target")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    (output / "prompts.jsonl").write_bytes(prompt_path.read_bytes())
    reps = config.get("repetitions", 5) if mode == "timed" else 1
    schedule = orders(list(config["variants"]), reps)
    manifest = {
        "schema": "binary_rescue_benchmark_v1",
        "mode": mode,
        "workload": config["diagnostic"]["label"] if diagnostic else "primary-development",
        "policy": policy,
        "schedule": schedule,
        "q4_variant": config.get("q4_variant", "q4_0"),
        "prompt_sha256": base.sha256(prompt_path),
        "environment": base.environment_manifest(Path(config["binary"])),
        "hashes": {
            "binary": base.sha256(Path(config["binary"])),
            "target": TARGET_SHA256,
            "drafts": {
                k: base.sha256(Path(v["draft"])) if v.get("draft") else None
                for k, v in config["variants"].items()
            },
        },
        "status": "running",
        "blocks": [],
        "records": [],
    }
    write(output / "manifest.json", manifest)
    url = f"http://127.0.0.1:{config.get('port', 18090)}"
    try:
        for rep, names in enumerate(schedule):
            for slot, name in enumerate(names):
                if not base.available_port("127.0.0.1", config.get("port", 18090)):
                    raise RuntimeError(
                        "benchmark port is occupied; refusing to query an unowned server"
                    )
                spec = config["variants"][name]
                block_records = []
                cell = output / f"r{rep:02d}-s{slot:02d}-{name}"
                cell.mkdir()
                env = server_env(config, spec, mode, cell)
                cmd = command(config, spec, policy, diagnostic=diagnostic)
                block = {
                    "variant": name,
                    "repetition": rep,
                    "order_slot": slot,
                    "directory": str(cell),
                    "command": cmd,
                    "env": {
                        k: v
                        for k, v in env.items()
                        if k.startswith((*CLEAR_PREFIXES, "CUDA_VISIBLE"))
                    },
                }
                manifest["blocks"].append(block)
                write(cell / "manifest.json", block)
                with GPUSampler(cell) as sampler, (cell / "server.log").open("wb") as log:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        block["server_pid"] = proc.pid
                        block["server_started_unix_ns"] = time.time_ns()
                        write(cell / "manifest.json", block)
                        base.wait_ready(proc, url, 300)
                        sampler.loaded_snapshot = base.gpu_snapshot()
                        # Reverse prompt traversal with each repetition; pairing uses stable IDs.
                        measured = prompts if rep % 2 == 0 else prompts[::-1]
                        requests = [
                            (True, prompts[i % len(prompts)])
                            for i in range(config.get("warmups", 2))
                        ]
                        requests += [(False, p) for p in measured]
                        for index, (warmup, prompt) in enumerate(requests):
                            request_dir = (
                                cell / f"{index:03d}-{'warmup-' if warmup else ''}{prompt['id']}"
                            )
                            request_dir.mkdir()
                            body = {
                                "messages": prompt["messages"],
                                "max_tokens": policy["tokens"],
                                "temperature": 0,
                                "seed": 42,
                                "stream": True,
                                "stream_options": {"include_usage": True},
                                "cache_prompt": False,
                                "chat_template_kwargs": {"enable_thinking": False},
                                "reasoning_format": "none",
                                "return_tokens": True,
                                "verbose": True,
                            }
                            write(request_dir / "request.json", body)
                            before_rounds = (
                                len(read_jsonl(cell / "rounds.jsonl")) if mode != "timed" else 0
                            )
                            before = base.metrics_text(url)
                            row = stream_request(url, body, request_dir)
                            after = base.metrics_text(url)
                            for tag, raw in (("before", before), ("after", after)):
                                if raw is not None:
                                    (request_dir / f"metrics-{tag}.prom").write_text(raw)
                            row.update(
                                {
                                    "variant": name,
                                    "repetition": rep,
                                    "order_slot": slot,
                                    "prompt_id": prompt["id"],
                                    "warmup": warmup,
                                    "speculative": base.counter_delta(before, after),
                                    "directory": str(request_dir),
                                }
                            )
                            if mode != "timed":
                                rounds = read_jsonl(cell / "rounds.jsonl")[before_rounds:]
                                if spec.get("draft") and not rounds:
                                    raise ValueError("speculative quality run lacks round trace")
                                write(request_dir / "rounds.json", rounds)
                                row["quality"] = round_quality(rounds)
                            write(request_dir / "record.json", row)
                            manifest["records"].append(row)
                            block_records.append(row)
                            write(output / "manifest.json", manifest)
                            print(
                                f"{mode} r{rep} {name} {'warmup' if warmup else prompt['id']} "
                                f"{row['request_wall_s']:.3f}s",
                                flush=True,
                            )
                    finally:
                        block["server_stop"] = base.stop_server(proc)
                        block["server_exit_code"] = proc.returncode
                        write(cell / "manifest.json", block)
                text = (cell / "server.log").read_text(errors="replace")
                attach_digests(text, block_records)
                for row in block_records:
                    write(Path(row["directory"]) / "record.json", row)
                markers = spec.get("required_markers", [])
                if any(marker not in text for marker in markers):
                    raise ValueError(f"missing required runtime marker for {name}")
                graph_markers = config.get("graph_required_markers", [])
                block["graph_marker_checks"] = {marker: marker in text for marker in graph_markers}
                block["graph_stats"] = graph_stats(text)
                block["graph_status"] = (
                    "verified_launches"
                    if block["graph_stats"]["verified_launches"]
                    else "unverified"
                )
                if graph_markers and not all(block["graph_marker_checks"].values()):
                    raise ValueError(f"missing graph execution evidence for {name}")
                block["gpu_telemetry"] = sampler.summary
                block["server_log_sha256"] = base.sha256(cell / "server.log")
                write(cell / "manifest.json", block)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        write(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("quality", "timed", "instrumented"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic", action="store_true", help="use separately frozen config.diagnostic workload"
    )
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, handle_stop_signal)
    run(json.loads(args.config.read_text()), args.mode, args.output, args.diagnostic)


if __name__ == "__main__":
    main()
