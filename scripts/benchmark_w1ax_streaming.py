#!/usr/bin/env python3
"""Separate eight-path W1Ax streaming latency diagnostic.

TTFT is measured from immediately before the HTTP request to receipt of the
first complete SSE event containing nonempty generated content. Full request
wall time ends at HTTP EOF. Neither number is nonstreaming decode throughput.
Run under scripts/remote_job.py on the RTX 2080 Ti; --dry-run uses no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import benchmark_native_eagle as base

ROOT = Path(__file__).resolve().parents[1]
PROMPT_IDS = (
    "context-prose-short",
    "context-code-medium",
    "context-reasoning-long",
)
CAPS = (32, 128)
WARMUPS = 2
REPETITIONS = 5
EXPECTED_CONTEXT_SHA256 = "5653cfe7599e5dd4ae44e057df27b816221f8ee89635056bc0fb24b9f44a21a3"
EXPECTED_TARGET_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
EXPECTED_W1AX_SHA256 = "098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c"


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def content_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


class SSEParser:
    """Parse complete SSE frames; timestamps refer to frame completion."""

    def __init__(self) -> None:
        self.data_lines: list[bytes] = []
        self.first_content_time: float | None = None
        self.text_parts: list[str] = []
        self.done = False
        self.finish_reason: str | None = None
        self.events = 0
        self.response_ids: set[str] = set()
        self.usage: dict[str, Any] | None = None

    def feed_line(self, line: bytes, received_at: float) -> None:
        if self.done and line.strip():
            raise ValueError("SSE data after [DONE]")
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            self._finish_event(received_at)
        elif stripped.startswith(b":"):
            return
        else:
            field, separator, value = stripped.partition(b":")
            if separator and value.startswith(b" "):
                value = value[1:]
            if field == b"data":
                self.data_lines.append(value)

    def finish(self, received_at: float) -> None:
        self._finish_event(received_at)
        if not self.done:
            raise ValueError("SSE response ended without [DONE]")
        if self.first_content_time is None:
            raise ValueError("SSE response had no generated content")

    def _finish_event(self, received_at: float) -> None:
        if not self.data_lines:
            return
        payload = b"\n".join(self.data_lines)
        self.data_lines.clear()
        if payload == b"[DONE]":
            self.done = True
            return
        try:
            row = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid SSE JSON event") from error
        if not isinstance(row, dict):
            raise ValueError("SSE JSON event is not an object")
        if row.get("error"):
            raise RuntimeError(f"SSE server error: {row['error']}")
        self.events += 1
        if isinstance(row.get("id"), str):
            self.response_ids.add(row["id"])
        if isinstance(row.get("usage"), dict):
            self.usage = row["usage"]
        choices = row.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("SSE choices is not a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("invalid SSE choice")
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ValueError("invalid SSE delta")
            piece = content_text(delta)
            if piece:
                self.text_parts.append(piece)
                if self.first_content_time is None:
                    self.first_content_time = received_at
            if isinstance(choice.get("finish_reason"), str):
                self.finish_reason = choice["finish_reason"]


def stream_request(url: str, body: dict[str, Any], timeout: float, raw_path: Path) -> dict[str, Any]:
    """The clock starts before urlopen; full wall ends after HTTP EOF."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    parser = SSEParser()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, raw_path.open("wb") as raw:
            if response.status != 200:
                raise RuntimeError(f"stream returned HTTP {response.status}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("stream response is not text/event-stream")
            while True:
                line = response.readline()
                received_at = time.perf_counter()
                if not line:
                    break
                raw.write(line)
                parser.feed_line(line, received_at)
                if received_at - started > timeout:
                    raise TimeoutError("stream exceeded request timeout")
    except urllib.error.HTTPError as error:
        raw_path.write_bytes(error.read())
        raise RuntimeError(f"stream returned HTTP {error.code}") from error
    ended = time.perf_counter()
    parser.finish(ended)
    return {
        "ttft_s": parser.first_content_time - started,
        "request_wall_s": ended - started,
        "ttft_boundary": "before HTTP urlopen to complete first SSE event with nonempty choice[0].delta.content",
        "wall_boundary": "before HTTP urlopen to HTTP EOF (includes prefill, generation, and transfer)",
        "completion_text_sha256": hashlib.sha256("".join(parser.text_parts).encode()).hexdigest(),
        "completion_chars": sum(map(len, parser.text_parts)),
        "sse_events": parser.events,
        "finish_reason": parser.finish_reason,
        "response_ids": sorted(parser.response_ids),
        "usage": parser.usage,
        "generated_token_ids_status": "unavailable_from_stream; no raw-ID parity claim",
        "raw_sse_path": str(raw_path),
        "raw_sse_sha256": base.sha256(raw_path),
    }


def validate_config(config: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    evaluation = config["evaluation"]
    variants = base.selected_variants(evaluation)
    expected = ("target_only", "ordinary_eagle", "draft_q8_0", "draft_q4_0", *base.W1AX_VARIANTS)
    if variants != expected:
        raise ValueError("stream diagnostic requires the full eight-path W1Ax matrix")
    specs = base.w1ax_specs(config, variants)
    base.weight_only_specs(config, variants)
    base.w1ax_policy(evaluation, True)
    if evaluation.get("w1ax_policy_diagnostic") or evaluation.get("round_trace", False):
        raise ValueError("stream diagnostic requires the primary policy without round tracing")
    if evaluation.get("temperature") != 0.0 or evaluation.get("enable_thinking") is not False:
        raise ValueError("stream diagnostic requires greedy non-thinking generation")
    if evaluation.get("warmup_requests") != WARMUPS or evaluation.get("repetitions") != REPETITIONS:
        raise ValueError("stream diagnostic requires two warmups and five repetitions")
    if config.get("precision", {}).get("target_weights") != "f16" or config["precision"].get("target_kv") != "f16" or config["precision"].get("draft_kv") != "f16":
        raise ValueError("stream diagnostic requires FP16 target and F16 KV")
    args = config["server"]["common_args"]
    required = (("--parallel", "1"), ("--cache-type-k", "f16"), ("--cache-type-v", "f16"))
    if any(not any(args[i:i + 2] == list(pair) for i in range(len(args) - 1)) for pair in required):
        raise ValueError("server requires concurrency one and f16 KV")
    if config["server"]["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("server must bind loopback")
    return variants, specs


def selected_prompts(path: Path) -> list[dict[str, Any]]:
    if base.sha256(path) != EXPECTED_CONTEXT_SHA256:
        raise ValueError("context prompt manifest differs from the frozen SHA256")
    by_id = {prompt["id"]: prompt for prompt in base.load_prompts(path)}
    if any(prompt_id not in by_id for prompt_id in PROMPT_IDS):
        raise ValueError("frozen context manifest lacks selected stable IDs")
    selected = [by_id[prompt_id] for prompt_id in PROMPT_IDS]
    if [prompt.get("context_bin") for prompt in selected] != ["short", "medium", "long"]:
        raise ValueError("selected prompt IDs do not match the three context bins")
    return selected


def run(config_path: Path, run_id: str, prompt_file: Path, *, dry_run: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run ID must be a safe relative name")
    config = tomllib.loads(config_path.read_text())
    variants, specs = validate_config(config)
    prompts = selected_prompts(prompt_file)
    paths = {
        "binary": base.resolve(ROOT, config["server"]["binary"]),
        "target": base.resolve(ROOT, config["models"]["target"]),
        "ordinary_draft": base.resolve(ROOT, config["models"]["ordinary_draft"]),
        **{variant: base.resolve(ROOT, config["weight_only_variants"][variant.removeprefix("draft_")]["draft"]) for variant in base.WEIGHT_ONLY_VARIANTS},
        **{variant: base.resolve(ROOT, config["w1ax"]["draft"]) for variant in base.W1AX_VARIANTS},
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if base.sha256(paths["target"]) != EXPECTED_TARGET_SHA256:
        raise ValueError("FP16 target differs from the frozen W1Ax study artifact")
    if base.sha256(paths["draft_w1a1"]) != EXPECTED_W1AX_SHA256:
        raise ValueError("shared all-nine W1Ax draft differs from the frozen study artifact")
    if not base.available_port(config["server"]["host"], config["server"]["port"]):
        raise RuntimeError("configured server port is occupied")
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, destination / "resolved-config.toml")
    shutil.copy2(prompt_file, destination / "context-prompts.jsonl")
    environment = {key: os.environ[key] for key in base.SAFE_INHERITED_ENV if key in os.environ}
    environment.update(config.get("environment", {}))
    for selector in ("GGML_CUDA_W1A1_MMA", "GGML_CUDA_W8A8_MMA", "GGML_CUDA_W4A4_MMA", "GGML_W1AX_ACT_BITS", "GGML_W1AX_A4_KERNEL", "W1AX_ROUND_TRACE_JSONL"):
        environment.pop(selector, None)
    variant_envs = {
        variant: {
            **environment,
            "GGML_CUDA_W1A1_MMA": "0",
            "GGML_CUDA_W8A8_MMA": "0",
            "GGML_CUDA_W4A4_MMA": "0",
            **({"GGML_W1AX_ACT_BITS": str(base.W1AX_BITS[variant])} if variant in specs else {}),
            **({"GGML_W1AX_A4_KERNEL": "bitserial"} if variant == "draft_w1a4" else {}),
        }
        for variant in variants
    }
    orders = base.schedule(REPETITIONS, variants)
    llama_gitlink = base.git_output("ls-tree", "HEAD", "third_party/llama.cpp").split()[2]
    llama_checkout_commit = base.git_output("-C", "third_party/llama.cpp", "rev-parse", "HEAD").strip()
    if llama_checkout_commit != llama_gitlink:
        raise ValueError("llama.cpp checkout differs from published project gitlink")
    manifest = {
        "schema": "w1ax_streaming_v1",
        "started_utc": now(),
        "platform": platform.platform(),
        "python": sys.version,
        "config_source": str(config_path),
        "config_sha256": base.sha256(config_path),
        "prompt_source": str(prompt_file),
        "prompt_sha256": base.sha256(prompt_file),
        "selected_prompt_ids": list(PROMPT_IDS),
        "selected_prompt_bins": {prompt["id"]: prompt["context_bin"] for prompt in prompts},
        "output_caps": list(CAPS),
        "request_options_excluding_messages_and_cap": {
            key: value for key, value in {**base.request_body(config, prompts[0]), "stream": True}.items()
            if key not in ("messages", "max_tokens")
        },
        "warmups_per_server": WARMUPS,
        "measured_repetitions": REPETITIONS,
        "variants": list(variants),
        "orders": orders,
        "model_and_binary_files": {key: {"path": str(path), "sha256": base.sha256(path), "bytes": path.stat().st_size} for key, path in paths.items()},
        "variant_environments": variant_envs,
        "commands": {variant: base.command_for(config, paths, variant) for variant in variants},
        "initial_gpu_snapshot": base.gpu_snapshot(),
        "environment_manifest": base.environment_manifest(paths["binary"]),
        "project_commit": base.git_output("rev-parse", "HEAD").strip(),
        "llama_gitlink": llama_gitlink,
        "llama_checkout_commit": llama_checkout_commit,
        "measurement": "streaming TTFT and full request wall; not nonstreaming decode throughput",
        "token_parity": "stream chunks may omit token IDs; no raw-ID parity claim",
    }
    write_json(destination / "manifest.json", manifest)
    if dry_run:
        return destination
    base_url = f"http://{config['server']['host']}:{config['server']['port']}"
    records: list[dict[str, Any]] = []
    try:
        for cap in CAPS:
            for repetition, order in enumerate(orders):
                for variant in order:
                    server_dir = destination / f"cap-{cap}" / f"rep-{repetition:02d}" / variant
                    server_dir.mkdir(parents=True)
                    write_json(server_dir / "gpu-before.json", base.gpu_snapshot())
                    write_json(server_dir / "environment.json", variant_envs[variant])
                    with (server_dir / "server.log").open("wb") as log:
                        process = subprocess.Popen(
                            manifest["commands"][variant], cwd=ROOT, env=variant_envs[variant],
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=os.name == "posix",
                        )
                        try:
                            base.wait_ready(process, base_url, config["server"]["startup_timeout_s"])
                            status, raw_props = base.request_json(base_url + "/props", None, 10)
                            (server_dir / "props.json").write_text(raw_props)
                            if status != 200:
                                raise RuntimeError(f"/props returned HTTP {status}")
                            write_json(server_dir / "gpu-loaded.json", base.gpu_snapshot())
                            for index in range(WARMUPS):
                                prompt = prompts[index % len(prompts)]
                                body = {**base.request_body(config, prompt), "max_tokens": cap, "stream": True}
                                stream_request(base_url + "/v1/chat/completions", body, config["server"]["request_timeout_s"], server_dir / "warmup" / f"request-{index:02d}.sse")
                            for prompt in prompts:
                                body = {**base.request_body(config, prompt), "max_tokens": cap, "stream": True}
                                raw_path = server_dir / "measured" / f"{prompt['id']}.sse"
                                measurement = stream_request(base_url + "/v1/chat/completions", body, config["server"]["request_timeout_s"], raw_path)
                                measurement.update({
                                    "cap": cap, "repetition": repetition, "variant": variant,
                                    "prompt_id": prompt["id"], "context_bin": prompt["context_bin"],
                                    "request_id": f"cap-{cap}/rep-{repetition:02d}/{variant}/{prompt['id']}",
                                    "server_request_index": WARMUPS + prompts.index(prompt),
                                    "w1ax_activation_bits_selector": variant_envs[variant].get("GGML_W1AX_ACT_BITS"),
                                })
                                write_json(raw_path.with_suffix(".json"), measurement)
                                records.append(measurement)
                                write_json(destination / "records.json", records)
                        finally:
                            base.stop_server(process)
                            log.flush()
                            server_log = (server_dir / "server.log").read_text(errors="replace")
                            evidence = base.w1ax_dispatch_evidence(server_log, specs[variant], specs) if variant in specs else {"status": "not_w1ax_variant"}
                            write_json(server_dir / "dispatch-evidence.json", evidence)
                            write_json(server_dir / "gpu-after.json", base.gpu_snapshot())
                    if variant in specs and not evidence["cuda_w1ax_dispatch_confirmed"]:
                        raise RuntimeError(f"{variant}: all-nine loader, graph, or CUDA dispatch marker missing")
        expected = len(CAPS) * REPETITIONS * len(variants) * len(prompts)
        if len(records) != expected:
            raise RuntimeError(f"incomplete streaming matrix: {len(records)}/{expected}")
        summary = {
            "status": "complete", "schema": "w1ax_streaming_summary_v1", "requests": len(records),
            "metrics": "streaming TTFT and full HTTP wall only; not decode tokens/s",
            "by_cap_bin_variant": [
                {
                    "cap": cap, "context_bin": prompt["context_bin"], "variant": variant,
                    "ttft_median_s": median(row["ttft_s"] for row in records if row["cap"] == cap and row["prompt_id"] == prompt["id"] and row["variant"] == variant),
                    "request_wall_median_s": median(row["request_wall_s"] for row in records if row["cap"] == cap and row["prompt_id"] == prompt["id"] and row["variant"] == variant),
                }
                for cap in CAPS for prompt in prompts for variant in variants
            ],
            "final_gpu_snapshot": base.gpu_snapshot(),
        }
        write_json(destination / "summary.json", summary)
    except BaseException as error:
        write_json(destination / "failure.json", {"type": type(error).__name__, "message": str(error), "time_utc": now()})
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="resolved full eight-path W1Ax TOML")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "data/w1ax-context/prompts.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(run(args.config.resolve(), args.run_id, args.prompt_file.resolve(), dry_run=args.dry_run))


if __name__ == "__main__":
    main()
