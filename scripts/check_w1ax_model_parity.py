#!/usr/bin/env python3
"""Compare CPU and CUDA draft placement for one frozen all-nine W1A1 draft.

This is a correctness diagnostic, not a timing benchmark. Run it beneath
scripts/remote_job.py on the RTX 2080 Ti so the supervisor owns the process
group and raw output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BITS = (1, 4, 8, 16)
LOADER_MARKER = "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)"
CUDA_MARKERS = {
    1: "CUDA packed W1A1 XOR/POPCOUNT dispatch",
    4: "CUDA packed W1A4 BITSERIAL dispatch",
    8: "CUDA packed W1A8 INT8 dispatch",
    16: "CUDA packed W1A16 FP16 SIGNADD dispatch",
}
PROMPT_FILE = ROOT / "configs" / "acceptance_prompts.jsonl"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_snapshot(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {
            "command": command, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_prompt(path: Path, prompt_id: str) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        prompt = json.loads(line)
        if prompt.get("id") == prompt_id:
            if not isinstance(prompt.get("messages"), list) or not prompt["messages"]:
                raise ValueError(f"prompt {prompt_id!r} has no messages")
            return prompt
    raise ValueError(f"prompt id {prompt_id!r} not found in {path}")


def token_ids(response: dict[str, Any]) -> list[int] | None:
    """Read generated IDs from llama-server's verbose response field."""
    verbose = response.get("__verbose")
    values = verbose.get("tokens") if isinstance(verbose, dict) else None
    if isinstance(values, list) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        return values
    return None


def selected_logprobs(response: dict[str, Any]) -> Any:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("logprobs")


def mode_evidence(bits: int, mode: str, server_command: list[str], server_log: str) -> dict[str, Any]:
    loader_seen = LOADER_MARKER in server_log
    graph_marker = f"EAGLE3 W1Ax activation bits: {bits}"
    graph_seen = re.search(re.escape(graph_marker) + r"(?!\d)", server_log) is not None
    cuda_marker = CUDA_MARKERS[bits]
    cuda_seen = cuda_marker in server_log
    wrong_mode_marker_seen = any(
        re.search(re.escape(f"EAGLE3 W1Ax activation bits: {other}") + r"(?!\d)", server_log)
        or CUDA_MARKERS[other] in server_log
        for other in BITS if other != bits
    )
    draft_ngl_index = server_command.index("--spec-draft-ngl") + 1
    actual_ngl = server_command[draft_ngl_index]
    expected_ngl = "0" if mode == "cpu" else "all"
    draft_device_index = server_command.index("--spec-draft-device") + 1
    actual_device = server_command[draft_device_index]
    expected_device = "none" if mode == "cpu" else "CUDA0"
    target_gpu_marker_seen = re.search(
        r"offloaded\s+[1-9]\d*/\d+ layers to GPU", server_log
    ) is not None
    if mode == "cuda":
        placement_seen = actual_ngl == expected_ngl and actual_device == expected_device and cuda_seen
    else:
        placement_seen = actual_ngl == expected_ngl and actual_device == expected_device and not cuda_seen
    return {
        "mode": mode,
        "target_gpu_offload_marker_seen": target_gpu_marker_seen,
        "spec_draft_ngl": actual_ngl,
        "expected_spec_draft_ngl": expected_ngl,
        "spec_draft_device": actual_device,
        "expected_spec_draft_device": expected_device,
        "all_nine_loader_marker": LOADER_MARKER,
        "all_nine_loader_confirmed": loader_seen,
        "activation_marker": graph_marker,
        "activation_marker_confirmed": graph_seen,
        "cuda_dispatch_marker": cuda_marker,
        "cuda_dispatch_marker_seen": cuda_seen,
        "wrong_mode_marker_seen": wrong_mode_marker_seen,
        "placement_confirmed": (
            placement_seen and target_gpu_marker_seen and loader_seen and graph_seen
            and not wrong_mode_marker_seen
        ),
    }


def build_server_command(
    binary: Path, target: Path, draft: Path, host: str, port: int, draft_ngl: str
) -> list[str]:
    draft_device = "none" if draft_ngl == "0" else "CUDA0"
    return [
        str(binary), "-m", str(target), "-md", str(draft),
        "--spec-type", "draft-eagle3", "--spec-draft-n-max", "5",
        "--spec-draft-p-min", "0", "--spec-draft-device", draft_device,
        "--spec-draft-ngl", draft_ngl,
        "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
        "--n-gpu-layers", "all", "--ctx-size", "2048", "--parallel", "1",
        "--fit", "off", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--jinja", "--metrics", "--perf", "-lv", "4",
        "--host", host, "--port", str(port),
    ]


def request_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, str]:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def wait_ready(process: subprocess.Popen[bytes], url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup ({process.returncode})")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"llama-server readiness timeout at {url}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    # Catch children that outlive the server leader, matching remote supervisor policy.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_one(
    *, run_dir: Path, bits: int, mode: str, binary: Path, target: Path, draft: Path,
    prompt: dict[str, Any], host: str, startup_timeout: float, request_timeout: float,
) -> dict[str, Any]:
    cell = run_dir / f"a{bits}-{mode}"
    cell.mkdir()
    port = free_port(host)
    command = build_server_command(binary, target, draft, host, port, "0" if mode == "cpu" else "all")
    env = os.environ.copy()
    env["GGML_W1AX_ACT_BITS"] = str(bits)
    env.pop("GGML_W1AX_A4_KERNEL", None)
    log_path = cell / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base_url = f"http://{host}:{port}"
        result = None
        try:
            wait_ready(process, base_url, startup_timeout)
            body = {
                "messages": prompt["messages"], "max_tokens": 32,
                "temperature": 0, "seed": 42, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_format": "none", "return_tokens": True, "verbose": True,
                "ignore_eos": True,
                "logprobs": True, "top_logprobs": 5,
            }
            json_write(cell / "request-with-logprobs.json", body)
            status, raw = request_json(base_url + "/v1/chat/completions", body, request_timeout)
            logprobs_requested = True
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            # Some server builds reject the optional OpenAI logprobs fields. Retry
            # the same greedy request without them and preserve both responses.
            if status != 200 and ("logprobs" in raw.lower() or "top_logprobs" in raw.lower()):
                (cell / "logprobs-rejected-response.json").write_text(raw)
                body.pop("logprobs")
                body.pop("top_logprobs")
                json_write(cell / "request.json", body)
                status, raw = request_json(base_url + "/v1/chat/completions", body, request_timeout)
                logprobs_requested = False
                parsed = None
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            if logprobs_requested:
                json_write(cell / "request.json", body)
            (cell / "response.json").write_text(raw)
            if status != 200 or not isinstance(parsed, dict):
                raise RuntimeError(f"HTTP {status}; see {cell / 'response.json'}")
            ids = token_ids(parsed)
            if ids is None:
                raise RuntimeError(f"server omitted __verbose.tokens; see {cell / 'response.json'}")
            evidence = mode_evidence(bits, mode, command, log_path.read_text(errors="replace"))
            result = {
                "bits": bits, "mode": mode, "command": command,
                "environment_selector": {"GGML_W1AX_ACT_BITS": str(bits)},
                "mode_evidence": evidence,
                "http_status": status,
                "generated_token_count": len(ids), "generated_token_ids": ids,
                "generated_token_ids_sha256": sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
                "completion_content": ((parsed.get("choices") or [{}])[0].get("message") or {}).get("content"),
                "logprobs_requested": logprobs_requested,
                "logprobs_available": selected_logprobs(parsed) is not None,
                "logprobs": selected_logprobs(parsed),
                "response_id": parsed.get("id"),
                "request_sha256": sha256_file(cell / "request.json"),
                "response_sha256": sha256_file(cell / "response.json"),
                "completion_tokens_reported": (parsed.get("usage") or {}).get("completion_tokens"),
                "finish_reason": ((parsed.get("choices") or [{}])[0]).get("finish_reason"),
                "server_log_sha256": sha256_file(log_path),
            }
            json_write(cell / "result.json", result)
            return result
        finally:
            stop_server(process)
            log.flush()
            if result is not None:
                result["server_log_sha256"] = sha256_file(log_path)
                json_write(cell / "result.json", result)


def run(args: argparse.Namespace) -> Path:
    binary = args.binary.resolve()
    target = args.target.resolve()
    draft = args.draft.resolve()
    prompt_path = args.prompts.resolve()
    config_path = args.config.resolve()
    for label, path in (("config", config_path), ("binary", binary), ("target", target), ("draft", draft), ("prompts", prompt_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    prompt = load_prompt(prompt_path, args.prompt_id)
    run_id = args.run_id or datetime.now(UTC).strftime("w1ax-parity-%Y%m%dT%H%M%SZ")
    run_dir = args.results.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n")
    outcomes = []
    for bits in BITS:
        pair = {}
        for mode in ("cpu", "cuda"):
            pair[mode] = run_one(
                run_dir=run_dir, bits=bits, mode=mode, binary=binary, target=target,
                draft=draft, prompt=prompt, host=args.host,
                startup_timeout=args.startup_timeout, request_timeout=args.request_timeout,
            )
            if not pair[mode]["mode_evidence"]["placement_confirmed"]:
                raise RuntimeError(f"{bits}-bit {mode} runtime markers did not confirm placement")
            if pair[mode]["generated_token_count"] != 32:
                raise RuntimeError(f"{bits}-bit {mode} emitted {pair[mode]['generated_token_count']} tokens, expected 32")
        cpu_ids, cuda_ids = pair["cpu"]["generated_token_ids"], pair["cuda"]["generated_token_ids"]
        outcomes.append({
            "bits": bits,
            "cpu_token_ids_sha256": pair["cpu"]["generated_token_ids_sha256"],
            "cuda_token_ids_sha256": pair["cuda"]["generated_token_ids_sha256"],
            "exact_token_id_match": cpu_ids == cuda_ids,
            "first_difference_index": next((i for i, (a, b) in enumerate(zip(cpu_ids, cuda_ids)) if a != b), None),
            "cpu_result": f"a{bits}-cpu/result.json",
            "cuda_result": f"a{bits}-cuda/result.json",
        })
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "same-model CPU-draft versus CUDA-draft greedy token parity; no timing claim",
        "run_id": run_id, "hardware_required": "RTX 2080 Ti / SM75",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "target": {"path": str(target), "sha256": sha256_file(target), "precision": "FP16", "placement": "CUDA via --n-gpu-layers all"},
        "draft": {"path": str(draft), "sha256": sha256_file(draft), "format": "same all-nine packed W1A1 GGUF for every cell"},
        "prompt": {"id": prompt["id"], "path": str(prompt_path), "sha256": sha256_bytes(json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())},
        "environment": {
            "gpu": command_snapshot(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
            "cuda_compiler": command_snapshot(["nvcc", "--version"]),
            "project_commit": command_snapshot(["git", "rev-parse", "HEAD"]),
            "llama_cpp_commit": command_snapshot(["git", "-C", str(ROOT / "third_party" / "llama.cpp"), "rev-parse", "HEAD"]),
        },
        "settings": {"activation_bits": list(BITS), "draft_ngl_modes": {"cpu": "0", "cuda": "all"}, "speculative_draft_length": 5, "min_draft_probability": 0.0, "max_generated_tokens": 32, "temperature": 0, "seed": 42, "thinking": False, "ignore_eos": True, "kv_precision": "f16"},
        "cells": outcomes,
        "all_token_ids_match": all(row["exact_token_id_match"] for row in outcomes),
        "timing_claim": None,
    }
    json_write(run_dir / "manifest.json", manifest)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="explicit llama-server executable")
    parser.add_argument("--config", required=True, type=Path, help="explicit frozen W1Ax evaluation config for provenance")
    parser.add_argument("--target", required=True, type=Path, help="explicit FP16 target GGUF")
    parser.add_argument("--draft", required=True, type=Path, help="explicit audited all-nine packed W1A1 GGUF")
    parser.add_argument("--prompts", type=Path, default=PROMPT_FILE)
    parser.add_argument("--prompt-id", default="prose-01", help="frozen historical prompt ID")
    parser.add_argument("--run-id", help="unique run directory name")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    run_dir = run(args)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
