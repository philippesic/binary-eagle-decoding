#!/usr/bin/env python3
"""Capture CUDA W1A1 activation inputs for a few frozen historical prompts.

Run inside a supervised ignored run directory, for example:

    python3 scripts/remote_job.py w1ax-activation-capture-01 -- \
      python3 scripts/capture_w1ax_activations.py \
      --config configs/native_benchmark_w1ax.toml \
      --run-dir runs/w1ax-activation-capture-01 \
      --capture-dir runs/w1ax-activation-capture-01/activations \
      --prompt-count 3

The script creates the capture directory before starting the server and never
clears it. Results are capture evidence only; request timings are not benchmark
data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOADER_MARKER = "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)"
GRAPH_MARKER = "EAGLE3 W1Ax activation bits: 1"
CUDA_MARKER = "CUDA packed W1A1 XOR/POPCOUNT dispatch"
MAX_OUTPUT_TOKENS = 32


class StopRequested(Exception):
    """Raised by signal handlers so the server cleanup path can run."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    ids = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        prompt_id = row.get("id") if isinstance(row, dict) else None
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in ids:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate prompt id")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number}: expected nonempty messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"{path}:{line_number}: invalid text message")
        ids.add(prompt_id)
        prompts.append(row)
    if not prompts:
        raise ValueError(f"prompt file is empty: {path}")
    return prompts


def generated_token_ids(response: dict[str, Any]) -> list[int] | None:
    candidates: list[Any] = [response.get("generated_token_ids"), response.get("token_ids")]
    verbose = response.get("__verbose")
    if isinstance(verbose, dict):
        candidates.append(verbose.get("tokens"))
    candidates.append(response.get("tokens"))
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        first = choices[0]
        candidates.extend(
            (first.get("generated_token_ids"), first.get("token_ids"), first.get("tokens"))
        )
        message = first.get("message")
        if isinstance(message, dict):
            candidates.append(message.get("tokens"))
    for value in candidates:
        if isinstance(value, list) and all(type(item) is int for item in value):
            return value
        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(item, dict) and type(item.get("id", item.get("token_id"))) is int
                for item in value
            )
        ):
            return [item.get("id", item.get("token_id")) for item in value]
    return None


def file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.glob("*"))
        if path.is_file()
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def capture_inventory(capture_dir: Path) -> dict[str, Any]:
    """Take a cheap per-request snapshot without hashing large activation tensors."""
    files = {
        path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(capture_dir.glob("op-*.bin"))
        if path.is_file()
    }
    rows = read_jsonl(capture_dir / "captures.jsonl")
    return {
        "files": files,
        "sequences": sorted(
            row["sequence"]
            for row in rows
            if isinstance(row.get("sequence"), int) and not isinstance(row.get("sequence"), bool)
        ),
        "sidecar_rows": rows,
    }


def trace_inventory(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def validate_capture_sidecar_row(row: dict[str, Any]) -> None:
    if row.get("schema_version") != 1:
        raise ValueError("unsupported captures.jsonl schema_version")
    sequence = row.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("capture sidecar row has invalid sequence")
    expected_file = f"op-{sequence:012d}.bin"
    if row.get("file") != expected_file:
        raise ValueError(f"capture sidecar sequence {sequence} must name {expected_file}")
    if not isinstance(row.get("weight_tensor"), str) or not row["weight_tensor"]:
        raise ValueError(f"capture sidecar row {sequence} has no weight_tensor")
    for field in ("k", "m", "n", "bits", "timestamp_us", "capture_start_us", "capture_end_us"):
        if type(row.get(field)) is not int:
            raise ValueError(f"capture sidecar row {sequence} has invalid {field}")
    if row["capture_end_us"] < row["capture_start_us"]:
        raise ValueError(f"capture sidecar row {sequence} has a negative capture span")


def validate_round_trace_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("round-trace.jsonl is empty; no EAGLE round evidence was recorded")
    for index, row in enumerate(rows):
        if row.get("schema") != "w1ax_eagle_round_v1":
            raise ValueError(f"round-trace row {index} has unsupported schema")
        if type(row.get("round_start_us")) is not int or type(row.get("round_end_us")) is not int:
            raise ValueError(f"round-trace row {index} has invalid round bounds")
        if row["round_end_us"] < row["round_start_us"]:
            raise ValueError(f"round-trace row {index} has a negative round span")


def validate_request_capture_delta(delta: dict[str, Any]) -> None:
    rows = delta["capture_events"]
    filenames = delta["new_filenames"]
    if not filenames:
        raise ValueError("request produced no activation captures")
    for row in rows:
        validate_capture_sidecar_row(row)
    file_counts = collections.Counter(filenames)
    row_counts = collections.Counter(row.get("file") for row in rows)
    if file_counts != row_counts:
        missing_rows = sorted((file_counts - row_counts).elements())
        missing_files = sorted((row_counts - file_counts).elements())
        raise ValueError(
            "request capture files and sidecar rows differ "
            f"(files without rows={missing_rows}, rows without files={missing_files})"
        )
    sequences = [row["sequence"] for row in rows]
    if len(sequences) != len(set(sequences)):
        raise ValueError("request has duplicate capture sidecar sequences")


def validate_capture_run(
    capture_dir: Path,
    round_trace_path: Path,
    requests: list[dict[str, Any]],
) -> None:
    sidecar_path = capture_dir / "captures.jsonl"
    if not sidecar_path.is_file():
        raise ValueError(
            "captures.jsonl is missing; selected runtime did not emit capture metadata"
        )
    snapshot = capture_inventory(capture_dir)
    rows = snapshot["sidecar_rows"]
    if not rows:
        raise ValueError("captures.jsonl contains no activation records")
    for row in rows:
        validate_capture_sidecar_row(row)
    sequences = sorted(row["sequence"] for row in rows)
    if sequences != list(range(len(rows))):
        raise ValueError("captures.jsonl sequences are incomplete or duplicated")
    file_counts = collections.Counter(snapshot["files"].keys())
    row_counts = collections.Counter(row["file"] for row in rows)
    if file_counts != row_counts:
        raise ValueError("capture binary files and captures.jsonl rows do not match one-to-one")
    if not round_trace_path.is_file():
        raise ValueError("round-trace.jsonl is missing")
    round_rows = read_jsonl(round_trace_path)
    validate_round_trace_rows(round_rows)
    round_capture_count = sum(
        event.get("phase") == "round"
        for request in requests
        for event in request.get("capture", {}).get("capture_events", [])
    )
    if round_capture_count == 0:
        raise ValueError("no activation capture span overlaps an EAGLE round")


def request_capture_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join new capture sidecar records to round spans by their shared ggml clock."""
    prior_rows = before["sidecar_rows"]
    prior_counts: dict[str, int] = {}
    for row in prior_rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        prior_counts[key] = prior_counts.get(key, 0) + 1
    new_rows = []
    for row in after["sidecar_rows"]:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        count = prior_counts.get(key, 0)
        if count:
            prior_counts[key] = count - 1
        else:
            new_rows.append(row)
    new_files = sorted(set(after["files"]) - set(before["files"]))
    intervals = [
        row
        for row in trace_rows
        if isinstance(row.get("round_start_us"), int) and isinstance(row.get("round_end_us"), int)
    ]
    enriched_rows = []
    for row in sorted(new_rows, key=lambda item: (item.get("sequence", -1), item.get("file", ""))):
        capture_start = row.get("capture_start_us", row.get("timestamp_us"))
        capture_end = row.get("capture_end_us", capture_start)
        overlaps = [
            trace
            for trace in intervals
            if isinstance(capture_start, int)
            and isinstance(capture_end, int)
            and capture_start <= trace["round_end_us"]
            and capture_end >= trace["round_start_us"]
        ]
        enriched_rows.append(
            {
                **row,
                "phase": "round" if overlaps else "prefill_or_outside_round",
                "round_indices": [trace.get("round_index") for trace in overlaps],
                "round_task_ids": [trace.get("task_id") for trace in overlaps],
            }
        )
    sequences = [row.get("sequence") for row in new_rows if isinstance(row.get("sequence"), int)]
    new_filenames_sorted = sorted(new_files)
    return {
        "files_before": before["files"],
        "files_after": after["files"],
        "sequences_before": before["sequences"],
        "sequences_after": after["sequences"],
        "sidecar_rows_before": len(before["sidecar_rows"]),
        "sidecar_rows_after": len(after["sidecar_rows"]),
        "new_filenames": new_files,
        "filename_range": (
            [new_filenames_sorted[0], new_filenames_sorted[-1]] if new_filenames_sorted else None
        ),
        "sequence_range": [min(sequences), max(sequences)] if sequences else None,
        "new_sequences": sorted(sequences),
        "capture_events": enriched_rows,
        "round_trace_rows": trace_rows,
        "round_trace_rows_added": len(trace_rows),
        "prefill_or_outside_round_count": sum(
            event["phase"] == "prefill_or_outside_round" for event in enriched_rows
        ),
    }


def server_command(config: dict[str, Any], binary: Path, target: Path, draft: Path) -> list[str]:
    server = config["server"]
    common_args = list(server["common_args"])
    # Force one request slot even if a local config was edited for another run.
    while "--parallel" in common_args:
        position = common_args.index("--parallel")
        del common_args[position : position + 2]
    return [
        str(binary),
        "-m",
        str(target),
        "-md",
        str(draft),
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        "5",
        "--spec-draft-p-min",
        "0.0",
        "--spec-draft-ngl",
        "all",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        *common_args,
        "--parallel",
        "1",
        "--host",
        server["host"],
        "--port",
        str(server["port"]),
    ]


def request_body(prompt: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": prompt["messages"],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": evaluation["temperature"],
        "seed": evaluation["seed"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": evaluation.get("enable_thinking", False)},
        "reasoning_format": "none",
        "return_tokens": True,
        "verbose": True,
    }


def benchmark_request_id(prompt_id: str) -> str:
    return f"rep-00/draft_w1a1/{prompt_id}"


def http_json(url: str, value: dict[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    data = json.dumps(value).encode() if value is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise ValueError("server returned a non-object JSON response")
    return decoded


def wait_ready(base_url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with status {process.returncode}"
            )
        try:
            http_json(f"{base_url}/health", timeout=2)
            return
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            time.sleep(1)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def stop_server(process: subprocess.Popen) -> dict[str, Any]:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            # Leave room for remote_job.py's ten second parent-group grace period.
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    # Ensure descendants in the server's dedicated session are also gone.
    if process.returncode is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"return_code": process.returncode, "stopped": process.poll() is not None}


def handle_stop_signal(signum: int, _frame: object) -> None:
    raise StopRequested(f"received signal {signum}; stopping llama-server")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    config_path = resolve(root, args.config)
    config = tomllib.loads(config_path.read_text())
    if config.get("precision", {}).get("target_weights", "").lower() != "f16":
        raise ValueError("capture requires the configured target weights to be FP16")
    if config.get("w1ax", {}).get("expected_loader_marker") != LOADER_MARKER:
        raise ValueError("config must require the all-nine W1A1 loader marker")
    if config.get("w1ax", {}).get("expected_cuda_markers", {}).get("1") != CUDA_MARKER:
        raise ValueError("config must require the CUDA W1A1 XOR/POPCOUNT marker")
    binary = resolve(root, args.binary or config["server"]["binary"])
    target = resolve(root, args.target or config["models"]["target"])
    draft = resolve(root, args.draft or config["w1ax"]["draft"])
    prompt_path = resolve(root, args.prompts or config["evaluation"]["prompt_file"])
    run_dir = resolve(root, args.run_dir)
    capture_dir = resolve(root, args.capture_dir)
    if not run_dir.is_dir():
        raise ValueError(f"run directory must already exist: {run_dir}")
    # The CUDA hook requires an existing path at process start. Create it here,
    # before setting GGML_W1AX_CAPTURE_DIR, and never clear existing contents.
    capture_dir.mkdir(parents=True, exist_ok=True)
    if (capture_dir / "captures.jsonl").exists() or any(capture_dir.glob("op-*.bin")):
        raise ValueError(
            "capture directory already contains W1Ax output; use a fresh run-specific directory"
        )
    for label, path in (
        ("server binary", binary),
        ("target GGUF", target),
        ("draft GGUF", draft),
        ("prompts", prompt_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if args.prompt_count not in (1, 2, 3):
        raise ValueError("--prompt-count must be 1, 2, or 3")
    prompts = load_prompts(prompt_path)[: args.prompt_count]
    port = args.port if args.port is not None else int(config["server"]["port"])
    host = args.host or config["server"]["host"]
    base_url = f"http://{host}:{port}"
    command_config = {**config, "server": {**config["server"], "host": host, "port": port}}
    command = server_command(command_config, binary, target, draft)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_copy = run_dir / "prompts.jsonl"
    prompt_copy.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in prompts
        )
    )
    round_trace_path = run_dir / "round-trace.jsonl"
    if round_trace_path.exists():
        raise ValueError(
            f"round trace already exists; use a fresh run directory: {round_trace_path}"
        )
    before_capture = file_inventory(capture_dir)
    env = os.environ.copy()
    env.update(
        {
            "GGML_W1AX_ACT_BITS": "1",
            "GGML_W1AX_CAPTURE_DIR": str(capture_dir),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "W1AX_ROUND_TRACE_JSONL": str(round_trace_path),
        }
    )
    env["CUDA_VISIBLE_DEVICES"] = config.get("environment", {}).get(
        "CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0")
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "activation input capture; no timing claim",
        "root": str(root),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "capture_driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "binary": {"path": str(binary), "sha256": sha256(binary), "bytes": binary.stat().st_size},
        "target": {"path": str(target), "sha256": sha256(target), "bytes": target.stat().st_size},
        "draft": {"path": str(draft), "sha256": sha256(draft), "bytes": draft.stat().st_size},
        "prompt_source": {"path": str(prompt_path), "sha256": sha256(prompt_path)},
        "prompt_copy": {"path": str(prompt_copy), "sha256": sha256(prompt_copy)},
        "prompt_ids": [row["id"] for row in prompts],
        "request_count": len(prompts),
        "request_max_tokens": MAX_OUTPUT_TOKENS,
        "policy": {"D": 5, "p_min": 0.0, "parallel": 1, "kv_k": "f16", "kv_v": "f16"},
        "environment": {
            key: env[key]
            for key in (
                "GGML_W1AX_ACT_BITS",
                "GGML_W1AX_CAPTURE_DIR",
                "GGML_CUDA_DISABLE_GRAPHS",
                "W1AX_ROUND_TRACE_JSONL",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "server_command": command,
        "capture_directory": str(capture_dir),
        "capture_files_before": before_capture,
        "server_stop": None,
        "requests": [],
        "validation": {
            "loader_marker": LOADER_MARKER,
            "graph_marker": GRAPH_MARKER,
            "cuda_marker": CUDA_MARKER,
        },
    }
    server_log = run_dir / "server.log"
    process = None
    error: str | None = None
    previous_handlers = {
        signum: signal.signal(signum, handle_stop_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with server_log.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            manifest["server_pid"] = process.pid
            manifest["server_process_group"] = process.pid
            wait_ready(base_url, process, args.startup_timeout)
            for index, prompt in enumerate(prompts):
                body = request_body(prompt, config["evaluation"])
                request_id = benchmark_request_id(prompt["id"])
                request_path = run_dir / f"request-{index + 1:02d}.json"
                response_path = run_dir / f"response-{index + 1:02d}.json"
                write_json(request_path, body)
                request_capture_before = capture_inventory(capture_dir)
                request_trace_before = trace_inventory(round_trace_path)
                started = time.monotonic()
                response = http_json(
                    f"{base_url}/v1/chat/completions", body, timeout=args.request_timeout
                )
                elapsed = time.monotonic() - started
                write_json(response_path, response)
                request_capture_after = capture_inventory(capture_dir)
                request_trace_after = trace_inventory(round_trace_path)
                old_trace_counts: dict[str, int] = {}
                for trace_row in request_trace_before:
                    key = json.dumps(trace_row, sort_keys=True, separators=(",", ":"))
                    old_trace_counts[key] = old_trace_counts.get(key, 0) + 1
                request_trace_delta = []
                for trace_row in request_trace_after:
                    key = json.dumps(trace_row, sort_keys=True, separators=(",", ":"))
                    count = old_trace_counts.get(key, 0)
                    if count:
                        old_trace_counts[key] = count - 1
                    else:
                        request_trace_delta.append(trace_row)
                capture_delta = request_capture_delta(
                    request_capture_before, request_capture_after, request_trace_delta
                )
                ids = generated_token_ids(response)
                manifest["requests"].append(
                    {
                        "prompt_id": prompt["id"],
                        "benchmark_request_id": request_id,
                        "request": {"path": str(request_path), "sha256": sha256(request_path)},
                        "response": {"path": str(response_path), "sha256": sha256(response_path)},
                        "generated_token_ids": ids,
                        "generated_token_ids_sha256": hashlib.sha256(
                            json.dumps(ids, separators=(",", ":")).encode()
                        ).hexdigest()
                        if ids is not None
                        else None,
                        "generated_token_count": len(ids) if ids is not None else None,
                        "client_elapsed_s": elapsed,
                        "capture": capture_delta,
                    }
                )
                validate_request_capture_delta(capture_delta)
                if ids is None:
                    raise RuntimeError(f"{prompt['id']}: server omitted raw generated token IDs")
                if len(ids) > MAX_OUTPUT_TOKENS:
                    raise RuntimeError(
                        f"{prompt['id']}: server returned more than {MAX_OUTPUT_TOKENS} token IDs"
                    )
        manifest["server_stop"] = stop_server(process)
        validate_capture_run(capture_dir, round_trace_path, manifest["requests"])
        log_text = server_log.read_text(errors="replace")
        marker_evidence = {
            "all_nine_loader": LOADER_MARKER in log_text,
            "activation_graph_mode_a1": GRAPH_MARKER in log_text,
            "cuda_w1a1_dispatch": CUDA_MARKER in log_text,
        }
        manifest["validation"]["markers_seen"] = marker_evidence
        if not all(marker_evidence.values()):
            raise RuntimeError(f"required CUDA/loader markers missing: {marker_evidence}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            manifest["server_stop"] = stop_server(process)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        manifest["capture_files_after"] = file_inventory(capture_dir)
        after = manifest["capture_files_after"]
        manifest["captured_files"] = {
            name: metadata
            for name, metadata in after.items()
            if before_capture.get(name) != metadata
        }
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        manifest["server_log"] = (
            {"path": str(server_log), "sha256": sha256(server_log)} if server_log.exists() else None
        )
        manifest["captures_sidecar"] = (
            {
                "path": str(capture_dir / "captures.jsonl"),
                "sha256": sha256(capture_dir / "captures.jsonl"),
            }
            if (capture_dir / "captures.jsonl").is_file()
            else None
        )
        try:
            round_trace_rows = len(read_jsonl(round_trace_path))
            round_trace_error = None
        except Exception as exc:
            round_trace_rows = None
            round_trace_error = f"{type(exc).__name__}: {exc}"
        manifest["round_trace"] = {
            "path": str(round_trace_path),
            "sha256": sha256(round_trace_path) if round_trace_path.is_file() else None,
            "rows": round_trace_rows,
            "schema": "w1ax_eagle_round_v1",
            "parse_error": round_trace_error,
        }
        manifest["error"] = error
        write_json(run_dir / "manifest.json", manifest)
    if error:
        raise RuntimeError(error)
    return run_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--config", type=Path, default=Path("configs/native_benchmark_w1ax.toml"))
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="existing ignored run directory"
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="CUDA-hook output directory; created if missing and never cleared",
    )
    parser.add_argument("--binary", type=Path, help="override configured llama-server")
    parser.add_argument("--target", type=Path, help="override configured FP16 target GGUF")
    parser.add_argument("--draft", type=Path, help="override configured all-nine W1A1 GGUF")
    parser.add_argument("--prompts", type=Path, help="override frozen prompt JSONL")
    parser.add_argument("--prompt-count", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--host", help="override configured HTTP bind host")
    parser.add_argument("--port", type=int, help="override configured HTTP port")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=180)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"capture failed: {error}")
        return 1
    print(f"capture complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
