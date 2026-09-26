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
    before_capture = file_inventory(capture_dir)
    env = os.environ.copy()
    env.update(
        {
            "GGML_W1AX_ACT_BITS": "1",
            "GGML_W1AX_CAPTURE_DIR": str(capture_dir),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
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
                request_path = run_dir / f"request-{index + 1:02d}.json"
                response_path = run_dir / f"response-{index + 1:02d}.json"
                write_json(request_path, body)
                started = time.monotonic()
                response = http_json(
                    f"{base_url}/v1/chat/completions", body, timeout=args.request_timeout
                )
                elapsed = time.monotonic() - started
                write_json(response_path, response)
                ids = generated_token_ids(response)
                if ids is None:
                    raise RuntimeError(f"{prompt['id']}: server omitted raw generated token IDs")
                if len(ids) > MAX_OUTPUT_TOKENS:
                    raise RuntimeError(
                        f"{prompt['id']}: server returned more than {MAX_OUTPUT_TOKENS} token IDs"
                    )
                manifest["requests"].append(
                    {
                        "prompt_id": prompt["id"],
                        "request": {"path": str(request_path), "sha256": sha256(request_path)},
                        "response": {"path": str(response_path), "sha256": sha256(response_path)},
                        "generated_token_ids": ids,
                        "generated_token_ids_sha256": hashlib.sha256(
                            json.dumps(ids, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "generated_token_count": len(ids),
                        "client_elapsed_s": elapsed,
                    }
                )
        manifest["server_stop"] = stop_server(process)
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
