#!/usr/bin/env python3
"""Run a frozen W1Ax diagnostic config suite sequentially, with resume state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
stop_signal: int | None = None
active_process: subprocess.Popen[bytes] | None = None


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def request_stop(signum: int, _frame: object) -> None:
    global stop_signal
    stop_signal = signum


def signal_benchmark(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def stop_active_process(process: subprocess.Popen[bytes], grace_seconds: float = 8) -> int:
    """Ask benchmark to unwind so its finally block shuts down llama-server."""
    signal_benchmark(process, signal.SIGINT)
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signal_benchmark(process, signal.SIGTERM)
    try:
        return process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        signal_benchmark(process, signal.SIGKILL)
        return process.wait()


def load_suite(suite_path: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    suite_path = suite_path.resolve()
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != "w1ax_diagnostic_suite_v1":
        raise ValueError("unsupported W1Ax suite manifest")
    verify_frozen_inputs(suite)
    configs = []
    for entry in suite.get("configs", []):
        path = Path(entry["path"]).resolve()
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"missing or changed frozen config: {path}")
        prompt = suite.get("prompt_sets", {}).get(entry.get("prompt_set"), {})
        if entry.get("prompt_file") != prompt.get("path") or entry.get("prompt_file_sha256") != prompt.get("sha256"):
            raise ValueError(f"manifest prompt hash records disagree for config: {path.name}")
        configs.append((path, entry))
    if len(configs) != 14:
        raise ValueError(f"expected 14 frozen configs, found {len(configs)}")
    return suite, configs


def verify_frozen_inputs(suite: dict[str, Any]) -> None:
    inputs = suite.get("input_files")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("suite manifest has no frozen input_files hashes")
    for key, entry in inputs.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"invalid frozen input manifest entry: {key}")
        path = Path(entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing ({key}): {path}")
        actual = sha256(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"frozen input hash changed ({key}): {path}")
    for key, digest in suite.get("model_server_hashes", {}).items():
        if key not in inputs or inputs[key].get("sha256") != digest:
            raise ValueError(f"manifest hash records disagree for frozen input: {key}")
    if suite.get("primary_config_sha256") != inputs.get("primary_config", {}).get("sha256"):
        raise ValueError("manifest primary config hash records disagree")
    for prompt_set, details in suite.get("prompt_sets", {}).items():
        key = f"prompt:{prompt_set}"
        recorded = inputs.get(key, {})
        if details.get("path") != recorded.get("path") or details.get("sha256") != recorded.get("sha256"):
            raise ValueError(f"manifest prompt hash records disagree for {prompt_set}")


def run_suite(suite_path: Path, *, root: Path = ROOT, python: str = sys.executable) -> int:
    global active_process, stop_signal
    stop_signal = None
    suite, configs = load_suite(suite_path)
    suite_path = suite_path.resolve()
    run_dir = suite_path.parent
    state_path = run_dir / "progress.json"
    logs_dir = run_dir / "launcher-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    benchmark = root / "scripts" / "benchmark_native_eagle.py"
    if not benchmark.is_file():
        raise FileNotFoundError(benchmark)
    suite_fingerprint = hashlib.sha256(
        json.dumps(
            {"schema": suite.get("schema"), "configs": [entry["sha256"] for _, entry in configs]},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("suite_fingerprint") != suite_fingerprint:
            raise ValueError("existing progress belongs to a different frozen suite")
    else:
        state = {
            "schema": "w1ax_diagnostic_progress_v1",
            "suite_path": str(suite_path),
            "suite_fingerprint": suite_fingerprint,
            "started_at_utc": now(),
            "configs": {},
        }
    entries = state.setdefault("configs", {})

    for config_path, config_entry in configs:
        # Detect a prompt/model/binary replacement between sequential cells too.
        verify_frozen_inputs(suite)
        config_hash = config_entry["sha256"]
        key = config_entry["name"]
        previous = entries.get(key)
        if previous and previous.get("config_sha256") != config_hash:
            raise ValueError(f"config hash changed since prior launch: {key}")
        if previous and previous.get("status") == "succeeded":
            continue
        if stop_signal is not None:
            break

        attempts = previous.get("attempts", []) if previous else []
        attempt_number = len(attempts) + 1
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", run_dir.name).strip("-.") or "w1ax"
        suffix = f"-{key.removesuffix('.toml')}-a{attempt_number}"
        run_id = f"{stem[: max(1, 79 - len(suffix))]}{suffix}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
            raise ValueError(f"generated unsafe benchmark run ID: {run_id}")
        results_path = root / "results" / run_id
        if results_path.exists():
            raise FileExistsError(
                f"benchmark result already exists for {run_id}; preserve it and rename the suite directory"
            )
        command = [python, str(benchmark), "--config", str(config_path), "--run-id", run_id]
        log_path = logs_dir / f"{run_id}.log"
        attempt = {
            "attempt": attempt_number,
            "run_id": run_id,
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "command": command,
            "started_at_utc": now(),
            "status": "running",
            "log_path": str(log_path),
        }
        if previous is None:
            previous = {"config_sha256": config_hash, "attempts": attempts}
            entries[key] = previous
        attempts.append(attempt)
        previous["attempts"] = attempts
        previous["status"] = "running"
        write_json(state_path, state)

        with log_path.open("wb") as log:
            active_process = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                while active_process.poll() is None and stop_signal is None:
                    time.sleep(0.2)
                if stop_signal is not None:
                    code = stop_active_process(active_process)
                else:
                    code = active_process.wait()
            finally:
                active_process = None
        attempt["ended_at_utc"] = now()
        attempt["exit_code"] = code
        attempt["status"] = "succeeded" if code == 0 else (
            "interrupted" if stop_signal is not None else "failed"
        )
        previous["status"] = attempt["status"]
        previous["last_exit_code"] = code
        write_json(state_path, state)
        if code != 0 or stop_signal is not None:
            state["ended_at_utc"] = now()
            state["status"] = "interrupted" if stop_signal is not None else "failed"
            write_json(state_path, state)
            return 128 + stop_signal if stop_signal is not None else code

    state["ended_at_utc"] = now()
    state["status"] = "complete" if all(
        entries.get(entry["name"], {}).get("status") == "succeeded"
        for _, entry in configs
    ) else "interrupted"
    write_json(state_path, state)
    return 0 if state["status"] == "complete" else (128 + stop_signal if stop_signal else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="suite.json emitted by prepare_w1ax_diagnostics.py")
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    return run_suite(args.suite)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
