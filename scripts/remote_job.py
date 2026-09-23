"""Supervise one remote experiment and terminate its entire process group on exit."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stop_signal: int | None = None


def request_stop(signum: int, _frame: object) -> None:
    global stop_signal
    stop_signal = signum


def stop_group(pgid: int, process: subprocess.Popen) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit before children it started; clean the group too.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="unique name containing letters, digits, dots, or hyphens")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.run_id):
        parser.error("invalid run_id")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("cache", "tmp", "torch", "huggingface", "cuda"):
        (run_dir / name).mkdir()
    env = os.environ.copy()
    env.update(
        XDG_CACHE_HOME=str(run_dir / "cache"),
        UV_CACHE_DIR=str(run_dir / "cache" / "uv"),
        HF_HOME=str(run_dir / "huggingface"),
        TORCH_HOME=str(run_dir / "torch"),
        CUDA_CACHE_PATH=str(run_dir / "cuda"),
        TMPDIR=str(run_dir / "tmp"),
    )
    record = {
        "run_id": args.run_id,
        "command": command,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "starting",
    }
    record_path = run_dir / "state.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    with (run_dir / "stdout.log").open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pgid = process.pid
        record.update(status="running", pid=process.pid, pgid=pgid)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            while process.poll() is None and stop_signal is None:
                time.sleep(0.5)
        finally:
            stop_group(pgid, process)
            record.update(
                status="interrupted" if stop_signal is not None else "finished",
                exit_code=process.poll(),
                ended_at_utc=datetime.now(UTC).isoformat(),
            )
            record_path.write_text(json.dumps(record, indent=2) + "\n")
    return 128 + stop_signal if stop_signal is not None else (process.returncode or 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileExistsError:
        print("Run ID already exists; choose a new one to preserve prior output", file=sys.stderr)
        sys.exit(2)
