"""Supervise one remote experiment and terminate its entire process group on exit."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stop_signal: int | None = None
stop_signal_at_utc: str | None = None


def request_stop(signum: int, _frame: object) -> None:
    global stop_signal, stop_signal_at_utc
    # Retain the first request even if another signal arrives during cleanup.
    if stop_signal is None:
        stop_signal = signum
        stop_signal_at_utc = datetime.now(UTC).isoformat()


def stop_receipt() -> dict | None:
    if stop_signal is None:
        return None
    return {"number": stop_signal, "name": signal.Signals(stop_signal).name,
            "received_at_utc": stop_signal_at_utc}


def stop_group(pgid: int, process: subprocess.Popen, grace_seconds: float = 10) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit before children it started; clean the group too.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class RotatingBinaryLog:
    """Bound the supervisor's raw stdout log while retaining recent output."""

    def __init__(self, path: Path, max_bytes: int, backups: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.stream = path.open("wb")
        self.size = 0

    def _rotate(self) -> None:
        self.stream.close()
        for ordinal in range(self.backups, 0, -1):
            old = self.path.with_name(f"{self.path.name}.{ordinal}")
            if ordinal == self.backups:
                old.unlink(missing_ok=True)
            elif old.exists():
                os.replace(old, self.path.with_name(f"{self.path.name}.{ordinal + 1}"))
        if self.path.exists():
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self.stream = self.path.open("wb")
        self.size = 0

    def write(self, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            if self.size == self.max_bytes:
                self._rotate()
            count = min(len(data) - offset, self.max_bytes - self.size)
            self.stream.write(data[offset:offset + count])
            self.size += count
            offset += count
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def _relay_output(pipe, log: RotatingBinaryLog, errors: list[BaseException]) -> None:
    try:
        while True:
            chunk = os.read(pipe.fileno(), 64 * 1024)
            if not chunk:
                break
            log.write(chunk)
    except BaseException as exc:
        errors.append(exc)
    finally:
        pipe.close()
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="unique name containing letters, digits, dots, or hyphens")
    parser.add_argument("--stop-grace-seconds", type=float, default=10,
                        help="seconds to wait after SIGTERM before killing the process group")
    parser.add_argument("--max-log-bytes", type=int, default=128 * 1024**2,
                        help="maximum size of each raw stdout log file")
    parser.add_argument("--log-backups", type=int, default=3,
                        help="number of rotated stdout logs to retain")
    argv = sys.argv[1:]
    if "--" in argv:
        separator = argv.index("--")
        option_args, command = argv[:separator], argv[separator + 1:]
    else:
        option_args, command = argv, []
    args = parser.parse_args(option_args)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.run_id):
        parser.error("invalid run_id")
    if not command:
        parser.error("provide a command after --")
    if not math.isfinite(args.stop_grace_seconds) or args.stop_grace_seconds <= 0:
        parser.error("--stop-grace-seconds must be finite and positive")
    if args.max_log_bytes < 1 or args.log_backups < 1:
        parser.error("--max-log-bytes and --log-backups must be positive")

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
        "supervisor_pid": os.getpid(),
        "supervisor_ppid": os.getppid(),
        "supervisor_pgid": os.getpgrp(),
        "supervisor_sid": os.getsid(0),
        "received_signal": None,
        "stop_grace_seconds": args.stop_grace_seconds,
        "log_rotation": {"max_bytes_per_file": args.max_log_bytes,
                         "backups": args.log_backups},
    }
    record_path = run_dir / "state.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    output_log = RotatingBinaryLog(run_dir / "stdout.log", args.max_log_bytes, args.log_backups)
    relay_errors: list[BaseException] = []
    relay = threading.Thread(target=_relay_output,
                             args=(process.stdout, output_log, relay_errors), daemon=True)
    relay.start()
    pgid = process.pid
    record.update(status="running", pid=process.pid, pgid=pgid)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        while process.poll() is None and stop_signal is None and not relay_errors:
            time.sleep(0.5)
    finally:
        if stop_signal is not None:
            record.update(status="stop_requested", received_signal=stop_receipt())
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        if relay_errors and stop_signal is None and process.poll() is None:
            # A log write failure must not leave a producer blocked on a full pipe.
            stop_signal_local = signal.SIGTERM
            try:
                os.killpg(pgid, stop_signal_local)
            except ProcessLookupError:
                pass
        stop_group(pgid, process, args.stop_grace_seconds)
        relay.join(timeout=args.stop_grace_seconds + 5)
        if relay.is_alive():
            relay_errors.append(RuntimeError("stdout relay did not exit after process-group stop"))
        record.update(
            status="interrupted" if stop_signal is not None else "finished",
            received_signal=stop_receipt(),
            exit_code=process.poll(),
            ended_at_utc=datetime.now(UTC).isoformat(),
        )
        if relay_errors:
            record["log_error"] = "; ".join(str(exc) for exc in relay_errors)
            record["status"] = "failed"
            record["exit_code"] = (
                record["exit_code"] if record["exit_code"] not in (0, None) else 74
            )
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    if relay_errors:
        return 74
    return 128 + stop_signal if stop_signal is not None else (process.returncode or 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileExistsError:
        print("Run ID already exists; choose a new one to preserve prior output", file=sys.stderr)
        sys.exit(2)
