#!/usr/bin/env python3
"""Fixed read-only global /dev/dxg FD census. No model, GPU API or command execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from pathlib import Path


def census(proc_root=Path("/proc")):
    boot = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    holders = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            for fd in (directory / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if target == "/dev/dxg":
                    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                    holders.append(
                        {
                            "pid": int(directory.name),
                            "start_ticks": int(fields[19]),
                            "boot_id": boot,
                        }
                    )
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {"boot_id": boot, "holders": sorted(holders, key=lambda item: item["pid"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.expected_sha256 or os.geteuid() != 0:
        raise ValueError("hash-bound privileged read-only observer required")

    def deadline(*_):
        raise TimeoutError("global DXG observation exceeded ten seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(10)
    try:
        result = census()
        print(
            json.dumps(
                {
                    "schema": "nine_model_read_only_dxg_census_v1",
                    "complete": True,
                    "read_only": True,
                    "effective_uid": os.geteuid(),
                    "proc_root": "/proc",
                    "pid_namespace": os.readlink("/proc/self/ns/pid"),
                    "observer_source_sha256": digest,
                    **result,
                },
                sort_keys=True,
            )
        )
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"DXG census failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
