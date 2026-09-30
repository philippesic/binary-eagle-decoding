#!/usr/bin/env python3
"""Read-only CPU health check. No torch, GPU query, capture, evaluation or restart."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def check_health(status_path: Path, *, now: float | None = None,
                 stale_seconds: float = 1800, min_disk_bytes: int = 5 * 1024**3,
                 supervisor_path: Path | None = None, check_process: bool = False,
                 verify_checkpoint: bool = True) -> dict:
    now = time.time() if now is None else now
    failures = []
    status_path = status_path.resolve()
    try:
        status = json.loads(status_path.read_text())
    except (OSError, ValueError) as exc:
        return {"healthy": False, "terminal": False, "failures": [f"status unreadable: {exc}"]}
    if not isinstance(status, dict):
        return {"healthy": False, "terminal": False, "failures": ["status must be an object"]}
    state = status.get("status")
    terminal = state in ("stopped", "completed", "failed")
    if state == "failed":
        failures.append(f"training failed: {status.get('error', 'see training log')}")
    elif state not in ("running", "smoke", "stopped", "completed"):
        failures.append(f"unknown training status: {state}")
    heartbeat = status.get("heartbeat_unix")
    if not terminal and (not isinstance(heartbeat, (int, float)) or
                         not math.isfinite(heartbeat) or now - heartbeat > stale_seconds or
                         heartbeat > now + 60):
        failures.append("training heartbeat missing, stale or invalid")
    models = status.get("models", {})
    if not isinstance(models, dict):
        failures.append("models status must be an object")
        models = {}
    steps = []
    for name in ("A8", "A1"):
        model = models.get(name)
        if not isinstance(model, dict):
            failures.append(f"{name} status missing")
            continue
        step = model.get("step")
        if not isinstance(step, int) or step < 0:
            failures.append(f"{name} step invalid")
        else:
            steps.append(step)
        if step:
            loss = model.get("loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                failures.append(f"{name} loss nonfinite or missing")
            if model.get("grad_finite") is not True:
                failures.append(f"{name} gradients not finite")
        hb = model.get("heartbeat_unix")
        if not terminal and (not isinstance(hb, (int, float)) or not math.isfinite(hb) or
                             now - hb > stale_seconds or hb > now + 60):
            failures.append(f"{name} heartbeat stale or invalid")
    if len(steps) == 2 and abs(steps[0] - steps[1]) > 1:
        failures.append("A8/A1 step imbalance exceeds one interleaved minibatch")
    disk = status.get("disk_free_bytes")
    if not isinstance(disk, (int, float)) or not math.isfinite(disk) or disk < min_disk_bytes:
        failures.append("disk headroom missing or below health threshold")
    # Snapshot metrics, never query the device from this checker.
    for key in ("cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_bytes"):
        value = status.get(key)
        if value is not None and (not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or value < 0):
            failures.append(f"{key} invalid")
    checkpoint = status.get("checkpoint")
    if checkpoint is not None:
        try:
            path = Path(checkpoint["path"])
            path = path if path.is_absolute() else status_path.parent / path
            path = path.resolve()
            if not path.is_relative_to(status_path.parent):
                raise ValueError("checkpoint outside experiment directory")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("checkpoint missing or empty")
            if verify_checkpoint:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != checkpoint["sha256"]:
                    raise ValueError("checkpoint hash mismatch")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            failures.append(f"checkpoint unhealthy: {exc}")
    elif terminal and any(steps):
        failures.append("terminal run has no checkpoint")
    if supervisor_path is not None:
        try:
            supervisor = json.loads(supervisor_path.read_text())
            if supervisor.get("status") in ("finished", "interrupted") and not terminal:
                failures.append("supervisor terminal but training status is nonterminal")
            if supervisor.get("status") == "finished" and supervisor.get("exit_code") != 0:
                failures.append(f"supervisor exited {supervisor.get('exit_code')}")
            if check_process and not terminal:
                pid = supervisor.get("pid")
                if not isinstance(pid, int) or pid <= 1:
                    raise ValueError("supervisor process id missing or invalid")
                os.kill(pid, 0)
        except (OSError, ValueError) as exc:
            failures.append(f"supervisor/process unhealthy: {exc}")
    return {"healthy": not failures, "terminal": terminal, "status": state,
            "failures": failures, "steps": dict(zip(("A8", "A1"), steps)),
            "checkpoint": checkpoint, "models": models,
            "heartbeat_unix": heartbeat, "disk_free_bytes": disk}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("status", type=Path)
    p.add_argument("--supervisor", type=Path)
    p.add_argument("--check-process", action="store_true",
                   help="only on the same host as training; never for copied snapshots")
    p.add_argument("--stale-seconds", type=float, default=1800)
    p.add_argument("--min-disk-gib", type=float, default=5)
    args = p.parse_args()
    if args.stale_seconds <= 0 or args.min_disk_gib < 0:
        p.error("positive stale threshold and nonnegative disk threshold required")
    result = check_health(args.status, supervisor_path=args.supervisor,
                          check_process=args.check_process,
                          stale_seconds=args.stale_seconds,
                          min_disk_bytes=int(args.min_disk_gib * 1024**3))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["healthy"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
