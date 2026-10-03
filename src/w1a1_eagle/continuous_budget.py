"""Crash-safe cumulative trainer time; unresolved attempts never refund work."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path


def boot_identity() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.is_file():
        return path.read_text().strip()
    # CPU fixtures have no Linux boot identity. Cross-process recovery falls
    # back to charging the full reservation, as on a changed remote boot.
    return f"{platform.node()}:{os.getpid()}"


def process_birth(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class TrainingBudget:
    """Write intent before work, then settle actual monotonic elapsed time.

    A killed attempt is charged through recovery time on the same Linux boot,
    including any uncertain downtime. After reboot its whole remaining reserved
    budget is consumed. Normal exits exclude standalone evaluation/startup.
    The run/GPU locks are still required; PID birth checks catch unsafe overlap.
    """

    def __init__(self, path: Path, source: str, maximum: float | None, atomic_write):
        self.path, self.source, self.maximum = Path(path), source, maximum
        self.write = atomic_write
        self.used = 0.0
        self.active = None

    def load(self, checkpoint_elapsed: float) -> float:
        self.used = checkpoint_elapsed
        if not self.path.exists():
            return self.used
        value = json.loads(self.path.read_text())
        if (
            value.get("schema") != "continuous_training_budget_v1"
            or value.get("source_sha256") != self.source
            or value.get("max_seconds") != self.maximum
        ):
            raise ValueError("persisted training budget contract differs")
        used = value.get("training_seconds")
        if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
            raise ValueError("persisted training budget is invalid")
        self.used = used
        active = value.get("active_attempt")
        if active is not None:
            if not isinstance(active, dict):
                raise ValueError("persisted training attempt is invalid")
            start, reserved = active.get("started_monotonic"), active.get("reserved_seconds")
            if (
                type(start) not in (int, float)
                or not math.isfinite(start)
                or start < 0
                or (
                    reserved is not None
                    and (
                        type(reserved) not in (int, float)
                        or not math.isfinite(reserved)
                        or reserved < 0
                    )
                )
            ):
                raise ValueError("persisted training attempt timing is invalid")
            same_boot = active.get("boot_id") == boot_identity()
            if same_boot and active.get("process_birth") is not None:
                if process_birth(active["pid"]) == active["process_birth"]:
                    raise ValueError("previous trainer budget owner is still alive")
            if same_boot:
                charged = max(0.0, time.monotonic() - start)
                if reserved is not None:
                    charged = min(charged, reserved)
            elif reserved is not None:
                charged = reserved
            else:
                raise ValueError("unbounded crashed training requires recoverable boot clock")
            self.used = max(checkpoint_elapsed, self.used + charged)
            self._persist(
                recovered_attempt={
                    **active,
                    "charged_seconds": charged,
                    "recovery_unix": time.time(),
                }
            )
        self.used = max(checkpoint_elapsed, self.used)
        return self.used

    def _persist(self, **extra):
        self.write(
            self.path,
            {
                "schema": "continuous_training_budget_v1",
                "source_sha256": self.source,
                "max_seconds": self.maximum,
                "training_seconds": self.used,
                "active_attempt": self.active,
                **extra,
            },
        )

    def begin(self, checkpoint_elapsed: float):
        self.load(checkpoint_elapsed)
        self.active = {
            "boot_id": boot_identity(),
            "pid": os.getpid(),
            "process_birth": process_birth(os.getpid()),
            "started_monotonic": time.monotonic(),
            "started_unix": time.time(),
            "reserved_seconds": None
            if self.maximum is None
            else max(0.0, self.maximum - self.used),
        }
        self._persist()

    def elapsed(self) -> float:
        if self.active is None:
            return self.used
        elapsed = max(0.0, time.monotonic() - self.active["started_monotonic"])
        reserved = self.active["reserved_seconds"]
        if reserved is not None:
            elapsed = min(elapsed, reserved)
        return self.used + elapsed

    def finish(self) -> float:
        self.used = self.elapsed()
        self.active = None
        self._persist()
        return self.used
