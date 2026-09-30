"""Linux/WSL host-RAM admission for user-started CUDA experiment stages.

No file or accelerator is inspected on import. CPU tests pass fixture /proc
files. MemAvailable is a kernel estimate, not a reservation; immediate stage
checks reduce risk but cannot guarantee another process will not consume RAM.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _kib_field(text: str, key: str) -> int:
    matches = [
        line.split(":", 1)[1].strip().split()
        for line in text.splitlines()
        if line.startswith(key + ":")
    ]
    if (
        len(matches) != 1
        or len(matches[0]) != 2
        or matches[0][1] != "kB"
        or not matches[0][0].isdigit()
    ):
        raise RuntimeError(f"Linux host-RAM source has missing/invalid {key}")
    return int(matches[0][0]) * 1024


def linux_host_memory(proc_root: Path = Path("/proc")) -> dict:
    """Read current Linux host availability and this process's resident RAM."""
    proc_root = Path(proc_root)
    try:
        meminfo = (proc_root / "meminfo").read_text()
        status = (proc_root / "self/status").read_text()
    except OSError as error:
        raise RuntimeError("manual CUDA host-RAM admission requires Linux/WSL /proc") from error
    available = _kib_field(meminfo, "MemAvailable")
    total = _kib_field(meminfo, "MemTotal")
    rss = _kib_field(status, "VmRSS")
    if available > total:
        raise RuntimeError("Linux MemAvailable exceeds MemTotal")
    return {
        "host_available_bytes": available,
        "host_total_bytes": total,
        "process_rss_bytes": rss,
        "host_memory_source": "Linux/WSL /proc/meminfo MemAvailable and /proc/self/status VmRSS",
    }


def require_host_memory(
    stats: dict, *, floor_bytes: int, additional_bytes: int = 0, stage: str
) -> dict:
    if floor_bytes < 0 or additional_bytes < 0:
        raise ValueError("host-RAM admission bounds must be nonnegative")
    required = floor_bytes + additional_bytes
    if stats["host_available_bytes"] < required:
        raise RuntimeError(
            f"host available RAM below {stage} admission: "
            f"requires {required} bytes including safety floor"
        )
    return {
        **stats,
        "host_additional_stage_bytes": additional_bytes,
        "host_required_available_bytes": required,
        "host_admission_stage": stage,
    }


def model_storage_bytes(model, *, device_type: str | None = None) -> int:
    """Unique model parameter/buffer storage, without optimizer allocation."""
    storages = {}
    for tensor in [*model.parameters(), *model.buffers()]:
        if device_type is not None and tensor.device.type != device_type:
            continue
        storage = tensor.untyped_storage()
        storages[storage._cdata] = storage.nbytes()
    return sum(storages.values())


def lane_storage_bytes(lanes, *, device_type: str | None = None) -> int:
    """Unique tensor-storage size for drafter/Adam state, counting aliases once."""
    storages = {}
    for lane in lanes:
        tensors = [*lane.drafter.parameters(), *lane.drafter.buffers()]
        tensors.extend(
            value
            for state in lane.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        for tensor in tensors:
            if device_type is not None and tensor.device.type != device_type:
                continue
            storage = tensor.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
    return sum(storages.values())


def checkpoint_host_buffer_bytes(lanes) -> int:
    """Conservative CPU transfer/export working set beyond current host RSS.

    A row NPZ retains all copied arrays for one lane; GPU-to-CPU transfer and
    NumPy F32 copying can temporarily overlap. Budget two full lane exports
    plus the largest selected matrix and 16 MiB serialization workspace.
    This excludes host-resident models/moments, already reflected by /proc.
    """
    exports = []
    largest = 0
    for lane in lanes:
        total = 0
        for module in lane.linears.values():
            latent = module.latent_sign.numel() * 4
            scales = module.scale_offset.numel() * 4
            total += latent + scales
            largest = max(largest, latent)
        exports.append(total)
    return 2 * max(exports, default=0) + largest + 16 * 1024**2
