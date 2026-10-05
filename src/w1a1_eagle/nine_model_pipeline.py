"""Fail-closed, out-of-process nine-model campaign lifecycle.

Import and bundle validation are CPU-only. Real launch needs a fresh explicit
5080 lease, a source-bound admission producer, and Linux resource observation.
Synthetic execution is a test API; its receipts never grant production readiness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

FAMILIES = ("eagle", "dspark", "dflash")
CANDIDATES = tuple(f"{f}_{p}" for f in FAMILIES for p in ("a8", "a1"))
CELLS = tuple(f"{f}_{p}" for f in FAMILIES for p in ("q4", "a8", "a1"))
GATES = {"resources", "kernel", "model", "backward", "memory", "capture_portability"}
ENV_PREFIXES = ("GGML_", "W1AX_", "EAGLE_", "DSPARK_", "DFLASH_", "TORCH_", "PYTORCH_")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_opaque_prompts(path):
    prompts = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(prompts, "prompt file empty")
    identifiers = set()
    for row in prompts:
        identifier = row.get("id")
        require(
            isinstance(identifier, str) and identifier and identifier not in identifiers,
            "opaque prompt ID missing or duplicate",
        )
        identifiers.add(identifier)
        messages = row.get("messages")
        require(isinstance(messages, list) and messages, "nonempty messages required")
        require(
            all(
                isinstance(m, dict)
                and m.get("role") in {"system", "user", "assistant"}
                and isinstance(m.get("content"), str)
                for m in messages
            ),
            "text message schema differs",
        )
    return prompts


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Files:
    """Hash once per in-process inode identity; never persist stat-only trust."""

    def __init__(self):
        self.cache = set()

    def opaque(self, record):
        require(
            isinstance(record, dict) and set(record) == {"path", "sha256"},
            "exact path/SHA256 locator required",
        )
        path = Path(record["path"])
        require(
            path.is_absolute() and path == path.resolve() and path.is_file(),
            "canonical regular artifact required",
        )
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "invalid SHA256")
        return path

    def check(self, record):
        path = self.opaque(record)
        stat = path.stat()
        key = (
            str(path),
            record["sha256"],
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if key not in self.cache:
            require(sha256(path) == record["sha256"], f"artifact changed: {path}")
            require(path.stat() == stat, f"artifact changed during hash: {path}")
            self.cache.add(key)
        return path


def clean_environment(declared=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(ENV_PREFIXES)
        and k
        not in {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONPATH",
        }
    }
    declared = declared or {}
    require(
        set(declared)
        <= {
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "OMP_NUM_THREADS",
            "PYTHONPATH",
        },
        "undeclared experimental environment",
    )
    env.update(declared)
    return env


def native_environment(family, activation_bits, declared=None):
    env = clean_environment(declared)
    require(family in FAMILIES or family == "target_only", "unknown native family")
    require(
        activation_bits is None or (type(activation_bits) is int and activation_bits in {1, 8}),
        "unsupported deployed activation arithmetic",
    )
    if family == "eagle" and activation_bits is not None:
        env["GGML_W1AX_ACT_BITS"] = str(activation_bits)
    return env


def validate_cuda_dispatch(text, audit, *, activation_bits):
    """Exact named integer/XOR kernel launches, parsed only outside clean timing."""
    records = []
    marker = "W1AX_ADMISSION_TRACE "
    for line in text.splitlines():
        if marker in line:
            records.append(json.loads(line.split(marker, 1)[1]))
    expected = {
        base + ".w1a1_packed": entry["shape"] for base, entry in audit["projections"].items()
    }
    matched = set()
    for record in records:
        packed = record.get("packed")
        if packed not in expected:
            continue
        rows, logical_k = expected[packed]
        require(
            record.get("schema") == "w1ax_cuda_dispatch_v1"
            and record.get("backend") == "CUDA"
            and record.get("device") == 0
            and record.get("activation_bits") == activation_bits
            and record.get("logical_k") == logical_k
            and record.get("rows") == rows
            and record.get("packed_type") == "i32"
            and record.get("packed_words") == (logical_k + 31) // 32
            and type(record.get("tokens")) is int
            and record["tokens"] > 0,
            "observed named W1 CUDA operation/shape/arithmetic differs",
        )
        matched.add(packed)
    require(matched == set(expected), "complete per-projection W1 CUDA execution evidence absent")
    require(
        "CUDA error" not in text and "dense fallback" not in text.lower(),
        "native CUDA error/dense fallback observed",
    )
    return {
        "schema": "nine_model_observed_w1_dispatch_v1",
        "status": "PASS",
        "activation_bits": activation_bits,
        "packed_names": sorted(matched),
        "selected_projection_count": len(matched),
        "actual_cuda_operations": True,
    }


def validate_bundle(path):
    files = Files()
    bundle = json.loads(Path(path).read_text())
    require(bundle.get("schema") == "nine_model_campaign_bundle_v1", "bundle schema differs")
    require(bundle.get("artifact_kind") == "production", "fixture cannot grant readiness")
    require(bundle.get("preparation_complete") is True, "preparation dependencies PENDING")
    require(set(bundle.get("candidates", {})) == set(CANDIDATES), "six candidates required")
    require(set(bundle.get("controls", {})) == set(FAMILIES), "three frozen Q4 controls required")
    require(
        bundle.get("target_policy") == {"immutable": True, "weights": "f16", "kv": "f16"},
        "immutable common target/F16 KV policy required",
    )
    require(
        {"protocol", "prompts"} <= set(bundle.get("inputs", {})),
        "protocol and opaque prompt locator missing",
    )
    protocol = json.loads(files.check(bundle["inputs"]["protocol"]).read_text())
    require(protocol.get("split") in {"development", "final"}, "explicit evaluation split required")
    for name, record in bundle.get("inputs", {}).items():
        if name == "prompts" and protocol["split"] == "final":
            files.opaque(record)
            continue  # Sealed bytes first accessed in fresh evaluator after freeze.
        files.check(record)
    require(
        {"target", "binary", "protocol", "prompts", "budget"} <= set(bundle.get("inputs", {})),
        "native runtime/protocol/prompt binding missing",
    )
    for records in bundle.get("source", {}).values():
        files.check(records)
    require(bundle.get("source"), "source inventory missing")
    require(set(bundle.get("fresh_gates", [])) == GATES, "fresh gate inventory differs")
    for family, record in bundle["controls"].items():
        require(record.get("frozen_original") is True, f"original Q4 control required: {family}")
        files.check(record["model"])
    for name, candidate in bundle["candidates"].items():
        require(
            candidate.get("profile") in {"fixed_reference", "direct_a1", "a8_to_a1_reset"},
            f"unsupported profile: {name}",
        )
        files.check(candidate["config"])
        require(
            set(candidate.get("stages", {})) == {"train", "export"},
            "candidate needs actual train/export commands",
        )
        require(
            candidate.get("native_markers")
            and all(isinstance(marker, str) and marker for marker in candidate["native_markers"]),
            "native loader/dispatch marker contract required",
        )
        expected = {"train": "train_nine_model_qat.py", "export": "export_nine_model_candidate.py"}
        for key, producer in expected.items():
            require(
                Path(candidate["stages"][key]["producer"]["path"]).name == producer,
                "unbound generic stage producer prohibited",
            )
        require(
            candidate["config"]["path"] in candidate["stages"]["train"]["argv"],
            "actual training config not passed to producer",
        )
    require(
        Path(bundle["admission"]["producer"]["path"]).name == "admit_nine_model_sm120.py"
        and Path(bundle["evaluation"]["producer"]["path"]).name == "evaluate_nine_model_native.py",
        "actual admission/evaluator producer required",
    )
    for stage in [
        bundle["admission"],
        bundle["evaluation"],
        *(s for c in bundle["candidates"].values() for s in c["stages"].values()),
    ]:
        validate_stage(stage, files)
        require(
            stage["producer"] in bundle["source"].values(),
            "actual stage producer does not join frozen source inventory",
        )
    for name, value in bundle["resource_policy"].items():
        require(type(value) is int and value >= 0, f"invalid resource policy: {name}")
    require(
        set(bundle["resource_policy"])
        == {
            "host_floor_bytes",
            "gpu_floor_bytes",
            "host_return_tolerance_bytes",
            "gpu_return_tolerance_bytes",
        },
        "exact resource floors/return policy required",
    )
    return bundle, files


def validate_stage(stage, files):
    require(
        isinstance(stage.get("argv"), list)
        and stage["argv"]
        and all(isinstance(a, str) and a for a in stage["argv"]),
        "argv list required",
    )
    require(
        type(stage.get("wall_seconds")) in (float, int)
        and math.isfinite(stage["wall_seconds"])
        and stage["wall_seconds"] > 0,
        "finite positive stage wall cap required",
    )
    files.check(stage["producer"])
    require(
        len(stage["argv"]) >= 2
        and stage["argv"][0] == sys.executable
        and stage["argv"][1] == stage["producer"]["path"],
        "command must execute the pinned producer with the current admitted Python",
    )


def require_available(path, bundle_hash, *, now=None):
    """Stale unpaused flags are insufficient: user-announced short lease required."""
    record = json.loads(Path(path).read_text())
    now = time.time() if now is None else now
    require(
        record.get("schema") == "nine_model_gpu_lease_v1"
        and record.get("host") == "rtx5080"
        and record.get("user_announced_available") is True
        and record.get("sole_owner") is True
        and record.get("pause_requested") is False
        and record.get("bundle_sha256") == bundle_hash,
        "exclusive 5080 availability lease absent",
    )
    require(
        type(record.get("granted_unix")) in (int, float)
        and type(record.get("expires_unix")) in (int, float)
        and record["granted_unix"] <= now < record["expires_unix"] <= record["granted_unix"] + 300,
        "5080 availability lease stale",
    )
    require(
        isinstance(record.get("gpu_uuid"), str)
        and re.fullmatch(r"GPU-[A-Za-z0-9-]+", record["gpu_uuid"]) is not None,
        "canonical GPU UUID absent",
    )
    return record


def process_identity(pid, proc_root=Path("/proc")):
    fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": int(pid),
        "start_ticks": int(fields[19]),
        "boot_id": (proc_root / "sys/kernel/random/boot_id").read_text().strip(),
    }


def descendant_identities(root_pid, proc_root=Path("/proc")):
    parents = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(directory.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    owned = {root_pid}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    identities = []
    for pid in sorted(owned):
        try:
            identities.append(process_identity(pid, proc_root))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return identities


def identity_active(identity):
    try:
        if process_identity(identity["pid"]) != identity:
            return False
        fields = (
            (Path("/proc") / str(identity["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        )
        return fields[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def cleanup_descendants(identities):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for identity in identities:
            if identity_active(identity):
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3
        while any(identity_active(i) for i in identities) and time.monotonic() < deadline:
            time.sleep(0.05)
    require(not any(identity_active(i) for i in identities), "owned descendant survived cleanup")


def _scan_dxg_holders(proc_root):
    """WSL device holders with kernel identities, including unlisted contexts."""
    holders = []
    require(proc_root.is_dir(), "Linux /proc context observer required")
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fds = directory / "fd"
            for fd in fds.iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target == "/dev/dxg":
                    holders.append(process_identity(int(directory.name), proc_root))
                    break
        except FileNotFoundError:
            continue
    return sorted(holders, key=lambda x: x["pid"])


def _privileged_dxg_holders():
    """One fixed, hash-bound, stdlib-only read observer; never elevate a trainer."""
    helper = Path(__file__).resolve().parents[2] / "scripts/read_only_dxg_census.py"
    expected = sha256(helper)
    before = helper.stat()
    try:
        require(
            os.environ.get("WSL_DISTRO_NAME") == "Ubuntu",
            "fixed Ubuntu observer requires current Ubuntu namespace",
        )
        environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
        if os.environ.get("WSL_INTEROP"):
            environment["WSL_INTEROP"] = os.environ["WSL_INTEROP"]
        result = subprocess.run(
            [
                "/mnt/c/Windows/System32/wsl.exe",
                "-d",
                "Ubuntu",
                "-u",
                "root",
                "--",
                "/usr/bin/python3",
                "-B",
                "-I",
                str(helper),
                "--expected-sha256",
                expected,
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            cwd="/",
            env=environment,
        )
        require(
            helper.stat() == before and sha256(helper) == expected, "DXG observer source changed"
        )
        record = json.loads(result.stdout)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        require(
            record.get("schema") == "nine_model_read_only_dxg_census_v1"
            and record.get("complete") is True
            and record.get("read_only") is True
            and record.get("effective_uid") == 0
            and record.get("proc_root") == "/proc"
            and record.get("observer_source_sha256") == expected
            and record.get("boot_id") == boot
            and record.get("pid_namespace") == os.readlink("/proc/self/ns/pid")
            and isinstance(record.get("holders"), list),
            "privileged DXG census proof unavailable; release PENDING",
        )
        holders = record["holders"]
        require(
            all(
                isinstance(item, dict)
                and set(item) == {"pid", "start_ticks", "boot_id"}
                and type(item["pid"]) is int
                and item["pid"] > 0
                and type(item["start_ticks"]) is int
                and item["start_ticks"] >= 0
                and item["boot_id"] == boot
                for item in holders
            )
            and len({item["pid"] for item in holders}) == len(holders),
            "privileged DXG holder identities invalid; release PENDING",
        )
        return sorted(holders, key=lambda x: x["pid"])
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as error:
        raise ValueError("WSL /dev/dxg privileged census unavailable; release PENDING") from error


def dxg_holders(proc_root=Path("/proc")):
    """Complete global device census; permission gaps cannot masquerade as idle."""
    try:
        return _scan_dxg_holders(proc_root)
    except PermissionError as error:
        if proc_root != Path("/proc"):
            raise ValueError("WSL /dev/dxg holder census denied; release PENDING") from error
        return _privileged_dxg_holders()


def group_members(pgid, proc_root=Path("/proc")):
    require(proc_root.is_dir(), "Linux /proc ownership observation required")
    members = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(proc, *, grace=3):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            # Leader may exit before a descendant. Always kill remaining group.
            continue
    proc.wait(timeout=grace)
    if Path("/proc").is_dir():
        deadline = time.monotonic() + grace
        while group_members(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        require(not group_members(proc.pid), "owned process group survived cleanup")


@contextmanager
def deferred_termination():
    """Complete a bounded cleanup/receipt write before delivering termination."""
    mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@contextmanager
def stop_signals():
    previous = {}
    requested = False

    def interrupted(signum, _frame):
        nonlocal requested
        if requested:
            return
        requested = True
        raise InterruptedError(f"campaign signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class SubprocessRunner:
    def __init__(self, root, environment=None, authorization=None):
        self.root = Path(root)
        self.environment = clean_environment(environment)
        self.authorization = authorization
        self.process_groups = []
        self.process_identities = []

    def run(self, argv, *, directory, stop_path, wall_seconds):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        proc = None
        start = time.monotonic()
        with (directory / "stdout.log").open("ab") as log:
            try:
                mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
                try:
                    proc = subprocess.Popen(
                        argv,
                        cwd=self.root,
                        env=self.environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask),
                    )
                    self.process_groups.append(proc.pid)
                    if Path("/proc").is_dir():
                        self.process_identities.append(process_identity(proc.pid))
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                atomic_json(
                    directory / "process.json",
                    {
                        "pid": proc.pid,
                        "pgid": proc.pid,
                        "argv": argv,
                        "started_unix": time.time(),
                        "kernel_identity": self.process_identities[-1]
                        if self.process_identities
                        else None,
                    },
                )
                while proc.poll() is None:
                    require(
                        (directory / "stdout.log").stat().st_size <= 128 * 1024**2,
                        "stage stdout safety cap reached; existing checkpoints preserved",
                    )
                    if Path("/proc").is_dir():
                        discovered = descendant_identities(proc.pid)
                        additions = [i for i in discovered if i not in self.process_identities]
                        if additions:
                            self.process_identities.extend(additions)
                            atomic_json(
                                directory / "process-lineage.json",
                                {"kernel_identities": self.process_identities},
                            )
                    if self.authorization is not None:
                        self.authorization()
                    if Path(stop_path).exists():
                        raise InterruptedError("campaign STOP requested")
                    if time.monotonic() - start >= wall_seconds:
                        raise TimeoutError("stage wall cap reached; checkpoint retained")
                    time.sleep(0.1)
                require(proc.returncode == 0, f"stage exit code {proc.returncode}; see {directory}")
            finally:
                if proc is not None:
                    original = sys.exception()
                    errors = []
                    with deferred_termination():
                        try:
                            stop_owned(proc, grace=30)
                        except BaseException as error:
                            errors.append(str(error))
                        finally:
                            if Path("/proc").is_dir():
                                try:
                                    cleanup_descendants(self.process_identities)
                                except BaseException as error:
                                    errors.append(str(error))
                        atomic_json(
                            directory / "process-exit.json",
                            {
                                "pid": proc.pid,
                                "exit_code": proc.returncode,
                                "owned_group_cleaned": Path("/proc").is_dir() and not errors,
                                "cleanup_status": "FAILED" if errors else "PASS",
                                "cleanup_errors": errors,
                            },
                        )
                    if errors and original is None:
                        raise RuntimeError("owned cleanup failed: " + "; ".join(errors))


class LinuxResources:
    """Whole-device free memory and process checks, not compute-app emptiness alone."""

    def __init__(self, gpu_uuid, *, hardware="rtx5080"):
        require(hardware in {"rtx5080", "rtx2080ti"}, "unsupported resource observer hardware")
        self.gpu_uuid, self.hardware = gpu_uuid, hardware

    def require_released(self, pgids, identities=()):
        for identity in identities:
            try:
                require(
                    not identity_active(identity),
                    f"owned kernel process still active: {identity['pid']}",
                )
            except FileNotFoundError:
                pass
        active_owned = [identity for identity in identities if identity_active(identity)]
        require(not active_owned, "owned kernel process remains active")
        # A exited leader can leave its original process group alive. A group
        # with a new leader identity belongs to another owner and is ignored.
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for pgid in pgids:
            origin = next((i for i in identities if i["pid"] == pgid), None)
            if origin is None or origin["boot_id"] != boot:
                continue
            try:
                current = process_identity(pgid)
            except FileNotFoundError:
                current = None
            if current is not None and current != origin:
                continue
            require(not group_members(pgid), "original owned process group remains active")
        utility = "/usr/lib/wsl/lib/nvidia-smi"
        if not Path(utility).is_file():
            utility = "nvidia-smi"
        query = subprocess.run(
            [
                utility,
                "-i",
                self.gpu_uuid,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        require(
            all(line.isdigit() for line in lines),
            "CUDA context PID census unavailable; resource release PENDING",
        )
        active = {int(line) for line in lines}
        require(
            not active.intersection({i["pid"] for i in active_owned}),
            "owned CUDA context still active",
        )
        return {
            "owned_process_groups_absent": True,
            "owned_cuda_pids_absent": True,
            "context_observer": "nvidia-smi compute-app PID census plus /proc and memory return",
            "other_context_pids": sorted(active),
        }

    def snapshot(self):
        mem = Path("/proc/meminfo").read_text()
        matches = re.findall(r"^MemAvailable:\s+(\d+) kB$", mem, re.MULTILINE)
        require(len(matches) == 1, "Linux MemAvailable missing")
        utility = "/usr/lib/wsl/lib/nvidia-smi"
        if not Path(utility).is_file():
            utility = "nvidia-smi"
        data = (
            subprocess.run(
                [
                    utility,
                    "-i",
                    self.gpu_uuid,
                    "--query-gpu=uuid,name,compute_cap,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            .stdout.strip()
            .split(",")
        )
        require(len(data) == 5 and data[0].strip() == self.gpu_uuid, "GPU UUID differs")
        expected = (
            ("12.0", "5080", [12, 0]) if self.hardware == "rtx5080" else ("7.5", "2080", [7, 5])
        )
        require(
            data[2].strip() == expected[0] and expected[1] in data[1],
            "actual declared CUDA hardware required",
        )
        return {
            "host_available_bytes": int(matches[0]) * 1024,
            "gpu_free_bytes": int(data[3].strip()) * 1024**2,
            "gpu_total_bytes": int(data[4].strip()) * 1024**2,
            "gpu_uuid": self.gpu_uuid,
            "hardware": data[1].strip(),
            "compute_capability": expected[2],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "dxg_holders": dxg_holders() if Path("/dev/dxg").exists() else [],
        }


class DiagnosticMemorySampler:
    """Diagnostic-only sampled maxima; never an allocator high-water claim.

    Per-process RSS is retained separately because shared mappings can overlap.
    This object must be stopped before the next native process launch.
    """

    def __init__(self, resources, root_pid, *, interval_seconds=1.0, max_samples=3600):
        require(
            math.isfinite(interval_seconds)
            and interval_seconds > 0
            and type(max_samples) is int
            and max_samples > 0,
            "bounded telemetry required",
        )
        self.resources, self.root_pid = resources, root_pid
        self.interval_seconds, self.max_samples = interval_seconds, max_samples
        self.done = threading.Event()
        self.thread = None
        self.samples = 0
        self.error = None
        self.gpu_used_max = 0
        self.host_available_min = None
        self.per_process = {}
        self.sample_start_times = []
        self.sample_durations = []

    def sample(self):
        started = time.monotonic()
        snapshot = self.resources.snapshot()
        self.gpu_used_max = max(
            self.gpu_used_max, snapshot["gpu_total_bytes"] - snapshot["gpu_free_bytes"]
        )
        available = snapshot["host_available_bytes"]
        self.host_available_min = (
            available
            if self.host_available_min is None
            else min(self.host_available_min, available)
        )
        for identity in descendant_identities(self.root_pid):
            try:
                text = (Path("/proc") / str(identity["pid"]) / "status").read_text()
            except FileNotFoundError:
                continue
            matches = re.findall(r"^VmRSS:\s+(\d+) kB$", text, re.MULTILINE)
            if not matches:
                continue
            key = json.dumps(identity, sort_keys=True)
            previous = self.per_process.get(key, {"kernel_identity": identity, "rss_max_bytes": 0})
            previous["rss_max_bytes"] = max(previous["rss_max_bytes"], int(matches[0]) * 1024)
            self.per_process[key] = previous
        self.sample_start_times.append(started)
        self.sample_durations.append(time.monotonic() - started)
        self.samples += 1

    def _loop(self):
        try:
            while not self.done.is_set() and self.samples < self.max_samples:
                self.sample()
                self.done.wait(self.interval_seconds)
        except BaseException as error:
            self.error = str(error)

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="diagnostic-memory", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
            require(not self.thread.is_alive(), "diagnostic telemetry context did not stop")
        require(self.error is None, "diagnostic telemetry failed: " + str(self.error))
        require(self.samples > 0, "diagnostic telemetry samples absent")
        gaps = [
            later - earlier
            for earlier, later in zip(self.sample_start_times, self.sample_start_times[1:])
        ]
        return {
            "schema": "nine_model_sampled_memory_v1",
            "diagnostic_only": True,
            "clean_timing_instrumented": False,
            "samples": self.samples,
            "requested_interval_seconds": self.interval_seconds,
            "max_samples": self.max_samples,
            "sample_limit_reached": self.samples >= self.max_samples,
            "observed_start_gap_seconds": {
                "count": len(gaps),
                "mean": sum(gaps) / len(gaps) if gaps else None,
                "min": min(gaps) if gaps else None,
                "max": max(gaps) if gaps else None,
            },
            "longest_observer_duration_seconds": max(self.sample_durations),
            "whole_device_used_max_bytes": self.gpu_used_max,
            "system_host_memavailable_min_bytes": self.host_available_min,
            "evaluator_and_descendant_process_rss_maxima": list(self.per_process.values()),
            "scope": "sampled maxima are lower bounds, not true allocator peaks; whole-device "
            "memory includes baseline/external use; RSS includes evaluator, native server and "
            "transient utility descendants, may overlap and is "
            "not summed; Torch training allocated/reserved peaks reported separately",
        }


def resource_gate(observed, baseline, policy):
    for name, metric in (("host", "host_available_bytes"), ("gpu", "gpu_free_bytes")):
        required = max(
            policy[f"{name}_floor_bytes"],
            baseline[metric] - policy[f"{name}_return_tolerance_bytes"],
        )
        require(observed[metric] >= required, f"{name} resource return below {required} bytes")
    require(observed["gpu_uuid"] == baseline["gpu_uuid"], "resource GPU changed")
    baseline_contexts = {json.dumps(x, sort_keys=True) for x in baseline.get("dxg_holders", [])}
    observed_contexts = {json.dumps(x, sort_keys=True) for x in observed.get("dxg_holders", [])}
    require(
        observed_contexts <= baseline_contexts,
        "new /dev/dxg context holder remains; resource release PENDING",
    )


class Campaign:
    def __init__(
        self,
        bundle,
        bundle_hash,
        files,
        root,
        run_dir,
        runner,
        resources,
        *,
        fixture=False,
        authorization=None,
    ):
        self.bundle, self.bundle_hash, self.files = bundle, bundle_hash, files
        self.root, self.run = Path(root).resolve(), Path(run_dir).resolve()
        self.runner, self.resources = runner, resources
        self.fixture, self.authorization = fixture, authorization
        self.state = {
            "schema": "nine_model_campaign_state_v1",
            "bundle_sha256": bundle_hash,
            "artifact_kind": "fixture" if fixture else "production",
            "completed": {},
        }

    def guard(self):
        if (self.run / "STOP").exists():
            raise InterruptedError("campaign STOP requested")
        if not self.fixture:
            require(self.authorization is not None, "production authorization observer absent")
            self.authorization()
        for record in self.bundle["source"].values():
            self.files.check(record)
        protocol = (
            json.loads(self.files.check(self.bundle["inputs"]["protocol"]).read_text())
            if "protocol" in self.bundle["inputs"]
            else {"split": "development"}
        )
        for name, record in self.bundle["inputs"].items():
            if name == "prompts" and protocol["split"] == "final":
                self.files.opaque(record)
                continue
            self.files.check(record)
        for control in self.bundle["controls"].values():
            self.files.check(control["model"])
        for candidate in self.bundle["candidates"].values():
            if "config" in candidate:
                self.files.check(candidate["config"])

    def publish(self, **changes):
        self.state.update(changes)
        atomic_json(self.run / "state.json", self.state)

    def stage(self, name, spec, *, resume=False, values=None):
        self.guard()
        output = self.run / name / "attempts" / uuid.uuid4().hex / "receipt.json"
        values = dict(
            values or {},
            bundle_sha256=self.bundle_hash,
            run_dir=str(self.run),
            receipt=str(output),
            stage_dir=str(output.parent),
            resume="--resume" if resume else "",
        )
        argv = [a.format_map(values) for a in spec["argv"]]
        argv = [a for a in argv if a]
        self.files.check(spec["producer"])
        self.publish(status="running", stage=name)
        self.runner.run(
            argv,
            directory=output.parent,
            stop_path=self.run / "STOP",
            wall_seconds=spec["wall_seconds"],
        )
        receipt = json.loads(output.read_text())
        require(
            receipt.get("schema") == "nine_model_stage_receipt_v1"
            and receipt.get("bundle_sha256") == self.bundle_hash
            and receipt.get("stage") == name
            and receipt.get("status") == "PASS",
            f"stage receipt invalid: {name}",
        )
        require(
            receipt.get("artifact_kind") == self.state["artifact_kind"],
            "fixture evidence cannot grant production readiness",
        )
        locator = {"path": str(output), "sha256": sha256(output)}
        self.state["completed"][name] = locator
        self.publish(status="stage_complete")
        return receipt

    def recover_training_receipt(self, name, config):
        recovered = []
        for path in (self.run / name / "attempts").glob("*/receipt.json"):
            receipt = json.loads(path.read_text())
            if (
                receipt.get("schema") == "nine_model_stage_receipt_v1"
                and receipt.get("bundle_sha256") == self.bundle_hash
                and receipt.get("stage") == name
                and receipt.get("status") == "PASS"
                and receipt.get("artifact_kind") == self.state["artifact_kind"]
                and receipt.get("committed") is True
                and receipt.get("completion_reason") == "approved_budget_complete"
            ):
                if not self.fixture:
                    require(
                        receipt.get("config_sha256") == config["sha256"],
                        "orphan completion config differs",
                    )
                self.files.check(receipt["checkpoint"])
                recovered.append((path, receipt))
        if not recovered:
            return
        require(
            len({r["checkpoint"]["sha256"] for _, r in recovered}) == 1,
            "multiple conflicting committed training endpoints require reconciliation",
        )
        path, _ = recovered[-1]
        self.state["completed"][name] = {"path": str(path), "sha256": sha256(path)}
        self.publish(recovered_committed_training_receipt=name)

    def returned(self, baseline):
        ownership = self.resources.require_released(
            self.runner.process_groups, self.runner.process_identities
        )
        observed = self.resources.snapshot()
        resource_gate(observed, baseline, self.bundle["resource_policy"])
        self.publish(last_resource_return=observed, owned_release=ownership)
        return observed

    def execute(self, *, resume=False):
        self.guard()
        self.run.mkdir(parents=True, exist_ok=resume)
        lock = self.run / ".owner.lock"
        import fcntl

        with lock.open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                old = json.loads((self.run / "state.json").read_text())
                require(
                    old["bundle_sha256"] == self.bundle_hash
                    and old["artifact_kind"] == self.state["artifact_kind"],
                    "resume bundle differs",
                )
                self.state = old
                for record in self.state["completed"].values():
                    self.files.check(record)
            try:
                with stop_signals():
                    observed = self.resources.snapshot()
                    baseline = self.state.get("resource_baseline", observed)
                    reboot = observed.get("boot_id") != baseline.get("boot_id")
                    if not reboot:
                        resource_gate(observed, baseline, self.bundle["resource_policy"])
                    self.publish(resource_baseline=baseline)
                    if resume:
                        for process_path in self.run.glob("**/process.json"):
                            process = json.loads(process_path.read_text())
                            identity = process.get("kernel_identity")
                            require(identity is not None, "prior process kernel identity absent")
                            self.runner.process_groups.append(process["pgid"])
                            self.runner.process_identities.append(identity)
                        for lineage_path in self.run.glob("**/process-lineage.json"):
                            self.runner.process_identities.extend(
                                json.loads(lineage_path.read_text())["kernel_identities"]
                            )
                        if reboot:
                            self.resources.require_released(
                                self.runner.process_groups, self.runner.process_identities
                            )
                            resource_gate(observed, observed, self.bundle["resource_policy"])
                            self.publish(prior_boot_baseline=baseline, resource_baseline=observed)
                            baseline = observed
                        self.returned(baseline)
                    admission = self.stage("admission", self.bundle["admission"])
                    require(
                        admission.get("gates") == dict.fromkeys(sorted(GATES), "PASS")
                        and admission.get("optimizer_updates") == 0
                        and admission.get("compute_capability") == [12, 0],
                        "fresh SM120 kernel/model/backward/memory/capture gates incomplete",
                    )
                    self.returned(baseline)
                    exports = {}
                    for candidate in CANDIDATES:
                        spec = self.bundle["candidates"][candidate]
                        train_name = candidate + "/train"
                        if resume and train_name not in self.state["completed"]:
                            self.recover_training_receipt(train_name, spec.get("config"))
                        if train_name in self.state["completed"]:
                            train = json.loads(
                                self.files.check(self.state["completed"][train_name]).read_text()
                            )
                        else:
                            selected = admission.get("candidate_admissions", {}).get(candidate)
                            if not self.fixture:
                                path = self.files.check(selected)
                                record = json.loads(path.read_text())
                                require(
                                    record.get("schema") == "nine_model_training_admission_v1"
                                    and record.get("bundle_sha256") == self.bundle_hash
                                    and record.get("config_sha256") == spec["config"]["sha256"]
                                    and record.get("status") == "PASS",
                                    "candidate training admission/config binding differs",
                                )
                            train = self.stage(
                                train_name,
                                spec["stages"]["train"],
                                resume=resume,
                                values={
                                    "admission": selected["path"]
                                    if selected
                                    else "fixture-not-production-admission"
                                },
                            )
                        require(
                            train.get("committed") is True
                            and train.get("completion_reason") == "approved_budget_complete",
                            "training did not reach successful committed budget endpoint",
                        )
                        require(
                            type(train.get("counters", {}).get("step")) is int
                            and train["counters"]["step"] > 0,
                            "zero-update endpoint cannot grant trained-candidate evaluation",
                        )
                        checkpoint = self.files.check(train["checkpoint"])
                        self.returned(baseline)
                        export = self.stage(
                            candidate + "/export",
                            spec["stages"]["export"],
                            values={
                                "checkpoint": str(checkpoint),
                                "train_receipt": self.state["completed"][train_name]["path"],
                                "candidate": candidate,
                            },
                        )
                        require(
                            export.get("serialization_audit_passed") is True
                            and export.get("selected_weights_packed") is True,
                            "native export contract incomplete",
                        )
                        self.files.check(export["model"])
                        exports[candidate] = export["model"]
                        self.returned(baseline)
                    self.guard()
                    self.returned(baseline)
                    evaluation_inputs = self.run / "evaluation-inputs.json"
                    atomic_json(
                        evaluation_inputs,
                        {
                            "bundle_sha256": self.bundle_hash,
                            "models": exports,
                            "training_endpoints": {
                                c: self.state["completed"][c + "/train"] for c in CANDIDATES
                            },
                            "export_endpoints": {
                                c: self.state["completed"][c + "/export"] for c in CANDIDATES
                            },
                            "controls": self.bundle["controls"],
                            "target": self.bundle["inputs"]["target"],
                        },
                    )
                    evaluation = self.stage(
                        "evaluation",
                        self.bundle["evaluation"],
                        values={"evaluation_inputs": str(evaluation_inputs)},
                    )
                    require(
                        evaluation.get("native") is True
                        and evaluation.get("cells") == list(CELLS)
                        and evaluation.get("target_only_diagnostic") is True,
                        "native matched nine-model evaluation incomplete",
                    )
                    self.files.check(evaluation["report"])
                    self.returned(baseline)
                    self.publish(status="complete", report=evaluation["report"])
            except BaseException as error:
                with deferred_termination():
                    release = {"status": "PENDING"}
                    if "resource_baseline" in self.state:
                        try:
                            self.returned(self.state["resource_baseline"])
                            release = {"status": "PASS"}
                        except BaseException as cleanup_error:
                            release = {"status": "FAILED", "reason": str(cleanup_error)}
                    self.publish(
                        release_after_failure=release,
                        status="stopped" if isinstance(error, InterruptedError) else "failed",
                        failure={"type": type(error).__name__, "reason": str(error)},
                        checkpoints_deleted=False,
                    )
                raise
        return self.state
