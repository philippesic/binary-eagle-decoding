import csv
import os
import re
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import cast

from benchmarks.comparison import Arm, Comparison
from benchmarks.measurement import now, pause, port_number, ready, request
from benchmarks.runner import native_counts, signature
from benchmarks.serving import server_arguments
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, canonical, read


def find_server(arguments: list[str]) -> ProcessIdentity:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (
                proc.stat().st_uid != os.getuid()
                or (proc / "exe").resolve() != Path(arguments[0]).resolve()
            ):
                continue
            actual = [
                part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part
            ]
            if actual[1:] == arguments[1:]:
                found.append(ProcessIdentity.read(int(proc.name)))
        except (FileNotFoundError, PermissionError):
            continue
    if len(found) != 1:
        raise ValueError("profiled server command/port ownership is ambiguous")
    return found[0]


@contextmanager
def traced_session(
    profiler: Path,
    spec: Comparison,
    arm: Arm,
    cell: Store,
    environment: dict[str, str],
    tracing: Sequence[str],
    log_name: str,
) -> Iterator[int]:
    port = port_number()
    session_name = "binary-eagle-" + uuid.uuid4().hex
    arguments = server_arguments(spec, arm, port)
    argv = [
        str(profiler),
        "launch",
        "--session-new=" + session_name,
        *tracing,
        "--cuda-graph-trace=node",
        "--show-output=true",
        *arguments,
    ]
    cell.atomic(cell.root / "command.json", canonical({"argv": argv}))
    owner: ProcessIdentity | None = None
    with (cell.root / log_name).open("wb") as log:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            deadline = now() + 180
            while not ready(port):
                if process.poll() is not None or now() >= deadline:
                    raise ValueError("profiled native server startup failed")
                pause(0.1)
            owner = find_server(arguments)
            subprocess.run(
                [
                    str(profiler),
                    "start",
                    "--session=" + session_name,
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--force-overwrite=false",
                    "--output=" + str(cell.root / "trace"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=30,
                check=True,
            )
            cell.atomic(cell.root / "owner.json", canonical(cast(Record, asdict(owner))))
            cell.atomic(
                cell.root / "library-maps.txt", Path(f"/proc/{owner.pid}/maps").read_bytes()
            )
            yield port
            subprocess.run(
                [str(profiler), "stop", "--session=" + session_name],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=120,
                check=True,
            )
        finally:
            subprocess.run(
                [
                    str(profiler),
                    "shutdown",
                    "--session=" + session_name,
                    "--kill=sigterm",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=30,
                check=False,
            )
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            cell.atomic(
                cell.root / "release.json",
                canonical(
                    {
                        "session": session_name,
                        "launcher_reaped": True,
                        "server_alive": owner.alive() if owner else None,
                    }
                ),
            )


def profile(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) - {"forbidden_kernels", "kernel_round_multipliers"}
        != {"schema", "comparison", "profiler", "expected_kernels"}
        or raw["schema"] != 1
        or output.exists()
    ):
        raise ValueError("explicit native profiling inputs and a new output required")
    root = specification.resolve().parent
    comparison, profiler = artifact(raw["comparison"], root), artifact(raw["profiler"], root)
    comparison.verify()
    profiler.verify()
    spec = Comparison.load(comparison.path)
    spec.verify()
    if spec.tier != "pilot" or spec.sampling.temperature:
        raise ValueError("kernel activation profiling requires an explicit greedy pilot")
    expected = record(raw["expected_kernels"])
    if set(expected) != {arm.name for arm in spec.arms}:
        raise ValueError("expected kernel assertions must identify every arm")
    with device_lease("cuda"):
        store = Store(output)
        store.atomic(store.root / "specification.json", canonical(raw))
        arms: Record = {}
        reference: Record | None = None
        for arm in spec.arms:
            cell = Store(store.root / arm.name)
            with traced_session(
                profiler.path,
                spec,
                arm,
                cell,
                native_environment(spec.binary.path.parent),
                ("--trace=cuda",),
                "server-profiler.log",
            ) as port:
                task = spec.tasks[0]
                rendered = request(
                    port,
                    "apply-template",
                    {"messages": task["messages"], "tools": task.get("tools", [])},
                )
                tokens = request(
                    port,
                    "tokenize",
                    {"content": rendered["prompt"], "add_special": True, "parse_special": True},
                )["tokens"]
                response = request(
                    port,
                    "completion",
                    {
                        "prompt": tokens,
                        "n_predict": task["output_cap"],
                        "temperature": 0,
                        "seed": 8101,
                        "cache_prompt": False,
                        "return_tokens": True,
                    },
                )
                cell.atomic(cell.root / "response.json", canonical(response))
                current = signature(response)
                if reference is None:
                    reference = current
                if current != reference:
                    raise ValueError("profiler changed greedy output/stopping across arms")
            report = cell.root / "trace.nsys-rep"
            if not report.is_file():
                raise ValueError("profiler did not publish a trace")
            with (cell.root / "kernel-summary.csv").open("wb") as table:
                subprocess.run(
                    [
                        str(profiler.path),
                        "stats",
                        "--report=cuda_gpu_kern_sum",
                        "--format=csv",
                        str(report),
                    ],
                    stdout=table,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            table_rows = cast(
                list[list[str]],
                list(csv.reader((cell.root / "kernel-summary.csv").read_text().splitlines())),
            )
            headers = [
                index
                for index, row in enumerate(table_rows)
                if "Name" in row and "Instances" in row
            ]
            if len(headers) != 1:
                raise ValueError("CUDA kernel statistics unavailable")
            header = table_rows[headers[0]]
            names = [
                row[header.index("Name")]
                for row in table_rows[headers[0] + 1 :]
                if len(row) == len(header)
            ]
            required = [string(value) for value in sequence(expected[arm.name])]
            forbidden = [
                string(value)
                for value in sequence(record(raw.get("forbidden_kernels", {})).get(arm.name, []))
            ]
            counts = {
                row[header.index("Name")]: int(row[header.index("Instances")])
                for row in table_rows[headers[0] + 1 :]
                if len(row) == len(header)
            }
            if any(any(kernel in name for name in names) for kernel in forbidden):
                raise ValueError("forbidden kernel executed in control arm: " + arm.name)
            rounds = cast(
                list[str],
                re.findall(
                    r"#gen drafts =\s*(\d+)", (cell.root / "server-profiler.log").read_text()
                ),
            )
            multipliers = record(record(raw.get("kernel_round_multipliers", {})).get(arm.name, {}))
            for kernel, multiple in multipliers.items():
                actual = sum(count for name, count in counts.items() if kernel in name)
                if len(rounds) != 1 or actual != int(rounds[0]) * integer(multiple):
                    raise ValueError(
                        "native kernel instances do not cover every draft layer: " + kernel
                    )
            if any(not any(kernel in name for name in names) for kernel in required):
                raise ValueError("expected native kernel did not execute: " + arm.name)
            arms[arm.name] = {
                "required_kernels": required,
                "observed_kernel_names": names,
                "kernel_instances": counts,
                "draft_rounds": int(rounds[0]) if len(rounds) == 1 else 0,
                "native": native_counts(response, arm.drafter is not None),
            }
        result: Record = {
            "status": "passed",
            "arms": arms,
            "scope": "instrumented CUDA kernel activation and greedy fidelity only; profiler timings are excluded from primary comparisons",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
