import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path

from benchmarks.comparison import Arm, Comparison
from benchmarks.measurement import now, request
from benchmarks.metrics import (
    draft_statistics,
    native_counts,
    reconcile,
    signature,
    valid_sample,
    verify_generation,
)
from benchmarks.runner import prompt_tokens
from benchmarks.runtime_policy import arguments, payload
from benchmarks.serving import session
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.identity import source_identity
from w1a1_eagle.records import artifact, canonical, identity, read, reference

WARMUP_TOKENS = 16
NEAR_BEST = 0.02
FIELDS = {
    "schema",
    "comparison",
    "drafter",
    "tasks",
    "cost_tasks",
    "survival_cap",
    "cost_cap",
    "max_depth",
    "exact",
}


def input_tokens(task: Record) -> int:
    return integer(record(task.get("ancestry", {})).get("input_tokens", 0))


def by_category(task: Record) -> tuple[str, int, str]:
    return string(task["category"]), input_tokens(task), string(task["id"])


def by_length(task: Record) -> tuple[int, str]:
    return input_tokens(task), string(task["id"])


Serve = Callable[[Comparison, Arm, Store, bool], AbstractContextManager[int]]


def served(
    spec: Comparison, arm: Arm, cell: Store, consistent: bool
) -> AbstractContextManager[int]:
    return session(spec, arm, cell, consistent=consistent)


def spread(items: Sequence[Record], count: int) -> tuple[Record, ...]:
    return tuple(items[(2 * index + 1) * len(items) // (2 * count)] for index in range(count))


@dataclass(frozen=True)
class Calibration:
    comparison: Artifact
    spec: Comparison
    drafter: Artifact
    tasks: tuple[Record, ...]
    cost_tasks: tuple[Record, ...]
    survival_cap: int
    cost_cap: int
    max_depth: int
    exact: bool

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        raw = read(path)
        if set(raw) != FIELDS or raw["schema"] != "depth-calibration-v1":
            raise ValueError("unsupported depth calibration specification")
        root = path.resolve().parent
        source = artifact(raw["comparison"], root)
        source.verify()
        spec = Comparison.load(source.path)
        if spec.role != "dev" or spec.diagnostic or spec.sampling.temperature or spec.references:
            raise ValueError("depth calibration requires a clean greedy DEV comparison")
        count, cost_count = integer(raw["tasks"]), integer(raw["cost_tasks"])
        survival_cap, cost_cap = integer(raw["survival_cap"]), integer(raw["cost_cap"])
        depth = integer(raw["max_depth"])
        if type(raw["exact"]) is not bool:
            raise ValueError("exact must be an explicit boolean")
        if (
            not 1 <= cost_count <= count <= len(spec.tasks)
            or not 2 <= depth <= 15
            or not WARMUP_TOKENS <= survival_cap <= 4096
            or not 8 * (depth + 1) <= cost_cap <= 4096
        ):
            raise ValueError("depth calibration bounds invalid")
        tasks = spread(sorted(spec.tasks, key=by_category), count)
        return cls(
            source,
            spec,
            artifact(raw["drafter"], root),
            tasks,
            spread(sorted(tasks, key=by_length), cost_count),
            survival_cap,
            cost_cap,
            depth,
            raw["exact"],
        )


def measure(
    spec: Comparison,
    exact: bool,
    arm: Arm,
    depth: int,
    tasks: tuple[Record, ...],
    cap: int,
    cell: Store,
    serve: Serve = served,
) -> Record:
    began = now()
    spec = replace(spec, depth=max(depth, 1))
    speculative = arm.drafter is not None
    responses: list[Record] = []
    rows: list[Record] = []
    with serve(spec, arm, cell, exact) as port:
        for index, task in enumerate((tasks[0], *tasks)):
            limit = WARMUP_TOKENS if index == 0 else min(cap, integer(task["output_cap"]))
            identifiers = prompt_tokens(port, task, limit, spec.context)
            query = payload(identifiers, limit, 0.0, 1.0, 0, 0.0, 8101)
            started = now()
            response = request(port, "completion", query)
            elapsed = now() - started
            verify_generation(response, query, speculative)
            if not valid_sample(response, limit):
                raise ValueError("invalid native tokens or stopping")
            responses.append(response)
            if index:
                timings = record(response["timings"])
                counts = native_counts(response, speculative)
                rows.append(
                    {
                        "task": task["id"],
                        "category": task["category"],
                        "input_tokens": len(identifiers),
                        "tokens": len(sequence(response["tokens"])),
                        "stop_type": response["stop_type"],
                        "wall_seconds": elapsed,
                        "prompt_ms": number(timings["prompt_ms"]),
                        "predicted_ms": number(timings["predicted_ms"]),
                        "proposed": counts["proposed"],
                        "accepted": counts["accepted"],
                        "output_sha256": identity(signature(response)),
                    }
                )
    if speculative:
        log = (cell.root / "server.log").read_text(errors="replace")
        reconcile(log, responses)
        statistics = draft_statistics(log)
        for row, before, after in zip(rows, statistics[:-1], statistics[1:], strict=True):
            row["draft_rounds"] = integer(after["rounds"]) - integer(before["rounds"])
            row["draft_ms"] = (
                number(after["draft_ms"]) - number(before["draft_ms"])
                if after["draft_ms"] is not None and before["draft_ms"] is not None
                else None
            )
            row["reached"] = [
                integer(late) - integer(early)
                for early, late in zip_longest(
                    sequence(before["reached"]), sequence(after["reached"]), fillvalue=0
                )
            ]
    cell.atomic(cell.root / "rows.json", canonical({"rows": rows}))
    return {"depth": depth, "rows": rows, "seconds": now() - began}


def rounds(rows: Sequence[Record]) -> int:
    return sum(integer(row["tokens"]) - integer(row["accepted"]) - 1 for row in rows)


def survival(rows: Sequence[Record], depth: int) -> tuple[int, list[float]]:
    drafted = sum(integer(row["draft_rounds"]) for row in rows)
    reached = [0] * depth
    for row in rows:
        for position, value in enumerate(sequence(row["reached"])[:depth]):
            reached[position] += integer(value)
    return drafted, [value / drafted if drafted else 0.0 for value in reached]


def tokens_per_round(shares: Sequence[float], depth: int) -> float:
    reach = [1.0, *shares, 0.0]
    exact = [reach[index] - reach[index + 1] for index in range(len(shares) + 1)]
    pieces = sum(share * math.ceil((index + 1) / (depth + 1)) for index, share in enumerate(exact))
    return (1.0 + sum(shares)) / pieces


def round_cost(session_record: Record) -> Record:
    rows = [record(row) for row in sequence(session_record["rows"])]
    count = rounds(rows)
    if count <= 0:
        raise ValueError("cost session produced no decoding rounds")
    predicted = sum(number(row["predicted_ms"]) for row in rows)
    drafted = [row["draft_ms"] for row in rows if "draft_ms" in row]
    drafter = sum(number(value) for value in drafted) / count if drafted else 0.0
    if any(value is None for value in drafted):
        raise ValueError("drafter timing counters unavailable")
    return {
        "depth": session_record["depth"],
        "requests": len(rows),
        "rounds": count,
        "ms_per_round": predicted / count,
        "drafter_ms_per_round": drafter,
        "target_ms_per_round": predicted / count - drafter,
        "accepted_per_round": sum(integer(row["accepted"]) for row in rows) / count,
        "overhead_seconds_per_request": sum(
            number(row["wall_seconds"]) - number(row["predicted_ms"]) / 1000 for row in rows
        )
        / len(rows),
        "seconds": session_record["seconds"],
    }


def best(values: dict[int, float]) -> Record:
    choice = max(sorted(values), key=values.__getitem__)
    return {
        "depth": choice,
        "near_best": [
            depth for depth in sorted(values) if values[depth] >= (1 - NEAR_BEST) * values[choice]
        ],
    }


def predict(survival_rows: Sequence[Record], costs: Sequence[Record], depth: int) -> Record:
    drafted, shares = survival(survival_rows, depth)
    if not drafted:
        raise ValueError("survival run produced no drafted rounds")
    requests = len(survival_rows)
    tokens = sum(integer(row["tokens"]) for row in survival_rows)
    overhead = sum(
        number(row["wall_seconds"]) - number(row["predicted_ms"]) / 1000 for row in survival_rows
    )
    cost = {integer(row["depth"]): record(row) for row in costs}
    drafting = [cost[value] for value in cost if value]
    target_overhead = number(cost[0]["overhead_seconds_per_request"]) / (
        sum(number(row["overhead_seconds_per_request"]) for row in drafting) / len(drafting)
    )
    rows: list[Record] = []
    complete: dict[int, float] = {}
    for value in sorted(cost):
        emitted = tokens_per_round(shares, value)
        milliseconds = number(cost[value]["ms_per_round"])
        seconds = (overhead * (target_overhead if value == 0 else 1.0)) + (
            tokens - requests
        ) / emitted * milliseconds / 1000
        complete[value] = tokens / seconds
        rows.append(
            {
                "depth": value,
                "tokens_per_round": emitted,
                "ms_per_round": milliseconds,
                "decode_tokens_per_second": 1000 * emitted / milliseconds,
                "complete_tokens_per_second": complete[value],
            }
        )
    choice = best({key: speed for key, speed in complete.items() if key})
    top = complete[integer(choice["depth"])]
    categories: Record = {}
    for category in sorted({string(row["category"]) for row in survival_rows}):
        selected = [row for row in survival_rows if row["category"] == category]
        count, reached = survival(selected, depth)
        decode = {
            value: 1000 * tokens_per_round(reached, value) / number(cost[value]["ms_per_round"])
            for value in cost
            if value
        }
        categories[category] = {
            "requests": len(selected),
            "drafted_rounds": count,
            "survival": reached,
            "decode_tokens_per_second": [decode[value] for value in sorted(decode)],
            **best(decode),
        }
    return {
        "survival": {
            "requests": requests,
            "tokens": tokens,
            "drafted_rounds": drafted,
            "share_reaching_position": shares,
            "ms_per_round": sum(number(row["predicted_ms"]) for row in survival_rows)
            / rounds(survival_rows),
            "accepted_per_round": sum(integer(row["accepted"]) for row in survival_rows)
            / rounds(survival_rows),
            "overhead_seconds": overhead,
        },
        "target_only_overhead_ratio": target_overhead,
        "predictions": [
            {**row, "relative_to_best": complete[integer(row["depth"])] / top} for row in rows
        ],
        "recommendation": {**choice, "speculation_gain": top / complete[0]},
        "by_category": categories,
    }


def calibrate(specification: Path, output: Path) -> Record:
    plan = Calibration.load(specification)
    plan.spec.verify()
    plan.drafter.verify()
    if output.exists():
        raise ValueError("calibration output must be new; preserve prior attempts")
    with device_lease("cuda"):
        began = now()
        store = Store(output)
        gpu = cuda_identity()
        store.atomic(store.root / "gpu.txt", gpu.encode())
        store.atomic(store.root / "specification.json", specification.read_bytes())
        code = source_identity(Path(__file__).resolve().parents[1])
        drafter = Arm("drafter", plan.drafter)
        try:
            deep = measure(
                plan.spec,
                plan.exact,
                drafter,
                plan.max_depth,
                plan.tasks,
                plan.survival_cap,
                Store(store.root / ("survival-d" + str(plan.max_depth))),
            )
            sessions = [
                measure(
                    plan.spec,
                    plan.exact,
                    drafter if depth else Arm("target", None),
                    depth,
                    plan.cost_tasks,
                    plan.cost_cap,
                    Store(store.root / ("cost-d" + str(depth))),
                )
                for depth in range(plan.max_depth + 1)
            ]
            costs = [round_cost(value) for value in sessions]
            survival_rows = [record(row) for row in sequence(deep["rows"])]
            model = predict(survival_rows, costs, plan.max_depth)
        except BaseException as error:
            store.atomic(
                store.root / "failure.json", canonical({"status": "failed", "error": str(error)})
            )
            raise
        result: Record = {
            "schema": "depth-calibration-result-v1",
            "status": "complete",
            "specification": reference(Artifact.capture(specification)),
            "comparison": reference(plan.comparison),
            "identities": {
                "runtime_build": reference(plan.spec.source),
                "server": reference(plan.spec.binary),
                "libraries": [reference(value) for value in plan.spec.libraries],
                "target": reference(plan.spec.target),
                "drafter": reference(plan.drafter),
                "workload": reference(plan.spec.workload),
            },
            "gpu": gpu,
            "code_sha256": code,
            "settings": {
                "context": plan.spec.context,
                "decode_consistent": "32" if plan.exact else "0",
                "max_depth": plan.max_depth,
                "survival_cap": plan.survival_cap,
                "cost_cap": plan.cost_cap,
                "warmup_tokens": WARMUP_TOKENS,
                "tasks": [task["id"] for task in plan.tasks],
                "cost_tasks": [task["id"] for task in plan.cost_tasks],
                "server_arguments": arguments(
                    Path("server"), Path("target"), Path("drafter"), plan.spec.context, 0, 0
                ),
            },
            "costs": costs,
            **model,
            "survival_seconds": deep["seconds"],
            "elapsed_seconds": now() - began,
            "scope": "round model from one deep survival run and short per-depth cost sessions; "
            "a recommendation for this target, drafter, runtime, GPU and DEV workload only",
        }
        path = store.root / "calibration.json"
        store.atomic(path, canonical(result))
    return {
        "record": reference(Artifact.capture(path)),
        "recommendation": result["recommendation"],
        "predictions": result["predictions"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
