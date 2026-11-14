import math
import re
import sqlite3
import subprocess
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from benchmarks.calibration import Serve, by_length, input_tokens, measure, served, spread
from benchmarks.comparison import Arm, Comparison
from benchmarks.measurement import now
from benchmarks.profiling import traced_session
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.identity import source_identity
from w1a1_eagle.records import artifact, canonical, read, reference

EDIT = "round_profile_nvtx_v1"
PASSES = ("clean", "phases", "nodes")
FIELDS = {
    "schema",
    "comparison",
    "diagnostic_build",
    "profiler",
    "drafters",
    "depths",
    "lengths",
    "output_cap",
    "exact",
}
PHASES = ("draft", "checkpoint", "target", "process", "verify")
DRAFTER_PHASES = ("draft", "process")
COPY_KINDS = {1: "host_to_device", 2: "device_to_host", 8: "device_to_device", 10: "peer"}
WEIGHT_ROLES: tuple[tuple[str, str], ...] = (
    (r"(^|\.)token_embd\.", "vocabulary"),
    (r"^fc\.", "fusion"),
    (r"^enc\.", "fusion"),
    (r"\.attn_(q|k|v|qkv|kv)\.", "attention.qkv_projection"),
    (r"\.attn_output\.", "attention.output_projection"),
    (r"\.attn_[a-z_]*norm\.", "attention.norms"),
    (r"\.ffn_gate\.", "ffn.gate"),
    (r"\.ffn_up\.", "ffn.up"),
    (r"\.ffn_down\.", "ffn.down"),
    (r"\.ffn_[a-z_]*norm\.", "ffn.norms"),
    (r"^output_norm\.", "final_norm"),
    (r"^output\.", "vocabulary"),
)
OP_ROLES = {
    "FLASH_ATTN_EXT": "attention.context",
    "SOFT_MAX": "attention.context",
    "ROPE": "attention.rope",
    "ARGMAX": "token_selection",
    "ARGSORT": "token_selection",
    "TOP_K": "token_selection",
}
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("feature handoff + context K/V update", ("handoff.",)),
    ("block embedding", ("draft.embedding",)),
    (
        "attention projections (q, k, v, output)",
        ("draft.attention.qkv_projection", "draft.attention.output_projection"),
    ),
    ("attention over context (score, softmax, mix)", ("draft.attention.context",)),
    (
        "attention norms, rope, block K/V write",
        ("draft.attention.norms", "draft.attention.rope", "draft.kv_update"),
    ),
    ("feed-forward gate+up (one fused kernel)", ("draft.ffn.gate+ffn.up",)),
    ("feed-forward gate", ("draft.ffn.gate",)),
    ("feed-forward up", ("draft.ffn.up",)),
    ("feed-forward down", ("draft.ffn.down",)),
    ("feed-forward norm", ("draft.ffn.norms",)),
    ("final norm + vocabulary scoring", ("draft.final_norm", "draft.vocabulary")),
    ("token selection kernels", ("draft.token_selection",)),
    ("residual and activation elementwise", ("draft.elementwise", "draft.weights.other")),
    ("copies (inputs, logits)", ("copies.",)),
    ("outside graph nodes", ("draft.unlabelled", "process.unlabelled")),
)


@dataclass(frozen=True)
class Band:
    name: str
    tasks: tuple[Record, ...]


@dataclass(frozen=True)
class Plan:
    comparison: Artifact
    spec: Comparison
    diagnostic: Artifact
    traced: Comparison
    profiler: Artifact
    drafters: tuple[Arm, ...]
    depths: tuple[int, ...]
    bands: tuple[Band, ...]
    cap: int
    exact: bool

    @classmethod
    def load(cls, path: Path) -> "Plan":
        raw = read(path)
        if set(raw) != FIELDS or raw["schema"] != "round-profile-v1":
            raise ValueError("unsupported round profile specification")
        root = path.resolve().parent
        comparison = artifact(raw["comparison"], root)
        comparison.verify()
        spec = Comparison.load(comparison.path)
        if spec.role != "dev" or spec.diagnostic or spec.sampling.temperature or spec.references:
            raise ValueError("round profiling requires a clean greedy DEV comparison")
        spec.source.verify()
        diagnostic = artifact(raw["diagnostic_build"], root)
        diagnostic.verify()
        built, clean = read(diagnostic.path), read(spec.source.path)
        if (
            built.get("status") != "built"
            or built.get("verification_trace", False) is not False
            or built.get("source_bundle") != clean.get("source_bundle")
            or sequence(built["runtime_repairs"]) != [*sequence(clean["runtime_repairs"]), EDIT]
        ):
            raise ValueError("diagnostic build must be the clean runtime plus " + EDIT + " only")
        produced = [
            artifact(value, diagnostic.path.parent) for value in sequence(built["artifacts"])
        ]
        servers = [value for value in produced if value.path.name == "llama-server"]
        names = {value.path.name for value in spec.libraries}
        libraries = tuple(value for value in produced if value.path.name in names)
        if len(servers) != 1 or {value.path.name for value in libraries} != names:
            raise ValueError("diagnostic build lacks the server or the comparison's libraries")
        drafters = tuple(
            Arm(string(name), artifact(value, root))
            for name, value in record(raw["drafters"]).items()
        )
        if not drafters or any(
            not arm.name.replace("-", "").replace("_", "").isalnum() for arm in drafters
        ):
            raise ValueError("named drafters required")
        depths = tuple(integer(value) for value in sequence(raw["depths"]))
        if not depths or len(set(depths)) != len(depths) or any(not 1 <= d <= 15 for d in depths):
            raise ValueError("distinct draft depths between 1 and 15 required")
        bands: list[Band] = []
        for value in sequence(raw["lengths"]):
            row = record(value)
            if set(row) != {"name", "minimum", "maximum", "tasks"}:
                raise ValueError("length bands need name, minimum, maximum and tasks")
            low, high, count = (
                integer(row["minimum"]),
                integer(row["maximum"]),
                integer(row["tasks"]),
            )
            eligible = sorted(
                (task for task in spec.tasks if low <= input_tokens(task) <= high), key=by_length
            )
            if not 1 <= count <= len(eligible):
                raise ValueError("length band has too few DEV tasks: " + string(row["name"]))
            bands.append(Band(string(row["name"]), spread(eligible, count)))
        identifiers = [string(task["id"]) for band in bands for task in band.tasks]
        if (
            not bands
            or len(set(identifiers)) != len(identifiers)
            or len({b.name for b in bands}) != len(bands)
        ):
            raise ValueError("length bands must be named uniquely and must not share tasks")
        cap = integer(raw["output_cap"])
        if type(raw["exact"]) is not bool or not 16 <= cap <= 4096:
            raise ValueError("explicit exact flag and an output cap of 16-4096 required")
        traced = replace(spec, source=diagnostic, binary=servers[0], libraries=libraries)
        return cls(
            comparison,
            spec,
            diagnostic,
            traced,
            artifact(raw["profiler"], root),
            drafters,
            depths,
            tuple(bands),
            cap,
            raw["exact"],
        )

    def tasks(self) -> tuple[Record, ...]:
        return tuple(task for band in self.bands for task in band.tasks)


def tracer(plan: Plan, nodes: bool) -> Serve:
    @contextmanager
    def serve(spec: Comparison, arm: Arm, cell: Store, consistent: bool) -> Iterator[int]:
        environment = native_environment(spec.binary.path.parent)
        if not consistent:
            environment["GGML_CUDA_DECODE_CONSISTENT"] = "0"
        if nodes:
            environment.update(GGML_CUDA_DISABLE_GRAPHS="1", BITDRAFT_NVTX_NODES="1")
        with traced_session(
            plan.profiler.path, spec, arm, cell, environment, ("--trace=cuda,nvtx",), "server.log"
        ) as port:
            yield port

    return serve


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Activity:
    start: int
    end: int
    launch: int
    kind: str
    name: str
    size: int


@dataclass(frozen=True)
class Source:
    slot: int
    name: str
    kind: str
    shape: tuple[int, ...]
    size: int
    weight: bool


@dataclass(frozen=True)
class Node:
    index: int
    op: str
    name: str
    shape: tuple[int, ...]
    sources: tuple[Source, ...]


def dimensions(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def parse_node(label: str) -> Node:
    parts = label.split("|")
    if len(parts) < 5 or parts[0] not in {"node", "fused"}:
        raise ValueError("malformed graph node label: " + label[:80])
    sources = []
    for part in parts[5:]:
        fields = part.split(":")
        if len(fields) < 6 or fields[-1] not in {"0", "1"}:
            raise ValueError("malformed graph node source: " + part[:80])
        sources.append(
            Source(
                int(fields[0]),
                ":".join(fields[1:-4]),
                fields[-4],
                dimensions(fields[-3]),
                int(fields[-2]),
                fields[-1] == "1",
            )
        )
    return Node(int(parts[1]), parts[2], parts[3], dimensions(parts[4]), tuple(sources))


def product(values: Sequence[int]) -> int:
    return math.prod(values)


def cost(node: Node) -> tuple[int, int, int]:
    weights = [source for source in node.sources if source.weight]
    rows = product(node.shape[1:])
    if node.op == "GET_ROWS" and weights:
        table = weights[0]
        size = table.size // max(product(table.shape[1:]), 1) * rows
    else:
        size = sum(source.size for source in weights)
    slots = {source.slot: source for source in node.sources}
    if node.op == "MUL_MAT" and 0 in slots:
        return slots[0].shape[0] * node.shape[0] * rows, size, rows
    if node.op == "W1A1_MUL_MAT" and 2 in slots:
        return slots[2].shape[0] * node.shape[0] * rows, size, rows
    if node.op == "FLASH_ATTN_EXT" and {0, 1, 2} <= set(slots):
        query, key, value = slots[0], slots[1], slots[2]
        tokens = query.shape[1] * query.shape[3]
        return (
            tokens * query.shape[2] * key.shape[1] * (query.shape[0] + value.shape[0]),
            size,
            tokens,
        )
    return 0, size, rows


def weight_role(name: str) -> str:
    base = name.split(" (")[0]
    for pattern, role in WEIGHT_ROLES:
        if re.search(pattern, base):
            return role
    return "weights.other"


def slot_of(source: Source) -> int:
    return source.slot


def span_order(span: Span) -> tuple[int, int]:
    return span.start, -span.end


def role(node: Node) -> str:
    if node.op == "GET_ROWS" and any(source.weight for source in node.sources):
        return "embedding"
    for source in sorted(node.sources, key=slot_of):
        if source.weight:
            return weight_role(source.name)
    if node.op in OP_ROLES:
        return OP_ROLES[node.op]
    if node.op in {"SET_ROWS", "CPY"} and (
        node.name.startswith("cache_") or any(s.name.startswith("cache_") for s in node.sources)
    ):
        return "kv_update"
    return "elementwise"


def layer(node: Node) -> int | None:
    for source in node.sources:
        found = re.search(r"blk\.(\d+)\.", source.name) if source.weight else None
        if found:
            return int(found[1])
    for name in (node.name, *(source.name for source in node.sources)):
        found = re.search(r"(?:-|_l)(\d+)(?:$| )", name)
        if found:
            return int(found[1])
    return None


def roles(nodes: Sequence[Node]) -> list[str]:
    assigned = [role(node) for node in nodes]
    for index in range(len(nodes) - 2, -1, -1):
        if nodes[index].op in {"RMS_NORM", "NORM"} and assigned[index] == "elementwise":
            assigned[index] = assigned[index + 1]
    return assigned


def rows_of(cursor: sqlite3.Cursor) -> list[tuple[object, ...]]:
    return cast(list[tuple[object, ...]], cursor.fetchall())


def optional(value: object) -> int:
    return integer(value) if value is not None else 0


def api_category(name: str) -> str:
    for marker, category in (
        ("GraphLaunch", "graph_launch"),
        ("LaunchKernel", "kernel_launch"),
        ("Synchronize", "synchronize"),
        ("Memcpy", "memcpy"),
        ("Memset", "memset"),
        ("Event", "event"),
    ):
        if marker in name:
            return category
    return "other"


def load_trace(path: Path) -> tuple[list[Span], list[Activity], list[Span]]:
    with closing(sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)) as db:
        tables = {string(row[0]) for row in rows_of(db.execute("SELECT name FROM sqlite_master"))}
        spans_by_thread: dict[int, list[Span]] = {}
        for start, end, label, thread in rows_of(
            db.execute(
                "SELECT e.start, e.end, coalesce(s.value, e.text), e.globalTid FROM NVTX_EVENTS e "
                "LEFT JOIN StringIds s ON e.textId = s.id WHERE e.eventType = 59 AND e.end IS NOT NULL"
            )
        ):
            spans_by_thread.setdefault(integer(thread), []).append(
                Span(integer(start), integer(end), string(label))
            )
        if not spans_by_thread:
            raise ValueError("trace has no NVTX ranges; was the diagnostic runtime traced?")
        drafts = {
            thread: sum(span.label == "srv.draft" for span in values)
            for thread, values in spans_by_thread.items()
        }
        main = max(drafts, key=drafts.__getitem__)
        launches: dict[int, int] = {}
        calls: list[Span] = []
        for correlation, start, end, thread, name in rows_of(
            db.execute(
                "SELECT r.correlationId, r.start, r.end, r.globalTid, s.value "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON r.nameId = s.id"
            )
        ):
            if integer(thread) != main:
                continue
            launches[integer(correlation)] = integer(start)
            label = api_category(name if isinstance(name, str) else "")
            calls.append(Span(integer(start), integer(end), label))
        activities: list[Activity] = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            for start, end, correlation, name in rows_of(
                db.execute(
                    "SELECT k.start, k.end, k.correlationId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "LEFT JOIN StringIds s ON k.shortName = s.id"
                )
            ):
                if integer(correlation) in launches:
                    activities.append(
                        Activity(
                            integer(start),
                            integer(end),
                            launches[integer(correlation)],
                            "kernel",
                            string(name) if name is not None else "?",
                            0,
                        )
                    )
        for table, kind in (
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "set"),
        ):
            if table not in tables:
                continue
            column = "copyKind" if kind == "copy" else "0"
            for start, end, correlation, size, mode in rows_of(
                db.execute(f"SELECT start, end, correlationId, bytes, {column} FROM {table}")
            ):
                if integer(correlation) in launches:
                    activities.append(
                        Activity(
                            integer(start),
                            integer(end),
                            launches[integer(correlation)],
                            kind,
                            COPY_KINDS.get(optional(mode), "other") if kind == "copy" else "memset",
                            optional(size),
                        )
                    )
    return sorted(spans_by_thread[main], key=span_order), activities, calls


def parents(spans: Sequence[Span]) -> list[int]:
    result = [-1] * len(spans)
    stack: list[int] = []
    for index, span in enumerate(spans):
        while stack and spans[stack[-1]].end <= span.start:
            stack.pop()
        result[index] = stack[-1] if stack else -1
        stack.append(index)
    return result


def enclosing(spans: Sequence[Span], times: Sequence[int]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = [()] * len(times)
    stack: list[int] = []
    cursor = 0
    for moment, position in sorted((moment, position) for position, moment in enumerate(times)):
        while cursor < len(spans) and spans[cursor].start <= moment:
            while stack and spans[stack[-1]].end <= spans[cursor].start:
                stack.pop()
            stack.append(cursor)
            cursor += 1
        while stack and spans[stack[-1]].end <= moment:
            stack.pop()
        result[position] = tuple(stack)
    return result


def merge(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        elif end > start:
            merged.append((start, end))
    return merged


def covered(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge(intervals))


def overlap(first: Sequence[tuple[int, int]], second: Sequence[tuple[int, int]]) -> int:
    a, b = merge(first), merge(second)
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def phase_of(label: str) -> str | None:
    if not label.startswith("srv."):
        return None
    return label[4:].split("|")[0]


def short(label: str) -> str:
    return label.split("|")[0]


@dataclass(frozen=True)
class Round:
    request: int
    start: int
    end: int


def rounds(spans: Sequence[Span], depth: int) -> list[Round]:
    drafts = [span.start for span in spans if span.label == "srv.draft"]
    prompts = sorted(
        span.start
        for span in spans
        if span.label.startswith("srv.target|") and int(span.label.split("|")[1]) > depth + 1
    )
    segments: list[int] = []
    for moment in prompts:
        if not segments or any(segments[-1] < start < moment for start in drafts):
            segments.append(moment)
    found: list[Round] = []
    for start, end in zip(drafts, drafts[1:]):
        position = bisect_right(prompts, start)
        if position < len(prompts) and prompts[position] < end:
            continue
        found.append(Round(bisect_right(segments, start), start, end))
    return found


def add(totals: dict[str, float], key: str, value: float) -> None:
    totals[key] = totals.get(key, 0.0) + value


def blank() -> dict[str, float]:
    return {"ms": 0.0, "nodes": 0.0, "weight_bytes": 0.0, "macs": 0.0, "macs_per_slot": 0.0}


def start_of(spans: Sequence[Span]) -> Callable[[int], int]:
    def key(index: int) -> int:
        return spans[index].start

    return key


def components_of(
    spans: Sequence[Span],
    entries: Sequence[tuple[int, str, float]],
    fused: dict[int, list[int]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    kernels: dict[int, float] = {}
    phases: dict[int, str] = {}
    for index, phase, duration in entries:
        kernels[index] = kernels.get(index, 0.0) + duration
        phases[index] = phase
    by_phase: dict[str, list[int]] = {}
    for index in sorted(kernels, key=start_of(spans)):
        by_phase.setdefault(phases[index], []).append(index)
    components: dict[str, dict[str, float]] = {}
    instances: dict[str, float] = {}
    for phase, indices in by_phase.items():
        graph = [parse_node(spans[index].label) for index in indices]
        prefix = {"process": "handoff", "draft": "draft"}.get(phase, phase)
        instance, previous = 0, -1
        for index, node, assigned in zip(indices, graph, roles(graph)):
            if node.index <= previous:
                instance += 1
            group_nodes = [
                node,
                *(parse_node(spans[child].label) for child in fused.get(index, [])),
            ]
            previous = group_nodes[-1].index
            kinds = sorted({assigned, *(role(member) for member in group_nodes[1:])})
            assigned = "+".join(kinds) if len(kinds) > 1 else assigned
            costs = [cost(member) for member in group_nodes]
            macs, size = sum(c[0] for c in costs), sum(c[1] for c in costs)
            per_slot = sum(c[0] / c[2] for c in costs if c[2])
            names = [prefix + "." + assigned]
            position = layer(node)
            if position is not None and assigned.startswith("attention"):
                names.append(names[0] + ".layer" + str(position))
            for name in names:
                entry = components.setdefault(name, blank())
                entry["ms"] += kernels[index] / 1e6
                entry["nodes"] += 1
                entry["weight_bytes"] += size
                entry["macs"] += macs
                if instance == 0:
                    entry["macs_per_slot"] += per_slot
        instances[phase] = float(instance + 1)
    return components, instances


def analyse(path: Path, depth: int) -> list[Record]:
    spans, activities, calls = load_trace(path)
    found = rounds(spans, depth)
    starts = [item.start for item in found]

    def locate(moment: int) -> int | None:
        position = bisect_right(starts, moment) - 1
        return position if position >= 0 and moment < found[position].end else None

    owners = parents(spans)
    fused: dict[int, list[int]] = {}
    for index, owner in enumerate(owners):
        if owner >= 0 and spans[index].label.startswith("fused|"):
            fused.setdefault(owner, []).append(index)
    root_of = list(range(len(spans)))
    children = [0] * len(spans)
    for index, owner in enumerate(owners):
        if owner >= 0:
            root_of[index] = root_of[owner]
            children[owner] += spans[index].end - spans[index].start
    host: list[dict[str, float]] = [{} for _ in found]
    windows: list[dict[str, list[tuple[int, int]]]] = [{} for _ in found]
    for index, span in enumerate(spans):
        slot = locate(span.start)
        if slot is None:
            continue
        phase = phase_of(spans[root_of[index]].label) or "other"
        add(host[slot], phase + "/" + short(span.label), span.end - span.start - children[index])
        if owners[index] < 0 and phase_of(span.label):
            windows[slot].setdefault(phase, []).append((span.start, span.end))
    stacks = enclosing(spans, [activity.launch for activity in activities])
    gpu: list[dict[str, float]] = [{} for _ in found]
    busy: list[list[tuple[int, int]]] = [[] for _ in found]
    drafter_busy: list[list[tuple[int, int]]] = [[] for _ in found]
    labelled: list[list[tuple[int, str, float]]] = [[] for _ in found]
    for activity, stack in zip(activities, stacks):
        slot = locate(activity.launch)
        if slot is None:
            continue
        duration = float(activity.end - activity.start)
        interval = (activity.start, activity.end)
        busy[slot].append(interval)
        phase = (phase_of(spans[stack[0]].label) if stack else None) or "other"
        add(gpu[slot], phase, duration)
        if phase in DRAFTER_PHASES:
            drafter_busy[slot].append(interval)
        node = [index for index in stack if spans[index].label.startswith("node|")]
        if activity.kind != "kernel":
            add(gpu[slot], "copies." + phase + "." + activity.name, duration)
        elif node:
            labelled[slot].append((node[-1], phase, duration))
        else:
            add(gpu[slot], phase + ".unlabelled", duration)
    api_count: list[dict[str, float]] = [{} for _ in found]
    api_time: list[dict[str, float]] = [{} for _ in found]
    for call, stack in zip(calls, enclosing(spans, [call.start for call in calls])):
        slot = locate(call.start)
        if slot is None:
            continue
        phase = (phase_of(spans[stack[0]].label) if stack else None) or "other"
        add(api_count[slot], phase + "/" + call.label, 1.0)
        add(api_time[slot], phase + "/" + call.label, (call.end - call.start) / 1e6)
    result: list[Record] = []
    for slot, item in enumerate(found):
        wall = float(item.end - item.start)
        phase_walls = {phase: float(covered(windows[slot].get(phase, []))) for phase in PHASES}
        drafter_windows = [w for phase in DRAFTER_PHASES for w in windows[slot].get(phase, [])]
        components, instances = components_of(spans, labelled[slot], fused)
        result.append(
            {
                "request": item.request,
                "wall_ms": wall / 1e6,
                "phase_ms": {phase: value / 1e6 for phase, value in phase_walls.items()},
                "other_ms": (wall - sum(phase_walls.values())) / 1e6,
                "host_exclusive_ms": {
                    key: value / 1e6 for key, value in sorted(host[slot].items())
                },
                "gpu_ms": {key: value / 1e6 for key, value in sorted(gpu[slot].items())},
                "gpu_busy_ms": covered(busy[slot]) / 1e6,
                "drafter_wall_ms": covered(drafter_windows) / 1e6,
                "drafter_gpu_ms": covered(drafter_busy[slot]) / 1e6,
                "drafter_gpu_in_window_ms": overlap(drafter_busy[slot], drafter_windows) / 1e6,
                "graph_instances": instances,
                "api_calls": api_count[slot],
                "api_ms": api_time[slot],
                "components": components,
            }
        )
    return result


def flatten(value: object, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], float]:
    if isinstance(value, dict):
        flat: dict[tuple[str, ...], float] = {}
        for key, nested in record(value).items():
            flat.update(flatten(nested, (*path, key)))
        return flat
    return {path: number(value)}


def mean_record(rows: Sequence[Record]) -> Record:
    totals: dict[tuple[str, ...], float] = {}
    for row in rows:
        for key, value in flatten({k: v for k, v in row.items() if k != "request"}).items():
            totals[key] = totals.get(key, 0.0) + value
    result: Record = {}
    for key in sorted(totals):
        target = result
        for part in key[:-1]:
            target = cast(Record, target.setdefault(part, {}))
        target[key[-1]] = totals[key] / len(rows)
    return result


def counters(rows: Sequence[Record]) -> Record:
    derived = sum(integer(row["tokens"]) - integer(row["accepted"]) - 1 for row in rows)
    drafted = sum(integer(row["draft_rounds"]) for row in rows)
    draft = [row["draft_ms"] for row in rows]
    if derived <= 0 or drafted <= 0 or any(value is None for value in draft):
        raise ValueError("server round or drafter counters unavailable")
    predicted = sum(number(row["predicted_ms"]) for row in rows)
    drafter = sum(number(value) for value in draft) / drafted
    return {
        "requests": len(rows),
        "derived_rounds": derived,
        "round_ms": predicted / derived,
        "draft_counter_ms": drafter,
        "remainder_ms": predicted / derived - drafter,
        "accepted_per_round": sum(integer(row["accepted"]) for row in rows) / derived,
    }


def group_of(name: str) -> str | None:
    if ".layer" in name or (
        name.startswith("copies.") and name.split(".")[1] not in DRAFTER_PHASES
    ):
        return None
    for title, prefixes in GROUPS:
        if any(name.startswith(prefix) for prefix in prefixes):
            return title
    return None


def group(components: Record, gpu: Record) -> dict[str, float]:
    totals: dict[str, float] = {name: 0.0 for name, _ in GROUPS}
    for name, value in components.items():
        title = group_of(name)
        if title is not None:
            totals[title] += number(record(value)["ms"])
    for name, value in gpu.items():
        title = group_of(name)
        if title is not None:
            totals[title] += number(value)
    return totals


def summarise(
    plan: Plan,
    depth: int,
    band: Band,
    cells: dict[str, Store],
    traces: dict[str, list[Record]],
) -> Record:
    identifiers = [string(task["id"]) for task in band.tasks]
    order = [string(task["id"]) for task in plan.tasks()]
    requests = {order.index(identifier) + 2 for identifier in identifiers}
    output: Record = {"depth": depth, "band": band.name, "tasks": identifiers}
    outputs: dict[str, list[object]] = {}
    for name in PASSES:
        rows = [
            record(row)
            for row in sequence(read(cells[name].root / "rows.json")["rows"])
            if string(record(row)["task"]) in identifiers
        ]
        outputs[name] = [row["output_sha256"] for row in rows]
        entry: Record = {"server_counters": counters(rows)}
        if name in traces:
            selected = [value for value in traces[name] if integer(value["request"]) in requests]
            if not selected:
                raise ValueError("no complete traced rounds for band " + band.name)
            averaged = mean_record(selected)
            averaged.setdefault("components", {})
            averaged.setdefault("graph_instances", {})
            averaged.setdefault("gpu_ms", {})
            averaged.setdefault("api_calls", {})
            averaged.setdefault("api_ms", {})
            entry.update(averaged)
            entry["traced_rounds"] = len(selected)
            drafter_wall = number(averaged["drafter_wall_ms"])
            in_window = number(averaged["drafter_gpu_in_window_ms"])
            entry["drafter_host_or_idle_ms"] = drafter_wall - in_window
            entry["groups"] = group(record(averaged["components"]), record(averaged["gpu_ms"]))
        output[name] = entry
    output["outputs_identical"] = outputs["clean"] == outputs["phases"] == outputs["nodes"]
    clean = record(record(output["clean"])["server_counters"])
    phases, nodes = record(output["phases"]), record(output["nodes"])
    draft_gpu = sum(number(record(phases["gpu_ms"]).get(p, 0.0)) for p in DRAFTER_PHASES)
    node_gpu = sum(number(record(nodes["gpu_ms"]).get(p, 0.0)) for p in DRAFTER_PHASES)
    attributed = sum(
        number(value)
        for title, value in record(nodes["groups"]).items()
        if title not in {"outside graph nodes"}
    )
    output["reconciliation"] = {
        "clean_round_ms": clean["round_ms"],
        "clean_draft_counter_ms": clean["draft_counter_ms"],
        "phases_round_ms": record(phases["server_counters"])["round_ms"],
        "phases_draft_counter_ms": record(phases["server_counters"])["draft_counter_ms"],
        "phases_traced_round_ms": phases["wall_ms"],
        "phases_drafter_wall_ms": phases["drafter_wall_ms"],
        "phases_drafter_gpu_ms": draft_gpu,
        "nodes_drafter_gpu_ms": node_gpu,
        "nodes_attributed_ms": attributed,
        "nodes_attribution_coverage": attributed / node_gpu if node_gpu else None,
        "kernel_time_ratio_nodes_to_phases": node_gpu / draft_gpu if draft_gpu else None,
        "instrumentation_round_ms": number(record(phases["server_counters"])["round_ms"])
        - number(clean["round_ms"]),
        "unexplained_draft_ms": number(clean["draft_counter_ms"])
        - number(record(phases["phase_ms"])["draft"]),
    }
    return output


def export(profiler: Path, cell: Store) -> Path:
    report, database = cell.root / "trace.nsys-rep", cell.root / "trace.sqlite"
    if not report.is_file():
        raise ValueError("profiler did not publish a trace: " + str(cell.root))
    if not database.is_file():
        subprocess.run(
            [str(profiler), "export", "--type=sqlite", "--output=" + str(database), str(report)],
            capture_output=True,
            check=True,
            timeout=1800,
        )
    return database


def table(result: Record) -> str:
    lines: list[str] = []
    for entry in sequence(result["profiles"]):
        profile = record(entry)
        lines.append(
            f"== {profile['drafter']} depth {profile['depth']} {profile['band']} "
            f"(outputs identical: {profile['outputs_identical']})"
        )
        clean = record(record(profile["clean"])["server_counters"])
        phases, nodes = record(profile["phases"]), record(profile["nodes"])
        lines.append(
            f"clean clock: round {number(clean['round_ms']):.3f} ms = drafter counter "
            f"{number(clean['draft_counter_ms']):.3f} + rest {number(clean['remainder_ms']):.3f}; "
            f"accepted/round {number(clean['accepted_per_round']):.2f}"
        )
        walls = record(phases["phase_ms"])
        lines.append(
            "phases clock (instrumented): round {:.3f} = ".format(number(phases["wall_ms"]))
            + " + ".join(f"{name} {number(walls[name]):.3f}" for name in PHASES)
            + f" + other {number(phases['other_ms']):.3f}"
        )
        lines.append(
            f"drafter wall {number(phases['drafter_wall_ms']):.3f} ms: GPU busy inside "
            f"{number(phases['drafter_gpu_in_window_ms']):.3f}, host or idle "
            f"{number(phases['drafter_host_or_idle_ms']):.3f}"
        )
        calls = record(phases["api_calls"])
        lines.append(
            "CUDA calls per round (phases clock): "
            + ", ".join(
                f"{key} {number(value):.1f}"
                for key, value in calls.items()
                if key.split("/")[1] in {"graph_launch", "kernel_launch", "synchronize"}
            )
        )
        host = record(phases["host_exclusive_ms"])
        for phase in DRAFTER_PHASES:
            parts = [
                f"{key.split('/')[1]} {number(value):.3f}"
                for key, value in host.items()
                if key.startswith(phase + "/") and number(value) >= 0.0005
            ]
            lines.append(f"{phase} host exclusive (phases clock, ms): " + ", ".join(parts))
        lines.append(
            f"{'component (nodes clock, GPU ms/round)':48} {'ms':>8} {'MiB':>8} "
            f"{'GB/s':>6} {'MMAC/slot':>10}"
        )
        components = record(nodes["components"])
        for title, value in record(nodes["groups"]).items():
            members = [record(v) for k, v in components.items() if group_of(k) == title]
            size = sum(number(member["weight_bytes"]) for member in members)
            macs = sum(number(member["macs_per_slot"]) for member in members)
            rate = size / (number(value) * 1e6) if number(value) > 0 and size else 0.0
            lines.append(
                f"{title:48} {number(value):8.3f} {size / 2**20:8.1f} {rate:6.0f} {macs / 1e6:10.1f}"
            )
        layered = sorted(
            (int(name.rsplit("layer", 1)[1]), number(record(value)["ms"]))
            for name, value in components.items()
            if name.startswith("draft.attention.context.layer")
        )
        if layered:
            lines.append(
                "attention over context by layer (ms): "
                + ", ".join(f"{index}: {value:.3f}" for index, value in layered)
            )
        recon = record(profile["reconciliation"])
        lines.append(
            "reconcile: clean draft counter {:.3f} vs traced draft wall {:.3f} (unexplained {:.3f}); "
            "drafter GPU phases {:.3f} vs nodes {:.3f}; node coverage {}".format(
                number(recon["clean_draft_counter_ms"]),
                number(walls["draft"]),
                number(recon["unexplained_draft_ms"]),
                number(recon["phases_drafter_gpu_ms"]),
                number(recon["nodes_drafter_gpu_ms"]),
                recon["nodes_attribution_coverage"],
            )
        )
        lines.append("")
    return "\n".join(lines)


def run(plan: Plan, store: Store) -> dict[tuple[str, int, str], Store]:
    cells: dict[tuple[str, int, str], Store] = {}
    tasks = plan.tasks()
    for depth in plan.depths:
        for arm in plan.drafters:
            for name in PASSES:
                cell = Store(store.root / arm.name / ("d" + str(depth)) / name)
                serve: Serve = served if name == "clean" else tracer(plan, name == "nodes")
                spec = plan.spec if name == "clean" else plan.traced
                measure(spec, plan.exact, arm, depth, tasks, plan.cap, cell, serve)
                cells[(arm.name, depth, name)] = cell
    return cells


def profile_rounds(specification: Path, output: Path) -> Record:
    plan = Plan.load(specification)
    plan.spec.verify()
    plan.profiler.verify()
    for value in (plan.traced.binary, *plan.traced.libraries, *(a.drafter for a in plan.drafters)):
        if value is not None:
            value.verify()
    if output.exists():
        raise ValueError("round profile output must be new; preserve prior attempts")
    began = now()
    with device_lease("cuda"):
        store = Store(output)
        gpu = cuda_identity()
        store.atomic(store.root / "gpu.txt", gpu.encode())
        store.atomic(store.root / "specification.json", specification.read_bytes())
        try:
            cells = run(plan, store)
        except BaseException as error:
            store.atomic(
                store.root / "failure.json", canonical({"status": "failed", "error": str(error)})
            )
            raise
    measured = now() - began
    return analyse_output(plan, specification, store, cells, gpu, measured)


def analyse_output(
    plan: Plan,
    specification: Path,
    store: Store,
    cells: dict[tuple[str, int, str], Store],
    gpu: str,
    measured: float,
) -> Record:
    profiles: list[Record] = []
    for depth in plan.depths:
        for arm in plan.drafters:
            selected = {name: cells[(arm.name, depth, name)] for name in PASSES}
            traces = {
                name: analyse(export(plan.profiler.path, selected[name]), depth)
                for name in PASSES
                if name != "clean"
            }
            for band in plan.bands:
                profiles.append(
                    {"drafter": arm.name, **summarise(plan, depth, band, selected, traces)}
                )
    result: Record = {
        "schema": "round-profile-result-v1",
        "status": "complete",
        "specification": reference(Artifact.capture(specification)),
        "identities": {
            "comparison": reference(plan.comparison),
            "clean_runtime": reference(plan.spec.source),
            "clean_server": reference(plan.spec.binary),
            "diagnostic_runtime": reference(plan.diagnostic),
            "diagnostic_server": reference(plan.traced.binary),
            "profiler": reference(plan.profiler),
            "target": reference(plan.spec.target),
            "drafters": {arm.name: reference(arm.drafter) for arm in plan.drafters if arm.drafter},
            "workload": reference(plan.spec.workload),
        },
        "gpu": gpu,
        "code_sha256": source_identity(Path(__file__).resolve().parents[1]),
        "settings": {
            "context": plan.spec.context,
            "decode_consistent": "32" if plan.exact else "0",
            "output_cap": plan.cap,
            "depths": list(plan.depths),
            "bands": {band.name: [task["id"] for task in band.tasks] for band in plan.bands},
        },
        "clocks": {
            "clean": "unmodified runtime, server counters only: decode ms per derived round and the drafter generate counter",
            "phases": "diagnostic runtime under Nsight with CUDA graphs as served: NVTX phase walls and GPU activity per round",
            "nodes": "diagnostic runtime under Nsight with CUDA graphs disabled and one NVTX range per graph node: GPU kernel time per component",
        },
        "measured_seconds": measured,
        "profiles": profiles,
    }
    path = store.root / "round-profile.json"
    store.atomic(path, canonical(result))
    store.atomic(store.root / "round-profile.txt", table(result).encode())
    return {"record": reference(Artifact.capture(path)), "measured_seconds": measured}


def reanalyse(specification: Path, output: Path) -> Record:
    plan = Plan.load(specification)
    store = Store(output)
    cells = {
        (arm.name, depth, name): Store(output / arm.name / ("d" + str(depth)) / name)
        for depth in plan.depths
        for arm in plan.drafters
        for name in PASSES
    }
    gpu = (output / "gpu.txt").read_text()
    previous = output / "round-profile.json"
    measured = number(read(previous)["measured_seconds"]) if previous.is_file() else 0.0
    return analyse_output(plan, specification, store, cells, gpu, measured)
