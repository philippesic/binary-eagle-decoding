import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from benchmarks.native import file_reference
from benchmarks.round_profile import (
    EDIT,
    Plan,
    Span,
    analyse,
    cost,
    counters,
    group,
    mean_record,
    parse_node,
    roles,
    rounds,
    table,
)
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.records import canonical

ATTN_Q = "node|0|MUL_MAT|Qcur-2|4096,4,1,1|0:blk.2.attn_q.weight:q4_0:2560,4096,1,1:5898240:1|1:attn_norm-2:f32:2560,4,1,1:40960:0"
FATTN = "node|1|FLASH_ATTN_EXT|node_7|128,32,4,1|0:q-2:f32:128,4,32,1:65536:0|1:cache_k_l2 (view):f16:128,256,8,1:524288:0|2:cache_v_l2 (view):f16:128,256,8,1:524288:0"
VOCAB = "node|2|MUL_MAT|result_output|248320,3,1,1|0:output.weight:q4_K:2560,248320,1,1:357580800:1|1:result_norm:f32:2560,3,1,1:30720:0"
UP = "node|3|W1A1_MUL_MAT|node_3|9216,4,1,1|0:blk.0.ffn_up.w1a1_packed:i32:80,9216,1,1:2949120:1|2:ffn_norm-0:f32:2560,4,1,1:40960:0"
GATE = "fused|4|W1A1_MUL_MAT|node_4|9216,4,1,1|0:blk.0.ffn_gate.w1a1_packed:i32:80,9216,1,1:2949120:1|2:ffn_norm-0:f32:2560,4,1,1:40960:0"
FUSION = "node|0|MUL_MAT|node_0|2560,2,1,1|0:fc.weight:q4_0:20480,2560,1,1:29491200:1|1:inp_target_features:f32:20480,2,1,1:163840:0"


def test_node_labels_give_weight_bytes_and_multiply_adds() -> None:
    query = parse_node(ATTN_Q)
    assert (query.op, query.name, query.shape) == ("MUL_MAT", "Qcur-2", (4096, 4, 1, 1))
    assert cost(query) == (2560 * 4096 * 4, 5898240, 4)
    attention = parse_node(FATTN)
    assert cost(attention) == (4 * 32 * 256 * 256, 0, 4)
    vocabulary = parse_node(VOCAB)
    assert cost(vocabulary)[0] // 3 == 2560 * 248320
    binary = parse_node(
        "node|9|W1A1_MUL_MAT|node_9|9216,4,1,1|0:blk.0.ffn_up.w1a1_packed:i32:80,9216,1,1:2949120:1"
        "|1:blk.0.ffn_up.w1a1_scale:f32:9216,1,1,1:36864:1|2:ffn_norm-0:f32:2560,4,1,1:40960:0"
        "|4:blk.0.ffn_up.w1ax_midpoint:f32:9216,1,1,1:36864:1"
    )
    assert cost(binary) == (2560 * 9216 * 4, 2949120 + 36864 + 36864, 4)
    rows = parse_node(
        "node|3|GET_ROWS|inp_noise_embd|2560,4,1,1|0:token_embd.weight:q4_K:2560,248320,1,1:357580800:1"
        "|1:inp_tokens:i32:4,1,1,1:16:0"
    )
    assert cost(rows) == (0, 357580800 // 248320 * 4, 4)
    with pytest.raises(ValueError, match="malformed"):
        parse_node("node|1|MUL_MAT")


def test_roles_come_from_weight_names_ops_and_following_norm_weights() -> None:
    labels = [
        "node|0|RMS_NORM|norm-0|2560,4,1,1|0:inpL:f32:2560,4,1,1:40960:0",
        "node|1|MUL|attn_norm-0|2560,4,1,1|0:norm-0:f32:2560,4,1,1:40960:0|1:blk.0.attn_norm.weight:f32:2560,1,1,1:10240:1",
        ATTN_Q,
        "node|3|ROPE|Qcur-0|128,32,4,1|0:Qcur:f32:128,32,4,1:65536:0",
        "node|4|SET_ROWS|cache_k_l0 (view)|1024,4,1,1|0:Kcur-0:f32:1024,4,1,1:16384:0|1:k_idxs:i64:4,1,1,1:32:0",
        FATTN,
        "node|6|MUL_MAT|node_6|2560,4,1,1|0:blk.0.ffn_down.weight:q4_K:9216,2560,1,1:13271040:1|1:x:f32:9216,4,1,1:1:0",
        "node|7|ADD|l_out-0|2560,4,1,1|0:a:f32:2560,4,1,1:1:0|1:b:f32:2560,4,1,1:1:0",
        VOCAB,
        "node|9|MUL_MAT|result_output|248320,3,1,1|0:token_embd.weight:q6_K:2560,248320,1,1:521000000:1",
        "node|10|GET_ROWS|inp_noise_embd|2560,4,1,1|0:token_embd.weight:q6_K:2560,248320,1,1:521000000:1",
    ]
    assert roles([parse_node(label) for label in labels]) == [
        "attention.norms",
        "attention.norms",
        "attention.qkv_projection",
        "attention.rope",
        "kv_update",
        "attention.context",
        "ffn.down",
        "elementwise",
        "vocabulary",
        "vocabulary",
        "embedding",
    ]


def test_rounds_skip_prompt_processing_and_number_requests() -> None:
    spans = [
        Span(0, 100, "srv.target|600"),
        Span(200, 260, "srv.draft"),
        Span(262, 300, "srv.target|4"),
        Span(400, 450, "srv.draft"),
        Span(600, 700, "srv.target|512"),
        Span(710, 720, "srv.target|3"),
        Span(800, 850, "srv.draft"),
        Span(900, 950, "srv.draft"),
    ]
    found = rounds(spans, 3)
    assert [(item.request, item.start, item.end) for item in found] == [
        (1, 200, 400),
        (2, 800, 900),
    ]


def write_trace(path: Path) -> None:
    ranges = [
        (0, 100, "srv.target|600"),
        (100, 110, "srv.process"),
        (200, 260, "srv.draft"),
        (204, 216, "llama.compute"),
        (206, 207, ATTN_Q),
        (207, 208, FATTN),
        (208, 209, VOCAB),
        (209, 212, UP),
        (211, 211, GATE),
        (216, 250, "llama.sync"),
        (250, 258, "dft.select"),
        (260, 262, "srv.checkpoint"),
        (262, 300, "srv.target|4"),
        (300, 305, "srv.process"),
        (301, 303, FUSION),
        (305, 320, "srv.verify"),
        (400, 450, "srv.draft"),
        (460, 480, "srv.target|4"),
        (600, 700, "srv.target|512"),
        (800, 850, "srv.draft"),
        (900, 950, "srv.draft"),
    ]
    launches = [(1, 206), (2, 207), (3, 208), (4, 214), (5, 302), (6, 263), (7, 805), (8, 210)]
    kernels = [
        (216, 226, 1, 1),
        (226, 236, 2, 1),
        (236, 246, 3, 1),
        (302, 312, 5, 1),
        (264, 290, 6, 1),
        (806, 810, 7, 1),
        (250, 258, 8, 1),
    ]
    with closing(sqlite3.connect(path)) as db:
        db.executescript(
            "CREATE TABLE StringIds (id INTEGER, value TEXT);"
            "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER, nameId INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, shortName INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER, bytes INTEGER, copyKind INTEGER);"
        )
        db.execute("INSERT INTO StringIds VALUES (1, 'kernel')")
        db.execute("INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel_v7000')")
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, ?, NULL, 7)",
            [(start, end, label) for start, end, label in ranges],
        )
        db.execute("INSERT INTO NVTX_EVENTS VALUES (1, 2, 59, 'srv.draft', NULL, 8)")
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, 7, 2)",
            [(moment, moment + 1, correlation) for correlation, moment in launches],
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (5, 6, 9, 8, 0)")
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", kernels)
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (6, 9, 9, 1)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (246, 248, 4, 3000000, 2)")
        db.commit()


def test_trace_analysis_attributes_kernels_to_phases_components_and_host_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.sqlite"
    write_trace(path)
    found = analyse(path, 3)
    assert [row["request"] for row in found] == [1, 2]
    first = found[0]
    assert first["wall_ms"] == pytest.approx(200e-6)
    phases = record(first["phase_ms"])
    assert phases["draft"] == pytest.approx(60e-6) and phases["target"] == pytest.approx(38e-6)
    assert first["other_ms"] == pytest.approx((200 - 60 - 2 - 38 - 5 - 15) * 1e-6)
    host = record(first["host_exclusive_ms"])
    assert host["draft/llama.sync"] == pytest.approx(34e-6)
    assert host["draft/llama.compute"] == pytest.approx(6e-6)
    assert host["draft/srv.draft"] == pytest.approx((60 - 12 - 34 - 8) * 1e-6)
    gpu = record(first["gpu_ms"])
    assert gpu["draft"] == pytest.approx(40e-6) and gpu["target"] == pytest.approx(26e-6)
    assert gpu["copies.draft.device_to_host"] == pytest.approx(2e-6)
    calls = {"draft/kernel_launch": 5.0, "process/kernel_launch": 1.0, "target/kernel_launch": 1.0}
    assert record(first["api_calls"]) == calls
    assert first["drafter_gpu_ms"] == pytest.approx(50e-6)
    assert first["drafter_gpu_in_window_ms"] == pytest.approx(43e-6)
    components = record(first["components"])
    assert set(components) == {
        "draft.attention.qkv_projection",
        "draft.attention.qkv_projection.layer2",
        "draft.attention.context",
        "draft.attention.context.layer2",
        "draft.vocabulary",
        "draft.ffn.gate+ffn.up",
        "handoff.fusion",
    }
    fused = record(components["draft.ffn.gate+ffn.up"])
    assert fused["ms"] == pytest.approx(8e-6) and fused["weight_bytes"] == 2 * 2949120
    assert fused["macs_per_slot"] == 2 * 2560 * 9216
    vocabulary = record(components["draft.vocabulary"])
    assert vocabulary["ms"] == pytest.approx(10e-6)
    assert vocabulary["weight_bytes"] == 357580800
    assert vocabulary["macs_per_slot"] == 2560 * 248320
    assert record(found[1]["gpu_ms"]) == {
        "draft": pytest.approx(4e-6),
        "draft.unlabelled": pytest.approx(4e-6),
    }
    averaged = mean_record(found)
    assert record(averaged["gpu_ms"])["draft"] == pytest.approx(22e-6)
    assert record(record(averaged["components"])["draft.vocabulary"])["ms"] == pytest.approx(5e-6)
    groups = group(record(averaged["components"]), record(averaged["gpu_ms"]))
    assert groups["final norm + vocabulary scoring"] == pytest.approx(5e-6)
    assert groups["copies (inputs, logits)"] == pytest.approx(1e-6)
    assert groups["outside graph nodes"] == pytest.approx(2e-6)


def test_server_counters_give_clean_round_split() -> None:
    rows: list[Record] = [
        {"tokens": 41, "accepted": 20, "draft_rounds": 21, "draft_ms": 42.0, "predicted_ms": 200.0},
        {"tokens": 21, "accepted": 0, "draft_rounds": 21, "draft_ms": 42.0, "predicted_ms": 200.0},
    ]
    result = counters(rows)
    assert result["derived_rounds"] == 40
    assert result["round_ms"] == pytest.approx(10.0)
    assert result["draft_counter_ms"] == pytest.approx(2.0)
    assert result["remainder_ms"] == pytest.approx(8.0)
    rows[0]["draft_ms"] = None
    with pytest.raises(ValueError, match="counters"):
        counters(rows)


def reference_to(path: Path, data: bytes) -> Record:
    path.write_bytes(data)
    return file_reference(path)


def plan_inputs(tmp_path: Path, repairs: list[str]) -> Path:
    library = tmp_path / "clean" / "libggml.so"
    library.parent.mkdir(parents=True)
    server = tmp_path / "clean" / "llama-server"
    clean = tmp_path / "clean" / "result.json"
    lib_ref, server_ref = reference_to(library, b"lib"), reference_to(server, b"server")
    clean_ref = reference_to(
        clean,
        canonical(
            {
                "status": "built",
                "source_bundle": {"sha256": "a"},
                "runtime_repairs": ["one"],
                "artifacts": [lib_ref, server_ref],
            }
        ),
    )
    (tmp_path / "diag").mkdir()
    diag_lib = reference_to(tmp_path / "diag" / "libggml.so", b"diag-lib")
    diag_server = reference_to(tmp_path / "diag" / "llama-server", b"diag-server")
    diag = reference_to(
        tmp_path / "diag" / "result.json",
        canonical(
            {
                "status": "built",
                "verification_trace": False,
                "source_bundle": {"sha256": "a"},
                "runtime_repairs": repairs,
                "artifacts": [diag_lib, diag_server],
            }
        ),
    )
    tasks = [
        {
            "id": name,
            "group": name,
            "role": "dev",
            "category": "code",
            "messages": [{"role": "user", "content": "x"}],
            "output_cap": 64,
            "ancestry": {"input_tokens": tokens},
        }
        for name, tokens in [("a", 100), ("b", 200), ("c", 9000), ("d", 12000)]
    ]
    workload = reference_to(tmp_path / "workload.json", canonical({"schema": 1, "tasks": tasks}))
    model = reference_to(tmp_path / "model.gguf", b"model")
    comparison = reference_to(
        tmp_path / "comparison.json",
        canonical(
            {
                "schema": "native-comparison-v1",
                "source": clean_ref,
                "binary": server_ref,
                "libraries": [lib_ref],
                "target": model,
                "arms": [
                    {"name": "target", "drafter": None},
                    {"name": "baseline", "drafter": model},
                ],
                "workload": workload,
                "repeats": 1,
                "warmups": 0,
                "context": 32768,
                "depth": 3,
                "role": "dev",
                "tier": "pilot",
            }
        ),
    )
    path = tmp_path / "spec.json"
    lengths: list[Record] = [
        {"name": "short", "minimum": 0, "maximum": 511, "tasks": 1},
        {"name": "long", "minimum": 8192, "maximum": 16384, "tasks": 2},
    ]
    depths: list[object] = [1, 3]
    drafters: Record = {"baseline": model, "s4": model}
    path.write_bytes(
        canonical(
            {
                "schema": "round-profile-v1",
                "comparison": comparison,
                "diagnostic_build": diag,
                "profiler": model,
                "drafters": drafters,
                "depths": depths,
                "lengths": lengths,
                "output_cap": 32,
                "exact": True,
            }
        )
    )
    return path


def test_plan_requires_clean_runtime_plus_only_the_trace_edit(tmp_path: Path) -> None:
    plan = Plan.load(plan_inputs(tmp_path, ["one", EDIT]))
    assert [arm.name for arm in plan.drafters] == ["baseline", "s4"]
    assert [[task["id"] for task in band.tasks] for band in plan.bands] == [["b"], ["c", "d"]]
    assert plan.traced.binary.path.parent.name == "diag"
    assert plan.traced.libraries[0].path.parent.name == "diag"
    plan.traced.verify()
    for repairs in (["one"], ["one", EDIT, "other"], ["two", EDIT]):
        with pytest.raises(ValueError, match="diagnostic build"):
            Plan.load(plan_inputs(tmp_path / str(len(repairs)) / repairs[0], repairs))


def test_table_reports_every_clock_under_its_own_name() -> None:
    nodes = {
        "phase_ms": {"draft": 1.0},
        "components": {
            "draft.vocabulary": {"ms": 0.5, "weight_bytes": 2**20, "macs_per_slot": 1e6}
        },
        "groups": {"final norm + vocabulary scoring": 0.5},
    }
    counters_row = {
        "round_ms": 9.0,
        "draft_counter_ms": 1.8,
        "remainder_ms": 7.2,
        "accepted_per_round": 1.5,
    }
    phases = {
        "server_counters": counters_row,
        "wall_ms": 9.5,
        "phase_ms": {"draft": 2.0, "checkpoint": 0.1, "target": 6.0, "process": 0.4, "verify": 0.5},
        "other_ms": 0.5,
        "drafter_wall_ms": 2.4,
        "drafter_gpu_in_window_ms": 1.4,
        "drafter_host_or_idle_ms": 1.0,
        "host_exclusive_ms": {"draft/dft.select": 0.3},
        "api_calls": {"draft/kernel_launch": 200.0},
    }
    text = table(
        {
            "profiles": [
                {
                    "drafter": "s4",
                    "depth": 3,
                    "band": "short",
                    "outputs_identical": True,
                    "clean": {"server_counters": counters_row},
                    "phases": phases,
                    "nodes": nodes,
                    "reconciliation": {
                        "clean_draft_counter_ms": 1.8,
                        "unexplained_draft_ms": -0.2,
                        "phases_drafter_gpu_ms": 1.4,
                        "nodes_drafter_gpu_ms": 1.5,
                        "nodes_attribution_coverage": 0.9,
                    },
                }
            ]
        }
    )
    assert "clean clock: round 9.000 ms" in text and "phases clock (instrumented)" in text
    assert "final norm + vocabulary scoring" in text and "1.0" in text
