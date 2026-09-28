#!/usr/bin/env python3
"""Replay candidate-D FFN on an identical native first-seed graph input.

This CPU-only diagnostic joins the first proposal's head state to one native
decoder graph execution, then feeds its captured post-attention norm directly
to the three candidate-D binary FFN projections. It does not rebuild attention,
run a target forward, train weights, or assess serving performance.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import D_SHA256, load_candidate_d_arrays  # noqa: E402

from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402

FFN_BASES = ("blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down")
WIDTH = 2560
INTERMEDIATE = 9728
EXPECTED_EXECUTION = 2
EXPECTED_COLUMN = 0


def _finite_vector(value: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (width,) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite F32 vector of width {width}")
    return array


def join_first_seed(
    records: list[dict],
    values: np.ndarray,
    head_row: np.ndarray,
    output_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prove a unique first-head join and return two taps from that execution."""
    head_row = _finite_vector(head_row, WIDTH, "native first head state")
    norm = _finite_vector(output_norm, WIDTH, "native output norm").astype(np.float64)
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in records:
        kind, execution, name = (
            row.get("group_kind"),
            row.get("group_execution"),
            row.get("tensor_name"),
        )
        if kind not in {"decoder", "encoder"} or type(execution) is not int or execution < 0:
            raise ValueError("native graph group identity is invalid")
        if not isinstance(name, str) or name in groups[(kind, execution)]:
            raise ValueError("native graph has a duplicate or invalid tensor name")
        groups[(kind, execution)][name] = row

    exact = []
    for (kind, execution), group in groups.items():
        if kind == "decoder" and "result_norm" in group:
            record = group["result_norm"]
            for column in range(record["n_tokens"]):
                if np.array_equal(_column(record, values, column), head_row):
                    exact.append((execution, column))
    if exact:
        if len(exact) != 1:
            raise ValueError("first head state has an ambiguous exact graph join")
        execution, column = exact[0]
        method = "exact_result_norm"
        join_error = {"rms": 0.0, "max_abs": 0.0, "next_best_rms": None}
    else:
        ranked = []
        for (kind, execution), group in groups.items():
            if kind != "decoder" or "eagle3_prenorm-0" not in group:
                continue
            record = group["eagle3_prenorm-0"]
            for column in range(record["n_tokens"]):
                prenorm = _column(record, values, column).astype(np.float64)
                if prenorm.shape != norm.shape or not np.isfinite(prenorm).all():
                    raise ValueError("native prenorm width or finiteness is invalid")
                reconstructed = prenorm * (1.0 / np.sqrt(np.mean(prenorm**2) + 1e-6)) * norm
                delta = reconstructed - head_row.astype(np.float64)
                ranked.append(
                    (
                        float(np.sqrt(np.mean(delta**2))),
                        float(np.max(np.abs(delta))),
                        execution,
                        column,
                    )
                )
        ranked.sort()
        if not ranked or ranked[0][0] >= 1e-4 or (len(ranked) > 1 and ranked[1][0] < 1e-4):
            raise ValueError("first head state lacks a unique reconstructed graph join")
        rms, max_abs, execution, column = ranked[0]
        method = "f32_output_norm_reconstruction"
        join_error = {
            "rms": rms,
            "max_abs": max_abs,
            "next_best_rms": ranked[1][0] if len(ranked) > 1 else None,
        }

    if (execution, column) != (EXPECTED_EXECUTION, EXPECTED_COLUMN):
        raise ValueError("first-seed graph join is not decoder execution 2, column 0")
    selected = groups[("decoder", execution)]
    if "post_attn_norm-0" not in selected or "ffn_out-0" not in selected:
        raise ValueError("joined native decoder execution lacks FFN input or output")
    input_record, output_record = selected["post_attn_norm-0"], selected["ffn_out-0"]
    if input_record["n_tokens"] != output_record["n_tokens"]:
        raise ValueError("joined native FFN taps have different token counts")
    if input_record.get("dtype") != "f32" or output_record.get("dtype") != "f32":
        raise ValueError("joined native FFN taps must be F32")
    native_input = _finite_vector(_column(input_record, values, column), WIDTH, "native FFN input")
    native_output = _finite_vector(
        _column(output_record, values, column), WIDTH, "native FFN output"
    )
    return (
        native_input,
        native_output,
        {
            "join_method": method,
            "join_error": join_error,
            "native_group_execution": execution,
            "native_token_column": column,
            "native_head_state_boundary": "native_output_norm_f32_before_head_operand_conversion",
            "ffn_input_tap": "post_attn_norm-0",
            "ffn_output_tap": "ffn_out-0",
            "input_identity": "one native F32 graph tap used unchanged for both arithmetic modes",
            "native_input_sha256": _array_sha256(native_input),
        },
    )


def _array_sha256(value: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("FFN comparison shapes or values are invalid")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    reference = expected.astype(np.float64)
    return {
        "elements": actual.size,
        "exact_elements": int(np.count_nonzero(actual.view(np.uint32) == expected.view(np.uint32))),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference)), 1e-12)),
    }


def _summary(value: torch.Tensor) -> dict:
    result = value.detach().to(torch.float64).numpy()
    return {
        "elements": result.size,
        "max_abs": float(np.max(np.abs(result))),
        "rms": float(np.sqrt(np.mean(result**2))),
        "zero_elements": int(np.count_nonzero(result == 0)),
    }


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def replay_ffn(
    native_input: np.ndarray,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    arithmetic: str,
    *,
    expected_intermediate: int | None = INTERMEDIATE,
) -> tuple[np.ndarray, dict]:
    """Instantiate only the three binary FFN linears and run one no-grad step."""
    native_input = np.asarray(native_input)
    if (
        native_input.ndim != 1
        or native_input.dtype != np.float32
        or not np.isfinite(native_input).all()
    ):
        raise ValueError("FFN input must be one finite F32 vector")
    if set(arrays) != set(FFN_BASES) or arithmetic not in {"native_order", "group_matmul"}:
        raise ValueError("FFN requires exactly gate/up/down arrays and known arithmetic")
    gate_packed, gate_scales = arrays[FFN_BASES[0]]
    up_packed, up_scales = arrays[FFN_BASES[1]]
    down_packed, down_scales = arrays[FFN_BASES[2]]
    intermediate = gate_packed.shape[0]
    if (
        (expected_intermediate is not None and intermediate != expected_intermediate)
        or up_packed.shape[0] != intermediate
        or down_packed.shape[0] != native_input.size
        or down_packed.shape[1] * 32 != intermediate
    ):
        raise ValueError("FFN arrays do not have the expected gate/up/down geometry")
    linears = {
        "gate": GroupedBinaryLinear.from_packed(
            gate_packed, gate_scales, in_features=native_input.size, arithmetic=arithmetic
        ),
        "up": GroupedBinaryLinear.from_packed(
            up_packed, up_scales, in_features=native_input.size, arithmetic=arithmetic
        ),
        "down": GroupedBinaryLinear.from_packed(
            down_packed, down_scales, in_features=intermediate, arithmetic=arithmetic
        ),
    }
    with torch.no_grad():
        x = torch.from_numpy(np.array(native_input, dtype=np.float32, copy=True))
        gate = linears["gate"](x)
        up = linears["up"](x)
        activated_gate = F.silu(gate)
        fused = activated_gate * up
        output = linears["down"](fused)
        if not all(torch.isfinite(value).all() for value in (gate, up, fused, output)):
            raise ValueError("FFN replay produced nonfinite values")
        summaries = {
            "gate": _summary(gate),
            "up": _summary(up),
            "silu_gate_times_up": _summary(fused),
        }
    return output.numpy().copy(), summaries


def audit(capture_dir: Path, candidate_d: Path, native_source: Path) -> dict:
    capture_dir, candidate_d, native_source = map(Path, (capture_dir, candidate_d, native_source))
    index = capture_dir / "heads.draft_graph.jsonl"
    raw = capture_dir / "heads.draft_graph.f32"
    heads_path = capture_dir / "heads.jsonl"
    states_path = capture_dir / "heads.f32"
    if not native_source.is_file():
        raise ValueError("native eagle3.cpp source is required for provenance")
    arrays, d_audit = load_candidate_d_arrays(candidate_d)
    if d_audit["gguf_sha256"] != D_SHA256:
        raise ValueError("candidate D differs from the pinned GGUF")
    records, values, footer = _read_graph(index, raw)
    heads = read_jsonl(heads_path)
    first = [row for row in heads if row.get("round_index") == 0 and row.get("depth") == 0]
    if (
        len(first) != 1
        or first[0].get("schema") != "eagle_head_state_v1"
        or first[0].get("state_dim") != WIDTH
        or first[0].get("state_boundary") != "native_output_norm_f32_before_head_operand_conversion"
        or type(first[0].get("state_row")) is not int
        or not 0 <= first[0]["state_row"] < len(heads)
    ):
        raise ValueError("capture lacks one valid first-round depth-zero head state")
    states = np.memmap(states_path, dtype="<f4", mode="r")
    if states.size != len(heads) * WIDTH:
        raise ValueError("native head-state payload length differs from index")
    head_row = states.reshape(-1, WIDTH)[first[0]["state_row"]]
    model = Model(candidate_d)
    output_norm = np.asarray(model.tensors["output_norm.weight"].data, dtype=np.float32)
    native_input, native_output, join = join_first_seed(records, values, head_row, output_norm)
    ffn_arrays = {base: arrays[base] for base in FFN_BASES}
    results = {}
    outputs = {}
    for arithmetic in ("native_order", "group_matmul"):
        output, summaries = replay_ffn(native_input, ffn_arrays, arithmetic)
        results[arithmetic] = {
            "versus_native_ffn_out": _metrics(output, native_output),
            "intermediate_summaries": summaries,
            "output_sha256": _array_sha256(output),
        }
        outputs[arithmetic] = output
    return {
        "schema": "recurrent_binary_ffn_same_input_cpu_v1",
        "execution_device": "cpu",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "precision": "A16 cast before each W1 group; F32 group sums, scales, SiLU and product",
        "arithmetic": ["native_order", "group_matmul"],
        "native_graph_footer": {
            "decoder_groups": footer.get("decoder_groups"),
            "encoder_groups": footer.get("encoder_groups"),
        },
        "native_head_state_row": first[0]["state_row"],
        "join": join,
        "native_ffn_output_sha256": _array_sha256(native_output),
        "results": results,
        "group_matmul_versus_native_order": _metrics(
            outputs["group_matmul"], outputs["native_order"]
        ),
        "source_sha256": {
            "candidate_d": d_audit["gguf_sha256"],
            "native_eagle3_cpp": sha256(native_source),
            "graph_index": sha256(index),
            "graph_values": sha256(raw),
            "head_index": sha256(heads_path),
            "head_states": sha256(states_path),
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "scope": (
            "one first-round seed column; no context rebuild, optimizer, GPU or target forward"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new path")
    torch.set_num_threads(1)
    result = audit(args.capture_dir, args.candidate_d, args.native_source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "native_order": result["results"]["native_order"]["versus_native_ffn_out"],
                "group_matmul": result["results"]["group_matmul"]["versus_native_ffn_out"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
