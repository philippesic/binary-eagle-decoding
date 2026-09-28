#!/usr/bin/env python3
"""Join one native draft decoder graph with the CPU adapter's seed-step taps.

The native decoder group is selected by an exact `result_norm` column match
to the selected round's first head state, or by uniquely reconstructing it from
`eagle3_prenorm` and the frozen output norm when graph fusion hides the
intermediate. Every captured tensor is compared within that same execution.
This is a CPU diagnostic, not a training or performance gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from audit_recurrent_binary_capture import read_jsonl, sha256
from export_binary_rescue import Model

DECODER_TAPS = (
    "inp_embd",
    "embd_norm-0",
    "g_norm-0",
    "concat_embd-0",
    "Qcur-0",
    "Kcur-0",
    "Vcur-0",
    "Qcur_rope-0",
    "Kcur_rope-0",
    "kqv_out-0",
    "ffn_inp-0",
    "post_attn_norm-0",
    "ffn_out-0",
    "eagle3_prenorm-0",
    "result_norm",
)
QK_HEADS = {"Qcur-0": 32, "Qcur_rope-0": 32, "Kcur-0": 8, "Kcur_rope-0": 8}
PINNED_D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"


def _read_graph(index_path: Path, values_path: Path) -> tuple[list[dict], np.ndarray, dict]:
    indexed = read_jsonl(index_path)
    footer = indexed[-1]
    rows = indexed[:-1]
    if (
        not rows
        or footer.get("schema") != "eagle_draft_graph_v1"
        or footer.get("event") != "capture_end"
        or footer.get("status") != "complete"
        or footer.get("reason") != ""
        or footer.get("tensor_rows") != len(rows)
        or footer.get("execution_count") != len(rows)
    ):
        raise ValueError("native draft graph capture is missing a complete footer")
    values = np.memmap(values_path, dtype="<f4", mode="r")
    cursor = 0
    for row in rows:
        shape = row.get("ne")
        token_axis = row.get("token_axis")
        width = row.get("token_width")
        tokens = row.get("n_tokens")
        count = row.get("f32_count")
        if (
            row.get("schema") != "eagle_draft_graph_v1"
            or row.get("event") != "tensor"
            or not isinstance(row.get("tensor_name"), str)
            or not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size < 1 for size in shape)
            or type(token_axis) is not int
            or not 0 <= token_axis < len(shape)
            or type(width) is not int
            or type(tokens) is not int
            or type(count) is not int
            or width < 1
            or tokens < 1
            or count != width * tokens
            or count != int(np.prod(shape))
            or shape[token_axis] != tokens
            or row.get("f32_offset") != cursor
            or row.get("f32_bytes") != count * 4
        ):
            raise ValueError("native graph index has inconsistent tensor shape or F32 offsets")
        payload = values[cursor : cursor + count]
        if row["tensor_name"] == "result_output":
            if np.isnan(payload).any() or np.isposinf(payload).any():
                raise ValueError("native mapped draft logits contain NaN or positive infinity")
        elif not np.isfinite(payload).all():
            raise ValueError("native graph intermediate contains nonfinite values")
        cursor += count
    if cursor != values.size or footer.get("bytes_written") != cursor * 4:
        raise ValueError("native graph F32 payload is truncated")
    return rows, values, footer


def _column(row: dict, values: np.ndarray, token_index: int) -> np.ndarray:
    tokens, width = row["n_tokens"], row["token_width"]
    if not 0 <= token_index < tokens:
        raise ValueError("native graph token column is out of bounds")
    start = row["f32_offset"] + token_index * width
    return np.asarray(values[start : start + width], dtype=np.float32)


def _python_qk_to_native_rows(value: np.ndarray, heads: int) -> np.ndarray:
    if value.shape != (heads * 128,):
        raise ValueError("Q/K tap does not have pinned EAGLE head geometry")
    return value.reshape(heads, 2, 64).swapaxes(1, 2).reshape(-1)


def compare(
    capture_dir: Path,
    adapter_taps_path: Path,
    *,
    graph_index_path: Path | None = None,
    graph_values_path: Path | None = None,
    draft_gguf: Path | None = None,
    native_output_norm: np.ndarray | None = None,
    round_index: int = 0,
) -> dict:
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round index must be a nonnegative integer")
    capture_dir = Path(capture_dir)
    index = graph_index_path or capture_dir / "heads.draft_graph.jsonl"
    raw = graph_values_path or capture_dir / "heads.draft_graph.f32"
    records, values, footer = _read_graph(index, raw)
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in records:
        kind = row.get("group_kind")
        execution = row.get("group_execution")
        name = row["tensor_name"]
        if kind not in {"decoder", "encoder"} or type(execution) is not int or execution < 0:
            raise ValueError("native graph group identity is invalid")
        group = groups[(kind, execution)]
        if name in group:
            raise ValueError("duplicate native graph tensor within execution")
        group[name] = row
    if footer.get("decoder_groups") != sum(kind == "decoder" for kind, _ in groups) or footer.get(
        "encoder_groups"
    ) != sum(kind == "encoder" for kind, _ in groups):
        raise ValueError("native graph footer group counts differ from tensor index")
    head_rows = read_jsonl(capture_dir / "heads.jsonl")
    matches = [
        row for row in head_rows if row.get("round_index") == round_index and row.get("depth") == 0
    ]
    if (
        len(matches) != 1
        or matches[0].get("schema") != "eagle_head_state_v1"
        or matches[0].get("state_dim") != 2560
        or type(matches[0].get("state_row")) is not int
        or not 0 <= matches[0]["state_row"] < len(head_rows)
    ):
        raise ValueError("selected native head row is not one first proposal state")
    state_row = matches[0]["state_row"]
    native_states = np.memmap(capture_dir / "heads.f32", dtype="<f4", mode="r")
    if native_states.size != len(head_rows) * 2560:
        raise ValueError("native head-state file has wrong dimensions")
    head_row = native_states.reshape(-1, 2560)[state_row]
    candidates = []
    for (kind, execution), group in groups.items():
        if kind != "decoder" or "result_norm" not in group:
            continue
        result = group["result_norm"]
        for token_index in range(result["n_tokens"]):
            if np.array_equal(_column(result, values, token_index), head_row):
                candidates.append((execution, token_index))
    join_method = "exact_result_norm"
    join_error = None
    if not candidates:
        if native_output_norm is None:
            if draft_gguf is None:
                raise ValueError("draft GGUF is required when native result_norm is not captured")
            if sha256(draft_gguf) != PINNED_D_SHA256:
                raise ValueError("draft GGUF differs from pinned candidate D")
            native_output_norm = np.asarray(
                Model(draft_gguf).tensors["output_norm.weight"].data, dtype=np.float32
            )
        norm = np.asarray(native_output_norm, dtype=np.float64)
        if norm.shape != (2560,) or not np.isfinite(norm).all():
            raise ValueError("native output norm must be one finite 2560-wide row")
        ranked = []
        for (kind, execution), group in groups.items():
            if kind != "decoder" or "eagle3_prenorm-0" not in group:
                continue
            record = group["eagle3_prenorm-0"]
            for column in range(record["n_tokens"]):
                prenorm = _column(record, values, column).astype(np.float64)
                if prenorm.shape != norm.shape:
                    continue
                reconstructed = prenorm * (1.0 / np.sqrt(np.mean(prenorm * prenorm) + 1e-6)) * norm
                delta = reconstructed - head_row.astype(np.float64)
                ranked.append(
                    (
                        float(np.sqrt(np.mean(delta * delta))),
                        float(np.max(np.abs(delta))),
                        execution,
                        column,
                    )
                )
        ranked.sort()
        if not ranked or ranked[0][0] >= 1e-4 or (len(ranked) > 1 and ranked[1][0] < 1e-4):
            raise ValueError("native prenorm cannot be joined uniquely to head state row zero")
        _, max_abs, execution, token_index = ranked[0]
        candidates = [(execution, token_index)]
        join_method = "f32_output_norm_reconstruction"
        join_error = {
            "rms": ranked[0][0],
            "max_abs": max_abs,
            "next_best_rms": ranked[1][0] if len(ranked) > 1 else None,
        }
    if len(candidates) != 1:
        raise ValueError("native result_norm cannot be joined uniquely to head state row zero")
    execution, token_index = candidates[0]
    selected = groups[("decoder", execution)]
    tap_names = DECODER_TAPS if "result_norm" in selected else DECODER_TAPS[:-1]
    missing = set(tap_names) - set(selected)
    if missing:
        raise ValueError(f"native decoder graph lacks intermediate taps: {sorted(missing)}")
    first = selected["inp_embd"]
    last = selected.get("result_output", selected[tap_names[-1]])
    if first.get("group_begin") is not True or last.get("group_end") is not True:
        raise ValueError("native decoder graph group lacks begin/end markers")
    with np.load(adapter_taps_path, allow_pickle=False) as adapter:
        if missing := (set(tap_names) - set(adapter.files)):
            raise ValueError(f"CPU adapter lacks intermediate taps: {sorted(missing)}")
        differences = {}
        for name in tap_names:
            record = selected[name]
            if record["n_tokens"] != last["n_tokens"]:
                raise ValueError("native decoder tap token count differs within execution")
            native = _column(record, values, token_index).astype(np.float64)
            python = np.asarray(adapter[name], dtype=np.float64).reshape(-1)
            if native.shape != python.shape or not np.isfinite(python).all():
                raise ValueError(f"adapter/native {name} width or finiteness differs")
            if name in QK_HEADS:
                python = _python_qk_to_native_rows(python, QK_HEADS[name])
            delta = python - native
            native_norm = float(np.linalg.norm(native))
            differences[name] = {
                "elements": native.size,
                "native_dtype": record.get("dtype"),
                "max_abs": float(np.max(np.abs(delta))),
                "rms": float(np.sqrt(np.mean(delta * delta))),
                "relative_l2": float(np.linalg.norm(delta) / max(native_norm, 1e-12)),
                "exact_elements": int(np.count_nonzero(delta == 0)),
                "python_qk_rows_permuted_to_native": name in QK_HEADS,
            }
        fusion_candidates = []
        if "fc_out" in adapter.files:
            expected_fusion = np.asarray(adapter["fc_out"], dtype=np.float64).reshape(-1)
            for (kind, candidate_execution), group in groups.items():
                if kind != "encoder" or "fc_out" not in group:
                    continue
                record = group["fc_out"]
                for column in range(record["n_tokens"]):
                    native = _column(record, values, column).astype(np.float64)
                    if native.shape != expected_fusion.shape:
                        continue
                    delta = expected_fusion - native
                    fusion_candidates.append(
                        {
                            "encoder_group_execution": candidate_execution,
                            "token_column": column,
                            "max_abs": float(np.max(np.abs(delta))),
                            "rms": float(np.sqrt(np.mean(delta * delta))),
                            "relative_l2": float(
                                np.linalg.norm(delta) / max(float(np.linalg.norm(native)), 1e-12)
                            ),
                        }
                    )
        nearest_fusion = (
            min(fusion_candidates, key=lambda item: item["rms"]) if fusion_candidates else None
        )
        mapped_draft_head = None
        if "result_output" in selected and "draft_logits" in adapter.files:
            if draft_gguf is None or sha256(draft_gguf) != PINNED_D_SHA256:
                raise ValueError("pinned D GGUF is required for mapped draft-head comparison")
            mapping = np.asarray(Model(draft_gguf).tensors["d2t"].data, dtype=np.int64)
            native_logits = _column(selected["result_output"], values, token_index)
            python_logits = np.asarray(adapter["draft_logits"], dtype=np.float32).reshape(-1)
            if (
                native_logits.shape != (151_936,)
                or python_logits.shape != (32_000,)
                or mapping.shape != (32_000,)
                or not np.array_equal(np.flatnonzero(np.isfinite(native_logits)), mapping)
            ):
                raise ValueError("native result_output is not the pinned mapped draft head")
            delta = python_logits.astype(np.float64) - native_logits[mapping].astype(np.float64)
            mapped_draft_head = {
                "finite_mapped_target_ids": len(mapping),
                "max_abs": float(np.max(np.abs(delta))),
                "rms": float(np.sqrt(np.mean(delta * delta))),
                "exact_elements": int(np.count_nonzero(delta == 0)),
                "python_top_target_id": int(mapping[int(np.argmax(python_logits))]),
                "native_top_target_id": int(np.argmax(native_logits)),
            }
    return {
        "schema": "recurrent_draft_graph_cpu_join_v1",
        "status": (
            "first_head_graph_join_verified_numeric_drift_measured"
            if join_method == "exact_result_norm"
            else "first_head_prenorm_join_supported_numeric_drift_measured"
        ),
        "execution_device": "cpu",
        "round_index": round_index,
        "native_head_state_row": state_row,
        "join_method": join_method,
        "join_error": join_error,
        "native_group_execution": execution,
        "native_token_column": token_index,
        "native_decoder_groups": sum(kind == "decoder" for kind, _ in groups),
        "tap_order": list(tap_names),
        "differences": differences,
        "nearest_encoder_fc_out_unverified_join": nearest_fusion,
        "mapped_draft_head": mapped_draft_head,
        "source_sha256": {
            "graph_index": sha256(index),
            "graph_values": sha256(raw),
            "heads": sha256(capture_dir / "heads.jsonl"),
            "head_states": sha256(capture_dir / "heads.f32"),
            "adapter_taps": sha256(adapter_taps_path),
            **({"draft_gguf": sha256(draft_gguf)} if draft_gguf is not None else {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--adapter-taps", type=Path, required=True)
    parser.add_argument("--graph-index", type=Path)
    parser.add_argument("--graph-values", type=Path)
    parser.add_argument("--draft-gguf", type=Path)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = compare(
        args.capture_dir,
        args.adapter_taps,
        graph_index_path=args.graph_index,
        graph_values_path=args.graph_values,
        draft_gguf=args.draft_gguf,
        round_index=args.round_index,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "execution": report["native_group_execution"]}))


if __name__ == "__main__":
    main()
