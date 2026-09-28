#!/usr/bin/env python3
"""Ablate one joined native/student draft attention query and physical K/V cache.

Every mode uses the same captured F16 mask and the pinned ggml CPU Flash
Attention helper. Student Q/K are tested in both Python half-split order and
the native interleaved row order; V is never permuted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256
from audit_recurrent_draft_cache import audit as audit_cache
from check_recurrent_native_attention_operator import (
    KV_WIDTH,
    Q_WIDTH,
    _group_decoder_taps,
    _metrics,
    reconstruct_slots,
    replay,
)
from compare_recurrent_draft_graph import _column, _read_graph, compare

from w1a1_eagle.native_attention_oracle import NativeCPUAttentionOracle

SLOTS = 256
MODE_LABELS = (
    "native_q_native_kv",
    "student_q_raw_native_kv",
    "student_q_native_rows_native_kv",
    "native_q_student_k_raw_v",
    "native_q_student_k_native_rows_v",
    "student_qk_raw_student_v",
    "student_qk_native_rows_student_v",
)


def python_qk_to_native_rows(value: np.ndarray, heads: int) -> np.ndarray:
    """Swap [2,64] and [64,2] independently for each 128-channel head."""
    value = np.asarray(value)
    if type(heads) is not int or heads < 1 or value.shape[-2:] != (heads, 128):
        raise ValueError("Q/K operand must have [...,heads,128] geometry")
    return value.reshape(*value.shape[:-2], heads, 2, 64).swapaxes(-2, -1).reshape(value.shape)


def native_qk_to_python_rows(value: np.ndarray, heads: int) -> np.ndarray:
    """Inverse row permutation for the same pinned head geometry."""
    value = np.asarray(value)
    if type(heads) is not int or heads < 1 or value.shape[-2:] != (heads, 128):
        raise ValueError("Q/K operand must have [...,heads,128] geometry")
    return value.reshape(*value.shape[:-2], heads, 64, 2).swapaxes(-2, -1).reshape(value.shape)


def _load_student_operands(
    taps_path: Path, cache_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(taps_path, allow_pickle=False) as taps:
        if "Qcur_rope-0" not in taps.files:
            raise ValueError("adapter taps lack first seed Qcur_rope-0")
        query = np.asarray(taps["Qcur_rope-0"])
    if query.dtype != np.dtype("<f4") or query.shape != (Q_WIDTH,) or not np.isfinite(query).all():
        raise ValueError("student first-seed query must be finite F32 [4096]")
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != {"key", "value"}:
            raise ValueError("post-seed cache must contain exactly key and value")
        key = np.asarray(cache["key"])
        value = np.asarray(cache["value"])
    if (
        key.dtype != np.dtype("<f4")
        or value.dtype != np.dtype("<f4")
        or key.ndim != 3
        or key.shape != value.shape
        or key.shape[0] != 8
        or key.shape[2] != 128
        or not 1 <= key.shape[1] <= SLOTS
    ):
        raise ValueError("post-seed cache must be F32 [8,T,128] for T in 1..256")
    for name, array in (("key", key), ("value", value)):
        if not np.isfinite(array).all() or not np.array_equal(
            array, array.astype("<f2").astype("<f4")
        ):
            raise ValueError(f"post-seed {name} must be finite and F16-exact")
    return query.reshape(32, 128).copy(), key.copy(), value.copy()


def build_modes(
    native_query: np.ndarray,
    native_keys: np.ndarray,
    native_values: np.ndarray,
    student_query: np.ndarray,
    student_keys: np.ndarray,
    student_values: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Replace only the live physical prefix; preserve reserve slots and V order."""
    if (
        native_query.shape != (32, 128)
        or student_query.shape != (32, 128)
        or native_keys.shape != (SLOTS, KV_WIDTH)
        or native_values.shape != native_keys.shape
        or native_keys.dtype != np.dtype("<u2")
        or native_values.dtype != np.dtype("<u2")
        or student_keys.ndim != 3
        or student_keys.shape != student_values.shape
        or student_keys.shape[0] != 8
        or student_keys.shape[2] != 128
        or not 1 <= student_keys.shape[1] <= SLOTS
    ):
        raise ValueError("attention operands have incompatible pinned geometry")
    prefix = student_keys.shape[1]
    student_slot_keys = student_keys.transpose(1, 0, 2)
    student_slot_values = student_values.transpose(1, 0, 2)
    student_slot_keys_native = python_qk_to_native_rows(student_slot_keys, 8)
    swapped = {}
    for label, key in (("raw", student_slot_keys), ("native_rows", student_slot_keys_native)):
        keys = native_keys.copy()
        values = native_values.copy()
        keys[:prefix] = (
            np.ascontiguousarray(key).astype("<f2").view("<u2").reshape(prefix, KV_WIDTH)
        )
        values[:prefix] = student_slot_values.astype("<f2").view("<u2").reshape(prefix, KV_WIDTH)
        swapped[label] = (keys, values)
    raw_q = student_query
    native_row_q = python_qk_to_native_rows(student_query, 32)
    modes = {
        "native_q_native_kv": (native_query, native_keys, native_values),
        "student_q_raw_native_kv": (raw_q, native_keys, native_values),
        "student_q_native_rows_native_kv": (native_row_q, native_keys, native_values),
        "native_q_student_k_raw_v": (native_query, *swapped["raw"]),
        "native_q_student_k_native_rows_v": (native_query, *swapped["native_rows"]),
        "student_qk_raw_student_v": (raw_q, *swapped["raw"]),
        "student_qk_native_rows_student_v": (native_row_q, *swapped["native_rows"]),
    }
    if tuple(modes) != MODE_LABELS:
        raise AssertionError("attention ablation mode ledger changed")
    return modes


def _run_helper(
    helper: Path, query: np.ndarray, keys: np.ndarray, values: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    if (
        query.shape != (32, 128)
        or keys.shape != (SLOTS, KV_WIDTH)
        or values.shape != keys.shape
        or mask.shape != (SLOTS,)
    ):
        raise ValueError("helper input has wrong physical attention geometry")
    with tempfile.TemporaryDirectory(prefix="recurrent-attention-operands-") as directory:
        operand_dir = Path(directory)
        np.asarray(query, dtype="<f4").tofile(operand_dir / "query.f32")
        np.asarray(keys, dtype="<u2").tofile(operand_dir / "keys.f16")
        np.asarray(values, dtype="<u2").tofile(operand_dir / "values.f16")
        np.asarray(mask, dtype="<u2").tofile(operand_dir / "mask.f16")
        subprocess.run(
            [str(helper.resolve()), str(operand_dir), "1", str(SLOTS), "10"],
            check=True,
            capture_output=True,
            text=True,
        )
        output_path = operand_dir / "attention.f32"
        if not output_path.is_file() or output_path.stat().st_size != Q_WIDTH * 4:
            raise ValueError("ggml helper returned a truncated attention output")
        output = np.fromfile(output_path, dtype="<f4").reshape(32, 128)
    if not np.isfinite(output).all():
        raise ValueError("ggml helper returned nonfinite attention output")
    return output


def ablate(
    capture_dir: Path,
    adapter_taps: Path,
    adapter_cache: Path,
    draft_gguf: Path,
    helper: Path,
    *,
    round_index: int = 0,
) -> dict:
    capture_dir = Path(capture_dir)
    NativeCPUAttentionOracle(helper, threads=10)  # pin binary identity before any replay
    joined = compare(capture_dir, adapter_taps, draft_gguf=draft_gguf, round_index=round_index)
    execution = joined["native_group_execution"]
    column = joined["native_token_column"]
    control = replay(capture_dir, helper, execution=execution, column=column, threads=10)
    if control["metrics"]["bitwise_equal_elements"] != Q_WIDTH:
        raise ValueError("native attention control failed exact ggml replay")
    audit_cache(capture_dir)
    events = read_jsonl(capture_dir / "heads.draft_cache.jsonl")
    executions = [event for event in events if event.get("event") == "execution"]
    rows = [event for event in events if event.get("event") == "row"]
    cache_bits = np.memmap(capture_dir / "heads.draft_cache.f16", dtype="<u2", mode="r")
    mask_bits = np.memmap(capture_dir / "heads.draft_cache.mask", dtype="<u2", mode="r")
    native_keys, native_values, selected_masks, ledger = reconstruct_slots(
        executions, rows, cache_bits, mask_bits, execution
    )
    if native_keys.shape != (SLOTS, KV_WIDTH) or selected_masks.shape[1] != SLOTS:
        raise ValueError("selected native attention execution is not the pinned 256-slot graph")
    student_query, student_keys, student_values = _load_student_operands(
        adapter_taps, adapter_cache
    )
    prefix = student_keys.shape[1]
    selected_rounds = [
        row
        for row in read_jsonl(capture_dir / "forced-rounds.jsonl")
        if row.get("round_index") == round_index
    ]
    if len(selected_rounds) != 1:
        raise ValueError("selected round lacks unique native ancestry")
    native_round = selected_rounds[0]
    if (
        not isinstance(native_round.get("prefix_token_ids"), list)
        or len(native_round["prefix_token_ids"]) != prefix
    ):
        raise ValueError("student post-seed cache length differs from native round prefix")
    selected_row = ledger["selected_rows"][column]
    selected_mask = selected_masks[column]
    if (
        selected_row.get("position") != prefix - 1
        or selected_row.get("slot") != prefix - 1
        or selected_row.get("token_id") != native_round.get("seed_token_id")
        or ledger["visible_slots_per_query"][column] != prefix
        or not np.all(selected_mask.view("<f2")[:prefix] == 0)
        or not np.all(np.isneginf(selected_mask.view("<f2")[prefix:]))
    ):
        raise ValueError("student post-seed cache does not join the native physical prefix")
    records, graph_values, _ = _read_graph(
        capture_dir / "heads.draft_graph.jsonl", capture_dir / "heads.draft_graph.f32"
    )
    group = _group_decoder_taps(records)[execution]
    if (
        group["Qcur_rope-0"]["n_tokens"] != selected_masks.shape[0]
        or group["kqv_out-0"]["n_tokens"] != selected_masks.shape[0]
    ):
        raise ValueError("joined graph/cache query batch lengths differ")
    native_query = _column(group["Qcur_rope-0"], graph_values, column).reshape(32, 128)
    native_output = _column(group["kqv_out-0"], graph_values, column).reshape(1, 32, 128)
    modes = build_modes(
        native_query, native_keys, native_values, student_query, student_keys, student_values
    )
    metrics = {
        label: _metrics(
            _run_helper(helper, query, keys, values, selected_mask).reshape(1, 32, 128),
            native_output,
        )
        for label, (query, keys, values) in modes.items()
    }
    if metrics["native_q_native_kv"]["bitwise_equal_elements"] != Q_WIDTH:
        raise ValueError("ablation native control differs from exact replay")
    root = Path(__file__).resolve().parents[1]
    return {
        "schema": "recurrent_draft_attention_operand_ablation_v1",
        "execution_device": "cpu",
        "round_index": round_index,
        "native_join": {
            "method": joined["join_method"],
            "execution": execution,
            "column": column,
            "native_head_state_row": joined["native_head_state_row"],
            "join_error": joined["join_error"],
        },
        "query_batch_tokens": selected_masks.shape[0],
        "physical_kv_slots": SLOTS,
        "post_seed_cache_positions": prefix,
        "mask_visible_slots": ledger["visible_slots_per_query"][column],
        "selected_native_cache_row": selected_row,
        "helper_threads": 10,
        "mode_order": list(MODE_LABELS),
        "metrics": metrics,
        "source_sha256": {
            **control["source_sha256"],
            "adapter_taps": sha256(adapter_taps),
            "adapter_post_seed_cache": sha256(adapter_cache),
            "draft_gguf": sha256(draft_gguf),
            "rounds": sha256(capture_dir / "forced-rounds.jsonl"),
            "ablation_source": sha256(
                root / "scripts/check_recurrent_attention_operand_ablation.py"
            ),
            "real_step_source": sha256(root / "scripts/check_recurrent_real_step.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--adapter-taps", type=Path, required=True)
    parser.add_argument("--adapter-cache", type=Path, required=True)
    parser.add_argument("--draft-gguf", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = ablate(
        args.capture_dir,
        args.adapter_taps,
        args.adapter_cache,
        args.draft_gguf,
        args.helper,
        round_index=args.round_index,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {label: value["max_abs_difference"] for label, value in report["metrics"].items()}
        )
    )


if __name__ == "__main__":
    main()
