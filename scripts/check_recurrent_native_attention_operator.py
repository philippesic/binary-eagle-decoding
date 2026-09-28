#!/usr/bin/env python3
"""Replay captured EAGLE CPU Flash Attention with archived physical F16 K/V.

The selected execution is computed as its complete native query batch. Cache
slots are reconstructed from chronological post-write rows; a visible slot
without known bytes is an error. No model forward or new inference is run.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from audit_recurrent_binary_capture import read_jsonl, sha256
from audit_recurrent_draft_cache import audit as audit_cache
from compare_recurrent_draft_graph import _column, _read_graph

Q_WIDTH = 32 * 128
KV_WIDTH = 8 * 128
SLOT_BYTES = 4096


def _group_decoder_taps(records: list[dict]) -> dict[int, dict[str, dict]]:
    groups: dict[int, dict[str, dict]] = {}
    for row in records:
        if row.get("group_kind") != "decoder":
            continue
        execution = row.get("group_execution")
        name = row.get("tensor_name")
        if type(execution) is not int or execution < 0 or not isinstance(name, str):
            raise ValueError("invalid decoder graph identity")
        group = groups.setdefault(execution, {})
        if name in group:
            raise ValueError("duplicate decoder graph tap")
        group[name] = row
    return groups


def reconstruct_slots(
    executions: list[dict],
    rows: list[dict],
    cache_bits: np.ndarray,
    mask_bits: np.ndarray,
    selected_execution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply completed executions' writes; return physical slots and selected mask."""
    if type(selected_execution) is not int or not 0 <= selected_execution < len(executions):
        raise ValueError("selected execution is out of bounds")
    if any(item.get("execution") != index for index, item in enumerate(executions)):
        raise ValueError("cache execution ordinals are not consecutive")
    slots = executions[selected_execution]["n_kv"]
    if type(slots) is not int or not 1 <= slots <= 2048:
        raise ValueError("invalid selected K/V slot count")
    keys = np.zeros((slots, KV_WIDTH), dtype="<u2")
    values = np.zeros_like(keys)
    known = np.zeros(slots, dtype=bool)
    selected_mask: np.ndarray | None = None
    selected_rows: list[dict] = []
    row_cursor = 0
    for index, execution in enumerate(executions[: selected_execution + 1]):
        n_tokens = execution.get("n_tokens")
        n_kv = execution.get("n_kv")
        if type(n_tokens) is not int or not 1 <= n_tokens <= 512 or n_kv != slots:
            raise ValueError("cache geometry changed before selected execution")
        if execution.get("mask_dtype") != "f16":
            raise ValueError("this replay requires captured F16 masks")
        offset = execution.get("mask_offset")
        length = execution.get("mask_bytes")
        if type(offset) is not int or type(length) is not int or length != 2 * n_tokens * slots:
            raise ValueError("selected mask dimensions or offset differ")
        if offset < 0 or offset + length > mask_bits.nbytes or offset % 2:
            raise ValueError("selected mask payload is truncated")
        execution_rows = rows[row_cursor : row_cursor + n_tokens]
        if len(execution_rows) != n_tokens:
            raise ValueError("cache execution lacks row writes")
        row_cursor += n_tokens
        seen_slots: set[int] = set()
        for column, row in enumerate(execution_rows):
            slot = row.get("slot")
            offset_bytes = row.get("row_offset")
            if (
                row.get("execution") != index
                or row.get("column") != column
                or type(slot) is not int
                or not 0 <= slot < slots
                or slot in seen_slots
                or row.get("position") != slot
                or type(offset_bytes) is not int
                or offset_bytes != (row_cursor - n_tokens + column) * SLOT_BYTES
            ):
                raise ValueError("ambiguous or invalid physical cache write")
            seen_slots.add(slot)
            start = offset_bytes // 2
            if start + SLOT_BYTES // 2 > cache_bits.size:
                raise ValueError("physical cache payload is truncated")
            keys[slot] = cache_bits[start : start + KV_WIDTH]
            values[slot] = cache_bits[start + KV_WIDTH : start + 2 * KV_WIDTH]
            known[slot] = True
        if index == selected_execution:
            selected_rows = execution_rows
            selected_mask = mask_bits[offset // 2 : (offset + length) // 2].reshape(n_tokens, slots)
    assert selected_mask is not None
    mask = selected_mask.view("<f2")
    if np.isnan(mask).any() or np.isposinf(mask).any():
        raise ValueError("native attention mask has NaN or positive infinity")
    visible = np.isfinite(mask)
    if np.any(visible & ~known[None, :]):
        raise ValueError("mask exposes a K/V slot without a captured write")
    if np.any(mask[visible] != 0) or np.any(~np.isneginf(mask[~visible])):
        raise ValueError("native attention mask is not exact-prefix zero/-infinity")
    return (
        keys,
        values,
        selected_mask.copy(),
        {
            "selected_rows": selected_rows,
            "known_slots": int(known.sum()),
            "visible_slots_per_query": visible.sum(axis=1).astype(int).tolist(),
            "writes_applied": row_cursor,
        },
    )


def _metrics(replayed: np.ndarray, native: np.ndarray) -> dict:
    if replayed.shape != native.shape or replayed.ndim != 3 or replayed.shape[1:] != (32, 128):
        raise ValueError("replayed/native attention geometry differs")
    if not np.isfinite(replayed).all() or not np.isfinite(native).all():
        raise ValueError("attention output has nonfinite elements")
    delta = replayed.astype(np.float64) - native.astype(np.float64)
    per_head = []
    for head in range(32):
        lhs = delta[:, head, :]
        rhs = native[:, head, :].astype(np.float64)
        denom = float(np.linalg.norm(rhs))
        per_head.append(float(np.linalg.norm(lhs) / denom) if denom else math.inf)
    return {
        "elements": int(replayed.size),
        "bitwise_equal_elements": int(np.count_nonzero(replayed.view("<u4") == native.view("<u4"))),
        "max_abs_difference": float(np.max(np.abs(delta))),
        "rms_difference": float(np.sqrt(np.mean(delta * delta))),
        "per_head_relative_l2": per_head,
    }


def replay(
    capture_dir: Path,
    helper: Path,
    *,
    execution: int = 2,
    column: int | None = 0,
    threads: int = 10,
) -> dict:
    capture_dir = Path(capture_dir)
    helper = Path(helper)
    if not helper.is_file() or type(threads) is not int or threads < 1:
        raise ValueError("native helper or thread count is invalid")
    audited = audit_cache(capture_dir)
    index_path = capture_dir / "heads.draft_cache.jsonl"
    rows_path = capture_dir / "heads.draft_cache.f16"
    mask_path = capture_dir / "heads.draft_cache.mask"
    graph_index = capture_dir / "heads.draft_graph.jsonl"
    graph_payload = capture_dir / "heads.draft_graph.f32"
    events = read_jsonl(index_path)
    executions = [event for event in events if event.get("event") == "execution"]
    rows = [event for event in events if event.get("event") == "row"]
    records, graph_values, _ = _read_graph(graph_index, graph_payload)
    groups = _group_decoder_taps(records)
    if execution not in groups:
        raise ValueError("selected graph execution is missing")
    group = groups[execution]
    if not {"Qcur_rope-0", "kqv_out-0", "inp_embd"} <= set(group):
        raise ValueError("selected graph execution lacks attention taps")
    n_tokens = executions[execution]["n_tokens"]
    if any(
        group[name]["n_tokens"] != n_tokens for name in ("Qcur_rope-0", "kqv_out-0", "inp_embd")
    ):
        raise ValueError("graph and cache execution token batches differ")
    if (
        group["Qcur_rope-0"]["token_width"] != Q_WIDTH
        or group["kqv_out-0"]["token_width"] != Q_WIDTH
    ):
        raise ValueError("selected attention tap has wrong width")
    if column is not None and (type(column) is not int or not 0 <= column < n_tokens):
        raise ValueError("selected query column is out of bounds")
    cache_bits = np.memmap(rows_path, dtype="<u2", mode="r")
    mask_bits = np.memmap(mask_path, dtype="<u2", mode="r")
    keys, values, mask, ledger = reconstruct_slots(
        executions, rows, cache_bits, mask_bits, execution
    )
    query = np.stack([_column(group["Qcur_rope-0"], graph_values, i) for i in range(n_tokens)])
    native = np.stack([_column(group["kqv_out-0"], graph_values, i) for i in range(n_tokens)])
    if not np.isfinite(query).all():
        raise ValueError("captured query has nonfinite values")
    with tempfile.TemporaryDirectory(prefix="native-recurrent-attn-") as temporary:
        operands = Path(temporary)
        np.asarray(query, dtype="<f4").tofile(operands / "query.f32")
        keys.tofile(operands / "keys.f16")
        values.tofile(operands / "values.f16")
        mask.tofile(operands / "mask.f16")
        subprocess.run(
            [str(helper.resolve()), str(operands), str(n_tokens), str(keys.shape[0]), str(threads)],
            check=True,
            capture_output=True,
            text=True,
        )
        result_path = operands / "attention.f32"
        if result_path.stat().st_size != n_tokens * Q_WIDTH * 4:
            raise ValueError("native helper output has wrong byte count")
        result = np.fromfile(result_path, dtype="<f4").reshape(n_tokens, 32, 128)
    native = native.reshape(n_tokens, 32, 128)
    result_for_metrics = result if column is None else result[column : column + 1]
    native_for_metrics = native if column is None else native[column : column + 1]
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "cache_index": index_path,
        "cache_rows": rows_path,
        "cache_masks": mask_path,
        "graph_index": graph_index,
        "graph_values": graph_payload,
        "capture_manifest": capture_dir / "manifest.json",
        "capture_server_log": capture_dir / "server.log",
        "helper_source": root / "scripts/native_recurrent_attention.cpp",
        "replay_source": root / "scripts/check_recurrent_native_attention_operator.py",
        "helper_binary": helper,
        "ggml_cmake_cache": helper.resolve().parent.parent / "CMakeCache.txt",
        "ggml_cpu_ops_source": root / "third_party/llama.cpp/ggml/src/ggml-cpu/ops.cpp",
        "eagle_graph_source": root / "third_party/llama.cpp/src/models/eagle3.cpp",
        "attention_graph_source": root / "third_party/llama.cpp/src/llama-graph.cpp",
    }
    cpu_dylib = helper.resolve().parent / "libggml-cpu.dylib"
    base_dylib = helper.resolve().parent / "libggml-base.dylib"
    if not cpu_dylib.exists() or not base_dylib.exists():
        raise ValueError("helper ggml CPU libraries are missing")
    source_paths["ggml_cpu_library"] = cpu_dylib
    source_paths["ggml_base_library"] = base_dylib
    log = (capture_dir / "server.log").read_text(errors="replace")
    match = re.search(r"n_threads = (\d+) \(n_threads_batch = (\d+)\)", log)
    captured_threads = [int(match.group(1)), int(match.group(2))] if match else None
    if captured_threads != [threads, threads] or "Flash Attention enabled" not in log:
        raise ValueError("helper threads or Flash Attention differ from the native CPU capture")
    capture_manifest = json.loads((capture_dir / "manifest.json").read_text())
    if capture_manifest.get("execution_device") != "cpu":
        raise ValueError("native source capture was not CPU")
    return {
        "schema": "recurrent_native_cpu_flash_attention_replay_v1",
        "execution_device": "cpu",
        "host_architecture": platform.machine(),
        "execution": execution,
        "selected_column": column,
        "query_batch_tokens": n_tokens,
        "physical_kv_slots": int(keys.shape[0]),
        "captured_threads": captured_threads,
        "helper_threads": threads,
        "flash_attention_enabled_log": True,
        "operator": {
            "scale": 1.0 / math.sqrt(128),
            "max_alibi_bias": 0.0,
            "logit_softcap": 0.0,
            "sinks": None,
            "n_kv_max": 0,
            "accumulator": "GGML_PREC_F32",
            "query_dtype": "f32",
            "stored_key_value_dtype": "f16",
            "mask_dtype": "f16",
        },
        "ledger": ledger,
        "metrics": _metrics(result_for_metrics, native_for_metrics),
        "cache_audit_status": audited["status"],
        "captured_model_and_binary_sha256": capture_manifest["source_sha256"],
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
    }


def replay_all(capture_dir: Path, helper: Path, *, threads: int = 10) -> dict:
    events = read_jsonl(Path(capture_dir) / "heads.draft_cache.jsonl")
    executions = [event for event in events if event.get("event") == "execution"]
    if not executions:
        raise ValueError("native cache capture has no decoder executions")
    reports = [
        replay(capture_dir, helper, execution=execution, column=None, threads=threads)
        for execution in range(len(executions))
    ]
    total = sum(report["metrics"]["elements"] for report in reports)
    return {
        "schema": "recurrent_native_cpu_flash_attention_all_executions_v1",
        "execution_device": "cpu",
        "host_architecture": platform.machine(),
        "decoder_executions": len(reports),
        "query_rows": sum(report["query_batch_tokens"] for report in reports),
        "attention_elements": total,
        "bitwise_equal_elements": sum(
            report["metrics"]["bitwise_equal_elements"] for report in reports
        ),
        "max_abs_difference": max(report["metrics"]["max_abs_difference"] for report in reports),
        "rms_difference": math.sqrt(
            sum(
                report["metrics"]["rms_difference"] ** 2 * report["metrics"]["elements"]
                for report in reports
            )
            / total
        ),
        "max_execution_head_relative_l2": max(
            max(report["metrics"]["per_head_relative_l2"]) for report in reports
        ),
        "operator": reports[0]["operator"],
        "captured_threads": reports[0]["captured_threads"],
        "helper_threads": threads,
        "captured_model_and_binary_sha256": reports[0]["captured_model_and_binary_sha256"],
        "source_sha256": reports[0]["source_sha256"],
        "execution_reports": [
            {
                "execution": report["execution"],
                "query_batch_tokens": report["query_batch_tokens"],
                "ledger": report["ledger"],
                "metrics": report["metrics"],
            }
            for report in reports
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--execution", type=int, default=2)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--all-columns", action="store_true")
    parser.add_argument("--all-executions", action="store_true")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    if args.all_executions:
        result = replay_all(args.capture_dir, args.helper, threads=args.threads)
    else:
        result = replay(
            args.capture_dir,
            args.helper,
            execution=args.execution,
            column=None if args.all_columns else args.column,
            threads=args.threads,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.all_executions:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "decoder_executions",
                        "query_rows",
                        "attention_elements",
                        "bitwise_equal_elements",
                        "max_abs_difference",
                        "rms_difference",
                        "max_execution_head_relative_l2",
                    )
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "execution": result["execution"],
                    "column": result["selected_column"],
                    "metrics": result["metrics"],
                }
            )
        )


if __name__ == "__main__":
    main()
