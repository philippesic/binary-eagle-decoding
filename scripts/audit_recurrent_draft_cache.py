#!/usr/bin/env python3
"""Audit stored EAGLE draft K/V bytes and causal masks against graph writes.

This accepts only the single-sequence, contiguous-slot diagnostic capture.
It compares post-write cache bytes, not merely projected graph operands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_recurrent_binary_capture import read_jsonl, sha256
from compare_recurrent_draft_graph import _column, _read_graph


def audit(capture_dir: Path) -> dict:
    capture_dir = Path(capture_dir)
    index_path = capture_dir / "heads.draft_cache.jsonl"
    rows_path = capture_dir / "heads.draft_cache.f16"
    masks_path = capture_dir / "heads.draft_cache.mask"
    graph_index_path = capture_dir / "heads.draft_graph.jsonl"
    graph_values_path = capture_dir / "heads.draft_graph.f32"
    events = read_jsonl(index_path)
    if not events or events[-1].get("event") != "capture_end":
        raise ValueError("native draft cache capture lacks a complete footer")
    if any(event.get("schema") != "eagle_draft_cache_v1" for event in events):
        raise ValueError("native draft cache capture schema differs")
    footer = events[-1]
    executions = [event for event in events if event.get("event") == "execution"]
    cache_rows = [event for event in events if event.get("event") == "row"]
    if len(events) != len(executions) + len(cache_rows) + 1:
        raise ValueError("unknown draft cache capture event")
    if (
        not executions
        or footer.get("executions") != len(executions)
        or footer.get("rows") != len(cache_rows)
        or footer.get("row_bytes") != rows_path.stat().st_size
        or footer.get("mask_bytes") != masks_path.stat().st_size
        or rows_path.stat().st_size != 4096 * len(cache_rows)
    ):
        raise ValueError("draft cache counts or binary payload lengths differ")
    max_rows = footer.get("max_rows")
    max_bytes = footer.get("max_bytes")
    if (
        type(max_rows) is not int
        or not 1 <= max_rows <= 65536
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= 1024 * 1024 * 1024
        or len(cache_rows) > max_rows
        or footer["row_bytes"] + footer["mask_bytes"] > max_bytes
    ):
        raise ValueError("draft cache capture limits are missing, exceeded or invalid")

    graph_records, graph_values, graph_footer = _read_graph(graph_index_path, graph_values_path)
    if graph_footer.get("decoder_groups") != len(executions):
        raise ValueError("draft cache executions do not join graph decoder groups")
    groups: dict[int, dict[str, dict]] = {}
    for record in graph_records:
        if record.get("group_kind") == "decoder":
            groups.setdefault(record["group_execution"], {})[record["tensor_name"]] = record
    if set(groups) != set(range(len(executions))):
        raise ValueError("draft graph decoder executions are missing or duplicated")

    cache_bits = np.memmap(rows_path, dtype="<u2", mode="r")
    mask_bytes = np.memmap(masks_path, dtype="u1", mode="r")
    next_row_offset = next_mask_offset = row_index = 0
    matched_keys = matched_values = mask_prefix_rows = 0
    position_rewrites: dict[int, int] = {}
    cache_devices: set[str] = set()
    mask_devices: set[str] = set()
    for execution_index, execution in enumerate(executions):
        n_tokens = execution.get("n_tokens")
        n_kv = execution.get("n_kv")
        dtype_name = execution.get("mask_dtype")
        cache_buffer_type = execution.get("cache_buffer_type")
        cache_buffer_is_host = execution.get("cache_buffer_is_host")
        mask_buffer_type = execution.get("mask_buffer_type")
        mask_buffer_is_host = execution.get("mask_buffer_is_host")
        def classify_buffer(buffer_type: object, is_host: object) -> str:
            if not isinstance(buffer_type, str) or not buffer_type or type(is_host) is not bool:
                raise ValueError("draft cache backend buffer metadata is invalid")
            if buffer_type.startswith("CUDA"):
                return "cuda_host" if is_host else "cuda"
            if is_host:
                return "cpu"
            raise ValueError("draft cache capture used an unsupported non-host backend")

        cache_devices.add(classify_buffer(cache_buffer_type, cache_buffer_is_host))
        mask_devices.add(classify_buffer(mask_buffer_type, mask_buffer_is_host))
        if (
            execution.get("execution") != execution_index
            or type(n_tokens) is not int
            or not 1 <= n_tokens <= 512
            or type(n_kv) is not int
            or not 1 <= n_kv <= 2048
            or dtype_name not in {"f16", "f32"}
            or execution.get("mask_offset") != next_mask_offset
        ):
            raise ValueError("draft cache execution metadata is invalid")
        mask_dtype = np.dtype("<f2" if dtype_name == "f16" else "<f4")
        expected_mask_bytes = n_kv * n_tokens * mask_dtype.itemsize
        if execution.get("mask_bytes") != expected_mask_bytes:
            raise ValueError("draft cache mask dimensions differ from payload")
        mask = np.frombuffer(
            mask_bytes[next_mask_offset : next_mask_offset + expected_mask_bytes],
            dtype=mask_dtype,
        ).reshape(n_tokens, n_kv)
        next_mask_offset += expected_mask_bytes

        group = groups[execution_index]
        if (
            not {"inp_embd", "Kcur_rope-0", "Vcur-0"} <= set(group)
            or group["inp_embd"]["n_tokens"] != n_tokens
        ):
            raise ValueError("draft cache execution does not join native projection group")
        for column in range(n_tokens):
            if row_index >= len(cache_rows):
                raise ValueError("draft cache execution lacks rows")
            row = cache_rows[row_index]
            row_index += 1
            position = row.get("position")
            slot = row.get("slot")
            if (
                row.get("execution") != execution_index
                or row.get("column") != column
                or row.get("row_offset") != next_row_offset
                or row.get("key_bytes") != 2048
                or row.get("value_bytes") != 2048
                or type(position) is not int
                or type(slot) is not int
                or not 0 <= position < n_kv
                or slot != position
            ):
                raise ValueError("draft cache row ancestry, slot or offset is invalid")
            next_row_offset += 4096
            allowed = mask[column, : position + 1]
            blocked = mask[column, position + 1 :]
            if not (np.all(allowed == 0) and np.all(np.isneginf(blocked))):
                raise ValueError("draft attention mask is not the captured exact prefix")
            mask_prefix_rows += 1
            position_rewrites[position] = position_rewrites.get(position, 0) + 1

            k_expected = _column(group["Kcur_rope-0"], graph_values, column).astype("<f2")
            v_expected = _column(group["Vcur-0"], graph_values, column).astype("<f2")
            if k_expected.size != 1024 or v_expected.size != 1024:
                raise ValueError("draft projection width differs from pinned geometry")
            element_offset = row["row_offset"] // 2
            k_stored = cache_bits[element_offset : element_offset + 1024]
            v_stored = cache_bits[element_offset + 1024 : element_offset + 2048]
            matched_keys += int(np.count_nonzero(k_expected.view("<u2") == k_stored))
            matched_values += int(np.count_nonzero(v_expected.view("<u2") == v_stored))
    if row_index != len(cache_rows) or next_mask_offset != mask_bytes.size:
        raise ValueError("draft cache has trailing or unjoined binary rows")
    elements = len(cache_rows) * 1024
    if matched_keys != elements or matched_values != elements:
        raise ValueError("stored draft cache bytes differ from native projected writes")
    return {
        "schema": "recurrent_stored_draft_cache_audit_v2",
        "status": "stored_f16_rows_and_exact_prefix_masks_compared",
        "execution_device": next(iter(cache_devices)) if len(cache_devices) == 1 else "mixed",
        "mask_device": next(iter(mask_devices)) if len(mask_devices) == 1 else "mixed",
        "cache_buffer_types": sorted({event["cache_buffer_type"] for event in executions}),
        "mask_buffer_types": sorted({event["mask_buffer_type"] for event in executions}),
        "decoder_executions": len(executions),
        "captured_rows": len(cache_rows),
        "key_elements": elements,
        "key_equal_elements": matched_keys,
        "value_elements": elements,
        "value_equal_elements": matched_values,
        "exact_prefix_mask_rows": mask_prefix_rows,
        "rewritten_positions": sum(count > 1 for count in position_rewrites.values()),
        "source_sha256": {
            "cache_index": sha256(index_path),
            "cache_rows": sha256(rows_path),
            "cache_masks": sha256(masks_path),
            "graph_index": sha256(graph_index_path),
            "graph_values": sha256(graph_values_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = audit(args.capture_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("captured_rows", "key_equal_elements", "value_equal_elements")
            }
        )
    )


if __name__ == "__main__":
    main()
