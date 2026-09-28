#!/usr/bin/env python3
"""Compare CPU student K/V writes with native graph projections for a round.

Rows are joined by exact borrowed token embeddings and near-identical fused
feature norms before comparing F16-rounded key/value operands. This checks
projected cache-write values, not the native cache bytes after rollback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_recurrent_binary_capture import read_jsonl, sha256
from check_recurrent_real_step import _build_drafter, _retained_feature_indices
from compare_recurrent_draft_graph import _column, _python_qk_to_native_rows, _read_graph

from w1a1_eagle.frozen_operands import FrozenOperands


def compare_kv_writes(
    capture_dir: Path,
    seed_graph_join: Path,
    target_gguf: Path,
    draft_gguf: Path,
    drafter_config: Path,
) -> dict:
    capture_dir = Path(capture_dir)
    joined = json.loads(seed_graph_join.read_text())
    round_index = joined.get("round_index")
    seed_execution = joined.get("native_group_execution")
    if (
        joined.get("schema") != "recurrent_draft_graph_cpu_join_v1"
        or type(round_index) is not int
        or type(seed_execution) is not int
        or round_index < 0
        or seed_execution < 1
    ):
        raise ValueError("seed graph join does not identify a valid native round")
    index_path = capture_dir / "heads.draft_graph.jsonl"
    values_path = capture_dir / "heads.draft_graph.f32"
    if joined.get("source_sha256", {}).get("graph_index") != sha256(index_path) or joined.get(
        "source_sha256", {}
    ).get("graph_values") != sha256(values_path):
        raise ValueError("seed graph join differs from native graph source")
    records, values, _ = _read_graph(index_path, values_path)
    groups: dict[int, dict[str, dict]] = {}
    for row in records:
        if row.get("group_kind") == "decoder" and row.get("group_execution") < seed_execution:
            groups.setdefault(row["group_execution"], {})[row["tensor_name"]] = row
    if not groups:
        raise ValueError("no native decoder context exists before selected seed")
    rounds = read_jsonl(capture_dir / "forced-rounds.jsonl")
    selected = [row for row in rounds if row.get("round_index") == round_index]
    if len(selected) != 1:
        raise ValueError("native round selection is ambiguous")
    round_record = selected[0]
    prefix = round_record["prefix_token_ids"]
    events = read_jsonl(capture_dir / "heads.target_features.jsonl")
    indices = _retained_feature_indices(events, prefix, round_record["task_id"])
    raw_values = np.memmap(capture_dir / "heads.target_features.f32", dtype="<f4", mode="r")
    if raw_values.size % 7680 or max(indices) >= raw_values.size // 7680:
        raise ValueError("retained feature row exceeds native F32 payload")
    raw_values = raw_values.reshape(-1, 7680)
    operands = FrozenOperands(target_gguf, draft_gguf)
    adapter = _build_drafter(drafter_config, draft_gguf, operands, "native_order")
    cache = adapter.new_cache()
    comparisons = []
    with torch.no_grad():
        for position in range(len(prefix) - 1):
            token = prefix[position + 1]
            raw = torch.from_numpy(np.array(raw_values[indices[position]], copy=True))
            fused = adapter.encode_feature(raw)
            taps: dict[str, np.ndarray] = {}

            def record(name: str, value: torch.Tensor) -> None:
                if name in {"inp_embd", "g_norm-0"}:
                    taps[name] = value.cpu().numpy().astype(np.float32, copy=True).reshape(-1)

            step = adapter.decode_step(
                token, fused, position, cache, compute_logits=False, trace_callback=record
            )
            cache = step.cache
            expected_key = (
                _python_qk_to_native_rows(cache.key[:, position, :].numpy().reshape(-1), 8)
                .astype(np.float16)
                .view(np.uint16)
                .reshape(8, 128)
            )
            expected_value = cache.value[:, position, :].numpy().astype(np.float16).view(np.uint16)
            candidates = []
            for execution, group in groups.items():
                required = {"inp_embd", "g_norm-0", "Kcur_rope-0", "Vcur-0"}
                if not required <= set(group):
                    continue
                for column in range(group["inp_embd"]["n_tokens"]):
                    embedding = _column(group["inp_embd"], values, column)
                    if not np.array_equal(embedding, taps["inp_embd"]):
                        continue
                    normalized = _column(group["g_norm-0"], values, column)
                    norm_rms = float(
                        np.sqrt(np.mean((normalized.astype(np.float64) - taps["g_norm-0"]) ** 2))
                    )
                    if norm_rms >= 1e-5:
                        continue
                    key = _column(group["Kcur_rope-0"], values, column).reshape(8, 128)
                    value = _column(group["Vcur-0"], values, column).reshape(8, 128)
                    native_key = key.astype(np.float16).view(np.uint16)
                    native_value = value.astype(np.float16).view(np.uint16)
                    candidates.append(
                        {
                            "group_execution": execution,
                            "token_column": column,
                            "fused_norm_rms_difference": norm_rms,
                            "key_equal_elements": int(np.count_nonzero(expected_key == native_key)),
                            "value_equal_elements": int(
                                np.count_nonzero(expected_value == native_value)
                            ),
                            "key_bits": native_key.copy(),
                            "value_bits": native_value.copy(),
                        }
                    )
            if not candidates:
                raise ValueError(f"context position {position} lacks a native graph input join")
            candidates.sort(key=lambda item: (item["group_execution"], item["token_column"]))
            chosen = candidates[-1]
            agree = all(
                np.array_equal(item["key_bits"], chosen["key_bits"])
                and np.array_equal(item["value_bits"], chosen["value_bits"])
                for item in candidates
            )
            comparisons.append(
                {
                    "decoder_position": position,
                    "input_token_id": token,
                    "native_candidate_rows": len(candidates),
                    "candidate_kv_values_agree": agree,
                    "latest_native_group_execution": chosen["group_execution"],
                    "latest_native_token_column": chosen["token_column"],
                    "max_fused_norm_rms_difference": max(
                        item["fused_norm_rms_difference"] for item in candidates
                    ),
                    "key_equal_elements": min(item["key_equal_elements"] for item in candidates),
                    "value_equal_elements": min(
                        item["value_equal_elements"] for item in candidates
                    ),
                }
            )
    elements = len(comparisons) * 8 * 128
    key_matches = sum(row["key_equal_elements"] for row in comparisons)
    value_matches = sum(row["value_equal_elements"] for row in comparisons)
    return {
        "schema": "recurrent_cpu_projected_kv_write_comparison_v1",
        "status": "projected_kv_write_values_compared_native_cache_unread",
        "execution_device": "cpu",
        "round_index": round_index,
        "prefix_tokens": len(prefix),
        "context_positions": len(comparisons),
        "key_elements": elements,
        "key_equal_elements": key_matches,
        "value_elements": elements,
        "value_equal_elements": value_matches,
        "all_native_duplicate_candidates_agree": all(
            row["candidate_kv_values_agree"] for row in comparisons
        ),
        "positions": comparisons,
        "source_sha256": {
            "seed_graph_join": sha256(seed_graph_join),
            "graph_index": sha256(index_path),
            "graph_values": sha256(values_path),
            "rounds": sha256(capture_dir / "forced-rounds.jsonl"),
            "feature_events": sha256(capture_dir / "heads.target_features.jsonl"),
            "feature_values": sha256(capture_dir / "heads.target_features.f32"),
            "target_gguf": sha256(target_gguf),
            "draft_gguf": sha256(draft_gguf),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--seed-graph-join", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--draft-gguf", type=Path, required=True)
    parser.add_argument("--drafter-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = compare_kv_writes(
        args.capture_dir,
        args.seed_graph_join,
        args.target_gguf,
        args.draft_gguf,
        args.drafter_config,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "round_index": report["round_index"],
                "context_positions": report["context_positions"],
                "key_equal": report["key_equal_elements"],
                "value_equal": report["value_equal_elements"],
            }
        )
    )


if __name__ == "__main__":
    main()
