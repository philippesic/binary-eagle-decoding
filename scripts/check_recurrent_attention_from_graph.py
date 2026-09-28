#!/usr/bin/env python3
"""Replay one native EAGLE attention row from captured CPU Q/K/V tensors.

This isolates native attention-kernel arithmetic after verifying that the
captured context decoder inputs are the exact frozen target embeddings for
the first prompt. It is a numeric diagnostic, not a parity or speed gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402


def _matrix(group: dict[str, dict], name: str, values: np.ndarray) -> np.ndarray:
    record = group[name]
    return np.stack([_column(record, values, index) for index in range(record["n_tokens"])])


def replay(
    capture_dir: Path,
    target_gguf: Path,
    draft_gguf: Path,
    *,
    context_execution: int = 1,
    seed_execution: int = 2,
    seed_column: int = 0,
) -> dict:
    capture_dir = Path(capture_dir)
    index_path = capture_dir / "heads.draft_graph.jsonl"
    values_path = capture_dir / "heads.draft_graph.f32"
    records, values, footer = _read_graph(index_path, values_path)
    groups: dict[int, dict[str, dict]] = {}
    for row in records:
        if row.get("group_kind") == "decoder":
            groups.setdefault(row["group_execution"], {})[row["tensor_name"]] = row
    context, seed = groups[context_execution], groups[seed_execution]
    prefix = read_jsonl(capture_dir / "forced-rounds.jsonl")[0]["prefix_token_ids"]
    if len(prefix) < 2 or context["inp_embd"]["n_tokens"] != len(prefix) - 1:
        raise ValueError("native context graph does not cover first prompt prefix")
    operands = FrozenOperands(target_gguf, draft_gguf)
    context_embeddings = _matrix(context, "inp_embd", values)
    if any(
        not np.array_equal(row, operands(int(token)).float().numpy())
        for row, token in zip(context_embeddings, prefix[1:])
    ):
        raise ValueError("native decoder context tokens differ from frozen prompt embeddings")
    if seed["inp_embd"]["n_tokens"] <= seed_column:
        raise ValueError("selected seed graph token column is missing")
    q = _column(seed["Qcur_rope-0"], values, seed_column).reshape(32, 128).copy()
    keys = np.concatenate(
        (
            _matrix(context, "Kcur_rope-0", values).reshape(-1, 8, 128),
            _column(seed["Kcur_rope-0"], values, seed_column).reshape(1, 8, 128),
        )
    )
    values_kv = np.concatenate(
        (
            _matrix(context, "Vcur-0", values).reshape(-1, 8, 128),
            _column(seed["Vcur-0"], values, seed_column).reshape(1, 8, 128),
        )
    )
    native = _column(seed["kqv_out-0"], values, seed_column).astype(np.float64)
    with torch.no_grad():
        query = torch.from_numpy(q)
        key_cache = (
            torch.from_numpy(keys.copy()).to(torch.float16).to(torch.float32).permute(1, 0, 2)
        )
        value_cache = (
            torch.from_numpy(values_kv.copy()).to(torch.float16).to(torch.float32).permute(1, 0, 2)
        )
        repeated_key = key_cache.repeat_interleave(4, dim=0)
        repeated_value = value_cache.repeat_interleave(4, dim=0)
        scores = torch.einsum("hd,htd->ht", query, repeated_key) / math.sqrt(128)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        replayed = torch.einsum("ht,htd->hd", probabilities, repeated_value).reshape(-1)
    delta = replayed.numpy().astype(np.float64) - native
    server_log = (capture_dir / "server.log").read_text(errors="replace")
    return {
        "schema": "recurrent_native_qkv_cpu_attention_replay_v1",
        "status": "native_qkv_replayed_attention_drift_measured",
        "execution_device": "cpu",
        "context_group_execution": context_execution,
        "seed_group_execution": seed_execution,
        "seed_token_column": seed_column,
        "context_embedding_rows_exact": len(prefix) - 1,
        "cache_positions": keys.shape[0],
        "native_flash_attention_enabled_log": "Flash Attention enabled" in server_log,
        "max_abs_attention_difference": float(np.max(np.abs(delta))),
        "rms_attention_difference": float(np.sqrt(np.mean(delta * delta))),
        "native_attention_rms": float(np.sqrt(np.mean(native * native))),
        "source_sha256": {
            "target_gguf": sha256(target_gguf),
            "draft_gguf": sha256(draft_gguf),
            "rounds": sha256(capture_dir / "forced-rounds.jsonl"),
            "graph_index": sha256(index_path),
            "graph_values": sha256(values_path),
            "server_log": sha256(capture_dir / "server.log"),
        },
        "capture_footer": {
            "status": footer["status"],
            "decoder_groups": footer["decoder_groups"],
            "bytes_written": footer["bytes_written"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--draft-gguf", type=Path, required=True)
    parser.add_argument("--context-execution", type=int, default=1)
    parser.add_argument("--seed-execution", type=int, default=2)
    parser.add_argument("--seed-column", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = replay(
        args.capture_dir,
        args.target_gguf,
        args.draft_gguf,
        context_execution=args.context_execution,
        seed_execution=args.seed_execution,
        seed_column=args.seed_column,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"max_abs": result["max_abs_attention_difference"], "status": result["status"]})
    )


if __name__ == "__main__":
    main()
