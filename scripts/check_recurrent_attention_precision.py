#!/usr/bin/env python3
"""Ablate CPU Flash Attention arithmetic on one archived native graph row.

The F16 dot and online paths model the current AArch64 NEON F16 code. Python
and NumPy cannot promise native instruction, expf, or compiler parity. This is
a model-free numeric diagnostic, not an end-to-end or performance gate.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from audit_recurrent_binary_capture import sha256
from compare_recurrent_draft_graph import _column, _read_graph


def _f32_softmax_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Match the earlier torch replay's F32 score, softmax, and value path."""
    with torch.no_grad():
        query = torch.from_numpy(q.copy())
        keys = torch.from_numpy(k.copy()).repeat_interleave(4, dim=0)
        values = torch.from_numpy(v.copy()).repeat_interleave(4, dim=0)
        scores = torch.einsum("hd,htd->ht", query, keys) / math.sqrt(q.shape[-1])
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.einsum("ht,htd->hd", probs, values).numpy().copy()


def _neon_f16_dot(q: np.ndarray, k: np.ndarray) -> np.float32:
    """Model GGML_F16_STEP=32, ARR=4, EPR=8 with F16 lane FMAs."""
    if q.shape != k.shape or q.ndim != 1 or q.size % 32:
        raise ValueError("NEON F16 dot needs equal 1D vectors divisible by 32")
    lanes = np.zeros((4, 8), dtype=np.float16)
    for base in range(0, q.size, 32):
        for group in range(4):
            sl = slice(base + group * 8, base + (group + 1) * 8)
            # Products of F16 values are exactly representable in F32 here;
            # one F16 rounding follows each modeled fused lane update.
            lanes[group] = (
                lanes[group].astype(np.float32)
                + q[sl].astype(np.float32) * k[sl].astype(np.float32)
            ).astype(np.float16)
    lanes[:2] = (lanes[:2].astype(np.float32) + lanes[2:].astype(np.float32)).astype(np.float16)
    reduced = (lanes[0].astype(np.float32) + lanes[1].astype(np.float32)).astype(np.float16)
    return np.float32(
        np.sum(reduced[:4].astype(np.float32) + reduced[4:].astype(np.float32), dtype=np.float32)
    )


def _f16_dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    heads, width = q.shape
    positions = k.shape[1]
    scores = np.empty((heads, positions), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(width))
    for head in range(heads):
        for pos in range(positions):
            scores[head, pos] = np.float32(_neon_f16_dot(q[head], k[head // 4, pos]) * scale)
    return scores


def _online_attention(scores: np.ndarray, values: np.ndarray, *, f16_values: bool) -> np.ndarray:
    """Model ggml's ordered online max/sum and optional F16 value numerator."""
    heads, positions = scores.shape
    output = np.empty((heads, values.shape[-1]), dtype=np.float32)
    for head in range(heads):
        numerator = np.zeros(values.shape[-1], dtype=np.float16 if f16_values else np.float32)
        maximum = -math.inf
        denominator = np.float32(0.0)
        for pos in range(positions):
            score = float(scores[head, pos])
            value = values[head // 4, pos]
            ms = np.float32(1.0)
            vs = np.float32(1.0)
            if score > maximum:
                ms = np.float32(math.exp(maximum - score))
                maximum = score
                if f16_values:
                    numerator = (numerator.astype(np.float32) * np.float16(ms)).astype(np.float16)
                else:
                    numerator = np.float32(numerator * ms)
            else:
                vs = np.float32(math.exp(score - maximum))
            if f16_values:
                numerator = (
                    numerator.astype(np.float32) + value.astype(np.float32) * np.float16(vs)
                ).astype(np.float16)
            else:
                numerator = np.float32(numerator + value.astype(np.float32) * vs)
            denominator = np.float32(denominator * ms + vs)
        output[head] = np.float32(numerator.astype(np.float32) / denominator)
    return output


def _metrics(output: np.ndarray, native: np.ndarray) -> dict[str, float]:
    difference = output.astype(np.float64).ravel() - native.astype(np.float64).ravel()
    return {
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
    }


def replay(
    capture_dir: Path, *, context_execution: int = 1, seed_execution: int = 2, seed_column: int = 0
) -> dict:
    capture_dir = Path(capture_dir)
    index_path = capture_dir / "heads.draft_graph.jsonl"
    values_path = capture_dir / "heads.draft_graph.f32"
    records, payload, footer = _read_graph(index_path, values_path)
    groups: dict[int, dict[str, dict]] = {}
    for record in records:
        if record.get("group_kind") == "decoder":
            groups.setdefault(record["group_execution"], {})[record["tensor_name"]] = record
    context, seed = groups[context_execution], groups[seed_execution]
    if seed["inp_embd"]["n_tokens"] <= seed_column:
        raise ValueError("selected seed graph token column is missing")
    q = _column(seed["Qcur_rope-0"], payload, seed_column).reshape(32, 128).copy()
    k = np.concatenate(
        [
            np.stack(
                [
                    _column(context["Kcur_rope-0"], payload, i)
                    for i in range(context["Kcur_rope-0"]["n_tokens"])
                ]
            ).reshape(-1, 8, 128),
            _column(seed["Kcur_rope-0"], payload, seed_column).reshape(1, 8, 128),
        ]
    )
    v = np.concatenate(
        [
            np.stack(
                [
                    _column(context["Vcur-0"], payload, i)
                    for i in range(context["Vcur-0"]["n_tokens"])
                ]
            ).reshape(-1, 8, 128),
            _column(seed["Vcur-0"], payload, seed_column).reshape(1, 8, 128),
        ]
    )
    native = _column(seed["kqv_out-0"], payload, seed_column).reshape(32, 128)
    k = k.astype(np.float16).astype(np.float32).transpose(1, 0, 2).copy()
    v = v.astype(np.float16).transpose(1, 0, 2).copy()
    q16 = q.astype(np.float16).astype(np.float32)
    scores = _f16_dot_scores(q16, k.astype(np.float16))
    stages = {
        "f32_softmax": _f32_softmax_attention(q, k, v.astype(np.float32)),
        "q_f16_softmax": _f32_softmax_attention(q16, k, v.astype(np.float32)),
    }
    repeated_v = torch.from_numpy(v.astype(np.float32)).repeat_interleave(4, dim=0)
    with torch.no_grad():
        probabilities = torch.softmax(torch.from_numpy(scores), dim=-1, dtype=torch.float32)
        stages["q_f16_dot_f16_softmax"] = (
            torch.einsum("ht,htd->hd", probabilities, repeated_v).numpy().copy()
        )
    stages["q_f16_dot_f16_online_f32"] = _online_attention(scores, v, f16_values=False)
    stages["q_f16_dot_f16_online_f16"] = _online_attention(scores, v, f16_values=True)
    report_sources = {
        "graph_index": sha256(index_path),
        "graph_values": sha256(values_path),
        "rounds": sha256(capture_dir / "forced-rounds.jsonl"),
        "server_log": sha256(capture_dir / "server.log"),
    }
    cpu_source = Path(__file__).resolve().parents[1] / "third_party/llama.cpp/ggml/src/ggml-cpu"
    for filename in ("ops.cpp", "vec.cpp", "vec.h", "simd-mappings.h"):
        report_sources[f"ggml_cpu_{filename}"] = sha256(cpu_source / filename)
    reference_path = capture_dir / "native_qkv_torch_attention.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        for name, digest in report_sources.items():
            if name in reference["source_sha256"] and reference["source_sha256"][name] != digest:
                raise ValueError(f"prior replay source hash differs: {name}")
        if not math.isclose(
            _metrics(stages["f32_softmax"], native)["max_abs_difference"],
            reference["max_abs_attention_difference"],
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError("F32 baseline differs from the prior graph replay")
        report_sources.update(
            {
                "target_gguf_from_prior_report": reference["source_sha256"]["target_gguf"],
                "draft_gguf_from_prior_report": reference["source_sha256"]["draft_gguf"],
                "prior_replay_report": sha256(reference_path),
            }
        )
    return {
        "schema": "recurrent_cpu_attention_precision_ablation_v1",
        "execution_device": "cpu",
        "execution_architecture": platform.machine(),
        "scope": "single archived native graph row; approximate NEON F16 arithmetic",
        "context_group_execution": context_execution,
        "seed_group_execution": seed_execution,
        "seed_token_column": seed_column,
        "cache_positions": int(k.shape[1]),
        "native_flash_attention_enabled_log": "Flash Attention enabled"
        in (capture_dir / "server.log").read_text(errors="replace"),
        "native_attention_rms": float(np.sqrt(np.mean(native.astype(np.float64) ** 2))),
        "stages": {name: _metrics(output, native) for name, output in stages.items()},
        "source_sha256": report_sources,
        "capture_footer": {
            key: footer[key] for key in ("status", "decoder_groups", "bytes_written")
        },
        "limitations": [
            "F16 dot uses NumPy rounding rather than native NEON instructions and reduction.",
            "online exponential uses Python math.exp rounded to F32 rather than C expf.",
            "graph Q/K/V and F16-rounded projected cache values are not actual stored cache bytes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--context-execution", type=int, default=1)
    parser.add_argument("--seed-execution", type=int, default=2)
    parser.add_argument("--seed-column", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = replay(
        args.capture_dir,
        context_execution=args.context_execution,
        seed_execution=args.seed_execution,
        seed_column=args.seed_column,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stages": result["stages"], "execution_device": "cpu"}))


if __name__ == "__main__":
    main()
