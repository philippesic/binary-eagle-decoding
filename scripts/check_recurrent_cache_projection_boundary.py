#!/usr/bin/env python3
"""Locate selected reasoning cache K/V differences at the fused-input boundary."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from audit_recurrent_draft_cache import audit as audit_cache  # noqa: E402
from check_recurrent_attention_operand_ablation import (  # noqa: E402
    native_qk_to_python_rows,
    python_qk_to_native_rows,
)
from check_recurrent_ffn_from_graph import _cpu_model, _metrics  # noqa: E402
from check_recurrent_ffn_stage_parity import verify_baseline_identity, verify_capture  # noqa: E402
from check_recurrent_native_attention_operator import (  # noqa: E402
    _group_decoder_taps,
    reconstruct_slots,
)
from check_recurrent_real_step import _build_drafter, _retained_feature_indices  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from load_recurrent_binary_init import D_SHA256, load_candidate_d_arrays  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_step import _frozen_rms_norm  # noqa: E402
from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402

POSITIONS = (0, 26, 45)


def _f16_exact(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("F16 comparison requires equal finite arrays")
    return int(
        np.count_nonzero(actual.astype("<f2").view("<u2") == expected.astype("<f2").view("<u2"))
    )


def _stage(actual: np.ndarray, expected: np.ndarray) -> dict:
    return {"f32": _metrics(actual, expected), "f16_exact": _f16_exact(actual, expected)}


def _student_rope_key(native_raw: np.ndarray, position: int, theta: float) -> np.ndarray:
    """Replay the adapter's half-split F32 key RoPE in native row order."""
    key = native_qk_to_python_rows(native_raw.reshape(8, 128), 8).copy()
    k = torch.from_numpy(key)
    theta_scale = torch.tensor(theta, dtype=torch.float32).pow(-2.0 / 128)
    angle = torch.empty(64, dtype=torch.float32)
    current = torch.tensor(float(position), dtype=torch.float32)
    for index in range(64):
        angle[index] = current
        current = current * theta_scale
    full_angle = torch.cat((angle, angle))
    cos, sin = full_angle.cos(), full_angle.sin()
    rotated = k * cos + torch.cat((-k[:, 64:], k[:, :64]), dim=-1) * sin
    native = python_qk_to_native_rows(rotated.numpy(), 8)
    return native.reshape(1024).copy()


def audit(
    capture: Path,
    target: Path,
    draft: Path,
    config: Path,
    student_cache: Path,
    ablation_report: Path,
) -> dict:
    manifest, seal = verify_capture(capture, require_stages=False)
    verify_baseline_identity(seal)
    previous = json.loads(ablation_report.read_text())
    if (
        previous.get("schema") != "recurrent_draft_attention_operand_ablation_v1"
        or previous["source_sha256"]["capture_manifest"] != seal["manifest"]
        or previous["source_sha256"]["adapter_post_seed_cache"] != sha256(student_cache)
        or previous["source_sha256"]["draft_gguf"] != sha256(draft)
        or sha256(draft) != D_SHA256
        or previous["native_join"]["execution"] != 2
        or previous["native_join"]["column"] != 0
    ):
        raise ValueError("sealed capture, D or student cache identity differs")
    cache_audit = audit_cache(capture)
    cache_events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in cache_events if row.get("event") == "execution"]
    rows = [row for row in cache_events if row.get("event") == "row"]
    native_keys, native_values, _, ledger = reconstruct_slots(
        executions,
        rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        2,
    )
    records, graph_values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    groups = _group_decoder_taps(records)
    if groups[1]["concat_embd-0"]["n_tokens"] != 46 or ledger["selected_rows"][0]["position"] != 46:
        raise ValueError("reasoning context/seed ancestry differs")
    rounds = [
        row for row in read_jsonl(capture / "forced-rounds.jsonl") if row.get("round_index") == 0
    ]
    if len(rounds) != 1 or len(rounds[0]["prefix_token_ids"]) != 47:
        raise ValueError("reasoning round lacks the expected 47-token prefix")
    record = rounds[0]
    prefix = record["prefix_token_ids"]
    feature_rows = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"), prefix, record["task_id"]
    )
    raw_values = np.memmap(capture / "heads.target_features.f32", dtype="<f4", mode="r")
    raw_values = raw_values.reshape(-1, 7680)
    if max(feature_rows) >= len(raw_values):
        raise ValueError("retained feature row exceeds sealed payload")
    operands = FrozenOperands(target, draft)
    adapter = _build_drafter(config, draft, operands, "group_matmul")
    with np.load(student_cache, allow_pickle=False) as saved:
        if set(saved.files) != {"key", "value"}:
            raise ValueError("student cache has unexpected tensors")
        expected_k = saved["key"]
        expected_v = saved["value"]
    if expected_k.shape != (8, 47, 128) or expected_v.shape != expected_k.shape:
        raise ValueError("student cache has wrong geometry")
    cache = adapter.new_cache()
    selected: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for position in range(46):
            taps: dict[str, np.ndarray] = {}

            def record_tap(name: str, value: torch.Tensor) -> None:
                taps[name] = value.numpy().astype("<f4", copy=True).reshape(-1)

            callback = record_tap
            feature = adapter.encode_feature(
                torch.from_numpy(np.array(raw_values[feature_rows[position]], copy=True)),
                trace_callback=callback,
            )
            cache = adapter.decode_step(
                prefix[position + 1],
                feature,
                position,
                cache,
                compute_logits=False,
                trace_callback=callback,
            ).cache
            selected[position] = taps
    for name, actual, expected in (
        ("key", cache.key.numpy(), expected_k[:, :46]),
        ("value", cache.value.numpy(), expected_v[:, :46]),
    ):
        if not np.array_equal(actual.view("<u4"), expected.view("<u4")):
            raise ValueError(f"rebuilt context {name} differs from prior student cache")

    student_k = python_qk_to_native_rows(expected_k.transpose(1, 0, 2), 8)
    student_k = student_k.astype("<f2").view("<u2").reshape(47, 1024)
    student_v = expected_v.transpose(1, 0, 2).astype("<f2").view("<u2").reshape(47, 1024)
    position_counts = [
        {
            "position": position,
            "key_differences": int(np.count_nonzero(student_k[position] != native_keys[position])),
            "value_differences": int(
                np.count_nonzero(student_v[position] != native_values[position])
            ),
        }
        for position in range(47)
    ]
    if (
        sum(row["key_differences"] for row in position_counts) != 70
        or sum(row["value_differences"] for row in position_counts) != 66
        or position_counts[46]["key_differences"]
        or position_counts[46]["value_differences"]
    ):
        raise ValueError("cache discrepancy no longer matches prior reasoning ablation")

    arrays, d_audit = load_candidate_d_arrays(draft)
    if d_audit["gguf_sha256"] != D_SHA256:
        raise ValueError("candidate D arrays differ from pinned GGUF")
    linears = {
        name: GroupedBinaryLinear.from_packed(
            *arrays[base], in_features=5120, arithmetic="native_order"
        )
        for name, base in (("Kcur-0", "blk.0.attn_k"), ("Vcur-0", "blk.0.attn_v"))
    }
    fc_ordered = GroupedBinaryLinear.from_packed(
        *arrays["fc"], in_features=7680, arithmetic="native_order"
    )
    all_context = []
    grouped_rope_matches_student = 0
    ordered_rope_matches_native = 0
    ordered_value_matches_native = 0
    for position in range(46):
        native_group = groups[1]
        with torch.no_grad():
            raw = torch.from_numpy(np.array(raw_values[feature_rows[position]], copy=True))
            ordered_feature = fc_ordered(raw)
            ordered_norm = (
                _frozen_rms_norm(ordered_feature, adapter.drafter.midlayer.hidden_norm)
                .numpy()
                .copy()
            )
        native_norm = _column(native_group["g_norm-0"], graph_values, position)
        native_fused = _column(native_group["concat_embd-0"], graph_values, position)
        ordered_fused = np.concatenate((selected[position]["embd_norm-0"], ordered_norm))
        row = {
            "position": position,
            "token_id": prefix[position + 1],
            "grouped_fused_f16_exact": _f16_exact(
                selected[position]["concat_embd-0"], native_fused
            ),
            "ordered_fc_g_norm_f32_exact": _metrics(ordered_norm, native_norm)["exact_elements"],
            "ordered_fc_fused_f32_exact": _metrics(ordered_fused, native_fused)["exact_elements"],
        }
        for name, linear in linears.items():
            with torch.no_grad():
                projected = linear(torch.from_numpy(ordered_fused.copy())).numpy().copy()
            native_projection = _column(native_group[name], graph_values, position)
            row[f"ordered_fc_{name}_f32_exact"] = _metrics(projected, native_projection)[
                "exact_elements"
            ]
            if name == "Kcur-0":
                grouped = selected[position][name]
                grouped_native = python_qk_to_native_rows(grouped.reshape(8, 128), 8)
                grouped_rope = _student_rope_key(
                    grouped_native.reshape(-1), position, adapter.rope_theta
                )
                ordered_rope = _student_rope_key(projected, position, adapter.rope_theta)
                grouped_bits = grouped_rope.astype("<f2").view("<u2")
                ordered_bits = ordered_rope.astype("<f2").view("<u2")
                grouped_exact = int(np.count_nonzero(grouped_bits == student_k[position]))
                ordered_exact = int(np.count_nonzero(ordered_bits == native_keys[position]))
                native_rope = _column(native_group["Kcur_rope-0"], graph_values, position)
                row["grouped_rope_student_cache_exact"] = grouped_exact
                row["ordered_rope_native_cache_exact"] = ordered_exact
                row["ordered_rope_native_f32"] = _metrics(ordered_rope, native_rope)
                row["ordered_rope_f16_mismatches"] = [
                    {
                        "index": int(index),
                        "student_f32": float(ordered_rope[index]),
                        "native_f32": float(native_rope[index]),
                        "student_f16_bits": int(ordered_bits[index]),
                        "native_f16_bits": int(native_keys[position, index]),
                    }
                    for index in np.flatnonzero(ordered_bits != native_keys[position])
                ]
                grouped_rope_matches_student += grouped_exact
                ordered_rope_matches_native += ordered_exact
            else:
                ordered_bits = projected.astype("<f2").view("<u2")
                value_exact = int(np.count_nonzero(ordered_bits == native_values[position]))
                row["ordered_value_native_cache_exact"] = value_exact
                ordered_value_matches_native += value_exact
        all_context.append(row)
    if grouped_rope_matches_student != 46 * 1024:
        raise ValueError("standalone RoPE replay does not reproduce student key cache")
    selected_results = {}
    for position in POSITIONS:
        native_group = groups[1]
        native_fused = _column(native_group["concat_embd-0"], graph_values, position)
        student_fused = selected[position]["concat_embd-0"]
        native_half = native_fused.astype("<f2").view("<u2")
        student_half = student_fused.astype("<f2").view("<u2")
        different = np.flatnonzero(student_half != native_half)
        repaired_fused = student_fused.copy()
        repaired_fused[different] = native_fused[different]
        row = {
            "token_id": prefix[position + 1],
            "fused_input": _stage(student_fused, native_fused),
            "f16_input_thresholds": [
                {
                    "index": int(index),
                    "student_f32": float(student_fused[index]),
                    "native_f32": float(native_fused[index]),
                    "student_f16_bits": int(student_half[index]),
                    "native_f16_bits": int(native_half[index]),
                }
                for index in different
            ],
            "g_norm": _stage(
                selected[position]["g_norm-0"],
                _column(native_group["g_norm-0"], graph_values, position),
            ),
        }
        with torch.no_grad():
            raw = torch.from_numpy(np.array(raw_values[feature_rows[position]], copy=True))
            ordered_feature = fc_ordered(raw)
            ordered_norm = (
                _frozen_rms_norm(ordered_feature, adapter.drafter.midlayer.hidden_norm)
                .numpy()
                .copy()
            )
        native_g_norm = _column(native_group["g_norm-0"], graph_values, position)
        row["ordered_fc_then_norm"] = _stage(ordered_norm, native_g_norm)
        ordered_fused = np.concatenate(
            (_column(native_group["embd_norm-0"], graph_values, position), ordered_norm)
        )
        row["ordered_fc_fused"] = _stage(ordered_fused, native_fused)
        for name, linear in linears.items():
            native_projection = _column(native_group[name], graph_values, position)
            current = selected[position][name]
            if name == "Kcur-0":
                current = python_qk_to_native_rows(current.reshape(8, 128), 8).reshape(-1)
            with torch.no_grad():
                replay_student = linear(torch.from_numpy(student_fused.copy())).numpy().copy()
                replay_repaired = linear(torch.from_numpy(repaired_fused.copy())).numpy().copy()
                replay_native = linear(torch.from_numpy(native_fused.copy())).numpy().copy()
            row[name] = {
                "student_grouped": _stage(current, native_projection),
                "student_fused_ordered": _stage(replay_student, native_projection),
                "repaired_thresholds_ordered": _stage(replay_repaired, native_projection),
                "native_fused_ordered": _stage(replay_native, native_projection),
            }
            if row[name]["repaired_thresholds_ordered"]["f32"]["exact_elements"] != 1024:
                raise ValueError(f"{name} repaired threshold replay is not exact")
        selected_results[str(position)] = row
    return {
        "schema": "recurrent_reasoning_cache_projection_boundary_cpu_v1",
        "scope": "sealed first-round context positions 0,26,45; all 47 stored K/V positions",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "arithmetic": "group_matmul student control; ordered A16/W1 projection interventions",
        "native_revision": manifest["native_revision"],
        "cache_audit_status": cache_audit["status"],
        "position_counts": position_counts,
        "all_context_ordered_fc": all_context,
        "all_context_cache_interventions": {
            "key_elements": 46 * 1024,
            "grouped_rope_student_key_exact": grouped_rope_matches_student,
            "ordered_rope_native_key_exact": ordered_rope_matches_native,
            "ordered_value_native_exact": ordered_value_matches_native,
        },
        "selected_positions": selected_results,
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "drafter_config": sha256(config),
            "student_cache": sha256(student_cache),
            "ablation_report": sha256(ablation_report),
            "probe": sha256(Path(__file__)),
            "student_adapter": sha256(
                Path(__file__).resolve().parents[1] / "src/w1a1_eagle/native_step.py"
            ),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--drafter-config", type=Path, required=True)
    parser.add_argument("--student-cache", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    torch.set_num_threads(10)
    result = audit(
        args.capture_dir,
        args.target_gguf,
        args.candidate_d,
        args.drafter_config,
        args.student_cache,
        args.ablation_report,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                position: {
                    "f16_input_thresholds": len(row["f16_input_thresholds"]),
                    "key_exact_after_repair": row["Kcur-0"]["repaired_thresholds_ordered"]["f32"][
                        "exact_elements"
                    ],
                    "value_exact_after_repair": row["Vcur-0"]["repaired_thresholds_ordered"]["f32"][
                        "exact_elements"
                    ],
                }
                for position, row in result["selected_positions"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
