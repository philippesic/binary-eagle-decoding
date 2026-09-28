#!/usr/bin/env python3
"""Compare one ordered CPU draft seed with adapter versus exact native K/V cache."""

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
from check_recurrent_cache_projection_boundary import _student_rope_key  # noqa: E402
from check_recurrent_ffn_from_graph import _array_sha256, _cpu_model, _metrics  # noqa: E402
from check_recurrent_ffn_stage_parity import (  # noqa: E402
    read_first_seed,
    verify_baseline_identity,
    verify_capture,
)
from check_recurrent_native_attention_operator import (  # noqa: E402
    _group_decoder_taps,
    reconstruct_slots,
)
from check_recurrent_real_step import _build_drafter, _retained_feature_indices  # noqa: E402
from check_recurrent_silu_arithmetic import ggml_silu  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import D_SHA256  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_attention_oracle import NativeCPUAttentionOracle  # noqa: E402
from w1a1_eagle.native_step import NativeStepCache, _frozen_rms_norm  # noqa: E402


def _cache(keys: np.ndarray, values: np.ndarray) -> NativeStepCache:
    if (
        keys.shape != (46, 1024)
        or values.shape != keys.shape
        or keys.dtype != np.dtype("<u2")
        or values.dtype != keys.dtype
    ):
        raise ValueError("first-seed cache requires 46 F16 K/V rows")
    native_k = keys.view("<f2").astype("<f4").reshape(46, 8, 128)
    python_k = native_qk_to_python_rows(native_k, 8).transpose(1, 0, 2).copy()
    python_v = values.view("<f2").astype("<f4").reshape(46, 8, 128).transpose(1, 0, 2)
    return NativeStepCache(torch.from_numpy(python_k), torch.from_numpy(python_v.copy()))


def audit(
    capture: Path,
    target: Path,
    draft: Path,
    config: Path,
    rope_report: Path,
    boundary_report: Path,
    attention_helper: Path,
    stage_capture: Path,
    stage_report: Path,
    silu_report: Path,
    silu_library: Path,
) -> dict:
    manifest, seal = verify_capture(capture, require_stages=False)
    verify_baseline_identity(seal)
    rope = json.loads(rope_report.read_text())
    boundary = json.loads(boundary_report.read_text())
    if (
        rope.get("schema") != "recurrent_reasoning_ggml_rope_oracle_cpu_v1"
        or rope["source_sha256"]["capture_manifest"] != seal["manifest"]
        or rope["source_sha256"]["cache_boundary_report"] != sha256(boundary_report)
        or rope["context_f32"]["exact_elements"] != 46 * 1024
        or boundary["source_sha256"]["candidate_d"] != sha256(draft)
        or sha256(draft) != D_SHA256
        or manifest.get("native_revision") != rope["native_revision"]
    ):
        raise ValueError("capture, candidate D or RoPE report identity differs")
    _, stage_seal = verify_capture(stage_capture, require_stages=True)
    stage_prior = json.loads(stage_report.read_text())
    silu_prior = json.loads(silu_report.read_text())
    if (
        stage_prior["source_sha256"]["new_capture_manifest"] != stage_seal["manifest"]
        or silu_prior["source_sha256"]["stage_report"] != sha256(stage_report)
        or silu_prior["source_sha256"]["ggml_cpu_library"] != sha256(silu_library)
        or silu_prior["source_sha256"]["capture_manifest"] != stage_seal["manifest"]
    ):
        raise ValueError("native FFN stage or ggml SiLU library identity differs")
    cache_audit = audit_cache(capture)
    oracle = NativeCPUAttentionOracle(attention_helper, threads=10)
    events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in events if row.get("event") == "execution"]
    rows = [row for row in events if row.get("event") == "row"]
    native_keys, native_values, _, ledger = reconstruct_slots(
        executions,
        rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        2,
    )
    if ledger["selected_rows"][0]["position"] != 46:
        raise ValueError("selected seed is not at decoder position 46")
    records, values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    groups = _group_decoder_taps(records)
    context, seed = groups[1], groups[2]
    if context["Kcur-0"]["n_tokens"] != 46 or seed["Qcur_rope-0"]["n_tokens"] != 1:
        raise ValueError("context or seed graph has unexpected geometry")
    operands = FrozenOperands(target, draft)
    adapter = _build_drafter(
        config,
        draft,
        operands,
        "native_order",
        attention_mode="native_forward_f32_backward",
        native_attention_oracle=oracle,
    )
    adapter_keys = np.stack(
        [
            _student_rope_key(
                _column(context["Kcur-0"], values, position),
                position,
                adapter.rope_theta,
            )
            .astype("<f2")
            .view("<u2")
            for position in range(46)
        ]
    )
    exact_adapter_keys = int(np.count_nonzero(adapter_keys == native_keys[:46]))
    if exact_adapter_keys != 46 * 1024 - 3:
        raise ValueError("adapter RoPE cache does not retain the three known key thresholds")
    native_cache = _cache(native_keys[:46].copy(), native_values[:46].copy())
    adapter_cache = _cache(adapter_keys, native_values[:46].copy())

    round_rows = [
        row for row in read_jsonl(capture / "forced-rounds.jsonl") if row.get("round_index") == 0
    ]
    if len(round_rows) != 1 or len(round_rows[0]["prefix_token_ids"]) != 47:
        raise ValueError("first reasoning round lacks the expected prefix")
    round_row = round_rows[0]
    indices = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"),
        round_row["prefix_token_ids"],
        round_row["task_id"],
    )
    feature_values = np.memmap(capture / "heads.target_features.f32", dtype="<f4", mode="r")
    feature_values = feature_values.reshape(-1, 7680)
    native_first = read_first_seed(capture, draft, require_stages=False)
    stage_first = read_first_seed(stage_capture, draft, require_stages=True)
    for name in ("input", "output", "head_state"):
        if native_first[name].tobytes() != stage_first[name].tobytes():
            raise ValueError(f"new FFN stage capture changed native first-seed {name}")
    expected_stage_hashes = stage_prior["results"]["native_order"]["native_stage_sha256"]
    if any(
        _array_sha256(stage_first["stages"][name]) != expected_stage_hashes[name]
        for name in expected_stage_hashes
    ):
        raise ValueError("new FFN stage values differ from sealed report")
    comparisons = {}
    with torch.no_grad():
        raw = torch.from_numpy(np.array(feature_values[indices[46]], copy=True))
        feature = adapter.encode_feature(raw)
        for mode, start_cache in (
            ("adapter_rope_cache", adapter_cache),
            ("native_cache", native_cache),
        ):
            taps: dict[str, np.ndarray] = {}

            def record_tap(name: str, tensor: torch.Tensor) -> None:
                taps[name] = tensor.numpy().astype("<f4", copy=True).reshape(-1)

            step = adapter.decode_step(
                round_row["seed_token_id"],
                feature,
                46,
                start_cache,
                compute_logits=False,
                trace_callback=record_tap,
            )
            stage_names = (
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
            )
            metrics = {}
            for name in stage_names:
                actual = taps[name]
                if name.startswith("Qcur"):
                    actual = python_qk_to_native_rows(actual.reshape(32, 128), 32).reshape(-1)
                elif name.startswith("Kcur"):
                    actual = python_qk_to_native_rows(actual.reshape(8, 128), 8).reshape(-1)
                native_stage = _column(seed[name], values, 0)
                metrics[name] = _metrics(actual, native_stage)
            metrics["result_norm"] = _metrics(taps["result_norm"], native_first["head_state"])
            metrics["pre_norm"] = _metrics(
                step.pre_norm.numpy(), _column(seed["eagle3_prenorm-0"], values, 0)
            )
            after_k = (
                python_qk_to_native_rows(step.cache.key[:, 46, :].numpy(), 8)
                .astype("<f2")
                .view("<u2")
                .reshape(-1)
            )
            after_v = step.cache.value[:, 46, :].numpy().astype("<f2").view("<u2").reshape(-1)
            comparisons[mode] = {
                "versus_native": metrics,
                "seed_key_f16_exact": int(np.count_nonzero(after_k == native_keys[46])),
                "seed_value_f16_exact": int(np.count_nonzero(after_v == native_values[46])),
            }
            if mode == "native_cache":
                mlp = adapter.drafter.midlayer.mlp
                post = torch.from_numpy(taps["post_attn_norm-0"].copy())
                gate = mlp.gate_proj(post)
                up = mlp.up_proj(post)
                silu_gate = torch.from_numpy(ggml_silu(silu_library, gate.numpy(), scalar=False))
                product = up * silu_gate
                down = mlp.down_proj(product)
                prenorm = torch.from_numpy(taps["ffn_inp-0"].copy()) + down
                normalized = _frozen_rms_norm(prenorm, adapter.drafter.norm)
                stage_tensors = {
                    "gate": gate.numpy(),
                    "up": up.numpy(),
                    "silu_gate": silu_gate.numpy(),
                    "mul": product.numpy(),
                    "down": down.numpy(),
                }
                comparisons[mode]["native_silu_intervention"] = {
                    "ffn_stages": {
                        name: _metrics(tensor, stage_first["stages"][name])
                        for name, tensor in stage_tensors.items()
                    },
                    "pre_norm": _metrics(
                        prenorm.numpy(), _column(seed["eagle3_prenorm-0"], values, 0)
                    ),
                    "result_norm": _metrics(normalized.numpy(), native_first["head_state"]),
                }
                head_rows = [
                    row
                    for row in read_jsonl(capture / "heads.jsonl")
                    if row.get("round_index") == 0 and row.get("depth") == 0
                ]
                if len(head_rows) != 1:
                    raise ValueError("native first seed lacks one head-logit record")
                native_head = head_rows[0]
                mapping = np.asarray(Model(draft).tensors["d2t"].data, dtype=np.int64)
                logits = adapter.drafter.lm_head(normalized).numpy().copy()
                if mapping.shape != logits.shape or len(set(mapping.tolist())) != mapping.size:
                    raise ValueError("candidate D head map is not one-to-one")
                predicted_draft = int(np.argmax(logits))
                predicted_target = int(mapping[predicted_draft])
                label_positions = np.flatnonzero(mapping == native_head["verifier_token_id"])
                if label_positions.size != 1:
                    raise ValueError("native verifier label is not in draft vocabulary")
                label_index = int(label_positions[0])
                probes = native_head["draft_logit_probe"]
                if [row["token_id"] for row in probes] != list(range(8)):
                    raise ValueError("native draft-logit probe is not the first eight IDs")
                native_probe = np.asarray([row["logit"] for row in probes], dtype=np.float32)
                predicted_probe = logits[:8]
                comparisons[mode]["native_silu_intervention"]["head"] = {
                    "probe": _metrics(predicted_probe, native_probe),
                    "mapped_argmax_id": predicted_target,
                    "native_argmax_id": native_head["draft_argmax_id"],
                    "argmax_logit": _metrics(
                        np.asarray([logits[predicted_draft]], dtype=np.float32),
                        np.asarray([native_head["draft_argmax_logit"]], dtype=np.float32),
                    ),
                    "label_logit": _metrics(
                        np.asarray([logits[label_index]], dtype=np.float32),
                        np.asarray([native_head["verifier_label_logit"]], dtype=np.float32),
                    ),
                    "target_rank": 1 + int(np.count_nonzero(logits > logits[label_index])),
                    "native_target_rank": native_head["target_rank"],
                }
    return {
        "schema": "recurrent_reasoning_seed_corrected_cache_cpu_v1",
        "scope": "one frozen first-round reasoning seed; no optimizer or target forward",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "context_adapter_key_f16_exact": exact_adapter_keys,
        "context_key_elements": 46 * 1024,
        "cache_audit_status": cache_audit["status"],
        "comparisons": comparisons,
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "candidate_d": D_SHA256,
            "target_gguf": sha256(target),
            "drafter_config": sha256(config),
            "rope_report": sha256(rope_report),
            "cache_boundary_report": sha256(boundary_report),
            "attention_helper": sha256(attention_helper),
            "stage_capture_manifest": stage_seal["manifest"],
            "stage_report": sha256(stage_report),
            "silu_report": sha256(silu_report),
            "silu_cpu_library": sha256(silu_library),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--drafter-config", type=Path, required=True)
    parser.add_argument("--rope-report", type=Path, required=True)
    parser.add_argument("--cache-boundary-report", type=Path, required=True)
    parser.add_argument("--attention-helper", type=Path, required=True)
    parser.add_argument("--stage-capture", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--silu-report", type=Path, required=True)
    parser.add_argument("--silu-cpu-library", type=Path, required=True)
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
        args.rope_report,
        args.cache_boundary_report,
        args.attention_helper,
        args.stage_capture,
        args.stage_report,
        args.silu_report,
        args.silu_cpu_library,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                mode: {
                    stage: comparison["versus_native"][stage]["max_abs"]
                    for stage in ("kqv_out-0", "ffn_inp-0", "ffn_out-0", "result_norm")
                }
                for mode, comparison in result["comparisons"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
