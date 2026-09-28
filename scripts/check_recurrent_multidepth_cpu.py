#!/usr/bin/env python3
"""Replay one sealed reasoning proposal chain with pinned CPU forward oracles."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
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
from check_recurrent_seed_corrected_cache import _cache  # noqa: E402
from check_recurrent_silu_arithmetic import ggml_silu  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import D_SHA256  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_attention_oracle import (  # noqa: E402
    NativeCPUAttentionOracle,
    native_forward_f32_backward,
)
from w1a1_eagle.native_step import _frozen_rms_norm  # noqa: E402

GRAPH_TAPS = (
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


def _ggml_query_rope(helper: Path, raw_native: np.ndarray, position: int) -> np.ndarray:
    """Use the pinned eight-head ggml RoPE helper as four 8-head Q blocks."""
    if raw_native.shape != (4096,) or raw_native.dtype != np.float32:
        raise ValueError("native raw Q must be one F32 [4096] vector")
    with tempfile.TemporaryDirectory(prefix="recurrent-query-rope-") as directory:
        operands = Path(directory)
        raw_native.astype("<f4").tofile(operands / "key_raw.f32")
        np.full(4, position, dtype="<i4").tofile(operands / "positions.i32")
        subprocess.run([str(helper.resolve()), str(operands), "4"], check=True)
        output = operands / "key_rope.f32"
        if output.stat().st_size != 4096 * 4:
            raise ValueError("ggml Q RoPE helper returned the wrong byte count")
        return np.fromfile(output, dtype="<f4")


def _head_comparison(logits: np.ndarray, mapping: np.ndarray, head: dict) -> dict:
    if logits.shape != (32000,) or mapping.shape != logits.shape:
        raise ValueError("candidate D head or vocabulary map has wrong geometry")
    label_positions = np.flatnonzero(mapping == head["verifier_token_id"])
    if label_positions.size != 1:
        raise ValueError("captured verifier label lacks one draft-vocabulary mapping")
    label = int(label_positions[0])
    predicted_draft = int(np.argmax(logits))
    predicted_target = int(mapping[predicted_draft])
    probes = head["draft_logit_probe"]
    if [row["token_id"] for row in probes] != list(range(8)):
        raise ValueError("native head probe does not contain draft rows 0..7")
    return {
        "probe": _metrics(
            logits[:8], np.asarray([row["logit"] for row in probes], dtype=np.float32)
        ),
        "argmax_logit": _metrics(
            np.asarray([logits[predicted_draft]], dtype=np.float32),
            np.asarray([head["draft_argmax_logit"]], dtype=np.float32),
        ),
        "label_logit": _metrics(
            np.asarray([logits[label]], dtype=np.float32),
            np.asarray([head["verifier_label_logit"]], dtype=np.float32),
        ),
        "mapped_argmax_id": predicted_target,
        "native_argmax_id": head["draft_argmax_id"],
        "target_rank": 1 + int(np.count_nonzero(logits > logits[label])),
        "native_target_rank": head["target_rank"],
    }


def audit(
    capture: Path,
    target: Path,
    draft: Path,
    config: Path,
    seed_report: Path,
    attention_helper: Path,
    silu_report: Path,
    silu_library: Path,
    rope_report: Path,
    rope_helper: Path,
) -> dict:
    manifest, seal = verify_capture(capture, require_stages=False)
    verify_baseline_identity(seal)
    prior = json.loads(seed_report.read_text())
    silu = json.loads(silu_report.read_text())
    rope = json.loads(rope_report.read_text())
    if (
        prior.get("schema") != "recurrent_reasoning_seed_corrected_cache_cpu_v1"
        or prior["source_sha256"]["capture_manifest"] != seal["manifest"]
        or prior["source_sha256"]["candidate_d"] != sha256(draft)
        or sha256(draft) != D_SHA256
        or prior["source_sha256"]["attention_helper"] != sha256(attention_helper)
        or prior["source_sha256"]["silu_cpu_library"] != sha256(silu_library)
        or silu["source_sha256"]["ggml_cpu_library"] != sha256(silu_library)
        or rope["source_sha256"]["capture_manifest"] != seal["manifest"]
        or rope["source_sha256"]["helper_binary"] != sha256(rope_helper)
        or prior["comparisons"]["native_cache"]["native_silu_intervention"]["result_norm"][
            "exact_elements"
        ]
        != 2560
    ):
        raise ValueError("sealed first seed, candidate D or pinned oracles differ")
    cache_audit = audit_cache(capture)
    events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in events if row.get("event") == "execution"]
    rows = [row for row in events if row.get("event") == "row"]
    native_k, native_v, _, ledger = reconstruct_slots(
        executions,
        rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        6,
    )
    if len(ledger["selected_rows"]) != 1 or ledger["selected_rows"][0]["position"] != 50:
        raise ValueError("fifth decoder execution is not at position 50")
    records, graph_values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    graph = _group_decoder_taps(records)
    if any(graph[2 + depth]["inp_embd"]["n_tokens"] != 1 for depth in range(5)):
        raise ValueError("reasoning proposal graph is not five single-token executions")
    heads = [row for row in read_jsonl(capture / "heads.jsonl") if row.get("round_index") == 0]
    if len(heads) != 5 or [row.get("depth") for row in heads] != list(range(5)):
        raise ValueError("first reasoning round lacks five ordered head records")
    native_states = np.memmap(capture / "heads.f32", dtype="<f4", mode="r").reshape(-1, 2560)
    rounds = [
        row for row in read_jsonl(capture / "forced-rounds.jsonl") if row.get("round_index") == 0
    ]
    if len(rounds) != 1 or len(rounds[0]["prefix_token_ids"]) != 47:
        raise ValueError("first reasoning round lacks the 47-token prefix")
    round_row = rounds[0]
    if [head["input_token_id"] for head in heads] != [
        round_row["seed_token_id"],
        *round_row["draft_token_ids"][:4],
    ]:
        raise ValueError("native head inputs do not follow the captured proposal chain")
    indices = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"),
        round_row["prefix_token_ids"],
        round_row["task_id"],
    )
    raw_values = np.memmap(capture / "heads.target_features.f32", dtype="<f4", mode="r")
    raw_values = raw_values.reshape(-1, 7680)
    operands = FrozenOperands(target, draft)
    oracle = NativeCPUAttentionOracle(attention_helper, threads=10)
    adapter = _build_drafter(
        config,
        draft,
        operands,
        "native_order",
        attention_mode="native_forward_f32_backward",
        native_attention_oracle=oracle,
    )
    mapping = np.asarray(Model(draft).tensors["d2t"].data, dtype=np.int64)
    if len(set(mapping.tolist())) != mapping.size:
        raise ValueError("draft vocabulary map is not one-to-one")
    cache = _cache(native_k[:46].copy(), native_v[:46].copy())
    results = []
    with torch.no_grad():
        raw = torch.from_numpy(np.array(raw_values[indices[46]], copy=True))
        feature = adapter.encode_feature(raw)
        for depth, head in enumerate(heads):
            position = 46 + depth
            if head.get("input_position") != position + 1 or head.get("state_row") is None:
                raise ValueError("head position or normalized-state row differs")
            taps: dict[str, np.ndarray] = {}

            def record_tap(name: str, tensor: torch.Tensor) -> None:
                taps[name] = tensor.numpy().astype("<f4", copy=True).reshape(-1)

            step = adapter.decode_step(
                head["input_token_id"],
                feature,
                position,
                cache,
                compute_logits=False,
                trace_callback=record_tap,
            )
            stages = {}
            for name in GRAPH_TAPS:
                actual = taps[name]
                if name.startswith("Qcur"):
                    actual = python_qk_to_native_rows(actual.reshape(32, 128), 32).reshape(-1)
                elif name.startswith("Kcur"):
                    actual = python_qk_to_native_rows(actual.reshape(8, 128), 8).reshape(-1)
                stages[name] = _metrics(actual, _column(graph[2 + depth][name], graph_values, 0))
            mlp = adapter.drafter.midlayer.mlp
            post = torch.from_numpy(taps["post_attn_norm-0"].copy())
            gate = mlp.gate_proj(post)
            up = mlp.up_proj(post)
            silu_gate = torch.from_numpy(ggml_silu(silu_library, gate.numpy(), scalar=False))
            product = up * silu_gate
            down = mlp.down_proj(product)
            corrected_prenorm = torch.from_numpy(taps["ffn_inp-0"].copy()) + down
            corrected_norm = _frozen_rms_norm(corrected_prenorm, adapter.drafter.norm)
            native_prenorm = _column(graph[2 + depth]["eagle3_prenorm-0"], graph_values, 0)
            state_row = head["state_row"]
            if type(state_row) is not int or not 0 <= state_row < len(native_states):
                raise ValueError("native normalized-state row is invalid")
            normalized_native = np.asarray(native_states[state_row])
            logits = adapter.drafter.lm_head(corrected_norm).numpy().copy()
            after_k = (
                python_qk_to_native_rows(step.cache.key[:, position, :].numpy(), 8)
                .astype("<f2")
                .view("<u2")
                .reshape(-1)
            )
            after_v = step.cache.value[:, position, :].numpy().astype("<f2").view("<u2").reshape(-1)
            results.append(
                {
                    "depth": depth,
                    "execution": 2 + depth,
                    "memory_position": position,
                    "input_target_token_id": head["input_token_id"],
                    "native_proposed_target_id": head["proposed_token_id"],
                    "adapter_stages": stages,
                    "corrected_ffn_output": _metrics(
                        down.numpy(), _column(graph[2 + depth]["ffn_out-0"], graph_values, 0)
                    ),
                    "corrected_pre_norm": _metrics(corrected_prenorm.numpy(), native_prenorm),
                    "corrected_head_state": _metrics(corrected_norm.numpy(), normalized_native),
                    "head": _head_comparison(logits, mapping, head),
                    "cache_key_f16_exact": int(np.count_nonzero(after_k == native_k[position])),
                    "cache_value_f16_exact": int(np.count_nonzero(after_v == native_v[position])),
                }
            )
            raw_q_native = python_qk_to_native_rows(taps["Qcur-0"].reshape(32, 128), 32).reshape(-1)
            ggml_q_native = _ggml_query_rope(rope_helper, raw_q_native, position)
            native_q = _column(graph[2 + depth]["Qcur_rope-0"], graph_values, 0)
            q_metrics = _metrics(ggml_q_native, native_q)
            q_python = native_qk_to_python_rows(ggml_q_native.reshape(32, 128), 32)
            oracle_attention = native_forward_f32_backward(
                torch.from_numpy(q_python.copy()), step.cache.key, step.cache.value, oracle
            ).reshape(-1)
            oracle_residual = feature + adapter.drafter.midlayer.self_attn.o_proj(oracle_attention)
            oracle_post = _frozen_rms_norm(
                oracle_residual, adapter.drafter.midlayer.post_attention_layernorm
            )
            oracle_gate = mlp.gate_proj(oracle_post)
            oracle_up = mlp.up_proj(oracle_post)
            oracle_silu = torch.from_numpy(
                ggml_silu(silu_library, oracle_gate.numpy(), scalar=False)
            )
            oracle_down = mlp.down_proj(oracle_up * oracle_silu)
            oracle_pre = oracle_residual + oracle_down
            oracle_norm = _frozen_rms_norm(oracle_pre, adapter.drafter.norm)
            oracle_logits = adapter.drafter.lm_head(oracle_norm).numpy().copy()
            results[-1]["native_query_rope_intervention"] = {
                "query_rope": q_metrics,
                "attention": _metrics(
                    oracle_attention.numpy(),
                    _column(graph[2 + depth]["kqv_out-0"], graph_values, 0),
                ),
                "ffn_input": _metrics(
                    oracle_residual.numpy(),
                    _column(graph[2 + depth]["ffn_inp-0"], graph_values, 0),
                ),
                "post_attn_norm": _metrics(
                    oracle_post.numpy(),
                    _column(graph[2 + depth]["post_attn_norm-0"], graph_values, 0),
                ),
                "ffn_output": _metrics(
                    oracle_down.numpy(),
                    _column(graph[2 + depth]["ffn_out-0"], graph_values, 0),
                ),
                "pre_norm": _metrics(oracle_pre.numpy(), native_prenorm),
                "head_state": _metrics(oracle_norm.numpy(), normalized_native),
                "head": _head_comparison(oracle_logits, mapping, head),
            }
            if depth == 4:
                native_query = _column(graph[6]["Qcur_rope-0"], graph_values, 0).reshape(32, 128)
                student_query = python_qk_to_native_rows(taps["Qcur_rope-0"].reshape(32, 128), 32)
                native_half = native_query.astype("<f2").view("<u2")
                student_half = student_query.astype("<f2").view("<u2")
                different = np.argwhere(native_half != student_half)
                if (1, 52) not in [tuple(index) for index in different]:
                    raise ValueError("expected depth-4 query threshold is absent")
                repaired_query = student_query.copy()
                repaired_query[1, 52] = native_query[1, 52]
                python_query = native_qk_to_python_rows(repaired_query, 32)
                repaired_attention = native_forward_f32_backward(
                    torch.from_numpy(python_query.copy()),
                    step.cache.key,
                    step.cache.value,
                    oracle,
                ).reshape(-1)
                residual = feature + adapter.drafter.midlayer.self_attn.o_proj(repaired_attention)
                repaired_post = _frozen_rms_norm(
                    residual, adapter.drafter.midlayer.post_attention_layernorm
                )
                mlp = adapter.drafter.midlayer.mlp
                gate = mlp.gate_proj(repaired_post)
                up = mlp.up_proj(repaired_post)
                silu_gate = torch.from_numpy(ggml_silu(silu_library, gate.numpy(), scalar=False))
                repaired_down = mlp.down_proj(up * silu_gate)
                repaired_pre = residual + repaired_down
                repaired_norm = _frozen_rms_norm(repaired_pre, adapter.drafter.norm)
                repaired_logits = adapter.drafter.lm_head(repaired_norm).numpy().copy()
                patched = {
                    "f16_query_differences": [
                        {
                            "head": int(h),
                            "channel": int(c),
                            "student_f16_bits": int(student_half[h, c]),
                            "native_f16_bits": int(native_half[h, c]),
                        }
                        for h, c in different
                    ],
                    "patched_head": 1,
                    "patched_channel": 52,
                    "attention": _metrics(
                        repaired_attention.numpy(),
                        _column(graph[6]["kqv_out-0"], graph_values, 0),
                    ),
                    "ffn_input": _metrics(
                        residual.numpy(), _column(graph[6]["ffn_inp-0"], graph_values, 0)
                    ),
                    "post_attn_norm": _metrics(
                        repaired_post.numpy(),
                        _column(graph[6]["post_attn_norm-0"], graph_values, 0),
                    ),
                    "ffn_output": _metrics(
                        repaired_down.numpy(),
                        _column(graph[6]["ffn_out-0"], graph_values, 0),
                    ),
                    "pre_norm": _metrics(repaired_pre.numpy(), native_prenorm),
                    "head_state": _metrics(repaired_norm.numpy(), normalized_native),
                    "head": _head_comparison(repaired_logits, mapping, head),
                }
                if patched["attention"]["exact_elements"] != 4096:
                    raise ValueError("single-query repair did not restore native attention")
                results[-1]["query_threshold_intervention"] = patched
            feature = oracle_pre
            cache = step.cache
    if results[0]["corrected_head_state"]["exact_elements"] != 2560:
        raise ValueError("multi-depth replay failed the established first-seed control")
    for row in results:
        intervention = row["native_query_rope_intervention"]
        head = intervention["head"]
        if (
            row["cache_key_f16_exact"] != 1024
            or row["cache_value_f16_exact"] != 1024
            or intervention["query_rope"]["exact_elements"] != 4096
            or intervention["attention"]["exact_elements"] != 4096
            or any(
                intervention[name]["exact_elements"] != 2560
                for name in ("ffn_input", "post_attn_norm", "ffn_output", "pre_norm", "head_state")
            )
            or head["probe"]["exact_elements"] != 8
            or head["argmax_logit"]["exact_elements"] != 1
            or head["label_logit"]["exact_elements"] != 1
            or head["mapped_argmax_id"] != head["native_argmax_id"]
            or head["target_rank"] != head["native_target_rank"]
        ):
            raise ValueError(f"native CPU forward diverged at depth {row['depth']}")
    return {
        "schema": "recurrent_reasoning_multidepth_cpu_diagnostic_v1",
        "scope": "one native-token-forced five-depth first-round reasoning chain; CPU only",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "forward_arithmetic": "ordered binary; native attention oracle; ggml vector SiLU",
        "backward_executed": False,
        "cache_audit_status": cache_audit["status"],
        "depths": results,
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "candidate_d": D_SHA256,
            "target_gguf": sha256(target),
            "drafter_config": sha256(config),
            "first_seed_report": sha256(seed_report),
            "attention_helper": sha256(attention_helper),
            "silu_report": sha256(silu_report),
            "silu_cpu_library": sha256(silu_library),
            "rope_report": sha256(rope_report),
            "rope_helper": sha256(rope_helper),
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
    parser.add_argument("--first-seed-report", type=Path, required=True)
    parser.add_argument("--attention-helper", type=Path, required=True)
    parser.add_argument("--silu-report", type=Path, required=True)
    parser.add_argument("--silu-cpu-library", type=Path, required=True)
    parser.add_argument("--rope-report", type=Path, required=True)
    parser.add_argument("--rope-helper", type=Path, required=True)
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
        args.first_seed_report,
        args.attention_helper,
        args.silu_report,
        args.silu_cpu_library,
        args.rope_report,
        args.rope_helper,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            [
                {
                    "depth": row["depth"],
                    "attention_exact": row["adapter_stages"]["kqv_out-0"]["exact_elements"],
                    "state_exact": row["corrected_head_state"]["exact_elements"],
                    "key_exact": row["cache_key_f16_exact"],
                    "value_exact": row["cache_value_f16_exact"],
                    "argmax_matches": row["head"]["mapped_argmax_id"]
                    == row["head"]["native_argmax_id"],
                }
                for row in result["depths"]
            ]
        )
    )


if __name__ == "__main__":
    main()
