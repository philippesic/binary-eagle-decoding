#!/usr/bin/env python3
"""Replay the sealed reasoning post-acceptance round with rebuilt CPU cache."""

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
from check_recurrent_multidepth_cpu import _ggml_query_rope, _head_comparison  # noqa: E402
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
from w1a1_eagle.native_attention_oracle import (  # noqa: E402
    NativeCPUAttentionOracle,
    native_forward_f32_backward,
)
from w1a1_eagle.native_step import NativeStepCache, _frozen_rms_norm  # noqa: E402

ROUND = 2
CONTEXT = 49
DEPTHS = 3
SEED_EXECUTION = 14
LAST_EXECUTION = SEED_EXECUTION + DEPTHS - 1


def _ggml_key_rope(helper: Path, raw: np.ndarray) -> np.ndarray:
    if raw.shape != (CONTEXT, 1024) or raw.dtype != np.float32:
        raise ValueError("post-acceptance raw K has wrong F32 geometry")
    with tempfile.TemporaryDirectory(prefix="postacceptance-key-rope-") as directory:
        operands = Path(directory)
        raw.astype("<f4").tofile(operands / "key_raw.f32")
        np.arange(CONTEXT, dtype="<i4").tofile(operands / "positions.i32")
        subprocess.run([str(helper.resolve()), str(operands), str(CONTEXT)], check=True)
        output = operands / "key_rope.f32"
        if output.stat().st_size != raw.nbytes:
            raise ValueError("ggml post-acceptance K RoPE returned wrong bytes")
        return np.fromfile(output, dtype="<f4").reshape(CONTEXT, 1024)


def _cache(keys: np.ndarray, values: np.ndarray) -> NativeStepCache:
    if keys.shape != (CONTEXT, 1024) or values.shape != keys.shape:
        raise ValueError("rebuilt post-acceptance cache has wrong geometry")
    k_native = keys.view("<f2").astype("<f4").reshape(CONTEXT, 8, 128)
    k_python = native_qk_to_python_rows(k_native, 8).transpose(1, 0, 2).copy()
    v_python = values.view("<f2").astype("<f4").reshape(CONTEXT, 8, 128)
    v_python = v_python.transpose(1, 0, 2).copy()
    return NativeStepCache(torch.from_numpy(k_python), torch.from_numpy(v_python))


def audit(
    capture: Path,
    target: Path,
    draft: Path,
    config: Path,
    attention_helper: Path,
    rope_helper: Path,
    rope_report: Path,
    silu_library: Path,
    silu_report: Path,
) -> dict:
    manifest, seal = verify_capture(capture, require_stages=False)
    verify_baseline_identity(seal)
    rope = json.loads(rope_report.read_text())
    silu = json.loads(silu_report.read_text())
    if (
        sha256(target) != manifest["source_sha256"]["target"]
        or sha256(draft) != D_SHA256
        or rope["source_sha256"]["helper_binary"] != sha256(rope_helper)
        or silu["source_sha256"]["ggml_cpu_library"] != sha256(silu_library)
    ):
        raise ValueError("reasoning model or pinned CPU operator identity differs")
    cache_audit = audit_cache(capture)
    cache_events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in cache_events if row.get("event") == "execution"]
    cache_rows = [row for row in cache_events if row.get("event") == "row"]
    cache_bits = np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r")
    mask_bits = np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r")
    before_k, before_v, _, before_ledger = reconstruct_slots(
        executions, cache_rows, cache_bits, mask_bits, SEED_EXECUTION
    )
    final_k, final_v, _, final_ledger = reconstruct_slots(
        executions, cache_rows, cache_bits, mask_bits, LAST_EXECUTION
    )
    if (
        len(before_ledger["selected_rows"]) != 1
        or before_ledger["selected_rows"][0]["position"] != CONTEXT
        or len(final_ledger["selected_rows"]) != 1
        or final_ledger["selected_rows"][0]["position"] != CONTEXT + DEPTHS - 1
        or not np.array_equal(before_k[:CONTEXT], final_k[:CONTEXT])
        or not np.array_equal(before_v[:CONTEXT], final_v[:CONTEXT])
    ):
        raise ValueError("native round-2 cache ancestry differs before/after proposals")
    latest_rows = {}
    rewrite_counts = {}
    for row in cache_rows:
        if row.get("execution") >= SEED_EXECUTION:
            break
        position = row["position"]
        latest_rows[position] = row
        rewrite_counts[position] = rewrite_counts.get(position, 0) + 1
    if not set(range(CONTEXT)).issubset(latest_rows):
        raise ValueError("post-acceptance context has missing physical writes")
    records, graph_values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    graph = _group_decoder_taps(records)
    round_rows = [
        row
        for row in read_jsonl(capture / "forced-rounds.jsonl")
        if row.get("round_index") == ROUND
    ]
    prior_rounds = [
        row
        for row in read_jsonl(capture / "forced-rounds.jsonl")
        if row.get("round_index") == ROUND - 1
    ]
    if len(round_rows) != 1 or len(round_rows[0]["prefix_token_ids"]) != CONTEXT + 1:
        raise ValueError("native reasoning round 2 has wrong accepted prefix")
    if len(prior_rounds) != 1 or prior_rounds[0].get("accepted_drafts") != 1:
        raise ValueError("selected round does not follow one accepted draft")
    round_row = round_rows[0]
    prefix = round_row["prefix_token_ids"]
    if round_row["accepted_drafts"] != 2 or round_row["seed_token_id"] != 1447:
        raise ValueError("selected round is not the frozen post-acceptance case")
    heads = [row for row in read_jsonl(capture / "heads.jsonl") if row.get("round_index") == ROUND]
    if len(heads) != DEPTHS or [row["depth"] for row in heads] != list(range(DEPTHS)):
        raise ValueError("post-acceptance round has wrong native head depth")
    if [row["input_token_id"] for row in heads] != [
        round_row["seed_token_id"],
        *round_row["draft_token_ids"][: DEPTHS - 1],
    ]:
        raise ValueError("post-acceptance native head tokens do not follow proposals")
    indices = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"), prefix, round_row["task_id"]
    )
    features = np.memmap(capture / "heads.target_features.f32", dtype="<f4", mode="r")
    features = features.reshape(-1, 7680)
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
    context_metrics = {
        name: {"exact": 0, "elements": 0, "max_abs": 0.0}
        for name in ("inp_embd", "embd_norm-0", "g_norm-0", "concat_embd-0", "Kcur-0", "Vcur-0")
    }
    context_sources = []
    raw_keys = []
    raw_values = []
    with torch.no_grad():
        for position in range(CONTEXT):
            row = latest_rows[position]
            if row["token_id"] != prefix[position + 1] or row["slot"] != position:
                raise ValueError(f"position {position} latest cache write is not accepted prefix")
            group = graph[row["execution"]]
            column = row["column"]
            feature = adapter.encode_feature(
                torch.from_numpy(np.array(features[indices[position]], copy=True))
            )
            embedding = adapter.embedding_lookup(prefix[position + 1]).to(torch.float32)
            emb_norm = _frozen_rms_norm(embedding, adapter.drafter.midlayer.input_layernorm)
            g_norm = _frozen_rms_norm(feature, adapter.drafter.midlayer.hidden_norm)
            fused = torch.cat((emb_norm, g_norm))
            key = adapter.drafter.midlayer.self_attn.k_proj(fused).numpy().copy()
            key = python_qk_to_native_rows(key.reshape(8, 128), 8).reshape(-1)
            value = adapter.drafter.midlayer.self_attn.v_proj(fused).numpy().copy()
            stages = {
                "inp_embd": embedding.numpy(),
                "embd_norm-0": emb_norm.numpy(),
                "g_norm-0": g_norm.numpy(),
                "concat_embd-0": fused.numpy(),
                "Kcur-0": key,
                "Vcur-0": value,
            }
            for name, actual in stages.items():
                native = _column(group[name], graph_values, column)
                metric = _metrics(actual, native)
                context_metrics[name]["exact"] += metric["exact_elements"]
                context_metrics[name]["elements"] += metric["elements"]
                context_metrics[name]["max_abs"] = max(
                    context_metrics[name]["max_abs"], metric["max_abs"]
                )
            context_sources.append(
                {
                    "position": position,
                    "token_id": row["token_id"],
                    "execution": row["execution"],
                    "column": column,
                    "writes_before_seed": rewrite_counts[position],
                }
            )
            raw_keys.append(key)
            raw_values.append(value)
    keys_f32 = np.stack(raw_keys)
    values_f32 = np.stack(raw_values)
    rope_keys = _ggml_key_rope(rope_helper, keys_f32)
    native_rope = np.stack(
        [
            _column(
                graph[latest_rows[i]["execution"]]["Kcur_rope-0"],
                graph_values,
                latest_rows[i]["column"],
            )
            for i in range(CONTEXT)
        ]
    )
    context_metrics["Kcur_rope-0"] = _metrics(rope_keys, native_rope)
    key_bits = rope_keys.astype("<f2").view("<u2")
    value_bits = values_f32.astype("<f2").view("<u2")
    key_exact = int(np.count_nonzero(key_bits == before_k[:CONTEXT]))
    value_exact = int(np.count_nonzero(value_bits == before_v[:CONTEXT]))
    cache = _cache(key_bits, value_bits)
    native_states = np.memmap(capture / "heads.f32", dtype="<f4", mode="r").reshape(-1, 2560)
    depth_results = []
    with torch.no_grad():
        feature = adapter.encode_feature(
            torch.from_numpy(np.array(features[indices[CONTEXT]], copy=True))
        )
        for depth, head in enumerate(heads):
            position = CONTEXT + depth
            taps: dict[str, np.ndarray] = {}

            def record_tap(name: str, value: torch.Tensor) -> None:
                taps[name] = value.numpy().astype("<f4", copy=True).reshape(-1)

            step = adapter.decode_step(
                head["input_token_id"],
                feature,
                position,
                cache,
                compute_logits=False,
                trace_callback=record_tap,
            )
            q_native = python_qk_to_native_rows(taps["Qcur-0"].reshape(32, 128), 32)
            q_rope_native = _ggml_query_rope(rope_helper, q_native.reshape(-1), position)
            q_python = native_qk_to_python_rows(q_rope_native.reshape(32, 128), 32)
            attention = native_forward_f32_backward(
                torch.from_numpy(q_python.copy()), step.cache.key, step.cache.value, oracle
            ).reshape(-1)
            residual = feature + adapter.drafter.midlayer.self_attn.o_proj(attention)
            post = _frozen_rms_norm(residual, adapter.drafter.midlayer.post_attention_layernorm)
            mlp = adapter.drafter.midlayer.mlp
            gate = mlp.gate_proj(post)
            up = mlp.up_proj(post)
            activated = torch.from_numpy(ggml_silu(silu_library, gate.numpy(), scalar=False))
            down = mlp.down_proj(up * activated)
            prenorm = residual + down
            normalized = _frozen_rms_norm(prenorm, adapter.drafter.norm)
            logits = adapter.drafter.lm_head(normalized).numpy().copy()
            state_row = head["state_row"]
            if type(state_row) is not int or not 0 <= state_row < len(native_states):
                raise ValueError("post-acceptance native head-state row is invalid")
            group = graph[SEED_EXECUTION + depth]
            after_k = python_qk_to_native_rows(step.cache.key[:, position, :].numpy(), 8)
            after_k = after_k.astype("<f2").view("<u2").reshape(-1)
            after_v = step.cache.value[:, position, :].numpy()
            after_v = after_v.astype("<f2").view("<u2").reshape(-1)
            depth_results.append(
                {
                    "depth": depth,
                    "execution": SEED_EXECUTION + depth,
                    "position": position,
                    "input_token_id": head["input_token_id"],
                    "native_proposed_token_id": head["proposed_token_id"],
                    "query_rope": _metrics(
                        q_rope_native, _column(group["Qcur_rope-0"], graph_values, 0)
                    ),
                    "attention": _metrics(
                        attention.numpy(), _column(group["kqv_out-0"], graph_values, 0)
                    ),
                    "ffn_input": _metrics(
                        residual.numpy(), _column(group["ffn_inp-0"], graph_values, 0)
                    ),
                    "post_attn_norm": _metrics(
                        post.numpy(), _column(group["post_attn_norm-0"], graph_values, 0)
                    ),
                    "ffn_output": _metrics(
                        down.numpy(), _column(group["ffn_out-0"], graph_values, 0)
                    ),
                    "pre_norm": _metrics(
                        prenorm.numpy(), _column(group["eagle3_prenorm-0"], graph_values, 0)
                    ),
                    "head_state": _metrics(normalized.numpy(), native_states[state_row]),
                    "head": _head_comparison(logits, mapping, head),
                    "key_write_exact": int(np.count_nonzero(after_k == final_k[position])),
                    "value_write_exact": int(np.count_nonzero(after_v == final_v[position])),
                }
            )
            feature = prenorm
            cache = step.cache
    for name, metric in context_metrics.items():
        exact = metric.get("exact", metric.get("exact_elements"))
        if exact != metric["elements"]:
            raise ValueError(f"post-acceptance context {name} differs from native graph")
    if key_exact != CONTEXT * 1024 or value_exact != CONTEXT * 1024:
        raise ValueError("rebuilt post-acceptance context cache differs from native bytes")
    for row in depth_results:
        head = row["head"]
        if (
            row["key_write_exact"] != 1024
            or row["value_write_exact"] != 1024
            or row["query_rope"]["exact_elements"] != 4096
            or row["attention"]["exact_elements"] != 4096
            or any(
                row[name]["exact_elements"] != 2560
                for name in ("ffn_input", "post_attn_norm", "ffn_output", "pre_norm", "head_state")
            )
            or head["probe"]["exact_elements"] != 8
            or head["argmax_logit"]["exact_elements"] != 1
            or head["label_logit"]["exact_elements"] != 1
            or head["mapped_argmax_id"] != head["native_argmax_id"]
            or head["target_rank"] != head["native_target_rank"]
        ):
            raise ValueError(f"post-acceptance forward diverged at depth {row['depth']}")
    return {
        "schema": "recurrent_reasoning_postacceptance_cpu_diagnostic_v1",
        "scope": "frozen reasoning round 2 with shifted accepted-prefix cache rewrites",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "round_index": ROUND,
        "accepted_drafts_in_prior_round": prior_rounds[0]["accepted_drafts"],
        "accepted_drafts_this_round": round_row["accepted_drafts"],
        "context_positions": CONTEXT,
        "context_sources": context_sources,
        "context_stage_metrics": context_metrics,
        "context_key_write_exact": key_exact,
        "context_value_write_exact": value_exact,
        "context_write_elements": CONTEXT * 1024,
        "depths": depth_results,
        "cache_audit_status": cache_audit["status"],
        "backward_executed": False,
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "drafter_config": sha256(config),
            "attention_helper": sha256(attention_helper),
            "rope_helper": sha256(rope_helper),
            "rope_report": sha256(rope_report),
            "silu_cpu_library": sha256(silu_library),
            "silu_report": sha256(silu_report),
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
    parser.add_argument("--attention-helper", type=Path, required=True)
    parser.add_argument("--rope-helper", type=Path, required=True)
    parser.add_argument("--rope-report", type=Path, required=True)
    parser.add_argument("--silu-cpu-library", type=Path, required=True)
    parser.add_argument("--silu-report", type=Path, required=True)
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
        args.attention_helper,
        args.rope_helper,
        args.rope_report,
        args.silu_cpu_library,
        args.silu_report,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "context_key_exact": result["context_key_write_exact"],
                "context_value_exact": result["context_value_write_exact"],
                "depth_state_exact": [
                    row["head_state"]["exact_elements"] for row in result["depths"]
                ],
                "depth_argmax_matches": [
                    row["head"]["mapped_argmax_id"] == row["head"]["native_argmax_id"]
                    for row in result["depths"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
