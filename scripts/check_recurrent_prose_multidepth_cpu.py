#!/usr/bin/env python3
"""Build a sealed prose context cache and replay its five CPU draft depths."""

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
from run_binary_head_capture import TARGET_F16_SHA256, TRAIN_PROMPTS_SHA256  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_attention_oracle import (  # noqa: E402
    NativeCPUAttentionOracle,
    native_forward_f32_backward,
)
from w1a1_eagle.native_step import NativeStepCache, _frozen_rms_norm  # noqa: E402

PROSE_MANIFEST_SHA256 = "4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169"
CONTEXT = 31
DEPTHS = 5


def _verify_prose(capture: Path, target: Path, draft: Path) -> tuple[dict, str]:
    manifest_path = capture / "manifest.json"
    digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        digest != PROSE_MANIFEST_SHA256
        or manifest.get("schema") != "recurrent_cpu_native_diagnostic_v1"
        or manifest.get("execution_device") != "cpu"
        or manifest.get("training_eligible") is not False
        or manifest.get("prompt_id") != "qat-revisit-train-prose-urban-waterways-01"
        or manifest.get("source_sha256", {}).get("target") != sha256(target)
        or sha256(target) != TARGET_F16_SHA256
        or manifest.get("source_sha256", {}).get("draft") != sha256(draft)
        or sha256(draft) != D_SHA256
        or manifest.get("source_sha256", {}).get("frozen_train_prompts") != TRAIN_PROMPTS_SHA256
    ):
        raise ValueError("prose capture or frozen model identity differs")
    ledger = manifest.get("files")
    if not isinstance(ledger, dict):
        raise ValueError("prose manifest lacks a file ledger")
    actual_files = {path.name for path in capture.iterdir() if path.is_file()} - {"manifest.json"}
    if actual_files != set(ledger):
        raise ValueError("prose capture file set differs from sealed ledger")
    for name, entry in ledger.items():
        file = capture / name
        if (
            Path(name).name != name
            or file.stat().st_size != entry.get("bytes")
            or sha256(file) != entry.get("sha256")
        ):
            raise ValueError(f"prose capture file differs from ledger: {name}")
    return manifest, digest


def _ggml_key_rope(helper: Path, native_raw: np.ndarray) -> np.ndarray:
    if native_raw.shape != (CONTEXT, 1024) or native_raw.dtype != np.float32:
        raise ValueError("context raw key rows have wrong F32 geometry")
    with tempfile.TemporaryDirectory(prefix="prose-key-rope-") as directory:
        operands = Path(directory)
        native_raw.astype("<f4").tofile(operands / "key_raw.f32")
        np.arange(CONTEXT, dtype="<i4").tofile(operands / "positions.i32")
        subprocess.run([str(helper.resolve()), str(operands), str(CONTEXT)], check=True)
        output = operands / "key_rope.f32"
        if output.stat().st_size != native_raw.nbytes:
            raise ValueError("ggml context key RoPE returned wrong byte count")
        return np.fromfile(output, dtype="<f4").reshape(CONTEXT, 1024)


def _student_cache(keys: np.ndarray, values: np.ndarray) -> NativeStepCache:
    if keys.shape != (CONTEXT, 1024) or values.shape != keys.shape:
        raise ValueError("constructed prose context cache has wrong geometry")
    key_native = keys.view("<f2").astype("<f4").reshape(CONTEXT, 8, 128)
    key_python = native_qk_to_python_rows(key_native, 8).transpose(1, 0, 2).copy()
    value_python = values.view("<f2").astype("<f4").reshape(CONTEXT, 8, 128)
    value_python = value_python.transpose(1, 0, 2).copy()
    return NativeStepCache(torch.from_numpy(key_python), torch.from_numpy(value_python))


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
    manifest, manifest_sha = _verify_prose(capture, target, draft)
    rope = json.loads(rope_report.read_text())
    silu = json.loads(silu_report.read_text())
    if rope["source_sha256"]["helper_binary"] != sha256(rope_helper) or silu["source_sha256"][
        "ggml_cpu_library"
    ] != sha256(silu_library):
        raise ValueError("pinned ggml RoPE or SiLU helper differs")
    cache_audit = audit_cache(capture)
    events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in events if row.get("event") == "execution"]
    rows = [row for row in events if row.get("event") == "row"]
    native_k, native_v, _, ledger = reconstruct_slots(
        executions,
        rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        2 + DEPTHS - 1,
    )
    if len(ledger["selected_rows"]) != 1 or ledger["selected_rows"][0]["position"] != 35:
        raise ValueError("fifth prose depth is not at memory position 35")
    graph_records, graph_values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    graph = _group_decoder_taps(graph_records)
    if graph[1]["inp_embd"]["n_tokens"] != CONTEXT or any(
        graph[2 + depth]["inp_embd"]["n_tokens"] != 1 for depth in range(DEPTHS)
    ):
        raise ValueError("native prose context or proposal graph has wrong token counts")
    round_rows = [
        row for row in read_jsonl(capture / "forced-rounds.jsonl") if row.get("round_index") == 0
    ]
    if len(round_rows) != 1 or len(round_rows[0]["prefix_token_ids"]) != CONTEXT + 1:
        raise ValueError("prose first round has wrong prefix")
    round_row = round_rows[0]
    heads = [row for row in read_jsonl(capture / "heads.jsonl") if row.get("round_index") == 0]
    if len(heads) != DEPTHS or [row["depth"] for row in heads] != list(range(DEPTHS)):
        raise ValueError("prose first round lacks five native head records")
    if [row["input_token_id"] for row in heads] != [
        round_row["seed_token_id"],
        *round_row["draft_token_ids"][:4],
    ]:
        raise ValueError("prose head inputs do not follow native proposals")
    indices = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"),
        round_row["prefix_token_ids"],
        round_row["task_id"],
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
    raw_keys = []
    raw_values = []
    context_metrics = {
        name: {"exact": 0, "elements": 0, "max_abs": 0.0}
        for name in ("inp_embd", "embd_norm-0", "g_norm-0", "concat_embd-0", "Kcur-0", "Vcur-0")
    }
    with torch.no_grad():
        for position in range(CONTEXT):
            raw = torch.from_numpy(np.array(features[indices[position]], copy=True))
            feature = adapter.encode_feature(raw)
            token = round_row["prefix_token_ids"][position + 1]
            embedding = adapter.embedding_lookup(token).to(torch.float32)
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
                native = _column(graph[1][name], graph_values, position)
                metric = _metrics(actual, native)
                context_metrics[name]["exact"] += metric["exact_elements"]
                context_metrics[name]["elements"] += metric["elements"]
                context_metrics[name]["max_abs"] = max(
                    context_metrics[name]["max_abs"], metric["max_abs"]
                )
            raw_keys.append(key)
            raw_values.append(value)
    keys_f32 = np.stack(raw_keys)
    values_f32 = np.stack(raw_values)
    rope_keys = _ggml_key_rope(rope_helper, keys_f32)
    graph_rope = np.stack(
        [_column(graph[1]["Kcur_rope-0"], graph_values, i) for i in range(CONTEXT)]
    )
    context_metrics["Kcur_rope-0"] = _metrics(rope_keys, graph_rope)
    key_bits = rope_keys.astype("<f2").view("<u2")
    value_bits = values_f32.astype("<f2").view("<u2")
    key_exact = int(np.count_nonzero(key_bits == native_k[:CONTEXT]))
    value_exact = int(np.count_nonzero(value_bits == native_v[:CONTEXT]))
    cache = _student_cache(key_bits, value_bits)
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
                raise ValueError("prose native head-state row is invalid")
            seed_graph = graph[2 + depth]
            after_k = python_qk_to_native_rows(step.cache.key[:, position, :].numpy(), 8)
            after_k = after_k.astype("<f2").view("<u2").reshape(-1)
            after_v = step.cache.value[:, position, :].numpy().astype("<f2").view("<u2").reshape(-1)
            depth_results.append(
                {
                    "depth": depth,
                    "position": position,
                    "input_token_id": head["input_token_id"],
                    "native_proposed_token_id": head["proposed_token_id"],
                    "query_rope": _metrics(
                        q_rope_native, _column(seed_graph["Qcur_rope-0"], graph_values, 0)
                    ),
                    "attention": _metrics(
                        attention.numpy(), _column(seed_graph["kqv_out-0"], graph_values, 0)
                    ),
                    "ffn_input": _metrics(
                        residual.numpy(), _column(seed_graph["ffn_inp-0"], graph_values, 0)
                    ),
                    "post_attn_norm": _metrics(
                        post.numpy(), _column(seed_graph["post_attn_norm-0"], graph_values, 0)
                    ),
                    "ffn_output": _metrics(
                        down.numpy(), _column(seed_graph["ffn_out-0"], graph_values, 0)
                    ),
                    "pre_norm": _metrics(
                        prenorm.numpy(), _column(seed_graph["eagle3_prenorm-0"], graph_values, 0)
                    ),
                    "head_state": _metrics(normalized.numpy(), native_states[state_row]),
                    "head": _head_comparison(logits, mapping, head),
                    "key_write_exact": int(np.count_nonzero(after_k == native_k[position])),
                    "value_write_exact": int(np.count_nonzero(after_v == native_v[position])),
                }
            )
            feature = prenorm
            cache = step.cache
    for name, metric in context_metrics.items():
        exact = metric.get("exact", metric.get("exact_elements"))
        if exact != metric["elements"]:
            raise ValueError(f"prose context {name} differs from native graph")
    if key_exact != CONTEXT * 1024 or value_exact != CONTEXT * 1024:
        raise ValueError("constructed prose context cache differs from native stored bytes")
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
            raise ValueError(f"prose draft forward diverged at depth {row['depth']}")
    return {
        "schema": "recurrent_prose_multidepth_cpu_diagnostic_v1",
        "scope": "one frozen prose first round, independently constructed context cache",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "context_positions": CONTEXT,
        "context_stage_metrics": context_metrics,
        "context_key_write_exact": key_exact,
        "context_value_write_exact": value_exact,
        "context_write_elements": CONTEXT * 1024,
        "depths": depth_results,
        "cache_audit_status": cache_audit["status"],
        "backward_executed": False,
        "source_sha256": {
            "capture_manifest": manifest_sha,
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "drafter_config": sha256(config),
            "attention_helper": sha256(attention_helper),
            "rope_report": sha256(rope_report),
            "rope_helper": sha256(rope_helper),
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
