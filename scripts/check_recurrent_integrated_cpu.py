#!/usr/bin/env python3
"""Check the opt-in native CPU student forward on three sealed training rounds."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from audit_recurrent_draft_cache import audit as audit_cache  # noqa: E402
from check_recurrent_attention_operand_ablation import python_qk_to_native_rows  # noqa: E402
from check_recurrent_ffn_from_graph import _cpu_model, _metrics  # noqa: E402
from check_recurrent_multidepth_cpu import _head_comparison  # noqa: E402
from check_recurrent_native_attention_operator import (  # noqa: E402
    _group_decoder_taps,
    reconstruct_slots,
)
from check_recurrent_real_step import _build_drafter, _retained_feature_indices  # noqa: E402
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import D_SHA256  # noqa: E402
from run_binary_head_capture import TARGET_F16_SHA256, TRAIN_PROMPTS_SHA256  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_cpu_diagnostic import NativeCPUDiagnosticOperators  # noqa: E402
from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache  # noqa: E402

CASES = {
    "reasoning_first": {
        "prompt_id": "qat-revisit-train-reasoning-rate-and-work-01",
        "manifest": "99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40",
        "round": 0,
        "context": 46,
        "execution": 2,
        "depths": 5,
    },
    "prose_first": {
        "prompt_id": "qat-revisit-train-prose-urban-waterways-01",
        "manifest": "4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169",
        "round": 0,
        "context": 31,
        "execution": 2,
        "depths": 5,
    },
    "reasoning_postaccept": {
        "prompt_id": "qat-revisit-train-reasoning-rate-and-work-01",
        "manifest": "99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40",
        "round": 2,
        "context": 49,
        "execution": 14,
        "depths": 3,
    },
}
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


def _verify_capture(capture: Path, case: dict, target: Path, draft: Path) -> tuple[dict, str]:
    manifest_path = capture / "manifest.json"
    digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get("source_sha256", {})
    if (
        digest != case["manifest"]
        or manifest.get("schema") != "recurrent_cpu_native_diagnostic_v1"
        or manifest.get("execution_device") != "cpu"
        or manifest.get("training_eligible") is not False
        or manifest.get("prompt_id") != case["prompt_id"]
        or sources.get("target") != sha256(target)
        or sources.get("draft") != sha256(draft)
        or sources.get("target") != TARGET_F16_SHA256
        or sources.get("draft") != D_SHA256
        or sources.get("frozen_train_prompts") != TRAIN_PROMPTS_SHA256
    ):
        raise ValueError("sealed CPU capture or frozen model identity differs")
    ledger = manifest.get("files")
    if not isinstance(ledger, dict):
        raise ValueError("sealed capture lacks a file ledger")
    actual_files = {path.name for path in capture.iterdir() if path.is_file()} - {"manifest.json"}
    if actual_files != set(ledger):
        raise ValueError("sealed capture file set differs from manifest")
    for name, entry in ledger.items():
        file = capture / name
        if (
            Path(name).name != name
            or file.stat().st_size != entry.get("bytes")
            or sha256(file) != entry.get("sha256")
        ):
            raise ValueError(f"sealed capture file differs: {name}")
    return manifest, digest


def _all_exact(row: dict) -> bool:
    if row["cache_key_f16_exact"] != 1024 or row["cache_value_f16_exact"] != 1024:
        return False
    for name, value in row["stages"].items():
        if value["exact_elements"] != value["elements"]:
            return False
    head = row["head"]
    return (
        head["probe"]["exact_elements"] == 8
        and head["argmax_logit"]["exact_elements"] == 1
        and head["label_logit"]["exact_elements"] == 1
        and head["mapped_argmax_id"] == head["native_argmax_id"]
        and head["target_rank"] == head["native_target_rank"]
    )


def audit(
    case_name: str,
    capture: Path,
    target: Path,
    draft: Path,
    config: Path,
    attention_helper: Path,
    rope_helper: Path,
    silu_library: Path,
) -> dict:
    started = time.monotonic()
    case = CASES[case_name]
    manifest, manifest_sha = _verify_capture(capture, case, target, draft)
    cache_audit = audit_cache(capture)
    events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in events if row.get("event") == "execution"]
    cache_rows = [row for row in events if row.get("event") == "row"]
    native_k, native_v, _, ledger = reconstruct_slots(
        executions,
        cache_rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        case["execution"] + case["depths"] - 1,
    )
    if (
        len(ledger["selected_rows"]) != 1
        or ledger["selected_rows"][0]["position"] != case["context"] + case["depths"] - 1
    ):
        raise ValueError("native final execution does not join selected round")
    records, graph_values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    graph = _group_decoder_taps(records)
    if any(
        graph[case["execution"] + depth]["inp_embd"]["n_tokens"] != 1
        for depth in range(case["depths"])
    ):
        raise ValueError("selected native proposal executions are not single-token")
    rounds = [
        row
        for row in read_jsonl(capture / "forced-rounds.jsonl")
        if row.get("round_index") == case["round"]
    ]
    if len(rounds) != 1 or len(rounds[0]["prefix_token_ids"]) != case["context"] + 1:
        raise ValueError("selected round has wrong accepted prefix")
    round_row = rounds[0]
    heads = [
        row
        for row in read_jsonl(capture / "heads.jsonl")
        if row.get("round_index") == case["round"]
    ]
    if len(heads) != case["depths"] or [row["depth"] for row in heads] != list(
        range(case["depths"])
    ):
        raise ValueError("selected round lacks ordered native head records")
    if [row["input_token_id"] for row in heads] != [
        round_row["seed_token_id"],
        *round_row["draft_token_ids"][: case["depths"] - 1],
    ]:
        raise ValueError("selected head inputs do not follow captured native proposals")
    feature_indices = _retained_feature_indices(
        read_jsonl(capture / "heads.target_features.jsonl"),
        round_row["prefix_token_ids"],
        round_row["task_id"],
    )
    feature_values = np.memmap(capture / "heads.target_features.f32", dtype="<f4", mode="r")
    feature_values = feature_values.reshape(-1, 7680)
    raw = torch.from_numpy(np.array(feature_values[feature_indices], copy=True))
    operands = FrozenOperands(target, draft)
    operators = NativeCPUDiagnosticOperators(rope_helper, silu_library, attention_helper)
    adapter = _build_drafter(
        config,
        draft,
        operands,
        "native_order",
        attention_mode="native_cpu_diagnostic",
        native_cpu_operators=operators,
    )
    mapping = np.asarray(Model(draft).tensors["d2t"].data, dtype=np.int64)
    with torch.no_grad():
        rebuilt = rebuild_prefix_cache(
            round_row["prefix_token_ids"] + [round_row["seed_token_id"]],
            raw,
            list(range(len(round_row["prefix_token_ids"]))),
            parent_position=case["context"],
            encode_feature=adapter.encode_feature,
            decode_context=adapter.decode_context,
            new_cache=adapter.new_cache,
        )
        if rebuilt.decoder_position != case["context"]:
            raise ValueError("rebuilt cache has the wrong decoder position")
        context_key = (
            python_qk_to_native_rows(rebuilt.cache.key.permute(1, 0, 2).numpy(), 8)
            .astype("<f2")
            .view("<u2")
            .reshape(case["context"], 1024)
        )
        context_value = rebuilt.cache.value.permute(1, 0, 2).numpy()
        context_value = context_value.astype("<f2").view("<u2").reshape(case["context"], 1024)
        context_key_exact = int(np.count_nonzero(context_key == native_k[: case["context"]]))
        context_value_exact = int(np.count_nonzero(context_value == native_v[: case["context"]]))
        feature = adapter.encode_feature(rebuilt.seed_raw_features)
        cache = rebuilt.cache
        native_states = np.memmap(capture / "heads.f32", dtype="<f4", mode="r")
        native_states = native_states.reshape(-1, 2560)
        depth_results = []
        for depth, head in enumerate(heads):
            position = case["context"] + depth
            if head["input_position"] != position + 1:
                raise ValueError("native head input position differs")
            taps: dict[str, np.ndarray] = {}

            def record_tap(name: str, value: torch.Tensor) -> None:
                taps[name] = value.numpy().astype("<f4", copy=True).reshape(-1)

            step = adapter.decode_step(
                head["input_token_id"],
                feature,
                position,
                cache,
                compute_logits=True,
                trace_callback=record_tap,
            )
            group = graph[case["execution"] + depth]
            stage_metrics = {}
            for name in GRAPH_TAPS:
                actual = taps[name]
                if name.startswith("Qcur"):
                    actual = python_qk_to_native_rows(actual.reshape(32, 128), 32).reshape(-1)
                elif name.startswith("Kcur"):
                    actual = python_qk_to_native_rows(actual.reshape(8, 128), 8).reshape(-1)
                stage_metrics[name] = _metrics(actual, _column(group[name], graph_values, 0))
            state_row = head["state_row"]
            if type(state_row) is not int or not 0 <= state_row < len(native_states):
                raise ValueError("native normalized head-state row is invalid")
            stage_metrics["result_norm"] = _metrics(taps["result_norm"], native_states[state_row])
            key_bits = (
                python_qk_to_native_rows(step.cache.key[:, position, :].numpy(), 8)
                .astype("<f2")
                .view("<u2")
                .reshape(-1)
            )
            value_bits = step.cache.value[:, position, :].numpy()
            value_bits = value_bits.astype("<f2").view("<u2").reshape(-1)
            depth_results.append(
                {
                    "depth": depth,
                    "memory_position": position,
                    "execution": case["execution"] + depth,
                    "input_token_id": head["input_token_id"],
                    "native_proposed_token_id": head["proposed_token_id"],
                    "cache_key_f16_exact": int(np.count_nonzero(key_bits == native_k[position])),
                    "cache_value_f16_exact": int(
                        np.count_nonzero(value_bits == native_v[position])
                    ),
                    "stages": stage_metrics,
                    "head": _head_comparison(step.logits.numpy(), mapping, head),
                }
            )
            feature = step.pre_norm
            cache = step.cache
    status = (
        "bitwise_exact"
        if context_key_exact == case["context"] * 1024
        and context_value_exact == case["context"] * 1024
        and all(_all_exact(row) for row in depth_results)
        else "mismatch"
    )
    return {
        "schema": "recurrent_integrated_native_cpu_diagnostic_v1",
        "case": case_name,
        "status": status,
        "scope": "one sealed native-token-following training round; no backward or optimizer",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "round_index": case["round"],
        "context_positions": case["context"],
        "context_key_f16_exact": context_key_exact,
        "context_value_f16_exact": context_value_exact,
        "context_elements": case["context"] * 1024,
        "depths": depth_results,
        "cache_audit_status": cache_audit["status"],
        "backward_executed": False,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "capture_manifest": manifest_sha,
            "target_gguf": sha256(target),
            "candidate_d": D_SHA256,
            "drafter_config": sha256(config),
            "attention_helper": sha256(attention_helper),
            "rope_helper": sha256(rope_helper),
            "silu_cpu_library": sha256(silu_library),
            "diagnostic_operators": sha256(
                Path(__file__).resolve().parents[1] / "src/w1a1_eagle/native_cpu_diagnostic.py"
            ),
            "native_step": sha256(
                Path(__file__).resolve().parents[1] / "src/w1a1_eagle/native_step.py"
            ),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__, "torch": torch.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--drafter-config", type=Path, required=True)
    parser.add_argument("--attention-helper", type=Path, required=True)
    parser.add_argument("--rope-helper", type=Path, required=True)
    parser.add_argument("--silu-cpu-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    torch.set_num_threads(10)
    result = audit(
        args.case,
        args.capture_dir,
        args.target_gguf,
        args.candidate_d,
        args.drafter_config,
        args.attention_helper,
        args.rope_helper,
        args.silu_cpu_library,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case": result["case"],
                "status": result["status"],
                "context_key_exact": result["context_key_f16_exact"],
                "context_value_exact": result["context_value_f16_exact"],
                "state_exact": [
                    row["stages"]["result_norm"]["exact_elements"] for row in result["depths"]
                ],
            }
        )
    )
    if result["status"] != "bitwise_exact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
