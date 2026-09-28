#!/usr/bin/env python3
"""Compare one real native D proposal chain with the CPU adapter.

This diagnostic rebuilds one round's D cache from accepted-prefix raw
target features and reports numeric drift. It never trains or uses an
accelerator. Grouped-matmul arithmetic is approximate to native reduction;
the report does not confer parity automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import load_candidate_d_arrays  # noqa: E402
from prepare_recurrent_native_rows import read_jsonl, sha256  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_attention_oracle import NativeCPUAttentionOracle  # noqa: E402
from w1a1_eagle.native_step import NativeStepAdapter, bind_frozen_norms  # noqa: E402
from w1a1_eagle.recurrent_binary import (  # noqa: E402
    CANDIDATE_D_BASE_TO_PATH,
    GroupedBinaryLinear,
    _native_qk_to_python_rows,
    unpack_signs,
)
from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache  # noqa: E402


def _norm(width: int, epsilon: float) -> nn.Module:
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))
    norm.variance_epsilon = epsilon
    return norm


def _build_drafter(
    config_path: Path,
    candidate_d: Path,
    operands: FrozenOperands,
    arithmetic: str,
    *,
    attention_mode: str = "f32",
    native_attention_oracle: NativeCPUAttentionOracle | None = None,
) -> NativeStepAdapter:
    config = json.loads(config_path.read_text())
    if (
        config.get("hidden_size") != 2560
        or config.get("head_dim") != 128
        or config.get("num_attention_heads") != 32
        or config.get("num_key_value_heads") != 8
        or config.get("intermediate_size") != 9728
        or config.get("rms_norm_eps") != 1e-6
    ):
        raise ValueError("drafter config differs from the pinned native geometry")
    arrays, _ = load_candidate_d_arrays(candidate_d)
    drafter = nn.Module()
    drafter.config = SimpleNamespace(**config)
    drafter.early_stop_method = None
    drafter.tree_mask = None
    drafter.midlayer = nn.Module()
    drafter.midlayer.self_attn = nn.Module()
    drafter.midlayer.mlp = nn.Module()
    for path in (
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
        "norm",
    ):
        parent_name, name = path.rsplit(".", 1) if "." in path else ("", path)
        parent = drafter.get_submodule(parent_name) if parent_name else drafter
        setattr(parent, name, _norm(2560, config["rms_norm_eps"]))
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        packed, scales = arrays[base]
        width = packed.shape[1] * 32
        if base in ("blk.0.attn_q", "blk.0.attn_k"):
            heads = 32 if base.endswith("_q") else 8
            signs = _native_qk_to_python_rows(unpack_signs(packed, width), heads)
            scale_rows = _native_qk_to_python_rows(scales, heads)
            binary = GroupedBinaryLinear(
                torch.from_numpy(np.ascontiguousarray(signs)),
                torch.from_numpy(np.ascontiguousarray(scale_rows)),
                group_size=128,
                arithmetic=arithmetic,
            )
        else:
            binary = GroupedBinaryLinear.from_packed(
                packed, scales, in_features=width, arithmetic=arithmetic
            )
        parent_name, name = path.rsplit(".", 1) if "." in path else ("", path)
        parent = drafter.get_submodule(parent_name) if parent_name else drafter
        setattr(parent, name, binary)
    bind_frozen_norms(drafter, operands.norm_arrays)
    return NativeStepAdapter(
        drafter,
        embedding_lookup=operands,
        attention_mode=attention_mode,
        native_attention_oracle=native_attention_oracle,
    )


def _attention_oracle(mode: str, helper: Path | None) -> NativeCPUAttentionOracle | None:
    if mode == "f32":
        if helper is not None:
            raise ValueError("native attention helper requires native attention mode")
        return None
    if mode != "native_forward_f32_backward" or helper is None:
        raise ValueError("native attention mode requires a pinned helper")
    return NativeCPUAttentionOracle(helper, threads=10)


def _retained_feature_indices(events: list[dict], prefix: list[int], task_id: int) -> list[int]:
    decoded = [row for row in events if row.get("event") == "decoded_row"]
    dispositions = {
        row.get("feature_row"): row for row in events if row.get("event") == "disposition"
    }
    indices = []
    for position, token in enumerate(prefix):
        matches = [
            row.get("feature_row")
            for row in decoded
            if row.get("task_id") == task_id
            and row.get("position") == position
            and row.get("token_id") == token
            and row.get("prefix_token_ids") == prefix[: position + 1]
            and dispositions.get(row.get("feature_row"), {}).get("retained_input") is True
        ]
        if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 0:
            raise ValueError("accepted prefix lacks exactly one retained native feature row")
        indices.append(matches[0])
    return indices


def check_round(
    capture_dir: Path,
    target_gguf: Path,
    draft_gguf: Path,
    drafter_config: Path,
    arithmetic: str = "group_matmul",
    tap_output: Path | None = None,
    round_index: int = 0,
    attention_mode: str = "f32",
    native_attention_helper: Path | None = None,
) -> dict:
    if (
        arithmetic not in {"native_order", "group_matmul"}
        or type(round_index) is not int
        or round_index < 0
    ):
        raise ValueError("unknown CPU binary arithmetic")
    start = time.monotonic()
    rounds = read_jsonl(capture_dir / "forced-rounds.jsonl")
    heads = read_jsonl(capture_dir / "heads.jsonl")
    events = read_jsonl(capture_dir / "heads.target_features.jsonl")
    selected = [record for record in rounds if record.get("round_index") == round_index]
    if len(selected) != 1:
        raise ValueError("capture lacks one selected native round")
    record = selected[0]
    prefix = record.get("prefix_token_ids")
    if (
        not isinstance(prefix, list)
        or len(prefix) < 2
        or not any(
            head.get("round_index") == round_index and head.get("depth") == 0 for head in heads
        )
    ):
        raise ValueError("selected native round/head row is not the expected prefix")
    feature_indices = _retained_feature_indices(events, prefix, record["task_id"])
    feature_path = capture_dir / "heads.target_features.f32"
    feature_values = np.memmap(feature_path, dtype="<f4", mode="r")
    if feature_values.size % 7680:
        raise ValueError("raw target feature file has wrong width")
    feature_values = feature_values.reshape(-1, 7680)
    if max(feature_indices) >= len(feature_values):
        raise ValueError("retained native feature row exceeds F32 payload")
    native_path = capture_dir / "heads.f32"
    native_values = np.memmap(native_path, dtype="<f4", mode="r")
    if native_values.size != len(heads) * 2560:
        raise ValueError("native head-state file has wrong row count")
    native_values = native_values.reshape(-1, 2560)
    operands = FrozenOperands(target_gguf, draft_gguf)
    oracle = _attention_oracle(attention_mode, native_attention_helper)
    adapter = _build_drafter(
        drafter_config,
        draft_gguf,
        operands,
        arithmetic,
        attention_mode=attention_mode,
        native_attention_oracle=oracle,
    )
    round_heads = [
        row
        for row in heads
        if row.get("round_index") == round_index and row.get("task_id") == record.get("task_id")
    ]
    proposed = record.get("draft_token_ids")
    if (
        not isinstance(proposed, list)
        or len(round_heads) != len(proposed)
        or [row.get("depth") for row in round_heads] != list(range(len(proposed)))
        or any(
            row.get("state_dim") != 2560
            or type(row.get("state_row")) is not int
            or not 0 <= row["state_row"] < len(heads)
            for row in round_heads
        )
    ):
        raise ValueError("selected native round has incomplete proposal head states")
    mapping = np.asarray(Model(draft_gguf).tensors["d2t"].data)
    depth_rows = []
    taps: dict[str, np.ndarray] = {}

    def record_tap(name: str, value: torch.Tensor) -> None:
        if name in taps:
            raise ValueError(f"duplicate adapter tap {name}")
        taps[name] = value.cpu().numpy().astype("<f4", copy=True).reshape(-1)

    with torch.no_grad():
        raw = torch.from_numpy(np.array(feature_values[feature_indices], copy=True))
        rebuilt = rebuild_prefix_cache(
            prefix + [record["seed_token_id"]],
            raw,
            list(range(len(prefix))),
            parent_position=len(prefix) - 1,
            encode_feature=adapter.encode_feature,
            decode_context=adapter.decode_context,
            new_cache=adapter.new_cache,
        )
        feature = adapter.encode_feature(
            rebuilt.seed_raw_features,
            trace_callback=record_tap if tap_output is not None else None,
        )
        token = rebuilt.seed_token
        cache = rebuilt.cache
        norm = adapter.drafter.norm
        for depth, native_row in enumerate(round_heads):
            result = adapter.decode_step(
                token,
                feature,
                rebuilt.decoder_position + depth,
                cache,
                trace_callback=record_tap if tap_output is not None and depth == 0 else None,
            )
            if tap_output is not None and depth == 0:
                record_tap("draft_logits", result.logits)
            state = result.pre_norm
            normalized = (
                state * torch.rsqrt(state.square().mean() + norm.variance_epsilon) * norm.weight
            )
            predicted = normalized.detach().numpy()
            original = np.asarray(native_values[native_row["state_row"]])
            delta = predicted.astype(np.float64) - original.astype(np.float64)
            predicted_draft_id = int(torch.argmax(result.logits))
            predicted_target_id = int(mapping[predicted_draft_id])
            depth_rows.append(
                {
                    "depth": depth,
                    "native_state_row": native_row["state_row"],
                    "max_abs_state_difference": float(np.max(np.abs(delta))),
                    "mean_abs_state_difference": float(np.mean(np.abs(delta))),
                    "rms_state_difference": float(np.sqrt(np.mean(delta * delta))),
                    "native_state_rms": float(np.sqrt(np.mean(original.astype(np.float64) ** 2))),
                    "predicted_draft_id": predicted_draft_id,
                    "predicted_target_id": predicted_target_id,
                    "native_proposed_target_id": proposed[depth],
                    "top1_target_id_matches_native": predicted_target_id == proposed[depth],
                }
            )
            token = proposed[depth]
            feature = result.pre_norm
            cache = result.cache
    first_depth = depth_rows[0]
    if tap_output is not None:
        if tap_output.exists():
            raise ValueError("adapter tap output must be new")
        tap_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tap_output, **taps)
    return {
        "schema": "recurrent_real_step_cpu_diagnostic_v1",
        "execution_device": "cpu",
        "arithmetic": arithmetic,
        "attention_mode": attention_mode,
        "attention_backward": (
            "f32_attention_surrogate" if oracle is not None else "f32_attention"
        ),
        "status": "numeric_drift_measured_parity_unproven",
        "round_index": round_index,
        "prefix_tokens": len(prefix),
        "retained_feature_indices": feature_indices,
        "cache_positions": result.cache.key.shape[1],
        "first_depth": first_depth,
        "per_depth": depth_rows,
        "all_top1_target_ids_match_native": all(
            row["top1_target_id_matches_native"] for row in depth_rows
        ),
        **(
            {
                "adapter_taps": {
                    "path": str(tap_output),
                    "sha256": sha256(tap_output),
                    "names": sorted(taps),
                }
            }
            if tap_output is not None
            else {}
        ),
        "elapsed_seconds": time.monotonic() - start,
        "source_sha256": {
            "target_gguf": sha256(target_gguf),
            "draft_gguf": sha256(draft_gguf),
            "drafter_config": sha256(drafter_config),
            "rounds": sha256(capture_dir / "forced-rounds.jsonl"),
            "heads": sha256(capture_dir / "heads.jsonl"),
            "head_states": sha256(native_path),
            "feature_events": sha256(capture_dir / "heads.target_features.jsonl"),
            "feature_values": sha256(feature_path),
            **(
                {"native_attention_helper": sha256(native_attention_helper)}
                if native_attention_helper is not None
                else {}
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--draft-gguf", type=Path, required=True)
    parser.add_argument("--drafter-config", type=Path, required=True)
    parser.add_argument(
        "--arithmetic", choices=("native_order", "group_matmul"), default="group_matmul"
    )
    parser.add_argument("--tap-output", type=Path)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument(
        "--attention-mode", choices=("f32", "native_forward_f32_backward"), default="f32"
    )
    parser.add_argument("--native-attention-helper", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = check_round(
        args.capture_dir,
        args.target_gguf,
        args.draft_gguf,
        args.drafter_config,
        args.arithmetic,
        args.tap_output,
        args.round_index,
        args.attention_mode,
        args.native_attention_helper,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "first_max_abs_state_difference": report["first_depth"]["max_abs_state_difference"],
                "all_top1_target_ids_match_native": report["all_top1_target_ids_match_native"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
