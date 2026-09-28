#!/usr/bin/env python3
"""Audit first-round and later-round CPU D proposal parity across captures.

This joins every per-round adapter report back to canonical native proposals,
source hashes, frozen training prompt IDs, continuity and response emissions.
It measures one bounded diagnostic sample; it cannot establish population
acceptance, trained quality or accelerator throughput.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_recurrent_binary_capture import read_jsonl, sha256
from audit_recurrent_continuity import audit_internal_continuity
from audit_recurrent_response import audit_response
from run_binary_head_capture import DRAFT_D_SHA256, TARGET_F16_SHA256, TRAIN_PROMPTS_SHA256


def summarize(capture_dirs: list[Path], frozen_train_prompts: Path) -> dict:
    if not capture_dirs or sha256(frozen_train_prompts) != TRAIN_PROMPTS_SHA256:
        raise ValueError("summary requires the frozen 96-prompt training split")
    frozen = {row["id"]: row for row in read_jsonl(frozen_train_prompts)}
    if len(frozen) != 96 or len(set(frozen)) != 96:
        raise ValueError("frozen training prompt IDs are incomplete")
    per_prompt = {}
    total_rounds = total_rows = total_matches = 0
    for directory in capture_dirs:
        directory = Path(directory)
        prompt = json.loads((directory / "prompt.json").read_text())
        prompt_id = prompt.get("id")
        if prompt_id not in frozen or prompt != frozen[prompt_id] or prompt_id in per_prompt:
            raise ValueError("capture prompt is not a unique frozen training prompt")
        task_map_path = directory / "task_prompt_ids.json"
        task_map = json.loads(task_map_path.read_text())
        if (
            not isinstance(task_map, dict)
            or len(task_map) != 1
            or list(task_map.values()) != [prompt_id]
        ):
            raise ValueError("capture task ownership differs from frozen prompt")
        task_id = next(iter(task_map))
        rounds_path = directory / "forced-rounds.jsonl"
        events_path = directory / "heads.target_features.jsonl"
        heads_path = directory / "heads.jsonl"
        head_states_path = directory / "heads.f32"
        feature_values_path = directory / "heads.target_features.f32"
        continuity = audit_internal_continuity(rounds_path, events_path, task_map_path)
        response = audit_response(
            rounds_path,
            directory / "rounds.jsonl",
            events_path,
            task_map_path,
            directory / "request.json",
            directory / "response.json",
            task_id,
        )
        rounds = read_jsonl(rounds_path)
        expected_hashes = {
            "target_gguf": TARGET_F16_SHA256,
            "draft_gguf": DRAFT_D_SHA256,
            "rounds": sha256(rounds_path),
            "heads": sha256(heads_path),
            "head_states": sha256(head_states_path),
            "feature_events": sha256(events_path),
            "feature_values": sha256(feature_values_path),
        }
        reports = []
        max_abs = max_rms = 0.0
        matches = rows_count = 0
        for index, native in enumerate(rounds):
            if native.get("round_index") != index or str(native.get("task_id")) != task_id:
                raise ValueError("canonical native round order or task changed")
            path = directory / f"real_round_{index:02d}_group_matmul.json"
            report = json.loads(path.read_text())
            source = report.get("source_sha256")
            if (
                report.get("schema") != "recurrent_real_step_cpu_diagnostic_v1"
                or report.get("status") != "numeric_drift_measured_parity_unproven"
                or report.get("execution_device") != "cpu"
                or report.get("arithmetic") != "group_matmul"
                or report.get("round_index") != index
                or report.get("prefix_tokens") != len(native["prefix_token_ids"])
                or not isinstance(source, dict)
                or any(source.get(key) != value for key, value in expected_hashes.items())
            ):
                raise ValueError("round report disagrees with canonical CPU capture sources")
            depths = report.get("per_depth")
            proposed = native["draft_token_ids"]
            if not isinstance(depths, list) or len(depths) != len(proposed):
                raise ValueError("CPU replay omitted a native proposal depth")
            round_matches = 0
            for depth, (row, token) in enumerate(zip(depths, proposed)):
                if (
                    row.get("depth") != depth
                    or row.get("native_proposed_target_id") != token
                    or type(row.get("predicted_target_id")) is not int
                    or row.get("top1_target_id_matches_native")
                    is not (row["predicted_target_id"] == token)
                ):
                    raise ValueError("CPU replay proposal ID or depth disagrees with native round")
                round_matches += row["predicted_target_id"] == token
                max_abs = max(max_abs, row["max_abs_state_difference"])
                max_rms = max(max_rms, row["rms_state_difference"])
            matches += round_matches
            rows_count += len(depths)
            reports.append(
                {
                    "round_index": index,
                    "report_sha256": sha256(path),
                    "proposal_rows": len(depths),
                    "mapped_top_matches": round_matches,
                }
            )
        per_prompt[prompt_id] = {
            "category": prompt["category"],
            "rounds": len(rounds),
            "proposal_rows": rows_count,
            "mapped_top_matches": matches,
            "accepted_drafts_by_round": [row["accepted_drafts"] for row in rounds],
            "max_abs_state_difference": max_abs,
            "max_rms_state_difference": max_rms,
            "response_tokens": response["response_tokens"],
            "continuity_status": continuity["status"],
            "response_status": response["status"],
            "source_sha256": {
                "rounds": expected_hashes["rounds"],
                "feature_events": expected_hashes["feature_events"],
                "head_states": expected_hashes["head_states"],
                "response": sha256(directory / "response.json"),
            },
            "reports": reports,
        }
        total_rounds += len(rounds)
        total_rows += rows_count
        total_matches += matches
    return {
        "schema": "recurrent_cpu_d_broader_proposal_parity_v1",
        "execution_device": "cpu",
        "training_eligible": False,
        "split": "train_diagnostic_subset",
        "frozen_train_prompts_sha256": TRAIN_PROMPTS_SHA256,
        "prompt_count": len(per_prompt),
        "rounds": total_rounds,
        "proposal_rows": total_rows,
        "mapped_top_matches": total_matches,
        "all_mapped_top_ids_match_native": total_matches == total_rows,
        "per_prompt": per_prompt,
        "unverified": [
            "full_drafter_numeric_and_kv_parity",
            "96_prompt_training_capture_and_loss_choice",
            "trained_model_quality_and_native_Q4_0_throughput",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--frozen-train-prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = summarize(args.capture_dir, args.frozen_train_prompts)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("prompt_count", "rounds", "proposal_rows", "mapped_top_matches")
            }
        )
    )


if __name__ == "__main__":
    main()
