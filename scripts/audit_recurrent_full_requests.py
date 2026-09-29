#!/usr/bin/env python3
"""Audit every frozen-training response against its native EAGLE emissions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_recurrent_binary_capture import sha256
from audit_recurrent_response import audit_response
from run_binary_head_capture import TRAIN_PROMPTS_SHA256


def audit_full_requests(
    capture_root: Path,
    *,
    expected_prompt_sha256: str = TRAIN_PROMPTS_SHA256,
    expected_prompt_count: int = 96,
) -> dict:
    capture_root = Path(capture_root)
    if (
        len(expected_prompt_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_prompt_sha256)
        or type(expected_prompt_count) is not int
        or expected_prompt_count < 1
    ):
        raise ValueError("expected prompt SHA256/count must be explicit valid values")
    cell = capture_root / "d_d"
    capture_manifest_path = capture_root / "capture-manifest.json"
    cell_manifest_path = cell / "manifest.json"
    task_map_path = capture_root / "task_prompt_ids.json"
    capture = json.loads(capture_manifest_path.read_text())
    manifest = json.loads(cell_manifest_path.read_text())
    task_map = json.loads(task_map_path.read_text())
    requests = manifest.get("requests")
    if (
        capture.get("schema") != "recurrent_binary_native_capture_v1"
        or capture.get("train_prompts_sha256") != expected_prompt_sha256
        or capture.get("train_prompt_count") != expected_prompt_count
        or capture.get("complete") is not True
        or manifest.get("complete") is not True
        or manifest.get("prompts_sha256") != expected_prompt_sha256
        or manifest.get("prompt_count") != expected_prompt_count
        or not isinstance(requests, list)
        or len(requests) != expected_prompt_count
        or not isinstance(task_map, dict)
        or len(task_map) != expected_prompt_count
    ):
        raise ValueError("full recurrent capture is incomplete or not the frozen training split")
    if set(task_map) != {item.get("task_id") for item in requests if isinstance(item, dict)}:
        raise ValueError("full recurrent request/task ownership is incomplete")

    rounds = cell / "forced-rounds.jsonl"
    traces = cell / "rounds.jsonl"
    features = cell / "heads.target_features.jsonl"
    totals: Counter[str] = Counter()
    results = []
    for index, item in enumerate(requests):
        if not isinstance(item, dict) or task_map.get(item.get("task_id")) != item.get("id"):
            raise ValueError("full recurrent request does not own its frozen prompt")
        request_dir = cell / f"request-{index:03d}"
        for filename, field in (
            ("prompt.json", "prompt_sha256"),
            ("request.json", "request_sha256"),
            ("response.json", "response_sha256"),
        ):
            if sha256(request_dir / filename) != item.get(field):
                raise ValueError(f"request {index} {filename} differs from cell manifest")
        report = audit_response(
            rounds,
            traces,
            features,
            task_map_path,
            request_dir / "request.json",
            request_dir / "response.json",
            item["task_id"],
        )
        if (
            report["status"] != "captured_response_emissions_verified"
            or report["prompt_id"] != item["id"]
            or report["response_tokens"] != len(item.get("generated_token_ids", []))
        ):
            raise ValueError(f"request {index} failed native response audit")
        totals["response_tokens"] += report["response_tokens"]
        totals["rounds"] += report["rounds"]
        totals["accepted_drafts"] += report["accepted_drafts"]
        totals["eos_clipped_final_rounds"] += report["eos_clipped_final_rounds"]
        totals[f"finish_{report['finish_reason']}"] += 1
        results.append(
            {
                "index": index,
                "prompt_id": item["id"],
                "task_id": item["task_id"],
                "response_tokens": report["response_tokens"],
                "rounds": report["rounds"],
                "accepted_drafts": report["accepted_drafts"],
                "eos_clipped_final_rounds": report["eos_clipped_final_rounds"],
                "finish_reason": report["finish_reason"],
                "response_sha256": item["response_sha256"],
            }
        )
    return {
        "schema": "recurrent_full_request_audit_v1",
        "status": "all_frozen_training_responses_joined_to_native_emissions",
        "requests": len(results),
        "totals": dict(totals),
        "items": results,
        "source_sha256": {
            "capture_manifest": sha256(capture_manifest_path),
            "cell_manifest": sha256(cell_manifest_path),
            "task_map": sha256(task_map_path),
            "rounds": sha256(rounds),
            "round_trace": sha256(traces),
            "feature_events": sha256(features),
        },
        "initial_seed_sampler_parity": "unverified",
        "terminal_sampler_parity": "unverified_where_observed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256", default=TRAIN_PROMPTS_SHA256)
    parser.add_argument("--expected-prompt-count", type=int, default=96)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = audit_full_requests(
        args.capture_root,
        expected_prompt_sha256=args.expected_prompt_sha256,
        expected_prompt_count=args.expected_prompt_count,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"requests": result["requests"], "totals": result["totals"]}))


if __name__ == "__main__":
    main()
