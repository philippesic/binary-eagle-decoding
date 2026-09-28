#!/usr/bin/env python3
"""Join one native request's raw response to canonical EAGLE round emissions.

This proves a captured response agrees with the recorded seed and verifier
emissions. It cannot independently validate the first or terminal target
sample, target feature values, or the model's sampling arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_recurrent_binary_capture import read_jsonl, sha256
from audit_recurrent_continuity import audit_internal_continuity

PINNED_TARGET_EOS_TOKEN_ID = 151_645


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{label} must contain nonnegative token IDs")
    return value


def audit_response(
    rounds_path: Path,
    round_trace_path: Path,
    feature_events_path: Path,
    task_map_path: Path,
    request_path: Path,
    response_path: Path,
    task_id: str,
) -> dict:
    """Fail on a response/round gap for one single-slot native request."""
    continuity = audit_internal_continuity(rounds_path, feature_events_path, task_map_path)
    task_map = json.loads(task_map_path.read_text())
    if task_id not in task_map:
        raise ValueError("task has no frozen prompt ownership")
    canonical = [row for row in read_jsonl(rounds_path) if str(row.get("task_id")) == task_id]
    trace = [row for row in read_jsonl(round_trace_path) if str(row.get("task_id")) == task_id]
    if not canonical or len(trace) not in (len(canonical), len(canonical) + 1):
        raise ValueError("canonical rounds and native timing trace differ")
    request = json.loads(request_path.read_text())
    response = json.loads(response_path.read_text())
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError("request and response must be JSON objects")
    cap = request.get("max_tokens")
    if type(cap) is not int or cap < 1 or cap > 128:
        raise ValueError("request has no bounded output token cap")
    verbose = response.get("__verbose")
    usage = response.get("usage")
    choices = response.get("choices")
    if (
        not isinstance(verbose, dict)
        or not isinstance(usage, dict)
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise ValueError("response lacks one raw-token choice and usage")
    output = _tokens(verbose.get("tokens"), "raw response")
    if not output or len(output) > cap:
        raise ValueError("response output length disagrees with request cap")
    prompt_tokens = len(canonical[0]["prefix_token_ids"])
    if (
        usage.get("completion_tokens") != len(output)
        or usage.get("prompt_tokens") != prompt_tokens
        or verbose.get("tokens_predicted") != len(output)
        or verbose.get("tokens_evaluated") != prompt_tokens
        or verbose.get("truncated") is not False
    ):
        raise ValueError("response token counts or prompt prefix disagree")
    reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    if reason not in {"length", "stop"} or (reason == "length" and len(output) != cap):
        raise ValueError("response stop reason disagrees with output cap")

    emitted = [canonical[0]["seed_token_id"]]
    accepted_total = 0
    eos_clipped_final_rounds = 0
    for index, (record, live) in enumerate(zip(canonical, trace[: len(canonical)])):
        verifier = _tokens(record.get("verifier_token_ids"), "verifier emission")
        proposed = _tokens(record.get("draft_token_ids"), "draft proposal")
        live_emitted = _tokens(live.get("emitted_token_ids"), "live verifier emission")
        eos_clip = (
            index == len(canonical) - 1
            and len(trace) == len(canonical)
            and reason == "stop"
            and bool(live_emitted)
            and live_emitted[-1] == PINNED_TARGET_EOS_TOKEN_ID
            and len(live_emitted) < len(verifier)
            and verifier[: len(live_emitted)] == live_emitted
        )
        if (
            live.get("schema") != "w1ax_eagle_round_v1"
            or live.get("status") != "complete"
            or live.get("replay") is not False
            or live.get("round_index") != record.get("round_index")
            or live.get("round_index") != index
            or live.get("n_proposed") != len(proposed)
            or live.get("n_accepted") != record.get("accepted_drafts")
            or live.get("n_emitted") != len(live_emitted)
            or live.get("proposed_token_ids") != proposed
            or (live_emitted != verifier and not eos_clip)
        ):
            raise ValueError("native round trace disagrees with canonical verifier emission")
        emitted.extend(live_emitted)
        accepted_total += record["accepted_drafts"]
        eos_clipped_final_rounds += int(eos_clip)
    terminal = trace[len(canonical) :]
    feature_rows = [
        row
        for row in read_jsonl(feature_events_path)
        if row.get("event") == "decoded_row"
        and row.get("phase") == "target_only"
        and str(row.get("task_id")) == task_id
    ]
    if terminal:
        last = terminal[0]
        token = _tokens(last.get("emitted_token_ids"), "terminal emission")
        if (
            last.get("schema") != "w1ax_eagle_round_v1"
            or last.get("status") != "no_proposal"
            or last.get("replay") is not False
            or last.get("round_index") != len(canonical)
            or last.get("n_proposed") != 0
            or last.get("n_accepted") != 0
            or last.get("proposed_token_ids") != []
            or last.get("n_emitted") != 1
            or len(token) != 1
            or len(feature_rows) != 1
        ):
            raise ValueError("terminal no-proposal trace is incomplete")
        emitted.extend(token)
    elif feature_rows:
        raise ValueError("target-only input has no terminal trace emission")
    if output != emitted:
        raise ValueError("raw response differs from native seed, round or terminal emissions")
    return {
        "schema": "recurrent_response_round_join_v1",
        "status": "captured_response_emissions_verified",
        "execution_device": "cpu",
        "task_id": task_id,
        "prompt_id": task_map[task_id],
        "rounds": len(canonical),
        "accepted_drafts": accepted_total,
        "eos_clipped_final_rounds": eos_clipped_final_rounds,
        "response_tokens": len(output),
        "round_emission_prefix_tokens": len(emitted) - len(terminal),
        "terminal_trace_tokens": len(terminal),
        "finish_reason": reason,
        "initial_seed_sampler_parity": "unverified",
        "terminal_sampler_parity": "unverified" if terminal else "not_observed",
        "source_sha256": {
            "rounds": sha256(rounds_path),
            "round_trace": sha256(round_trace_path),
            "feature_events": sha256(feature_events_path),
            "task_map": sha256(task_map_path),
            "request": sha256(request_path),
            "response": sha256(response_path),
        },
        "continuity_status": continuity["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--round-trace", type=Path, required=True)
    parser.add_argument("--feature-events", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    report = audit_response(
        args.rounds,
        args.round_trace,
        args.feature_events,
        args.task_map,
        args.request,
        args.response,
        args.task_id,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "response_tokens": report["response_tokens"]}))


if __name__ == "__main__":
    main()
