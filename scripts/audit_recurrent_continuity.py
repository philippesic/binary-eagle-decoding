#!/usr/bin/env python3
"""Prove internal accepted-prefix continuity across native EAGLE rounds on CPU.

This checks canonical proposal rounds against raw target-feature inputs and
their post-verification dispositions. It does not prove initial target
sampling, terminal EOS/emission, request completeness or numeric feature
values; those remain separate gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from prepare_recurrent_native_features import RAW_SCHEMA  # noqa: E402


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def audit_internal_continuity(
    rounds_path: Path, feature_events_path: Path, task_map_path: Path
) -> dict:
    """Fail on an unexplained proposal gap or feature/acceptance mismatch."""
    task_map = json.loads(task_map_path.read_text())
    if not isinstance(task_map, dict) or not task_map:
        raise ValueError("missing task ownership map")
    rounds = read_jsonl(rounds_path)
    events = read_jsonl(feature_events_path)
    decoded, dispositions = {}, {}
    for event in events:
        if event.get("schema") != RAW_SCHEMA:
            raise ValueError("unknown target-feature event schema")
        row = _integer(event.get("feature_row"), "feature_row")
        if event.get("event") == "decoded_row":
            if row in decoded:
                raise ValueError("duplicate decoded target-feature row")
            decoded[row] = event
        elif event.get("event") == "disposition":
            if row in dispositions:
                raise ValueError("duplicate target-feature disposition")
            dispositions[row] = event
        else:
            raise ValueError("unknown target-feature event")
    if set(decoded) != set(dispositions) or set(decoded) != set(range(len(decoded))):
        raise ValueError("target-feature rows or dispositions are missing")
    by_task_round = defaultdict(list)
    prefill = defaultdict(list)
    target_only = defaultdict(list)
    slots = defaultdict(set)
    for row_index, row in decoded.items():
        task = str(row.get("task_id"))
        if task not in task_map or dispositions[row_index].get("task_id") != row.get("task_id"):
            raise ValueError("feature event task is unmapped or mismatched")
        slot = _integer(row.get("slot_id"), "slot_id")
        if dispositions[row_index].get("slot_id") != slot:
            raise ValueError("feature disposition slot differs from decoded row")
        slots[task].add(slot)
        phase = row.get("phase")
        if phase == "speculative":
            key = task, _integer(row.get("round_index"), "feature round_index")
            by_task_round[key].append((row_index, row, dispositions[row_index]))
        elif phase == "prefill":
            prefill[task].append((row_index, row, dispositions[row_index]))
        elif phase == "target_only":
            target_only[task].append((row_index, row, dispositions[row_index]))
        else:
            raise ValueError("replay or unknown feature phase is outside the strict chain")
    grouped = defaultdict(list)
    for record in rounds:
        if record.get("schema") != "eagle_forced_round_v1":
            raise ValueError("unknown native round schema")
        task = str(record.get("task_id"))
        if task not in task_map:
            raise ValueError("native round task is unmapped")
        grouped[task].append(record)
    for task, group in grouped.items():
        if [record.get("round_index") for record in group] != list(range(len(group))):
            raise ValueError(f"task {task} proposal round indexes have an unexplained gap")
    if set(grouped) != set(task_map) or set(by_task_round) != {
        (task, row["round_index"]) for task, group in grouped.items() for row in group
    }:
        raise ValueError("rounds and speculative feature groups differ")
    checked_inputs = checked_prefill = rejected = 0
    for task, group in grouped.items():
        if not group:
            raise ValueError("task has no proposal round")
        if len(slots[task]) != 1:
            raise ValueError("first capture contract requires one slot per task")
        first_prefix = group[0].get("prefix_token_ids")
        if not isinstance(first_prefix, list) or not first_prefix:
            raise ValueError("first round has no complete prefill prefix")
        prefill_rows = sorted(prefill[task], key=lambda item: item[1].get("position", -1))
        if len(prefill_rows) != len(first_prefix):
            raise ValueError("first round prefix lacks complete captured prefill")
        previous_prefill_ordinal = -1
        for position, (_, row, status) in enumerate(prefill_rows):
            if (
                row.get("position") != position
                or row.get("prefix_token_ids") != first_prefix[: position + 1]
                or row.get("token_id") != first_prefix[position]
                or _integer(row.get("decode_ordinal"), "prefill decode_ordinal")
                < previous_prefill_ordinal
                or status.get("retained_input") is not True
                or status.get("reason") != "accepted_prefix"
            ):
                raise ValueError("prefill target feature ancestry is incomplete")
            previous_prefill_ordinal = row["decode_ordinal"]
        checked_prefill += len(prefill_rows)
        previous = None
        previous_ordinal = max(item[1]["decode_ordinal"] for item in prefill_rows)
        for record in group:
            prefix = record.get("prefix_token_ids")
            seed = _integer(record.get("seed_token_id"), "seed_token_id")
            proposed = record.get("draft_token_ids")
            accepted = _integer(record.get("accepted_drafts"), "accepted_drafts")
            verifier = record.get("verifier_token_ids")
            if (
                not isinstance(prefix, list)
                or not prefix
                or record.get("pos0") != len(prefix)
                or not isinstance(proposed, list)
                or not proposed
                or not isinstance(verifier, list)
                or accepted > len(proposed)
                or len(verifier) != accepted + 1
                or verifier[:accepted] != proposed[:accepted]
            ):
                raise ValueError("native round proposal/acceptance record is invalid")
            if previous is not None:
                old_prefix, old_seed, old_verifier = previous
                if (
                    prefix != old_prefix + [old_seed] + old_verifier[:-1]
                    or seed != old_verifier[-1]
                ):
                    raise ValueError("consecutive proposal rounds break accepted-prefix continuity")
            entries = sorted(
                by_task_round[(task, record["round_index"])],
                key=lambda item: item[1].get("spec_input_row", -1),
            )
            if len(entries) != len(proposed) + 1:
                raise ValueError("round has missing or extra speculative target inputs")
            ordinals = {row.get("decode_ordinal") for _, row, _ in entries}
            if len(ordinals) != 1 or next(iter(ordinals)) <= previous_ordinal:
                raise ValueError("speculative decode ordinal is missing or out of order")
            previous_ordinal = next(iter(ordinals))
            for j, (_, row, status) in enumerate(entries):
                expected_prefix = prefix + [seed] + proposed[:j]
                retained = j <= accepted
                if (
                    row.get("spec_input_row") != j
                    or row.get("position") != len(prefix) + j
                    or row.get("prefix_token_ids") != expected_prefix
                    or row.get("token_id") != expected_prefix[-1]
                    or status.get("round_index") != record["round_index"]
                    or status.get("accepted_drafts") != accepted
                    or status.get("retained_input") is not retained
                    or status.get("reason")
                    != ("accepted_prefix" if retained else "rejected_suffix")
                ):
                    raise ValueError("speculative feature row disagrees with canonical round")
                checked_inputs += 1
                rejected += not retained
            previous = prefix, seed, verifier
        extras = target_only[task]
        if extras:
            if len(extras) != 1:
                raise ValueError(
                    "intervening or multiple target-only rows lack sampling provenance"
                )
            _, row, status = extras[0]
            old_prefix, old_seed, old_verifier = previous
            expected = old_prefix + [old_seed] + old_verifier
            if (
                row.get("prefix_token_ids") != expected
                or row.get("position") != len(expected) - 1
                or row.get("token_id") != expected[-1]
                or _integer(row.get("decode_ordinal"), "terminal decode_ordinal")
                <= previous_ordinal
                or status.get("retained_input") is not True
                or status.get("reason") != "accepted_prefix"
            ):
                raise ValueError("terminal target-only input lacks accepted-prefix ancestry")
    return {
        "schema": "recurrent_internal_continuity_v1",
        "status": "internal_accepted_prefix_continuity_verified",
        "execution_device": "cpu",
        "rounds": len(rounds),
        "prefill_inputs": checked_prefill,
        "speculative_inputs": checked_inputs,
        "rejected_suffix_inputs": rejected,
        "source_sha256": {
            "rounds": sha256(rounds_path),
            "feature_events": sha256(feature_events_path),
            "task_map": sha256(task_map_path),
        },
        "initial_target_sample": "unverified",
        "terminal_emission_and_stop": "unverified",
        "request_completeness": "unverified",
        "numeric_feature_values": "unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--feature-events", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = audit_internal_continuity(args.rounds, args.feature_events, args.task_map)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rounds": result["rounds"], "status": result["status"]}))


if __name__ == "__main__":
    main()
