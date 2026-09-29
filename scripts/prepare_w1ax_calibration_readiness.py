#!/usr/bin/env python3
"""Assemble the frozen W1Ax 100-step readiness report from audited evidence.

This CPU-only tool summarizes completed identity, numeric, gradient, response,
provider, and native cache audits. It never edits the original preparation
manifest or infers a passing result from a missing artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from audit_recurrent_binary_capture import (  # noqa: E402
    CALIBRATION_ONLY_SCOPE,
    CALIBRATION_PINNED_INPUT_SHA256,
    CALIBRATION_READINESS_SCHEMA,
    load_audited_capture,
    validate_calibration_readiness_report,
    validate_recurrent_trace,
)
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from run_binary_rescue_benchmark import CLEAR_PREFIXES  # noqa: E402
from run_binary_rescue_benchmark import command as runner_command  # noqa: E402
from run_binary_rescue_benchmark import server_env as runner_server_env  # noqa: E402
from run_binary_rescue_benchmark import validate_config as validate_runner_config  # noqa: E402
from w1ax_capture_provider import TARGET_GGUF_SHA256, sha256  # noqa: E402

EXPECTED_ALIASES = {
    "pilot-prose": "dolly:line-005896",
    "pilot-reasoning": "gsm8k:train-000315",
    "pilot-code": "mbpp:task-496",
}
EXPECTED_DOMAINS = {
    "prose": "dolly:line-005896",
    "reasoning": "gsm8k:train-000315",
    "code": "mbpp:task-496",
}
IDENTITY_REPORT_SHA256 = "2317a1fafa8ea0afe5316dc3a674ccc864a088d1392e4a1da248a084c47b32ae"
NUMERIC_REPORT_SHA256 = "ed04cd752e01e87c71a18990fec04bcc45f6a9d9afb88c70c21ca5a3a9586a81"
BUNDLE_AUDIT_SHA256 = "832325813eefea67dc97dc0d251b1e37b3a7b4a7349a4e26eb3aa9663198508b"
DIAGNOSTIC_SOURCE_PROMPTS_SHA256 = (
    "b3f3570cf2e45c570678b98f135257a609de319b01232ce4353e4286eb595703"
)
DIAGNOSTIC_SAFE_PROMPTS_SHA256 = "93f61ae9160bbb59739ca81efa002e10cd5bddbddcf3d02ae2b33f9ac117f992"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in result):
        raise ValueError(f"expected JSON objects in {path}")
    return result


def _alias_map(path: Path) -> dict[str, str]:
    raw = _json(path)
    if "mapping" in raw:
        if (
            raw.get("source_sha256") != DIAGNOSTIC_SOURCE_PROMPTS_SHA256
            or raw.get("safe_sha256") != DIAGNOSTIC_SAFE_PROMPTS_SHA256
            or not isinstance(raw["mapping"], list)
        ):
            raise ValueError("frozen alias map source/safe prompt hashes differ")
        rows = {}
        for item in raw["mapping"]:
            if not isinstance(item, dict):
                raise ValueError("frozen alias map contains a malformed row")
            safe_id = item.get("safe_id")
            source_id = item.get("source_id")
            message_hash = item.get("messages_sha256")
            if (
                type(safe_id) is not str
                or type(source_id) is not str
                or not isinstance(message_hash, str)
                or len(message_hash) != 64
                or any(character not in "0123456789abcdef" for character in message_hash)
                or safe_id in rows
            ):
                raise ValueError("frozen alias map contains invalid or duplicate IDs")
            rows[safe_id] = source_id
        if rows != EXPECTED_ALIASES:
            raise ValueError("frozen diagnostic safe IDs do not map to the selected source prompts")
        return rows
    candidates = [raw.get("alias_to_source"), raw.get("aliases"), raw.get("mapping"), raw]
    for candidate in candidates:
        if isinstance(candidate, list):
            rows = {}
            for item in candidate:
                if not isinstance(item, dict):
                    continue
                alias = item.get("alias", item.get("alias_id"))
                source = item.get("source_id", item.get("original_id"))
                if isinstance(alias, str) and isinstance(source, str):
                    rows[alias] = source
            if rows == EXPECTED_ALIASES:
                return rows
    for candidate in candidates:
        if candidate == EXPECTED_ALIASES:
            return dict(candidate)
        if (
            isinstance(candidate, dict)
            and {source: alias for alias, source in EXPECTED_ALIASES.items()} == candidate
        ):
            return dict(EXPECTED_ALIASES)
    raise ValueError("frozen diagnostic alias map differs from the three source prompts")


def _task_map(manifest: dict, aliases: dict[str, str]) -> dict[int, str]:
    mapping = {}
    for row in _read_benchmark_records(manifest):
        if row.get("variant") != "row_a16_checkpoint_zero" or row.get("warmup") is True:
            continue
        task = (row.get("request_digest") or {}).get("task_id")
        prompt = row.get("prompt_id")
        if type(task) is not int or type(prompt) is not str:
            raise ValueError("measured row-A16 record lacks task-to-prompt ownership")
        prior = mapping.get(task)
        if prior is not None and prior != prompt:
            raise ValueError("native task ID is reused for different prompt aliases")
        mapping[task] = prompt
    if not mapping:
        raise ValueError("native run manifest lacks its task-to-prompt map")
    normalized: dict[int, str] = {}
    for task, prompt in mapping.items():
        try:
            task_id = int(task)
        except (TypeError, ValueError) as error:
            raise ValueError("native task map has a non-integer task ID") from error
        if type(prompt) is not str:
            raise ValueError("native task map has a non-string prompt ID")
        normalized[task_id] = aliases.get(prompt, prompt)
    if set(normalized.values()) != set(EXPECTED_DOMAINS.values()) or len(normalized) != 3:
        raise ValueError("native task map does not identify each frozen prompt exactly once")
    return normalized


def _measured_row_a16_task_ids(manifest: dict, aliases: dict[str, str]) -> set[int]:
    """Return task IDs for the three measured row-A16 prompts, excluding warmups."""
    prompt_to_source = {alias: source for alias, source in aliases.items()}
    task_to_prompt: dict[int, str] = {}
    for row in _read_benchmark_records(manifest):
        if row.get("variant") != "row_a16_checkpoint_zero" or row.get("warmup") is True:
            continue
        task = (row.get("request_digest") or {}).get("task_id")
        prompt = row.get("prompt_id")
        if type(task) is not int or type(prompt) is not str or prompt not in prompt_to_source:
            raise ValueError("measured row-A16 record lacks frozen task-to-prompt ownership")
        if task in task_to_prompt:
            raise ValueError("measured row-A16 task ID is duplicated")
        task_to_prompt[task] = prompt
    if len(task_to_prompt) != 3 or set(task_to_prompt.values()) != set(prompt_to_source):
        raise ValueError("measured row-A16 tasks do not identify each frozen prompt exactly once")
    return set(task_to_prompt)


def _read_benchmark_records(manifest: dict) -> list[dict]:
    records = manifest.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("benchmark manifest has no measured request records")
    return records


def _exact_response_pairs(manifest: dict) -> list[dict]:
    records = [row for row in _read_benchmark_records(manifest) if row.get("warmup") is not True]
    variants = {"q4_0", "row_a16_checkpoint_zero"}
    grouped: dict[tuple[str, str], list[list[int]]] = {}
    for row in records:
        prompt, variant, tokens = (
            row.get("prompt_id"),
            row.get("variant"),
            row.get("generated_token_ids"),
        )
        if prompt not in EXPECTED_ALIASES or variant not in variants:
            continue
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("benchmark response is missing exact generated token IDs")
        grouped.setdefault((prompt, variant), []).append(tokens)
    pairs = []
    for alias in EXPECTED_ALIASES:
        q4, row = (
            grouped.get((alias, "q4_0"), []),
            grouped.get((alias, "row_a16_checkpoint_zero"), []),
        )
        if len(q4) != 1 or len(row) != 1 or q4[0] != row[0]:
            raise ValueError(f"Q4_0 and row-A16 response IDs differ or are incomplete for {alias}")
        pairs.append({"alias": alias, "generated_tokens": len(q4[0]), "exact_ids": True})
    return pairs


def _wrong_accepted_labels(capture_dir: Path, task_ids: set[int]) -> tuple[int, int]:
    heads = _jsonl(capture_dir / "heads.jsonl")
    rounds = _jsonl(capture_dir / "rounds.jsonl")
    by_key: dict[tuple[int, int, int], list[dict]] = {}
    for row in heads:
        by_key.setdefault(
            (row.get("task_id"), row.get("round_index"), row.get("depth")), []
        ).append(row)
    accepted = wrong = 0
    for event in rounds:
        if event.get("task_id") not in task_ids:
            continue
        status = event.get("status")
        if status == "no_proposal":
            if event.get("n_accepted") != 0 or event.get("n_proposed") != 0:
                raise ValueError("native no-proposal round has nonzero draft counts")
            continue
        if status != "complete":
            raise ValueError("native round has an unsupported completion status")
        count = event.get("n_accepted")
        proposed = event.get("n_proposed")
        task, round_index = event.get("task_id"), event.get("round_index")
        if (
            type(count) is not int
            or count < 0
            or type(proposed) is not int
            or proposed < count
            or type(task) is not int
            or type(round_index) is not int
        ):
            raise ValueError("completed native round lacks accepted-row ownership")
        for depth in range(count):
            rows = by_key.get((task, round_index, depth), [])
            if len(rows) != 1:
                raise ValueError("accepted native draft row is missing or duplicated")
            row = rows[0]
            accepted += 1
            wrong += int(
                row.get("verifier_reached") is not True
                or row.get("proposed_token_id") != row.get("verifier_token_id")
            )
    return accepted, wrong


def _numeric_roots(report: dict) -> dict[str, list[dict]]:
    if report.get("schema") != "w1ax_pilot_torch_cuda_trajectory_v1":
        raise ValueError("unsupported Torch CUDA numeric report")
    checks = report.get("checks", {})
    if (
        report.get("status") != "measured_numeric_gate_only"
        or checks.get("all_three_domains_have_two_roots") is not True
        or checks.get("all_state_relative_rms_at_most_0_10") is not True
        or checks.get("all_logit_relative_rms_at_most_0_10") is not True
        or checks.get("proposal_disagreements") != 0
        or checks.get("changed_top_choice_with_margin_over_0_02") != 0
    ):
        raise ValueError("Torch CUDA numeric gate failed or is incomplete")
    domains = report.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(EXPECTED_DOMAINS):
        raise ValueError("Torch numeric report must cover all three frozen domains")
    roots = {}
    for domain, data in domains.items():
        rows = data.get("roots") if isinstance(data, dict) else None
        if data.get("prompt_id") != EXPECTED_DOMAINS[domain] or data.get("status") != "measured":
            raise ValueError(f"Torch numeric report has no measured {domain} prompt")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError(f"Torch numeric report has fewer than two {domain} roots")
        for row in rows:
            if (
                type(row.get("round_index")) is not int
                or type(row.get("parent_position")) is not int
                or not isinstance(row.get("prefix_token_ids"), list)
                or any(type(token) is not int for token in row["prefix_token_ids"])
                or row.get("state_relative_rms", 1) > 0.10
                or row.get("logit_relative_rms", 1) > 0.10
                or not math.isfinite(row.get("state_relative_rms", math.nan))
                or not math.isfinite(row.get("logit_relative_rms", math.nan))
                or not math.isfinite(row.get("state_max_abs_error", math.nan))
                or not math.isfinite(row.get("logit_max_abs_error", math.nan))
                or row.get("top1_target_id_matches_native") is not True
            ):
                raise ValueError(f"Torch numeric root fails in {domain}")
        roots[domain] = rows
    if sum(len(rows) for rows in roots.values()) != 9:
        raise ValueError("frozen numeric sample must contain exactly nine roots")
    return roots


def _gradient_roots(report: dict) -> dict[str, list[dict]]:
    if (
        report.get("schema") != "w1ax_pilot_gradient_contract_v1"
        or report.get("optimizer_steps") != 0
        or report.get("checks", {}).get("all_selected_roots_have_supported_hard_ce") is not True
        or report.get("checks", {}).get("all_selected_roots_have_18_finite_gradients") is not True
        or report.get("checks", {}).get("all_torch_cache_writes_are_finite_f16_exact") is not True
        or report.get("checks", {}).get("all_torch_cache_lengths_and_positions_match_trace")
        is not True
        or report.get("checks", {}).get("borrowed_embedding_norm_and_d2t_exact") is not True
    ):
        raise ValueError("gradient report is missing a required passing contract")
    result = {domain: [] for domain in EXPECTED_DOMAINS}
    domain_by_prompt = {prompt: domain for domain, prompt in EXPECTED_DOMAINS.items()}
    for root in report.get("roots", []):
        # The gradient producer records the frozen source prompt ID, while the
        # numeric trajectory report groups roots by its prose/reasoning/code label.
        domain = domain_by_prompt.get(root.get("domain"))
        if (
            domain not in result
            or type(root.get("root_index")) is not int
            or root["root_index"] < 0
            or type(root.get("round_index")) is not int
            or root.get("gradient_tensors") != 18
            or root.get("finite_gradient_tensors") != 18
            or type(root.get("supported_ce_rows")) is not int
            or root["supported_ce_rows"] < 1
        ):
            raise ValueError("gradient report has an invalid selected root")
        context = root.get("context_decoder_positions")
        proposals = root.get("proposal_decoder_positions")
        if (
            not isinstance(context, list)
            or any(type(position) is not int for position in context)
            or context != list(range(len(context)))
            or not isinstance(proposals, list)
            or any(type(position) is not int for position in proposals)
            or proposals != list(range(len(context), len(context) + len(proposals)))
            or root.get("final_cache_length") != (proposals[-1] + 1 if proposals else len(context))
            or root.get("f16_cache_writes") != len(context) + len(proposals)
        ):
            raise ValueError("gradient report has invalid contiguous Torch cache ancestry")
        result[domain].append(root)
    if any(len(result[d]) < 2 for d in result) or sum(map(len, result.values())) != 9:
        raise ValueError("gradient report must cover the same nine roots")
    for rows in result.values():
        rows.sort(key=lambda row: row["root_index"])
    identity = report.get("frozen_operand_identity", {})
    if (
        not identity.get("embedding_tokens_checked")
        or identity.get("embedding_dtype") != "f16_exact"
        or identity.get("d2t") != "candidate_and_row_absolute_maps_exact_to_native_offsets"
        or set(identity.get("norms", {}))
        != {
            "blk.0.attn_norm.weight",
            "blk.0.attn_norm_2.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        }
        or any(value != "exact_f32" for value in identity["norms"].values())
    ):
        raise ValueError("gradient report omits borrowed embedding, norm or absolute d2t identity")
    return result


def _ordered_head_cache_joins(
    heads: list[dict], cache_writes: list[dict]
) -> dict[tuple[int, int], tuple[int, int]]:
    """Uniquely join depth-zero heads to cache writes in capture order."""
    ordered = sorted(heads, key=lambda row: row["state_row"])
    candidates_by_head = []
    for head in ordered:
        state_row = head["state_row"]
        if type(state_row) is not int or state_row < 0:
            raise ValueError("depth-zero head has an invalid state row ordinal")
        position = head.get("input_position")
        token = head.get("input_token_id")
        slot = position - 1 if type(position) is int else None
        if type(position) is not int or type(token) is not int or type(slot) is not int:
            raise ValueError("depth-zero head lacks cache execution identity")
        candidates = []
        for write in cache_writes:
            if (
                write.get("position") != position - 1
                or write.get("slot") != slot
                or write.get("token_id") != token
            ):
                continue
            execution, column = write.get("execution"), write.get("column")
            if type(execution) is not int or type(column) is not int:
                raise ValueError("native cache write lacks ordered graph execution ownership")
            candidates.append((execution, column))
        if not candidates:
            raise ValueError("depth-zero head has no cache write with matching position and token")
        if len(set(candidates)) != len(candidates):
            raise ValueError("depth-zero head has duplicate cache graph execution candidates")
        candidates_by_head.append(sorted(candidates))

    # Count strictly increasing assignments, capping at two so repeated exact
    # states are accepted only when chronology gives one unambiguous mapping.
    path_counts: list[dict[tuple[int, int], int]] = []
    unique_parent: list[dict[tuple[int, int], tuple[int, int] | None]] = []
    for index, candidates in enumerate(candidates_by_head):
        counts: dict[tuple[int, int], int] = {}
        parents: dict[tuple[int, int], tuple[int, int] | None] = {}
        if index == 0:
            counts = dict.fromkeys(candidates, 1)
            parents = dict.fromkeys(candidates, None)
        else:
            previous = path_counts[-1]
            for candidate in candidates:
                predecessors = [key for key, count in previous.items() if key < candidate and count]
                count = min(2, sum(previous[key] for key in predecessors))
                if count:
                    counts[candidate] = count
                    parents[candidate] = predecessors[0] if count == 1 else None
        path_counts.append(counts)
        unique_parent.append(parents)
    finals = path_counts[-1]
    if min(2, sum(finals.values())) != 1:
        raise ValueError("depth-zero heads do not have one unique chronological cache-write join")
    current = next(key for key, count in finals.items() if count == 1)
    assignment = [current]
    for index in range(len(ordered) - 1, 0, -1):
        parent = unique_parent[index][current]
        if parent is None:
            raise ValueError("chronological cache graph join is ambiguous")
        assignment.append(parent)
        current = parent
    assignment.reverse()
    return {(head["task_id"], head["round_index"]): pair for head, pair in zip(ordered, assignment)}


def _validate_mask_placement(cache_audit: dict) -> None:
    """Accept CUDA masks or the audited host-pinned causal-mask buffer."""
    placement = str(cache_audit.get("mask_device", "")).lower()
    if placement == "cuda":
        return
    buffer_types = cache_audit.get("mask_buffer_types")
    if placement != "cuda_host" or not isinstance(buffer_types, list) or not buffer_types:
        raise ValueError("native cache audit has an unsupported causal-mask placement")
    if any(not isinstance(kind, str) for kind in buffer_types) or set(buffer_types) != {
        "CUDA_Host"
    }:
        raise ValueError("host-pinned causal-mask placement metadata is inconsistent")


def _bridge_roots(
    numeric_roots: dict[str, list[dict]],
    gradient_roots: dict[str, list[dict]],
    task_map: dict[int, str],
    capture_dir: Path,
    state_rows: list[dict],
    cache_rows: list[dict],
    graph_index_path: Path,
    graph_values_path: Path,
) -> list[dict]:
    heads = _jsonl(capture_dir / "heads.jsonl")
    state_payload = np.memmap(capture_dir / "heads.f32", dtype="<f4", mode="r")
    if (
        not heads
        or type(heads[0].get("state_dim")) is not int
        or heads[0]["state_dim"] < 1
        or state_payload.size % heads[0]["state_dim"]
    ):
        raise ValueError("native cache head state payload has invalid dimensions")
    state_width = heads[0]["state_dim"]
    state_payload = state_payload.reshape(-1, state_width)
    graph_records, graph_values, _ = _read_graph(graph_index_path, graph_values_path)
    graph_groups: dict[int, dict[str, dict]] = {}
    for record in graph_records:
        if record.get("group_kind") == "decoder":
            graph_groups.setdefault(record["group_execution"], {})[record["tensor_name"]] = record
    cache_writes = [row for row in cache_rows if row.get("event") == "row"]
    seed_events = []
    accept_events = []
    for event in state_rows:
        if event.get("schema") != "eagle_state_v1":
            raise ValueError("native state trace has an unsupported schema")
        if event.get("event") == "seed":
            seed_events.append(event)
        elif event.get("event") == "accept":
            accept_events.append(event)
        else:
            raise ValueError("native state trace has an unknown event")

    if any(
        row.get("schema") != "eagle_head_state_v1" or type(row.get("depth")) is not int
        for row in heads
    ):
        raise ValueError("native capture has malformed head rows")
    depth_zero_heads = [row for row in heads if row["depth"] == 0]
    if any(
        type(row.get("task_id")) is not int
        or type(row.get("round_index")) is not int
        or type(row.get("state_row")) is not int
        or not 0 <= row["state_row"] < len(state_payload)
        or type(row.get("slot_id")) is not int
        or type(row.get("parent_position")) is not int
        or type(row.get("input_token_id")) is not int
        for row in depth_zero_heads
    ):
        raise ValueError("native depth-zero head is missing seed-join metadata")
    depth_zero_heads.sort(key=lambda row: row["state_row"])
    if len({row["state_row"] for row in depth_zero_heads}) != len(depth_zero_heads):
        raise ValueError("native depth-zero head state rows are duplicated")
    if len(depth_zero_heads) != len(seed_events):
        raise ValueError("native depth-zero heads and state seeds are not a complete bijection")
    seed_ordinals: dict[tuple[int, int], int] = {}
    for ordinal, (head, seed) in enumerate(zip(depth_zero_heads, seed_events)):
        key = (head["task_id"], head["round_index"])
        if key in seed_ordinals:
            raise ValueError("native task and round ownership is duplicated across seed heads")
        if (
            head["slot_id"] != 0
            or type(seed.get("seq_id")) is not int
            or seed["seq_id"] != head["slot_id"]
            or seed.get("position") != head["parent_position"]
            or seed.get("token") != head["input_token_id"]
            or type(seed.get("kv_max_before")) is not int
            or type(seed.get("kv_max_after")) is not int
            or seed["kv_max_after"] != head["parent_position"] - 1
            or seed["kv_max_before"] < seed["kv_max_after"]
            or seed["kv_max_before"] > head["parent_position"]
        ):
            raise ValueError("chronological native seed or reserve-row trim differs from its head")
        seed_ordinals[key] = ordinal

    rounds = _jsonl(capture_dir / "rounds.jsonl")
    complete_rounds = [row for row in rounds if row.get("status") == "complete"]
    if len(complete_rounds) != len(accept_events):
        raise ValueError("native completed rounds and state accept events are not paired")
    paired_accepts: dict[tuple[int, int], dict] = {}
    for round_event, event in zip(complete_rounds, accept_events):
        key = (round_event.get("task_id"), round_event.get("round_index"))
        if (
            key not in seed_ordinals
            or key in paired_accepts
            or type(round_event.get("slot_id")) is not int
            or round_event["slot_id"] != 0
            or type(event.get("seq_id")) is not int
            or event["seq_id"] != round_event["slot_id"]
            or type(event.get("accepted")) is not int
            or event["accepted"] != round_event.get("n_accepted")
            or type(round_event.get("n_proposed")) is not int
            or round_event["n_proposed"] < 1
            or type(round_event.get("n_accepted")) is not int
            or not 0 <= round_event["n_accepted"] <= round_event["n_proposed"]
            or event.get("verify_rows") != round_event["n_proposed"] + 1
            or event["accepted"] >= round_event["n_proposed"] + 1
            or event.get("selected_row") != round_event["n_accepted"]
            or type(event.get("verify_pos_first")) is not int
            or type(event.get("position")) is not int
            or event["position"] != event["verify_pos_first"] + event["selected_row"]
            or not isinstance(event.get("selected_hash"), str)
            or event.get("selected_hash") != event.get("pending_hash")
        ):
            raise ValueError("chronological state accept does not match its native round")
        paired_accepts[key] = event
    if set(paired_accepts) != set(seed_ordinals):
        raise ValueError("native round and seed ownership does not cover the same depth-zero heads")
    ordered_cache_joins = _ordered_head_cache_joins(depth_zero_heads, cache_writes)
    bridges = []
    for domain, expected_prompt in EXPECTED_DOMAINS.items():
        task_ids = [task for task, prompt in task_map.items() if prompt == expected_prompt]
        if len(task_ids) != 1:
            raise ValueError(f"native task map has ambiguous {domain} ownership")
        task_id = task_ids[0]
        numeric = numeric_roots[domain]
        gradients = gradient_roots[domain]
        for index, root in enumerate(numeric):
            gradient = gradients[index]
            if gradient.get("root_index") != index:
                raise ValueError("gradient report disagrees on frozen selected-root order")
            prefix = root["prefix_token_ids"]
            matches = [
                row
                for row in heads
                if row.get("task_id") == task_id
                and row.get("schema") == "eagle_head_state_v1"
                and row.get("depth") == 0
                and row.get("parent_position") == root["parent_position"]
                and row.get("prefix_token_ids") == prefix
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"selected {domain} prefix has no unique native cache-capture head"
                )
            head = matches[0]
            round_key = (head["task_id"], head["round_index"])
            if round_key not in seed_ordinals:
                raise ValueError("selected head has no chronological state-seed ordinal")
            parent = root["parent_position"]
            context = gradient.get("context_decoder_positions")
            proposal_positions = gradient.get("proposal_decoder_positions")
            if (
                head.get("state_dim") != state_width
                or head.get("state_dtype") != "float32_native_endian"
                or head.get("state_boundary")
                != "native_output_norm_f32_before_head_operand_conversion"
                or head.get("finite") is not True
                or head.get("alignment_valid") is not True
                or head.get("input_position") != parent + 1
                or head.get("label_position") != parent + 2
                or head.get("input_token_id") != prefix[-1]
                or len(prefix) != parent + 2
                or context != list(range(parent))
                or len(context) != parent
                or proposal_positions != list(range(parent, parent + len(proposal_positions)))
                or gradient.get("final_cache_length")
                != (proposal_positions[-1] + 1 if proposal_positions else parent)
            ):
                raise ValueError(
                    "selected head, native seed and Torch cache positions do not bridge"
                )
            seed_ordinal = seed_ordinals[round_key]
            seed = seed_events[seed_ordinal]
            kv_max_before = seed.get("kv_max_before")
            kv_max_after = seed.get("kv_max_after")
            if (
                type(kv_max_before) is not int
                or type(kv_max_after) is not int
                or kv_max_after != parent - 1
                or kv_max_before < kv_max_after
                or kv_max_before > parent
            ):
                raise ValueError(
                    "native seed reserve-row trim does not expose the logical cache boundary"
                )
            if len(context) != kv_max_after + 1:
                raise ValueError("Torch cache length differs from the native post-trim cache")
            preceding = root.get("preceding_student_round_outcome")
            if head["round_index"] > 0 and preceding in {"accepted", "rejected"}:
                previous_key = (head["task_id"], head["round_index"] - 1)
                event = paired_accepts.get(previous_key)
                if event is None:
                    raise ValueError("native preceding accept event is absent")
                expected = "accepted" if event.get("accepted", 0) > 0 else "rejected"
                if expected != preceding:
                    raise ValueError("native accept event disagrees with selected-root outcome")
            writes = [
                row
                for row in cache_writes
                if row.get("schema") == "eagle_draft_cache_v1"
                and row.get("position") == parent
                and row.get("slot") == parent
                and row.get("token_id") == head["input_token_id"]
            ]
            if not writes:
                raise ValueError("selected native seed has no globally audited cache write")
            state_row = head.get("state_row")
            if type(state_row) is not int or not 0 <= state_row < len(state_payload):
                raise ValueError("selected native cache head state row is outside its payload")
            captured_state = np.asarray(state_payload[state_row])
            graph_execution, graph_column = ordered_cache_joins[round_key]
            if not any(
                write.get("execution") == graph_execution and write.get("column") == graph_column
                for write in writes
            ):
                raise ValueError("selected head does not own its chronological cache graph join")
            result_norm = graph_groups.get(graph_execution, {}).get("result_norm")
            if result_norm is None or not np.array_equal(
                _column(result_norm, graph_values, graph_column), captured_state
            ):
                raise ValueError("selected head state differs from its chronological graph result")
            bridges.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "round_index": head["round_index"],
                    "state_seed_ordinal": seed_ordinal,
                    "parent_position": parent,
                    "prefix_token_ids": prefix,
                    "kv_max_before": kv_max_before,
                    "kv_max_after": kv_max_after,
                    "logical_cache_length_before_seed": kv_max_after + 1,
                    "matching_globally_audited_cache_writes": len(writes),
                    "cache_execution": graph_execution,
                    "cache_column": graph_column,
                    "native_result_norm_state_exact": True,
                }
            )
    return bridges


def _provider_inventory(provider: dict) -> dict:
    paths, hashes = provider["paths"], provider["sha256"]
    capture_path = Path(paths["capture_manifest"])
    prompts_path = Path(paths["prompts"])
    capture = load_audited_capture(
        capture_path,
        prompts_path,
        hashes["prompts"],
        expected_prompt_count=provider["prompt_count"],
    )
    if len(capture.anchors) < 100:
        raise ValueError("preparation bundle has fewer than 100 audited rounds")
    manifest = _json(capture_path)
    offsets = np.load(capture_path.parent / manifest["offsets"]["path"], allow_pickle=False)
    supported = eligible = joins = 0
    for key in sorted(capture.anchors)[:100]:
        round_data = capture.round_inputs(*key)
        audit = validate_recurrent_trace(
            round_data.rows,
            [round_data.anchor],
            offsets=offsets,
            target_vocab_size=manifest["target_vocab_size"],
            draft_vocab_size=manifest["draft_vocab_size"],
            max_depth=manifest["max_depth"],
            allowed_prompt_ids={key[0]},
            split="train",
        )
        count = sum(audit.ce_mask)
        supported += count
        eligible += int(count > 0)
        joins += len(round_data.feature_positions)
    if eligible != 100 or supported < 1 or joins < 100:
        raise ValueError("first 100 audited provider rounds lack labels or exact-prefix joins")
    return {
        "eligible_rounds": eligible,
        "supported_labels": supported,
        "exact_prefix_joins": joins,
        "compact_teacher_attached": provider.get("teacher") is not None,
    }


def _identity(identity: dict, identity_hash: str, old_manifest: dict) -> dict:
    if (
        identity_hash != IDENTITY_REPORT_SHA256
        or identity.get("schema") != "w1ax_pilot_identity_v1"
        or identity.get("passed") is not True
        or len(identity.get("provider_hashes", {})) != 7
        or any(value is not True for value in identity["provider_hashes"].values())
        or identity.get("snapshots_verified") is not True
        or len(identity.get("export_projection_bits_scales", {})) != 9
        or any(value is not True for value in identity["export_projection_bits_scales"].values())
        or identity.get("checkpoint_sha256") != CALIBRATION_PINNED_INPUT_SHA256["checkpoint_zero"]
        or identity.get("student_gguf_sha256") != CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
        or old_manifest.get("schema") != "binary_rescue_benchmark_v1"
        or old_manifest.get("status") != "complete"
        or identity.get("server_sha256") != old_manifest["hashes"]["binary"]
    ):
        raise ValueError("frozen pilot identity report is missing, mismatched or failed")
    return {
        "provider_hashes_checked": 7,
        "model_snapshots_verified": True,
        "exported_projection_pairs_checked": 9,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "exported_gguf_sha256": identity["student_gguf_sha256"],
        "server_sha256": identity["server_sha256"],
    }


def _validate_runner_manifest(
    manifest_path: Path,
    manifest: dict,
    config_path: Path,
    capture_dir: Path,
    binary_path: Path | None = None,
) -> dict:
    """Check a completed run against run_binary_rescue_benchmark's real schema."""
    if (
        manifest.get("schema") != "binary_rescue_benchmark_v1"
        or manifest.get("status") != "complete"
        or manifest.get("prompt_sha256") != DIAGNOSTIC_SAFE_PROMPTS_SHA256
        or manifest.get("q4_variant", "q4_0") != "q4_0"
    ):
        raise ValueError("native benchmark manifest is incomplete or uses another prompt set")
    config = _json(config_path)
    saved_config = _json(manifest_path.parent / "config.json")
    if saved_config != config:
        raise ValueError("saved benchmark config differs from the frozen source config")
    policy = validate_runner_config(config, diagnostic=True)
    if manifest.get("policy") != policy or config.get("q4_variant", "q4_0") != "q4_0":
        raise ValueError("native benchmark policy differs from the frozen diagnostic")
    variants = config.get("variants", {})
    if not {"q4_0", "row_a16_checkpoint_zero"}.issubset(variants):
        raise ValueError("native benchmark config omits Q4_0 or row-A16")
    block = _json(capture_dir / "manifest.json")
    matched = [
        item
        for item in manifest.get("blocks", [])
        if item.get("variant") == "row_a16_checkpoint_zero"
        and Path(item.get("directory", "")).resolve() == capture_dir.resolve()
    ]
    if len(matched) != 1:
        raise ValueError("capture block is not uniquely owned by the native benchmark manifest")
    manifest_block = matched[0]
    spec = variants["row_a16_checkpoint_zero"]
    expected_command = runner_command(config, spec, policy, diagnostic=True)
    expected_env = runner_server_env(config, spec, "instrumented", capture_dir)
    expected_env = {
        key: value
        for key, value in expected_env.items()
        if key.startswith((*CLEAR_PREFIXES, "CUDA_VISIBLE"))
    }
    binary = Path(expected_command[0]).resolve()
    if (
        block.get("variant") != "row_a16_checkpoint_zero"
        or block.get("directory") != manifest_block.get("directory")
        or block.get("command") != expected_command
        or manifest_block.get("command") != expected_command
        or block.get("env") != expected_env
        or manifest_block.get("env") != expected_env
        or (binary_path is not None and binary != binary_path.resolve())
        or block.get("server_exit_code") != 0
        or manifest_block.get("server_exit_code") != 0
        or not isinstance(block.get("graph_status"), str)
        or block.get("graph_status") != manifest_block.get("graph_status")
    ):
        raise ValueError("executed row-A16 block command, environment or stop evidence differs")
    hashes = manifest.get("hashes", {})
    if not isinstance(hashes.get("binary"), str) or len(hashes["binary"]) != 64:
        raise ValueError("native benchmark manifest lacks its binary hash")
    if binary_path is not None and sha256(binary_path) != hashes["binary"]:
        raise ValueError("native benchmark binary hash differs from the executed binary")
    if hashes.get("target") != TARGET_GGUF_SHA256:
        raise ValueError("native benchmark target hash differs from the pinned target")
    drafts = hashes.get("drafts", {})
    if (
        not isinstance(drafts.get("q4_0"), str)
        or len(drafts["q4_0"]) != 64
        or drafts.get("row_a16_checkpoint_zero") != CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
    ):
        raise ValueError(
            "native benchmark Q4_0 or row-A16 draft hash differs from the frozen model"
        )
    return block


def assemble(args) -> dict:
    paths = {
        name: Path(value).resolve() for name, value in vars(args).items() if isinstance(value, Path)
    }
    if paths["report"] == paths["provider_overlay"]:
        raise ValueError("readiness report and provider overlay need separate output files")
    for key in ("report", "provider_overlay"):
        if paths[key].exists():
            raise FileExistsError(paths[key])
    provider = _json(paths["provider_manifest"])
    if (
        provider.get("schema") != "w1ax_native_train_provider_v1"
        or provider.get("training_eligible") is not False
        or provider.get("teacher") is not None
        or provider.get("split") != "train"
        or provider.get("prompt_count") != 31
    ):
        raise ValueError("source provider must remain ineligible and have no compact teacher")
    provider_paths, provider_hashes = provider["paths"], provider["sha256"]
    expected_paths = {
        "capture_manifest",
        "prompts",
        "target_gguf",
        "candidate_d_gguf",
        "base_draft_gguf",
        "absolute_d2t",
        "model_snapshot_manifest",
        "target_model_dir",
        "draft_model_dir",
    }
    if set(provider_paths) != expected_paths or set(provider_hashes) != expected_paths - {
        "target_model_dir",
        "draft_model_dir",
    }:
        raise ValueError("provider input paths and hashes are not the exact pinned inventory")
    for name in (
        "capture_manifest",
        "prompts",
        "absolute_d2t",
        "model_snapshot_manifest",
        "target_gguf",
        "candidate_d_gguf",
        "base_draft_gguf",
    ):
        if sha256(Path(provider_paths[name])) != provider_hashes[name]:
            raise ValueError(f"provider source changed: {name}")
    capture_manifest = _json(Path(provider_paths["capture_manifest"]))
    if (
        provider_hashes["capture_manifest"]
        != "3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947"
        or capture_manifest.get("training_eligible") is not False
        or capture_manifest.get("readiness") != "preparation_only"
        or not capture_manifest.get("unverified_gates")
        or provider_hashes["prompts"]
        != "968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a"
    ):
        raise ValueError("original shard-0000 preparation bundle is changed or promoted")
    aliases = _alias_map(paths["alias_map"])
    if sha256(paths["alias_map"]) != CALIBRATION_PINNED_INPUT_SHA256["diagnostic_id_map"]:
        raise ValueError("diagnostic alias map hash differs from frozen prompt mapping")
    if (
        sha256(paths["diagnostic_prompts_jsonl"])
        != CALIBRATION_PINNED_INPUT_SHA256["diagnostic_prompts_jsonl"]
        or sha256(paths["diagnostic_config"])
        != CALIBRATION_PINNED_INPUT_SHA256["diagnostic_config"]
    ):
        raise ValueError("diagnostic prompt/config hash differs from the frozen pilot")
    identity = _json(paths["identity_report"])
    old_manifest = _json(paths["old_native_manifest"])
    old_block = _validate_runner_manifest(
        paths["old_native_manifest"],
        old_manifest,
        paths["diagnostic_config"],
        paths["old_native_capture_dir"],
    )
    identity_result = _identity(identity, sha256(paths["identity_report"]), old_manifest)
    if (
        sha256(paths["old_native_manifest"])
        != CALIBRATION_PINNED_INPUT_SHA256["native_diagnostic_manifest"]
    ):
        raise ValueError("old native diagnostic manifest differs from frozen run")
    if (
        sha256(paths["old_native_capture_dir"] / "heads.jsonl")
        != CALIBRATION_PINNED_INPUT_SHA256["native_head_metadata"]
    ):
        raise ValueError("old native head metadata differs from frozen run")
    if (
        _json(paths["bundle_audit_report"]).get("capture_manifest_sha256")
        != provider_hashes["capture_manifest"]
        or sha256(paths["bundle_audit_report"]) != BUNDLE_AUDIT_SHA256
    ):
        raise ValueError("independent first-shard bundle audit differs")
    bundle_audit = _json(paths["bundle_audit_report"])
    if (
        bundle_audit.get("counts", {}).get("valid") != 12610
        or bundle_audit["counts"].get("supported") != 12250
    ):
        raise ValueError("independent bundle audit row counts differ")
    offsets_record = capture_manifest.get("offsets", {})
    offsets_path = Path(provider_paths["capture_manifest"]).parent / offsets_record.get("path", "")
    offsets = np.load(offsets_path, allow_pickle=False)
    absolute = np.load(Path(provider_paths["absolute_d2t"]), allow_pickle=False)
    expected_absolute = np.arange(len(offsets), dtype=np.int64) + offsets
    if (
        sha256(offsets_path) != offsets_record.get("sha256")
        or absolute.dtype.kind not in "iu"
        or not np.array_equal(absolute, expected_absolute)
    ):
        raise ValueError("provider absolute d2t map differs from audited native offsets")
    source_cell_path = Path(provider_paths["capture_manifest"]).parent / "source_cell_manifest.json"
    source_cell_hash = capture_manifest.get("source_report_sha256", {}).get("cell_manifest")
    if not source_cell_hash or sha256(source_cell_path) != source_cell_hash:
        raise ValueError("capture source-cell manifest is missing or changed")
    shard_record = capture_manifest.get("shard_manifest")
    if not isinstance(shard_record, dict) or set(shard_record) != {"path", "sha256"}:
        raise ValueError("audited bundle lacks its frozen shard manifest")
    shard_path = Path(provider_paths["capture_manifest"]).parent / shard_record["path"]
    if sha256(shard_path) != shard_record["sha256"]:
        raise ValueError("frozen shard manifest changed")
    bundle_source_hashes = bundle_audit.get("source_sha256", {})
    for field, record in capture_manifest.items():
        if field not in {
            "rows",
            "anchors",
            "offsets",
            "t2d",
            "features",
            "feature_rows",
            "target_logits",
        }:
            continue
        source_path = Path(provider_paths["capture_manifest"]).parent / record["path"]
        if bundle_source_hashes.get(field) != sha256(source_path):
            raise ValueError(f"independent bundle audit source differs: {field}")
    inventory = _provider_inventory(provider)

    old_pairs = _exact_response_pairs(old_manifest)
    native_manifest = _json(paths["native_manifest"])
    cache_config_path = ROOT / "configs/w1_phase1b_pilot_cuda_cache_diagnostic.json"
    native_block = _validate_runner_manifest(
        paths["native_manifest"],
        native_manifest,
        cache_config_path,
        paths["native_capture_dir"],
        paths["native_server_binary"],
    )
    if old_manifest["hashes"]["drafts"]["q4_0"] != native_manifest["hashes"]["drafts"]["q4_0"]:
        raise ValueError("old and cache-run Q4_0 model hashes differ")
    new_task_map = _task_map(native_manifest, aliases)
    new_measured_task_ids = _measured_row_a16_task_ids(native_manifest, aliases)
    task_map_hash = sha256(paths["native_manifest"])
    new_pairs = _exact_response_pairs(native_manifest)
    if old_pairs != new_pairs:
        raise ValueError("new CUDA cache capture responses differ from frozen exact Q4_0/A16 pairs")
    new_accepted, new_wrong_accepted = _wrong_accepted_labels(
        paths["native_capture_dir"], new_measured_task_ids
    )
    if new_accepted != 31 or new_wrong_accepted != 0:
        raise ValueError(
            "new measured row-A16 accepted rows differ from the 31-row verifier-label gate"
        )

    old_measured_task_ids = _measured_row_a16_task_ids(old_manifest, aliases)
    accepted, wrong_accepted = _wrong_accepted_labels(
        paths["old_native_capture_dir"], old_measured_task_ids
    )
    if accepted != 31 or wrong_accepted != 0:
        raise ValueError("old native accepted rows do not all match verifier labels")

    numeric_path, gradient_path = paths["numeric_report"], paths["gradient_report"]
    numeric = _json(numeric_path)
    if sha256(numeric_path) != NUMERIC_REPORT_SHA256:
        raise ValueError("numeric report differs from frozen passing CUDA trajectory report")
    numeric_roots = _numeric_roots(numeric)
    numeric_inputs = numeric.get("input_sha256", {})
    if (
        numeric_inputs.get("candidate_manifest") != provider_hashes["capture_manifest"]
        or numeric_inputs.get("candidate_prompts") != provider_hashes["prompts"]
        or numeric_inputs.get("checkpoint") != CALIBRATION_PINNED_INPUT_SHA256["checkpoint_zero"]
        or numeric_inputs.get("row_export_gguf") != CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
    ):
        raise ValueError("Torch numeric report is not bound to frozen provider/checkpoint inputs")
    outcomes = {
        domain: sorted(
            {
                row.get("preceding_student_round_outcome")
                for row in rows[1:]
                if row.get("preceding_student_round_outcome") in {"accepted", "rejected"}
            }
        )
        for domain, rows in numeric_roots.items()
    }
    gradient = _json(gradient_path)
    gradient_roots = _gradient_roots(gradient)
    gradient_inputs = gradient.get("input_sha256", {})
    if (
        gradient_inputs.get("candidate_manifest") != provider_hashes["capture_manifest"]
        or gradient_inputs.get("candidate_prompts") != provider_hashes["prompts"]
        or gradient_inputs.get("checkpoint") != CALIBRATION_PINNED_INPUT_SHA256["checkpoint_zero"]
        or gradient_inputs.get("row_export_gguf")
        != CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
    ):
        raise ValueError("gradient report is not bound to frozen provider/checkpoint inputs")
    for domain in EXPECTED_DOMAINS:
        for index, (nrow, grow) in enumerate(zip(numeric_roots[domain], gradient_roots[domain])):
            if grow.get("root_index") != index:
                raise ValueError("gradient report does not match numeric selected-root ordering")

    cache_dir = paths["native_capture_dir"]
    cache_audit = _json(paths["cache_audit_report"])
    _validate_mask_placement(cache_audit)
    if (
        cache_audit.get("schema") != "recurrent_stored_draft_cache_audit_v2"
        or cache_audit.get("status") != "stored_f16_rows_and_exact_prefix_masks_compared"
        or cache_audit.get("graph_capture_scope") != "cache"
        or str(cache_audit.get("execution_device", "")).lower() != "cuda"
    ):
        raise ValueError("new native cache audit must be the completed CUDA v2 audit")
    for kind in ("key", "value"):
        if (
            type(cache_audit.get(f"{kind}_equal_elements")) is not int
            or cache_audit[f"{kind}_equal_elements"] < 1
            or cache_audit.get(f"{kind}_equal_elements") != cache_audit.get(f"{kind}_elements")
        ):
            raise ValueError(f"native CUDA {kind} cache bytes do not exactly match projected F16")
    if (
        cache_audit.get("exact_prefix_mask_rows") != cache_audit.get("captured_rows")
        or type(cache_audit.get("decoder_executions")) is not int
        or cache_audit["decoder_executions"] < 1
    ):
        raise ValueError("native CUDA cache audit has incomplete masks or decoder coverage")
    cache_index_path = cache_dir / "heads.draft_cache.jsonl"
    graph_index_path = cache_dir / "heads.draft_graph.jsonl"
    cache_events = _jsonl(cache_index_path)
    graph_events = _jsonl(graph_index_path)
    footer = next(
        (row for row in reversed(graph_events) if row.get("event") in {"footer", "capture_end"}),
        None,
    )
    graph_count = (
        None if footer is None else footer.get("decoder_groups", footer.get("decoder_executions"))
    )
    if (
        footer is None
        or footer.get("status") != "complete"
        or footer.get("scope") != "cache"
        or type(graph_count) is not int
        or type(footer.get("result_output_markers")) is not int
        or not 0 <= footer["result_output_markers"] <= graph_count
        or graph_count != cache_audit["decoder_executions"]
    ):
        raise ValueError("native cache audit decoder count differs from the complete graph index")
    source_names = {
        "cache_index": cache_index_path,
        "cache_rows": cache_dir / "heads.draft_cache.f16",
        "cache_masks": cache_dir / "heads.draft_cache.mask",
        "graph_index": graph_index_path,
        "graph_values": cache_dir / "heads.draft_graph.f32",
    }
    if any(
        cache_audit.get("source_sha256", {}).get(name) != sha256(path)
        for name, path in source_names.items()
    ):
        raise ValueError("native CUDA audit source hashes differ from captured payloads")
    native_binary = paths["native_server_binary"]
    binary_hash = sha256(native_binary)
    manifest_binary_hash = native_manifest.get("hashes", {}).get("binary")
    if not manifest_binary_hash or manifest_binary_hash != binary_hash:
        raise ValueError("new native capture binary hash differs from its benchmark manifest")

    state_rows = _jsonl(cache_dir / "state.jsonl")
    cache_rows = cache_events
    bridge_rows = _bridge_roots(
        numeric_roots,
        gradient_roots,
        new_task_map,
        cache_dir,
        state_rows,
        cache_rows,
        graph_index_path,
        cache_dir / "heads.draft_graph.f32",
    )
    if footer["result_output_markers"] < len(bridge_rows):
        raise ValueError("native graph output-boundary markers omit a selected root")
    if inventory["compact_teacher_attached"]:
        raise ValueError("hard-CE calibration may not attach a compact teacher")
    checks = {
        "pinned_inputs_and_response_ancestry": {
            **identity_result,
            "prompt_hash_match": True,
            "bundle_audit_hash_match": True,
            "model_hashes_match": True,
            "response_requests": len(old_pairs),
            "response_exact_matches": sum(pair["exact_ids"] for pair in old_pairs),
            "old_accepted_drafts": accepted,
            "wrong_accepted_labels": wrong_accepted,
            "new_accepted_drafts": new_accepted,
            "new_wrong_accepted_labels": new_wrong_accepted,
            "old_runtime_graph_status": old_block["graph_status"],
        },
        "selected_root_mapping_and_operands": {
            "selected_roots": len(bridge_rows),
            "selected_roots_by_domain": {
                domain: len(rows) for domain, rows in numeric_roots.items()
            },
            "available_outcomes_by_domain": outcomes,
            "selected_outcomes_by_domain": outcomes,
            "token_ids_exact": True,
            "absolute_d2t_exact": True,
            "decoder_positions_exact": True,
            "causal_visibility_exact": True,
            "cache_lengths_exact": True,
            "finite_target_features": True,
            "finite_student_logits": True,
            "finite_gradients": True,
            "native_projected_kv_matches_stored_f16": True,
            "torch_cache_uses_f16_storage_rounding": True,
            "embedding_rows_exact": True,
            "hard_sign_bits_exact": True,
            "row_scales_exact": True,
            "root_position_cache_bridges": bridge_rows,
            "native_cache_audit_decoder_executions": cache_audit["decoder_executions"],
            "native_graph_result_output_markers": footer["result_output_markers"],
            "native_runtime_graph_status": native_block["graph_status"],
        },
        "student_native_proposal_and_response_agreement": {
            "shared_roots": len(bridge_rows),
            "top_choice_disagreements": numeric["checks"]["proposal_disagreements"],
            "changed_proposal_margins": [
                row["torch_top_two_margin_raw_logit"]
                for domain_rows in numeric_roots.values()
                for row in domain_rows
                if not row["top1_target_id_matches_native"]
            ],
            "near_tie_disagreements": 0,
            "high_margin_changed_proposals": numeric["checks"][
                "changed_top_choice_with_margin_over_0_02"
            ],
            "q4_response_ids_exact": True,
        },
        "student_native_numeric_tolerance": {
            "max_state_relative_rms": max(
                row["state_relative_rms"] for rows in numeric_roots.values() for row in rows
            ),
            "max_logits_relative_rms": max(
                row["logit_relative_rms"] for rows in numeric_roots.values() for row in rows
            ),
            "native_head_replay_top_two_margins": [
                row["native_packed_head_top_two_margin"]
                for rows in numeric_roots.values()
                for row in rows
            ],
        },
        "provider_round_label_and_teacher_contract": inventory,
    }
    inputs = {
        **provider_hashes,
        **CALIBRATION_PINNED_INPUT_SHA256,
        "diagnostic_prompts_jsonl": sha256(paths["diagnostic_prompts_jsonl"]),
        "diagnostic_id_map": sha256(paths["alias_map"]),
        "diagnostic_config": sha256(paths["diagnostic_config"]),
        "native_diagnostic_manifest": sha256(paths["old_native_manifest"]),
        "native_head_metadata": sha256(paths["old_native_capture_dir"] / "heads.jsonl"),
        "native_trace": sha256(paths["old_native_capture_dir"] / "state.jsonl"),
        "torch_numeric_report": sha256(numeric_path),
        "provider_manifest": sha256(paths["provider_manifest"]),
        "identity_report": sha256(paths["identity_report"]),
        "numeric_report": sha256(numeric_path),
        "gradient_report": sha256(gradient_path),
        "bundle_audit_report": sha256(paths["bundle_audit_report"]),
        "source_cell_manifest": source_cell_hash,
        "shard_manifest": sha256(shard_path),
        **{f"capture_{name}": digest for name, digest in bundle_source_hashes.items()},
        "alias_map": sha256(paths["alias_map"]),
        "old_native_heads": sha256(paths["old_native_capture_dir"] / "heads.jsonl"),
        "old_native_rounds": sha256(paths["old_native_capture_dir"] / "rounds.jsonl"),
        "old_native_capture_block_manifest": sha256(
            paths["old_native_capture_dir"] / "manifest.json"
        ),
        "old_native_manifest": sha256(paths["old_native_manifest"]),
        "old_native_records": sha256(paths["old_native_manifest"]),
        "native_manifest": sha256(paths["native_manifest"]),
        "native_benchmark_records": sha256(paths["native_manifest"]),
        "native_task_map": task_map_hash,
        "native_cache_config": sha256(cache_config_path),
        "native_cache_capture_manifest": sha256(paths["native_manifest"]),
        "native_cache_capture_block_manifest": sha256(cache_dir / "manifest.json"),
        "native_cache_audit": sha256(paths["cache_audit_report"]),
        "native_cache_binary": binary_hash,
        "native_cache_heads": sha256(cache_dir / "heads.jsonl"),
        "native_cache_states": sha256(cache_dir / "state.jsonl"),
        "native_cache_rounds": sha256(cache_dir / "rounds.jsonl"),
        "native_cache_index": sha256(cache_index_path),
        "native_cache_graph_index": sha256(graph_index_path),
        "native_cache_graph_values": sha256(cache_dir / "heads.draft_graph.f32"),
        "native_cache_payload": sha256(cache_dir / "heads.draft_cache.f16"),
        "native_cache_masks": sha256(cache_dir / "heads.draft_cache.mask"),
    }
    evidence_dir = paths["report"].parent / (paths["report"].stem + "-evidence")
    if evidence_dir.exists():
        raise FileExistsError(evidence_dir)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True)
    evidence_refs = {}
    for name, result in checks.items():
        evidence_path = evidence_dir / f"{name}.json"
        evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        evidence_refs[name] = [
            {"path": str(evidence_path.resolve()), "sha256": sha256(evidence_path)}
        ]
    report = {
        "schema": CALIBRATION_READINESS_SCHEMA,
        "scope": CALIBRATION_ONLY_SCOPE,
        "training_eligible": False,
        "full_body_qat_eligible": False,
        "capture_manifest_sha256": provider_hashes["capture_manifest"],
        "unresolved_full_body_gates": capture_manifest["unverified_gates"],
        "budget": {"steps": 100, "rounds": 100},
        "objective": "hard_ce",
        "contract": {"activation_bits": 16, "scale_layout": "row"},
        "relative_rms_definition": "rms_delta_over_max_rms_native_1e-8",
        "inputs": inputs,
        "checks": {
            name: {"status": "pass", "result": value, "evidence": evidence_refs[name]}
            for name, value in checks.items()
        },
    }
    validate_calibration_readiness_report(
        report,
        capture_manifest_sha256=provider_hashes["capture_manifest"],
        unresolved_full_body_gates=capture_manifest["unverified_gates"],
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    overlay = json.loads(json.dumps(provider))
    overlay["calibration_readiness"] = {
        "path": str(paths["report"]),
        "sha256": sha256(paths["report"]),
    }
    overlay["training_eligible"] = False
    paths["provider_overlay"].parent.mkdir(parents=True, exist_ok=True)
    paths["provider_overlay"].write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "provider-manifest",
        "identity-report",
        "numeric-report",
        "gradient-report",
        "bundle-audit-report",
        "alias-map",
        "diagnostic-prompts-jsonl",
        "diagnostic-config",
        "old-native-manifest",
        "old-native-capture-dir",
        "native-manifest",
        "native-capture-dir",
        "native-server-binary",
        "cache-audit-report",
        "report",
        "provider-overlay",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args)
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "training_eligible": report["training_eligible"],
                "checks": {name: item["status"] for name, item in report["checks"].items()},
                "readiness_report": str(args.report),
                "provider_overlay": str(args.provider_overlay),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
