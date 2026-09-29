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
from w1ax_capture_provider import sha256  # noqa: E402

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


def _task_map(manifest: dict, manifest_path: Path, aliases: dict[str, str]) -> dict[int, str]:
    mapping = manifest.get("task_prompt_ids", manifest.get("task_map"))
    if mapping is None:
        record = manifest.get("files", {}).get("task_prompt_ids")
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            task_path = Path(record["path"])
            if not task_path.is_absolute():
                task_path = manifest_path.parent / task_path
            if record.get("sha256") != sha256(task_path):
                raise ValueError("native task-to-prompt map hash differs from run manifest")
            mapping = _json(task_path)
    if mapping is None and isinstance(manifest.get("records"), list):
        mapping = {}
        for row in manifest["records"]:
            if (
                not isinstance(row, dict)
                or row.get("variant") != "row_a16_checkpoint_zero"
                or row.get("warmup") is True
            ):
                continue
            task = (row.get("request_digest") or {}).get("task_id")
            prompt = row.get("prompt_id")
            if type(task) is not int or type(prompt) is not str:
                raise ValueError("measured row-A16 record lacks task-to-prompt ownership")
            prior = mapping.get(task)
            if prior is not None and prior != prompt:
                raise ValueError("native task ID is reused for different prompt aliases")
            mapping[task] = prompt
    if isinstance(mapping, list):
        mapping = {
            row.get("task_id"): row.get("prompt_id", row.get("id"))
            for row in mapping
            if isinstance(row, dict)
        }
    if not isinstance(mapping, dict) or not mapping:
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


def _read_benchmark_records(manifest_path: Path, manifest: dict) -> list[dict]:
    records = manifest.get("records")
    if isinstance(records, list):
        return records
    record = manifest.get("files", {}).get("records")
    path = (
        Path(record["path"])
        if isinstance(record, dict) and record.get("path")
        else (manifest_path.parent / "records.json")
    )
    if not path.is_absolute():
        path = manifest_path.parent / path
    if isinstance(record, dict) and record.get("sha256") != sha256(path):
        raise ValueError("benchmark records hash differs from manifest")
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("benchmark manifest has no measured request records")
    return value


def _records_path(manifest_path: Path, manifest: dict) -> Path | None:
    if isinstance(manifest.get("records"), list):
        return None
    record = manifest.get("files", {}).get("records")
    path = (
        Path(record["path"])
        if isinstance(record, dict) and record.get("path")
        else manifest_path.parent / "records.json"
    )
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _exact_response_pairs(manifest_path: Path, manifest: dict) -> list[dict]:
    records = [
        row
        for row in _read_benchmark_records(manifest_path, manifest)
        if row.get("warmup") is not True
    ]
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


def _wrong_accepted_labels(capture_dir: Path) -> tuple[int, int]:
    heads = _jsonl(capture_dir / "heads.jsonl")
    rounds = _jsonl(capture_dir / "rounds.jsonl")
    by_key: dict[tuple[int, int, int], list[dict]] = {}
    for row in heads:
        by_key.setdefault(
            (row.get("task_id"), row.get("round_index"), row.get("depth")), []
        ).append(row)
    accepted = wrong = 0
    for event in rounds:
        if event.get("complete") is not True:
            continue
        count = event.get("n_accepted")
        task, round_index = event.get("task_id"), event.get("round_index")
        if (
            type(count) is not int
            or count < 0
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
    for root in report.get("roots", []):
        domain = root.get("domain")
        if (
            domain not in result
            or root.get("gradient_tensors") != 18
            or root.get("finite_gradient_tensors") != 18
        ):
            raise ValueError("gradient report has an invalid selected root")
        result[domain].append(root)
    if any(len(result[d]) < 2 for d in result) or sum(map(len, result.values())) != 9:
        raise ValueError("gradient report must cover the same nine roots")
    for rows in result.values():
        rows.sort(key=lambda row: row.get("root_index", -1))
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
    seeds: dict[int, list[dict]] = {}
    accepts: dict[int, list[dict]] = {}
    for event in state_rows:
        if type(event.get("seq_id")) is not int:
            continue
        bucket = (
            seeds
            if event.get("event") == "seed"
            else accepts
            if event.get("event") == "accept"
            else None
        )
        if bucket is not None:
            bucket.setdefault(event["seq_id"], []).append(event)
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
            if gradient.get("round_index") != root["round_index"]:
                raise ValueError("numeric and gradient reports disagree on selected root order")
            prefix = root["prefix_token_ids"]
            matches = [
                row
                for row in heads
                if row.get("task_id") == task_id
                and row.get("depth") == 0
                and row.get("parent_position") == root["parent_position"]
                and row.get("prefix_token_ids") == prefix
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"selected {domain} prefix has no unique native cache-capture head"
                )
            head = matches[0]
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
            seq_seeds = seeds.get(task_id, [])
            if not 0 <= head.get("round_index", -1) < len(seq_seeds):
                raise ValueError("selected native seed ordinal is absent from state trace")
            seed = seq_seeds[head["round_index"]]
            if (
                seed.get("position") != parent
                or seed.get("token") != head.get("input_token_id")
                or seed.get("kv_max_before") != parent - 1
            ):
                raise ValueError("native seed position, token or logical cache length differs")
            preceding = root.get("preceding_student_round_outcome")
            if head["round_index"] > 0 and preceding in {"accepted", "rejected"}:
                seq_accepts = accepts.get(task_id, [])
                if head["round_index"] - 1 >= len(seq_accepts):
                    raise ValueError("native preceding accept event is absent")
                event = seq_accepts[head["round_index"] - 1]
                expected = "accepted" if event.get("accepted", 0) > 0 else "rejected"
                if expected != preceding:
                    raise ValueError("native accept event disagrees with selected-root outcome")
            writes = [
                row
                for row in cache_writes
                if row.get("position") == parent and row.get("token_id") == head["input_token_id"]
            ]
            if not writes:
                raise ValueError("selected native seed has no globally audited cache write")
            state_row = head.get("state_row")
            if type(state_row) is not int or not 0 <= state_row < len(state_payload):
                raise ValueError("selected native cache head state row is outside its payload")
            captured_state = np.asarray(state_payload[state_row])
            exact_graph_joins = []
            for write in writes:
                execution = write.get("execution")
                column = write.get("column")
                group = graph_groups.get(execution, {})
                result_norm = group.get("result_norm")
                if result_norm is None or type(column) is not int:
                    continue
                graph_state = _column(result_norm, graph_values, column)
                if graph_state.shape == captured_state.shape and np.array_equal(
                    graph_state, captured_state
                ):
                    exact_graph_joins.append((execution, column))
            if len(exact_graph_joins) != 1:
                raise ValueError(
                    "selected head state lacks one exact ordered result_norm/cache join"
                )
            graph_execution, graph_column = exact_graph_joins[0]
            bridges.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "round_index": head["round_index"],
                    "parent_position": parent,
                    "prefix_token_ids": prefix,
                    "state_seed_ordinal": head["round_index"],
                    "kv_max_before": seed["kv_max_before"],
                    "torch_cache_length_before_seed": len(context),
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
        or identity.get("exportstudent_gguf") != CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
        or identity.get("server_sha256") != old_manifest["files"]["binary"]["sha256"]
    ):
        raise ValueError("frozen pilot identity report is missing, mismatched or failed")
    return {
        "provider_hashes_checked": 7,
        "model_snapshots_verified": True,
        "exported_projection_pairs_checked": 9,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "exported_gguf_sha256": identity["exportstudent_gguf"],
        "server_sha256": identity["server_sha256"],
    }


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

    old_pairs = _exact_response_pairs(paths["old_native_manifest"], old_manifest)
    native_manifest = _json(paths["native_manifest"])
    if not native_manifest.get("prompt_ids") or set(native_manifest["prompt_ids"]) != set(
        EXPECTED_ALIASES
    ):
        raise ValueError("new native cache run does not use the frozen alias prompts")
    cache_config_path = ROOT / "configs/w1_phase1b_pilot_cuda_cache_diagnostic.json"
    if native_manifest.get("config_sha256") != sha256(cache_config_path):
        raise ValueError("new native cache run does not use the frozen 3-prompt, 128-token config")
    if not {"q4_0", "row_a16_checkpoint_zero"}.issubset(set(native_manifest.get("variants", []))):
        raise ValueError("new native cache run omits Q4_0 or row-A16 responses")
    native_capture_manifest = _json(paths["native_capture_manifest"])
    new_task_map = _task_map(native_manifest, paths["native_manifest"], aliases)
    task_map_hash = sha256(paths["native_manifest"])
    new_pairs = _exact_response_pairs(paths["native_manifest"], native_manifest)
    if old_pairs != new_pairs:
        raise ValueError("new CUDA cache capture responses differ from frozen exact Q4_0/A16 pairs")
    new_accepted, new_wrong_accepted = _wrong_accepted_labels(paths["native_capture_dir"])
    if new_accepted < 1 or new_wrong_accepted != 0:
        raise ValueError("new native capture has no accepted labels or a wrong accepted label")

    accepted, wrong_accepted = _wrong_accepted_labels(paths["old_native_capture_dir"])
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
            if grow.get("root_index") != index or grow.get("round_index") != nrow["round_index"]:
                raise ValueError("gradient report does not match numeric selected-root ordering")

    cache_dir = paths["native_capture_dir"]
    cache_audit = _json(paths["cache_audit_report"])
    if (
        cache_audit.get("schema") != "recurrent_stored_draft_cache_audit_v2"
        or cache_audit.get("status") != "stored_f16_rows_and_exact_prefix_masks_compared"
        or cache_audit.get("graph_capture_scope") != "cache"
        or str(cache_audit.get("execution_device", "")).lower() != "cuda"
        or str(cache_audit.get("mask_device", "")).lower() != "cuda"
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
    manifest_binary_hash = native_manifest.get("files", {}).get("binary", {}).get("sha256")
    if not manifest_binary_hash or manifest_binary_hash != binary_hash:
        raise ValueError("new native capture binary hash differs from its benchmark manifest")
    capture_source_hashes = native_capture_manifest.get("source_sha256", {})
    captured_binary_hash = next(
        (
            capture_source_hashes.get(name)
            for name in ("native_binary", "server_binary", "binary")
            if capture_source_hashes.get(name) is not None
        ),
        None,
    )
    if captured_binary_hash is None:
        captured_binary_hash = (
            native_capture_manifest.get("files", {}).get("binary", {}).get("sha256")
        )
    if captured_binary_hash != binary_hash:
        raise ValueError("new native cache capture does not bind the current native binary")

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
        "old_native_manifest": sha256(paths["old_native_manifest"]),
        "old_native_records": sha256(
            _records_path(paths["old_native_manifest"], old_manifest)
            or paths["old_native_manifest"]
        ),
        "native_manifest": sha256(paths["native_manifest"]),
        "native_benchmark_records": sha256(
            _records_path(paths["native_manifest"], native_manifest) or paths["native_manifest"]
        ),
        "native_task_map": task_map_hash,
        "native_cache_config": sha256(cache_config_path),
        "native_cache_capture_manifest": sha256(paths["native_capture_manifest"]),
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
        "native-capture-manifest",
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
