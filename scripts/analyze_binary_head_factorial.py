#!/usr/bin/env python3
"""Audit forced-history head captures. Diagnostic agreement is not live acceptance."""
from __future__ import annotations

import argparse
import array
import json
import math
import statistics
from pathlib import Path

from capture_w1ax_activations import read_jsonl, sha256

ARMS = {"q4_q4": ("Q4_0", "Q4_0"), "d_d": ("D", "D"),
        "q4_d": ("Q4_0", "D"), "d_q4": ("D", "Q4_0"), "d_f16": ("D", "FP16")}
BOUNDARY = "native_output_norm_f32_before_head_operand_conversion"


def audit_rows(rows: list[dict], states_path: Path, joins: dict[str, str]) -> dict:
    if not rows:
        raise ValueError("missing native head capture rows")
    width = rows[0].get("state_dim")
    if type(width) is not int or width < 1:
        raise ValueError("invalid state dimension")
    seen = set()
    previous = None
    for index, row in enumerate(rows):
        if (row.get("schema") != "eagle_head_state_v1" or row.get("state_row") != index
                or row.get("state_dim") != width or row.get("state_boundary") != BOUNDARY
                or row.get("state_dtype") != "float32_native_endian"):
            raise ValueError("native state schema/order/boundary mismatch")
        prompt = joins.get(str(row.get("task_id")))
        if prompt is None:
            raise ValueError("unjoined native task ID")
        depth, parent = row.get("depth"), row.get("parent_position")
        if type(depth) is not int or not 0 <= depth < 5 or type(parent) is not int:
            raise ValueError("invalid depth or parent position")
        prefix = row.get("prefix_token_ids")
        if (row.get("alignment_valid") is not True or row.get("finite") is not True
                or row.get("valid") is not True or row.get("is_bonus") is not False
                or row.get("verifier_row") != depth or row.get("input_position") != parent+depth+1
                or row.get("label_position") != parent+depth+2
                or not isinstance(prefix, list) or len(prefix) != row["label_position"]
                or not prefix or prefix[-1] != row.get("input_token_id")
                or any(type(token) is not int or token < 0 for token in prefix)
                or type(row.get("verifier_token_id")) is not int
                or row["verifier_token_id"] < 0):
            raise ValueError("invalid state/verifier alignment or finite mask")
        key = (prompt, row.get("round_index"), depth)
        if key in seen or type(key[1]) is not int or key[1] < 0:
            raise ValueError("duplicate or invalid prompt/round/depth")
        seen.add(key)
        same_round = previous and key[:2] == previous[0][:2]
        if same_round:
            prior = previous[1]
            if (depth != prior["depth"]+1 or parent != prior["parent_position"]
                    or prefix != prior["prefix_token_ids"]+[prior["proposed_token_id"]]):
                raise ValueError("capture does not follow the body's recurrent prefix")
        elif depth != 0:
            raise ValueError("capture round must begin at depth zero")
        previous = (key, row)
        supported = row.get("label_supported")
        if type(supported) is not bool:
            raise ValueError("missing supported-label mask")
        if supported:
            if (type(row.get("target_rank")) is not int or row["target_rank"] < 1
                    or not isinstance(row.get("target_margin"), (float, int))
                    or not math.isfinite(row["target_margin"])):
                raise ValueError("invalid supported label rank/margin")
        elif row.get("target_rank") is not None or row.get("target_margin") is not None:
            raise ValueError("unsupported labels must have null rank/margin")
    if states_path.stat().st_size != len(rows)*width*4:
        raise ValueError("state F32 payload size mismatch")
    with states_path.open("rb") as stream:
        while block := stream.read(1024*1024):
            values = array.array("f")
            values.frombytes(block)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite F32 state payload")
    return {"rows": len(rows), "state_dim": width, "states_sha256": sha256(states_path)}


def head_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid"]]
    supported = [r for r in valid if r["label_supported"]]
    agreement = sum(r["draft_argmax_id"] == r["verifier_token_id"] for r in valid)
    ranks = [r["target_rank"] for r in supported]
    margins = [r["target_margin"] for r in supported]
    reached = [r for r in valid if r.get("verifier_reached")]
    return {"total": len(rows), "valid": len(valid), "supported": len(supported),
            "unsupported": len(valid)-len(supported), "agreement_count": agreement,
            "agreement_rate_valid": agreement/len(valid) if valid else None,
            "agreement_rate_supported": agreement/len(supported) if supported else None,
            "target_rank_mean_supported": statistics.mean(ranks) if ranks else None,
            "target_rank_median_supported": statistics.median(ranks) if ranks else None,
            "target_margin_mean_supported": statistics.mean(margins) if margins else None,
            "verifier_reached": len(reached),
            "raw_argmax_sampler_disagreements": sum(
                r["raw_verifier_argmax_id"] != r["verifier_token_id"] for r in valid),
            "reached_sampler_disagreements": sum(
                r["verifier_token_id"] != r.get("verifier_sampled_token_id") for r in reached)}


def history_rows(rows: list[dict], joins: dict[str, str]) -> list:
    return [(joins[str(r["task_id"])], r["round_index"], r["depth"], r["parent_position"],
             r["input_position"], r["label_position"], r["prefix_token_ids"],
             r["proposed_token_id"], r["verifier_token_id"]) for r in rows]


def round_history(rows: list[dict], joins: dict[str, str]) -> list:
    result = []
    for row in rows:
        if row.get("schema") != "eagle_forced_round_v1":
            raise ValueError("invalid forced round schema")
        result.append((joins[str(row["task_id"])], row["round_index"], row["prefix_token_ids"],
                       row["seed_token_id"], row["pos0"], row["draft_token_ids"],
                       row["accepted_drafts"], row["verifier_token_ids"]))
    return result


def analyze(directory: Path) -> dict:
    manifests = {name: json.loads((directory/name/"manifest.json").read_text())
                 for name in ["canonical", *ARMS]}
    reference = manifests["canonical"]
    canonical_rows = read_jsonl(directory/"canonical"/"heads.jsonl")
    canonical_audit = audit_rows(canonical_rows, directory/"canonical"/"heads.f32",
                                 reference["task_prompt_ids"])
    baseline_rounds = round_history(read_jsonl(directory/"canonical"/"forced-rounds.jsonl"),
                                    reference["task_prompt_ids"])
    if not baseline_rounds:
        raise ValueError("empty canonical forced-round trace")
    outputs = [(r["id"], r["generated_token_ids"]) for r in reference["requests"]]
    reference_history = None
    body_hashes = {}
    results = {}
    for name, (body, head) in ARMS.items():
        cell, manifest = directory/name, manifests[name]
        joins = manifest["task_prompt_ids"]
        if manifest["spec"].get("body") != body or manifest["spec"].get("head") != head:
            raise ValueError("incorrect factorial arm identity")
        rows = read_jsonl(cell/"heads.jsonl")
        audit = audit_rows(rows, cell/"heads.f32", joins)
        for field in ("binary_sha256", "target_sha256", "prompts_sha256"):
            if not manifest.get(field) or manifest[field] != reference.get(field):
                raise ValueError(f"{name}: common {field} differs or missing")
        if not all(r.get("forced") is True for r in rows):
            raise ValueError("factorial contains unforced head states")
        if round_history(read_jsonl(cell/"forced-rounds.jsonl"), joins) != baseline_rounds:
            raise ValueError(f"{name}: forced round histories/boundaries differ")
        if [(r["id"], r["generated_token_ids"]) for r in manifest["requests"]] != outputs:
            raise ValueError(f"{name}: target output IDs differ")
        history = history_rows(rows, joins)
        if name == "q4_q4" and (
                history != history_rows(canonical_rows, reference["task_prompt_ids"])
                or audit["states_sha256"] != canonical_audit["states_sha256"]):
            raise ValueError("Q4 canonical versus forced replay differs")
        if reference_history is None:
            reference_history = history
        elif history != reference_history:
            raise ValueError(f"{name}: capture prefix/verifier labels differ")
        if body in body_hashes and audit["states_sha256"] != body_hashes[body]:
            raise ValueError(f"{name}: same-body normalized head states differ")
        body_hashes[body] = audit["states_sha256"]
        results[name] = {"body": body, "head": head, "audit": audit,
                         "all": head_metrics(rows),
                         "first": head_metrics([r for r in rows if r["depth"] == 0]),
                         "later": head_metrics([r for r in rows if r["depth"] > 0]),
                         "by_depth": {str(d): head_metrics([r for r in rows if r["depth"] == d])
                                      for d in range(5)}}
    return {"passed": True, "kind": "forced_history_diagnostic_not_live_acceptance",
            "rounds": len(baseline_rounds), "arms": results,
            "gates": ["same_prefixes", "same_round_boundaries", "same_verifier_labels",
                      "same_target_outputs", "byte_identical_states_within_body"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    text = json.dumps(result, indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
