"""Summarize native scale-screen quality, raw IDs and boundary-state checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def state_audit(events, rounds=None):
    counts = {
        "initial": 0,
        "zero": 0,
        "partial": 0,
        "full": 0,
        "completed_pairs": 0,
        "paired_zero": 0,
        "paired_partial": 0,
        "paired_full": 0,
        "accept": 0,
        "seed": 0,
    }
    errors = []
    pending = {}
    proposing = [
        r for r in (rounds or []) if r["status"] != "checkpoint_replay" and r["n_proposed"] > 0
    ]
    accepts = [e for e in events if e["event"] == "accept"]
    mapped = len(accepts) == len(proposing)
    if not mapped and accepts:
        errors.append("accept/round mapping count mismatch")
    accept_index = 0
    for event in events:
        seq = event["seq_id"]
        if event["event"] == "accept":
            counts["accept"] += 1
            accepted = event["accepted"]
            proposed = proposing[accept_index]["n_proposed"] if mapped else 5
            if mapped and proposing[accept_index]["n_accepted"] != accepted:
                errors.append("accept/round count differs")
            accept_index += 1
            kind = "zero" if accepted == 0 else ("full" if accepted == proposed else "partial")
            if accepted >= event["verify_rows"]:
                errors.append("accept indexing unexpectedly clamped")
            counts[kind] += 1
            expected = min(accepted, event["verify_rows"] - 1)
            if event["selected_row"] != expected:
                errors.append("selected row differs from accepted prefix")
            if event["position"] != event["verify_pos_first"] + expected:
                errors.append("selected position mismatch")
            if event["selected_hash"] != event["pending_hash"]:
                errors.append("selected/pending state differs")
            pending[seq] = {**event, "transition_kind": kind}
        elif event["event"] == "seed":
            counts["seed"] += 1
            if event["kv_max_after"] >= event["position"]:
                errors.append("draft cache not truncated before seed")
            prior = pending.pop(seq, None)
            if prior is None:
                counts["initial"] += 1
            else:
                counts["completed_pairs"] += 1
                counts["paired_" + prior["transition_kind"]] += 1
            if prior is not None and (
                prior["position"] != event["position"]
                or prior["pending_hash"] != event["pending_hash"]
            ):
                errors.append("next seed differs from retained acceptance state")
    return {
        "counts": counts,
        "errors": errors,
        "observed": bool(events),
        "passed": bool(events) and not errors,
        "scope": "boundary feature copy and cache truncation extent; not cached K/V tensor parity",
    }


def analyze(directory):
    summary = read(directory / "summary.json")
    reference = summary.get("target_only")
    result = {
        "variants": {},
        "definitions": {
            "accepted_per_round": "verifier accepted drafts / non-replay rounds",
            "prefix_survival_k": "count(accepted>=k) / count(proposed>=k); first proposal is k=1",
            "terminal": "verifier decisions include accepted drafts not emitted at stop",
            "timing": "quality screen only; no throughput claim",
        },
    }
    for name, variant in summary.items():
        entry = {"quality": variant["quality"], "per_prompt": [], "state_audits": []}
        target = (
            {r["id"]: r["generated_token_ids"] for r in reference["requests"]} if reference else {}
        )
        for index, request in enumerate(variant["requests"]):
            ids = request["generated_token_ids"]
            ref = target.get(request["id"])
            first = None
            if ref is not None:
                first = next(
                    (
                        i
                        for i in range(max(len(ids), len(ref)))
                        if i >= len(ids) or i >= len(ref) or ids[i] != ref[i]
                    ),
                    None,
                )
            entry["per_prompt"].append(
                {
                    "id": request["id"],
                    "quality": request["quality"],
                    "output_tokens": len(ids),
                    "target_only_match": ids == ref if ref is not None else None,
                    "first_difference": first,
                    "ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
            path = directory / name / f"{index:03d}-{request['id']}" / "state.json"
            if path.exists():
                entry["state_audits"].append(
                    {
                        "id": request["id"],
                        **state_audit(read(path), read(path.parent / "rounds.json")),
                    }
                )
        entry["target_only_mismatch_prompts"] = sum(
            r["target_only_match"] is False for r in entry["per_prompt"]
        )
        result["variants"][name] = entry
    # Diagnostic endpoint equivalence pairs, if present.
    for first, second in [("A_legacy", "A"), ("ordinary", "cast_only")]:
        if first not in summary or second not in summary:
            continue
        checks = []
        for i, (a, b) in enumerate(
            zip(summary[first]["requests"], summary[second]["requests"], strict=True)
        ):
            if a["id"] != b["id"]:
                raise ValueError("pair prompt order differs")
            pa = directory / first / f"{i:03d}-{a['id']}"
            pb = directory / second / f"{i:03d}-{b['id']}"
            keys = (
                "status",
                "n_accepted",
                "n_proposed",
                "n_emitted",
                "proposed_token_ids",
                "emitted_token_ids",
            )
            ra = [{k: r[k] for k in keys} for r in read(pa / "rounds.json")]
            rb = [{k: r[k] for k in keys} for r in read(pb / "rounds.json")]
            checks.append(
                {
                    "id": a["id"],
                    "output_match": a["generated_token_ids"] == b["generated_token_ids"],
                    "rounds_match": ra == rb,
                    "state_match": read(pa / "state.json") == read(pb / "state.json"),
                }
            )
        result[f"{first}_vs_{second}"] = checks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("variant accepted/round first prefix2 prefix3 target_mismatch")
    for name, entry in result["variants"].items():
        q = entry["quality"]
        columns = [name, f"{q['accepted']}/{q['rounds']}"]
        columns += [
            f"{q['depth'][str(d)]['survived']}/{q['depth'][str(d)]['eligible']}" for d in (1, 2, 3)
        ]
        columns += [str(entry["target_only_mismatch_prompts"])]
        print(" ".join(columns))


if __name__ == "__main__":
    main()
