"""CPU-only one-point audit against a completed two-arm timing pair.

This report is neither an admission waiver nor a numeric-receipt consumer.
Runtime/model pins are inherited from the already checked run configurations;
only preserved JSON artifacts are hashed/read, and no model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from check_export import sha256

ARMS = ("eagle_q4_0", "dspark_3")
CASES = ("warmup-00", "warmup-01", *(f"prompt-{i:02d}" for i in range(6)))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def actual_point(records, task, position, ids):
    points = [r for r in records if r["task_id"] == task and r["generated_position"] == position and
              r.get("sampled") and r.get("emitted") and not r.get("replay")]
    need(len(points) == 1, "selected point is not one actual emitted nonreplay row")
    row = points[0]
    need(row.get("has_logits") and row.get("nan_logit_count") == 0 and
         row.get("sampled_token_id") == row.get("selected_token_id") == row.get("emitted_token_id") == ids[position],
         "raw sample/emission/logit join failed")
    top = row["raw_top5"]
    need(len(top) == 5 and len({p["token_id"] for p in top}) == 5 and
         all(isinstance(p.get("logit"), (int, float)) and math.isfinite(p["logit"]) for p in top),
         "raw top5 is incomplete/nonfinite")
    need(top[0]["token_id"] == ids[position] and all(top[i]["logit"] >= top[i+1]["logit"] for i in range(4)),
         "actual sample is not the eligible raw argmax")
    return row


def reached_prefix(row, rounds, ids):
    actual = [r for r in rounds if r["task_id"] == row["task_id"] and
              r["status"] in ("complete", "no_proposal") and not r.get("replay")]
    actual.sort(key=lambda r: r["round_index"])
    flat = [t for r in actual for t in r["emitted_token_ids"]]
    offset = 0 if flat == ids else 1 if flat == ids[1:] else None
    need(offset is not None, "native complete emitted-round stream differs from observed IDs")
    cursor, matches = offset, []
    for r in actual:
        if cursor + row["row"] == row["generated_position"] and row["row"] < len(r["emitted_token_ids"]):
            matches.append((cursor, r))
        cursor += len(r["emitted_token_ids"])
    need(len(matches) == 1, "selected retained-prefix round join is ambiguous")
    base, r = matches[0]
    j = row["row"]
    need(row.get("mode") == "speculative_verify" and row.get("base_generated_position") == base and
         r["round_index"] + 1 == row["round_index"] and r["n_accepted"] >= j and
         r["proposed_token_ids"][:j] == ids[base:base+j] == row["verifier_prefix_draft_ids"] and
         r["verified_token_ids"][j] == r["emitted_token_ids"][j] == ids[base+j],
         "selected row used an unreached/rejected input prefix")
    return {"task_id": row["task_id"], "native_round_index": r["round_index"], "verify_round_index": row["round_index"],
            "base_generated_position": base, "row": j, "n_accepted": r["n_accepted"],
            "retained_draft_prefix": row["verifier_prefix_draft_ids"], "untraced_prefill_tokens": offset}


def audit(diagnostic: Path, reference: Path, gate_path: Path, case="prompt-05", position=90):
    evidence = {}
    report = {"schema": "completed_native_pair_audit_v1", "passed": False, "case": case,
              "generated_position": position, "arms": list(ARMS), "evidence_sha256": evidence,
              "scope": "one completed-pair diagnostic only; no admission waiver or consumer change",
              "caveat": "no universal bit parity or confirmed arithmetic cause; no claim for unobserved paths"}
    def load(path, lines=False):
        path = Path(path).resolve()
        evidence[str(path)] = sha256(path)
        return [json.loads(line) for line in path.read_text().splitlines() if line] if lines else json.loads(path.read_text())
    try:
        need(case in CASES and position >= 0, "unfrozen selected diagnostic case")
        cfg, full_cfg = load(diagnostic / "config.json"), load(reference / "config.json")
        protocol = load(reference / "protocol.json")
        protocol_sha = evidence[str((reference / "protocol.json").resolve())]
        need(cfg["protocol_sha256"] == full_cfg["protocol_sha256"] == protocol_sha, "protocol pin changed")
        need(all(protocol.get(k) == v for k, v in {"context_tokens": 2048, "batch_tokens": 32,
             "microbatch_tokens": 32, "max_output_tokens": 128, "temperature": 0.0,
             "seed": 42, "enable_thinking": False, "cache_prompt": False}.items()), "frozen settings changed")
        for key in ("binary", "target", "eagle_q4_0", "dspark"):
            need(cfg[key]["path"] == full_cfg[key]["path"] and cfg[key]["sha256"] == full_cfg[key]["sha256"],
                 f"runtime/model pin changed: {key}")
        report["runtime_sha256"] = {key: cfg[key]["sha256"] for key in ("binary", "target", "eagle_q4_0", "dspark")}
        report["protocol_sha256"] = protocol_sha
        need(set(cfg["diagnostic_arms"]) == set(ARMS) and cfg.get("verify_positions") == list(range(88, 96)),
             "diagnostic arm/position scope changed")
        gate = load(gate_path)
        need(gate["max_abs_competing_logit_gap"] == gate["max_centered_common_top5_difference"] == 0.05,
             "predeclared numeric threshold changed")
        readiness = load(diagnostic / "readiness.json")
        need(readiness.get("cases") == list(CASES) and readiness.get("output_cap") == 128,
             "diagnostic did not preserve original warmups/prefix case order and cap")
        data, traces, rounds, reproduced = {}, {}, {}, []
        for arm in ARMS:
            launch = load(diagnostic / arm / "launch.json")
            full_launch = load(reference / "rep-00" / arm / "launch.json")
            def without_port(cmd):
                result = list(cmd)
                if "--port" in result:
                    i = result.index("--port")
                    del result[i:i+2]
                return result
            need(without_port(launch["command"]) == without_port(full_launch["command"]), "native launch/sampler parameters changed")
            data[arm] = {}
            for name in CASES:
                folders = (diagnostic / arm / name, reference / "rep-00" / arm / name)
                values = []
                for folder in folders:
                    m, req, resp = (load(folder / f) for f in ("measurement.json", "request.json", "response.json"))
                    ids = m["generated_token_ids"]
                    verbose = resp.get("__verbose", {})
                    termination = {"finish_reason": m.get("finish_reason"), "stop": verbose.get("stop"), "stop_type": verbose.get("stop_type")}
                    need(ids and req.get("max_tokens") == 128 and req.get("temperature") == 0 and req.get("seed") == 42 and
                         req.get("cache_prompt") is False and req.get("chat_template_kwargs", {}).get("enable_thinking") is False,
                         "actual request/ID evidence changed")
                    need(verbose.get("prompt") is not None and termination["stop"] is True, "actual rendered prompt/termination missing")
                    values.append({"ids": ids, "request": req, "prompt": verbose["prompt"], "termination": termination})
                need(values[0] == values[1], f"complete diagnostic/timing reproduction failed: {arm}/{name}")
                data[arm][name] = values[0]
                reproduced.append({"arm": arm, "case": name, "ids_sha256": token_hash(values[0]["ids"]),
                                   "termination": values[0]["termination"]})
            traces[arm] = load(diagnostic / arm / "verify.jsonl", True)
            rounds[arm] = load(diagnostic / arm / "rounds.jsonl", True)
        report["complete_reproduction"] = reproduced
        points, ancestry = [], []
        for arm in ARMS:
            tasks = list(dict.fromkeys(r["task_id"] for r in traces[arm]))
            eligible = [name for name in CASES if len(data[arm][name]["ids"]) > 88]
            need(len(tasks) == len(eligible), "sequential actual request/task map changed")
            task = dict(zip(eligible, tasks))[case]
            for neighbor in range(88, 96):
                actual_point(traces[arm], task, neighbor, data[arm][case]["ids"])
            point = actual_point(traces[arm], task, position, data[arm][case]["ids"])
            points.append(point)
            ancestry.append(reached_prefix(point, rounds[arm], data[arm][case]["ids"]))
        left, right = (data[arm][case] for arm in ARMS)
        need(left["request"] == right["request"] and left["prompt"] == right["prompt"] and
             left["ids"][:position] == right["ids"][:position] and left["ids"][position] != right["ids"][position],
             "actual input/generated prefix is not identical through the selected point")
        first = next((i for i, (a, b) in enumerate(zip(left["ids"], right["ids"])) if a != b), None)
        need(first == position, "selected point is not the first completed-pair divergence")
        pair = [left["ids"][position], right["ids"][position]]
        scores = [{r["token_id"]: r["logit"] for r in point["raw_top5"]} for point in points]
        need(all(set(pair) <= s.keys() for s in scores), "competing IDs missing from raw top5")
        gaps = [s[pair[0]] - s[pair[1]] for s in scores]
        need(all(abs(g) <= 0.05 for g in gaps) and gaps[0]*gaps[1] <= 0, "opposing near-tie gaps exceed gate")
        common = sorted(scores[0].keys() & scores[1].keys())
        means = [sum(s[t] for t in common)/len(common) for s in scores]
        centered = max(abs((scores[0][t]-means[0])-(scores[1][t]-means[1])) for t in common)
        need(centered <= 0.05, "centered common-top5 difference exceeds gate")
        report.update(passed=True, competing_ids=pair, signed_gaps=gaps, centered_common_top5_max=centered,
                      full_input_prefix={"rendered_prompt": left["prompt"], "generated_token_ids": left["ids"][:position]},
                      actual_raw_points=points, causal_retained_prefix=ancestry)
    except (ValueError, KeyError, IndexError, OSError, TypeError) as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", default="prompt-05")
    parser.add_argument("--position", type=int, default=90)
    args = parser.parse_args()
    result = audit(args.diagnostic, args.reference, args.gate, args.case, args.position)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
