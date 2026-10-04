"""Compose two bounded native near-tie edges from preserved CPU-readable data.

No model is loaded. Runtime hashes come from the already checked run configs;
the admission validator independently checks binary/target/model pins as usual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from check_export import sha256

CASES = ("warmup-00", "warmup-01", "prompt-00", "prompt-01", "eos")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def produce(inputs):
    need(1 <= len(inputs["edges"]) <= 2, "numeric receipt needs one or two scoped edges")
    allowed_edges = [("target_only", "eagle_q4_0", 89), ("eagle_q4_0", "dspark_3", 94)]
    need([(edge["left"], edge["right"]) for edge in inputs["edges"]] ==
         [(a, b) for a, b, _ in allowed_edges[:len(inputs["edges"])]], "unapproved numeric edge scope")
    evidence = {}
    def json_file(path, lines=False):
        path = Path(path).resolve()
        evidence[str(path)] = sha256(path)
        return [json.loads(line) for line in path.read_text().splitlines() if line] if lines else read(path)
    manifest_path = Path(inputs["manifest"]).resolve()
    manifest = json_file(manifest_path)
    probe_cfg = json_file(manifest_path.parent / "config.json")
    protocol_path = Path(inputs["protocol"]).resolve()
    json_file(protocol_path)
    protocol_sha = evidence[str(protocol_path)]
    runtime = {key: probe_cfg[key]["sha256"] for key in ("binary", "target")}
    def run_config(root):
        cfg = json_file(Path(root) / "config.json")
        need(all(cfg[key]["sha256"] == runtime[key] for key in runtime), "numeric runtime pins changed")
        need(cfg.get("protocol_sha256", cfg.get("protocol", {}).get("sha256")) == protocol_sha,
             "numeric protocol pin changed")
        return cfg
    run_config(manifest_path.parent)
    primary = Path(inputs["primary"])
    primary_cfg = run_config(primary)
    def cases(root, arm):
        result = {}
        for name in CASES:
            folder = Path(root) / arm / name
            measurement = json_file(folder / "measurement.json")
            request = json_file(folder / "request.json")
            response = json_file(folder / "response.json")
            ids = measurement["generated_token_ids"]
            need(isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids), "missing numeric raw IDs")
            need(request.get("temperature") == 0 and request.get("seed") == 42 and
                 request.get("max_tokens") == 128 and request.get("cache_prompt") is False and
                 request.get("chat_template_kwargs", {}).get("enable_thinking") is False,
                 "numeric sampler/request contract changed")
            prompt = response.get("__verbose", {}).get("prompt")
            need(prompt is not None, "missing actual rendered prompt for full-prefix join")
            termination = {"finish_reason": measurement.get("finish_reason"),
                           "stop": response.get("__verbose", {}).get("stop"),
                           "stop_type": response.get("__verbose", {}).get("stop_type")}
            result[name] = {"ids": ids, "request": request, "prompt": prompt, "termination": termination,
                            "measurement": str((folder / "measurement.json").resolve())}
        for a, b in (("warmup-00", "prompt-00"), ("warmup-01", "prompt-01")):
            need(result[a]["request"] == result[b]["request"] and result[a]["ids"] == result[b]["ids"],
                 "numeric within-arm full-request repeatability failed")
        return result
    nodes = {"target_only": cases(primary, "target_only")}
    primary_q4 = cases(primary, "eagle_q4_0")
    edges, q4_reference_ancestry = [], {}
    reference_assets = {}
    for key in ("dspark", "dflash"):
        model = probe_cfg[key]
        reference_assets[key] = model
        if "reference" not in model or "precision" not in model:
            continue
        from validate_native import check_q4_precision
        precision = json_file(model["precision"]["path"])
        export = json_file(model["export"]["path"])
        need(evidence[str(Path(model["precision"]["path"]).resolve())] == model["precision"]["sha256"] and
             evidence[str(Path(model["export"]["path"]).resolve())] == model["export"]["sha256"],
             "numeric Q4 precision/export receipt changed")
        check_q4_precision(export, precision, model["sha256"], model["export"]["sha256"])
        need(model["reference"]["sha256"] == export["draft_sha256"], "numeric Q4 reference pin changed")
        reference_assets[key] = model["reference"]
        q4_reference_ancestry[key] = {"candidate_sha256": model["sha256"],
            "reference_sha256": model["reference"]["sha256"], "precision_sha256": model["precision"]["sha256"],
            "scope": "exact observed output+termination path only; no candidate Q4 raw-logit margin claim"}
    reached = {name: {"target_only"} for name in CASES}

    def raw_point(trace, task, position, ids):
        found = [r for r in trace if r["task_id"] == task and r["generated_position"] == position and
                 r.get("sampled") and r.get("emitted") and not r.get("replay")]
        need(len(found) == 1, "numeric point is not one actual emitted nonreplay draw")
        row = found[0]
        need(row.get("has_logits") and row.get("nan_logit_count") == 0 and
             row["sampled_token_id"] == row["emitted_token_id"] == ids[position], "numeric sampled/raw ID join failed")
        top = row["raw_top5"]
        need(len(top) == 5 and all(isinstance(t.get("logit"), (int, float)) and math.isfinite(t["logit"]) for t in top),
             "numeric raw top5 is nonfinite/incomplete")
        need(top[0]["token_id"] == ids[position] and all(top[i]["logit"] >= top[i+1]["logit"] for i in range(4)),
             "actual sampled ID differs from raw argmax")
        return row

    def causal(row, rounds, ids):
        if row["mode"] == "target_only":
            need(row["row"] == 0, "target-only numeric row is not causal row0")
            return {"mode": "target_only"}
        actual = [r for r in rounds if r["task_id"] == row["task_id"] and
                  r["status"] in ("complete", "no_proposal") and not r.get("replay")]
        actual.sort(key=lambda r: r["round_index"])
        emitted = [token for r in actual for token in r["emitted_token_ids"]]
        offset = 0 if emitted == ids else 1 if emitted == ids[1:] else None
        need(offset is not None, "causal round stream does not reproduce complete output IDs")
        cursor = offset
        matches = []
        for round_row in actual:
            if cursor + row["row"] == row["generated_position"] and row["row"] < len(round_row["emitted_token_ids"]):
                matches.append((cursor, round_row))
            cursor += len(round_row["emitted_token_ids"])
        need(len(matches) == 1, "numeric causal retained-prefix round is ambiguous")
        base, round_row = matches[0]
        j = row["row"]
        need(row.get("base_generated_position") == base and round_row["round_index"] + 1 == row["round_index"] and
             round_row["n_accepted"] >= j and
             round_row["proposed_token_ids"][:j] == ids[base:base+j] == row["verifier_prefix_draft_ids"] and
             round_row["verified_token_ids"][j] == round_row["emitted_token_ids"][j] == ids[base+j],
             "numeric row contains an unreached/rejected-prefix input")
        return {"task_id": row["task_id"], "round_index": round_row["round_index"], "row": j,
                "n_accepted": round_row["n_accepted"], "base_generated_position": base,
                "prior_draft_prefix": row["verifier_prefix_draft_ids"]}

    for edge_index, spec in enumerate(inputs["edges"]):
        root, left, right = Path(spec["results"]), spec["left"], spec["right"]
        need(left in nodes, "numeric edge is disconnected from the frozen target")
        cfg = run_config(root)
        for arm in (left, right):
            if arm != "target_only":
                key = "eagle_q4_0" if arm == "eagle_q4_0" else arm.rsplit("_", 1)[0]
                expected = primary_cfg[key] if key == "eagle_q4_0" else reference_assets[key]
                need(cfg[key]["sha256"] == expected["sha256"], "numeric model source pin changed")
            launch = json_file(root / arm / "launch.json")
            cmd = launch["command"]
            need(cmd[0] == cfg["binary"]["path"] and cmd[cmd.index("-m")+1] == cfg["target"]["path"] and
                 "--backend-sampling" not in cmd and "-bs" not in cmd, "numeric target/sampler launch changed")
        gate = json_file(spec["gate"])
        need(gate["max_abs_competing_logit_gap"] == gate["max_centered_common_top5_difference"] == 0.05,
             "numeric predeclared threshold changed")
        position = allowed_edges[edge_index][2]
        positions = cfg.get("verify_positions", gate["positions"])
        need(position in positions, "scoped numeric point was not captured")
        data = {arm: cases(root, arm) for arm in (left, right)}
        traces = {arm: json_file(root / arm / "verify.jsonl", True) for arm in (left, right)}
        round_data = {arm: json_file(root / arm / "rounds.jsonl", True) if arm != "target_only" else [] for arm in (left, right)}
        task_map = {}
        for arm in (left, right):
            tasks = list(dict.fromkeys(r["task_id"] for r in traces[arm]))
            names = [name for name in CASES if len(data[arm][name]["ids"]) > min(positions)]
            need(len(tasks) == len(names), "numeric sequential request/task map changed")
            task_map[arm] = dict(zip(names, tasks))
        edge_cases = {}
        for name in CASES:
            a, b = data[left][name], data[right][name]
            need(a["request"] == b["request"] == nodes[left][name]["request"] and
                 a["prompt"] == b["prompt"] == nodes[left][name]["prompt"] and
                 a["ids"] == nodes[left][name]["ids"], "numeric diagnostic changed actual input/known output path")
            if right == "eagle_q4_0":
                need(b["ids"] == primary_q4[name]["ids"], "diagnostic Q4 differs from PRIMARY Q4 complete IDs")
                need(b["termination"] == primary_q4[name]["termination"], "diagnostic PRIMARY Q4 termination changed")
            if a["ids"] == b["ids"]:
                edge_cases[name] = {"exact": True}
                if left in reached[name]: reached[name].add(right)
                continue
            differences = [i for i in range(min(len(a["ids"]), len(b["ids"]))) if a["ids"][i] != b["ids"][i]]
            need(differences and differences[0] == position and a["ids"][:position] == b["ids"][:position],
                 "uncovered numeric first divergence/shared prefix")
            pair = gate["competing_token_ids"] if edge_index == 0 else [a["ids"][position], b["ids"][position]]
            need(set(pair) == {a["ids"][position], b["ids"][position]}, "numeric competing IDs changed")
            points = [raw_point(traces[arm], task_map[arm][name], position, data[arm][name]["ids"]) for arm in (left, right)]
            scores = [{t["token_id"]: t["logit"] for t in point["raw_top5"]} for point in points]
            need(all(set(pair) <= s.keys() for s in scores), "competing IDs missing from numeric top5")
            gaps = [s[pair[0]] - s[pair[1]] for s in scores]
            need(all(abs(g) <= 0.05 for g in gaps) and gaps[0]*gaps[1] <= 0, "numeric competing margin exceeds gate")
            common = sorted(scores[0].keys() & scores[1].keys())
            means = [sum(s[t] for t in common)/len(common) for s in scores]
            centered = max(abs((scores[0][t]-means[0])-(scores[1][t]-means[1])) for t in common)
            need(centered <= 0.05, "centered common-top5 numeric difference exceeds gate")
            ancestry = [causal(point, round_data[arm], data[arm][name]["ids"]) for arm, point in zip((left, right), points)]
            if left == "target_only" and right == "eagle_q4_0":
                need(ancestry[1]["row"] == 0 and ancestry[1]["n_accepted"] == 0 and points[1]["status"] == "rejected",
                     "PRIMARY Q4 control is not a zero-acceptance target correction")
            edge_cases[name] = {"exact": False, "first_difference": position,
                "full_prefix": {"actual_rendered_prompt": a["prompt"], "generated_token_ids": a["ids"][:position]},
                "competing_ids": pair, "signed_gaps": gaps, "centered_common_top5_max": centered,
                "raw_points": points, "causal_retained_prefix": ancestry,
                "suffix_status": "scoped complete native-path sequence; downstream cascade recorded, not bit parity"}
            if left in reached[name]: reached[name].add(right)
        nodes[right] = data[right]
        edges.append({"left": left, "right": right, "cases": edge_cases})
    covered, uncovered = [], []
    for cell in manifest["cells"]:
        for output in cell["outputs"]:
            path = Path(output["actual"])
            name = path.parent.name
            need(name in CASES, "numeric receipt contains an unfrozen case")
            actual, reference = json_file(path), json_file(output["reference"])
            response = json_file(path.parent / "response.json") if q4_reference_ancestry else None
            request = json_file(path.parent / "request.json")
            need(request == nodes["target_only"][name]["request"] and reference["generated_token_ids"] == nodes["target_only"][name]["ids"],
                 "original probe no longer reproduces numeric target/request")
            termination = {"finish_reason": actual.get("finish_reason"),
                           "stop": response.get("__verbose", {}).get("stop"),
                           "stop_type": response.get("__verbose", {}).get("stop_type")} if response else None
            if q4_reference_ancestry:
                need(termination["finish_reason"] in ("length", "stop") and termination["stop"] is True and
                     termination["stop_type"] in ("limit", "eos", "word"), "numeric Q4 termination evidence missing")
            compatible = [arm for arm in reached[name] if actual["generated_token_ids"] == nodes[arm][name]["ids"] and
                          (not q4_reference_ancestry or termination == nodes[arm][name]["termination"])]
            row = {"actual_measurement": str(path.resolve()), "actual_sha256": evidence[str(path.resolve())],
                   "reference_measurement": str(Path(output["reference"]).resolve()),
                   "reference_sha256": evidence[str(Path(output["reference"]).resolve())], "request_sha256": digest(request),
                   "actual_ids_sha256": digest(actual["generated_token_ids"]),
                   "reference_ids_sha256": digest(reference["generated_token_ids"]),
                   "kind": cell["kind"], "maximum": cell["maximum"], "case": name}
            if q4_reference_ancestry:
                row["termination"] = termination
            if compatible:
                covered.append({**row, "validated_native_paths": sorted(compatible)})
            else:
                uncovered.append(row)
    result = {"schema": "dspark_scoped_numeric_receipt_v1", "passed": True,
            "scope": "bounded numeric edges only; structural/native admission remains separate",
            "producer_inputs": inputs, "binary_sha256": runtime["binary"], "target_sha256": runtime["target"],
            "model_sha256": {"eagle_q4_0": primary_cfg["eagle_q4_0"]["sha256"]},
            "protocol_sha256": protocol_sha, "manifest_semantic_sha256": digest({k: v for k, v in manifest.items() if k != "numeric_gate"}),
            "edges": edges, "covered_outputs": covered, "uncovered_outputs": uncovered,
            "evidence_sha256": evidence, "correctness_status": "native near-tie sensitivity; not exact greedy parity or confirmed batch cause"}
    if q4_reference_ancestry:
        result["q4_reference_ancestry"] = q4_reference_ancestry
    return result


def consume(asset, manifest, binary_sha, target_sha):
    need(sha256(Path(asset["path"])) == asset["sha256"], "numeric receipt changed")
    receipt = read(asset["path"])
    need(receipt == produce(receipt["producer_inputs"]), "numeric receipt does not reproduce raw evidence")
    need(receipt["binary_sha256"] == binary_sha and receipt["target_sha256"] == target_sha and
         receipt["manifest_semantic_sha256"] == digest({k: v for k, v in manifest.items() if k != "numeric_gate"}),
         "numeric receipt scope/runtime changed")
    return receipt


def covered_output(receipt, actual_path, reference_path, request, actual_ids, reference_ids):
    return any(row["actual_sha256"] == sha256(Path(actual_path)) and row["reference_sha256"] == sha256(Path(reference_path)) and
               row["request_sha256"] == digest(request) and row["actual_ids_sha256"] == digest(actual_ids) and
               row["reference_ids_sha256"] == digest(reference_ids) for row in receipt["covered_outputs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(produce(read(args.inputs)), indent=2) + "\n")
