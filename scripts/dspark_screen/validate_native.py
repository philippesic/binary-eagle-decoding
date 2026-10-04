"""Validate source-bound native admission evidence; never synthesize pass flags.

The manifest format is documented in README.md. Actual inference is owned by
the sole GPU operator and runs under remote_job.py; this validator is CPU-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_export import sha256


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def initial_setup_masks(state):
    """Recognize only the exact observed pre-injection, unframed setup pair.

    This does not classify it outside target binding: the actual completed
    probe records it immediately after binding_begin. Callers retain the raw
    records and account for the recognized initialization rows separately.
    """
    begin = next((i for i, r in enumerate(state) if r["event"] == "binding_begin"), None)
    require(begin == 0, "unknown records before binding begin")
    require(state[-1]["event"] == "binding_end", "unknown records after binding end")
    prefix = []
    index = 1
    while index < len(state) and state[index]["event"] == "mask":
        prefix.append(state[index])
        index += 1
    if not prefix:
        return state, 0
    expected = [{"schema": "dspark_admission_v1", "event": "mask", "seq_id": 0,
                 "query_position": query, "anchor_position": 0, "max_visible_clean_position": -1,
                 "visible_noise_positions": [0, 1]} for query in (0, 1)]
    require(prefix == expected, "unrecognized initial setup mask records")
    # Neither setup row may be carried as stale noise: actual target feature
    # injection must overwrite both positions before the first noise event.
    initial = []
    first_noise = None
    for row in state[index:]:
        if row["event"] == "noise":
            first_noise = row
            break
        require(row["event"] == "inject", "unknown event between setup masks and actual injection")
        initial.append(row)
    require(first_noise and first_noise["anchor_position"] >= 2 and
            any(row["first_position"] == 0 and row["last_position"] >= 1 and row["rc"] == 0 for row in initial),
            "initial setup mask rows were not overwritten by target features")
    return [state[0], *state[index:]], 2


def check_state(state, rounds, maximum):
    raw_state = state
    starts = [r for r in state if r["event"] == "binding_begin"]
    ends = [r for r in state if r["event"] == "binding_end"]
    require(len(starts) == len(ends) == 1, "missing complete tensor binding lifetime")
    for key in ("target_embedding_identity", "target_head_identity", "embedding_hash_fnv1a64",
                "head_hash_fnv1a64", "target_embedding_bytes", "target_head_bytes",
                "target_embedding_dtype", "target_head_dtype", "borrows_embedding", "borrows_head",
                "draft_n_batch", "draft_n_ubatch", "draft_n_outputs_max", "draft_n_outputs_max_per_seq", "draft_backend_sampling"):
        require(starts[0][key] == ends[0][key], f"target binding changed: {key}")
    require(starts[0]["target_embedding_dtype"] == starts[0]["target_head_dtype"] == "f16",
            "fixed target precision changed")
    require(starts[0]["draft_n_batch"] >= 7 and starts[0]["draft_n_ubatch"] >= 7 and
            starts[0]["draft_n_outputs_max"] >= 7 and
            (not starts[0]["draft_backend_sampling"] or starts[0]["draft_n_outputs_max_per_seq"] >= 7),
            "draft batch/output capacity cannot cover seven computed noise rows")
    state, setup_count = initial_setup_masks(state)
    noise = [r for r in state if r["event"] == "noise"]
    injections = [r for r in state if r["event"] == "inject"]
    require(noise and injections, "no actual noise/injection evidence")
    for r in noise:
        require(r["n_noise_tokens"] == 7 and r["first_read_slot"] == 0 and
                r["sample_from_anchor"] and r["all_noise_rows_bidirectional"] and
                r["n_proposal_requested"] == maximum and r["mask_token_id"] == 151669,
                "wrong author noise/read-slot convention")
        require(r["kv_max_before"] < r["anchor_position"], "stale noise or pending anchor in draft cache")
        require(r["anchor_position"] == len(r["prefix_token_ids"]), "prefix/absolute-position ancestry differs")
    masks = [r for r in state if r["event"] == "mask"]
    require(len(masks) == 7 * len(noise), "missing actual seven-row attention masks")
    for index, n in enumerate(noise):
        block = masks[7*index:7*(index + 1)]
        expected = list(range(n["anchor_position"], n["anchor_position"] + 7))
        require(sorted(r["query_position"] for r in block) == expected, "actual noise query rows differ")
        for r in block:
            require(r["anchor_position"] == n["anchor_position"] and
                    r["visible_noise_positions"] == expected and
                    r["max_visible_clean_position"] < n["anchor_position"], "actual attention mask violates author block visibility")
    for r in injections:
        require(r["rc"] == 0 and r["target_taps"] == [2, 10, 18, 26, 34] and r["feature_hash_fnv1a64"],
                "failed or wrong target feature injection")
        require(r["last_position"] - r["first_position"] + 1 == r["n_tokens"], "noncontiguous injection")
    prior_noise = None
    pending_injections = []
    for r in state:
        if r["event"] == "inject":
            pending_injections.append(r)
        if r["event"] != "noise":
            continue
        if prior_noise and len(r["prefix_token_ids"]) > len(prior_noise["prefix_token_ids"]) and r["prefix_token_ids"][:len(prior_noise["prefix_token_ids"])] == prior_noise["prefix_token_ids"]:
            require(any(i["first_position"] <= prior_noise["anchor_position"] and
                        i["last_position"] >= r["anchor_position"] - 1 for i in pending_injections),
                    "retained target ancestry was not reinjected before next block")
        prior_noise = r
        pending_injections = []
    complete = [r for r in rounds if r["status"] == "complete" and not r.get("replay")]
    no_noise_boundary = [r for r in complete if r.get("n_draft_max") == 0 and r["n_proposed"] == 0]
    for r in no_noise_boundary:
        require(r["n_accepted"] == 0 and r["proposed_token_ids"] == [] and
                r["n_emitted"] == len(r["emitted_token_ids"]), "invalid no-noise output-cap boundary")
    complete = [r for r in complete if r not in no_noise_boundary]
    require(len(complete) == len(noise), "missing complete noise/verification round join")
    require(any(r["n_proposed"] == maximum for r in complete), "no native proposal reached configured maximum")
    proposal_map = {}
    for n, r in zip(noise, complete):
        proposed = r["proposed_token_ids"]
        verified = r["verified_token_ids"]
        emitted = r["emitted_token_ids"]
        accepted = r["n_accepted"]
        require(0 <= accepted <= len(proposed) and 0 < len(proposed) <= maximum and len(proposed) == r["n_proposed"], "bad or empty proposal counts")
        require(proposed[:accepted] == verified[:accepted] and len(verified) == accepted + 1,
                "not a greedy matched-prefix plus correction/bonus round")
        require(emitted == verified[:len(emitted)] and len(emitted) == r["n_emitted"], "bad emitted prefix")
        require(r["n_accepted_usable_prefix"] == min(accepted, len(emitted)), "bad EOS/cap consumed-prefix count")
        if 151645 in emitted:
            require(emitted[-1] == 151645, "tokens emitted after EOS")
        key = (tuple(n["prefix_token_ids"]), n["anchor_token_id"])
        proposal_map[key] = proposed
    return proposal_map, {"rounds": len(complete), "injections": len(injections),
                          "initial_unframed_setup_masks": setup_count,
                          "initial_setup_mask_records": raw_state[1:3] if setup_count else [],
                          "initial_setup_scope": "inside binding lifetime; exact pre-injection pair overwritten before real noise",
                          "no_noise_output_cap_boundaries": len(no_noise_boundary),
                          "zero_acceptance": sum(r["n_accepted"] == 0 for r in complete),
                          "partial_acceptance": sum(0 < r["n_accepted"] < r["n_proposed"] for r in complete),
                          "full_acceptance": sum(r["n_accepted"] == r["n_proposed"] for r in complete),
                          "eos_rounds": sum(151645 in r["emitted_token_ids"] for r in complete),
                          "boundary_caveat": "only observed trajectories claimed; unsampled EOS/acceptance branches remain source/fixture evidence"}


def validate(manifest):
    target_sha = sha256(Path(manifest["target"]))
    require(sha256(Path(manifest["binary"])) == manifest["binary_sha256"], "native binary changed")
    env = read(manifest["environment"])
    capability = env["gpu_capability_query"]
    require(capability.get("exit_code") == 0 and "RTX 2080 Ti" in capability["stdout"] and
            "7.5" in capability["stdout"], "actual RTX2080Ti/SM75 evidence missing")
    maps, details = {}, []
    pins = {"target": target_sha, "binary": manifest["binary_sha256"],
            "environment": sha256(Path(manifest["environment"]))}
    for cell in manifest["cells"]:
        kind, maximum = cell["kind"], cell["maximum"]
        require(kind in ("dspark", "dflash") and maximum in (3, 7), "unfrozen admission arm")
        export = read(cell["export"])
        require(export["passed"] and export["target_sha256"] == target_sha and
                export["draft_sha256"] == sha256(Path(cell["draft"])) and
                export["source_sha256"] == sha256(Path(cell["source"])) and
                export["config_sha256"] == sha256(Path(cell["conversion_config"])) and
                export["canonical_comparison_sha256"] == sha256(Path(export["canonical_comparison_path"])), "export ancestry changed")
        launch = read(cell["launch"])
        command, trace_env = launch["command"], launch["trace_env"]
        require(Path(command[0]).resolve() == Path(manifest["binary"]).resolve() and
                Path(command[command.index("-m") + 1]).resolve() == Path(manifest["target"]).resolve() and
                Path(command[command.index("-md") + 1]).resolve() == Path(cell["draft"]).resolve(), "launch/model path changed")
        for flag, value in (("--spec-draft-n-max", str(maximum)), ("--batch-size", "32"), ("--ubatch-size", "32"),
                            ("--n-gpu-layers", "all"),
                            ("--spec-draft-ngl", "all"), ("--fit", "off"), ("--parallel", "1"),
                            ("--cache-type-k", "f16"), ("--cache-type-v", "f16"),
                            ("--spec-draft-type-k", "f16"), ("--spec-draft-type-v", "f16")):
            require(flag in command and command[command.index(flag) + 1] == value, f"launch setting changed: {flag}")
        require(command[command.index("--ctx-size") + 1] == "2048" and
                command[command.index("--spec-type") + 1] == "draft-dspark" and
                command[command.index("--spec-draft-p-min") + 1] == "0" and
                trace_env.get("DSPARK_REQUIRE_AUTHOR_LAYOUT") == "1", "launch convention changed")
        log = Path(cell["server_log"]).read_text(errors="replace")
        require("CUDA0" in log and "sample_from_anchor=true" in log and "block_size=7" in log,
                "actual CUDA/layout load evidence missing")
        # Successful model load plus actual native requests is the residency gate.
        # Keep peak/free memory numbers and actual dispatch in the operator report.
        require("failed to load" not in log and "out of memory" not in log.lower(), "model admission failed")
        require(not any(error in log for error in ("backend sampling supports at most", "llama_decode returned",
                                                   "llama_decode(ctx_dft) failed", "CUDA error:")),
                "native draft decode/sampler did not complete successfully")
        state = rows(cell["state"])
        proposal_map, summary = check_state(state, rows(cell["rounds"]), maximum)
        start = next(r for r in state if r["event"] == "binding_begin")
        if export.get("target_tied_head_fallback"):
            require(export["target_head_source_tensor"] == "token_embd.weight" and
                    sha256(Path(export["target_config_path"])) == export["target_config_sha256"] and
                    read(export["target_config_path"]).get("tie_word_embeddings") is True,
                    "target tied-head configuration proof changed")
            for embedding, head in (("embedding_hash_fnv1a64", "head_hash_fnv1a64"),
                                    ("target_embedding_bytes", "target_head_bytes"),
                                    ("target_embedding_dtype", "target_head_dtype")):
                require(start[embedding] == start[head], "tied target roles differ in native source bytes/type/size")
        require(start["borrows_embedding"] == export["borrows_embedding"] and
                start["borrows_head"] == export["borrows_head"], "native/private tensor ownership differs from export")
        for pair in cell["outputs"]:
            actual, reference = read(pair["actual"]), read(pair["reference"])
            require(actual["generated_token_ids"] is not None and
                    actual["generated_token_ids"] == reference["generated_token_ids"], "target-only token mismatch")
            actual_request_path = Path(pair["actual"]).parent / "request.json"
            reference_request_path = Path(pair["reference"]).parent / "request.json"
            actual_request, reference_request = read(actual_request_path), read(reference_request_path)
            require(actual_request == reference_request == read(pair["prompt"]), "actual/reference request identity differs")
            require(actual_request.get("temperature") == 0 and actual_request.get("seed") == 42 and
                    actual_request.get("chat_template_kwargs", {}).get("enable_thinking") is False and
                    actual_request.get("reasoning_format") == "none" and
                    actual_request.get("cache_prompt") is False and
                    actual_request.get("max_tokens") == 128, "request greedy/nonthinking/cap contract changed")
            require(pair["prompt_sha256"] == sha256(Path(pair["prompt"])), "prompt changed")
            pins[f"{kind}_{maximum}_request_{len(pins)}"] = {"actual": sha256(actual_request_path),
                    "reference": sha256(reference_request_path), "actual_measurement": sha256(Path(pair["actual"])),
                    "reference_measurement": sha256(Path(pair["reference"]))}
        require(cell["outputs"], "no actual native output comparison")
        maps[kind, maximum] = proposal_map
        details.append({"kind": kind, "maximum": maximum, **summary})
        for key in ("source", "conversion_config", "export", "state", "rounds", "launch", "server_log"):
            pins[f"{kind}_{maximum}_{key}"] = sha256(Path(cell[key]))
    for kind in ("dspark", "dflash"):
        require((kind, 3) in maps and (kind, 7) in maps, "both valid proposal lengths required")
        common = maps[kind, 3].keys() & maps[kind, 7].keys()
        require(common, "no same-prefix short/max native join")
        for key in common:
            require(maps[kind, 3][key][:3] == maps[kind, 7][key][:3], "short proposal changes author first-three logits/decisions")
    return {"schema": "dspark_native_admission_v1", "passed": True, "target_sha256": target_sha,
            "memory": True, "anchor_first": True, "target_immutable": True,
            "cache_contract": True, "greedy_semantics": True, "evidence_sha256": pins,
            "cells": details, "limitations": "bounded actual trajectories; no bit-exact HF parity or serving-capacity claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(validate(read(a.manifest)), indent=2) + "\n")
