"""Summarize actual paired screen artifacts without supplying missing measurements."""

import argparse
import json
from pathlib import Path

from benchmark_dspark_screen import ARMS, round_summary
from benchmark_native_eagle import json_write


def summarize(source: Path) -> dict:
    data = json.loads((source / "measurements.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    measured = [r for r in data["records"] if not r["warmup"]]
    target = {(r["repetition"], r["prompt_id"]): r for r in measured if r["arm"] == "target_only"}
    primary = {(r["repetition"], r["prompt_id"]): r for r in measured if r["arm"] == "eagle_q4_0"}
    expected = protocol["prompt_count"] * protocol["repetitions"]
    summary = {"diagnostic": data["diagnostic"], "inference_s": data["inference_s"], "arms": {}}
    for arm in ARMS:
        items = [r for r in measured if r["arm"] == arm]
        pairs = {(r["repetition"], r["prompt_id"]) for r in items}
        if not data["diagnostic"] and (len(items) != expected or len(pairs) != expected or pairs != set(target)):
            raise ValueError(f"incomplete or duplicate paired measurements: {arm}")
        if any(not isinstance(r["completion_tokens"], int) or not isinstance(r["server_predicted_ms"], (int, float))
               or r["server_predicted_ms"] <= 0 for r in items):
            raise ValueError(f"missing actual output/decode counts: {arm}")
        tokens = sum(r["completion_tokens"] for r in items)
        decode_s = sum(r["server_predicted_ms"] for r in items) / 1000
        wall_s = sum(r["request_wall_s"] for r in items)
        mismatches, primary_mismatches = [], []
        for r in items:
            ref = target.get((r["repetition"], r["prompt_id"]))
            if ref is None or ref["generated_token_ids"] is None or r["generated_token_ids"] is None:
                raise ValueError("missing paired raw output IDs")
            if r["generated_token_ids"] != ref["generated_token_ids"]:
                common = next((i for i, (a, b) in enumerate(zip(ref["generated_token_ids"], r["generated_token_ids"]))
                               if a != b), min(len(ref["generated_token_ids"]), len(r["generated_token_ids"])))
                mismatches.append({"repetition": r["repetition"], "prompt_id": r["prompt_id"], "common_prefix": common})
            q4 = primary.get((r["repetition"], r["prompt_id"]))
            if q4 is None or q4["generated_token_ids"] is None:
                raise ValueError("missing primary Q4 paired raw output IDs")
            if r["generated_token_ids"] != q4["generated_token_ids"]:
                common = next((i for i, (a, b) in enumerate(zip(q4["generated_token_ids"], r["generated_token_ids"]))
                               if a != b), min(len(q4["generated_token_ids"]), len(r["generated_token_ids"])))
                primary_mismatches.append({"repetition": r["repetition"], "prompt_id": r["prompt_id"], "common_prefix": common})
        entry = {"requests": len(items), "output_tokens": tokens, "decode_s": decode_s, "request_s": wall_s,
                 "pooled_decode_tps": tokens / decode_s if decode_s else None,
                 "pooled_request_tps": tokens / wall_s if wall_s else None,
                 "output_id_mismatches": mismatches, "primary_q4_output_id_mismatches": primary_mismatches,
                 "within_arm_variable_prompt_ids": sorted({r["prompt_id"] for r in items
                     if len({tuple(p["generated_token_ids"]) for p in items if p["prompt_id"] == r["prompt_id"]}) > 1}),
                 "repetitions": {}}
        for rep in sorted({r["repetition"] for r in items}):
            cell = [r for r in items if r["repetition"] == rep]
            entry["repetitions"][str(rep)] = {
                "output_tokens": sum(r["completion_tokens"] for r in cell),
                "decode_tps": sum(r["completion_tokens"] for r in cell) * 1000 / sum(r["server_predicted_ms"] for r in cell),
                "request_tps": sum(r["completion_tokens"] for r in cell) / sum(r["request_wall_s"] for r in cell)}
        if arm != "target_only":
            all_rounds = []
            for r in items:
                directory = (source / r["artifact_path"]).resolve()
                if not directory.is_relative_to(source.resolve()):
                    raise ValueError("request artifact escapes run directory")
                meta = json.loads((directory / "measurement.json").read_text())
                if meta.get("server_response_id") != r["server_response_id"]:
                    raise ValueError("raw native request/round join failed")
                all_rounds += json.loads((directory / "rounds.json").read_text())
            maximum = protocol["eagle_length"] if arm == "eagle_q4_0" else int(arm[-1])
            entry["native_rounds"] = round_summary(all_rounds, maximum)
            # Preserve counts at output/EOS boundaries separately: verification can accept beyond the cap.
            entry["counting_caveat"] = "accepted counts are verifier matches; emitted round IDs include terminal tokens; completion usage follows server response"
        summary["arms"][arm] = entry
    baseline = summary["arms"]["eagle_q4_0"]
    for entry in summary["arms"].values():
        entry["decode_ratio_to_q4_0"] = entry["pooled_decode_tps"] / baseline["pooled_decode_tps"]
        entry["request_ratio_to_q4_0"] = entry["pooled_request_tps"] / baseline["pooled_request_tps"]
        for rep, cell in entry["repetitions"].items():
            cell["decode_ratio_to_q4_0"] = cell["decode_tps"] / baseline["repetitions"][rep]["decode_tps"]
            cell["request_ratio_to_q4_0"] = cell["request_tps"] / baseline["repetitions"][rep]["request_tps"]
    summary["timing_scope"] = "concurrency-one pooled output token rates; decode server predicted_ms; request client HTTP wall; no maximum serving-capacity claim"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    json_write(args.source / "summary.json", summarize(args.source))
