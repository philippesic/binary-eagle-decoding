"""Separate diagnostic CUDA node costs; never a throughput estimator."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmark_native_eagle import json_write, sha256


def union(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("negative CUDA event interval")
        total += right-left if end is None or left >= end else max(0, right-end)
        end = right if end is None else max(right, end)
    return total


def analyze(path: Path):
    events = []
    truncation = False
    with path.open(errors="replace") as log:
        for line in log:
            if "CUDA_EAGLE_EVENT " not in line:
                continue
            event = json.loads(line.split("CUDA_EAGLE_EVENT ", 1)[1])
            if event["kind"] == "truncation":
                truncation = True
            elif event["kind"] == "node" and event.get("model_arch") == "dflash":
                events.append(event)
    if truncation:
        raise ValueError("CUDA event inventory truncated")
    if not events:
        raise ValueError("no actual CUDA DFlash/DSpark nodes")
    if any(e.get("cuda_ms") is None or e.get("captured_inventory_only") for e in events):
        raise ValueError("captured graph inventory is not measured node time")
    frames = defaultdict(list)
    for e in events:
        frames[(e["llama_context"], e["context"], e["stream"], e["frame"])].append(e)
    groups = defaultdict(list)
    for key, frame in frames.items():
        head = [e["node"] for e in frame if e["tensor"] == "result_output" and e["op"] == "MUL_MAT"]
        if len(head) > 1:
            raise ValueError("ambiguous full-head boundary in graph frame")
        for e in frame:
            if e["n_outputs"] == 0:
                category = "feature_fusion" if e["tensor"] == "dspark_feature_fusion" else "context_injection_other"
            elif e["tensor"] == "result_output" and e["op"] == "MUL_MAT":
                category = "full_head"
            elif e["tensor"].startswith("dspark_confidence_"):
                category = "confidence_projection"
            elif e["tensor"].startswith("dspark_markov_") or head and e["node"] > head[0]:
                category = "markov_and_output_assembly"
            else:
                category = "draft_body"
            groups[key, category].append((e["gpu_begin_ms"], e["gpu_end_ms"]))
    costs = defaultdict(lambda: {"node_intervals": 0, "cuda_ms_union": 0.0})
    for (_key, category), spans in groups.items():
        costs[category]["node_intervals"] += len(spans)
        costs[category]["cuda_ms_union"] += union(spans)
    return {"schema": "dspark_cuda_components_v1", "source": str(path.resolve()),
            "source_sha256": sha256(path), "draft_contexts": sorted({e["llama_context"] for e in events}),
            "graph_frames": len(frames), "costs": dict(costs),
            "scope": "separate synchronized CUDA-event node intervals, including stream idle; load/warmup/request stages all retained in raw log",
            "aggregation": "union within each context/stream/frame/category; graph parents and transfer grandchildren excluded",
            "limitations": "perturbed diagnostic execution; not throughput or uninstrumented kernel latency; frame scopes must be joined to request windows before per-request claims"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    json_write(args.output, analyze(args.log))
