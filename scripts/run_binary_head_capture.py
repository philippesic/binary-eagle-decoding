#!/usr/bin/env python3
"""Supervised native head capture/factorial runner; heavy diagnostics, never timing.

Run under scripts/remote_job.py. Factorial records Q4_0 canonical rounds before
replaying q4_q4,d_d,q4_d,d_q4,d_f16. Train captures unchanged D own histories.
Diagnostic permits up to three historical prompts and the same five-arm gate.
Raw captures retain every state; prepare_frozen_d_head_data.py selects at most
17 states per prompt/depth (8,160 total) without opening development data.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from analyze_binary_head_factorial import ARMS, analyze, audit_rows
from capture_w1ax_activations import (
    generated_token_ids,
    handle_stop_signal,
    http_json,
    load_prompts,
    read_jsonl,
    sha256,
    stop_server,
    wait_ready,
)
from run_binary_scale_screen import quality

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def validate_inputs(args, prompts: list[dict], variants: dict) -> None:
    if not 1 <= args.tokens <= 128:
        raise ValueError("capture output token cap must be 1..128")
    if args.mode != "diagnostic" and args.tokens != 128:
        raise ValueError("frozen primary capture requires128 output cap")
    if "final" in str(args.prompts).lower() or any(
            "final" in p["id"].lower() for p in prompts):
        raise ValueError("reserved-final prompts prohibited")
    if sys.byteorder != "little":
        raise ValueError("training preparation currently requires little-endian native captures")
    if args.mode == "diagnostic":
        if len(prompts) > 3 or any(p["id"].startswith("qat-") for p in prompts):
            raise ValueError("diagnostic requires at most3 historical prompts")
    else:
        split, expected = ("train", 96) if args.mode == "train" else ("development", 24)
        if len(prompts) != expected or any(
                not p["id"].startswith(f"qat-revisit-{split}-") for p in prompts):
            raise ValueError(f"expected exactly{expected} frozen {split} prompts")
        if args.prompts_sha256:
            if sha256(args.prompts) != args.prompts_sha256:
                raise ValueError("frozen prompt hash/count mismatch")
        else:
            frozen = args.prompt_manifest or args.prompts.parent/"manifest.json"
            spec = json.loads(frozen.read_text())["splits"][split]
            if spec["sha256"] != sha256(args.prompts) or spec["prompts"] != expected:
                raise ValueError("frozen prompt manifest hash/count mismatch")
    required = {"d_d"} if args.mode == "train" else set(ARMS)
    if set(variants) != required:
        raise ValueError(f"expected variants {sorted(required)}")
    for name, spec in variants.items():
        if (spec.get("body"), spec.get("head")) != ARMS[name] or not spec.get("draft"):
            raise ValueError(f"incorrect body/head identity for {name}")
        for key in spec.get("env", {}):
            if key in ("EAGLE_CAPTURE_FULL_LOGITS", "EAGLE_CAPTURE_FULL_LOGITS_LIMIT"):
                continue
            if key.startswith(("EAGLE_", "W1AX_")) or key in (
                    "CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS"):
                raise ValueError(f"variant cannot override capture control {key}")
    if args.mode == "train" and (not args.d2t or not args.target_vocab_size):
        raise ValueError("train requires --d2t and --target-vocab-size")


def server_command(args, spec):
    return [str(args.binary.resolve()), "-m", str(args.target.resolve()),
            "--n-gpu-layers", "all", "--ctx-size", "2048", "--parallel", "1",
            "--fit", "off", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--jinja", "--metrics", "--perf", "-lv", "4", "--host", "127.0.0.1",
            "--port", str(args.port), "-md", str(Path(spec["draft"]).resolve()),
            "--spec-type", "draft-eagle3", "--spec-draft-n-max", "5",
            "--spec-draft-p-min", "0", "--spec-draft-ngl", "all",
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
            "--no-spec-draft-backend-sampling"]


def run_cell(args, name, spec, prompts, forced=None):
    cell = args.output.resolve()/name
    cell.mkdir()
    trace = cell/"rounds.jsonl"
    rows_path = cell/"heads.jsonl"
    rounds_path = cell/"forced-rounds.jsonl"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GGML_W1", "GGML_EAGLE", "W1AX_", "EAGLE_"))}
    env.update(spec.get("env", {}))
    env.update({"CUDA_VISIBLE_DEVICES": "0", "GGML_CUDA_DISABLE_GRAPHS": "1",
                "EAGLE_CAPTURE_PREFIX": str(cell/"heads"),
                "EAGLE_RECORD_ROUNDS_JSONL": str(rounds_path),
                "EAGLE_STATE_TRACE_JSONL": str(cell/"state.jsonl"),
                "EAGLE_REQUEST_DIGEST": "1", "W1AX_ROUND_TRACE_JSONL": str(trace)})
    if forced:
        env["EAGLE_FORCE_ROUNDS_JSONL"] = str(forced)
    if name == "d_f16":
        env.setdefault("EAGLE_CAPTURE_FULL_LOGITS", "1")
        env.setdefault("EAGLE_CAPTURE_FULL_LOGITS_LIMIT", "32")
    cmd = server_command(args, spec)
    manifest = {"schema": "binary_head_capture_cell_v1", "name": name, "spec": spec,
                "command": cmd, "env": {k: v for k, v in env.items()
                                         if k.startswith(("GGML_", "EAGLE_", "W1AX_", "CUDA_"))},
                "binary_sha256": sha256(args.binary), "target_sha256": sha256(args.target),
                "draft_sha256": sha256(Path(spec["draft"])),
                "prompts_sha256": sha256(args.prompts), "task_prompt_ids": {}, "requests": [],
                "kind": "instrumented_quality_capture_not_timing", "complete": False}
    write(cell/"manifest.json", manifest)
    with (cell/"server.log").open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            url = f"http://127.0.0.1:{args.port}"
            wait_ready(url, proc, 300)
            for index, prompt in enumerate(prompts):
                before = len(read_jsonl(rows_path))
                rounds_before = len(read_jsonl(rounds_path))
                trace_before = len(read_jsonl(trace))
                body = {"messages": prompt["messages"], "max_tokens": args.tokens,
                        "temperature": 0, "seed": 42, "stream": False, "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_format": "none", "return_tokens": True, "verbose": True}
                request = cell/f"request-{index:03d}"
                request.mkdir()
                write(request/"prompt.json", prompt)
                write(request/"request.json", body)
                response = http_json(url+"/v1/chat/completions", body, 600)
                write(request/"response.json", response)
                ids = generated_token_ids(response)
                if not ids:
                    raise ValueError("missing raw output token IDs")
                rows = read_jsonl(rows_path)[before:]
                rounds = read_jsonl(rounds_path)[rounds_before:]
                traces = read_jsonl(trace)[trace_before:]
                if not rows or not rounds or not traces:
                    raise ValueError("missing request head states or native rounds")
                task_ids = {str(r["task_id"]) for r in rows}
                if len(task_ids) != 1 or task_ids != {str(r["task_id"]) for r in rounds}:
                    raise ValueError("ambiguous request-to-task join")
                task_id = next(iter(task_ids))
                if task_id in manifest["task_prompt_ids"]:
                    raise ValueError("native task ID reused across requests")
                manifest["task_prompt_ids"][task_id] = prompt["id"]
                if any(r["forced"] != bool(forced) for r in rows):
                    raise ValueError("wrong native forced/own-history state")
                write(request/"rounds.json", traces)
                item = {"id": prompt["id"], "task_id": task_id, "generated_token_ids": ids,
                        "capture_rows": [before, before+len(rows)],
                        "forced_round_rows": [rounds_before, rounds_before+len(rounds)],
                        "request_sha256": sha256(request/"request.json"),
                        "prompt_sha256": sha256(request/"prompt.json"),
                        "response_sha256": sha256(request/"response.json"),
                        "quality": quality(traces)}
                manifest["requests"].append(item)
                write(cell/"manifest.json", manifest)
                progress = f"{name} {index+1}/{len(prompts)} {prompt['id']} rows={len(rows)}"
                print(progress, flush=True)
        finally:
            manifest["server_stop"] = stop_server(proc)
            write(cell/"manifest.json", manifest)
    if not manifest["server_stop"].get("stopped"):
        raise RuntimeError("server process group did not stop")
    for marker in spec.get("required_markers", []):
        if marker not in (cell/"server.log").read_text(errors="replace"):
            raise ValueError(f"missing runtime marker: {marker}")
    manifest["audit"] = audit_rows(read_jsonl(rows_path), cell/"heads.f32",
                                  manifest["task_prompt_ids"])
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in (rows_path, cell/"heads.f32", rounds_path, cell/"server.log")}
    manifest["complete"] = True
    write(cell/"manifest.json", manifest)
    return manifest


def run(args):
    prompts = load_prompts(args.prompts)
    variants = json.loads(args.variants.read_text())
    validate_inputs(args, prompts, variants)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"config.json", {"mode": args.mode, "tokens": args.tokens,
                                      "variants": variants, "prompts_sha256": sha256(args.prompts)})
    if args.mode == "train":
        cell = run_cell(args, "d_d", variants["d_d"], prompts)
        write(args.output/"capture-manifest.json", {
            "split": "train", "body": "D", "trajectory": "own_history",
            "train_prompts_sha256": sha256(args.prompts), "prompt_ids": [p["id"] for p in prompts],
            "d2t_path": str(args.d2t.resolve()), "d2t_sha256": sha256(args.d2t),
            "target_vocab_size": args.target_vocab_size,
            "selection_budget": "prepare:17 per prompt/depth, at most8160; raw capture unfiltered",
            "captures": [{"states_path": "d_d/heads.f32", "rows_path": "d_d/heads.jsonl",
                          "state_dim": cell["audit"]["state_dim"],
                          "task_prompt_ids": cell["task_prompt_ids"]}]})
    else:
        run_cell(args, "canonical", variants["q4_q4"], prompts)
        for name in ARMS:
            run_cell(args, name, variants[name], prompts,
                     args.output/"canonical"/"forced-rounds.jsonl")
        write(args.output/"factorial-analysis.json", analyze(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("factorial", "train", "diagnostic"), required=True)
    for name in ("binary", "target", "prompts", "variants", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--prompts-sha256",
                        help="frozen split hash, alternative to prompt manifest")
    parser.add_argument("--d2t", type=Path)
    parser.add_argument("--target-vocab-size", type=int)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handle_stop_signal)
    run(args)


if __name__ == "__main__":
    main()
