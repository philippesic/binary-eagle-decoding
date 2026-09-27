"""Supervised native-chain scale screen/capture. Quality only; no timing claims."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
SELECTED = {
    "fc.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quality(rows):
    rows = [r for r in rows if r["status"] != "checkpoint_replay"]
    for row in rows:
        if not 0 <= row["n_accepted"] <= row["n_proposed"] <= 5:
            raise ValueError("invalid acceptance count")
        if len(row["proposed_token_ids"]) != row["n_proposed"]:
            raise ValueError("proposal count mismatch")
    accepted = sum(r["n_accepted"] for r in rows)
    emitted = sum(r.get("n_emitted", 0) for r in rows)
    accepted_emitted = sum(min(r["n_accepted"], r.get("n_emitted", 0)) for r in rows)
    depth = {}
    for d in range(1, 6):
        eligible = sum(r["n_proposed"] >= d for r in rows)
        survived = sum(r["n_accepted"] >= d for r in rows)
        depth[str(d)] = {
            "survived": survived,
            "eligible": eligible,
            "rate": survived / eligible if eligible else None,
        }
    return {
        "rounds": len(rows),
        "accepted": accepted,
        "emitted": emitted,
        "accepted_emitted": accepted_emitted,
        "accepted_not_emitted": accepted - accepted_emitted,
        "proposed": sum(r["n_proposed"] for r in rows),
        "accepted_per_round": accepted / len(rows) if rows else None,
        "depth": depth,
    }


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prompts = load_prompts(args.prompts)
    if args.capture and len(prompts) != 96 and not args.diagnostic:
        raise ValueError("training capture requires exactly96 prompts")
    if not args.capture and len(prompts) != 24 and not args.diagnostic:
        raise ValueError("development screen requires exactly24 prompts")
    if "final" in args.prompts.name.lower():
        raise ValueError("reserved-final input prohibited")
    config = json.loads(args.variants.read_text())
    all_results = {}
    for name, spec in config.items():
        cell = out / name
        cell.mkdir()
        cap = cell / "activations"
        cap.mkdir()
        trace = cell / "rounds.jsonl"
        env = os.environ.copy()
        for k in list(env):
            if k.startswith(("GGML_W1", "GGML_EAGLE", "W1AX_", "EAGLE_")):
                env.pop(k)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "W1AX_ROUND_TRACE_JSONL": str(trace),
                "EAGLE_STATE_TRACE_JSONL": str(cell / "state.jsonl"),
            }
        )
        env.update(spec.get("env", {}))
        if args.capture:
            env["GGML_W1AX_CAPTURE_DIR"] = str(cap)
            env.setdefault("GGML_EAGLE_DENSE_A16", "1")
        cmd = [
            str(args.binary.resolve()),
            "-m",
            str(args.target.resolve()),
            "--n-gpu-layers",
            "all",
            "--ctx-size",
            "2048",
            "--parallel",
            "1",
            "--fit",
            "off",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--jinja",
            "--metrics",
            "--perf",
            "-lv",
            "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        if spec.get("draft"):
            cmd += [
                "-md",
                str(Path(spec["draft"]).resolve()),
                "--spec-type",
                "draft-eagle3",
                "--spec-draft-n-max",
                "5",
                "--spec-draft-p-min",
                "0",
                "--spec-draft-ngl",
                "all",
                "--spec-draft-type-k",
                "f16",
                "--spec-draft-type-v",
                "f16",
            ]
        else:
            cmd += ["--spec-type", "none"]
        manifest = {
            "name": name,
            "command": cmd,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith(("GGML_", "W1AX_", "EAGLE_", "CUDA_VISIBLE"))
            },
            "binary_sha256": sha256(args.binary),
            "target_sha256": sha256(args.target),
            "draft_sha256": sha256(Path(spec["draft"])) if spec.get("draft") else None,
            "prompts_sha256": sha256(args.prompts),
            "spec": spec,
            "requests": [],
        }
        write(cell / "manifest.json", manifest)
        all_rounds = []
        capture_prompts = []
        with (cell / "server.log").open("wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                url = f"http://127.0.0.1:{args.port}"
                wait_ready(url, proc, 300)
                for index, prompt in enumerate(prompts):
                    before = len(read_jsonl(trace))
                    state_trace = cell / "state.jsonl"
                    state_before = len(read_jsonl(state_trace))
                    files_before = set(p.name for p in cap.glob("op-*.bin"))
                    body = {
                        "messages": prompt["messages"],
                        "max_tokens": args.tokens,
                        "temperature": 0,
                        "seed": 42,
                        "stream": False,
                        "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_format": "none",
                        "return_tokens": True,
                        "verbose": True,
                    }
                    request = cell / f"{index:03d}-{prompt['id']}"
                    request.mkdir()
                    write(request / "request.json", body)
                    response = http_json(url + "/v1/chat/completions", body, 600)
                    write(request / "response.json", response)
                    ids = generated_token_ids(response)
                    if not ids:
                        raise ValueError("missing raw output token IDs")
                    rounds = read_jsonl(trace)[before:]
                    if spec.get("draft") and not rounds:
                        raise ValueError("missing native round traces")
                    write(request / "rounds.json", rounds)
                    state_events = read_jsonl(state_trace)[state_before:]
                    write(request / "state.json", state_events)
                    all_rounds.extend(rounds)
                    item = {
                        "id": prompt["id"],
                        "generated_token_ids": ids,
                        "quality": quality(rounds),
                        "response_sha256": sha256(request / "response.json"),
                    }
                    if args.capture:
                        files = sorted(set(p.name for p in cap.glob("op-*.bin")) - files_before)
                        sidecar = [
                            r for r in read_jsonl(cap / "captures.jsonl") if r["file"] in set(files)
                        ]
                        names = {r["weight_tensor"] for r in sidecar}
                        if not SELECTED <= names:
                            raise ValueError(f"missing captured layer(s): {SELECTED - names}")
                        cp = {"id": prompt["id"], "capture_dir": str(cap), "files": files}
                        capture_prompts.append(cp)
                        item["capture"] = cp
                        write(cell / "capture-manifest.json", {"prompts": capture_prompts})
                    manifest["requests"].append(item)
                    write(cell / "manifest.json", manifest)
                    print(
                        name,
                        f"{index + 1}/{len(prompts)}",
                        prompt["id"],
                        item["quality"]["accepted_per_round"],
                        flush=True,
                    )
                manifest["quality"] = quality(all_rounds)
            finally:
                manifest["server_stop"] = stop_server(proc)
                write(cell / "manifest.json", manifest)
        server_text = (cell / "server.log").read_text(errors="replace")
        for marker in spec.get("required_markers", []):
            if marker not in server_text:
                raise ValueError(f"missing runtime marker {marker}")
        manifest["server_log_sha256"] = sha256(cell / "server.log")
        write(cell / "manifest.json", manifest)
        all_results[name] = manifest
    write(out / "summary.json", all_results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handle_stop_signal)
    run(args)


if __name__ == "__main__":
    main()
