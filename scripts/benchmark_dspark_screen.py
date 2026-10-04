"""Frozen released-drafter screen; admission and native evidence are mandatory."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from benchmark_native_eagle import (
    available_port, environment_manifest, execute_request, json_write,
    load_prompts, request_body, sha256, wait_ready,
)

ARMS = ("target_only", "eagle_q4_0", "dspark_3", "dspark_7", "dflash_3", "dflash_7")
ROOT = Path(__file__).resolve().parents[1]
_launching = False
_pending_signal = None


def stop_owned_server(proc: subprocess.Popen, grace_s: float = 3) -> None:
    # Default throughput cleanup fits remote_job.py's 10s grace. Admission can
    # use 15s to finish device hashes with a corresponding 30s supervisor grace.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=3)


def order(rep: int) -> list[str]:
    # Six-row Williams design balances every position and directed predecessor.
    offsets = (0, 1, 5, 2, 4, 3)
    return [ARMS[(offset + rep) % len(ARMS)] for offset in offsets]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def duration_sum(value) -> int | float:
    if isinstance(value, list):
        return sum(duration_sum(v) for v in value)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid native duration")
    return value


def round_summary(records: list[dict], maximum: int) -> dict:
    complete = [r for r in records if r.get("status") == "complete" and not r.get("replay")]
    no_proposal = [r for r in records if r.get("status") == "no_proposal" and not r.get("replay")]
    for r in no_proposal:
        if r["n_proposed"] != 0 or r["n_accepted"] != 0 or r["proposed_token_ids"] or r["n_emitted"] != 1 or len(r["emitted_token_ids"]) != 1:
            raise ValueError("invalid native no-proposal round")
    for r in complete:
        if not 0 <= r["n_accepted"] <= r["n_proposed"] <= maximum:
            raise ValueError("invalid native round counts")
        if len(r["proposed_token_ids"]) != r["n_proposed"] or len(r["emitted_token_ids"]) != r["n_emitted"]:
            raise ValueError("native round token/count mismatch")
        if "n_accepted_usable_prefix" in r and not 0 <= r["n_accepted_usable_prefix"] <= min(r["n_accepted"], r["n_emitted"]):
            raise ValueError("invalid usable accepted prefix")
        if "verified_token_ids" in r and r["emitted_token_ids"] != r["verified_token_ids"][:r["n_emitted"]]:
            raise ValueError("emitted IDs differ from consumed verifier prefix")
    proposed = sum(r["n_proposed"] for r in complete)
    accepted = sum(r["n_accepted"] for r in complete)
    usable = sum(r["n_accepted_usable_prefix"] for r in complete) if complete and all("n_accepted_usable_prefix" in r for r in complete) else None
    position = {}
    for pos in range(1, maximum + 1):
        # Conditional survival at a reached prefix; also preserve unconditional totals.
        eligible = sum(r["n_proposed"] >= pos for r in complete)
        reached = sum(r["n_proposed"] >= pos and r["n_accepted"] >= pos - 1 for r in complete)
        survived = sum(r["n_accepted"] >= pos for r in complete)
        position[str(pos)] = dict(eligible=eligible, reached=reached, survived=survived,
                                 conditional_survival=survived / reached if reached else None)
    timings = {key: sum(duration_sum(r[key]) for r in complete) if complete and all(key in r for r in complete) else None for key in (
        "round_us", "begin_us", "draft_us", "target_decode_sync_us", "process_us", "kv_repair_us",
        "process_feature_copy_us", "process_draft_decode_us", "draft_seed_decode_us", "draft_step_decode_us", "draft_sampler_us")}
    return dict(rounds=len(complete), proposed=proposed, accepted=accepted,
                no_proposal_rounds=len(no_proposal),
                traced_decode_rounds=len(complete) + len(no_proposal),
                accepted_per_traced_decode_round=accepted / (len(complete) + len(no_proposal)) if complete or no_proposal else None,
                traced_emitted_tokens=sum(r["n_emitted"] for r in complete + no_proposal),
                round_denominator_scope="accepted_per_round uses complete draft rounds; traced_decode_rounds also includes no-proposal records; initial seed may be outside the trace",
                accepted_usable_prefix=usable,
                usable_acceptance=usable / proposed if usable is not None and proposed else None,
                usable_accepted_per_round=usable / len(complete) if usable is not None and complete else None,
                emitted=sum(r["n_emitted"] for r in complete),
                acceptance=accepted / proposed if proposed else None,
                accepted_per_round=accepted / len(complete) if complete else None,
                position=position, cpu_wall_totals_us=timings,
                first_rejection={str(p): sum(r["n_accepted"] == p and r["n_proposed"] > p for r in complete)
                                 for p in range(maximum)},
                other_records=len(records)-len(complete))


def command(config: dict, protocol: dict, arm: str, port: int) -> list[str]:
    cmd = [config["binary"]["path"], "-m", config["target"]["path"],
           "--n-gpu-layers", "all", "--ctx-size", str(protocol["context_tokens"]),
           "--batch-size", str(protocol.get("batch_tokens", 32)), "--ubatch-size", str(protocol.get("microbatch_tokens", 32)),
           "--parallel", "1", "--fit", "off", "--cache-type-k", "f16", "--cache-type-v", "f16",
           "--jinja", "--metrics", "--perf", "-lv", "4", "--host", "127.0.0.1", "--port", str(port)]
    if arm == "target_only":
        return cmd + ["--spec-type", "none"]
    key = "eagle_q4_0" if arm == "eagle_q4_0" else arm.split("_")[0]
    n = protocol["eagle_length"] if arm == "eagle_q4_0" else int(arm.rsplit("_", 1)[1])
    return cmd + ["-md", config[key]["path"], "--spec-type",
                  "draft-eagle3" if arm == "eagle_q4_0" else "draft-dspark",
                  "--spec-draft-n-max", str(n), "--spec-draft-p-min", "0",
                  "--spec-draft-ngl", "all", "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16"]


def prior_timing_seconds(config: dict) -> float:
    total = 0.0
    for entry in config.get("prior_timing_receipts", []):
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("previous timing receipt changed")
        receipt = json.loads(path.read_text())
        spent = receipt.get("inference_s")
        if receipt.get("diagnostic") is not False or not isinstance(spent, (int, float)) or spent < 0:
            raise ValueError("previous receipt is not actual non-diagnostic timing")
        if not isinstance(receipt.get("records"), list):
            raise ValueError("previous timing records missing")
        total += spent
    return total


def check_admission(config: dict, admission: dict) -> None:
    if not admission.get("passed") or admission.get("target_sha256") != config["target"]["sha256"]:
        raise ValueError("missing target-bound actual native admission")
    if admission.get("binary_sha256") != config["binary"]["sha256"] or admission.get("protocol_sha256") != config["protocol_sha256"]:
        raise ValueError("native admission runtime/protocol changed")
    for kind in ("dspark", "dflash", "eagle_q4_0"):
        if admission.get("model_sha256", {}).get(kind) != config[kind]["sha256"]:
            raise ValueError(f"native admission model changed: {kind}")
    if not all(admission.get(k) for k in ("memory", "anchor_first", "target_immutable", "cache_contract", "greedy_semantics")):
        raise ValueError("incomplete native admission")


def run(config_path: Path, destination: Path, diagnostic: bool = False) -> None:
    global _launching, _pending_signal
    config = json.loads(config_path.read_text())
    protocol_path = ROOT / "configs/dspark-screen/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    prior_s = prior_timing_seconds(config)
    if prior_s >= protocol["measurement_budget_s"]:
        raise TimeoutError("combined timing budget exhausted")
    admission = json.loads(Path(config["admission"]["path"]).read_text())
    if sha256(protocol_path) != config["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for key in ("binary", "target", "eagle_q4_0", "dspark", "dflash", "admission"):
        if sha256(Path(config[key]["path"])) != config[key]["sha256"]:
            raise ValueError(f"artifact changed: {key}")
    check_admission(config, admission)
    prompt_path = ROOT / protocol["prompt_file"]
    if sha256(prompt_path) != protocol["prompt_sha256"]:
        raise ValueError("frozen prompts changed")
    prompts = load_prompts(prompt_path)
    if len(prompts) != 24 or any(p.get("split") != "development" for p in prompts):
        raise ValueError("requires 24 explicit unsealed development prompts")
    destination.mkdir(parents=True, exist_ok=False)
    json_write(destination / "config.json", config)
    json_write(destination / "protocol.json", protocol)
    json_write(destination / "environment.json", environment_manifest(Path(config["binary"]["path"])))
    # Reuse established HTTP/token/metrics helpers; the configuration remains frozen.
    request_config = {"evaluation": {"max_output_tokens": protocol["max_output_tokens"],
                                     "temperature": 0.0, "seed": 42, "enable_thinking": False}}
    records = []
    inference_s = 0.0
    for rep in range(1 if diagnostic else protocol["repetitions"]):
        for arm in order(rep):
            cell = destination / f"rep-{rep:02d}" / arm
            cell.mkdir(parents=True)
            trace = cell / "rounds.jsonl"
            port = config.get("port", 18290)
            if not available_port("127.0.0.1", port):
                raise RuntimeError("screen port is occupied")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "W1AX_", "EAGLE_", "DSPARK_"))}
            env.update(CUDA_VISIBLE_DEVICES="0", W1AX_ROUND_TRACE_JSONL=str(trace.resolve()))
            if arm.startswith(("dspark_", "dflash_")):
                env["DSPARK_REQUIRE_AUTHOR_LAYOUT"] = "1"
            # Diagnostic state traces stay outside throughput runs.
            if diagnostic:
                env["DSPARK_ADMISSION_JSONL"] = str((cell / "state.jsonl").resolve())
            cmd = command(config, protocol, arm, port)
            json_write(cell / "launch.json", {"command": cmd, "trace_env": {k: v for k, v in env.items()
                       if k.startswith(("GGML_", "W1AX_", "EAGLE_", "DSPARK_", "CUDA_"))}})
            proc = None
            start = time.monotonic()
            with (cell / "server.log").open("wb") as log:
                try:
                    _launching = True
                    try:
                        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        json_write(cell / "owned_process.json", {"pid": proc.pid, "pgid": proc.pid, "command": cmd})
                    finally:
                        _launching = False
                    if _pending_signal is not None:
                        signum, _pending_signal = _pending_signal, None
                        interrupted(signum, None)
                    wait_ready(proc, f"http://127.0.0.1:{port}", 300)
                    json_write(cell / "startup.json", {"load_admission_s": time.monotonic()-start, "pid": proc.pid})
                    cases = [(True, i, prompts[i]) for i in range(protocol["warmups_per_cell"])]
                    cases += [(False, i, p) for i, p in enumerate(prompts[:2] if diagnostic else prompts)]
                    for warmup, idx, prompt in cases:
                        remaining = protocol["measurement_budget_s"] - prior_s - inference_s
                        if remaining <= 0:
                            raise TimeoutError("cumulative inference budget reached")
                        before = len(rows(trace))
                        request_dir = cell / (f"warmup-{idx:02d}" if warmup else f"prompt-{idx:02d}")
                        started = time.monotonic()
                        try:
                            result = execute_request(f"http://127.0.0.1:{port}", request_body(request_config, prompt),
                                                     min(180, remaining), request_dir)
                        finally:
                            inference_s += time.monotonic()-started
                            # Failed requests still consumed the shared inference allowance.
                            json_write(destination / "progress.json", {"diagnostic": diagnostic, "inference_s": inference_s,
                                       "prior_timing_s": prior_s, "combined_timing_s": prior_s + inference_s,
                                       "records": records, "last_request_artifact": str(request_dir.relative_to(destination))})
                        native = rows(trace)[before:]
                        json_write(request_dir / "rounds.json", native)
                        result.update(arm=arm, repetition=rep, prompt_id=prompt["id"], warmup=warmup,
                                      artifact_path=str(request_dir.relative_to(destination)))
                        if result["generated_token_ids"] is None:
                            raise RuntimeError("native server omitted raw generated token IDs")
                        if arm != "target_only":
                            maximum = protocol["eagle_length"] if arm == "eagle_q4_0" else int(arm[-1])
                            result["round_summary"] = round_summary(native, maximum)
                            if not result["round_summary"]["rounds"]:
                                raise RuntimeError("no complete native rounds")
                        records.append(result)
                        json_write(destination / "progress.json", {"diagnostic": diagnostic, "inference_s": inference_s,
                                   "prior_timing_s": prior_s, "combined_timing_s": prior_s + inference_s, "records": records})
                finally:
                    if proc is not None:
                        stop_owned_server(proc)
    json_write(destination / "measurements.json", {"diagnostic": diagnostic, "inference_s": inference_s,
               "prior_timing_s": prior_s, "combined_timing_s": prior_s + inference_s, "records": records})


def interrupted(signum: int, _frame) -> None:
    global _pending_signal
    if _launching:
        _pending_signal = signum
        return
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.config, args.output.resolve(), args.diagnostic)
