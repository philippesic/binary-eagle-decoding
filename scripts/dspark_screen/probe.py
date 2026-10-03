"""Produce actual native admission evidence inside caller-owned remote_job.py.

This is a GPU transaction when executed by the sole operator. It does not take
ownership from another job, connect to any host, or consume an existing admission.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from check_export import sha256
from validate_native import validate

ARMS = ("target_only", "dspark_3", "dspark_7", "dflash_3", "dflash_7")
EOS_FIXTURE = {"id": "admission-one-word-eos", "split": "development",
               "messages": [{"role": "user", "content": "Reply with the single word YES and then stop. Do not add punctuation or explanation."}]}
ROOT = Path(__file__).resolve().parents[2]
_launching = False
_pending_signal = None


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helpers(project: Path):
    sys.path.insert(0, str(project / "scripts"))
    return importlib.import_module("benchmark_dspark_screen"), importlib.import_module("benchmark_native_eagle")


def supervisor_receipt():
    temp = os.environ.get("TMPDIR")
    require(temp, "probe requires caller remote_job.py with --stop-grace-seconds 30")
    path = Path(temp).resolve().parent / "state.json"
    require(path.is_file(), "caller remote_job.py receipt is missing")
    receipt = read(path)
    require(receipt.get("stop_grace_seconds", 0) >= 30 and receipt.get("supervisor_pid", 0) > 0 and
            any(Path(str(arg)).name == "probe.py" for arg in receipt.get("command", [])),
            "caller remote_job.py needs a probe command and >=30-second cleanup grace")
    return {"path": str(path), "supervisor_pid": receipt["supervisor_pid"],
            "stop_grace_seconds": receipt["stop_grace_seconds"], "run_id": receipt["run_id"]}


def preflight(config: dict, project: Path):
    """Pin immutable input files before any process/model launch."""
    cache = {}
    def pin(asset):
        path = Path(asset["path"]).resolve()
        require(path.is_file(), f"missing input {path}")
        actual = cache.setdefault(str(path), None)
        if actual is None:
            actual = cache[str(path)] = sha256(path)
        require(actual == asset["sha256"], f"input hash changed: {path}")
        return path
    pin(config["binary"])
    target = pin(config["target"])
    if "protocol" in config:
        protocol_path = pin(config["protocol"])
    else:
        protocol_path = pin({"path": str(project / "configs/dspark-screen/protocol.json"),
                             "sha256": config["protocol_sha256"]})
    protocol = read(protocol_path)
    expected = {"context_tokens": 2048, "kv_precision": "f16", "target_precision": "f16",
                "max_output_tokens": 128, "temperature": 0.0, "seed": 42,
                "enable_thinking": False, "cache_prompt": False, "concurrency": 1,
                "candidate_lengths": [3, 7], "native_noise_tokens": 7, "warmups_per_cell": 2}
    require(all(protocol.get(k) == v for k, v in expected.items()), "unfrozen probe protocol")
    prompts_path = pin({"path": str(project / protocol["prompt_file"]), "sha256": protocol["prompt_sha256"]})
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line]
    require(len(prompts) == 24 and all(p.get("split") == "development" for p in prompts),
            "probe requires the frozen 24 unsealed development prompts")
    for kind in ("dspark", "dflash"):
        row = config[kind]
        pin(row)
        for key in ("source", "conversion_config", "canonical", "export"):
            pin(row[key])
        canonical, export = read(row["canonical"]["path"]), read(row["export"]["path"])
        if canonical.get("target_tied_head_fallback"):
            pin({"path": canonical["target_config_path"], "sha256": canonical["target_config_sha256"]})
            pin({"path": canonical["target_loader_path"], "sha256": canonical["target_loader_sha256"]})
            require(export.get("target_tied_head_fallback") and
                    export.get("target_head_source_tensor") == "token_embd.weight",
                    "export tied-head role provenance changed")
        require(canonical["source_sha256"] == row["source"]["sha256"] and
                canonical["target_sha256"] == config["target"]["sha256"], "canonical ancestry changed")
        require(export.get("passed") and export["source_sha256"] == row["source"]["sha256"] and
                export["config_sha256"] == row["conversion_config"]["sha256"] and
                export["draft_sha256"] == row["sha256"] and export["target_sha256"] == config["target"]["sha256"] and
                export["canonical_comparison_sha256"] == row["canonical"]["sha256"] and
                Path(export["canonical_comparison_path"]).resolve() == Path(row["canonical"]["path"]).resolve(),
                "export/canonical ancestry changed")
        for role in ("embedding", "head"):
            if export["borrows_" + role]:
                require(canonical["safe_to_borrow_" + role], f"unsafe target {role} borrowing")
        require(export["matrix_storage_types"] == ["BF16"], "released matrix precision changed")
    return protocol, protocol_path, prompts, cache


def environment_for(cell: Path, arm: str):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "W1AX_", "EAGLE_", "DSPARK_"))}
    env.update(CUDA_VISIBLE_DEVICES="0", W1AX_ROUND_TRACE_JSONL=str(cell / "rounds.jsonl"))
    if arm != "target_only":
        env.update(DSPARK_REQUIRE_AUTHOR_LAYOUT="1", DSPARK_ADMISSION_JSONL=str(cell / "state.jsonl"))
    return env


def run(config_path: Path, destination: Path, project: Path = ROOT):
    global _launching, _pending_signal
    screen, common = load_helpers(project)
    require("grace_s" in inspect.signature(screen.stop_owned_server).parameters,
            "root stop_owned_server helper needs the reviewed grace_s argument")
    config = read(config_path)
    protocol, protocol_path, prompts, pins = preflight(config, project)
    supervisor = supervisor_receipt()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "config.json", config)
    write(destination / "protocol.json", protocol)
    write(destination / "input_pins.json", pins)
    write(destination / "supervisor.json", supervisor)
    write(destination / "eos_fixture.json", EOS_FIXTURE)
    environment_path = destination / "environment.json"
    environment = common.environment_manifest(Path(config["binary"]["path"]))
    write(environment_path, environment)
    capability = environment.get("gpu_capability_query", {})
    require(capability.get("exit_code") == 0 and "RTX 2080 Ti" in capability.get("stdout", "") and
            "7.5" in capability.get("stdout", "") and
            len(capability.get("stdout", "").strip().splitlines()) == 1,
            "fresh single-device RTX2080Ti/SM75 hardware proof missing")
    write(destination / "gpu-before.json", common.gpu_snapshot())
    request_config = {"evaluation": {"max_output_tokens": 128, "temperature": 0.0,
                                    "seed": 42, "enable_thinking": False}}
    cases = [("warmup-00", prompts[0], True), ("warmup-01", prompts[1], True),
             ("prompt-00", prompts[0], False), ("prompt-01", prompts[1], False),
             ("eos", EOS_FIXTURE, False)]
    manifest = {"target": config["target"]["path"], "binary": config["binary"]["path"],
                "binary_sha256": config["binary"]["sha256"], "environment": str(environment_path), "cells": []}
    records = []
    try:
        for arm in ARMS:
            cell = destination / arm
            cell.mkdir()
            port = config.get("port", 18290)
            require(common.available_port("127.0.0.1", port), "probe port is occupied")
            env = environment_for(cell, arm)
            command = screen.command(config, protocol, arm, port)
            launch_path = cell / "launch.json"
            write(launch_path, {"command": command, "trace_env": {k: v for k, v in env.items()
                  if k.startswith(("GGML_", "W1AX_", "EAGLE_", "DSPARK_", "CUDA_"))}})
            proc = None
            cleanup = {"pid": None, "pgid": None, "started": False, "stopped": False}
            start = time.monotonic()
            with (cell / "server.log").open("wb") as log:
                try:
                    # Defer cancellation only across spawn/receipt publication;
                    # no signal mask is inherited by the native server.
                    _launching = True
                    try:
                        proc = subprocess.Popen(command, cwd=project, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                        cleanup.update(pid=proc.pid, pgid=proc.pid, started=True, command=command)
                        write(cell / "owned_process.json", cleanup)
                    finally:
                        _launching = False
                    if _pending_signal is not None:
                        signum, _pending_signal = _pending_signal, None
                        interrupted(signum, None)
                    screen.wait_ready(proc, f"http://127.0.0.1:{port}", 300)
                    write(cell / "startup.json", {"load_admission_s": time.monotonic()-start, "pid": proc.pid})
                    write(cell / "gpu-loaded.json", common.gpu_snapshot())
                    for name, prompt, warmup in cases:
                        directory = cell / name
                        before = len(screen.rows(cell / "rounds.jsonl"))
                        result = common.execute_request(f"http://127.0.0.1:{port}",
                                                        common.request_body(request_config, prompt), 180, directory)
                        require(result["generated_token_ids"] is not None, "server omitted exact output IDs")
                        require(isinstance(result["prompt_tokens"], int) and result["prompt_tokens"] + 128 + 7 <= 2048,
                                "probe lacks seven-slot/context/output reserve")
                        round_rows = screen.rows(cell / "rounds.jsonl")[before:]
                        write(directory / "rounds.json", round_rows)
                        if name == "eos":
                            response = read(directory / "response.json")
                            text = response["choices"][0]["message"].get("content", "")
                            require(text.strip() == "YES" and result["finish_reason"] == "stop", "one-word EOS fixture did not stop")
                            if arm != "target_only":
                                require(any(151645 in r.get("emitted_token_ids", []) for r in round_rows),
                                        "EOS fixture lacks actual native EOS emission")
                        records.append({"arm": arm, "case": name, "warmup": warmup, **result})
                        write(destination / "progress.json", {"records": records})
                finally:
                    if proc is not None:
                        cleanup_start = time.monotonic()
                        screen.stop_owned_server(proc, grace_s=15)
                        if hasattr(proc, "poll"):
                            proc.poll()
                        cleanup.update(stopped=proc.returncode is not None, returncode=proc.returncode,
                                       cleanup_wall_s=time.monotonic()-cleanup_start)
                        write(cell / "owned_process.json", cleanup)
                        require(cleanup["stopped"], "owned server has no verified terminal status")
            if arm == "target_only":
                continue
            kind, length = arm.rsplit("_", 1)
            row = config[kind]
            outputs = [{"actual": str(cell / name / "measurement.json"),
                        "reference": str(destination / "target_only" / name / "measurement.json"),
                        "prompt": str(cell / name / "request.json"),
                        "prompt_sha256": sha256(cell / name / "request.json")} for name, _, _ in cases]
            manifest["cells"].append({"kind": kind, "maximum": int(length), "draft": row["path"],
                "source": row["source"]["path"], "conversion_config": row["conversion_config"]["path"],
                "export": row["export"]["path"], "launch": str(launch_path),
                "server_log": str(cell / "server.log"), "state": str(cell / "state.jsonl"),
                "rounds": str(cell / "rounds.jsonl"), "outputs": outputs})
        write(destination / "manifest.json", manifest)
        admission = validate(manifest)
        admission["probe_evidence"] = {"protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
                                       "manifest_sha256": sha256(destination / "manifest.json"),
                                       "eos_fixture_sha256": sha256(destination / "eos_fixture.json")}
        write(destination / "admission.json", admission)
        write(destination / "measurements.json", {"scope": "admission; no throughput verdict", "records": records})
    except BaseException as error:
        write(destination / "failure.json", {"type": type(error).__name__, "error": str(error), "records_completed": len(records)})
        raise


def interrupted(signum, _frame):
    global _pending_signal
    if _launching:
        _pending_signal = signum
        return
    # One cancellation is sufficient; repeated signals must not interrupt the
    # bounded owned-server cleanup before remote_job.py's required 30s grace.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.config.resolve(), args.output.resolve(), args.project_root.resolve())
