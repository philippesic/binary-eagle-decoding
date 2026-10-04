#!/usr/bin/env python3
"""Bounded real EAGLE CUDA graph smoke on authenticated TRAIN prefix only.

Zero optimizer updates, no quality or throughput claim. Exact source/config pins
and loader/dispatch observations are required; this is not a synthetic gate.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmark_dspark_screen import stop_owned_server
from benchmark_native_eagle import (
    available_port,
    execute_request,
    request_body,
    wait_ready,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    Files,
    LinuxResources,
    atomic_json,
    clean_environment,
    load_opaque_prompts,
    process_identity,
    resource_gate,
    sha256,
    stop_signals,
)
from w1a1_eagle.nine_model_pipeline import require as require_value  # noqa: E402


def smoke(config_path, receipt_path):
    config = json.loads(config_path.read_text())
    require_value(
        config.get("schema") == "eagle_native_graph_smoke_config_v1", "smoke config differs"
    )
    files = Files()
    required = {"binary", "model", "target", "export_audit", "train_prompts", "train_capture"}
    require_value(required <= set(config["inputs"]), "smoke ancestry inventory incomplete")
    paths = {name: files.check(record) for name, record in config["inputs"].items()}
    audit = json.loads(paths["export_audit"].read_text())
    bits = config["activation_bits"]
    require_value(
        bits in {1, 8}
        and audit.get("activation_bits") == bits
        and audit.get("serialization_audit_passed") is True
        and audit.get("output", {}).get("sha256") == config["inputs"]["model"]["sha256"],
        "actual packed EAGLE export audit absent",
    )
    capture = json.loads(paths["train_capture"].read_text())
    require_value(
        capture.get("split") == "train"
        and capture.get("target_sha256") == config["inputs"]["target"]["sha256"]
        and capture.get("prompts_sha256") == config["inputs"]["train_prompts"]["sha256"],
        "native TRAIN prefix/target ancestry absent",
    )
    prompts = load_opaque_prompts(paths["train_prompts"])
    require_value(
        prompts and all(p.get("split") == "train" for p in prompts),
        "TRAIN role required; no prompt ID prefix assumptions",
    )
    observer = LinuxResources(config["gpu_uuid"], hardware=config.get("hardware", "rtx5080"))
    baseline = observer.snapshot()
    directory = receipt_path.parent / ("smoke-" + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    env = clean_environment(config.get("environment"))
    port = config["port"]
    require_value(available_port("127.0.0.1", port), "native smoke port occupied")
    command = [
        str(paths["binary"]),
        "-m",
        str(paths["target"]),
        "-md",
        str(paths["model"]),
        "--n-gpu-layers",
        "all",
        "--spec-draft-ngl",
        "all",
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        "5",
        "--spec-draft-p-min",
        "0",
        "--parallel",
        "1",
        "--ctx-size",
        str(config["context_tokens"]),
        "--fit",
        "off",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--jinja",
        "--metrics",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = None
    owned = None
    with stop_signals(), (directory / "server.log").open("wb") as log:
        try:
            old_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
            )
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask),
                )
                owned = process_identity(proc.pid)
                atomic_json(directory / "process.json", {"pid": proc.pid, "kernel_identity": owned})
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            wait_ready(proc, f"http://127.0.0.1:{port}", config["startup_wall_seconds"])
            request = {
                "evaluation": {
                    "max_output_tokens": 8,
                    "temperature": 0.0,
                    "seed": 42,
                    "enable_thinking": False,
                }
            }
            result = execute_request(
                f"http://127.0.0.1:{port}",
                request_body(request, prompts[0]),
                config["request_wall_seconds"],
                directory / "request",
            )
            require_value(result["generated_token_ids"] is not None, "native token IDs absent")
        finally:
            if proc is not None:
                stop_owned_server(proc, grace_s=15)
    release = observer.require_released([proc.pid], [owned])
    resource_gate(observer.snapshot(), baseline, config["resource_policy"])
    text = (directory / "server.log").read_text(errors="replace")
    loader = "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)"
    dispatch = (
        "CUDA packed W1A8 INT8 dispatch" if bits == 8 else "CUDA packed W1A1 XOR/POPCOUNT dispatch"
    )
    require_value(
        loader in text and dispatch in text, "actual all-nine W1 CUDA graph dispatch absent"
    )
    require_value("dense fallback" not in text.lower(), "native dense fallback observed")
    atomic_json(
        receipt_path,
        {
            "schema": "eagle_native_graph_smoke_v1",
            "status": "PASS",
            "family": "eagle",
            "model_sha256": config["inputs"]["model"]["sha256"],
            "target_sha256": config["inputs"]["target"]["sha256"],
            "binary_sha256": config["inputs"]["binary"]["sha256"],
            "cuda_dispatch_observed": True,
            "selected_projection_count": 9,
            "activation_bits": bits,
            "device": baseline,
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "inputs": config["inputs"],
            "optimizer_updates": 0,
            "prompt_split": "train",
            "max_output_tokens": 8,
            "actual_w1_cuda_dispatch": dispatch,
            "loader": loader,
            "owned_release": release,
            "quality_evaluation": False,
            "throughput_evaluation": False,
            "server_log": {
                "path": str(directory / "server.log"),
                "sha256": sha256(directory / "server.log"),
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.config.resolve(), args.receipt.resolve())
