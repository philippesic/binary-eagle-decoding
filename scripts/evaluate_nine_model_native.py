#!/usr/bin/env python3
"""Fresh-process native nine-model evaluation using actual llama-server requests."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from benchmark_dspark_screen import round_summary, rows, stop_owned_server
from benchmark_native_eagle import (
    available_port,
    execute_request,
    request_body,
    wait_ready,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    CELLS,
    LinuxResources,
    atomic_json,
    load_opaque_prompts,
    native_environment,
    require,
    sha256,
    stop_signals,
    validate_bundle,
)
from w1a1_eagle.nine_model_report import aggregate  # noqa: E402


def order(rep):
    # Alternating direction and rotation keeps order fixed before any result.
    cells = (*CELLS, "target_only")
    shifted = cells[rep % len(cells) :] + cells[: rep % len(cells)]
    return shifted if rep % 2 == 0 else tuple(reversed(shifted))


def native_command(bundle, protocol, cell, model, port):
    cmd = [
        bundle["inputs"]["binary"]["path"],
        "-m",
        bundle["inputs"]["target"]["path"],
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        str(protocol["context_tokens"]),
        "--batch-size",
        str(protocol["batch_tokens"]),
        "--ubatch-size",
        str(protocol["microbatch_tokens"]),
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
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if cell == "target_only":
        return cmd + ["--spec-type", "none"]
    family = cell.split("_")[0]
    return cmd + [
        "-md",
        model["path"],
        "--spec-type",
        "draft-eagle3" if family == "eagle" else "draft-dspark",
        "--spec-draft-n-max",
        str(protocol["draft_lengths"][family]),
        "--spec-draft-p-min",
        "0",
        "--spec-draft-ngl",
        "all",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
    ]


def run(args):
    require(sha256(args.bundle) == args.bundle_sha256, "bundle differs")
    bundle, files = validate_bundle(args.bundle)
    inputs = json.loads(args.evaluation_inputs.read_text())
    require(
        inputs["bundle_sha256"] == args.bundle_sha256
        and inputs["controls"] == bundle["controls"]
        and inputs["target"] == bundle["inputs"]["target"],
        "evaluation ancestry differs",
    )
    for candidate in CANDIDATES:
        training = json.loads(files.check(inputs["training_endpoints"][candidate]).read_text())
        exported = json.loads(files.check(inputs["export_endpoints"][candidate]).read_text())
        require(
            training.get("bundle_sha256") == args.bundle_sha256
            and training.get("committed") is True
            and training.get("completion_reason") == "approved_budget_complete"
            and training.get("artifact_kind") == "production"
            and exported.get("bundle_sha256") == args.bundle_sha256
            and exported.get("model") == inputs["models"][candidate],
            "evaluation requires successful committed endpoint/export ancestry",
        )
    coverage = {}
    for candidate in CANDIDATES:
        coverage[candidate] = bundle["candidates"][candidate].get("deployment_coverage")
    for family, control in inputs["controls"].items():
        coverage[family + "_q4"] = control.get("deployment_coverage")
    require(
        all(isinstance(value, dict) and value for value in coverage.values()),
        "explicit tensor coverage/normal precision exception inventory missing",
    )
    models = dict(inputs["models"])
    for family, control in inputs["controls"].items():
        models[family + "_q4"] = control["model"]
    require(set(models) == set(CELLS), "matched nine model artifacts required")
    for model in models.values():
        files.check(model)
    protocol = json.loads(files.check(bundle["inputs"]["protocol"]).read_text())
    require(
        protocol["repetitions"] >= 5 and protocol["warmups_per_cell"] >= 2,
        "five repetitions/two warmups required",
    )
    require(protocol["split"] in {"development", "final"}, "explicit evaluation split required")
    # Final evaluation is admitted only after every recipe/model/policy has frozen;
    # the pipeline never reads final prompts during preparation or selection.
    if protocol["split"] == "final":
        require(bundle.get("final_set_authorized") is True, "final set remains sealed")
    prompts = load_opaque_prompts(files.check(bundle["inputs"]["prompts"]))
    require(
        prompts and all(p.get("split") == protocol["split"] for p in prompts),
        "prompt role differs; no fabricated ID prefix rules",
    )
    require(len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt IDs")
    destination = args.completion_output.parent / "native-results"
    destination.mkdir(parents=True, exist_ok=False)
    observer = LinuxResources(bundle["gpu_uuid"])
    hardware = observer.snapshot()
    records, diagnostics = [], []
    started = time.monotonic()
    request_config = {
        "evaluation": {
            "max_output_tokens": protocol["max_output_tokens"],
            "temperature": 0.0,
            "seed": protocol["seed"],
            "enable_thinking": False,
        }
    }
    import subprocess

    with stop_signals():
        # Clean timing and detailed round tracing are separate process launches.
        passes = [(r, False) for r in range(protocol["repetitions"])] + [(0, True)]
        for rep, diagnostic in passes:
            for cell in order(rep):
                require(not (args.run_dir / "STOP").exists(), "campaign STOP requested")
                label = "diagnostic" if diagnostic else f"rep-{rep:02d}"
                directory = destination / label / cell
                directory.mkdir(parents=True)
                bits = (8 if cell.endswith("a8") else 1) if cell.endswith(("a8", "a1")) else None
                env = native_environment(
                    cell.split("_")[0] if cell != "target_only" else cell,
                    bits,
                    bundle.get("environment"),
                )
                if cell in {"eagle_a8", "eagle_a1"}:
                    bits = 8 if cell.endswith("a8") else 1
                    exported = json.loads(files.check(inputs["export_endpoints"][cell]).read_text())
                    audit = json.loads(files.check(exported["audit"]).read_text())
                    require(
                        audit.get("activation_bits") == bits
                        and audit.get("output", {}).get("sha256") == models[cell]["sha256"],
                        "EAGLE activation arithmetic differs from frozen GGUF audit",
                    )
                if cell.startswith(("dspark", "dflash")):
                    env["DSPARK_REQUIRE_AUTHOR_LAYOUT"] = "1"
                trace = directory / "rounds.jsonl"
                if diagnostic:
                    env["W1AX_ROUND_TRACE_JSONL"] = str(trace)
                port = protocol["port"]
                require(available_port("127.0.0.1", port), "native evaluator port occupied")
                cmd = native_command(bundle, protocol, cell, models.get(cell), port)
                proc = None
                with (directory / "server.log").open("wb") as log:
                    try:
                        # Block cancellation while installing owned process identity.
                        mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
                        try:
                            proc = subprocess.Popen(
                                cmd,
                                cwd=ROOT,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                                preexec_fn=lambda: signal.pthread_sigmask(
                                    signal.SIG_SETMASK, old_mask
                                ),
                            )
                            atomic_json(
                                directory / "owned-process.json",
                                {"pid": proc.pid, "pgid": proc.pid, "argv": cmd},
                            )
                        finally:
                            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                        wait_ready(
                            proc, f"http://127.0.0.1:{port}", protocol["startup_wall_seconds"]
                        )
                        cases = [
                            (True, i, prompts[i % len(prompts)])
                            for i in range(protocol["warmups_per_cell"])
                        ]
                        cases += [(False, i, p) for i, p in enumerate(prompts)]
                        for warmup, i, prompt in cases:
                            require(not (args.run_dir / "STOP").exists(), "campaign STOP requested")
                            remaining = protocol["evaluation_wall_seconds"] - (
                                time.monotonic() - started
                            )
                            require(remaining > 0, "evaluation wall cap exhausted")
                            before = len(rows(trace)) if diagnostic else 0
                            request_dir = directory / (
                                f"warmup-{i:04d}" if warmup else f"prompt-{i:04d}"
                            )
                            result = execute_request(
                                f"http://127.0.0.1:{port}",
                                request_body(request_config, prompt),
                                min(180, remaining),
                                request_dir,
                            )
                            require(
                                result["generated_token_ids"] is not None,
                                "native token IDs omitted",
                            )
                            if warmup:
                                continue
                            row = {
                                "cell": cell,
                                "repetition": rep,
                                "prompt_id": prompt["id"],
                                "output_tokens": result["completion_tokens"],
                                "latency_s": result["request_wall_s"],
                                "generated_token_ids": result["generated_token_ids"],
                                "speculative": result["speculative"],
                                "raw_result": str(request_dir / "measurement.json"),
                            }
                            if diagnostic and cell != "target_only":
                                summary = round_summary(
                                    rows(trace)[before:],
                                    protocol["draft_lengths"][cell.split("_")[0]],
                                )
                                require(summary["rounds"] > 0, "native round trace absent")
                                row["round_summary"] = summary
                            (diagnostics if diagnostic else records).append(row)
                            atomic_json(
                                destination / "progress.json",
                                {"clean_records": records, "diagnostic_records": diagnostics},
                            )
                    finally:
                        if proc is not None:
                            stop_owned_server(proc)
                if cell.endswith(("a8", "a1")):
                    text = (directory / "server.log").read_text(errors="replace")
                    markers = bundle["candidates"][cell]["native_markers"]
                    require(
                        markers and all(marker in text for marker in markers),
                        f"actual native loader/dispatch proof absent: {cell}",
                    )
                    require(
                        "dense fallback" not in text.lower(), "native dense fallback prohibited"
                    )
    # Greedy verifier semantics: all nine must reproduce the target-only token prefix.
    target = {(r["repetition"], r["prompt_id"]): r for r in records if r["cell"] == "target_only"}
    for row in records:
        require(
            row["generated_token_ids"]
            == target[(row["repetition"], row["prompt_id"])]["generated_token_ids"],
            f"verifier output mismatch: {row['cell']}/{row['prompt_id']}",
        )
    measurements = {
        "schema": "nine_model_native_measurements_v1",
        "artifact_kind": "production",
        "native": True,
        "instrumentation_in_clean_timing": False,
        "hardware": hardware["hardware"],
        "compute_capability": hardware["compute_capability"],
        "protocol": bundle["inputs"]["protocol"],
        "target": inputs["target"],
        "model_ancestry": models,
        "deployment_coverage": coverage,
        "records": records,
        "diagnostic_records": diagnostics,
    }
    atomic_json(destination / "measurements.json", measurements)
    report = destination / "report.json"
    atomic_json(report, aggregate(measurements))
    atomic_json(
        args.completion_output,
        {
            "schema": "nine_model_stage_receipt_v1",
            "stage": "evaluation",
            "status": "PASS",
            "artifact_kind": "production",
            "bundle_sha256": args.bundle_sha256,
            "native": True,
            "cells": list(CELLS),
            "target_only_diagnostic": True,
            "report": {"path": str(report), "sha256": sha256(report)},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--evaluation-inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    run(parser.parse_args())
