#!/usr/bin/env python3
"""Uninstrumented A8/Q4_0/target-only request measurements inside one existing eval deadline.

This reuses benchmark_native_eagle response parsing/aggregation and the frozen
run_binary_head_capture server policy. Capture elapsed time is never a rate.
The nonstreaming endpoint exposes server prefill/decode spans, but no TTFT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import benchmark_native_eagle as benchmark

VARIANTS = ("Q4_0", "A8", "target_only")
NATIVE_COMMIT = "9e2c7a90051e738751aab7d7bd7c2d8201fb76e3"
EVALUATION_ENV = {"GGML_EAGLE_SHARED_PACK": "1", "GGML_EAGLE_PRUNE_UNUSED_HEAD": "1"}
HISTORICAL_CAPTURE_BINARY = "b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c"


def remaining(deadline: float, stop_file: Path | None = None, cap: float = 180) -> float:
    if stop_file is not None and stop_file.exists():
        raise InterruptedError("user STOP requested during native request timing")
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("shared development deadline exhausted during native request timing")
    return min(cap, seconds)


def clean_environment(sources: dict, variant: str) -> dict[str, str]:
    if sources.get("evaluation_env") != EVALUATION_ENV:
        raise ValueError("native timing requires requested shared pack and unused-head pruning")
    # Whitelist ordinary process env: inherited capture/trace/selector hooks cannot leak.
    env = {key: os.environ[key] for key in benchmark.SAFE_INHERITED_ENV if key in os.environ}
    env.update(
        LD_LIBRARY_PATH=sources["native_runtime"]["ld_library_path"],
        CUDA_VISIBLE_DEVICES="0",
        # Matches the frozen capture execution policy, without tensor/trace instrumentation.
        GGML_CUDA_DISABLE_GRAPHS="1",
        **sources["evaluation_env"],
    )
    if variant == "A8":
        env["GGML_W1AX_ACT_BITS"] = "8"
    return env


def request_orders(repetitions: int) -> list[list[str]]:
    # Six balanced permutations; the primary A8/Q4 pair alternates on every repeat.
    balanced = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2))
    return [[VARIANTS[index] for index in balanced[repeat % 6]] for repeat in range(repetitions)]


def request_command(args, variant: str, drafts: dict[str, Path]) -> list[str]:
    from run_binary_head_capture import server_command

    if variant != "target_only":
        return server_command(args, {"draft": str(drafts[variant])})
    # Build the identical frozen target policy, then remove each draft control.
    command = server_command(args, {"draft": str(args.target)})
    valued_options = {
        "-md",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-draft-p-min",
        "--spec-draft-ngl",
        "--spec-draft-type-k",
        "--spec-draft-type-v",
    }
    flags = {"--no-spec-draft-backend-sampling"}
    target_command = []
    index = 0
    while index < len(command):
        option = command[index]
        if option in valued_options:
            index += 2
        elif option in flags:
            index += 1
        else:
            target_command.append(option)
            index += 1
    if any("spec-draft" in option for option in target_command):
        raise ValueError("target-only policy retained an unknown speculative control")
    return [*target_command, "--spec-type", "none"]


def _json_request(url: str, body: dict | None, deadline: float, stop_file: Path | None):
    result = benchmark.request_json(url, body, remaining(deadline, stop_file))
    remaining(deadline, stop_file)
    return result


def _metrics(url: str, deadline: float, stop_file: Path | None) -> str | None:
    try:
        status, raw = _json_request(url + "/metrics", None, deadline, stop_file)
    except (OSError, urllib.error.URLError):
        # A deadline is a failure, never an unavailable optional counter.
        remaining(deadline, stop_file)
        return None
    return raw if status == 200 else None


def measure_request(
    url: str, body: dict, directory: Path, *, deadline: float, stop_file: Path | None
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    benchmark.json_write(directory / "request.json", body)
    before = _metrics(url, deadline, stop_file)
    if before is not None:
        (directory / "metrics-before.prom").write_text(before)
    remaining(deadline, stop_file)
    started = time.perf_counter()
    status, raw = benchmark.request_json(
        url + "/v1/chat/completions", body, remaining(deadline, stop_file)
    )
    elapsed = time.perf_counter() - started
    (directory / "response.json").write_text(raw)
    remaining(deadline, stop_file)
    if status != 200:
        raise RuntimeError(f"native request HTTP {status}; see {directory / 'response.json'}")
    after = _metrics(url, deadline, stop_file)
    if after is not None:
        (directory / "metrics-after.prom").write_text(after)
    row = benchmark.extract_record(json.loads(raw), elapsed, benchmark.counter_delta(before, after))
    tokens = row["completion_tokens"]
    ids = row["generated_token_ids"]
    if type(tokens) is not int or tokens <= 0 or not ids or len(ids) != tokens:
        raise ValueError("native request needs actual positive output count and matching token IDs")
    for field in ("server_prompt_ms", "server_predicted_ms"):
        value = row[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            row[field] = None
    if not row["server_predicted_ms"]:
        row["decode_tokens_per_s"] = None
    row.update(
        prefill_server_s=(row["server_prompt_ms"] / 1000)
        if isinstance(row["server_prompt_ms"], (int, float))
        else None,
        decode_boundary="server timings.predicted_ms; excludes server prompt_ms",
        ttft_status="unavailable_nonstreaming_endpoint",
    )
    benchmark.json_write(directory / "measurement.json", row)
    return row


def _wait_ready(process, url: str, deadline: float, stop_file: Path | None):
    startup_deadline = min(deadline, time.monotonic() + 120)
    while True:
        budget = remaining(startup_deadline, stop_file, cap=3)
        if process.poll() is not None:
            raise RuntimeError(f"native request server exited during startup: {process.returncode}")
        try:
            status, raw = benchmark.request_json(url + "/health", None, budget)
            remaining(startup_deadline, stop_file)
            if status == 200 and json.loads(raw).get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            remaining(startup_deadline, stop_file)
        time.sleep(min(0.1, remaining(startup_deadline, stop_file)))


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stop_owned(process, deadline: float) -> dict:
    # Cleanup never starts a fresh ten-second wait beyond the aggregate budget.
    _kill_group(process)
    try:
        process.wait(timeout=max(0.001, min(1, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        return {"stopped": False, "process_group_gone": False, "return_code": None}
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        gone = True
    else:
        gone = False
    return {
        "stopped": process.poll() is not None,
        "process_group_gone": gone,
        "return_code": process.returncode,
    }


def _summary(rows: list[dict], complete: bool) -> dict:
    if not complete:
        return {
            "variants": None,
            "speedup_vs_q4_0": None,
            "speedup_vs_no_speculation": None,
            "metrics_status": "unavailable_incomplete_five_repetition_contract",
        }
    aggregates = benchmark.aggregate(rows, VARIANTS)
    for variant, aggregate in aggregates.items():
        selected = [row for row in rows if row["variant"] == variant]
        prefill = [row["prefill_server_s"] for row in selected]
        aggregate["prefill_server_s"] = (
            sum(prefill) if all(isinstance(value, (int, float)) for value in prefill) else None
        )
        aggregate["request_wall_distribution_s"] = _distribution(
            [row["request_wall_s"] for row in selected]
        )
        aggregate["decode_distribution_s"] = (
            _distribution(
                [
                    row["server_predicted_ms"] / 1000
                    for row in selected
                    if isinstance(row["server_predicted_ms"], (int, float))
                ]
            )
            if all(isinstance(row["server_predicted_ms"], (int, float)) for row in selected)
            else None
        )
        aggregate["time_to_first_token_s"] = None
    q4, candidate = aggregates["Q4_0"], aggregates["A8"]
    speedups = {}
    for name in ("request_tokens_per_s", "decode_tokens_per_s"):
        reference, measured = q4[name], candidate[name]
        speedups[name] = measured / reference if reference and measured is not None else None
    target = aggregates["target_only"]
    no_spec_speedups = {}
    for name in ("request_tokens_per_s", "decode_tokens_per_s"):
        reference, measured = target[name], candidate[name]
        no_spec_speedups[name] = (
            measured / reference if reference and measured is not None else None
        )
    parity = benchmark.generated_token_id_matches(rows, VARIANTS, "Q4_0")
    return {
        "variants": aggregates,
        "speedup_vs_q4_0": speedups,
        "speedup_vs_no_speculation": no_spec_speedups,
        "generated_token_id_matches_q4_0": parity,
        "metrics_status": "complete_request_metrics"
        if all(
            row["server_predicted_ms"] is not None and row["server_predicted_ms"] > 0
            for row in rows
        )
        else "complete_requests_decode_unavailable",
    }


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def measure_a8_requests(
    sources: dict,
    prompts: Path,
    draft: Path,
    output: Path,
    *,
    deadline: float,
    stop_file: Path | None = None,
    repetitions: int = 5,
    warmup_requests: int = 1,
    tokens: int = 128,
) -> dict:
    """Measure matched native requests; deadline is caller's ABSOLUTE monotonic time.

    On timeout return an explicit incomplete report, preserving raw partial runs.
    Other failures save that report and raise. No partial-run performance claim.
    Caller must bind the returned manifest and reject incomplete evaluations.
    """
    from run_binary_head_capture import verify_mapped_runtime
    from w1ax_capture_provider import TARGET_GGUF_SHA256
    from w1ax_continuous_stages import (
        Q4_0_GGUF_SHA256,
        native_cancellation,
        require_unsealed_prompts,
        verify_sources,
    )

    if not math.isfinite(deadline) or deadline - time.monotonic() > 1200:
        raise ValueError("request timing must share the caller's <=1200s evaluation deadline")
    if (
        type(repetitions) is not int
        or repetitions < 5
        or type(warmup_requests) is not int
        or warmup_requests < 1
    ):
        raise ValueError("native timing requires at least five repetitions and one warmup")
    if tokens != 128:
        raise ValueError("frozen development output cap is128")
    prompts, draft, output = Path(prompts), Path(draft), Path(output)
    require_unsealed_prompts(prompts, sources=sources)
    verify_sources(sources)
    if sources["sha256"]["target_gguf"] != TARGET_GGUF_SHA256 or not sources.get("native_runtime"):
        raise ValueError("frozen F16 target and immutable native runtime required")
    if (
        sources.get("evaluation_native_commit") != NATIVE_COMMIT
        or sources["sha256"].get("binary") == HISTORICAL_CAPTURE_BINARY
    ):
        raise ValueError(
            "timing requires explicit current9e2 evaluation runtime, not teacher capture b4"
        )
    clean_environment(sources, "A8")
    if sources.get("sha256", {}).get("q4_0_draft") != Q4_0_GGUF_SHA256:
        raise ValueError("frozen Q4_0 SHA256 required")
    prompt_rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    ids = [prompt["id"] for prompt in prompt_rows]
    if (
        len(ids) != 24
        or len(set(ids)) != 24
        or any(
            not name.startswith("qat-revisit-development-")
            or "final" in name.lower()
            or Path(name).name != name
            for name in ids
        )
    ):
        raise ValueError("timing requires the same24 unsealed development prompts")
    remaining(deadline, stop_file)
    port = sources.get("port", 18092)
    if not benchmark.available_port("127.0.0.1", port):
        raise RuntimeError("native timing port already in use; refusing another owner")
    output.mkdir(parents=True, exist_ok=False)
    drafts = {"A8": draft, "Q4_0": Path(sources["q4_0_draft"])}
    args = SimpleNamespace(
        binary=Path(sources["binary"]),
        target=Path(sources["target_gguf"]),
        port=port,
        mode="recurrent-train",
    )
    request_config = {
        "evaluation": {
            "max_output_tokens": 128,
            "temperature": 0,
            "seed": 42,
            "enable_thinking": False,
        }
    }
    orders = request_orders(repetitions)
    report = {
        "schema": "a8_native_request_metrics_v1",
        "kind": "native_request_timing_no_tensor_capture",
        "complete": False,
        "status": "running",
        "repetitions": repetitions,
        "warmup_requests_per_server": warmup_requests,
        "orders": orders,
        "prompts_sha256": benchmark.sha256(prompts),
        "prompt_ids": ids,
        "sources_sha256": sources["sha256"],
        "native_runtime": sources["native_runtime"],
        "evaluation_native_commit": sources["evaluation_native_commit"],
        "hardware": sources.get("evaluation_hardware"),
        "hardware_status": "caller_manifest_required"
        if not sources.get("evaluation_hardware")
        else "caller_supplied_environment_manifest",
        "drafts_sha256": {
            **{key: benchmark.sha256(path) for key, path in drafts.items()},
            "target_only": None,
        },
        "request_options": {
            key: value
            for key, value in benchmark.request_body(request_config, prompt_rows[0]).items()
            if key != "messages"
        },
        "deadline_contract": "caller absolute monotonic deadline; no independent timing budget",
        "timing_definition": (
            "output tokens/full HTTP wall incl prefill; decode uses server predicted_ms"
        ),
        "ttft_status": "unavailable_nonstreaming_endpoint",
        "servers": [],
        "records": [],
    }
    manifest = output / "manifest.json"
    benchmark.json_write(manifest, report)
    url = f"http://127.0.0.1:{port}"
    try:
        with native_cancellation() as guard:
            for repetition, order in enumerate(orders):
                for variant in order:
                    remaining(deadline, stop_file)
                    directory = output / f"rep-{repetition:02d}" / variant
                    directory.mkdir(parents=True)
                    command = request_command(args, variant, drafts)
                    env = clean_environment(sources, variant)
                    cell = {
                        "repetition": repetition,
                        "variant": variant,
                        "command": command,
                        "env": env,
                        "directory": str(directory.resolve()),
                    }
                    report["servers"].append(cell)
                    process, timer = None, None
                    with (directory / "server.log").open("wb") as log:
                        try:
                            with guard.defer():
                                process = subprocess.Popen(
                                    command,
                                    env=env,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                                cell.update(server_pid=process.pid, server_pgid=process.pid)
                                # Abort stalled reads at the aggregate deadline.
                                timer = threading.Timer(
                                    max(0, deadline - time.monotonic()),
                                    _kill_group,
                                    args=(process,),
                                )
                                timer.daemon = True
                                timer.start()
                            benchmark.json_write(manifest, report)
                            _wait_ready(process, url, deadline, stop_file)
                            cell["mapped_runtime"] = verify_mapped_runtime(
                                process.pid, sources["native_runtime"]
                            )
                            for index in range(warmup_requests):
                                measure_request(
                                    url,
                                    benchmark.request_body(request_config, prompt_rows[index % 24]),
                                    directory / "warmup" / f"{index:03d}",
                                    deadline=deadline,
                                    stop_file=stop_file,
                                )
                            for prompt in prompt_rows:
                                row = measure_request(
                                    url,
                                    benchmark.request_body(request_config, prompt),
                                    directory / "measured" / prompt["id"],
                                    deadline=deadline,
                                    stop_file=stop_file,
                                )
                                row.update(
                                    repetition=repetition, variant=variant, prompt_id=prompt["id"]
                                )
                                report["records"].append(row)
                                benchmark.json_write(manifest, report)
                        finally:
                            if timer is not None:
                                timer.cancel()
                            if process is not None:
                                cell["server_stop"] = _stop_owned(process, deadline)
                            benchmark.json_write(manifest, report)
                    if not cell["server_stop"]["process_group_gone"]:
                        raise RuntimeError("owned native timing process group did not return")
                    server_log = (directory / "server.log").read_text(errors="replace")
                    required = (
                        [benchmark.W1AX_LOADER_MARKER, benchmark.W1AX_CUDA_MARKERS["8"]]
                        if variant == "A8"
                        else []
                    )
                    cell["required_dispatch_markers"] = required
                    cell["dispatch_confirmed"] = all(marker in server_log for marker in required)
                    if not cell["dispatch_confirmed"]:
                        raise RuntimeError("A8 native timing dispatch not confirmed")
        verify_sources(sources)
        if (
            any(
                benchmark.sha256(path) != report["drafts_sha256"][key]
                for key, path in drafts.items()
            )
            or benchmark.sha256(prompts) != report["prompts_sha256"]
        ):
            raise ValueError("timing source inputs changed during measurement")
        remaining(deadline, stop_file)
        report.update(complete=True, status="complete")
    except TimeoutError as error:
        report.update(status="deadline_exhausted", error=str(error))
    except BaseException as error:
        expired = time.monotonic() >= deadline and isinstance(error, Exception)
        report.update(
            status="deadline_exhausted" if expired else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        report.update(_summary(report["records"], False))
        benchmark.json_write(manifest, report)
        if not expired:
            raise
    report.update(_summary(report["records"], report["complete"]))
    benchmark.json_write(manifest, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    args = parser.parse_args()
    result = measure_a8_requests(
        json.loads(args.sources.read_text()),
        args.prompts,
        args.draft,
        args.output,
        deadline=args.deadline_monotonic,
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
