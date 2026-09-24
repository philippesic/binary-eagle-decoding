"""Paired, sequential llama-server benchmark with raw artifact preservation.

Run from a repo checkout with converted GGUF files present. This is a harness,
not evidence of a speedup until its reports contain a completed same-device run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("target_only", "ordinary_eagle", "packed_head_w1a1")
MMA_VARIANT = "packed_head_w1a1_mma"
SPEC_COUNTERS = {
    "proposed": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted": "llamacpp:spec_decode_num_accepted_tokens_total",
    "rounds": "llamacpp:spec_decode_num_drafts_total",
}
SAFE_INHERITED_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "OMP_NUM_THREADS",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
)
METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
SPEC_TIMING_LINE = re.compile(
    r"statistics\s+draft-eagle3:.*?dur\(b,g,a\)\s*=\s*"
    r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*ms"
)


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.draw",
        "--format=csv,noheader",
    ]
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "command": command, "error": str(error)}
    result = {
        "available": True,
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        fallback = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader",
        ]
        try:
            simpler = subprocess.run(fallback, capture_output=True, text=True, timeout=15)
            result["fallback"] = {
                "command": fallback,
                "exit_code": simpler.returncode,
                "stdout": simpler.stdout,
                "stderr": simpler.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            result["fallback"] = {"command": fallback, "error": str(error)}
    return result


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    ids = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = row.get("id")
        messages = row.get("messages")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in ids:
            raise ValueError(f"{path}:{number}: missing or duplicate prompt ID")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", prompt_id):
            raise ValueError(f"{path}:{number}: unsafe prompt ID")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{number}: expected nonempty messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"{path}:{number}: invalid text message")
        ids.add(prompt_id)
        prompts.append(row)
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def selected_variants(evaluation: dict[str, Any]) -> tuple[str, ...]:
    enabled = evaluation.get("binary_mma", False)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.binary_mma must be a boolean")
    return (*VARIANTS, MMA_VARIANT) if enabled else VARIANTS


def schedule(repetitions: int, variants: tuple[str, ...] = VARIANTS) -> list[list[str]]:
    if repetitions < 5:
        raise ValueError("at least five measured repetitions are required")
    base = list(variants)
    if len(base) == 4:
        balanced = (
            (0, 1, 2, 3),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (3, 2, 1, 0),
        )
        return [[base[index] for index in balanced[rep % 4]] for rep in range(repetitions)]
    orders = []
    for rep in range(repetitions):
        rotated = base[rep % len(base) :] + base[: rep % len(base)]
        orders.append(rotated if rep % 2 == 0 else list(reversed(rotated)))
    return orders


def parse_metrics(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    metrics = {}
    for line in raw.splitlines():
        match = METRIC_LINE.fullmatch(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def counter_delta(before: str | None, after: str | None) -> dict[str, int | None]:
    old, new = parse_metrics(before), parse_metrics(after)
    result = {}
    for label, name in SPEC_COUNTERS.items():
        if old is None or new is None or name not in old or name not in new:
            result[label] = None
            continue
        difference = new[name] - old[name]
        if difference < 0 or not difference.is_integer():
            result[label] = None
        else:
            result[label] = int(difference)
    return result


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def metrics_text(base_url: str) -> str | None:
    try:
        status, body = request_json(base_url + "/metrics", None, 10)
        return body if status == 200 else None
    except (OSError, TimeoutError, urllib.error.URLError):
        return None


def available_port(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def wait_ready(process: subprocess.Popen[bytes], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup ({process.returncode})")
        try:
            status, body = request_json(base_url + "/health", None, 3)
            if status == 200:
                health = json.loads(body)
                if health.get("status") == "ok":
                    return
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server readiness timeout")


def dispatch_evidence(server_log: str, variant: str) -> dict[str, Any]:
    """Distinguish a packed model load from proof of actual CUDA op execution."""
    packed_loaded = "EAGLE3 using packed W1A1 draft head" in server_log
    cuda_backend = bool(
        re.search(r"CUDA\d+.*(?:model buffer|compute buffer|KV buffer)|ggml_cuda_init", server_log)
    )
    legacy_portable_marker = "CUDA packed W1A1 XOR/POPCOUNT dispatch"
    portable_marker = "CUDA packed W1A1 portable XOR/POPCOUNT dispatch"
    mma_marker = "CUDA packed W1A1 binary MMA dispatch"
    portable_seen = portable_marker in server_log or legacy_portable_marker in server_log
    mma_seen = mma_marker in server_log
    expected_seen = mma_seen if variant == MMA_VARIANT else portable_seen
    unexpected_seen = portable_seen if variant == MMA_VARIANT else mma_seen
    confirmed = (
        variant in ("packed_head_w1a1", MMA_VARIANT) and expected_seen and not unexpected_seen
    )
    return {
        "packed_head_loader_log": packed_loaded,
        "cuda_backend_log": cuda_backend,
        "explicit_cuda_w1a1_op_log": portable_seen or mma_seen,
        "portable_cuda_dispatch_log": portable_seen,
        "binary_mma_cuda_dispatch_log": mma_seen,
        "cuda_w1a1_dispatch_marker": (
            mma_marker
            if mma_seen
            else portable_marker
            if portable_marker in server_log
            else legacy_portable_marker
            if portable_seen
            else None
        ),
        "cuda_w1a1_dispatch_confirmed": True if confirmed else None,
        "interpretation": (
            "The expected explicit CUDA op marker confirms the selected packed dispatch; "
            "model load and general CUDA backend logs alone do not."
        ),
    }


def speculative_timing(
    server_log: str, variant: str, warmup_requests: int, measured_requests: int
) -> dict[str, Any]:
    """Read cumulative EAGLE impl host timings without mixing warmup into measurement."""
    if variant == "target_only":
        return {"status": "not_applicable"}
    cumulative = [
        tuple(float(value) for value in match.groups())
        for match in SPEC_TIMING_LINE.finditer(server_log)
    ]
    expected = warmup_requests + measured_requests
    if len(cumulative) != expected:
        return {
            "status": "unavailable",
            "reason": "cumulative speculative timing lines do not match request count",
            "expected": expected,
            "observed": len(cumulative),
        }
    previous = (0.0, 0.0, 0.0)
    deltas = []
    for current in cumulative:
        delta = tuple(now - before for now, before in zip(current, previous, strict=True))
        if any(value < -1e-9 for value in delta):
            return {"status": "unavailable", "reason": "cumulative timing counter decreased"}
        deltas.append(dict(zip(("begin_ms", "draft_ms", "accept_ms"), delta, strict=True)))
        previous = current
    measured = deltas[warmup_requests:]
    return {
        "status": "available",
        "source": "common_speculative_print_stats cumulative host wall times",
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "measured_per_request": measured,
        "measured_totals_ms": {
            key: sum(item[key] for item in measured)
            for key in ("begin_ms", "draft_ms", "accept_ms")
        },
    }


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        else:
            # A child in the process group can survive the server leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def command_for(config: dict[str, Any], paths: dict[str, Path], variant: str) -> list[str]:
    if variant not in (*VARIANTS, MMA_VARIANT):
        raise ValueError(f"unknown benchmark variant: {variant}")
    command = [str(paths["binary"]), "-m", str(paths["target"])]
    if variant == "target_only":
        command += ["--spec-type", "none"]
    else:
        draft_key = "ordinary_draft" if variant == "ordinary_eagle" else "packed_head_draft"
        command += [
            "-md",
            str(paths[draft_key]),
            "--spec-type",
            "draft-eagle3",
            "--spec-draft-n-max",
            str(config["evaluation"]["max_draft_tokens"]),
            "--spec-draft-ngl",
            "all",
            "--spec-draft-type-k",
            "f16",
            "--spec-draft-type-v",
            "f16",
        ]
    command += list(config["server"]["common_args"])
    command += ["--host", config["server"]["host"], "--port", str(config["server"]["port"])]
    return command


def request_body(config: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "messages": prompt["messages"],
        "max_tokens": evaluation["max_output_tokens"],
        "temperature": evaluation["temperature"],
        "seed": evaluation["seed"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": evaluation["enable_thinking"]},
        "reasoning_format": "none",
    }


def extract_record(
    response: dict[str, Any], elapsed_s: float, counters: dict[str, int | None]
) -> dict[str, Any]:
    timing = response.get("timings") or {}
    usage = response.get("usage") or {}
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int):
        tokens = timing.get("predicted_n") if isinstance(timing.get("predicted_n"), int) else None
    predicted_ms = timing.get("predicted_ms")
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    content = choices[0].get("message", {}).get("content") if choices else None
    return {
        "request_wall_s": elapsed_s,
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "server_predicted_n": timing.get("predicted_n"),
        "server_predicted_ms": predicted_ms,
        "server_prompt_n": timing.get("prompt_n"),
        "server_prompt_ms": timing.get("prompt_ms"),
        "request_tokens_per_s": tokens / elapsed_s
        if isinstance(tokens, int) and elapsed_s > 0
        else None,
        "decode_tokens_per_s": tokens * 1000 / predicted_ms
        if isinstance(tokens, int) and isinstance(predicted_ms, (int, float)) and predicted_ms > 0
        else None,
        "time_to_first_token_s": None,  # nonstreaming endpoint does not expose it
        "finish_reason": finish_reason,
        "completion_sha256": (
            hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None
        ),
        "speculative": counters,
    }


def execute_request(
    base_url: str, body: dict[str, Any], timeout: float, directory: Path
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    json_write(directory / "request.json", body)
    before = metrics_text(base_url)
    if before is not None:
        (directory / "metrics-before.prom").write_text(before)
    start = time.perf_counter()
    status, raw = request_json(base_url + "/v1/chat/completions", body, timeout)
    elapsed = time.perf_counter() - start
    (directory / "response.json").write_text(raw)
    after = metrics_text(base_url)
    if after is not None:
        (directory / "metrics-after.prom").write_text(after)
    if status != 200:
        raise RuntimeError(f"request failed with HTTP {status}; see {directory / 'response.json'}")
    response = json.loads(raw)
    result = extract_record(response, elapsed, counter_delta(before, after))
    json_write(directory / "measurement.json", result)
    return result


def aggregate(
    records: list[dict[str, Any]], variants: tuple[str, ...] = VARIANTS
) -> dict[str, Any]:
    output = {}
    for variant in variants:
        rows = [row for row in records if row["variant"] == variant]
        n_tokens = [row["completion_tokens"] for row in rows]
        wall = [row["request_wall_s"] for row in rows]
        decode = [row["server_predicted_ms"] for row in rows]
        counters = {label: [row["speculative"][label] for row in rows] for label in SPEC_COUNTERS}
        sums = {
            label: sum(values) if all(v is not None for v in values) else None
            for label, values in counters.items()
        }
        valid_tokens = all(isinstance(v, int) for v in n_tokens)
        valid_decode = all(isinstance(v, (int, float)) and v > 0 for v in decode)
        output[variant] = {
            "requests": len(rows),
            "completion_tokens": sum(n_tokens) if valid_tokens else None,
            "request_wall_s": sum(wall),
            "decode_server_s": sum(decode) / 1000 if valid_decode else None,
            "request_tokens_per_s": sum(n_tokens) / sum(wall)
            if valid_tokens and sum(wall) > 0
            else None,
            "decode_tokens_per_s": 1000 * sum(n_tokens) / sum(decode)
            if valid_tokens and valid_decode
            else None,
            "speculative": sums,
            "accepted_per_round": sums["accepted"] / sums["rounds"]
            if sums["accepted"] is not None and sums["rounds"]
            else None,
            "acceptance_rate": sums["accepted"] / sums["proposed"]
            if sums["accepted"] is not None and sums["proposed"]
            else None,
            "emitted_tokens_per_round": (
                sum(n_tokens) / sums["rounds"] if valid_tokens and sums["rounds"] else None
            ),
            "finish_reason_counts": {
                reason: sum(row.get("finish_reason") == reason for row in rows)
                for reason in sorted(
                    {row.get("finish_reason") for row in rows if row.get("finish_reason")}
                )
            },
        }
    return output


def completion_text_matches(
    records: list[dict[str, Any]], variants: tuple[str, ...] = VARIANTS
) -> dict[str, Any]:
    """Compare decoded response text hashes for matched repetition/prompt pairs."""
    by_key = {(row["repetition"], row["prompt_id"], row["variant"]): row for row in records}
    result = {}
    for variant in variants:
        if variant == "target_only":
            continue
        matched = 0
        unavailable = 0
        mismatches = []
        for row in records:
            if row["variant"] != variant:
                continue
            key = (row["repetition"], row["prompt_id"], "target_only")
            reference = by_key.get(key)
            if (
                reference is None
                or not row.get("completion_sha256")
                or not reference.get("completion_sha256")
            ):
                unavailable += 1
            elif row["completion_sha256"] == reference["completion_sha256"]:
                matched += 1
            else:
                mismatches.append({"repetition": row["repetition"], "prompt_id": row["prompt_id"]})
        result[variant] = {
            "matched_text": matched,
            "mismatched_text": len(mismatches),
            "unavailable": unavailable,
            "mismatch_pairs": mismatches,
        }
    return result


def relative_speedups(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Compute packed/anchor ratios from pooled token/time totals only."""
    packed = aggregated["packed_head_w1a1"]
    result = {}
    for anchor in ("target_only", "ordinary_eagle"):
        baseline = aggregated[anchor]
        result[anchor] = {}
        for metric in ("request_tokens_per_s", "decode_tokens_per_s"):
            numerator = packed[metric]
            denominator = baseline[metric]
            result[anchor][metric] = (
                numerator / denominator
                if isinstance(numerator, (int, float))
                and isinstance(denominator, (int, float))
                and denominator > 0
                else None
            )
    return result


def mma_speedup_vs_portable(aggregated: dict[str, Any]) -> dict[str, float | None]:
    """Use pooled token/time rates for an MMA/portable comparison."""
    mma = aggregated[MMA_VARIANT]
    portable = aggregated["packed_head_w1a1"]
    result = {}
    for metric in ("request_tokens_per_s", "decode_tokens_per_s"):
        numerator = mma[metric]
        denominator = portable[metric]
        result[metric] = (
            numerator / denominator
            if isinstance(numerator, (int, float))
            and isinstance(denominator, (int, float))
            and denominator > 0
            else None
        )
    return result


def repetition_summaries(
    records: list[dict[str, Any]], variants: tuple[str, ...] = VARIANTS
) -> list[dict[str, Any]]:
    """Preserve run-to-run spread without averaging speedup ratios globally."""
    summaries = []
    for repetition in sorted({row["repetition"] for row in records}):
        selected = [row for row in records if row["repetition"] == repetition]
        aggregated = aggregate(selected, variants)
        summary = {
            "repetition": repetition,
            "aggregation": aggregated,
            "packed_speedup_vs": relative_speedups(aggregated),
        }
        if MMA_VARIANT in variants:
            summary["mma_speedup_vs_portable"] = mma_speedup_vs_portable(aggregated)
        summaries.append(summary)
    return summaries


def summarize_draft_timings(
    timings: list[dict[str, Any]],
    aggregated: dict[str, Any],
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    """Pool measured-only speculative impl timings; keep unavailable explicit."""
    result = {}
    for variant in variants:
        if variant == "target_only":
            continue
        rows = [entry["timing"] for entry in timings if entry["variant"] == variant]
        if not rows or any(row["status"] != "available" for row in rows):
            result[variant] = {"status": "unavailable"}
            continue
        totals = {
            key: sum(row["measured_totals_ms"][key] for row in rows)
            for key in ("begin_ms", "draft_ms", "accept_ms")
        }
        rounds = aggregated[variant]["speculative"]["rounds"]
        result[variant] = {
            "status": "available",
            "measured_totals_ms": totals,
            "draft_ms_per_verification_round": totals["draft_ms"] / rounds if rounds else None,
            "definition": (
                "host wall time inside common_speculative_impl draft calls, "
                "excluding warmup; not a standalone GPU kernel duration"
            ),
        }
    return result


def all_dispatches_confirmed(destination: Path, repetitions: int, variant: str) -> bool | None:
    return (
        True
        if all(
            json.loads(
                (destination / f"rep-{rep:02d}" / variant / "dispatch-evidence.json").read_text()
            )["cuda_w1a1_dispatch_confirmed"]
            for rep in range(repetitions)
        )
        else None
    )


def run(config_path: Path, run_id: str, *, dry_run: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run ID must be a safe, relative name")
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    evaluation = config["evaluation"]
    variants = selected_variants(evaluation)
    orders = schedule(evaluation["repetitions"], variants)
    if evaluation["warmup_requests"] < 0 or evaluation["max_output_tokens"] <= 0:
        raise ValueError("invalid warmup or max output setting")
    if config["server"]["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("server must bind loopback for this local harness")
    paths = {
        name: resolve(ROOT, value)
        for name, value in {
            "binary": config["server"]["binary"],
            **config["models"],
            "prompt_file": evaluation["prompt_file"],
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    prompts = load_prompts(paths["prompt_file"])
    if not available_port(config["server"]["host"], config["server"]["port"]):
        raise RuntimeError("configured server port is already occupied")
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(config_path, destination / "config.toml")
        shutil.copy2(paths["prompt_file"], destination / "prompts.jsonl")
        env = {key: os.environ[key] for key in SAFE_INHERITED_ENV if key in os.environ}
        env.update(config.get("environment", {}))
        env.pop("GGML_CUDA_W1A1_MMA", None)
        variant_environments = {
            variant: {**env, "GGML_CUDA_W1A1_MMA": "1" if variant == MMA_VARIANT else "0"}
            for variant in variants
        }
        (destination / "project-diff.patch").write_text(git_output("diff", "--binary", "HEAD"))
        manifest = {
            "started_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "config_source": str(config_path),
            "config_sha256": sha256(config_path),
            "harness_sha256": sha256(Path(__file__)),
            "files": {
                name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in paths.items()
            },
            "environment": env,
            "variant_environments": variant_environments,
            "variants": list(variants),
            "orders": orders,
            "commands": {variant: command_for(config, paths, variant) for variant in variants},
            "request_options": {
                key: value
                for key, value in request_body(config, prompts[0]).items()
                if key != "messages"
            },
            "prompt_ids": [prompt["id"] for prompt in prompts],
            "precision": config.get("precision"),
            "initial_gpu_snapshot": gpu_snapshot(),
            "project_commit": git_output("rev-parse", "HEAD").strip(),
            "project_status": git_output("status", "--porcelain"),
            "project_diff_sha256": sha256(destination / "project-diff.patch"),
            "llama_gitlink": git_output("ls-tree", "HEAD", "third_party/llama.cpp").split()[2],
        }
        json_write(destination / "manifest.json", manifest)
        if dry_run:
            return destination
        base_url = f"http://{config['server']['host']}:{config['server']['port']}"
        records = []
        timing_rows = []
        try:
            for repetition, order in enumerate(orders):
                for variant in order:
                    server_dir = destination / f"rep-{repetition:02d}" / variant
                    server_dir.mkdir(parents=True)
                    json_write(server_dir / "gpu-before.json", gpu_snapshot())
                    command = manifest["commands"][variant]
                    variant_env = variant_environments[variant]
                    json_write(server_dir / "environment.json", variant_env)
                    with (server_dir / "server.log").open("wb") as log:
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=variant_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name == "posix",
                        )
                        try:
                            wait_ready(process, base_url, config["server"]["startup_timeout_s"])
                            props_status, props_raw = request_json(base_url + "/props", None, 10)
                            (server_dir / "props.json").write_text(props_raw)
                            if props_status != 200:
                                raise RuntimeError(f"/props returned HTTP {props_status}")
                            json_write(server_dir / "gpu-loaded.json", gpu_snapshot())
                            for index in range(evaluation["warmup_requests"]):
                                execute_request(
                                    base_url,
                                    request_body(config, prompts[index % len(prompts)]),
                                    config["server"]["request_timeout_s"],
                                    server_dir / "warmup" / f"request-{index:02d}",
                                )
                            for prompt in prompts:
                                measurement = execute_request(
                                    base_url,
                                    request_body(config, prompt),
                                    config["server"]["request_timeout_s"],
                                    server_dir / "measured" / prompt["id"],
                                )
                                measurement.update(
                                    {
                                        "repetition": repetition,
                                        "variant": variant,
                                        "prompt_id": prompt["id"],
                                        "w1a1_mma_selector": variant_env["GGML_CUDA_W1A1_MMA"],
                                    }
                                )
                                records.append(measurement)
                                json_write(destination / "records.json", records)
                        finally:
                            stop_server(process)
                            log.flush()
                            evidence = dispatch_evidence(
                                (server_dir / "server.log").read_text(errors="replace"), variant
                            )
                            json_write(server_dir / "dispatch-evidence.json", evidence)
                            timing = speculative_timing(
                                (server_dir / "server.log").read_text(errors="replace"),
                                variant,
                                evaluation["warmup_requests"],
                                len(prompts),
                            )
                            json_write(server_dir / "speculative-timing.json", timing)
                            timing_rows.append(
                                {"repetition": repetition, "variant": variant, "timing": timing}
                            )
                            json_write(server_dir / "gpu-after.json", gpu_snapshot())
            aggregated = aggregate(records, variants)
            report = {
                "status": "complete",
                "variants": list(variants),
                "aggregation": aggregated,
                "packed_speedup_vs": relative_speedups(aggregated),
                "repetitions": repetition_summaries(records, variants),
                "greedy_text_match_vs_target_only": completion_text_matches(records, variants),
                "draft_timing": summarize_draft_timings(timing_rows, aggregated, variants),
                "records": len(records),
                "native_cuda_dispatch_confirmed": all_dispatches_confirmed(
                    destination, evaluation["repetitions"], "packed_head_w1a1"
                ),
                "definition": {
                    "request": "completion tokens / client wall time including prefill",
                    "decode": "completion tokens / sum of server predicted_ms (excludes prompt_ms)",
                    "acceptance": (
                        "accepted draft tokens / proposed draft tokens "
                        "from per-request metrics deltas"
                    ),
                    "greedy_text_match": (
                        "SHA256 equality of decoded response text for matched prompt/repetition; "
                        "does not prove token-ID or stochastic distribution equivalence"
                    ),
                },
            }
            if MMA_VARIANT in variants:
                report["mma_cuda_dispatch_confirmed"] = all_dispatches_confirmed(
                    destination, evaluation["repetitions"], MMA_VARIANT
                )
                report["mma_speedup_vs_portable"] = mma_speedup_vs_portable(aggregated)
            json_write(destination / "report.json", report)
        except BaseException as error:
            json_write(
                destination / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "time_utc": datetime.now(UTC).isoformat(),
                },
            )
            raise
        return destination
    except BaseException:
        # Keep manifest and raw files for diagnosis, including failed preflight.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native_benchmark.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and write a manifest without launching the server",
    )
    args = parser.parse_args()
    print(run(args.config.resolve(), args.run_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
