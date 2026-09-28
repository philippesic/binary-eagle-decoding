#!/usr/bin/env python3
"""Capture one frozen training prompt with pinned D on a checked native server.

CPU mode disables every accelerator and GPU layer. CUDA mode requires only
the CUDA accelerator, offloads both models, and omits CPU-only graph/cache
hooks. Both modes stop the server process group and avoid final prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from audit_recurrent_continuity import audit_internal_continuity
from audit_recurrent_draft_cache import audit as audit_draft_cache
from audit_recurrent_response import audit_response
from capture_w1ax_activations import (
    generated_token_ids,
    http_json,
    read_jsonl,
    sha256,
    stop_server,
    wait_ready,
)
from run_binary_head_capture import (
    DRAFT_D_SHA256,
    TARGET_F16_SHA256,
    TRAIN_PROMPTS_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE_REVISION = "0abe6e5868d32eb74fa0d4c3dcb963b6a55fe568"
CPU_BACKENDS = (
    "CUDA",
    "METAL",
    "VULKAN",
    "SYCL",
    "HIP",
    "RPC",
    "OPENCL",
    "BLAS",
    "ACCELERATE",
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_inputs(
    binary: Path,
    cmake_cache: Path,
    target: Path,
    draft: Path,
    train_prompts: Path,
    prompt_id: str,
    max_tokens: int,
    device: str = "cpu",
) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError("native diagnostic device must be cpu or cuda")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 16:
        raise ValueError("native diagnostic output cap must be 1..16")
    if not prompt_id.startswith("qat-revisit-train-") or "final" in prompt_id.lower():
        raise ValueError("native diagnostic requires a frozen training prompt")
    if binary.resolve() != (cmake_cache.parent / "bin/llama-server").resolve():
        raise ValueError("server binary must belong to the checked CPU-only CMake build")
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/llama.cpp"), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != NATIVE_REVISION:
        raise ValueError("native diagnostic source revision changed")
    cache = cmake_cache.read_text()
    for backend in CPU_BACKENDS:
        required = "ON" if backend == "CUDA" and device == "cuda" else "OFF"
        if f"GGML_{backend}:BOOL={required}" not in cache.splitlines():
            raise ValueError(f"{device.upper()} diagnostic requires GGML_{backend}={required}")
    if (
        sha256(target) != TARGET_F16_SHA256
        or sha256(draft) != DRAFT_D_SHA256
        or sha256(train_prompts) != TRAIN_PROMPTS_SHA256
    ):
        raise ValueError("native diagnostic target, D draft or training split hash changed")
    prompts = [json.loads(line) for line in train_prompts.read_text().splitlines() if line.strip()]
    matches = [row for row in prompts if row.get("id") == prompt_id]
    if len(prompts) != 96 or len(matches) != 1 or not isinstance(matches[0].get("messages"), list):
        raise ValueError("native diagnostic prompt ID is missing from frozen training split")
    return matches[0]


def native_command(
    binary: Path, target: Path, draft: Path, port: int, flash_attention: str, device: str = "cpu"
) -> list[str]:
    if (
        device not in {"cpu", "cuda"}
        or flash_attention not in {"auto", "off"}
        or not 1024 <= port <= 65535
    ):
        raise ValueError(
            "native diagnostic needs cpu/cuda, a valid port and auto/off Flash Attention"
        )
    gpu_layers = "all" if device == "cuda" else "0"
    return [
        str(binary.resolve()),
        "-m",
        str(target.resolve()),
        "--n-gpu-layers",
        gpu_layers,
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
        str(port),
        "-md",
        str(draft.resolve()),
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-n-max",
        "5",
        "--spec-draft-p-min",
        "0",
        "--spec-draft-ngl",
        gpu_layers,
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--no-spec-draft-backend-sampling",
        "--no-context-shift",
        "--no-cache-prompt",
        "--cache-reuse",
        "0",
        "--ctx-checkpoints",
        "0",
        "--flash-attn",
        flash_attention,
    ]


def capture_one(args: argparse.Namespace) -> dict:
    if args.capture_cache and args.device != "cpu":
        raise ValueError("stored draft cache capture requires CPU mode")
    if args.capture_cache and args.flash_attention != "auto":
        raise ValueError("draft cache capture requires Flash Attention auto")
    prompt = validate_inputs(
        args.binary,
        args.cmake_cache,
        args.target,
        args.draft,
        args.train_prompts,
        args.prompt_id,
        args.max_tokens,
        args.device,
    )
    command = native_command(
        args.binary, args.target, args.draft, args.port, args.flash_attention, args.device
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    body = {
        "messages": prompt["messages"],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
        "return_tokens": True,
        "verbose": True,
    }
    _write(output / "prompt.json", prompt)
    _write(output / "request.json", body)
    capture_env = {
        "CUDA_VISIBLE_DEVICES": "0" if args.device == "cuda" else "",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "GGML_W1AX_ACT_BITS": "16",
        "EAGLE_CAPTURE_PREFIX": str(output / "heads"),
        "EAGLE_RECORD_ROUNDS_JSONL": str(output / "forced-rounds.jsonl"),
        "EAGLE_STATE_TRACE_JSONL": str(output / "state.jsonl"),
        "EAGLE_REQUEST_DIGEST": "1",
        "W1AX_ROUND_TRACE_JSONL": str(output / "rounds.jsonl"),
        "EAGLE_CAPTURE_TARGET_LOGITS": "1",
        "EAGLE_CAPTURE_TARGET_LOGITS_LIMIT": "32",
        "EAGLE_CAPTURE_TARGET_FEATURES": "1",
        "EAGLE_CAPTURE_TARGET_FEATURES_LIMIT": "1024",
    }
    if args.device == "cpu":
        capture_env.update(
            EAGLE_CAPTURE_DRAFT_GRAPH="1",
            EAGLE_CAPTURE_DRAFT_GRAPH_MAX_EXECUTIONS="512",
            EAGLE_CAPTURE_DRAFT_GRAPH_MAX_BYTES=str(128 * 1024 * 1024),
        )
    if args.capture_cache:
        capture_env["EAGLE_CAPTURE_DRAFT_CACHE"] = "1"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GGML_", "CUDA_", "LLAMA_ARG_", "EAGLE_", "W1AX_"))
    }
    environment.update(capture_env)
    _write(output / "command.json", {"cmd": command, "env": capture_env})
    with (output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            wait_ready(base_url, process, 300)
            response = http_json(base_url + "/v1/chat/completions", body, 300)
            _write(output / "response.json", response)
        finally:
            stopped = stop_server(process)
            _write(output / "server_stop.json", stopped)
    if not stopped["stopped"] or stopped["return_code"] != 0:
        raise RuntimeError("native diagnostic server did not stop cleanly")
    ids = generated_token_ids(response)
    if not ids or len(ids) > args.max_tokens:
        raise ValueError("native diagnostic response has no bounded raw output IDs")
    heads = read_jsonl(output / "heads.jsonl")
    rounds = read_jsonl(output / "forced-rounds.jsonl")
    events = read_jsonl(output / "heads.target_features.jsonl")
    tasks = {str(row.get("task_id")) for row in heads + rounds}
    if len(tasks) != 1 or not heads or not rounds or not events:
        raise ValueError("native diagnostic raw capture lacks one owned native task")
    task_id = next(iter(tasks))
    task_map = output / "task_prompt_ids.json"
    _write(task_map, {task_id: prompt["id"]})
    continuity = audit_internal_continuity(
        output / "forced-rounds.jsonl", output / "heads.target_features.jsonl", task_map
    )
    _write(output / "continuity.json", continuity)
    response_audit = audit_response(
        output / "forced-rounds.jsonl",
        output / "rounds.jsonl",
        output / "heads.target_features.jsonl",
        task_map,
        output / "request.json",
        output / "response.json",
        task_id,
    )
    _write(output / "response_round_join.json", response_audit)
    footer = None
    if args.device == "cpu":
        graph_rows = read_jsonl(output / "heads.draft_graph.jsonl")
        footer = graph_rows[-1]
        if footer.get("event") != "capture_end" or footer.get("status") != "complete":
            raise ValueError("CPU diagnostic draft graph capture is incomplete")
    if args.capture_cache:
        cache_rows = read_jsonl(output / "heads.draft_cache.jsonl")
        cache_footer = cache_rows[-1] if cache_rows else {}
        if (
            cache_footer.get("event") != "capture_end"
            or cache_footer.get("executions") != footer["decoder_groups"]
            or cache_footer.get("rows") != sum(row.get("event") == "row" for row in cache_rows)
            or cache_footer.get("row_bytes") != (output / "heads.draft_cache.f16").stat().st_size
            or cache_footer.get("mask_bytes") != (output / "heads.draft_cache.mask").stat().st_size
        ):
            raise ValueError("CPU diagnostic draft cache capture is incomplete")
        _write(output / "cache_audit.json", audit_draft_cache(output))
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema": (
            "recurrent_cpu_native_diagnostic_v1"
            if args.device == "cpu"
            else "recurrent_cuda_native_diagnostic_v1"
        ),
        "execution_device": args.device,
        "training_eligible": False,
        "kind": "instrumented_capture_not_timing",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "prompt_id": prompt["id"],
        "generated_token_ids": ids,
        "flash_attention": args.flash_attention,
        "native_revision": NATIVE_REVISION,
        "source_sha256": {
            "binary": sha256(args.binary),
            "cmake_cache": sha256(args.cmake_cache),
            "target": sha256(args.target),
            "draft": sha256(args.draft),
            "frozen_train_prompts": sha256(args.train_prompts),
        },
        "capture_counts": {
            "rounds": len(rounds),
            "head_rows": len(heads),
            "target_feature_rows": sum(row.get("event") == "decoded_row" for row in events),
            "draft_decoder_groups": footer["decoder_groups"] if footer else None,
            "draft_encoder_groups": footer["encoder_groups"] if footer else None,
            "draft_cache_rows": cache_footer["rows"] if args.capture_cache else None,
        },
        "files": files,
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--flash-attention", choices=("auto", "off"), default="auto")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--capture-cache", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--port", type=int, default=18557)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = capture_one(args)
    print(
        json.dumps(
            {
                "prompt_id": manifest["prompt_id"],
                "output_ids": len(manifest["generated_token_ids"]),
                "rounds": manifest["capture_counts"]["rounds"],
                "decoder_groups": manifest["capture_counts"]["draft_decoder_groups"],
            }
        )
    )


if __name__ == "__main__":
    main()
