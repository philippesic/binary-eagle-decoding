#!/usr/bin/env python3
"""Build and run one supervised RTX 5080 block-0 operator diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from check_target_block0_capture import audit, prepare, sha256

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, timeout: int) -> None:
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    if not (run_dir / "state.json").is_file():
        parser.error("start this probe through scripts/remote_job.py")
    ladder = args.ladder_dir.resolve()
    target = args.target_gguf.resolve()
    build = run_dir / "build"
    helper = build / "bin" / "native-target-block0-capture"
    capture = run_dir / "block0"
    cmake = [
        "cmake",
        "-S",
        str(ROOT / "third_party/llama.cpp"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DBUILD_SHARED_LIBS=ON",
    ]
    started = time.monotonic()
    recipe = {
        "schema": "target_block0_run_recipe_v1",
        "run_id": args.run_id,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            "helper": sha256(ROOT / "scripts/native_target_block0_capture.cpp"),
            "validator": sha256(ROOT / "scripts/check_target_block0_capture.py"),
            "runner": sha256(Path(__file__)),
        },
        "cmake": cmake,
        "build_target": ["llama", "ggml-cuda"],
        "target_gguf": str(target),
        "ladder_dir": str(ladder),
    }
    (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    _run(cmake, timeout=900)
    _run(["cmake", "--build", str(build), "--target", "llama", "ggml-cuda", "-j", "8"], timeout=900)
    compiler = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I" + str(ROOT / "third_party/llama.cpp/include"),
        "-I" + str(ROOT / "third_party/llama.cpp/ggml/include"),
        str(ROOT / "scripts/native_target_block0_capture.cpp"),
        "-L" + str(build / "bin"),
        "-Wl,-rpath,$ORIGIN",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-lggml-cpu",
        "-o",
        str(helper),
    ]
    _run(compiler, timeout=120)
    prepare(ladder, target, capture)
    _run([str(helper), str(target), str(capture / "tokens.i32"), str(capture)], timeout=300)
    report = audit(ladder, target, capture, helper)
    report["elapsed_seconds"] = time.monotonic() - started
    report["source_sha256"]["recipe"] = sha256(run_dir / "recipe.json")
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "block_output": report["block_output"]}))
    if report["status"] != "same_native_block_output":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
