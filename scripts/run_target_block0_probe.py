#!/usr/bin/env python3
"""Build and run one supervised RTX 5080 block-0 operator diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from check_target_block0_capture import CAPTURE_MODES, audit, prepare, sha256

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, timeout: int) -> None:
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--reuse-build-run-id")
    parser.add_argument("--capture-mode", choices=tuple(CAPTURE_MODES), default="all")
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    if not (run_dir / "state.json").is_file():
        parser.error("start this probe through scripts/remote_job.py")
    ladder = args.ladder_dir.resolve()
    target = args.target_gguf.resolve()
    cmake_exe = Path(sys.executable).parent / "cmake"
    if not cmake_exe.is_file():
        parser.error("project Python environment lacks its bundled CMake")
    cuda_compat = ROOT.parents[1] / "results/cuda-glibc-compat/include"
    if not (cuda_compat / "crt/host_config.h").is_file():
        parser.error("registered GPU project lacks its pinned CUDA/glibc compatibility headers")
    build = run_dir / "build"
    helper = build / "bin" / "native-target-block0-capture"
    capture = run_dir / "block0"
    cmake = [
        str(cmake_exe),
        "-S",
        str(ROOT / "third_party/llama.cpp"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA=ON",
        f"-DCMAKE_CUDA_FLAGS=-I{cuda_compat}",
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
        "reused_build_run_id": args.reuse_build_run_id,
        "capture_mode": args.capture_mode,
    }
    (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    if args.reuse_build_run_id is not None:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.reuse_build_run_id)
            or args.reuse_build_run_id == args.run_id
        ):
            parser.error("reuse-build-run-id must name a distinct existing run")
        source_run = ROOT / "runs" / args.reuse_build_run_id
        source_state = json.loads((source_run / "state.json").read_text())
        source_recipe = json.loads((source_run / "recipe.json").read_text())
        source_gitlink = subprocess.check_output(
            ["git", "ls-tree", source_recipe["source_commit"], "third_party/llama.cpp"],
            cwd=ROOT,
            text=True,
        ).split()[2]
        current_gitlink = subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/llama.cpp"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if (
            source_state.get("status") != "finished"
            or source_gitlink != current_gitlink
            or not (source_run / "build/CMakeCache.txt").is_file()
            or not (source_run / "build/bin/native-target-block0-capture").is_file()
        ):
            raise ValueError("reused build is unfinished or differs from pinned llama.cpp")
        build.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_run / "build/CMakeCache.txt", build / "CMakeCache.txt")
        shutil.copytree(source_run / "build/bin", build / "bin", symlinks=True)
        recipe["source_sha256"]["reused_cmake_cache"] = sha256(source_run / "build/CMakeCache.txt")
        (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    else:
        _run(cmake, timeout=900)
        _run(
            [str(cmake_exe), "--build", str(build), "--target", "llama", "ggml-cuda", "-j", "8"],
            timeout=900,
        )
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
    helper_command = [str(helper), str(target), str(capture / "tokens.i32"), str(capture)]
    if args.capture_mode != "all":
        helper_command.append(args.capture_mode)
    _run(helper_command, timeout=300)
    report = audit(ladder, target, capture, helper, mode=args.capture_mode)
    report["elapsed_seconds"] = time.monotonic() - started
    report["source_sha256"]["recipe"] = sha256(run_dir / "recipe.json")
    report["source_sha256"]["cmake_cache"] = sha256(build / "CMakeCache.txt")
    for library in ("libllama.so", "libggml-base.so", "libggml-cuda.so"):
        path = build / "bin" / library
        if not path.is_file():
            raise ValueError(f"built target backend library is missing: {library}")
        report["source_sha256"][library] = sha256(path)
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "block_output": report["block_output"]}))
    if report["status"] != "same_native_block_output":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
