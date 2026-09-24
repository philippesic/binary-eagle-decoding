"""Build the pinned upstream runtime without modifying its source tree."""

import argparse
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "metal", "cuda"))
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    parser.add_argument("--cuda-arch", default="75", help="CUDA compute capability, e.g. 120 or 75")
    parser.add_argument(
        "--cuda-include-root",
        type=Path,
        help="Private CUDA include tree for a documented host-toolkit compatibility fix",
    )
    parser.add_argument("--with-tests", action="store_true", help="Build test-backend-ops too")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not re.fullmatch(r"[0-9]+", args.cuda_arch):
        parser.error("--cuda-arch must be a numeric compute capability")
    if args.cuda_include_root is not None:
        if args.backend != "cuda":
            parser.error("--cuda-include-root requires the CUDA backend")
        if not args.cuda_include_root.is_dir():
            parser.error("--cuda-include-root must be an existing directory")
    source = ROOT / "third_party" / "llama.cpp"
    if not (source / "CMakeLists.txt").exists():
        parser.error("Missing llama.cpp: run make setup first")
    for command in ("cmake", "ninja"):
        if not shutil.which(command):
            parser.error(f"Missing {command}: run through uv after make setup")
    if args.backend == "metal" and platform.system() != "Darwin":
        parser.error("Metal builds require macOS")
    if args.backend == "cuda" and not shutil.which("nvcc"):
        parser.error("CUDA builds require nvcc on PATH (benchmark target: SM75)")

    build = ROOT / "build" / f"llama-{args.backend}"
    options = [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DLLAMA_BUILD_TESTS={'ON' if args.with_tests else 'OFF'}",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_OPENSSL=OFF",  # Local model files; no HTTPS dependency for smoke builds.
        f"-DGGML_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}",
        f"-DGGML_METAL={'ON' if args.backend == 'metal' else 'OFF'}",
    ]
    if args.backend == "cuda":
        options.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
        if args.cuda_include_root is not None:
            options.append(f"-DCMAKE_CUDA_FLAGS=-I{args.cuda_include_root.resolve()}")
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", *options], check=True
    )
    if not args.configure_only:
        targets = ["llama-server", "llama-cli", "llama-bench"]
        if args.with_tests:
            targets.append("test-backend-ops")
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--parallel",
                str(args.jobs),
                "--target",
                *targets,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
