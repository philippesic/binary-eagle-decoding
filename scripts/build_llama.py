"""Build the pinned upstream runtime without modifying its source tree."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "metal", "cuda"))
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
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
        "-DLLAMA_BUILD_TESTS=OFF",
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
        options.append("-DCMAKE_CUDA_ARCHITECTURES=75")
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", *options], check=True
    )
    if not args.configure_only:
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--parallel",
                str(args.jobs),
                "--target",
                "llama-server",
                "llama-cli",
                "llama-bench",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
