#!/usr/bin/env python3
"""Forward observed Nsight injection through a fresh profiling-only TOML config.

Run this wrapper inside `nsys profile`, under the existing remote job supervisor.
It replaces itself with the requested Python runner without creating a process group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

NSIGHT_KEYS = (
    "NSYS_CONTROL_CUPTI_FEATURES",
    "CUDA_INJECTION64_PATH",
    "NSYS_TARGET_STD_REDIRECT_ACTION",
    "NSYS_SESSION_TEMP_DIR",
    "NSYS_PROFILING_SESSION_ID",
    "NSYS_TSC_SUPPORT",
    "NSYS_CUDA_GPU_TIME_CONVERSION",
    "NSYS_AGENT_TMP_DIR",
    "LD_PRELOAD",
)
REQUIRED_KEYS = ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "NSYS_PROFILING_SESSION_ID")
ENVIRONMENT_HEADER = re.compile(r"(?m)^[ \t]*\[environment\][ \t]*(?:#[^\r\n]*)?$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profiler_environment(environ: Mapping[str, str], nsys_root: Path) -> tuple[dict[str, str], list[str]]:
    if not nsys_root.is_absolute() or not nsys_root.is_dir():
        raise ValueError("--nsys-root must be an existing absolute Nsight installation directory")
    root = nsys_root.resolve()
    missing = [key for key in REQUIRED_KEYS if not environ.get(key, "").strip()]
    if missing:
        raise ValueError("live Nsight environment missing: " + ", ".join(missing))
    selected = {key: environ[key] for key in NSIGHT_KEYS if key in environ}
    libraries = [selected["CUDA_INJECTION64_PATH"], *re.split(r"[\s:]+", selected["LD_PRELOAD"].strip())]
    resolved_libraries = []
    for value in libraries:
        library = Path(value)
        if not library.is_absolute() or not library.is_file() or not library.resolve().is_relative_to(root):
            raise ValueError(f"profiler injection library must be a file within --nsys-root: {value}")
        resolved_libraries.append(str(library.resolve()))
    return selected, resolved_libraries


def inject_environment(source: bytes, selected: dict[str, str]) -> bytes:
    """Insert only missing environment keys, preserving the original TOML text."""
    text = source.decode("utf-8")
    before = tomllib.loads(text)
    environment = before.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("config environment must be a TOML table")
    for key, value in selected.items():
        if key in environment and environment[key] != value:
            raise ValueError(f"config conflicts with observed Nsight environment: {key}")
    additions = "".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
        for key, value in selected.items() if key not in environment
    )
    header = ENVIRONMENT_HEADER.search(text)
    if header:
        text = text[:header.end()] + "\n" + additions + text[header.end():]
    elif "environment" not in before:
        text = text.rstrip("\n") + "\n\n[environment]\n" + additions
    else:
        raise ValueError("config must use an explicit [environment] table")
    expected = {**before, "environment": {**environment, **selected}}
    if tomllib.loads(text) != expected:
        raise ValueError("profile config changed settings outside the allowed environment keys")
    return text.encode("utf-8")


def prepare_profile(
    runner: Path,
    config: Path,
    profile_config: Path,
    nsys_root: Path,
    runner_args: list[str],
    environ: Mapping[str, str],
) -> list[str]:
    if not runner.is_absolute() or not runner.is_file():
        raise ValueError("--runner must be an existing absolute Python entrypoint")
    for argument in runner_args:
        option = argument.split("=", 1)[0]
        if option.startswith("--") and "--config".startswith(option):
            raise ValueError("runner arguments must not override or abbreviate --config")
    runner, config, profile_config = runner.resolve(), config.resolve(), profile_config.absolute()
    sidecar = profile_config.with_name(profile_config.name + ".provenance.json")
    if os.path.lexists(profile_config) or os.path.lexists(sidecar):
        raise FileExistsError("profile config and provenance sidecar must both be fresh paths")
    selected, libraries = profiler_environment(environ, nsys_root)
    source = config.read_bytes()
    generated = inject_environment(source, selected)
    command = [sys.executable, str(runner), "--config", str(profile_config), *runner_args]
    driver = Path(__file__).resolve()
    provenance = {
        "schema": "w1ax_nsight_profile_wrapper_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "driver": {"path": str(driver), "sha256": sha256_bytes(driver.read_bytes())},
        "runner": {"path": str(runner), "sha256": sha256_bytes(runner.read_bytes())},
        "source_config": {"path": str(config), "sha256": sha256_bytes(source)},
        "profile_config": {"path": str(profile_config), "sha256": sha256_bytes(generated)},
        "nsys_root": str(nsys_root.resolve()),
        "validated_injection_libraries": libraries,
        "passed_keys": list(selected),
        "passed_environment": selected,
        "environment_source": "observed live environment of this wrapper inside nsys profile",
        "settings_check": "TOML equals source except the explicitly listed environment additions",
        "command": command,
        "process_group": "inherited from supervisor; runner replaces wrapper via execv",
    }
    profile_config.parent.mkdir(parents=True, exist_ok=True)
    with profile_config.open("xb") as stream:
        stream.write(generated)
    with sidecar.open("x") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--nsys-root", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.runner_args and args.runner_args[0] != "--":
        parser.error("runner arguments must follow --")
    command = prepare_profile(
        args.runner, args.config, args.profile_config, args.nsys_root,
        args.runner_args[1:], os.environ,
    )
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
