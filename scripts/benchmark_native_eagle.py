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
GROUP_VARIANTS = (
    "packed_fusion_w1a1",
    "packed_attention_w1a1",
    "packed_ffn_w1a1",
    "packed_head_w1a1",
    "packed_all_w1a1",
)
GROUP_NAMES = ("fusion", "attention", "ffn", "head", "all")
WEIGHT_ONLY_VARIANTS = ("draft_q4_0", "draft_q8_0")
WEIGHT_ONLY_NAMES = ("q4_0", "q8_0")
W1AX_VARIANTS = ("draft_w1a16", "draft_w1a8", "draft_w1a4", "draft_w1a1")
W1AX_BITS = dict(zip(W1AX_VARIANTS, (16, 8, 4, 1), strict=True))
W1AX_LOADER_MARKER = "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)"
W1AX_A4_CONVENTIONAL_MARKER = "CUDA packed W1A4 CONVENTIONAL dispatch"
W1AX_CUDA_MARKERS = {
    "16": "CUDA packed W1A16 FP16 SIGNADD dispatch",
    "8": "CUDA packed W1A8 INT8 dispatch",
    "4": "CUDA packed W1A4 BITSERIAL dispatch",
    "1": "CUDA packed W1A1 XOR/POPCOUNT dispatch",
}
W1AX_DIAGNOSTIC_DRAFT_LENGTHS = (1, 2, 3, 5)
W1AX_DIAGNOSTIC_CONFIDENCE_FLOORS = (0.0, 0.1, 0.3)
NATIVE_OPERAND_VARIANTS = ("draft_w8a8", "draft_w4a4")
NATIVE_OPERAND_NAMES = ("w8a8", "w4a4")
NATIVE_OPERAND_MMA_VARIANTS = ("draft_w8a8_mma", "draft_w4a4_mma")
NATIVE_OPERAND_MMA_DEFAULTS = dict(
    zip(NATIVE_OPERAND_MMA_VARIANTS, NATIVE_OPERAND_VARIANTS, strict=True)
)
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


def executable_path(name: str) -> str | None:
    """Resolve PATH tools, including the WSL GPU utility outside nonlogin PATH."""
    found = shutil.which(name)
    if found is not None:
        return found
    wsl_nvidia_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if name == "nvidia-smi" and wsl_nvidia_smi.is_file() and os.access(wsl_nvidia_smi, os.X_OK):
        return str(wsl_nvidia_smi)
    return None


def gpu_snapshot() -> dict[str, Any]:
    executable = executable_path("nvidia-smi")
    command = [
        executable or "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.draw",
        "--format=csv,noheader",
    ]
    if executable is None:
        return {"available": False, "command": command, "executable_path": None}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "command": command, "executable_path": executable, "error": str(error)}
    result = {
        "available": True,
        "command": command,
        "executable_path": executable,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        fallback = [
            executable,
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


def command_snapshot(command: list[str], timeout: int = 15) -> dict[str, Any]:
    executable = executable_path(command[0])
    if executable is None:
        return {"available": False, "command": command, "executable_path": None}
    command = [executable, *command[1:]]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "command": command, "executable_path": executable, "error": str(error)}
    return {
        "available": True,
        "command": command,
        "executable_path": executable,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def environment_manifest(binary: Path) -> dict[str, Any]:
    """Collect host/build facts once per run, outside the measured request loop."""
    full_query = command_snapshot(["nvidia-smi", "-q"], timeout=30)
    cuda_query = command_snapshot(["nvcc", "--version"])
    runtime_links = command_snapshot(["ldd", str(binary)])
    cache_files = []
    for parent in (binary.parent, *binary.parents):
        candidate = parent / "CMakeCache.txt"
        if candidate.is_file():
            cache_files.append(candidate)
            break
    cmake_cache: dict[str, str] = {}
    cache_path = cache_files[0] if cache_files else None
    if cache_path:
        wanted = (
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER:FILEPATH",
            "CMAKE_CXX_FLAGS",
            "CMAKE_CUDA_COMPILER:FILEPATH",
            "CMAKE_CUDA_FLAGS",
            "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES",
            "GGML_CUDA",
            "GGML_CUDA_FORCE_CUBLAS",
            "GGML_CUDA_W1A1_MMA",
        )
        for line in cache_path.read_text(errors="replace").splitlines():
            if line.startswith("//") or not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if any(key == prefix or key.startswith(prefix + ":") for prefix in wanted):
                cmake_cache[key] = value
    cxx_path = cmake_cache.get("CMAKE_CXX_COMPILER:FILEPATH")
    cxx_version = (
        command_snapshot([cxx_path, "--version"])
        if cxx_path
        else {"available": False, "reason": "CMake cache has no C++ compiler path"}
    )
    capability = command_snapshot(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    runtime_text = runtime_links.get("stdout") or ""
    runtime_libs = [line.strip() for line in runtime_text.splitlines() if "libcudart" in line]
    compile_commands_path = cache_path.parent / "compile_commands.json" if cache_path else None
    compile_commands: dict[str, Any] = {"available": False}
    build_files = []
    if cache_path:
        for name in ("compile_commands.json", "build.ninja", "Makefile"):
            candidate = cache_path.parent / name
            if candidate.is_file():
                build_files.append(
                    {
                        "path": str(candidate),
                        "sha256": sha256(candidate),
                        "bytes": candidate.stat().st_size,
                    }
                )
    if compile_commands_path and compile_commands_path.is_file():
        try:
            commands = json.loads(compile_commands_path.read_text())
            cuda_commands = [row for row in commands if str(row.get("file", "")).endswith(".cu")]
            cxx_commands = [
                row
                for row in commands
                if str(row.get("file", "")).endswith((".cc", ".cpp", ".cxx"))
            ]

            def compact_command(row: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: row.get(key)
                    for key in ("directory", "file", "command", "arguments")
                    if row.get(key) is not None
                }

            compile_commands = {
                "available": True,
                "path": str(compile_commands_path),
                "sha256": sha256(compile_commands_path),
                "entries": len(commands),
                "cuda_command_samples": [compact_command(row) for row in cuda_commands[:5]],
                "cxx_command_samples": [compact_command(row) for row in cxx_commands[:5]],
            }
        except (OSError, json.JSONDecodeError, TypeError) as error:
            compile_commands = {
                "available": True,
                "path": str(compile_commands_path),
                "error": str(error),
            }
    return {
        "nvidia_smi_path": full_query.get("executable_path"),
        "gpu_capability_query": capability,
        "nvidia_smi_q": full_query,
        "cuda_toolkit_nvcc": cuda_query,
        "cxx_compiler": cxx_version,
        "binary_dynamic_runtime_links": runtime_links,
        "cuda_runtime_libraries": runtime_libs,
        "binary": {
            "path": str(binary),
            "sha256": sha256(binary) if binary.is_file() else None,
            "bytes": binary.stat().st_size if binary.is_file() else None,
        },
        "cmake_cache": {
            "path": str(cache_path) if cache_path else None,
            "available": cache_path is not None,
            "selected_build_flags": cmake_cache,
        },
        "compile_commands": compile_commands,
        "build_files": build_files,
        "runtime_version": {
            "status": "driver compatibility and linked CUDA runtime captured where available",
            "driver_cuda_compatibility": full_query.get("stdout")
            if full_query.get("available")
            else None,
            "toolkit_nvcc": cuda_query.get("stdout") if cuda_query.get("available") else None,
            "linked_runtime_libraries": runtime_libs if runtime_libs else None,
        },
    }


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
    w1ax_matrix = evaluation.get("w1ax_matrix", False)
    policy_diagnostic = evaluation.get("w1ax_policy_diagnostic", False)
    if not isinstance(w1ax_matrix, bool):
        raise ValueError("evaluation.w1ax_matrix must be a boolean")
    if not isinstance(policy_diagnostic, bool):
        raise ValueError("evaluation.w1ax_policy_diagnostic must be a boolean")
    if policy_diagnostic and not w1ax_matrix:
        raise ValueError("evaluation.w1ax_policy_diagnostic requires w1ax_matrix")
    if w1ax_matrix:
        incompatible = (
            "binary_mma", "group_matrix", "weight_only_matrix", "native_operand_matrix",
            "native_operand_variants", "native_operand_mma_variants",
        )
        if any(evaluation.get(key) for key in incompatible):
            raise ValueError("evaluation.w1ax_matrix cannot be combined with other matrices")
        return ("target_only", "ordinary_eagle", "draft_q8_0", "draft_q4_0", *W1AX_VARIANTS)
    enabled = evaluation.get("binary_mma", False)
    group_matrix = evaluation.get("group_matrix", False)
    weight_only_matrix = evaluation.get("weight_only_matrix", False)
    native_operand_matrix = evaluation.get("native_operand_matrix", False)
    native_operand_names = evaluation.get("native_operand_variants")
    native_mma_names = evaluation.get("native_operand_mma_variants", [])
    if not all(
        isinstance(value, bool)
        for value in (enabled, group_matrix, weight_only_matrix, native_operand_matrix)
    ):
        raise ValueError(
            "evaluation.binary_mma, group_matrix, weight_only_matrix, "
            "and native_operand_matrix must be booleans"
        )
    if group_matrix and enabled:
        raise ValueError("evaluation.group_matrix and binary_mma cannot be enabled together")
    if native_operand_names is not None:
        if native_operand_matrix:
            raise ValueError(
                "evaluation.native_operand_matrix and native_operand_variants cannot both be set"
            )
        if (
            not isinstance(native_operand_names, list)
            or any(name not in NATIVE_OPERAND_NAMES for name in native_operand_names)
            or len(native_operand_names) != len(set(native_operand_names))
        ):
            raise ValueError(
                "evaluation.native_operand_variants must list distinct w8a8/w4a4 names"
            )
    if (
        not isinstance(native_mma_names, list)
        or any(name not in NATIVE_OPERAND_NAMES for name in native_mma_names)
        or len(native_mma_names) != len(set(native_mma_names))
    ):
        raise ValueError(
            "evaluation.native_operand_mma_variants must list distinct w8a8/w4a4 names"
        )
    base = (
        (*VARIANTS[:2], *GROUP_VARIANTS)
        if group_matrix
        else ((*VARIANTS, MMA_VARIANT) if enabled else VARIANTS)
    )
    if weight_only_matrix:
        base = (*base, *WEIGHT_ONLY_VARIANTS)
    selected_native = (
        NATIVE_OPERAND_VARIANTS
        if native_operand_matrix
        else tuple(
            variant
            for variant, name in zip(NATIVE_OPERAND_VARIANTS, NATIVE_OPERAND_NAMES, strict=True)
            if native_operand_names is not None and name in native_operand_names
        )
    )
    if any(
        name in native_mma_names and variant not in selected_native
        for variant, name in zip(NATIVE_OPERAND_VARIANTS, NATIVE_OPERAND_NAMES, strict=True)
    ):
        raise ValueError("native operand MMA selection requires its default operand variant")
    selected_mma = tuple(
        variant
        for variant, name in zip(NATIVE_OPERAND_MMA_VARIANTS, NATIVE_OPERAND_NAMES, strict=True)
        if name in native_mma_names
    )
    return (*base, *selected_native, *selected_mma)


def w1ax_policy(evaluation: dict[str, Any], enabled: bool) -> dict[str, Any] | None:
    """Keep the primary comparison frozen while allowing declared development-grid cells."""
    if not enabled:
        return None
    diagnostic = evaluation.get("w1ax_policy_diagnostic", False)
    draft_length = evaluation.get("max_draft_tokens")
    confidence_floor = evaluation.get("min_draft_probability")
    if type(draft_length) is not int or type(confidence_floor) not in (int, float):
        raise ValueError("W1Ax policy requires numeric D and p_min")
    if diagnostic:
        if (
            draft_length not in W1AX_DIAGNOSTIC_DRAFT_LENGTHS
            or confidence_floor not in W1AX_DIAGNOSTIC_CONFIDENCE_FLOORS
        ):
            raise ValueError("W1Ax policy diagnostic must use the predeclared D/p_min grid")
        if evaluation.get("prompt_set") != "qat_development":
            raise ValueError("W1Ax policy diagnostic requires prompt_set = qat_development")
        mode = "policy_diagnostic"
    else:
        if draft_length != 5 or confidence_floor != 0.0 or evaluation["warmup_requests"] < 2:
            raise ValueError("W1Ax primary matrix requires D=5, p_min=0.0 and two warmups")
        mode = "primary_matrix"
    return {
        "mode": mode,
        "prompt_set": evaluation.get("prompt_set", "unspecified"),
        "max_draft_tokens": draft_length,
        "min_draft_probability": confidence_floor,
        "warmup_requests": evaluation["warmup_requests"],
        "repetitions": evaluation["repetitions"],
    }


def packed_specs(config: dict[str, Any], variants: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Return packed-model metadata; group variants require a configurable evidence contract."""
    if GROUP_VARIANTS[0] not in variants:
        return {}
    raw = config.get("packed_variants")
    if not isinstance(raw, dict) or set(raw) != set(GROUP_NAMES):
        raise ValueError("group_matrix requires [packed_variants.<group>] for all five groups")
    specs = {}
    for variant, name in zip(GROUP_VARIANTS, GROUP_NAMES, strict=True):
        spec = raw[name]
        if not isinstance(spec, dict):
            raise ValueError(f"packed_variants.{name} must be a table")
        for key in (
            "draft",
            "weight_coverage",
            "activation_coverage",
            "expected_loader_marker",
            "expected_cuda_marker",
        ):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"packed_variants.{name}.{key} must be a nonempty string")
        marker = spec["expected_cuda_marker"]
        if "CUDA" not in marker or "W1A1" not in marker:
            raise ValueError(
                f"packed_variants.{name}.expected_cuda_marker must name CUDA W1A1 dispatch"
            )
        loader_marker = spec["expected_loader_marker"]
        if "EAGLE3" not in loader_marker or "W1A1" not in loader_marker:
            raise ValueError(
                f"packed_variants.{name}.expected_loader_marker must identify loaded W1A1 groups"
            )
        specs[variant] = {
            "draft": spec["draft"],
            "weight_coverage": spec["weight_coverage"],
            "activation_coverage": spec["activation_coverage"],
            "expected_cuda_marker": marker,
            "expected_loader_marker": loader_marker,
        }
    return specs


def weight_only_specs(
    config: dict[str, Any], variants: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    selected = tuple(variant for variant in WEIGHT_ONLY_VARIANTS if variant in variants)
    if not selected:
        return {}
    raw = config.get("weight_only_variants")
    if not isinstance(raw, dict) or set(raw) != set(WEIGHT_ONLY_NAMES):
        raise ValueError("weight_only_matrix requires [weight_only_variants.q4_0] and .q8_0")
    specs = {}
    for variant, name in zip(WEIGHT_ONLY_VARIANTS, WEIGHT_ONLY_NAMES, strict=True):
        spec = raw[name]
        if not isinstance(spec, dict):
            raise ValueError(f"weight_only_variants.{name} must be a table")
        for key in ("draft", "weight_format", "activation_precision", "backend_precision"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"weight_only_variants.{name}.{key} must be a nonempty string")
        if spec["weight_format"].upper() != name.upper():
            raise ValueError(f"weight_only_variants.{name}.weight_format must be {name.upper()}")
        specs[variant] = {
            "draft": spec["draft"],
            "weight_format": spec["weight_format"],
            "weight_coverage": "all draft weights, weight-only quantization",
            "activation_precision": spec["activation_precision"],
            "backend_precision": spec["backend_precision"],
        }
    return specs


def w1ax_specs(config: dict[str, Any], variants: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if not any(variant in variants for variant in W1AX_VARIANTS):
        return {}
    raw = config.get("w1ax")
    if not isinstance(raw, dict) or set(raw) != {"draft", "expected_loader_marker", "expected_cuda_markers"}:
        raise ValueError("W1Ax matrix requires [w1ax] draft, loader marker, and CUDA markers")
    if not isinstance(raw["draft"], str) or not raw["draft"].strip():
        raise ValueError("w1ax.draft must be a nonempty string")
    if raw["expected_loader_marker"] != W1AX_LOADER_MARKER:
        raise ValueError("w1ax.expected_loader_marker must prove all nine W1A1 projections")
    markers = raw["expected_cuda_markers"]
    if markers != W1AX_CUDA_MARKERS:
        raise ValueError("w1ax.expected_cuda_markers must match the four runtime dispatch markers")
    return {
        variant: {
            "draft": raw["draft"],
            "activation_bits": bits,
            "weight_format": "W1 packed i32 plus F32 row scale",
            "weight_coverage": "all nine draft linears",
            "activation_coverage": "all nine draft linears",
            "activation_precision": f"A{bits}",
            "expected_loader_marker": raw["expected_loader_marker"],
            "expected_graph_marker": f"EAGLE3 W1Ax activation bits: {bits}",
            "expected_cuda_marker": markers[str(bits)],
        }
        for variant, bits in W1AX_BITS.items()
    }


def native_operand_specs(
    config: dict[str, Any], variants: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Require model and exact runtime evidence contracts for true W8A8/W4A4 rows."""
    selected = tuple(
        (variant, mma_variant, name)
        for variant, mma_variant, name in zip(
            NATIVE_OPERAND_VARIANTS,
            NATIVE_OPERAND_MMA_VARIANTS,
            NATIVE_OPERAND_NAMES,
            strict=True,
        )
        if variant in variants
    )
    if not selected:
        return {}
    raw = config.get("native_operand_variants")
    if (
        not isinstance(raw, dict)
        or not {name for _, _, name in selected} <= set(raw)
        or not set(raw) <= set(NATIVE_OPERAND_NAMES)
    ):
        raise ValueError(
            "selected native operand variants require their [native_operand_variants.<name>] tables"
        )
    specs = {}
    for variant, mma_variant, name in selected:
        spec = raw[name]
        if not isinstance(spec, dict):
            raise ValueError(f"native_operand_variants.{name} must be a table")
        required = (
            "draft",
            "weight_format",
            "activation_precision",
            "operator",
            "weight_coverage",
            "activation_coverage",
            "expected_loader_marker",
            "expected_cuda_marker",
        )
        for key in required:
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"native_operand_variants.{name}.{key} must be a nonempty string")
        if spec["weight_format"].upper() != name.upper():
            raise ValueError(f"native_operand_variants.{name}.weight_format must be {name.upper()}")
        if name.upper() not in spec["activation_precision"].upper():
            raise ValueError(
                f"native_operand_variants.{name}.activation_precision must identify {name.upper()}"
            )
        if spec["operator"] not in spec["expected_cuda_marker"]:
            raise ValueError(
                f"native_operand_variants.{name}.expected_cuda_marker must name its operator"
            )
        if (
            name.upper() not in spec["expected_cuda_marker"].upper()
            or "CUDA" not in spec["expected_cuda_marker"].upper()
        ):
            raise ValueError(
                f"native_operand_variants.{name}.expected_cuda_marker "
                f"must name CUDA {name.upper()} dispatch"
            )
        if (
            name.upper() not in spec["expected_loader_marker"].upper()
            or "EAGLE3" not in spec["expected_loader_marker"].upper()
        ):
            raise ValueError(
                f"native_operand_variants.{name}.expected_loader_marker "
                f"must identify EAGLE3 {name.upper()} load"
            )
        specs[variant] = {
            **{key: spec[key] for key in required},
            "backend_precision": spec["operator"],
        }
        if mma_variant in variants:
            for key in ("mma_operator", "expected_mma_cuda_marker"):
                if not isinstance(spec.get(key), str) or not spec[key].strip():
                    raise ValueError(
                        f"native_operand_variants.{name}.{key} must be a nonempty string"
                    )
            mma_marker = spec["expected_mma_cuda_marker"]
            default_marker = spec["expected_cuda_marker"]
            if (
                "MMA" not in mma_marker.upper()
                or spec["mma_operator"] not in mma_marker
                or mma_marker in default_marker
                or default_marker in mma_marker
            ):
                raise ValueError(
                    f"native_operand_variants.{name}.expected_mma_cuda_marker "
                    "must distinctly identify the MMA operator"
                )
            specs[variant]["forbidden_cuda_marker"] = mma_marker
            specs[mma_variant] = {
                **specs[variant],
                "operator": spec["mma_operator"],
                "backend_precision": spec["mma_operator"],
                "expected_cuda_marker": mma_marker,
                "forbidden_cuda_marker": default_marker,
                "comparison_default_variant": variant,
            }
    return specs


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
    if len(base) == len(VARIANTS):
        orders = []
        for rep in range(repetitions):
            offset = rep % len(base)
            rotated = base[offset:] + base[:offset]
            orders.append(rotated if rep % 2 == 0 else list(reversed(rotated)))
        return orders
    orders = []
    for rep in range(repetitions):
        offset = rep % len(base)
        orders.append(base[offset:] + base[:offset])
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


def dispatch_evidence(
    server_log: str,
    variant: str,
    expected_marker: str | None = None,
    expected_loader_marker: str | None = None,
) -> dict[str, Any]:
    """Distinguish a packed model load from proof of actual CUDA op execution."""
    packed_loaded = "EAGLE3 using packed W1A1 draft head" in server_log
    expected_loader_seen = (
        expected_loader_marker in server_log
        if expected_loader_marker is not None
        else packed_loaded
    )
    cuda_backend = bool(
        re.search(r"CUDA\d+.*(?:model buffer|compute buffer|KV buffer)|ggml_cuda_init", server_log)
    )
    legacy_portable_marker = "CUDA packed W1A1 XOR/POPCOUNT dispatch"
    portable_marker = "CUDA packed W1A1 portable XOR/POPCOUNT dispatch"
    mma_marker = "CUDA packed W1A1 binary MMA dispatch"
    portable_seen = portable_marker in server_log or legacy_portable_marker in server_log
    mma_seen = mma_marker in server_log
    expected_seen = (
        expected_marker in server_log
        if expected_marker is not None
        else mma_seen
        if variant == MMA_VARIANT
        else portable_seen
    )
    unexpected_seen = portable_seen if variant == MMA_VARIANT else mma_seen
    if expected_marker is not None:
        unexpected_seen = unexpected_seen or not expected_seen
    packed_variant = variant in (*GROUP_VARIANTS, "packed_head_w1a1", MMA_VARIANT)
    confirmed = (
        packed_variant
        and expected_seen
        and (expected_loader_marker is None or expected_loader_seen)
        and not unexpected_seen
    )
    return {
        "packed_head_loader_log": packed_loaded,
        "configured_expected_loader_marker": expected_loader_marker,
        "configured_expected_loader_marker_seen": expected_loader_seen,
        "cuda_backend_log": cuda_backend,
        "explicit_cuda_w1a1_op_log": portable_seen or mma_seen,
        "portable_cuda_dispatch_log": portable_seen,
        "binary_mma_cuda_dispatch_log": mma_seen,
        "configured_expected_cuda_dispatch_marker": expected_marker,
        "configured_expected_cuda_dispatch_marker_seen": expected_seen,
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


def native_operand_dispatch_evidence(server_log: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Confirm the configured draft load and the actual CUDA operand operator."""
    loader_seen = spec["expected_loader_marker"] in server_log
    lines = server_log.splitlines()
    operator_seen = any(spec["expected_cuda_marker"] in line for line in lines)
    forbidden_marker = spec.get("forbidden_cuda_marker")
    forbidden_seen = bool(forbidden_marker and any(forbidden_marker in line for line in lines))
    return {
        "expected_loader_marker": spec["expected_loader_marker"],
        "expected_loader_marker_seen": loader_seen,
        "expected_cuda_marker": spec["expected_cuda_marker"],
        "expected_cuda_marker_seen": operator_seen,
        "forbidden_cuda_marker": forbidden_marker,
        "forbidden_cuda_marker_seen": forbidden_seen,
        "operator": spec["operator"],
        "cuda_native_operand_dispatch_confirmed": (
            loader_seen and operator_seen and not forbidden_seen
        ),
    }


def w1ax_dispatch_evidence(
    server_log: str, spec: dict[str, Any], all_specs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Require all-nine loading, selected graph precision, and exclusive CUDA op dispatch."""
    loader_seen = spec["expected_loader_marker"] in server_log
    def graph_marker_seen(marker: str) -> bool:
        return re.search(re.escape(marker) + r"(?!\d)", server_log) is not None

    graph_seen = graph_marker_seen(spec["expected_graph_marker"])
    dispatch_seen = spec["expected_cuda_marker"] in server_log
    other_graph_markers = [
        other["expected_graph_marker"] for other in all_specs.values() if other is not spec
    ]
    other_dispatch_markers = [
        other["expected_cuda_marker"] for other in all_specs.values() if other is not spec
    ]
    wrong_mode_seen = any(graph_marker_seen(marker) for marker in other_graph_markers) or any(
        marker in server_log for marker in (*other_dispatch_markers, W1AX_A4_CONVENTIONAL_MARKER)
    )
    return {
        "expected_loader_marker": spec["expected_loader_marker"],
        "expected_loader_marker_seen": loader_seen,
        "expected_graph_marker": spec["expected_graph_marker"],
        "expected_graph_marker_seen": graph_seen,
        "expected_cuda_marker": spec["expected_cuda_marker"],
        "expected_cuda_marker_seen": dispatch_seen,
        "wrong_mode_marker_seen": wrong_mode_seen,
        "cuda_w1ax_dispatch_confirmed": loader_seen and graph_seen and dispatch_seen and not wrong_mode_seen,
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
    if variant not in (
        *VARIANTS,
        MMA_VARIANT,
        *GROUP_VARIANTS,
        *WEIGHT_ONLY_VARIANTS,
        *W1AX_VARIANTS,
        *NATIVE_OPERAND_VARIANTS,
        *NATIVE_OPERAND_MMA_VARIANTS,
    ):
        raise ValueError(f"unknown benchmark variant: {variant}")
    command = [str(paths["binary"]), "-m", str(paths["target"])]
    if variant == "target_only":
        command += ["--spec-type", "none"]
    else:
        if variant == "ordinary_eagle":
            draft_key = "ordinary_draft"
        elif variant in GROUP_VARIANTS and variant != "packed_head_w1a1":
            draft_key = variant
        elif variant in GROUP_VARIANTS and config["evaluation"].get("group_matrix", False):
            draft_key = variant
        elif variant in (
            *WEIGHT_ONLY_VARIANTS,
            *W1AX_VARIANTS,
            *NATIVE_OPERAND_VARIANTS,
            *NATIVE_OPERAND_MMA_VARIANTS,
        ):
            draft_key = variant
        else:
            draft_key = "packed_head_draft"
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
        if config["evaluation"].get("w1ax_matrix", False):
            command += [
                "--spec-draft-p-min", str(config["evaluation"]["min_draft_probability"])
            ]
    command += list(config["server"]["common_args"])
    command += ["--host", config["server"]["host"], "--port", str(config["server"]["port"])]
    return command


def request_body(config: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    body = {
        "messages": prompt["messages"],
        "max_tokens": evaluation["max_output_tokens"],
        "temperature": evaluation["temperature"],
        "seed": evaluation["seed"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": evaluation["enable_thinking"]},
        "reasoning_format": "none",
        # Kept identical for every variant so raw generation IDs can be paired.
        "return_tokens": True,
        "verbose": True,
    }
    return body


def generated_token_ids(response: dict[str, Any]) -> list[int] | None:
    """Extract server-returned generated IDs across supported response layouts."""
    candidates: list[Any] = [response.get("generated_token_ids"), response.get("token_ids")]
    verbose = response.get("__verbose")
    if isinstance(verbose, dict):
        candidates.append(verbose.get("tokens"))
    choices = response.get("choices") or []
    if choices:
        first = choices[0]
        message = first.get("message")
        if not isinstance(message, dict):
            message = {}
        logprobs = first.get("logprobs")
        candidates.extend(
            [
                first.get("generated_token_ids"),
                first.get("token_ids"),
                first.get("tokens"),
                message.get("tokens"),
                logprobs.get("content") if isinstance(logprobs, dict) else None,
            ]
        )
    candidates.append(response.get("tokens"))
    for value in candidates:
        if not isinstance(value, list):
            continue
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return value
        object_ids = []
        for item in value:
            if not isinstance(item, dict):
                object_ids = []
                break
            token_id = item.get("id", item.get("token_id"))
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                object_ids = []
                break
            object_ids.append(token_id)
        if object_ids:
            return object_ids
    return None


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
    token_ids = generated_token_ids(response)
    return {
        "server_response_id": response.get("id"),
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
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest()
        if token_ids is not None
        else None,
        "generated_token_ids_status": "available" if token_ids is not None else "omitted_by_server",
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
    records: list[dict[str, Any]],
    variants: tuple[str, ...] = VARIANTS,
    reference_variant: str = "target_only",
) -> dict[str, Any]:
    """Compare decoded response text hashes for matched repetition/prompt pairs."""
    if reference_variant not in variants:
        raise ValueError("reference variant is unavailable")
    by_key = {(row["repetition"], row["prompt_id"], row["variant"]): row for row in records}
    result = {}
    for variant in variants:
        if variant == reference_variant:
            continue
        matched = 0
        unavailable = 0
        mismatches = []
        for row in records:
            if row["variant"] != variant:
                continue
            key = (row["repetition"], row["prompt_id"], reference_variant)
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


def generated_token_id_matches(
    records: list[dict[str, Any]],
    variants: tuple[str, ...] = VARIANTS,
    reference_variant: str = "target_only",
) -> dict[str, Any]:
    """Compare raw generated IDs for paired requests; retain first-difference evidence."""
    if reference_variant not in variants:
        raise ValueError("reference variant is unavailable")
    by_key = {(row["repetition"], row["prompt_id"], row["variant"]): row for row in records}
    result = {}
    for variant in variants:
        if variant == reference_variant:
            continue
        matched = 0
        unavailable = 0
        mismatches = []
        for row in records:
            if row["variant"] != variant:
                continue
            reference = by_key.get((row["repetition"], row["prompt_id"], reference_variant))
            reference_ids = reference.get("generated_token_ids") if reference else None
            candidate_ids = row.get("generated_token_ids")
            if not isinstance(reference_ids, list) or not isinstance(candidate_ids, list):
                unavailable += 1
                continue
            if reference_ids == candidate_ids:
                matched += 1
                continue
            first_difference = next(
                (
                    index
                    for index, (expected, actual) in enumerate(zip(reference_ids, candidate_ids))
                    if expected != actual
                ),
                min(len(reference_ids), len(candidate_ids)),
            )
            mismatches.append(
                {
                    "repetition": row["repetition"],
                    "prompt_id": row["prompt_id"],
                    "first_mismatch_index": first_difference,
                    "reference_id": reference_ids[first_difference]
                    if first_difference < len(reference_ids)
                    else None,
                    "candidate_id": candidate_ids[first_difference]
                    if first_difference < len(candidate_ids)
                    else None,
                    "reference_length": len(reference_ids),
                    "candidate_length": len(candidate_ids),
                }
            )
        result[variant] = {
            "matched_sequences": matched,
            "mismatched_sequences": len(mismatches),
            "unavailable": unavailable,
            "mismatch_pairs": mismatches,
        }
    return result


def relative_speedups(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Compute legacy head ratios plus every packed/anchor pooled-rate ratio."""
    result = {anchor: {} for anchor in ("target_only", "ordinary_eagle")}
    anchors = ("target_only", "ordinary_eagle") + (
        ("draft_q4_0",) if "draft_q4_0" in aggregated else ()
    )
    by_variant = {}
    for variant, packed in aggregated.items():
        if not (variant.startswith("packed_") or variant.startswith(("draft_q", "draft_w"))):
            continue
        by_variant[variant] = {}
        for anchor in anchors:
            baseline = aggregated[anchor]
            by_variant[variant][anchor] = {}
            for metric in ("request_tokens_per_s", "decode_tokens_per_s"):
                numerator = packed[metric]
                denominator = baseline[metric]
                value = (
                    numerator / denominator
                    if isinstance(numerator, (int, float))
                    and isinstance(denominator, (int, float))
                    and denominator > 0
                    else None
                )
                by_variant[variant][anchor][metric] = value
                if variant == "packed_head_w1a1":
                    result[anchor][metric] = value
    result["by_variant"] = by_variant
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


def native_mma_speedup_vs_default(
    aggregated: dict[str, Any], variants: tuple[str, ...]
) -> dict[str, dict[str, float | None]]:
    result = {}
    for mma_variant, default_variant in NATIVE_OPERAND_MMA_DEFAULTS.items():
        if mma_variant not in variants:
            continue
        result[mma_variant] = {}
        for metric in ("request_tokens_per_s", "decode_tokens_per_s"):
            numerator = aggregated[mma_variant][metric]
            denominator = aggregated[default_variant][metric]
            result[mma_variant][metric] = (
                numerator / denominator
                if isinstance(numerator, (int, float))
                and isinstance(denominator, (int, float))
                and denominator > 0
                else None
            )
    return result


def require_complete_pairs(
    records: list[dict[str, Any]],
    variants: tuple[str, ...],
    repetitions: int,
    prompts: list[dict[str, Any]],
) -> None:
    expected = {
        (rep, prompt["id"], variant)
        for rep in range(repetitions)
        for prompt in prompts
        for variant in variants
    }
    observed = [(row["repetition"], row["prompt_id"], row["variant"]) for row in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("benchmark records are not complete prompt/repetition pairs")


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


def all_native_operand_dispatches_confirmed(
    destination: Path, repetitions: int, variant: str
) -> bool:
    return all(
        json.loads(
            (destination / f"rep-{rep:02d}" / variant / "dispatch-evidence.json").read_text()
        )["cuda_native_operand_dispatch_confirmed"]
        for rep in range(repetitions)
    )


def all_w1ax_dispatches_confirmed(destination: Path, repetitions: int, variant: str) -> bool:
    return all(
        json.loads(
            (destination / f"rep-{rep:02d}" / variant / "dispatch-evidence.json").read_text()
        )["cuda_w1ax_dispatch_confirmed"]
        for rep in range(repetitions)
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
    group_specs = packed_specs(config, variants)
    weight_specs = weight_only_specs(config, variants)
    ax_specs = w1ax_specs(config, variants)
    native_specs = native_operand_specs(config, variants)
    if evaluation["warmup_requests"] < 0 or evaluation["max_output_tokens"] <= 0:
        raise ValueError("invalid warmup or max output setting")
    policy = w1ax_policy(evaluation, bool(ax_specs))
    if not isinstance(evaluation.get("round_trace", False), bool):
        raise ValueError("evaluation.round_trace must be a boolean")
    if config["server"]["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("server must bind loopback for this local harness")
    model_paths = dict(config["models"])
    if ax_specs:
        model_paths.pop("packed_head_draft", None)
    model_paths.update({variant: spec["draft"] for variant, spec in group_specs.items()})
    model_paths.update({variant: spec["draft"] for variant, spec in weight_specs.items()})
    model_paths.update({variant: spec["draft"] for variant, spec in ax_specs.items()})
    model_paths.update({variant: spec["draft"] for variant, spec in native_specs.items()})
    paths = {
        name: resolve(ROOT, value)
        for name, value in {
            "binary": config["server"]["binary"],
            **model_paths,
            "prompt_file": evaluation["prompt_file"],
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    prompts = load_prompts(paths["prompt_file"])
    if policy and policy["mode"] == "policy_diagnostic" and len(prompts) != 24:
        raise ValueError("W1Ax policy diagnostic requires 24 QAT-development prompts")
    if not available_port(config["server"]["host"], config["server"]["port"]):
        raise RuntimeError("configured server port is already occupied")
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(config_path, destination / "config.toml")
        shutil.copy2(paths["prompt_file"], destination / "prompts.jsonl")
        env = {key: os.environ[key] for key in SAFE_INHERITED_ENV if key in os.environ}
        env.update(config.get("environment", {}))
        for selector in ("GGML_CUDA_W1A1_MMA", "GGML_CUDA_W8A8_MMA", "GGML_CUDA_W4A4_MMA", "GGML_W1AX_ACT_BITS", "GGML_W1AX_A4_KERNEL", "W1AX_ROUND_TRACE_JSONL"):
            env.pop(selector, None)
        variant_environments = {
            variant: {
                **env,
                "GGML_CUDA_W1A1_MMA": "1" if variant == MMA_VARIANT else "0",
                "GGML_CUDA_W8A8_MMA": "1" if variant == "draft_w8a8_mma" else "0",
                "GGML_CUDA_W4A4_MMA": "1" if variant == "draft_w4a4_mma" else "0",
                **({"GGML_W1AX_ACT_BITS": str(W1AX_BITS[variant])} if variant in ax_specs else {}),
                **({"GGML_W1AX_A4_KERNEL": "bitserial"} if variant == "draft_w1a4" else {}),
            }
            for variant in variants
        }
        (destination / "project-diff.patch").write_text(git_output("diff", "--binary", "HEAD"))
        (destination / "llama-diff.patch").write_text(
            git_output("-C", "third_party/llama.cpp", "diff", "--binary", "HEAD")
        )
        llama_gitlink = git_output("ls-tree", "HEAD", "third_party/llama.cpp").split()[2]
        llama_checkout_commit = git_output(
            "-C", "third_party/llama.cpp", "rev-parse", "HEAD"
        ).strip()
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
            "round_trace_enabled": evaluation.get("round_trace", False),
            "round_trace_path_template": "rep-{repetition:02d}/{variant}/round-trace.jsonl",
            "round_trace_schema": "w1ax_eagle_round_v1",
            "round_trace_request_mapping": (
                "Concurrency is one; server_request_index counts warmups then measured calls "
                "in order within each server process. Trace task_id is server-internal. "
                "Exclude checkpoint_replay rows from accepted-per-round quality aggregates."
            ),
            "variants": list(variants),
            "policy": policy,
            "variant_specs": {
                **{
                    variant: {
                        "weight_coverage": group_specs[variant]["weight_coverage"],
                        "activation_coverage": group_specs[variant]["activation_coverage"],
                        "weight_format": "W1A1 packed",
                        "activation_precision": "runtime sign with f32 scale",
                        "backend_precision": (
                            "must be established by per-variant CUDA dispatch evidence"
                        ),
                        "expected_cuda_marker": group_specs[variant]["expected_cuda_marker"],
                        "expected_loader_marker": group_specs[variant]["expected_loader_marker"],
                        "draft_model_path": str(paths[variant]),
                        "draft_model_sha256": sha256(paths[variant]),
                    }
                    for variant in group_specs
                },
                **{
                    variant: {
                        **weight_specs[variant],
                        "draft_model_path": str(paths[variant]),
                        "draft_model_sha256": sha256(paths[variant]),
                    }
                    for variant in weight_specs
                },
                **{
                    variant: {
                        **native_specs[variant],
                        "draft_model_path": str(paths[variant]),
                        "draft_model_sha256": sha256(paths[variant]),
                    }
                    for variant in native_specs
                },
                **{
                    variant: {
                        **ax_specs[variant],
                        "draft_model_path": str(paths[variant]),
                        "draft_model_sha256": sha256(paths[variant]),
                    }
                    for variant in ax_specs
                },
            },
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
            "environment_manifest": environment_manifest(paths["binary"]),
            "project_commit": git_output("rev-parse", "HEAD").strip(),
            "project_status": git_output("status", "--porcelain"),
            "project_diff_sha256": sha256(destination / "project-diff.patch"),
            "llama_gitlink": llama_gitlink,
            "llama_checkout_commit": llama_checkout_commit,
            "llama_checkout_matches_gitlink": llama_checkout_commit == llama_gitlink,
            "llama_checkout_status": git_output(
                "-C", "third_party/llama.cpp", "status", "--porcelain"
            ),
            "llama_checkout_diff_sha256": sha256(destination / "llama-diff.patch"),
        }
        json_write(destination / "manifest.json", manifest)
        if dry_run:
            return destination
        draft_hashes = {}
        for variant in variants:
            if variant == "target_only":
                draft_hashes[variant] = None
            elif variant == "ordinary_eagle":
                draft_hashes[variant] = manifest["files"]["ordinary_draft"]["sha256"]
            elif variant in group_specs or variant in weight_specs or variant in native_specs or variant in ax_specs:
                draft_hashes[variant] = manifest["files"][variant]["sha256"]
            else:
                draft_hashes[variant] = manifest["files"]["packed_head_draft"]["sha256"]
        base_url = f"http://{config['server']['host']}:{config['server']['port']}"
        records = []
        timing_rows = []
        try:
            for repetition, order in enumerate(orders):
                for variant in order:
                    server_dir = destination / f"rep-{repetition:02d}" / variant
                    server_dir.mkdir(parents=True)
                    pending_native_records = []
                    json_write(server_dir / "gpu-before.json", gpu_snapshot())
                    command = manifest["commands"][variant]
                    variant_env = variant_environments[variant]
                    if evaluation.get("round_trace", False):
                        variant_env = {
                            **variant_env,
                            "W1AX_ROUND_TRACE_JSONL": str(server_dir / "round-trace.jsonl"),
                        }
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
                            for prompt_index, prompt in enumerate(prompts):
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
                                        "request_id": f"rep-{repetition:02d}/{variant}/{prompt['id']}",
                                        "policy_mode": policy["mode"] if policy else None,
                                        "max_draft_tokens": policy["max_draft_tokens"] if policy else None,
                                        "min_draft_probability": policy["min_draft_probability"] if policy else None,
                                        "server_request_index": evaluation["warmup_requests"] + prompt_index,
                                        "w1ax_activation_bits_selector": variant_env.get("GGML_W1AX_ACT_BITS"),
                                        "round_trace_path": variant_env.get("W1AX_ROUND_TRACE_JSONL"),
                                        "w1a1_mma_selector": variant_env["GGML_CUDA_W1A1_MMA"],
                                        "w8a8_mma_selector": variant_env["GGML_CUDA_W8A8_MMA"],
                                        "w4a4_mma_selector": variant_env["GGML_CUDA_W4A4_MMA"],
                                        "weight_coverage": group_specs.get(variant, {}).get(
                                            "weight_coverage"
                                        )
                                        or native_specs.get(variant, {}).get("weight_coverage")
                                        or ax_specs.get(variant, {}).get("weight_coverage"),
                                        "activation_coverage": group_specs.get(variant, {}).get(
                                            "activation_coverage"
                                        )
                                        or native_specs.get(variant, {}).get("activation_coverage")
                                        or ax_specs.get(variant, {}).get("activation_coverage"),
                                        "weight_format": group_specs.get(variant, {}).get(
                                            "weight_format"
                                        )
                                        or weight_specs.get(variant, {}).get("weight_format")
                                        or native_specs.get(variant, {}).get("weight_format")
                                        or ax_specs.get(variant, {}).get("weight_format"),
                                        "operator": native_specs.get(variant, {}).get("operator"),
                                        "activation_precision": group_specs.get(variant, {}).get(
                                            "activation_precision"
                                        )
                                        or weight_specs.get(variant, {}).get("activation_precision")
                                        or native_specs.get(variant, {}).get(
                                            "activation_precision"
                                        ) or ax_specs.get(variant, {}).get("activation_precision"),
                                        "backend_precision": group_specs.get(variant, {}).get(
                                            "backend_precision"
                                        )
                                        or weight_specs.get(variant, {}).get("backend_precision")
                                        or native_specs.get(variant, {}).get("backend_precision")
                                        or ax_specs.get(variant, {}).get("expected_cuda_marker"),
                                        "draft_model_sha256": draft_hashes[variant],
                                    }
                                )
                                json_write(server_dir / "measured" / prompt["id"] / "measurement.json", measurement)
                                if ax_specs and measurement["generated_token_ids"] is None:
                                    raise RuntimeError(
                                        f"{variant}/{prompt['id']}: server omitted raw generated token IDs"
                                    )
                                if variant in native_specs or variant in ax_specs:
                                    pending_native_records.append(measurement)
                                else:
                                    records.append(measurement)
                                    json_write(destination / "records.json", records)
                        finally:
                            stop_server(process)
                            log.flush()
                            evidence = dispatch_evidence(
                                (server_dir / "server.log").read_text(errors="replace"),
                                variant,
                                group_specs.get(variant, {}).get("expected_cuda_marker"),
                                group_specs.get(variant, {}).get("expected_loader_marker"),
                            )
                            if variant in native_specs:
                                evidence = native_operand_dispatch_evidence(
                                    (server_dir / "server.log").read_text(errors="replace"),
                                    native_specs[variant],
                                )
                            if variant in ax_specs:
                                evidence = w1ax_dispatch_evidence(
                                    (server_dir / "server.log").read_text(errors="replace"),
                                    ax_specs[variant], ax_specs,
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
                    if (
                        variant in native_specs
                        and not evidence["cuda_native_operand_dispatch_confirmed"]
                    ):
                        raise RuntimeError(
                            f"{variant}: expected draft load and CUDA operand dispatch "
                            "markers were not both observed"
                        )
                    if variant in ax_specs and not evidence["cuda_w1ax_dispatch_confirmed"]:
                        raise RuntimeError(
                            f"{variant}: all-nine W1Ax load, selected graph and CUDA dispatch "
                            "markers were not confirmed"
                        )
                    if pending_native_records:
                        records.extend(pending_native_records)
                        json_write(destination / "records.json", records)
            require_complete_pairs(records, variants, evaluation["repetitions"], prompts)
            aggregated = aggregate(records, variants)
            report = {
                "status": "complete",
                "variants": list(variants),
                "policy": policy,
                "aggregation": aggregated,
                "packed_speedup_vs": relative_speedups(aggregated),
                "repetitions": repetition_summaries(records, variants),
                "greedy_text_match_vs_target_only": completion_text_matches(records, variants),
                "generated_token_id_match_vs_target_only": generated_token_id_matches(
                    records, variants
                ),
                "draft_timing": summarize_draft_timings(timing_rows, aggregated, variants),
                "records": len(records),
                "native_cuda_dispatch_confirmed": all_dispatches_confirmed(
                    destination, evaluation["repetitions"], "packed_head_w1a1"
                )
                if "packed_head_w1a1" in variants
                else None,
                "cuda_dispatch_confirmed_by_variant": {
                    variant: all_dispatches_confirmed(
                        destination, evaluation["repetitions"], variant
                    )
                    for variant in variants
                    if variant.startswith("packed_")
                },
                "native_operand_dispatch_confirmed_by_variant": {
                    variant: all_native_operand_dispatches_confirmed(
                        destination, evaluation["repetitions"], variant
                    )
                    for variant in native_specs
                },
                "w1ax_dispatch_confirmed_by_variant": {
                    variant: all_w1ax_dispatches_confirmed(
                        destination, evaluation["repetitions"], variant
                    )
                    for variant in ax_specs
                },
                "w1ax_variant_specs": {
                    variant: manifest["variant_specs"][variant] for variant in ax_specs
                },
                "round_trace_enabled": evaluation.get("round_trace", False),
                "native_operand_variant_specs": {
                    variant: manifest["variant_specs"][variant] for variant in native_specs
                },
                "native_operand_mma_speedup_vs_default": native_mma_speedup_vs_default(
                    aggregated, variants
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
                report["greedy_text_match_mma_vs_portable"] = completion_text_matches(
                    records, variants, "packed_head_w1a1"
                )[MMA_VARIANT]
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
