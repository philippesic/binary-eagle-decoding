#!/usr/bin/env python3
"""Instrumented real block GGUF load/injection/seven-slot smoke, never timing.

The native test executable executes selected W1 operators and reports each
packed tensor's actual output buffer/backend and activation bits. This checker
binds current binary/model/export hashes and refuses incomplete coverage or CPU
execution when CUDA was requested. CUDA kernel admission itself is separate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from export_block_binary import sha256


def check(
    binary: Path,
    model: Path,
    export: Path,
    *,
    gpu_layers: int,
    require_cuda: bool,
    timeout_seconds=180,
    target: Path | None = None,
    target_sha256: str | None = None,
) -> dict:
    binary, model, export = map(Path, (binary, model, export))
    receipt = json.loads(export.read_text())
    if (
        receipt.get("schema") != "block_binary_export_v1"
        or receipt.get("serialization_audit_passed") is not True
        or receipt.get("output", {}).get("sha256") != sha256(model)
    ):
        raise ValueError("block model/export binding mismatch")
    expected = {name + ".w1a1_packed" for name in receipt["projections"]}
    if len(expected) not in (15, 16) or receipt.get("activation_bits") not in (1, 8):
        raise ValueError("invalid admitted export profile")
    if (
        type(gpu_layers) is not int
        or not 0 <= gpu_layers <= 999
        or require_cuda
        and gpu_layers <= 0
    ):
        raise ValueError("invalid CUDA placement requirement")
    if (target is None) != (target_sha256 is None):
        raise ValueError("paired target requires both path and immutable hash")
    if target is not None:
        target = Path(target)
        if sha256(target) != target_sha256:
            raise ValueError("paired immutable target hash mismatch")
    with tempfile.TemporaryDirectory(prefix="block-native-proof-") as tmp:
        proof = Path(tmp) / "proof.json"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("GGML_W1", "GGML_EAGLE", "LLAMA_EAGLE"))
        }
        command = [str(binary), str(model), str(gpu_layers), str(proof)]
        if target is not None:
            command.append(str(target))
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        if result.returncode or not proof.is_file():
            raise RuntimeError(
                f"native block smoke failed exit {result.returncode}: {result.stderr[-6000:]}"
            )
        observed = json.loads(proof.read_text())
    if (
        observed.get("schema") != "block_native_graph_smoke_v1"
        or observed.get("passed") is not True
    ):
        raise ValueError("invalid native graph proof")
    nodes = observed.get("nodes", [])
    if len(nodes) != len(expected) or {node.get("packed") for node in nodes} != expected:
        raise ValueError("native selected operator coverage mismatch")
    for node in nodes:
        if (
            node.get("activation_bits") != receipt["activation_bits"]
            or node.get("packed_type") != "i32"
        ):
            raise ValueError("native binary precision/weight type mismatch")
        if require_cuda and "CUDA" not in node.get("output_buffer", "").upper():
            raise ValueError("selected binary operation did not execute in CUDA buffer")
    return {
        **observed,
        "schema": "block_binary_native_admission_v1",
        "source_export_sha256": sha256(export),
        "model_sha256": sha256(model),
        "binary_sha256": sha256(binary),
        "profile": receipt["profile"],
        "activation_bits": receipt["activation_bits"],
        "cuda_required": require_cuda,
        "target_sha256": target_sha256,
        "scope": "instrumented_model_operator_smoke_only",
        "throughput_or_quality_claim": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "export", "receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu-layers", type=int, required=True)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--target", type=Path)
    p.add_argument("--target-sha256")
    args = p.parse_args()
    if args.receipt.exists() or args.receipt.resolve() in {
        item.resolve() for item in (args.binary, args.model, args.export)
    }:
        p.error("receipt must be a distinct new file")
    report = check(
        args.binary,
        args.model,
        args.export,
        gpu_layers=args.gpu_layers,
        require_cuda=args.require_cuda,
        target=args.target,
        target_sha256=args.target_sha256,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
