#!/usr/bin/env python3
"""Bounded native target-only TRAIN golden replay on the actual CUDA device.

Five native layer-input taps and exact full-vocabulary final-prefix logits are
compared to saved authenticated producer bytes. This is a data portability gate,
not held-out quality, drafter throughput, or convergence evidence. The sole GPU
operator runs this command inside the existing supervised resource lifecycle.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from w1a1_eagle.block_data import TAPS, BlockDataset, file_sha256, token_sha256  # noqa: E402


def compare_matrix(reference, current, *, atol, rtol, chunk_rows=16):
    if (
        reference.shape != current.shape
        or reference.dtype != np.float32
        or current.dtype != np.float32
        or reference.ndim < 2
        or not np.isfinite(atol)
        or atol <= 0
        or not np.isfinite(rtol)
        or rtol <= 0
    ):
        raise ValueError(
            "numeric portability requires matching F32 arrays and positive finite tolerances"
        )
    max_abs = max_relative = 0.0
    failures = nonfinite = 0
    for first in range(0, len(reference), chunk_rows):
        gold = np.asarray(reference[first : first + chunk_rows])
        actual = np.asarray(current[first : first + chunk_rows])
        finite = np.isfinite(gold) & np.isfinite(actual)
        nonfinite += int(np.count_nonzero(~finite))
        if not finite.all():
            continue
        error = np.abs(actual.astype(np.float64) - gold.astype(np.float64))
        magnitude = np.abs(gold.astype(np.float64))
        max_abs = max(max_abs, float(np.max(error)))
        max_relative = max(max_relative, float(np.max(error / np.maximum(magnitude, 1e-6))))
        failures += int(np.count_nonzero(error > atol + rtol * magnitude))
    return {
        "shape": list(reference.shape),
        "values": int(reference.size),
        "dtype": "float32",
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": max_abs,
        "max_relative_error_with_1e_minus6_floor": max_relative,
        "out_of_tolerance_values": failures,
        "nonfinite_values": nonfinite,
        "status": "PASS" if failures == 0 and nonfinite == 0 else "FAIL",
    }


def cuda_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    rows = [r.strip().split(",") for r in result.stdout.splitlines() if r.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("portability gate requires one unambiguous visible CUDA device")
    name, compute, uuid = (v.strip() for v in rows[0])
    major, minor = compute.split(".")
    return {"name": name, "compute_capability": [int(major), int(minor)], "uuid": uuid}


def select_goldens(dataset, max_tokens, max_cases):
    selected = []
    domains = set()
    for domain in ("prose", "code", "reasoning"):
        for cid in sorted(dataset.chains):
            chain = dataset.chains[cid]
            if chain["domain"] != domain:
                continue
            tokens, _, logits = dataset._arrays[cid]
            if logits is None:
                continue
            eligible = [
                (index, anchor)
                for index, anchor in enumerate(chain["anchors"])
                if 1 <= anchor + 1 <= max_tokens
            ]
            if not eligible:
                continue
            index, anchor = eligible[0]
            # Block load verifies current source stat and consumed finite rows.
            dataset.load_block(cid, index, require_teacher=True)
            selected.append((cid, anchor + 1))
            domains.add(domain)
            break
    if domains != {"prose", "code", "reasoning"}:
        raise ValueError(
            "bounded portability goldens require full-logit original TRAIN in all three domains"
        )
    if max_cases < len(selected):
        raise ValueError("case bound cannot cover all required domains")
    return selected


def check_producer(receipt, dataset, device, *, binary_sha256, source_revision):
    execution = receipt.get("executed_result_buffers", [])
    buffers = receipt.get("target_storage_buffers", {})
    hardware = receipt.get("hardware", [])
    if (
        receipt.get("schema") != "block_native_teacher_request_v1"
        or receipt.get("complete") is not True
        or receipt.get("optimizer_updates") != 0
        or receipt.get("target_sha256") != dataset.manifest["producer"]["target_sha256"]
        or receipt.get("target_precision") != "F16"
        or receipt.get("kv_type") != "F16"
        or receipt.get("producer_binary_sha256") != binary_sha256
        or receipt.get("producer_source_revision") != source_revision
        or receipt.get("teacher_context_reset_between_requests") is not True
        or receipt.get("prefix_contract") != "teacher_forced_exact_caller_token_ids"
        or receipt.get("tap_ids") != list(TAPS)
        or receipt.get("gpu_layers", 0) <= 0
        or not execution
        or not all(
            isinstance(name, str) and re.fullmatch(r"CUDA[0-9]+", name) for name in execution
        )
        or not any(re.fullmatch(r"CUDA[0-9]+", name) for name in buffers)
        or not any(device["name"] in str(name) for name in hardware)
    ):
        raise ValueError(
            "native producer lacks matching actual CUDA target/source/offload/device proof"
        )


def run_gate(args, *, teacher_factory=None, device_query=cuda_inventory):
    production = teacher_factory is None
    if args.output.exists() or args.output_root.exists():
        raise ValueError("refuse to overwrite portability evidence")
    if file_sha256(args.binary) != args.binary_sha256:
        raise ValueError("native binary differs from admitted SHA256")
    dataset = BlockDataset(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        admission_path=args.admission,
        admission_sha256=args.admission_sha256,
    )
    target_pin = dataset.manifest["producer"]["target_sha256"]
    if file_sha256(args.target) != target_pin:
        raise ValueError("portability target differs from golden native target")
    device = device_query()
    if device["compute_capability"] != list(args.expected_compute_capability):
        raise ValueError("actual CUDA compute capability differs from required gate device")
    cases = select_goldens(dataset, args.max_tokens, args.max_cases)
    if teacher_factory is None:
        from capture_block_qat_teacher import NativeTeacher

        teacher_factory = NativeTeacher
    report = {
        "schema": "nine_model_capture_portability_v1",
        "status": "FAIL",
        "artifact_kind": "production" if production else "synthetic_fixture",
        "family": dataset.manifest["family"],
        "compute_capability": device["compute_capability"],
        "device": device,
        "target_sha256": target_pin,
        "optimizer_updates": 0,
        "manifest_sha256": dataset.sha256,
        "native_binary_sha256": args.binary_sha256,
        "native_source_revision": args.producer_source_revision,
        "gate_source_sha256": file_sha256(Path(__file__)),
        "producer_hardware": dataset.manifest["producer"]["hardware"],
        "numeric_checks": [],
        "failure": None,
        "decision_scope": (
            "exact full-vocabulary native target argmax; "
            "drafter trajectory is a separate model gate"
        ),
    }
    if production:
        report["teacher_client_source_sha256"] = file_sha256(
            ROOT / "scripts/capture_block_qat_teacher.py"
        )
    teacher = None
    old_handlers = {}

    def stopped(signum, frame):
        raise InterruptedError(f"portability STOP signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stopped)
        teacher = teacher_factory(
            args.binary,
            args.target,
            args.output_root,
            target_sha256=target_pin,
            max_tokens=args.max_tokens,
            gpu_layers=args.gpu_layers,
            producer_source_revision=args.producer_source_revision,
            timeout_seconds=args.timeout_seconds,
        )
        for cid, length in cases:
            chain = dataset.chains[cid]
            tokens, saved_features, saved_logits = dataset._arrays[cid]
            prefix = [int(t) for t in tokens[:length]]
            ancestry = {
                k: chain[k] for k in ("prompt_id", "prompt_sha256", "domain", "prompt_length")
            }
            ancestry["source_split"] = "TRAIN"
            receipt = teacher.capture_prefix(
                prefix, TAPS, logits_mode="last", chain_ancestry=ancestry
            )
            check_producer(
                receipt,
                dataset,
                device,
                binary_sha256=args.binary_sha256,
                source_revision=args.producer_source_revision,
            )
            if (
                report["artifact_kind"] == "production"
                and receipt.get("client_source_sha256") != report["teacher_client_source_sha256"]
            ):
                raise ValueError("actual teacher client source differs from pinned local module")
            if receipt.get("tokens") != prefix or receipt.get("chain_ancestry") != ancestry:
                raise ValueError("fresh teacher prefix/chain ancestry changed")
            fresh_features = teacher.array(receipt, "features")
            fresh_logits = teacher.array(receipt, "logits")
            feature_check = compare_matrix(
                saved_features[:length],
                fresh_features,
                atol=args.feature_atol,
                rtol=args.feature_rtol,
            )
            logit_check = compare_matrix(
                saved_logits[length - 1 : length],
                fresh_logits,
                atol=args.logit_atol,
                rtol=args.logit_rtol,
            )
            original_decision = int(np.argmax(saved_logits[length - 1]))
            fresh_decision = int(np.argmax(fresh_logits[0]))
            files = receipt["files"]
            for record in files.values():
                if file_sha256(Path(record["path"])) != record["sha256"]:
                    raise ValueError("fresh native producer bytes differ from receipt")
            report["numeric_checks"].append(
                {
                    "chain_id": cid,
                    "domain": chain["domain"],
                    "prompt_sha256": chain["prompt_sha256"],
                    "prefix_sha256": token_sha256(prefix),
                    "prefix_tokens": length,
                    "features": feature_check,
                    "full_vocab_logits": logit_check,
                    "saved_target_argmax": original_decision,
                    "fresh_target_argmax": fresh_decision,
                    "decision_changed": original_decision != fresh_decision,
                    "fresh_artifact_pins": files,
                }
            )
        if any(
            case["decision_changed"]
            or case["features"]["status"] != "PASS"
            or case["full_vocab_logits"]["status"] != "PASS"
            for case in report["numeric_checks"]
        ):
            raise ValueError(
                "native capture numeric/target-decision gate failed; "
                "investigate labels/conditioning"
            )
        report["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if teacher is not None:
            try:
                teacher.close()
            except Exception as error:
                report["status"] = "FAIL"
                report["failure"] = {
                    "type": type(error).__name__,
                    "message": f"producer cleanup failed: {error}",
                }
            report["producer_closed"] = teacher.closed
            if hasattr(teacher, "process"):
                report["producer_pid"] = teacher.process.pid
                report["producer_returncode"] = teacher.process.poll()
                if report["producer_returncode"] is None:
                    report["status"] = "FAIL"
                    report["failure"] = {
                        "type": "ResourceError",
                        "message": "owned native producer remains live after cleanup",
                    }
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "admission", "binary", "target", "output-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "manifest-sha256",
        "admission-sha256",
        "binary-sha256",
        "producer-source-revision",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-compute-capability", type=int, nargs=2, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--feature-atol", type=float, default=0.002)
    parser.add_argument("--feature-rtol", type=float, default=0.002)
    parser.add_argument("--logit-atol", type=float, default=0.02)
    parser.add_argument("--logit-rtol", type=float, default=0.002)
    args = parser.parse_args()
    try:
        report = run_gate(args)
    except Exception as error:
        if args.output.exists():
            raise
        report = {
            "schema": "nine_model_capture_portability_v1",
            "status": "FAIL",
            "artifact_kind": "production",
            "optimizer_updates": 0,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failure": report["failure"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
