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

from w1a1_eagle.block_data import (  # noqa: E402
    TAPS,
    BlockDataset,
    crop_decode_history,
    file_sha256,
    token_sha256,
    validate_decode_history,
)

EAGLE_TAPS = (2, 18, 33)


class NativeCaptureGoldens:
    """New bounded TRAIN target-only golden prefixes, separate from training data.

    EAGLE speculative/tree captures are never relabeled as these autoregressive
    prefixes. Native receipts and the original TRAIN inventory are externally
    pinned. The original corpus audits remain valid independently.
    """

    def __init__(
        self,
        manifest_path,
        *,
        expected_sha256,
        admission_path=None,
        admission_sha256=None,
        max_capture_bytes=256 * 1024 * 1024,
    ):
        if admission_path is not None or admission_sha256 is not None:
            raise ValueError("bounded golden receipts do not accept block-corpus admission aliases")
        self.path = Path(manifest_path).resolve()
        if file_sha256(self.path) != expected_sha256:
            raise ValueError("golden manifest differs from external SHA256 pin")
        self.sha256 = expected_sha256
        manifest = json.loads(self.path.read_text())
        keys = {
            "schema",
            "family",
            "tap_ids",
            "vocab_size",
            "target_width",
            "target_sha256",
            "train_inventory",
            "cases",
        }
        if (
            set(manifest) != keys
            or manifest["schema"] != "nine_model_train_capture_goldens_v1"
            or manifest["family"] not in ("eagle", "dspark", "dflash")
        ):
            raise ValueError("unsupported bounded native TRAIN golden profile")
        self.tap_ids = EAGLE_TAPS if manifest["family"] == "eagle" else TAPS
        if manifest["tap_ids"] != list(self.tap_ids):
            raise ValueError("golden taps differ from source-pinned native family layer inputs")
        self.vocab_size, self.target_width = manifest["vocab_size"], manifest["target_width"]
        if any(
            type(n) is not int or n <= 0
            for n in (self.vocab_size, self.target_width, max_capture_bytes)
        ):
            raise ValueError("golden vocabulary/width/storage bound must be positive integers")
        self._fingerprints = {}
        self._arrays, self.chains = {}, {}
        self.native_receipts = {}

        def file(record, directory):
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError("golden requires pinned original file descriptors")
            path = (directory / record["path"]).resolve()
            if file_sha256(path) != record["sha256"]:
                raise ValueError("golden producer artifact SHA256 differs")
            stat = path.stat()
            self._fingerprints[path] = (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_dev,
                stat.st_ino,
            )
            return path

        inventory = json.loads(file(manifest["train_inventory"], self.path.parent).read_text())
        if inventory.get("schema") != "block_train_inventory_v1":
            raise ValueError("golden needs original TRAIN inventory")
        producers = set()
        hardware = None
        total_bytes = 0
        for case in manifest["cases"]:
            if not isinstance(case, dict) or set(case) != {"chain_id", "native_receipt"}:
                raise ValueError("golden case must bind exact original native receipt")
            cid = case["chain_id"]
            if not isinstance(cid, str) or not cid or cid in self.chains:
                raise ValueError("duplicate/invalid native golden chain")
            receipt_path = file(case["native_receipt"], self.path.parent)
            native = json.loads(receipt_path.read_text())
            ancestry = native.get("chain_ancestry", {})
            if (
                set(ancestry)
                != {"prompt_id", "prompt_sha256", "domain", "source_split", "prompt_length"}
                or ancestry["source_split"] != "TRAIN"
                or inventory.get("prompts", {}).get(ancestry["prompt_id"])
                != {
                    "sha256": ancestry["prompt_sha256"],
                    "domain": ancestry["domain"],
                    "split": "TRAIN",
                }
            ):
                raise ValueError("golden prefix is not original authenticated TRAIN")
            tokens = native.get("tokens")
            if (
                not isinstance(tokens, list)
                or not tokens
                or len(tokens) > 32768
                or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens)
                or type(ancestry["prompt_length"]) is not int
                or not 1 <= ancestry["prompt_length"] <= len(tokens)
            ):
                raise ValueError("golden exact native prefix/prompt boundary invalid")
            execution = native.get("executed_result_buffers", [])
            if (
                native.get("schema") != "block_native_teacher_request_v1"
                or native.get("complete") is not True
                or native.get("optimizer_updates") != 0
                or native.get("target_sha256") != manifest["target_sha256"]
                or native.get("target_precision") != "F16"
                or native.get("kv_type") != "F16"
                or native.get("tap_ids") != list(self.tap_ids)
                or native.get("teacher_context_reset_between_requests") is not True
                or native.get("prefix_contract") != "teacher_forced_exact_caller_token_ids"
                or native.get("prefix_freshness") != "caller_current_student_prefix"
                or any(k in native for k in ("generation", "prompt", "prompt_source_sha256"))
                or not execution
                or not all(isinstance(n, str) and re.fullmatch(r"CUDA[0-9]+", n) for n in execution)
            ):
                raise ValueError("golden lacks exact native CUDA/F16/target/prefix/tap provenance")
            producer = tuple(
                native.get(k)
                for k in (
                    "producer_source_revision",
                    "producer_binary_sha256",
                    "client_source_sha256",
                )
            )
            validate_decode_history(native.get("decode_history"), len(tokens))
            if (
                not isinstance(producer[0], str)
                or len(producer[0]) != 40
                or any(not isinstance(h, str) or len(h) != 64 for h in producer[1:])
            ):
                raise ValueError("golden native source/binary/client identities missing")
            producers.add(producer)
            if hardware is not None and hardware != native.get("hardware"):
                raise ValueError("mixed original golden producer hardware")
            hardware = native.get("hardware")
            arrays = []
            for name in ("features", "logits"):
                record = native.get("files", {}).get(name, {})
                shape = (
                    [len(tokens), len(self.tap_ids), self.target_width]
                    if name == "features"
                    else (
                        [len(tokens) if native.get("logits_mode") == "all" else 1, self.vocab_size]
                    )
                )
                if (
                    set(record) != {"path", "sha256", "shape", "dtype"}
                    or record["shape"] != shape
                    or record["dtype"] != "float32"
                    or native.get("logits_mode") not in ("all", "last")
                ):
                    raise ValueError(
                        "golden raw native feature/full-vocabulary logit shape differs"
                    )
                path = (receipt_path.parent / record["path"]).resolve()
                total_bytes += path.stat().st_size
                if total_bytes > max_capture_bytes:
                    raise MemoryError("bounded golden capture storage exceeds declared cap")
                path = file({"path": str(path), "sha256": record["sha256"]}, receipt_path.parent)
                if path.stat().st_size != 4 * int(np.prod(shape)):
                    raise ValueError("golden raw artifact byte count differs")
                value = np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))
                for first in range(0, len(value), 16):
                    if not np.isfinite(value[first : first + 16]).all():
                        raise ValueError("golden raw native teacher nonfinite")
                arrays.append(value)
            self._arrays[cid] = (np.asarray(tokens, dtype=np.int64), *arrays)
            self.chains[cid] = ancestry | {"chain_id": cid, "anchors": [len(tokens) - 1]}
            self.native_receipts[cid] = native
        if not self.chains or len(producers) != 1:
            raise ValueError("empty or mixed-source native golden capture")
        self.manifest = manifest | {
            "producer": {
                "target_sha256": manifest["target_sha256"],
                "hardware": json.dumps(hardware, sort_keys=True),
            }
        }

    def load_block(self, chain_id, block_index, *, require_teacher=False):
        if chain_id not in self.chains or block_index != 0:
            raise ValueError("golden prefix index invalid")
        for path, expected in self._fingerprints.items():
            stat = path.stat()
            if (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_dev,
                stat.st_ino,
            ) != expected:
                raise ValueError("golden source changed after admission")
        return None


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
        or receipt.get("prefix_freshness") != "caller_current_student_prefix"
        or any(k in receipt for k in ("generation", "prompt", "prompt_source_sha256"))
        or receipt.get("tap_ids") != list(getattr(dataset, "tap_ids", TAPS))
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


def run_gate(
    args,
    *,
    teacher_factory=None,
    device_query=cuda_inventory,
    dataset_factory=BlockDataset,
    expected_family=None,
):
    production = teacher_factory is None
    if args.output.exists() or args.output_root.exists():
        raise ValueError("refuse to overwrite portability evidence")
    if file_sha256(args.binary) != args.binary_sha256:
        raise ValueError("native binary differs from admitted SHA256")
    dataset = dataset_factory(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        admission_path=args.admission,
        admission_sha256=args.admission_sha256,
    )
    if expected_family is not None and dataset.manifest["family"] != expected_family:
        raise ValueError("native golden manifest family differs from requested gate")
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
        "tap_ids": list(getattr(dataset, "tap_ids", TAPS)),
        "reference_profile": "native_target_only_exact_prefix_partitioned_train",
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
            native_source = dataset.native_receipts.get(cid)
            if native_source is None:
                raise ValueError("portability requires original native decode partitions")
            decode_history = crop_decode_history(native_source["decode_history"], length)
            receipt = teacher.capture_prefix(
                prefix,
                getattr(dataset, "tap_ids", TAPS),
                logits_mode="last",
                chain_ancestry=ancestry,
                decode_history=decode_history,
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
            if receipt.get("decode_history") != decode_history:
                raise ValueError("fresh teacher changed source decode partitions or KV ancestry")
            fresh_features = teacher.array(receipt, "features")
            fresh_logits = teacher.array(receipt, "logits")
            feature_check = compare_matrix(
                saved_features[:length],
                fresh_features,
                atol=args.feature_atol,
                rtol=args.feature_rtol,
            )
            saved_last = (
                saved_logits[:1] if len(saved_logits) == 1 else saved_logits[length - 1 : length]
            )
            logit_check = compare_matrix(
                saved_last,
                fresh_logits,
                atol=args.logit_atol,
                rtol=args.logit_rtol,
            )
            original_decision = int(np.argmax(saved_last[0]))
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
                    "decode_history": decode_history,
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


def main(*, expected_family=None, golden_only=False):
    parser = argparse.ArgumentParser(
        description=(
            "Bounded native EAGLE three-tap TRAIN golden CUDA portability gate."
            if expected_family == "eagle"
            else __doc__
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--golden-manifest", type=Path)
    for name in ("binary", "target", "output-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    for name in (
        "manifest-sha256",
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
    use_goldens = golden_only or args.golden_manifest is not None
    if args.golden_manifest is not None:
        args.manifest = args.golden_manifest
    try:
        if not use_goldens and (args.admission is None or args.admission_sha256 is None):
            raise ValueError("block-corpus portability requires pinned completed admission")
        report = run_gate(
            args,
            dataset_factory=NativeCaptureGoldens if use_goldens else BlockDataset,
            expected_family=expected_family,
        )
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
