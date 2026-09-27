#!/usr/bin/env python3
"""One bounded target-aligned head fit; never loads or trains a body/target.

Optimizer constants are frozen below, with no CLI hyperparameter overrides.
Requires audited train data plus native zero-export/behavior and logits gates.
Exports strict FP16 .npy heads at0/100/250/500 and any time-stop endpoint.
Native default MMVF half partial accumulation differs from this F32 matmul
surrogate; measured parity errors and source arithmetic evidence are mandatory.
Live native development acceptance selects the endpoint outside this script.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .prepare_frozen_d_head_data import SCHEMA, resolve, sha256
except ImportError:
    from prepare_frozen_d_head_data import SCHEMA, resolve, sha256

PROTOCOL = {
    "optimizer": "AdamW", "learning_rate": 1e-4, "batch_size": 64, "weight_decay": 0.0,
    "relative_l2_coefficient": 0.01, "relative_l2_definition": "mean((W-W0)^2)/(mean(W0^2)+1e-12)",
    "seed": 42, "max_steps": 500, "max_optimizer_seconds": 2700,
    "checkpoints": [0, 100, 250, 500], "selection": "live native dev accepted/round; tie earlier checkpoint",
    "time_stop_endpoint_eligible": True, "target": "mapped cloned native verifier sampler at each actual proposal prefix",
    "weight_forward": "F32 master straight-through FP16 roundtrip", "accumulation": "F32 BLAS surrogate; TF32 disabled",
    "data_order": "seed42 shuffled complete passes without replacement",
    "parity_atol": 0.05, "parity_rtol": 0.005,
}


def roundtrip_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float16).to(torch.float32)


def head_logits(states: torch.Tensor, weight: torch.Tensor, input_cast: str) -> torch.Tensor:
    if input_cast not in ("fp16", "f32"):
        raise ValueError("input cast must be native-audited fp16 or f32")
    inputs = states.to(torch.float16).to(torch.float32) if input_cast == "fp16" else states.float()
    return F.linear(inputs, roundtrip_weight(weight))


def objective(states, labels, weight, initial, input_cast):
    logits = head_logits(states, weight, input_cast)
    ce = F.cross_entropy(logits, labels)
    relative_l2 = (weight - initial).square().mean() / (initial.square().mean() + 1e-12)
    return ce + PROTOCOL["relative_l2_coefficient"] * relative_l2, ce, relative_l2


def load_training_data(directory: Path):
    report = json.loads((directory / "manifest.json").read_text())
    if report.get("schema") != SCHEMA or any(report.get(k) != v for k, v in
            {"split": "train", "body": "D", "trajectory": "own_history", "training_prompts": 96}.items()):
        raise ValueError("unrecognized/nontraining prepared data")
    if sha256(directory / "dataset.npz") != report.get("dataset_sha256"):
        raise ValueError("prepared dataset hash mismatch")
    if sha256(directory / "rows.jsonl") != report.get("rows_sha256"):
        raise ValueError("prepared row provenance hash mismatch")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    states, labels = data["states"], data["labels"]
    if (states.dtype != np.float32 or states.ndim != 2 or not np.isfinite(states).all()
            or labels.dtype != np.int64 or labels.shape != (len(states),)
            or not 0 < len(states) <= 8160 or len(states) != report["selected_rows"]
            or np.any(labels < 0) or np.any(labels >= len(data["d2t"]))):
        raise ValueError("invalid selected training states/labels")
    if not np.array_equal(data["selected_indices"], np.flatnonzero(data["selected_mask"])):
        raise ValueError("selected row index/mask mismatch")
    if np.any(data["selected_mask"] & ~(data["valid_mask"] & data["supported_mask"])):
        raise ValueError("selection includes invalid or unsupported labels")
    return data, report


def parity_gate(path: Path, dataset_hash: str, head_path: Path, weight: torch.Tensor, device: str) -> dict:
    gate = json.loads(path.read_text())
    if gate.get("initial_head_sha256") != sha256(head_path) or gate.get("dataset_sha256") != dataset_hash:
        raise ValueError("parity evidence is not bound to this initialization and dataset")
    for key in ("native_export_zero_passed", "native_zero_behavior_passed"):
        if gate.get(key) is not True:
            raise ValueError(f"missing required gate: {key}")
    input_cast = gate.get("input_cast")
    if input_cast not in ("fp16", "f32") or not gate.get("native_input_cast_evidence"):
        raise ValueError("native input cast must have source/dispatch evidence")
    arrays = {}
    for key in ("states", "logits"):
        file = resolve(path.parent, gate[f"{key}_path"])
        if sha256(file) != gate.get(f"{key}_sha256"):
            raise ValueError(f"native parity {key} hash mismatch")
        arrays[key] = np.load(file, allow_pickle=False)
    x, native = arrays["states"], arrays["logits"]
    if (x.ndim != 2 or x.shape[1] != weight.shape[1] or not len(x)
            or native.shape != (len(x), weight.shape[0])
            or not np.isfinite(x).all() or not np.isfinite(native).all()):
        raise ValueError("invalid native parity states/logits")
    with torch.no_grad():
        calculated = head_logits(torch.from_numpy(x).to(device), weight, input_cast).cpu().numpy()
    error = np.abs(calculated.astype(np.float64) - native.astype(np.float64))
    tolerance = PROTOCOL["parity_atol"] + PROTOCOL["parity_rtol"] * np.abs(native)
    native_top = native.argmax(1)
    offline_top = calculated.argmax(1)
    mismatches = native_top != offline_top
    tie_mismatches = []
    for row in np.flatnonzero(mismatches):
        a, b = native_top[row], offline_top[row]
        gap = float(native[row, a] - native[row, b])
        allowed = float(tolerance[row, a] + tolerance[row, b])
        tie_mismatches.append({"row": int(row), "native_argmax": int(a), "offline_argmax": int(b),
                               "native_gap": gap, "tolerance_sum": allowed, "within_tolerance": gap <= allowed})
    passed = bool(np.all(error <= tolerance) and all(r["within_tolerance"] for r in tie_mismatches))
    result = {"passed": passed, "rows": len(x), "logit_count": int(native.size),
              "max_absolute_error": float(error.max()), "p95_absolute_error": float(np.quantile(error, .95)),
              "mean_absolute_error": float(error.mean()), "max_tolerance_ratio": float((error / tolerance).max()),
              "argmax_mismatches": int(mismatches.sum()), "tie_mismatches": tie_mismatches,
              "input_cast": input_cast, "native_input_cast_evidence": gate["native_input_cast_evidence"],
              "gate_sha256": sha256(path), "atol": PROTOCOL["parity_atol"], "rtol": PROTOCOL["parity_rtol"]}
    if not passed:
        raise ValueError("native/offline logits parity failed: " + json.dumps(result))
    return result


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def export_checkpoint(directory: Path, step: int, weight: torch.Tensor, initial: torch.Tensor) -> dict:
    exported = weight.detach().half().cpu().numpy()
    if not np.isfinite(exported).all():
        raise ValueError("nonfinite deployment export")
    path = directory / f"head-step-{step:04d}.npy"
    np.save(path, exported, allow_pickle=False)
    if not np.array_equal(np.load(path, allow_pickle=False), exported):
        raise ValueError("checkpoint export/reload mismatch")
    delta = (weight.detach() - initial).double()
    return {"step": step, "path": path.name, "sha256": sha256(path), "dtype": "float16",
            "shape": list(exported.shape), "weight_delta_l2": float(delta.square().sum().sqrt()),
            "weight_delta_max_abs": float(delta.abs().max()), "eligible_for_native_selection": True}


def optimize(states, labels, initial, input_cast, directory, device,
             max_steps=PROTOCOL["max_steps"], max_seconds=PROTOCOL["max_optimizer_seconds"]):
    """Internal smaller bounds support CPU tests; CLI always uses frozen constants."""
    if not 0 <= max_steps <= PROTOCOL["max_steps"] or not 0 < max_seconds <= PROTOCOL["max_optimizer_seconds"]:
        raise ValueError("optimizer bound exceeds approved protocol")
    random.seed(PROTOCOL["seed"])
    np.random.seed(PROTOCOL["seed"])
    torch.manual_seed(PROTOCOL["seed"])
    weight = initial.detach().clone().float().to(device).requires_grad_(True)
    initial = initial.detach().float().to(device)
    x, y = states.to(device), labels.to(device)
    optimizer = torch.optim.AdamW([weight], lr=PROTOCOL["learning_rate"], weight_decay=0.0)
    checkpoints = [export_checkpoint(directory, 0, weight, initial)]
    exposure = np.zeros(len(x), dtype=np.int64)
    rng = np.random.default_rng(PROTOCOL["seed"])
    order, offset, completed_passes, seconds, step = rng.permutation(len(x)), 0, 0, 0.0, 0
    stop_reason = "max_steps"
    with (directory / "steps.jsonl").open("w") as log:
        while step < max_steps:
            if seconds >= max_seconds:
                stop_reason = "optimizer_time_cap"
                break
            indices = order[offset:offset + PROTOCOL["batch_size"]]
            offset += len(indices)
            exposure[indices] += 1
            if offset == len(order):
                completed_passes += 1
                order, offset = rng.permutation(len(x)), 0
            synchronize(device)
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, ce, penalty = objective(x[indices], y[indices], weight, initial, input_cast)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite optimization objective; stopped")
            loss.backward()
            if weight.grad is None or not torch.isfinite(weight.grad).all():
                raise ValueError("nonfinite optimization gradient; stopped")
            grad_norm = float(weight.grad.detach().double().square().sum().sqrt())
            optimizer.step()
            if not torch.isfinite(weight).all():
                raise ValueError("nonfinite optimized weight; stopped")
            synchronize(device)
            seconds += time.monotonic() - start
            step += 1
            record = {"step": step, "loss": float(loss.detach()), "cross_entropy": float(ce.detach()),
                      "relative_l2": float(penalty.detach()), "gradient_l2": grad_norm,
                      "optimizer_seconds": seconds, "exposures": int(exposure.sum()),
                      "completed_passes": completed_passes, "fractional_passes": float(exposure.sum()/len(x)),
                      "batch_rows": len(indices)}
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            if step in PROTOCOL["checkpoints"]:
                checkpoints.append(export_checkpoint(directory, step, weight, initial))
        if checkpoints[-1]["step"] != step:
            checkpoints.append(export_checkpoint(directory, step, weight, initial))
    # Keep the final F32 master for reproducibility, but only FP16 heads deploy.
    np.save(directory / "final-master-f32.npy", weight.detach().cpu().numpy(), allow_pickle=False)
    np.save(directory / "row-exposures.npy", exposure, allow_pickle=False)
    return {"status": "completed", "steps": step, "optimizer_seconds": seconds, "stop_reason": stop_reason,
            "time_cap_overshoot_seconds": max(0.0, seconds-max_seconds), "checkpoints": checkpoints,
            "exposures": int(exposure.sum()), "completed_passes": completed_passes,
            "fractional_passes": float(exposure.sum()/len(x)), "minimum_row_exposures": int(exposure.min()),
            "maximum_row_exposures": int(exposure.max()), "selection_pending_native_development": True}


def fit(args):
    data, manifest = load_training_data(args.data_dir)
    initial = np.load(args.initial_head, allow_pickle=False)
    if (initial.dtype != np.float16 or initial.shape != (len(data["d2t"]), data["states"].shape[1])
            or not np.isfinite(initial).all()):
        raise ValueError("initial head must be finite original FP16 [draft_vocab,H]")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    original = torch.from_numpy(initial.astype(np.float32)).to(args.device)
    parity = parity_gate(args.parity_gate, manifest["dataset_sha256"], args.initial_head, original, args.device)
    if args.check_only:
        return {"status": "gates_passed", "parity": parity, "protocol": PROTOCOL}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "frozen_d_head_fit_v1", "status": "running", "protocol": PROTOCOL,
              "parity": parity, "initial_head_sha256": sha256(args.initial_head),
              "dataset_sha256": manifest["dataset_sha256"], "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "device": args.device,
              "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU",
              "training_valid_rows": manifest["valid_rows"], "training_unsupported_rows": manifest["unsupported_rows"],
              "selected_rows": len(data["states"])}
    report_path = args.output_dir / "fit-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        report.update(optimize(torch.from_numpy(data["states"]), torch.from_numpy(data["labels"]),
                               original, parity["input_cast"], args.output_dir, args.device))
        if report["checkpoints"][0]["sha256"] != sha256(args.initial_head):
            # .npy header versions may differ; require bit-identical payload instead.
            zero = np.load(args.output_dir / report["checkpoints"][0]["path"], allow_pickle=False)
            if not np.array_equal(zero.view(np.uint16), initial.view(np.uint16)):
                raise ValueError("zero checkpoint differs from original initialization")
    except Exception as error:
        report.update({"status": "failed", "failure": str(error), "selection_pending_native_development": False})
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial-head", type=Path, required=True)
    parser.add_argument("--parity-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
