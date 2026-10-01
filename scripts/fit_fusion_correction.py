#!/usr/bin/env python3
"""Fit one raw-input fusion residual from explicitly joined training operands.

This script has no capture/model/GPU access. Preparation-only or diagnostic
operands cannot be relabeled as training. Real-data fitting needs a separate
explicit invocation with --allow-real-data-fit; development uses synthetic data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from w1a1_eagle.fusion_correction import FusionCorrection, FusionCorrectionConfig  # noqa: E402

SOURCE_KEYS = {"base_weights_sha256", "reference_weights_sha256", "quantizer_sha256"}
ARRAY_KEYS = {
    "raw_input", "binary_output", "reference_output", "raw_join_ids",
    "binary_join_ids", "reference_join_ids",
}
ROW_KEYS = {
    "row_id", "prompt_id", "prompt_sha256", "domain", "depth", "position", "split",
    "source_quantizer_sha256", "raw_input_sha256", "binary_output_sha256",
    "reference_output_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_sha256(row: np.ndarray) -> str:
    """Canonical little-endian F32 row digest, never a float-value approximation."""
    return hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest()


def _is_sha256(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def load_joined_calibration(manifest_path: Path, *, allow_real_data_fit: bool = False):
    """Fail closed on missing joins, ancestry, prompt split, or training eligibility."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    keys = {"version", "projection", "raw_input_stage", "source_data_split", "eligibility",
            "synthetic", "source", "operands", "operands_sha256", "rows"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError("calibration manifest fields differ from version 1")
    if (manifest["version"] != 1 or type(manifest["version"]) is not int
            or manifest["projection"] != "fc"
            or manifest["raw_input_stage"] != "pre_activation_quantization"
            or manifest["source_data_split"] != "train"
            or manifest["eligibility"] != "training_allowed"):
        raise ValueError("only joined train-eligible raw fusion operands may be fitted")
    if type(manifest["synthetic"]) is not bool:
        raise ValueError("synthetic flag must be a boolean")
    if not manifest["synthetic"] and not allow_real_data_fit:
        raise ValueError("real-data fitting requires explicit --allow-real-data-fit")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError("source weight/reference/quantizer identity is required")
    if not all(_is_sha256(value) for value in source.values()):
        raise ValueError("source identities must be SHA256")
    if type(manifest["operands"]) is not str or not _is_sha256(manifest["operands_sha256"]):
        raise ValueError("operand path and SHA256 required")
    operands_path = (manifest_path.parent / manifest["operands"]).resolve()
    if sha256(operands_path) != manifest["operands_sha256"]:
        raise ValueError("operand archive SHA256 differs")
    with np.load(operands_path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError("operand archive must include raw/binary/reference join IDs")
        arrays = {name: archive[name].copy() for name in archive.files}
    rows = manifest["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibration rows are required")
    for name in ("raw_input", "binary_output", "reference_output"):
        value = arrays[name]
        if (value.ndim != 2 or value.shape[0] != len(rows) or value.shape[1] < 1
                or value.dtype != np.dtype("float32") or not np.isfinite(value).all()):
            raise ValueError(f"{name}: expected finite F32 matrix matching rows")
    if arrays["binary_output"].shape != arrays["reference_output"].shape:
        raise ValueError("binary/reference output shapes differ")
    for name in ("raw_join_ids", "binary_join_ids", "reference_join_ids"):
        if arrays[name].shape != (len(rows),) or arrays[name].dtype.kind != "U":
            raise ValueError(f"{name}: expected one Unicode join ID per row")
    prompt_ownership, row_ids, coordinates = {}, set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("row metadata must include prompt/depth/domain and hashes")
        if any(type(row[name]) is not str or not row[name]
               for name in ("row_id", "prompt_id", "domain")):
            raise ValueError("row/prompt/domain IDs must be nonempty strings")
        if any(type(row[name]) is not int or row[name] < 0 for name in ("depth", "position")):
            raise ValueError("depth and position must be nonnegative integers")
        if row["split"] not in ("train", "validation"):
            raise ValueError("calibration only allows train and prompt-held-out validation")
        if row["row_id"] in row_ids:
            raise ValueError("duplicate calibration row ID")
        row_ids.add(row["row_id"])
        coordinate = (row["prompt_id"], row["depth"], row["position"])
        if coordinate in coordinates:
            raise ValueError("duplicate prompt/depth/position join")
        coordinates.add(coordinate)
        if not _is_sha256(row["prompt_sha256"]):
            raise ValueError("prompt identity hash required")
        owner = (row["prompt_sha256"], row["domain"], row["split"])
        if prompt_ownership.setdefault(row["prompt_id"], owner) != owner:
            raise ValueError("prompt crosses split, domain, or content identity")
        if row["source_quantizer_sha256"] != source["quantizer_sha256"]:
            raise ValueError("source quantizer differs: fit each quantizer separately")
        for name in ("raw_join_ids", "binary_join_ids", "reference_join_ids"):
            if str(arrays[name][index]) != row["row_id"]:
                raise ValueError("raw/binary/reference row joins disagree")
        for name in ("raw_input", "binary_output", "reference_output"):
            if row[f"{name}_sha256"] != row_sha256(arrays[name][index]):
                raise ValueError(f"{name}: row operand SHA256 differs")
    # Content-identical prompts under different aliases cannot leak into validation.
    train_hashes = {r["prompt_sha256"] for r in rows if r["split"] == "train"}
    validation_hashes = {r["prompt_sha256"] for r in rows if r["split"] == "validation"}
    if not train_hashes or not validation_hashes or train_hashes & validation_hashes:
        raise ValueError("training and validation must hold out disjoint prompt content")
    return arrays, manifest, {
        "manifest_sha256": sha256(manifest_path), "operands_sha256": sha256(operands_path),
        "source": source, "synthetic": manifest["synthetic"],
    }


def fit_reduced_rank(
    raw_input: torch.Tensor, residual: torch.Tensor, config: FusionCorrectionConfig,
    *, ridge: float = 1e-3,
) -> FusionCorrection:
    """Train-only ridge reduced-rank regression; dual solve avoids in-width cubed cost."""
    if not config.enabled:
        raise ValueError("fitting requires enabled fusion correction")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite positive")
    if (raw_input.ndim != 2 or residual.ndim != 2
            or raw_input.shape[0] != residual.shape[0] or raw_input.shape[0] <= config.rank
            or min(raw_input.shape[1], residual.shape[1]) < config.rank):
        raise ValueError("fitting needs matching 2D operands with more train rows than rank")
    if raw_input.device.type != "cpu" or residual.device.type != "cpu":
        raise ValueError("calibration fitting is CPU-only")
    if not bool(torch.isfinite(raw_input).all()) or not bool(torch.isfinite(residual).all()):
        raise ValueError("calibration operands must be finite")
    x, y = raw_input.double(), residual.double()
    xmean, ymean = x.mean(0), y.mean(0)
    if config.output_bias:
        x, y = x - xmean, y - ymean
    gram = x @ x.T
    dual = torch.linalg.solve(gram + ridge * torch.eye(len(x), dtype=x.dtype), y)
    coefficient = x.T @ dual
    predicted = x @ coefficient
    _, _, vh = torch.linalg.svd(predicted, full_matrices=False)
    output_basis = vh[:config.rank].T
    module = FusionCorrection(x.shape[1], y.shape[1], config)
    with torch.no_grad():
        module.u.copy_(output_basis.float())
        module.v.copy_((coefficient @ output_basis).T.float())
        if config.output_bias:
            # Bias is a separate bounded mean-offset control, not an extra rank.
            module.output_bias.copy_(
                (ymean - module(xmean.float())).float().clamp(-config.bias_bound, config.bias_bound)
            )
    module.project_()
    return module


def fit_manifest(manifest_path: Path, config: FusionCorrectionConfig, *, ridge: float = 1e-3,
                 allow_real_data_fit: bool = False):
    arrays, manifest, ancestry = load_joined_calibration(
        manifest_path, allow_real_data_fit=allow_real_data_fit,
    )
    train = torch.tensor([r["split"] == "train" for r in manifest["rows"]])
    raw = torch.from_numpy(arrays["raw_input"])
    residual = torch.from_numpy(arrays["reference_output"] - arrays["binary_output"])
    module = fit_reduced_rank(raw[train], residual[train], config, ridge=ridge)
    with torch.no_grad():
        error = residual - module(raw)
    metrics = {}
    for split in ("train", "validation"):
        indices = [i for i, row in enumerate(manifest["rows"]) if row["split"] == split]
        metrics[split] = {
            "rows": len(indices),
            "prompts": len({manifest["rows"][i]["prompt_id"] for i in indices}),
            "base_mse": float(residual[indices].square().mean()),
            "corrected_mse": float(error[indices].square().mean()),
        }
    strata = {}
    for index, row in enumerate(manifest["rows"]):
        key = (row["split"], row["domain"], row["depth"], row["source_quantizer_sha256"])
        strata.setdefault(key, []).append(index)
    report = {
        "version": 1,
        "execution": "synthetic_fit" if ancestry["synthetic"] else "train_calibration",
        "hardware": "CPU F64 ridge fit, F32 reductions with F16 factors for evaluation",
        "ancestry": ancestry, "ridge": ridge, "correction": module.manifest_payload(),
        "metrics": metrics,
        "strata": [
            {"split": key[0], "domain": key[1], "depth": key[2], "quantizer_sha256": key[3],
             "rows": len(indices), "base_mse": float(residual[indices].square().mean()),
             "corrected_mse": float(error[indices].square().mean())}
            for key, indices in sorted(strata.items())
        ],
        "validation_contract": "prompt_content_held_out_within_one_source_quantizer",
        "native_acceptance_measured": False, "native_throughput_measured": False,
    }
    return module, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(1, 4), default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output-bias", action="store_true")
    parser.add_argument("--bias-bound", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-real-data-fit", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    config = FusionCorrectionConfig(True, args.rank, args.output_bias, args.bias_bound, args.seed)
    module, report = fit_manifest(
        args.manifest, config, ridge=args.ridge, allow_real_data_fit=args.allow_real_data_fit,
    )
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "fusion_correction.pt"
    torch.save({"correction": module.state_payload(), "ancestry": report["ancestry"]}, checkpoint)
    report["checkpoint_sha256"] = sha256(checkpoint)
    (args.output_dir / "fit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
