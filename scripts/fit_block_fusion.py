#!/usr/bin/env python3
"""Calibrate fixed A8/A1 fusion from authenticated prompt-disjoint TRAIN chains.

Reference is the original F32-promoted fusion weight CPU product, not native
teacher logits. Raw and post-fusion RMS diagnostics are explicitly surrogates.
Output fc latent/scales can be merged into the native owner's complete binary
checkpoint; fusion-only NPZ does not claim a complete model export.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from w1a1_eagle.block_data import BlockDataset, file_sha256  # noqa: E402
from w1a1_eagle.block_fusion import (  # noqa: E402
    FusionFitConfig,
    diagnostics,
    fit_fusion,
    project,
    quantize,
    rms_norm_reference,
    validate_norm_descriptor,
)


def select_rows(dataset, split, rows_per_chain, max_total_rows):
    chunks, selected, remaining = [], [], max_total_rows
    for cid in sorted(dataset.chains):
        chain = dataset.chains[cid]
        if chain["split"] != split:
            continue
        # Features are layer inputs at committed context rows, before anchor.
        stop = chain["anchors"][-1]
        count = min(stop, rows_per_chain)
        if count == 0:
            raise ValueError("calibration chain has no committed context rows")
        if count > remaining:
            raise MemoryError("calibration row cap cannot cover all selected prompt groups")
        positions = np.unique(np.linspace(0, stop - 1, count).astype(int))
        chunks.append(dataset.copy_feature_rows(cid, positions).reshape(len(positions), -1))
        selected.append(
            {
                "chain_id": cid,
                "prompt_id": chain["prompt_id"],
                "prompt_sha256": chain["prompt_sha256"],
                "domain": chain["domain"],
                "positions": positions.tolist(),
            }
        )
        remaining -= len(positions)
    if not chunks:
        raise ValueError(f"missing prompt-disjoint {split} captures")
    if {r["domain"] for r in selected} != {"prose", "code", "reasoning"}:
        raise ValueError("fusion calibration requires prose/code/reasoning in each split")
    return np.ascontiguousarray(np.concatenate(chunks)), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--norm-sha256", required=True)
    parser.add_argument("--norm-metadata", type=Path, required=True)
    parser.add_argument("--norm-metadata-sha256", required=True)
    parser.add_argument("--norm-epsilon", type=float, required=True)
    parser.add_argument("--activation-bits", type=int, choices=(1, 8), required=True)
    parser.add_argument("--orientation-rescue", action="store_true")
    parser.add_argument("--coordinate-flips", type=int, default=0)
    parser.add_argument(
        "--latent-initialization",
        choices=("preserve_reference_magnitudes", "unit_probe"),
        default="preserve_reference_magnitudes",
    )
    parser.add_argument("--rows-per-chain", type=int, default=32)
    parser.add_argument("--max-total-rows", type=int, default=512)
    parser.add_argument("--max-array-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("refuse to overwrite historical fit")
    if min(args.rows_per_chain, args.max_total_rows, args.max_array_bytes) <= 0:
        raise ValueError("positive row/memory bounds required")
    if (args.admission is None) != (args.admission_sha256 is None):
        raise ValueError("completed admission requires both path and external SHA256 pin")
    dataset = BlockDataset(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        admission_path=args.admission,
        admission_sha256=args.admission_sha256,
    )
    for path, pin in ((args.weights, args.weights_sha256), (args.norm, args.norm_sha256)):
        if file_sha256(path) != pin:
            raise ValueError("original fusion/norm source differs from external pin")
    if file_sha256(args.norm_metadata) != args.norm_metadata_sha256:
        raise ValueError("original model norm metadata differs from external pin")
    reference = json.loads(args.norm_metadata.read_text())
    args.norm_epsilon = validate_norm_descriptor(
        reference,
        family=dataset.manifest["family"],
        weights_sha256=args.weights_sha256,
        norm_sha256=args.norm_sha256,
        supplied_epsilon=args.norm_epsilon,
    )
    weight = np.load(args.weights, mmap_mode="r", allow_pickle=False)
    norm = np.load(args.norm, mmap_mode="r", allow_pickle=False)
    if (
        weight.dtype != np.float32
        or weight.ndim != 2
        or weight.shape[1] != 5 * dataset.target_width
        or norm.dtype != np.float32
        or norm.shape != (len(weight),)
        or reference["fc_source"]["shape"] != list(weight.shape)
        or reference["norm_source"]["shape"] != list(norm.shape)
    ):
        raise ValueError("original fusion/norm projection shape/type differs")
    # Deliberately conservative workspace bound including design/sign/teacher
    # temporaries; process RSS is a separate independently measured gate.
    estimate = weight.nbytes * 6 + args.max_total_rows * (weight.shape[1] * 32 + len(weight) * 32)
    if estimate > args.max_array_bytes:
        raise MemoryError(f"estimated fusion workspace {estimate} exceeds declared array cap")
    fit_x, fit_rows = select_rows(
        dataset, "calibration_fit", args.rows_per_chain, args.max_total_rows
    )
    validation_x, validation_rows = select_rows(
        dataset, "calibration_validation", args.rows_per_chain, args.max_total_rows
    )
    fit_y = (fit_x @ weight.T).astype(np.float32)
    validation_y = (validation_x @ weight.T).astype(np.float32)
    config = FusionFitConfig(
        args.activation_bits,
        args.orientation_rescue,
        args.coordinate_flips,
        args.max_seconds,
        args.latent_initialization,
    )
    candidate = fit_fusion(fit_x, fit_y, weight, config)
    # Freeze/export before reading validation teacher through candidate; signs
    # and scales are never selected or changed using validation values.
    args.output_dir.mkdir(parents=True)
    artifact = args.output_dir / "fusion_candidate.npz"
    np.savez(artifact, **{"fc.latent": candidate["latent"], "fc.scale": candidate["scale"]})
    control_path = args.output_dir / "scale_only_control.npz"
    np.savez(
        control_path,
        **{"fc.latent": candidate["control_latent"], "fc.scale": candidate["control_scale"]},
    )
    codes, beta = quantize(validation_x, args.activation_bits)
    predicted = project(codes, beta, candidate["hard_signs"], candidate["scale"])
    control = project(codes, beta, candidate["control_hard_signs"], candidate["control_scale"])
    report = {
        "schema": "block_fusion_fit_v1",
        "config": asdict(config),
        "data_sha256": dataset.sha256,
        "weights_sha256": args.weights_sha256,
        "norm_sha256": args.norm_sha256,
        "reference_metadata_sha256": args.norm_metadata_sha256,
        "original_model_sha256": reference["model_sha256"],
        "norm_epsilon_f32_bits": reference["epsilon_f32_bits"],
        "norm_epsilon": args.norm_epsilon,
        "producer_hardware": dataset.manifest["producer"]["hardware"],
        "calibration_hardware": "CPU NumPy F32 product; native execution pending",
        "teacher_arithmetic": "original_promoted_f32_weight_cpu_blas_raw_features",
        "estimated_array_bytes": estimate,
        "fit_rows": fit_rows,
        "validation_rows": validation_rows,
        "fit": candidate["report"],
        "artifact_sha256": file_sha256(artifact),
        "control_sha256": file_sha256(control_path),
        "validation": {
            "raw_candidate": diagnostics(predicted, validation_y),
            "raw_scale_only": diagnostics(control, validation_y),
            "post_norm_candidate": diagnostics(
                rms_norm_reference(predicted, norm, args.norm_epsilon),
                rms_norm_reference(validation_y, norm, args.norm_epsilon),
            ),
            "post_norm_scale_only": diagnostics(
                rms_norm_reference(control, norm, args.norm_epsilon),
                rms_norm_reference(validation_y, norm, args.norm_epsilon),
            ),
        },
        "native_export_status": (
            "fusion-only initializer; merge into full checkpoint then validate native export"
        ),
        "quality_status": "PENDING native trajectory/quality evaluation on authorized final device",
    }
    (args.output_dir / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"artifact_sha256": report["artifact_sha256"], "validation": report["validation"]}
        )
    )


if __name__ == "__main__":
    main()
