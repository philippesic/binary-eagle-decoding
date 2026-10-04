#!/usr/bin/env python3
import argparse
import json
import platform
import resource
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / "scripts"))
sys.path.insert(0, str(source / "src"))
import fit_fusion_binary_discrete as authenticated  # noqa: E402

from w1a1_eagle.block_fusion import (  # noqa: E402
    FusionFitConfig,
    diagnostics,
    fit_fusion,
    quantize,
    rms_norm_reference,
)
from w1a1_eagle.block_fusion import project as projection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Bounded saved authenticated EAGLE TRAIN A8/A1 fusion calibration"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--source-weights", type=Path, required=True)
    parser.add_argument("--base-gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    dataset = args.data
    receipt_pin = args.receipt_sha256
    config = authenticated.load_config(source / "configs/fusion_binary_discrete_a8.json")
    x, weight, mask, manifest, ancestry = authenticated.load_operands(
        dataset / "fusion-manifest.json",
        config,
        provenance_receipt_path=dataset / "provenance-receipt.json",
        provenance_receipt_sha256=receipt_pin,
        source_weights_path=args.source_weights,
        base_gguf_path=args.base_gguf,
    )
    with safe_open(str(args.source_weights), framework="pt", device="cpu") as f:
        gamma = f.get_tensor("midlayer.hidden_norm.weight").float().numpy()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    fit_y = (x[mask] @ weight.T).astype(np.float32)
    reports = {}
    for bits in (8, 1):
        cfg = FusionFitConfig(bits, True, 0, 60, reference_kind="eagle_fixed_reference_0.5")
        candidate = fit_fusion(x[mask], fit_y, weight, cfg)
        path = output / f"fusion-a{bits}.npz"
        np.savez(path, **{"fc.latent": candidate["latent"], "fc.scale": candidate["scale"]})
        control = output / f"control-a{bits}.npz"
        np.savez(
            control,
            **{"fc.latent": candidate["control_latent"], "fc.scale": candidate["control_scale"]},
        )
        # Validation starts after immutable saved initializers, with no updates.
        validation_y = (x[~mask] @ weight.T).astype(np.float32)
        codes, beta = quantize(x[~mask], bits)
        prediction = projection(codes, beta, candidate["hard_signs"], candidate["scale"])
        baseline = projection(
            codes, beta, candidate["control_hard_signs"], candidate["control_scale"]
        )
        reports[str(bits)] = {
            "config": asdict(cfg),
            "fit": candidate["report"],
            "artifact": str(path),
            "artifact_sha256": authenticated.sha256(path),
            "control": str(control),
            "control_sha256": authenticated.sha256(control),
            "validation_raw_candidate": diagnostics(prediction, validation_y),
            "validation_raw_control": diagnostics(baseline, validation_y),
            "validation_post_norm_candidate": diagnostics(
                rms_norm_reference(prediction, gamma, 1e-6),
                rms_norm_reference(validation_y, gamma, 1e-6),
            ),
            "validation_post_norm_control": diagnostics(
                rms_norm_reference(baseline, gamma, 1e-6),
                rms_norm_reference(validation_y, gamma, 1e-6),
            ),
        }
    report = {
        "schema": "saved_eagle_fusion_preparation_v1",
        "source_commit": args.source_commit,
        "source_module_sha256": authenticated.sha256(source / "src/w1a1_eagle/block_fusion.py"),
        "receipt_sha256": receipt_pin,
        "ancestry": ancestry,
        "rows": manifest["rows"],
        "calibration_hardware": platform.platform() + " / " + platform.machine() + " CPU",
        "producer_hardware": "historical authenticated RTX5080 target capture; no new GPU action",
        "fit_rows": int(mask.sum()),
        "validation_rows": int((~mask).sum()),
        "validation_limitation": (
            "historical validation has prose/reasoning only; no code validation"
        ),
        "target_precision": "frozen F16; original BF16 FC promoted F32 CPU teacher",
        "normalization": (
            "post-fusion midlayer.hidden_norm RMS F32 diagnostic epsilon1e-6; "
            "native reduction gate pending"
        ),
        "profiles": reports,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "status": (
            "actual TRAIN calibration only; native operator/quality/throughput "
            "and SM120 admission pending"
        ),
        "command_script_sha256": authenticated.sha256(Path(__file__)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    "sha256": v["artifact_sha256"],
                    "rescued": len(v["fit"]["events"]),
                    "raw_validation": v["validation_raw_candidate"]["relative_squared_error"],
                    "control_validation": v["validation_raw_control"]["relative_squared_error"],
                }
                for k, v in reports.items()
            }
        )
    )
    print("peak_rss_bytes", report["peak_rss_bytes"])


if __name__ == "__main__":
    main()
