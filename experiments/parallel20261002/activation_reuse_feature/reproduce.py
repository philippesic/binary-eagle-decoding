"""Small source-bound operation/storage receipt; CPU synthetic graph only."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

from test_parallel20261002_activation_reuse_feature import graph_census  # noqa: E402

from research.parallel20261002.auxiliary_vjp.reference.audit import (  # noqa: E402
    combined,
    compare,
)
from w1a1_eagle.native_step import NativeStepAdapter  # noqa: E402


def main():
    control = json.loads(
        Path(
            "/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json"
        ).read_text()
    )
    if (
        control["research_stop"]
        or control["reset_observed"]
        or control["last_reset_unix"] != 1791049896
        or 100 - control["last_weekly_used_percent"] <= 1
    ):
        raise SystemExit("research stopped by shared control")
    torch.set_num_threads(1)
    baseline = combined(4, "single_forward")
    optimized = NativeStepAdapter(combined(4, "single_forward").drafter, activation_reuse=True)
    expected, base_calls, base_storage = graph_census(baseline)
    actual, reuse_calls, reuse_storage = graph_census(optimized)
    errors = compare(actual, expected)
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
    sources = [
        "src/w1a1_eagle/" + name + ".py"
        for name in (
            "activation_reuse",
            "native_step",
            "learned_activation",
            "recurrent_qat",
            "affine_binary",
            "fusion_correction",
            "recurrent_provider",
        )
    ] + [
        "tests/test_parallel20261002_activation_reuse_feature.py",
        "research/parallel20261002/auxiliary_vjp/reference/audit.py",
    ]
    receipt = {
        "hardware": platform.platform(),
        "device": "CPU",
        "torch": torch.__version__,
        "precision": "F32 masters/arithmetic; F16 K/V/fusion factors",
        "profile": "synthetic A4; learned activation + affine all + raw FC rank-1 correction",
        "attached_draft_steps": 3,
        "exact_logits": True,
        "baseline_attached_quantizer_computations": dict(base_calls),
        "reused_attached_quantizer_computations": dict(reuse_calls),
        "baseline_unique_ste_saved_storages": base_storage,
        "reused_unique_ste_saved_storages": reuse_storage,
        "vjps_atol": 2e-6,
        "vjps_rtol": 5e-5,
        "max_vjp_abs_error": max(
            row["max_abs_error"] for name, row in errors.items() if name not in ("logits", "loss")
        ),
        "effective_state": optimized.activation_reuse_state(),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources
        },
        "native_throughput_measured": False,
        "gpu_work": False,
        "qat_config_bound": False,
        "live_optimization_adopted": False,
    }
    output = Path(__file__).with_name("source-proof.json")
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "device",
                    "torch",
                    "exact_logits",
                    "max_vjp_abs_error",
                    "baseline_attached_quantizer_computations",
                    "reused_attached_quantizer_computations",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
