"""Emit compact tracked evidence and detailed ignored CPU results."""

import hashlib
import json
import platform
from pathlib import Path

import torch

from research.parallel20261002.activation_reuse.reference.test_reuse import (
    graph_gate,
    invalidation_gate,
    update_gate,
)
from research.parallel20261002.auxiliary_vjp.reference.audit import check_control

ROOT = Path(__file__).resolve().parents[4]


def run():
    check_control()
    torch.set_num_threads(1)
    result = dict(
        device="CPU", hardware=platform.machine(), torch=torch.__version__,
        precision="F32 masters/arithmetic; F16 K/V/factors", atol=2e-6, rtol=5e-5,
        graph=[graph_gate(b) for b in (1, 4, 8)], update=[update_gate(b) for b in (1, 4, 8)],
        invalidation=invalidation_gate(),
    )
    paths = ["src/w1a1_eagle/" + n + ".py" for n in (
        "native_step", "recurrent_qat", "learned_activation", "affine_binary",
        "fusion_correction", "recurrent_provider",
    )] + [
        "research/parallel20261002/auxiliary_vjp/reference/audit.py",
        "research/parallel20261002/activation_reuse/reference/prototype.py",
    ]
    result["source_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
    }
    raw = ROOT / "runs/parallel20261002/activation-reuse/results.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("graph", "update")}
    summary.update(raw_path=str(raw.relative_to(ROOT)),
                   raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    summary["graph"] = [dict(
        bits=g["bits"], groups=g["groups"], baseline_calls=g["baseline_calls"],
        reuse_calls=g["reuse_calls"], exact_logits=True,
        exact_projection_outputs=g["exact_projection_outputs"],
        max_vjp_error=max(v["max_abs_error"] for k, v in g["vjps"].items()
                          if k not in ("logits", "loss")),
    ) for g in result["graph"]]
    summary["update"] = [dict(
        bits=u["bits"], clipped_norm=u["clipped_norm"],
        max_update_error=max(v["max_abs_error"] for v in u["updates"].values()),
        max_clipped_error=max(v["max_abs_error"] for v in u["clipped"].values()),
    ) for u in result["update"]]
    out = ROOT / "experiments/parallel20261002/activation_reuse/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("3 learned profiles, exact projection outputs/logits, 3 clipped updates passed")


if __name__ == "__main__":
    run()
