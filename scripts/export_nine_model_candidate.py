#!/usr/bin/env python3
"""Freeze a committed training export through the real family GGUF serializer."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    atomic_json,
    require,
    sha256,
    validate_bundle,
)


def export(args):
    require(sha256(args.bundle) == args.bundle_sha256, "bundle SHA256 differs")
    bundle, files = validate_bundle(args.bundle)
    train = json.loads(args.train_receipt.read_text())
    require(
        train.get("schema") == "nine_model_stage_receipt_v1"
        and train.get("stage") == args.candidate + "/train"
        and train.get("artifact_kind") == "production"
        and train.get("bundle_sha256") == args.bundle_sha256
        and train.get("committed") is True
        and train.get("status") == "PASS"
        and train.get("completion_reason") == "approved_budget_complete",
        "committed production training endpoint absent",
    )
    family, precision = args.candidate.split("_")
    record = train["exports"][args.candidate]
    checkpoint, manifest = files.check(record["checkpoint"]), files.check(record["manifest"])
    model_record = bundle["candidates"][args.candidate]["base_model"]
    base = files.check(model_record)
    require(record["base_gguf_sha256"] == model_record["sha256"], "base model ancestry differs")
    target = args.completion_output.parent / ("model-" + sha256(args.train_receipt)[:16])
    target.mkdir(parents=True, exist_ok=False)
    output = target / "candidate.gguf"
    serializer = importlib.import_module(
        "export_recurrent_binary" if family == "eagle" else "export_block_binary"
    )
    audit = serializer.export_model(base, checkpoint, manifest, output)
    atomic_json(target / "export-audit.json", audit)
    require(
        output.is_file() and audit.get("serialization_audit_passed") is True,
        "native serializer audit did not PASS",
    )
    require(
        audit.get("activation_bits") == int(precision[1:])
        and audit.get("output", {}).get("sha256") == sha256(output)
        and audit.get("projections"),
        "export arithmetic/packed inventory differs",
    )
    # This is serialization proof. Fresh native load/graph/backward admission and
    # actual evaluator loader/dispatch checks remain separate requirements.
    atomic_json(
        args.completion_output,
        {
            "schema": "nine_model_stage_receipt_v1",
            "stage": args.candidate + "/export",
            "bundle_sha256": args.bundle_sha256,
            "status": "PASS",
            "artifact_kind": "production",
            "serialization_audit_passed": True,
            "selected_weights_packed": True,
            "runtime_dense_fallback_checked": False,
            "model": {"path": str(output.resolve()), "sha256": sha256(output)},
            "training_receipt": {
                "path": str(args.train_receipt),
                "sha256": sha256(args.train_receipt),
            },
            "audit": {
                "path": str(target / "export-audit.json"),
                "sha256": sha256(target / "export-audit.json"),
            },
            "native_runtime_gate": "PENDING_fresh_load_graph_dispatch",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--train-receipt", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    export(parser.parse_args())
