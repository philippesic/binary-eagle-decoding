#!/usr/bin/env python3
"""Prepare association receipts from frozen final authority and opaque indexes only.

This executes no native inference and grants no runtime permission. Missing real
indexes/selection/authority remain PENDING; prompt payloads are never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from w1a1_eagle.nine_model_endpoint_collection import (  # noqa: E402
    CANDIDATE_FIELDS,
    endpoint_validators,
    final_admission,
    final_admissions,
    source_context,
)
from w1a1_eagle.nine_model_pipeline import CANDIDATES, Files, atomic_json, require  # noqa: E402


def prepare(value, output, *, inspect_draft=False, files=None, validators=None, source_loader=None):
    require(
        value.get("schema") == "nine_model_endpoint_collection_inputs_v1",
        "collection input schema differs",
    )
    missing = [key for key in ("evaluation_source", "candidates") if not value.get(key)]
    for name in CANDIDATES:
        selected = value.get("candidates", {}).get(name, {})
        missing.extend(
            name + "/" + key
            for key in sorted(CANDIDATE_FIELDS - {"heldout_admission"})
            if not selected.get(key)
        )
    if inspect_draft:
        return {
            "status": "PENDING",
            "missing": missing or ["strict final evidence not validated"],
            "execution_allowed": False,
            "prompt_payload_read": False,
            "gpu_queried": False,
        }
    require(not missing, "final evidence PENDING: " + "; ".join(missing))
    require(set(value["candidates"]) == set(CANDIDATES), "exact six origins required")
    files = files or Files()
    source = (source_loader or source_context)(value["evaluation_source"], files)
    protocol = json.loads(files.check(source["protocol"]).read_text())
    require(
        protocol.get("schema") == "nine_model_native_protocol_v1"
        and protocol.get("split") == "final",
        "explicit frozen final protocol required",
    )
    validate_endpoint, validate_export = validators or endpoint_validators()
    contexts, exports = {}, {}
    for name, selected in value["candidates"].items():
        contexts[name] = validate_endpoint(
            *(
                selected[key]
                for key in ("frozen_lane", "lane_state", "supervisor_state", "training_receipt")
            ),
            files=files,
        )
        require(
            contexts[name]["lane"]["candidate"] == name
            and contexts[name]["target"] == source["target"],
            "lane target/name differs",
        )
        exports[name] = validate_export(contexts[name], selected["export_receipt"], files=files)
    proof = final_admissions(value, source, contexts, exports, files, check_admissions=False)
    require(
        output is not None and not output.exists(), "new immutable admission directory required"
    )
    output.mkdir(parents=True)
    for name in CANDIDATES:
        atomic_json(
            output / (name + ".json"), final_admission(value, source, contexts, exports, name)
        )
    return {
        "status": "VALIDATED_FINAL_EVIDENCE_RUNTIME_PENDING",
        "prompt_count": proof["prompt_count"],
        "execution_allowed": False,
        "prompt_payload_read": False,
        "gpu_queried": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-draft", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                json.loads(args.inputs.read_text()), args.output, inspect_draft=args.inspect_draft
            ),
            sort_keys=True,
        )
    )
