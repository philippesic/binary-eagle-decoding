#!/usr/bin/env python3
"""Freeze authentic independent endpoints; inspect missing inputs without execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from w1a1_eagle.nine_model_endpoint_collection import (  # noqa: E402
    SCHEMA,
    inspection,
    missing_inputs,
    validate_collection,
)
from w1a1_eagle.nine_model_pipeline import atomic_json, require  # noqa: E402


def prepare(value, output=None, *, inspect_draft=False):
    require(
        value.get("schema") == "nine_model_endpoint_collection_inputs_v1", "input schema differs"
    )
    require(
        set(value)
        <= {
            "schema",
            "artifact_kind",
            "campaign_complete",
            *("candidates", "evaluation_source", "controls", "hardware_stage"),
        },
        "unknown collection input fields",
    )
    require(
        value.get("artifact_kind", "production") == "production"
        and value.get("campaign_complete", False) is False,
        "fixture/readiness claim cannot create production collection",
    )
    missing = missing_inputs(value)
    if inspect_draft:
        return inspection(missing=missing or ["strict validation not requested"])
    require(not missing, "collection inputs PENDING: " + "; ".join(missing))
    collection = dict(value, schema=SCHEMA, artifact_kind="production", campaign_complete=False)
    context = validate_collection(collection)
    require(output is not None and not output.exists(), "new immutable collection output required")
    atomic_json(output, collection)
    return inspection(context)


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
