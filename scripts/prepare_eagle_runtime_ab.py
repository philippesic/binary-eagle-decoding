#!/usr/bin/env python3
"""Prepare matched runtime A/B configs without starting a server or GPU job."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SELECTORS = {
    "compact_logits": "GGML_EAGLE_COMPACT_LOGITS",
    "kv_only": "GGML_EAGLE_KV_ONLY_CATCHUP",
    "shared_pack": "GGML_EAGLE_SHARED_PACK",
    "device_state": "GGML_EAGLE_DEVICE_STATE",
    "warp_reduce": "GGML_W1AX_WARP_REDUCE",
}


def prepare(source: dict, name: str) -> tuple[dict, dict]:
    if name not in SELECTORS:
        raise ValueError("unknown runtime selector")
    if source.get("schema_version") != 1:
        raise ValueError("source must use benchmark schema 1")
    variants = source.get("variants", {})
    if source.get("q4_variant", "q4_0") not in variants:
        raise ValueError("source must include the Q4_0 primary control")
    if name in ("shared_pack", "warp_reduce") and not any(
        spec.get("env", {}).get("GGML_W1AX_ACT_BITS", "") in ("1", "4", "8")
        for spec in variants.values()
        if spec.get("draft")
    ):
        raise ValueError("packing/reduction needs an explicit A1/A4/A8 packed variant")
    baseline = copy.deepcopy(source)
    env = baseline.setdefault("graph_env", {})
    for selector in SELECTORS.values():
        env[selector] = "0"
        for spec in baseline["variants"].values():
            spec.get("env", {}).pop(selector, None)
    if name == "compact_logits":
        baseline["draft_backend_sampling"] = False
    enabled = copy.deepcopy(baseline)
    enabled["graph_env"][SELECTORS[name]] = "1"
    return baseline, enabled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selector", choices=sorted(SELECTORS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    raw = args.source.read_bytes()
    configs = prepare(json.loads(raw), args.selector)
    args.output.mkdir(parents=True)
    hashes = {}
    for mode, config in zip(("off", "on"), configs, strict=True):
        body = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
        (args.output / f"{mode}.json").write_bytes(body)
        hashes[mode] = hashlib.sha256(body).hexdigest()
    manifest = {
        "schema": "eagle_runtime_ab_preparation_v1",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "selector": SELECTORS[args.selector],
        "config_sha256": hashes,
        "draft_backend_sampling": configs[0].get("draft_backend_sampling", True),
        "gpu_jobs_started": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
