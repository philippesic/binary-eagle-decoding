#!/usr/bin/env python3
"""Produce a new validated experiment config; never load models or start CUDA."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from w1a1_eagle.continuous_qat import ContinuousConfig  # noqa: E402


def prepare(base, profiles, name, native_commit):
    if profiles.get("schema") != "qat_optimization_profiles_v1":
        raise ValueError("unsupported optimization profile inventory")
    if base.get("schema") != "continuous_w1ax_experiment_v1":
        raise ValueError("unsupported base experiment")
    if name not in profiles["profiles"]:
        raise ValueError("unknown optimization profile")
    if not isinstance(native_commit, str) or re.fullmatch("[0-9a-f]{40}", native_commit) is None:
        raise ValueError("published native revision must be a full lowercase commit")
    result = copy.deepcopy(base)
    training = dict(result["training"])
    training.update(profiles["profiles"][name])
    training["seeds"] = tuple(training["seeds"])
    training["optimization_readiness"] = None
    config = ContinuousConfig(**training)
    result["training"] = asdict(config)
    result["native"] = {"expected_commit": native_commit}
    result["optimization_profile"] = name
    result["preparation_note"] = (
        "New source/config experiment; no exact resume of the frozen preparation run. "
        "Run zero-update native and CUDA readiness before manual optimizer launch. "
        "No real-data optimizer updates are authorized by producing this file."
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/continuous_w1ax.json")
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "configs/qat_optimization_profiles.json"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--native-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(
        json.loads(args.base_config.read_text()),
        json.loads(args.profiles.read_text()),
        args.profile,
        args.native_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "configuration_valid": True,
                "profile": args.profile,
                "output": str(args.output.resolve()),
                "optimizer_updates": 0,
                "cuda_started": False,
                "gpu_ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
