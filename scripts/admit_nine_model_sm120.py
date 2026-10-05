#!/usr/bin/env python3
"""Run the frozen fresh SM120 gate on the already-authorized campaign host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from w1a1_eagle.nine_model_admission import Admission, validate_plan  # noqa: E402
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    LinuxResources,
    SubprocessRunner,
    require,
    sha256,
    stop_signals,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run-dir", "receipt", "gpu-control"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "bundle-sha256", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)

    def authorization():
        control = json.loads(args.gpu_control.read_text())
        if control.get("rtx5080", {}).get("pause_requested") is not False:
            raise InterruptedError("RTX5080 paused or control absent")

    authorization()
    require(sha256(args.plan) == args.plan_sha256, "frozen admission plan differs")
    require(
        len(args.bundle_sha256) == 64 and all(c in "0123456789abcdef" for c in args.bundle_sha256),
        "invalid bundle SHA256",
    )
    plan, files = validate_plan(args.plan)

    resources = LinuxResources(args.gpu_uuid)
    environment = dict(plan.get("environment", {}), CUDA_VISIBLE_DEVICES=args.gpu_uuid)
    runner = SubprocessRunner(ROOT, environment, authorization=authorization)
    with stop_signals():
        result = Admission(
            plan, files, runner, resources, args.bundle_sha256, args.run_dir
        ).execute(args.receipt)
    print(
        json.dumps(
            {"status": result["status"], "receipt": str(args.receipt), "optimizer_updates": 0}
        )
    )


if __name__ == "__main__":
    main()
