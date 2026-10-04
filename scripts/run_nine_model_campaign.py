#!/usr/bin/env python3
"""Inspect or run a frozen campaign on the already-authorized local RTX5080 host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    Campaign,
    LinuxResources,
    SubprocessRunner,
    require,
    require_available,
    sha256,
    validate_bundle,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--availability", type=Path)
    parser.add_argument(
        "--gpu-control",
        type=Path,
        default=Path.home() / ".config/binary-eagle-decoding/gpu-control.json",
    )
    args = parser.parse_args(argv)
    require(sha256(args.bundle) == args.bundle_sha256, "frozen bundle SHA256 differs")
    bundle, files = validate_bundle(args.bundle)
    if not args.start:
        print(
            json.dumps(
                {
                    "schema": "nine_model_plan_v1",
                    "portable_bundle_valid": True,
                    "gpu_queried": False,
                    "training_started": False,
                    "fresh_sm120_gates": sorted(bundle["fresh_gates"]),
                }
            )
        )
        return
    require(args.run_dir and args.availability, "start needs run-dir and availability lease")
    lease = require_available(args.availability, args.bundle_sha256)

    def authorization():
        control = json.loads(args.gpu_control.read_text())
        if control.get("rtx5080", {}).get("pause_requested") is not False:
            raise InterruptedError("RTX5080 paused or control absent")
        # Lease freshness is checked once before long QAT; owner/pause identity is
        # checked at every boundary. A five-minute lease is not a five-minute job cap.
        current = json.loads(args.availability.read_text())
        require(current == lease, "availability ownership changed during campaign")

    authorization()
    resources = LinuxResources(lease["gpu_uuid"])
    runner = SubprocessRunner(ROOT, bundle.get("environment"), authorization=authorization)
    gpu_lock = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with gpu_lock.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = Campaign(
            bundle,
            args.bundle_sha256,
            files,
            ROOT,
            args.run_dir.resolve(),
            runner,
            resources,
            authorization=authorization,
        ).execute(resume=args.resume)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
