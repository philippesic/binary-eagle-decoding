#!/usr/bin/env python3
"""Aggregate complete paired native nine-model measurements."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.nine_model_pipeline import atomic_json  # noqa: E402
from w1a1_eagle.nine_model_report import aggregate  # noqa: E402

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("measurements", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    atomic_json(args.output, aggregate(json.loads(args.measurements.read_text())))
