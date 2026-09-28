#!/usr/bin/env python3
"""Hash the pinned frozen GGUF operands used by the CPU binary drafter step."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.frozen_operands import (  # noqa: E402
    DRAFT_D_SHA256,
    TARGET_F16_SHA256,
    FrozenOperands,
)


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft-d", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    operands = FrozenOperands(args.target, args.draft_d)
    report = {
        "schema": "recurrent_frozen_operands_cpu_v1",
        "execution_device": "cpu",
        "target_gguf_sha256": TARGET_F16_SHA256,
        "draft_d_gguf_sha256": DRAFT_D_SHA256,
        "vocab_size": operands.vocab_size,
        "hidden_size": operands.hidden_size,
        "embedding_dtype": "float16",
        "sampled_embedding_row_sha256": {
            str(token): tensor_hash(operands(token)) for token in (0, operands.vocab_size - 1)
        },
        "norm_dtype": "float32",
        "norm_sha256": {name: tensor_hash(value) for name, value in operands.norm_arrays.items()},
        "target_forward_executed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sampled_rows": 2, "norms": len(report["norm_sha256"])}))


if __name__ == "__main__":
    main()
