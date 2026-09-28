#!/usr/bin/env python3
"""Audit deferred post-RoPE Q against safe K/output and earlier intrusive Q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from check_target_block0_capture import (
    HIDDEN,
    LADDER_MANIFEST_SHA256,
    PROMPT_ID,
    TOKENS,
    _index,
    _tensor,
    sha256,
)
from check_target_block0_hf import _metrics

Q_WIDTH = 4096


def _one(root: Path, name: str, mode: str) -> np.ndarray:
    entries, hashes = _index(root / "block0", mode=mode)
    report = json.loads((root / "comparison.json").read_text())
    if (
        report.get("schema") != "target_block0_native_cuda_capture_v1"
        or report.get("capture_mode") != mode
        or report.get("status") != "same_native_block_output"
        or report.get("prompt_id") != PROMPT_ID
        or report["block_output"]["exact_elements"] != TOKENS * HIDDEN
        or report["source_sha256"]["ladder_manifest"] != LADDER_MANIFEST_SHA256
        or any(report["source_sha256"].get(key) != value for key, value in hashes.items())
        or len(entries[name]) != 1
    ):
        raise ValueError(f"invalid output-preserving {mode} capture")
    return _tensor(root / "block0", entries[name][0]).reshape(TOKENS, -1)


def compare(
    run_dir: Path, deferred: Path, safe_k: Path, output_only: Path, intrusive: Path
) -> dict:
    if not (run_dir / "state.json").is_file():
        raise ValueError("start through scripts/remote_job.py")
    q = _one(deferred, "Qcur-0", "q_deferred_k_norm")
    k = _one(deferred, "Kcur_normed-0", "q_deferred_k_norm")
    k_reference = _one(safe_k, "Kcur_normed-0", "k_norm")
    block = _one(deferred, "l_out-0", "q_deferred_k_norm")
    block_reference = _one(output_only, "l_out-0", "output_only")
    if q.shape != (TOKENS, Q_WIDTH) or not np.isfinite(q).all():
        raise ValueError("deferred Q is nonfinite or has wrong geometry")
    if not np.array_equal(k.view("<u4"), k_reference.view("<u4")):
        raise ValueError("deferred capture changed the safe K values")
    if not np.array_equal(block.view("<u4"), block_reference.view("<u4")):
        raise ValueError("deferred capture changed block output")

    old_index = (intrusive / "block0/index.tsv").read_text().splitlines()
    old_q = [line.split("\t") for line in old_index if line.startswith("Qcur-0\t")]
    if (
        len(old_q) != 3
        or old_q[-1][1] != "2"
        or tuple(map(int, old_q[-1][4:8])) != (128, 32, 29, 1)
    ):
        raise ValueError("intrusive post-RoPE Q reference has wrong identity")
    old_file = intrusive / "block0" / old_q[-1][3]
    if old_file.stat().st_size != q.nbytes:
        raise ValueError("intrusive post-RoPE Q has wrong size")
    old_q_rows = np.fromfile(old_file, dtype="<f4").reshape(TOKENS, Q_WIDTH)
    if not np.isfinite(old_q_rows).all():
        raise ValueError("intrusive Q contains nonfinite values")
    report = {
        "schema": "target_q_deferred_audit_v1",
        "status": "safe_q_deferred_capture",
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "safe_k_exact": int(k.size),
        "safe_block_exact": int(block.size),
        "q_vs_intrusive_post_rope": _metrics(q, old_q_rows),
        "source_sha256": {
            "deferred_report": sha256(deferred / "comparison.json"),
            "safe_k_report": sha256(safe_k / "comparison.json"),
            "output_only_report": sha256(output_only / "comparison.json"),
            "deferred_q": sha256(deferred / "block0/Qcur-0_0.f32"),
            "intrusive_q": sha256(old_file),
            "probe": sha256(Path(__file__)),
        },
    }
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deferred-run", type=Path, required=True)
    parser.add_argument("--safe-k-run", type=Path, required=True)
    parser.add_argument("--output-only-run", type=Path, required=True)
    parser.add_argument("--intrusive-run", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        Path(__file__).resolve().parents[1] / "runs" / args.run_id,
        args.deferred_run,
        args.safe_k_run,
        args.output_only_run,
        args.intrusive_run,
    )
    print(json.dumps({"status": result["status"], "metrics": result["q_vs_intrusive_post_rope"]}))


if __name__ == "__main__":
    main()
