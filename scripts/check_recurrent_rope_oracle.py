#!/usr/bin/env python3
"""Compare the sealed reasoning K RoPE graph with a pinned ggml CPU replay."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from audit_recurrent_draft_cache import audit as audit_cache  # noqa: E402
from check_recurrent_ffn_from_graph import _cpu_model, _metrics  # noqa: E402
from check_recurrent_ffn_stage_parity import verify_baseline_identity, verify_capture  # noqa: E402
from check_recurrent_native_attention_operator import (  # noqa: E402
    _group_decoder_taps,
    reconstruct_slots,
)
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402


def audit(
    capture: Path,
    boundary_report: Path,
    prior_ablation: Path,
    helper: Path,
    cpu_library: Path,
    rope_source: Path,
) -> dict:
    manifest, seal = verify_capture(capture, require_stages=False)
    verify_baseline_identity(seal)
    boundary = json.loads(boundary_report.read_text())
    ablation = json.loads(prior_ablation.read_text())
    if (
        boundary.get("schema") != "recurrent_reasoning_cache_projection_boundary_cpu_v1"
        or boundary["source_sha256"]["capture_manifest"] != seal["manifest"]
        or len(boundary["all_context_ordered_fc"]) != 46
        or any(
            row["ordered_fc_Kcur-0_f32_exact"] != 1024 for row in boundary["all_context_ordered_fc"]
        )
        or ablation["source_sha256"]["capture_manifest"] != seal["manifest"]
        or ablation["source_sha256"]["ggml_cpu_library"] != sha256(cpu_library)
        or ablation["source_sha256"]["ggml_cpu_ops_source"] != sha256(rope_source)
    ):
        raise ValueError("capture, ordered raw K or pinned CPU library identity differs")
    cache_audit = audit_cache(capture)
    events = read_jsonl(capture / "heads.draft_cache.jsonl")
    executions = [row for row in events if row.get("event") == "execution"]
    rows = [row for row in events if row.get("event") == "row"]
    native_keys, _, _, ledger = reconstruct_slots(
        executions,
        rows,
        np.memmap(capture / "heads.draft_cache.f16", dtype="<u2", mode="r"),
        np.memmap(capture / "heads.draft_cache.mask", dtype="<u2", mode="r"),
        2,
    )
    if ledger["selected_rows"][0]["position"] != 46:
        raise ValueError("first seed does not follow 46 context positions")
    records, values, _ = _read_graph(
        capture / "heads.draft_graph.jsonl", capture / "heads.draft_graph.f32"
    )
    group = _group_decoder_taps(records)[1]
    if group["Kcur-0"]["n_tokens"] != 46 or group["Kcur_rope-0"]["n_tokens"] != 46:
        raise ValueError("context raw/rotated K graph has wrong token count")
    raw = np.stack([_column(group["Kcur-0"], values, i) for i in range(46)])
    expected = np.stack([_column(group["Kcur_rope-0"], values, i) for i in range(46)])
    if raw.shape != (46, 1024) or expected.shape != raw.shape or raw.dtype != np.float32:
        raise ValueError("native K graph has wrong F32 geometry")
    with tempfile.TemporaryDirectory(prefix="recurrent-rope-oracle-") as directory:
        operand_dir = Path(directory)
        raw.astype("<f4").tofile(operand_dir / "key_raw.f32")
        np.arange(46, dtype="<i4").tofile(operand_dir / "positions.i32")
        subprocess.run([str(helper.resolve()), str(operand_dir), "46"], check=True)
        output_path = operand_dir / "key_rope.f32"
        if output_path.stat().st_size != expected.nbytes:
            raise ValueError("ggml RoPE helper returned a truncated F32 output")
        actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
    aggregate = _metrics(actual, expected)
    actual_bits = actual.astype("<f2").view("<u2")
    stored = native_keys[:46]
    exact_stored = int(np.count_nonzero(actual_bits == stored))
    per_position = [
        {
            "position": i,
            "f32_exact": _metrics(actual[i], expected[i])["exact_elements"],
            "stored_f16_exact": int(np.count_nonzero(actual_bits[i] == stored[i])),
        }
        for i in range(46)
    ]
    return {
        "schema": "recurrent_reasoning_ggml_rope_oracle_cpu_v1",
        "scope": "sealed first-round reasoning context; same native raw K at 46 positions",
        "hardware": {"machine": platform.machine(), "cpu_model": _cpu_model()},
        "native_revision": manifest["native_revision"],
        "ggml_rope": {
            "mode": "normal",
            "n_dims": 128,
            "freq_base": 1000000,
            "freq_scale": 1.0,
            "n_ctx_orig": 40960,
            "ext_factor": 0.0,
            "attn_factor": 1.0,
        },
        "context_f32": aggregate,
        "stored_key_f16_elements": stored.size,
        "stored_key_f16_exact": exact_stored,
        "per_position": per_position,
        "cache_audit_status": cache_audit["status"],
        "source_sha256": {
            "capture_manifest": seal["manifest"],
            "cache_boundary_report": sha256(boundary_report),
            "prior_ablation": sha256(prior_ablation),
            "helper_binary": sha256(helper),
            "helper_source": sha256(Path(__file__).with_name("native_recurrent_rope.cpp")),
            "ggml_cpu_library": sha256(cpu_library),
            "ggml_cpu_rope_source": sha256(rope_source),
            "probe": sha256(Path(__file__)),
        },
        "software": {"numpy": np.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--cache-boundary-report", type=Path, required=True)
    parser.add_argument("--prior-ablation", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--ggml-cpu-library", type=Path, required=True)
    parser.add_argument("--ggml-cpu-rope-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    result = audit(
        args.capture_dir,
        args.cache_boundary_report,
        args.prior_ablation,
        args.helper,
        args.ggml_cpu_library,
        args.ggml_cpu_rope_source,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "f32_exact": result["context_f32"]["exact_elements"],
                "f32_elements": result["context_f32"]["elements"],
                "stored_key_f16_exact": result["stored_key_f16_exact"],
                "stored_key_f16_elements": result["stored_key_f16_elements"],
            }
        )
    )


if __name__ == "__main__":
    main()
