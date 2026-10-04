#!/usr/bin/env python3
"""Bounded CPU-only final endpoint join for the existing A8 comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource

import torch


ROOT = Path(
    "/home/philip/binary-eagle-decoding/checkouts/a8-qat-run-583480c7/runs/"
    "qat-a8-comparison-20261003-01/arms"
)
ARMS = ("candidate", "reference")


def read(path: Path):
    return json.loads(path.read_text())


def audit_arm(arm: str) -> dict:
    run = ROOT / arm
    status = read(run / "status.json")
    budget = read(run / "budget-used.json")
    latest = read(run / "latest.json")
    request = read(run / "development-request.json")
    result = read(run / "development-result.json")
    checkpoint_path = Path(latest["path"])
    manifest_path = checkpoint_path.parent / "manifest.json"
    manifest = read(manifest_path)

    assert status["status"] == "completed"
    assert status["final_training_complete"] is True
    assert budget["training_seconds"] == budget["max_seconds"] == 7200.0
    assert budget["active_attempt"] is None
    assert request["final_training_complete"] is True
    assert request["training_elapsed_seconds"] == 7200.0
    assert latest == request["checkpoint"] == result["checkpoint"]
    assert result["completed"] is True
    assert manifest["sha256"] == latest["sha256"]

    # The original checkpoint is loaded to CPU only. Never print or serialize tensors.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a8 = payload["metrics"]["A8"]
    counts = {
        "checkpoint_step": payload["step"],
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_cursor": payload["cursor"],
        "checkpoint_presented_supervised_tokens": payload["tokens"],
        "checkpoint_unique_prompts": len(payload["unique_prompts"]),
        "checkpoint_unique_supervised_rows": len(payload["unique_rows"]),
        "checkpoint_A8_exact_cumulative_sign_flips": a8["cumulative_sign_flips"],
        "status_step": status["step"],
        "status_cursor": status["cursor"],
        "status_training_elapsed_seconds": status["training_elapsed_seconds"],
        "status_unique_prompts": status["unique_prompts"],
        "status_unique_supervised_rows": status["unique_supervised_rows"],
        "status_presented_supervised_tokens": status["presented_supervised_tokens"],
        "status_A8_exact_cumulative_sign_flips": status["models"]["A8"]["cumulative_sign_flips"],
    }
    assert counts["checkpoint_step"] == counts["status_step"]
    assert counts["checkpoint_cursor"] == counts["status_cursor"]
    assert counts["checkpoint_presented_supervised_tokens"] == counts["status_presented_supervised_tokens"]
    assert counts["checkpoint_unique_prompts"] == counts["status_unique_prompts"]
    assert counts["checkpoint_unique_supervised_rows"] == counts["status_unique_supervised_rows"]
    assert counts["checkpoint_A8_exact_cumulative_sign_flips"] == counts["status_A8_exact_cumulative_sign_flips"]

    recipe_path = run / "recipe-audit.json"
    recipe = read(recipe_path)
    telemetry = recipe["telemetry"]
    saved_signs = payload["lanes"]["A8"]["recipe_telemetry"]["signs"]
    outer_recipe_step = recipe["step"]
    sample_step = saved_signs["last_step"]
    telemetry_summary = {
        "path": str(recipe_path),
        "sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "outer_recipe_step": outer_recipe_step,
        "sample_observation_step_from_saved_sign_state": sample_step,
        "checkpoint_step": status["step"],
        "sample_gap_to_final_checkpoint_steps": status["step"] - sample_step,
        "saved_sign_state_last_step": saved_signs["last_step"],
        "sample_step_joins_saved_sign_state": saved_signs["last_step"] == sample_step,
        "saved_observations": saved_signs["observations"],
        "telemetry_observations_match_saved_state": telemetry["diagnostic_observations"] == saved_signs["observations"],
        "sampled_cumulative_flips_match_saved_state": telemetry["cumulative_sign_flips"] == saved_signs["cumulative_flips"],
        "sampled_cumulative_flip_backs_match_saved_state": telemetry["cumulative_flip_backs"] == saved_signs["cumulative_flip_backs"],
        "observation_gap_steps": telemetry["observation_gap_steps"],
        "diagnostic_observations": telemetry["diagnostic_observations"],
        "sampled_sign_flips": telemetry["sign_flips"],
        "sampled_flip_backs": telemetry["flip_backs"],
        "sampled_net_sign_disagreement": telemetry["net_sign_disagreement"],
        "sampled_unique_flipped_signs": telemetry["unique_flipped_signs"],
        "sampled_sustained_disagreement": telemetry["sustained_disagreement"],
        "sampled_cumulative_sign_flips": telemetry["cumulative_sign_flips"],
        "sampled_cumulative_flip_backs": telemetry["cumulative_flip_backs"],
        "near_zero_threshold": telemetry["near_zero_threshold"],
        "near_zero_fraction": telemetry["near_zero_fraction"],
        "minimum_latent_distance_to_zero": telemetry["minimum_latent_distance_to_zero"],
        "mean_latent_magnitude": telemetry["mean_latent_magnitude"],
        "family_movement_l1_since_initialization": telemetry["family_movement_l1_since_initialization"],
        "exact_status_cumulative_sign_flips": counts["status_A8_exact_cumulative_sign_flips"],
        "sampled_counts_not_native_acceptance": telemetry["sampled_counts_not_native_acceptance"],
    }
    assert telemetry_summary["sample_step_joins_saved_sign_state"]
    assert telemetry_summary["telemetry_observations_match_saved_state"]
    assert telemetry_summary["sampled_cumulative_flips_match_saved_state"]
    assert telemetry_summary["sampled_cumulative_flip_backs_match_saved_state"]
    del payload

    return {
        "arm": arm,
        "run_dir": str(run),
        "status_path": str(run / "status.json"),
        "budget_path": str(run / "budget-used.json"),
        "request_path": str(run / "development-request.json"),
        "result_path": str(run / "development-result.json"),
        "checkpoint_manifest_path": str(manifest_path),
        "checkpoint_sha256": latest["sha256"],
        "report_path": result["report_path"],
        "report_sha256": result["report_sha256"],
        "budget_training_seconds": budget["training_seconds"],
        "budget_max_seconds": budget["max_seconds"],
        "budget_active_attempt": budget["active_attempt"],
        "final_training_complete": request["final_training_complete"],
        "counts_join": counts,
        "counts_join_passed": True,
        "recipe_telemetry": telemetry_summary,
    }


print(
    json.dumps(
        {
            "schema": "a8_final_endpoint_cpu_audit_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "checkpoint_load_map_location": "cpu",
            "checkpoint_load_weights_only": False,
            "tensor_values_printed": False,
            "arms": [audit_arm(arm) for arm in ARMS],
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        separators=(",", ":"),
        allow_nan=False,
    )
)
