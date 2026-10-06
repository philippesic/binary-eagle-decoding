#!/usr/bin/env python3
"""Freeze one production lane without promoting the unfinished nine-model campaign."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import prepare_nine_model_bundle as builder  # noqa: E402

from w1a1_eagle.nine_model_admission import validate_plan  # noqa: E402
from w1a1_eagle.nine_model_pipeline import CANDIDATES, Files, atomic_json, require  # noqa: E402


def build_lane(descriptor, output):
    require(descriptor.get("schema") == "nine_model_lane_inputs_v1", "staged descriptor required")
    (name,) = builder.selected_candidates(descriptor)
    selected = descriptor["candidates"][name]
    files = Files()
    budget = json.loads(files.check(descriptor["budget"]).read_text())
    builder.selected_budget(budget, (name,), staged=True)
    require(
        budget.get("schema") == "nine_model_selected_budget_v1", "operational budget schema differs"
    )
    ledger = json.loads(files.check(descriptor["qa_ledger"]).read_text())
    pending = builder.ledger_pending(ledger, (name,)) + builder.source_pending(ledger)
    require(not pending, "selected lane portable prelaunch PENDING: " + "; ".join(pending))
    config = files.check(selected["config"])
    trainer = importlib.import_module("train_nine_model_qat")
    spec = trainer.load_spec(config)
    require(spec["candidate"] == name, "selected config candidate differs")
    require(
        selected["profile"] in {"fixed_reference", "direct_a1"}, "staged lane uses direct precision"
    )
    require(spec.get("precision_stage", "direct") == "direct", "warm stage not selected")
    initializer = spec.get("initialization")
    require(initializer is not None, "calibrated fusion initializer missing")
    require(
        {key: initializer[key] for key in ("path", "sha256")} == selected["fusion_calibration"],
        "selected fusion initializer differs",
    )
    files.check(selected["fusion_calibration"])
    limits = budget["candidates"][name]
    require(limits.get("wall_seconds", 0) > 0, "bounded training wall cap required")
    if spec["family"] == "eagle":
        _, continuous = importlib.import_module("train_continuous_w1ax").load_config(
            files.check(spec["eagle_config"])
        )
        actual_limits = {
            key: getattr(continuous, key)
            for key in ("max_steps", "max_tokens", "max_seconds", "max_epochs")
        }
        require(
            continuous.development_lifecycle == "standalone", "fresh evaluation lifecycle required"
        )
    else:
        actual_limits = spec["limits"]
    require(
        actual_limits == limits["training_limits"], "operational limits differ from trainer config"
    )
    require(
        actual_limits.get("max_seconds") is not None
        and limits["wall_seconds"] > actual_limits["max_seconds"],
        "outer wall cap must allow overhead beyond cumulative trainer time",
    )
    plan_locator = descriptor["inputs"]["admission_plan"]
    plan, _ = validate_plan(files.check(plan_locator))
    require(
        plan["schema"] == "nine_model_lane_sm120_plan_v1" and set(plan["candidates"]) == {name},
        "selected admission scope differs",
    )
    lane = plan["candidates"][name]
    for role, chosen in (
        ("config", "config"),
        ("model", "initial_model"),
        ("export", "initial_export_audit"),
    ):
        require(lane[role] == selected[chosen], "admission/selected artifact differs: " + role)
    require(
        lane["source_bindings"]
        == builder.admission_source_bindings(selected, spec, files, plan["target"]["sha256"]),
        "completed production data source differs",
    )
    source = dict(plan["source"])
    for path in (Path(__file__), ROOT / "scripts/run_nine_model_lane.py"):
        source[str(path.relative_to(ROOT))] = builder.pin(path)
    value = {
        "schema": "nine_model_training_lane_v1",
        "artifact_kind": "production",
        "candidate": name,
        "preparation_complete": True,
        "campaign_complete": False,
        "remaining_candidates": [other for other in CANDIDATES if other != name],
        "evaluation_status": "PENDING",
        "target_policy": {"immutable": True, "weights": "f16", "kv": "f16"},
        "config": selected["config"],
        "admission_plan": plan_locator,
        "qa_ledger": descriptor["qa_ledger"],
        "budget": descriptor["budget"],
        "authorization": budget["authorization"],
        "source": source,
        "gpu_uuid": descriptor["gpu_uuid"],
        "gpu_control_path": descriptor["gpu_control_path"],
        "environment": plan.get("environment", {}),
        "resource_policy": plan["resource_policy"],
        "training_wall_seconds": limits["wall_seconds"],
    }
    if descriptor.get("timed_evaluation_plan"):
        from evaluate_nine_model_timed_checkpoint import validate_plan as validate_timed_plan

        locator = descriptor["timed_evaluation_plan"]
        validate_timed_plan(locator, value, plan, files)
        value["timed_evaluation_plan"] = locator
        timed = json.loads(files.check(locator).read_text())
        require(
            all(
                key not in value["source"] or value["source"][key] == record
                for key, record in timed["source"].items()
            ),
            "timed plan cannot replace original source pins",
        )
        value["source"].update(timed["source"])
    require(not output.exists(), "preserve previous lane publication")
    atomic_json(output, value)
    return {
        "status": "PASS",
        "lane": builder.pin(output),
        "gpu_queried": False,
        "optimizer_updates": 0,
        "campaign_complete": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_lane(json.loads(args.inputs.read_text()), args.output)))


if __name__ == "__main__":
    main()
