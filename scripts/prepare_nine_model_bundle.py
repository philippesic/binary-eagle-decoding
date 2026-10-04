#!/usr/bin/env python3
"""Freeze resolved production artifacts; draft inspection exposes dependencies.

No CUDA, SSH, model load, corpus scan or optimizer update. Existing completed
source-bound data admissions are required; the builder never launches a capture.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    CELLS,
    FAMILIES,
    GATES,
    Files,
    atomic_json,
    require,
    sha256,
    validate_bundle,
)

PORTABLE = ("source", "data", "ownership", "hard_forward", "resume", "export", "lifecycle")
CRITICAL_SOURCE = (
    "src/w1a1_eagle/nine_model_pipeline.py",
    "src/w1a1_eagle/block_qat.py",
    "src/w1a1_eagle/block_training.py",
    "src/w1a1_eagle/block_data.py",
    "src/w1a1_eagle/continuous_qat.py",
    "scripts/train_nine_model_qat.py",
    "scripts/export_nine_model_candidate.py",
    "scripts/export_block_binary.py",
    "scripts/export_recurrent_binary.py",
    "scripts/evaluate_nine_model_native.py",
    "scripts/admit_nine_model_sm120.py",
)


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def ledger_pending(ledger):
    pending = []
    require(ledger.get("schema") == "nine_model_qa_ledger_v1", "independent QA schema differs")
    for profile in CELLS:
        record = ledger.get("profiles", {}).get(profile, {})
        if record.get("status") != "PASS":
            pending.append(profile + ": profile production dependencies PENDING")
        if profile.endswith("q4"):
            required = ("source", "export")
        else:
            required = PORTABLE
        for name in required:
            requirement = record.get("requirements", {}).get(name, {})
            if requirement.get("status") != "PASS":
                pending.append(profile + ": " + name + " PENDING")
                continue
            if name in {"data", "hard_forward", "export"} and not any(
                e.get("scope") not in {None, "cpu_synthetic"}
                for e in requirement.get("evidence", [])
            ):
                pending.append(profile + ": " + name + " has only synthetic/unscoped evidence")
    return pending


def source_pending(ledger):
    pending = []
    bindings = ledger.get("source_files", {})
    for name in sorted(set(CRITICAL_SOURCE) | set(bindings)):
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "QA source paths must remain repository relative",
        )
        path = ROOT / name
        if not path.is_file() or bindings.get(name) != sha256(path):
            pending.append("independent current source evidence PENDING: " + name)
    return pending


def inspect_descriptor(descriptor, files):
    require(
        descriptor.get("schema") == "nine_model_bundle_inputs_v1",
        "resolved descriptor schema differs",
    )
    pending = list(descriptor.get("pending_dependencies", []))
    for key in ("inputs", "controls", "candidates", "qa_ledger", "budget", "gpu_uuid"):
        if key not in descriptor:
            pending.append("missing " + key)
    if "candidates" in descriptor:
        for candidate in CANDIDATES:
            record = descriptor["candidates"].get(candidate, {})
            for key in (
                "config",
                "base_model",
                "initial_model",
                "initial_export_audit",
                "data_admission",
                "fusion_calibration",
                "deployment_coverage",
                "profile",
                "native_markers",
            ):
                if key not in record:
                    pending.append(candidate + ": missing " + key)
    if descriptor.get("qa_ledger"):
        ledger = json.loads(files.check(descriptor["qa_ledger"]).read_text())
        pending.extend(ledger_pending(ledger))
        pending.extend(source_pending(ledger))
    return sorted(set(pending))


def materialize_configs(descriptor, directory, *, source_validate=True):
    """Six actual source configs from selected artifact references and budget.

    This is source configuration preparation, never production data/readiness.
    No model or GPU is loaded. Unsupported recipe options fail source parsing.
    """
    files = Files()
    budget = json.loads(files.check(descriptor["budget"]).read_text())
    require(
        budget.get("human_selected") is True
        and set(budget.get("candidates", {})) == set(CANDIDATES),
        "human-selected six-candidate allocation required for source configs",
    )
    directory = Path(directory).resolve()
    require(not directory.exists(), "config publication already exists")
    directory.mkdir(parents=True)
    resolved = copy.deepcopy(descriptor)
    trainer = importlib.import_module("train_nine_model_qat") if source_validate else None
    configs = {}
    for name in CANDIDATES:
        selected = resolved["candidates"][name]
        family, precision = name.split("_")
        bits = 8 if precision == "a8" else 1
        initializer = selected["initialization"]
        init_bits = 8 if selected["profile"] == "a8_to_a1_reset" else bits
        require(
            initializer.get("activation_bits") == init_bits,
            "initializer arithmetic must match first QAT stage",
        )
        policy = initializer["latent_initialization"]["policy"]
        require(
            policy in {"preserve_reference_magnitudes", "unit_probe"},
            "unsupported explicit latent magnitude policy",
        )
        require(
            policy != "unit_probe" or selected.get("unit_probe_selected") is True,
            "unit latent policy is an off-default recipe probe",
        )
        spec = {
            "schema": "nine_model_qat_training_v1",
            "family": family,
            "candidate": name,
            "device": "cuda:0",
            "initialization": initializer,
            "resource_floors": selected.get("resource_floors", {}),
        }
        limits = budget["candidates"][name]["training_limits"]
        if family == "eagle":
            require(
                selected["profile"] != "a8_to_a1_reset",
                "EAGLE curriculum configuration source integration PENDING",
            )
            base_config = files.check(selected["eagle_config_template"])
            eagle = json.loads(base_config.read_text())
            eagle["training"].update(limits)
            eagle["training"].update(
                activation_bits=[bits],
                development_lifecycle="standalone",
                activation_quantization="fixed",
                initialization_sha256=initializer["sha256"],
                initialization_policy=policy,
            )
            lane_config = directory / (name + "-continuous.json")
            atomic_json(lane_config, eagle)
            if source_validate:
                importlib.import_module("train_continuous_w1ax").load_config(lane_config)
            spec.update(eagle_config=pin(lane_config), prepared=selected["prepared"])
        else:
            require(
                selected["deployment_coverage"]["profile"] in {"ffn15", "ffn15_fusion"},
                "canonical native block coverage required",
            )
            qat = dict(selected.get("qat_overrides", {}))
            qat.update(
                family=family,
                activation_bits=bits,
                profile=selected["deployment_coverage"]["profile"],
                latent_initialization=policy,
            )
            # Defaults come from the real source dataclass. Only explicitly
            # selected shape/objective/optimizer/probe overrides are supplied.
            spec.update(
                qat=qat,
                model=selected["base_model"],
                data=selected["data"],
                limits=limits,
                checkpoint_every=selected["checkpoint_every"],
                precision_stage="a8_to_a1" if selected["profile"] == "a8_to_a1_reset" else "direct",
            )
            if spec["precision_stage"] == "a8_to_a1":
                spec["a8_warmup_steps"] = selected["a8_warmup_steps"]
                max_steps = limits.get("max_steps")
                require(
                    max_steps is None or max_steps > selected["a8_warmup_steps"],
                    "A8-to-A1 budget must permit final A1 optimizer exposure",
                )
            if selected.get("teacher") is not None:
                spec["teacher"] = selected["teacher"]
        path = directory / (name + ".json")
        atomic_json(path, spec)
        if source_validate:
            trainer.load_spec(path)
        locator = pin(path)
        configs[name] = locator
        selected["config"] = locator
        selected["fusion_calibration"] = {key: initializer[key] for key in ("path", "sha256")}
    atomic_json(directory / "resolved-inputs.json", resolved)
    receipt = {
        "schema": "nine_model_source_configs_v1",
        "configs": configs,
        "source_validation": "PASS" if source_validate else "PENDING",
        "production_preparation_ready": False,
        "model_loaded": False,
        "gpu_queried": False,
        "optimizer_updates": 0,
    }
    atomic_json(directory / "source-configs.json", receipt)
    return receipt


def build(descriptor, output, *, inspect_draft=False):
    files = Files()
    pending = inspect_descriptor(descriptor, files)
    if inspect_draft:
        return {
            "schema": "nine_model_bundle_inspection_v1",
            "status": "PENDING" if pending else "PASS",
            "pending": pending,
            "model_loaded": False,
            "gpu_queried": False,
            "optimizer_updates": 0,
            "production_bundle_created": False,
        }
    require(not pending, "production preparation PENDING: " + "; ".join(pending))
    budget = json.loads(files.check(descriptor["budget"]).read_text())
    require(
        budget.get("schema") == "nine_model_selected_budget_v1"
        and budget.get("human_selected") is True
        and set(budget.get("candidates", {})) == set(CANDIDATES)
        and budget.get("evaluation_wall_seconds", 0) > 0,
        "human-selected six-candidate/evaluation allocation absent",
    )
    require(set(descriptor["controls"]) == set(FAMILIES), "three original controls required")
    source = {}
    for folder in (ROOT / "src/w1a1_eagle", ROOT / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            source[str(path.relative_to(ROOT))] = pin(path)
    for required in (
        "train_nine_model_qat.py",
        "export_nine_model_candidate.py",
        "evaluate_nine_model_native.py",
        "admit_nine_model_sm120.py",
    ):
        require((ROOT / "scripts" / required).is_file(), "missing actual producer " + required)
    trainer = importlib.import_module("train_nine_model_qat")
    candidates = {}
    for name in CANDIDATES:
        selected = descriptor["candidates"][name]
        config_path = files.check(selected["config"])
        spec = trainer.load_spec(config_path)  # The real source API, never a copied config parser.
        require(
            spec["candidate"] == name and spec["family"] == name.split("_")[0],
            "candidate/family config identity differs",
        )
        for key in ("base_model", "initial_model", "initial_export_audit", "data_admission"):
            files.check(selected[key])
        require(
            spec.get("initialization") is not None,
            "deployed-arithmetic calibrated fusion initializer missing",
        )
        initializer = spec["initialization"]
        calibration_record = selected["fusion_calibration"]
        files.check(calibration_record)
        require(
            {key: initializer[key] for key in ("path", "sha256")} == calibration_record,
            "training initializer differs from selected fused calibration artifact",
        )
        initial_audit = json.loads(files.check(selected["initial_export_audit"]).read_text())
        require(
            initial_audit.get("serialization_audit_passed") is True
            and initial_audit.get("output", {}).get("sha256")
            == selected["initial_model"]["sha256"],
            "initial candidate native export audit differs",
        )
        # Reuse actual completed admissions; do not redo multi-hour corpus scans.
        if spec["family"] != "eagle":
            data = spec["data"]
            files.check(data)
            module = importlib.import_module("w1a1_eagle.block_data")
            module.BlockDataset(
                data["path"],
                expected_sha256=data["sha256"],
                allow_synthetic=False,
                verify_artifacts=False,
                admission_path=selected["data_admission"]["path"],
                admission_sha256=selected["data_admission"]["sha256"],
            )
        else:
            ready = selected["data_admission"]
            evidence = json.loads(files.check(ready).read_text())
            require(
                evidence.get("schema") == "continuous_w1ax_preparation_ready_v1"
                and evidence.get("preparation_complete") is True
                and evidence.get("optimization_started") is False,
                "authenticated completed original EAGLE prepared data admission required",
            )
        precision = 8 if name.endswith("a8") else 1
        if spec["family"] != "eagle":
            require(
                spec["qat"]["activation_bits"] == precision,
                "candidate deployment arithmetic differs",
            )
        else:
            eagle_api = importlib.import_module("train_continuous_w1ax")
            _, eagle_config = eagle_api.load_config(files.check(spec["eagle_config"]))
            require(
                eagle_config.activation_bits == (precision,)
                and eagle_config.development_lifecycle == "standalone",
                "EAGLE candidate arithmetic/fresh-process lifecycle differs",
            )
        if selected["profile"] == "a8_to_a1_reset":
            require(
                spec["family"] != "eagle" and spec.get("precision_stage") == "a8_to_a1",
                "EAGLE transition launcher not admitted; block transition must be explicit",
            )
        limits = budget["candidates"][name]
        require(
            limits.get("wall_seconds", 0) > 0 and limits.get("export_wall_seconds", 0) > 0,
            "positive candidate train/export wall caps required",
        )
        if spec["family"] != "eagle":
            require(
                spec["limits"] == limits["training_limits"],
                "selected budget differs from actual config",
            )
        else:
            caps = {
                key: getattr(eagle_config, key)
                for key in ("max_steps", "max_tokens", "max_seconds", "max_epochs")
            }
            require(
                caps == limits["training_limits"],
                "selected EAGLE budget differs from actual continuous config",
            )
        train_argv = [
            sys.executable,
            str(ROOT / "scripts/train_nine_model_qat.py"),
            "--config",
            str(config_path),
            "--run-dir",
            "{run_dir}/candidates/" + name,
            "--bundle-sha256",
            "{bundle_sha256}",
            "--stage-name",
            name + "/train",
            "--completion-output",
            "{receipt}",
            "--allow-cuda",
            "--admission",
            "{admission}",
            "{resume}",
        ]
        export_argv = [
            sys.executable,
            str(ROOT / "scripts/export_nine_model_candidate.py"),
            "--bundle",
            str(output),
            "--bundle-sha256",
            "{bundle_sha256}",
            "--candidate",
            name,
            "--train-receipt",
            "{train_receipt}",
            "--completion-output",
            "{receipt}",
        ]
        candidates[name] = {
            **selected,
            "stages": {
                "train": {
                    "argv": train_argv,
                    "producer": pin(ROOT / "scripts/train_nine_model_qat.py"),
                    "wall_seconds": limits["wall_seconds"],
                },
                "export": {
                    "argv": export_argv,
                    "producer": pin(ROOT / "scripts/export_nine_model_candidate.py"),
                    "wall_seconds": limits["export_wall_seconds"],
                },
            },
        }
    inputs = dict(
        descriptor["inputs"], budget=descriptor["budget"], qa_ledger=descriptor["qa_ledger"]
    )
    protocol = json.loads(files.check(inputs["protocol"]).read_text())
    for name, record in inputs.items():
        if name == "prompts" and protocol.get("split") == "final":
            files.opaque(record)
            continue  # Keep final prompt bytes sealed until the final evaluator.
        files.check(record)
    plan = files.check(inputs["admission_plan"])
    environment = dict(descriptor.get("environment", {}))
    require(
        environment.get("CUDA_VISIBLE_DEVICES", descriptor["gpu_uuid"]) == descriptor["gpu_uuid"],
        "CUDA visibility differs from selected physical GPU UUID",
    )
    environment["CUDA_VISIBLE_DEVICES"] = descriptor["gpu_uuid"]
    bundle = {
        "schema": "nine_model_campaign_bundle_v1",
        "artifact_kind": "production",
        "preparation_complete": True,
        "gpu_uuid": descriptor["gpu_uuid"],
        "source": source,
        "inputs": inputs,
        "candidates": candidates,
        "controls": descriptor["controls"],
        "target_policy": {"immutable": True, "weights": "f16", "kv": "f16"},
        "resource_policy": descriptor["resource_policy"],
        "fresh_gates": sorted(GATES),
        "environment": environment,
        "final_set_authorized": descriptor.get("final_set_authorized", False),
        "admission": {
            "producer": pin(ROOT / "scripts/admit_nine_model_sm120.py"),
            "wall_seconds": budget["admission_wall_seconds"],
            "argv": [
                sys.executable,
                str(ROOT / "scripts/admit_nine_model_sm120.py"),
                "--plan",
                str(plan),
                "--plan-sha256",
                inputs["admission_plan"]["sha256"],
                "--bundle-sha256",
                "{bundle_sha256}",
                "--run-dir",
                "{run_dir}/fresh-sm120",
                "--receipt",
                "{receipt}",
                "--gpu-uuid",
                descriptor["gpu_uuid"],
                "--gpu-control",
                descriptor["gpu_control_path"],
            ],
        },
        "evaluation": {
            "producer": pin(ROOT / "scripts/evaluate_nine_model_native.py"),
            "wall_seconds": budget["evaluation_wall_seconds"],
            "argv": [
                sys.executable,
                str(ROOT / "scripts/evaluate_nine_model_native.py"),
                "--bundle",
                str(output),
                "--bundle-sha256",
                "{bundle_sha256}",
                "--evaluation-inputs",
                "{evaluation_inputs}",
                "--run-dir",
                "{run_dir}",
                "--completion-output",
                "{receipt}",
            ],
        },
    }
    require(not output.exists(), "frozen bundle destination exists; preserve original")
    temporary = output.with_name(output.name + ".validation")
    require(not temporary.exists(), "validation publication already exists")
    atomic_json(temporary, bundle)
    try:
        validate_bundle(temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema": "nine_model_frozen_bundle_v1",
        "status": "PASS",
        "bundle": pin(output),
        "gpu_queried": False,
        "optimizer_updates": 0,
        "fresh_sm120": "PENDING_at_authorized_start",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--inspect-draft", action="store_true")
    cli.add_argument("--materialize-configs", type=Path)
    args = cli.parse_args()
    if args.materialize_configs is not None:
        result = materialize_configs(json.loads(args.inputs.read_text()), args.materialize_configs)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0)
    require(args.inspect_draft or args.output is not None, "production output required")
    result = build(
        json.loads(args.inputs.read_text()),
        args.output.resolve() if args.output else None,
        inspect_draft=args.inspect_draft,
    )
    print(json.dumps(result, sort_keys=True))
