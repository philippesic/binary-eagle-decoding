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
    for key in (
        "inputs",
        "controls",
        "candidates",
        "qa_ledger",
        "budget",
        "gpu_uuid",
        "resource_policy",
        "gpu_control_path",
    ):
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
            warm = selected["profile"] == "a8_to_a1_reset"
            base_config = files.check(selected["eagle_config_template"])
            eagle = json.loads(base_config.read_text())
            eagle["training"].update(limits)
            eagle["training"].update(
                activation_bits=[8 if warm else bits],
                development_lifecycle="standalone",
                activation_quantization="fixed",
                initialization_sha256=initializer["sha256"],
                initialization_policy=policy,
            )
            if "encoding" in initializer:
                eagle["training"]["initialization_encoding"] = initializer["encoding"]
            lane_config = directory / (name + "-continuous.json")
            atomic_json(lane_config, eagle)
            if source_validate:
                importlib.import_module("train_continuous_w1ax").load_config(lane_config)
            spec.update(eagle_config=pin(lane_config), prepared=selected["prepared"])
            if warm:
                require(
                    not source_validate or hasattr(trainer, "run_eagle_curriculum"),
                    "actual EAGLE curriculum source API integration PENDING",
                )
                curriculum = copy.deepcopy(budget["candidates"][name]["curriculum"])
                require(
                    [stage["activation_bits"] for stage in curriculum["stages"]] == [8, 1]
                    and curriculum.get("optimizer_transition") == "fresh",
                    "EAGLE curriculum must declare A8-to-A1 with fresh optimizer",
                )
                if source_validate:
                    from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage
                    from w1a1_eagle.qat_curriculum_runner import RunnerConfig

                    CurriculumConfig(
                        tuple(PrecisionStage(**stage) for stage in curriculum["stages"]),
                        curriculum["optimizer_transition"],
                    )
                    RunnerConfig(**selected.get("curriculum_runner", {}))
                spec.update(
                    precision_stage="a8_to_a1",
                    curriculum=curriculum,
                    curriculum_runner=selected.get("curriculum_runner", {}),
                )
                for quota in ("min_a1_updates", "min_a1_supervised_tokens"):
                    if quota in budget["candidates"][name]:
                        spec[quota] = budget["candidates"][name][quota]
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
                data={
                    **selected["data"],
                    **(
                        {"admission": selected["data_admission"]}
                        if selected.get("data_admission")
                        else {}
                    ),
                },
                limits=limits,
                checkpoint_every=selected["checkpoint_every"],
                precision_stage="a8_to_a1" if selected["profile"] == "a8_to_a1_reset" else "direct",
            )
            if spec["precision_stage"] == "a8_to_a1":
                spec["a8_warmup_steps"] = selected["a8_warmup_steps"]
                for quota in ("min_a1_updates", "min_a1_supervised_tokens"):
                    if quota in budget["candidates"][name]:
                        spec[quota] = budget["candidates"][name][quota]
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


def inspect_admission_inputs(descriptor):
    pending = []
    for role in ("backend_binary", "block_native_binary", "teacher_binary", "binary", "target"):
        if role not in descriptor.get("inputs", {}):
            pending.append("missing actual admission binary/target: " + role)
    if not descriptor.get("preflight", {}).get("native_source_revision"):
        pending.append("missing actual native producer source revision")
    for family in FAMILIES:
        if (
            not descriptor.get("preflight", {})
            .get("portability", {})
            .get(family, {})
            .get("golden_manifest")
        ):
            pending.append(family + ": bounded native TRAIN golden manifest PENDING")
    for role in ("eagle_smoke_prompts", "eagle_smoke_capture"):
        if role not in descriptor.get("inputs", {}):
            pending.append("missing authenticated EAGLE TRAIN native smoke input: " + role)
    for name in CANDIDATES:
        for role in ("config", "initial_model", "initial_export_audit", "data_admission"):
            if role not in descriptor.get("candidates", {}).get(name, {}):
                pending.append(name + ": " + role + " PENDING")
    for role in ("resource_policy", "gpu_uuid", "controls"):
        if role not in descriptor:
            pending.append("missing actual admission binding: " + role)
    return sorted(pending)


def admission_source_bindings(selected, spec, files, target_sha256):
    """Metadata-only exact trainer source join; never constructs a model."""
    if spec["family"] == "eagle":
        ready_path = files.check(selected["data_admission"])
        require(
            ready_path == Path(spec["prepared"]["run_dir"]).resolve() / "preparation-ready.json"
            and selected["data_admission"]["sha256"] == spec["prepared"]["ready_sha256"],
            "EAGLE prepared metadata locator differs from actual source config",
        )
        ready = json.loads(ready_path.read_text())
        require(
            ready.get("schema") == "continuous_w1ax_preparation_ready_v1"
            and ready.get("preparation_complete") is True
            and ready.get("optimization_started") is False,
            "actual completed EAGLE teacher source metadata absent",
        )
        source = ready["teacher_coverage"]["source"]
        require(
            source["common_source_sha256"]["target_gguf"] == target_sha256,
            "EAGLE native teacher target differs",
        )
        require("bundle_sha256" not in source, "EAGLE provider source must not bind wrapper bundle")
        return source
    data = spec["data"]
    manifest = json.loads(files.check({key: data[key] for key in ("path", "sha256")}).read_text())
    require(
        data.get("admission") == selected["data_admission"],
        "completed current-host block data admission not wired into trainer config",
    )
    admission = json.loads(files.check(selected["data_admission"]).read_text())
    require(
        admission.get("schema") == "block_data_completed_admission_v1"
        and admission.get("manifest_sha256") == data["sha256"],
        "actual completed block data metadata admission absent",
    )
    require(
        manifest["producer"]["target_sha256"] == target_sha256,
        "block native teacher target differs",
    )
    return {
        "base_gguf_sha256": spec["model"]["sha256"],
        "data_manifest_sha256": data["sha256"],
        "synthetic": False,
        "target_sha256": target_sha256,
    }


def materialize_admission_plan(descriptor, output, *, fixture=False, inspect_draft=False, api=None):
    """Wire known source CLIs from materialized pins, without GPU/model load.

    Plan is an execution request, not an admission receipt or readiness result.
    Only a later actual fresh hardware producer can grant training admission.
    """
    pending = inspect_admission_inputs(descriptor)
    if inspect_draft:
        return {
            "schema": "nine_model_admission_plan_inspection_v1",
            "status": "PENDING" if pending else "PASS",
            "pending": pending,
            "model_loaded": False,
            "gpu_queried": False,
            "production_ready": False,
        }
    require(not pending, "actual admission preparation PENDING: " + "; ".join(pending))
    require(not output.exists(), "preserve existing frozen admission plan")
    files = Files()
    inputs = descriptor["inputs"]
    for name in (
        "backend_binary",
        "block_native_binary",
        "teacher_binary",
        "binary",
        "target",
        "eagle_smoke_prompts",
        "eagle_smoke_capture",
    ):
        files.check(inputs[name])
    preflight = descriptor["preflight"]
    if not fixture:
        from check_block_capture_portability import NativeCaptureGoldens

        for family in FAMILIES:
            record = preflight["portability"][family]["golden_manifest"]
            NativeCaptureGoldens(files.check(record), expected_sha256=record["sha256"])

    revision = preflight["native_source_revision"]
    require(
        isinstance(revision, str)
        and len(revision) == 40
        and all(c in "0123456789abcdef" for c in revision),
        "native source revision required",
    )
    directory = output.parent / (output.name + "-inputs")
    require(not directory.exists(), "preserve previous admission input publications")
    directory.mkdir(parents=True)
    trainer = importlib.import_module("train_nine_model_qat")
    source = {}
    for folder in (ROOT / "src/w1a1_eagle", ROOT / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            source[str(path.relative_to(ROOT))] = pin(path)
    for name, record in inputs.items():
        if name in {
            "backend_binary",
            "block_native_binary",
            "teacher_binary",
            "binary",
            "target",
            "eagle_smoke_prompts",
            "eagle_smoke_capture",
        } or name.startswith("runtime_library_"):
            files.check(record)
            source["artifact:" + name] = record
    training_source = trainer.training_source_identity()
    for name, digest in training_source.items():
        require(source[name]["sha256"] == digest, "actual trainer source inventory differs")
    environment = dict(descriptor.get("environment", {}))
    require(
        environment.get("CUDA_VISIBLE_DEVICES", descriptor["gpu_uuid"]) == descriptor["gpu_uuid"],
        "admission device visibility differs from selected physical UUID",
    )
    environment["CUDA_VISIBLE_DEVICES"] = descriptor["gpu_uuid"]
    candidates = {}
    for name in CANDIDATES:
        selected = descriptor["candidates"][name]
        config_path = files.check(selected["config"])
        spec = trainer.load_spec(config_path)
        family, precision = name.split("_")
        require(
            spec["candidate"] == name and spec["family"] == family,
            "preflight candidate config family/name differs",
        )
        bits = int(precision[1:])
        model = files.check(selected["initial_model"])
        export_path = files.check(selected["initial_export_audit"])
        audit = json.loads(export_path.read_text())
        require(
            audit.get("serialization_audit_passed") is True
            and audit.get("activation_bits") == bits
            and audit.get("output", {}).get("sha256") == selected["initial_model"]["sha256"],
            "initial native candidate bits/export/model ancestry differs",
        )
        bindings = admission_source_bindings(selected, spec, files, inputs["target"]["sha256"])
        if not fixture and family != "eagle":
            from w1a1_eagle.block_data import BlockDataset

            dataset = BlockDataset(
                spec["data"]["path"],
                expected_sha256=spec["data"]["sha256"],
                allow_synthetic=False,
                verify_artifacts=False,
                admission_path=selected["data_admission"]["path"],
                admission_sha256=selected["data_admission"]["sha256"],
            )
            require(
                dataset.manifest["family"] == family
                and dataset.manifest["producer"]["target_sha256"] == inputs["target"]["sha256"],
                "production block data/native source family/target differs",
            )
        source["candidate:" + name + ":data_admission"] = selected["data_admission"]
        if family != "eagle":
            source["candidate:" + name + ":data_manifest"] = {
                key: spec["data"][key] for key in ("path", "sha256")
            }
        if spec.get("initialization"):
            source["candidate:" + name + ":initializer"] = {
                key: spec["initialization"][key] for key in ("path", "sha256")
            }
        native_script = (
            ROOT
            / "scripts"
            / (
                "check_eagle_binary_native.py"
                if family == "eagle"
                else "check_block_binary_native.py"
            )
        )
        if family == "eagle":
            smoke_config = directory / (name + "-native.json")
            atomic_json(
                smoke_config,
                {
                    "schema": "eagle_native_graph_smoke_config_v1",
                    "inputs": {
                        "binary": inputs["binary"],
                        "model": selected["initial_model"],
                        "target": inputs["target"],
                        "export_audit": selected["initial_export_audit"],
                        "train_prompts": inputs["eagle_smoke_prompts"],
                        "train_capture": inputs["eagle_smoke_capture"],
                        **{k: v for k, v in inputs.items() if k.startswith("runtime_library_")},
                    },
                    "activation_bits": bits,
                    "gpu_uuid": descriptor["gpu_uuid"],
                    "hardware": "rtx5080",
                    "port": preflight.get("eagle_port", 18290),
                    "context_tokens": preflight.get("context_tokens", 2048),
                    "startup_wall_seconds": preflight.get("startup_wall_seconds", 300),
                    "request_wall_seconds": preflight.get("request_wall_seconds", 120),
                    "resource_policy": descriptor["resource_policy"],
                    "environment": environment,
                },
            )
            source[str(smoke_config)] = pin(smoke_config)
            native_argv = [
                sys.executable,
                str(native_script),
                "--config",
                str(smoke_config),
                "--receipt",
                "{receipt}",
            ]
        else:
            native_argv = [
                sys.executable,
                str(native_script),
                "--binary",
                inputs["block_native_binary"]["path"],
                "--model",
                str(model),
                "--export",
                str(export_path),
                "--gpu-layers",
                "999",
                "--require-cuda",
                "--target",
                inputs["target"]["path"],
                "--target-sha256",
                inputs["target"]["sha256"],
                "--receipt",
                "{receipt}",
            ]
        candidates[name] = {
            "family": family,
            "final_bits": bits,
            "config": selected["config"],
            "model": selected["initial_model"],
            "export": selected["initial_export_audit"],
            "profile": selected["profile"],
            "precision_stage": spec.get("precision_stage", "direct"),
            "source_bindings": bindings,
            "native": {
                "producer": pin(native_script),
                "argv": native_argv,
                "wall_seconds": preflight.get("native_wall_seconds", 600),
            },
            "backward": {
                "producer": pin(ROOT / "scripts/train_nine_model_qat.py"),
                "argv": [
                    sys.executable,
                    str(ROOT / "scripts/train_nine_model_qat.py"),
                    "--config",
                    str(config_path),
                    "--run-dir",
                    "{run_dir}",
                    "--bundle-sha256",
                    "{bundle_sha256}",
                    "--stage-name",
                    name + "/preflight",
                    "--completion-output",
                    "{receipt}",
                    "--allow-cuda",
                    "--smoke-zero-updates",
                ],
                "wall_seconds": preflight.get("backward_wall_seconds", 1200),
            },
        }
    portable = {}
    for family in FAMILIES:
        settings = preflight["portability"][family]
        golden = files.check(settings["golden_manifest"])
        manifest = json.loads(golden.read_text())
        require(
            manifest.get("schema") == "nine_model_train_capture_goldens_v1"
            and manifest.get("family") == family
            and manifest.get("target_sha256") == inputs["target"]["sha256"],
            "native TRAIN golden family/target differs; speculative corpus cannot substitute",
        )
        script = (
            ROOT
            / "scripts"
            / (
                "check_eagle_capture_portability.py"
                if family == "eagle"
                else "check_block_capture_portability.py"
            )
        )
        source[str(golden)] = settings["golden_manifest"]
        argv = [
            sys.executable,
            str(script),
            "--golden-manifest",
            str(golden),
            "--manifest-sha256",
            settings["golden_manifest"]["sha256"],
            "--binary",
            inputs["teacher_binary"]["path"],
            "--binary-sha256",
            inputs["teacher_binary"]["sha256"],
            "--target",
            inputs["target"]["path"],
            "--producer-source-revision",
            revision,
            "--output-root",
            "{run_dir}/fresh-capture",
            "--output",
            "{receipt}",
            "--expected-compute-capability",
            "12",
            "0",
            "--gpu-layers",
            "999",
            "--max-tokens",
            str(settings.get("max_tokens", 1024)),
            "--max-cases",
            "3",
            "--timeout-seconds",
            str(settings.get("timeout_seconds", 120)),
        ]
        for key, default in (
            ("feature-atol", 0.002),
            ("feature-rtol", 0.002),
            ("logit-atol", 0.02),
            ("logit-rtol", 0.002),
        ):
            argv.extend(["--" + key, str(settings.get(key.replace("-", "_"), default))])
        portable[family] = {
            "producer": pin(script),
            "argv": argv,
            "wall_seconds": settings.get("wall_seconds", 900),
        }
    require(
        set(descriptor["controls"]) == set(FAMILIES),
        "three immutable original Q4 controls required",
    )
    for family, control in descriptor["controls"].items():
        require(control.get("frozen_original") is True, "original Q4 control must remain frozen")
        files.check(control["model"])
    plan = {
        "schema": "nine_model_sm120_plan_v1",
        "artifact_kind": "fixture" if fixture else "production",
        "source": source,
        "training_source_files": training_source,
        "backend_binary": inputs["backend_binary"],
        "target": inputs["target"],
        "controls": descriptor["controls"],
        "candidates": candidates,
        "portability": portable,
        "resource_policy": descriptor["resource_policy"],
        "environment": environment,
    }
    if api is None:
        api = importlib.import_module("w1a1_eagle.nine_model_admission")
    temporary = output.with_name(output.name + ".validation")
    require(not temporary.exists(), "preserve previous plan validation publication")
    atomic_json(temporary, plan)
    try:
        api.validate_plan(temporary, fixture=fixture)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    resolved = copy.deepcopy(descriptor)
    resolved.setdefault("inputs", {})["admission_plan"] = pin(output)
    resolved_path = directory / "resolved-inputs.json"
    atomic_json(resolved_path, resolved)
    return {
        "schema": "nine_model_admission_plan_prepared_v1",
        "plan": pin(output),
        "resolved_inputs": pin(resolved_path),
        "artifact_kind": plan["artifact_kind"],
        "production_ready": False,
        "model_loaded": False,
        "gpu_queried": False,
        "optimizer_updates": 0,
    }


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
            files.check({key: data[key] for key in ("path", "sha256")})
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
            initial_precision = 8 if selected["profile"] == "a8_to_a1_reset" else precision
            require(
                eagle_config.activation_bits == (initial_precision,)
                and eagle_config.development_lifecycle == "standalone",
                "EAGLE candidate arithmetic/fresh-process lifecycle differs",
            )
        if selected["profile"] == "a8_to_a1_reset":
            require(
                spec.get("precision_stage") == "a8_to_a1",
                "warm precision transition must use actual source launcher",
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
            if selected["profile"] == "a8_to_a1_reset":
                from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage
                from w1a1_eagle.qat_curriculum_runner import RunnerConfig

                CurriculumConfig(
                    tuple(PrecisionStage(**stage) for stage in spec["curriculum"]["stages"]),
                    spec["curriculum"]["optimizer_transition"],
                )
                RunnerConfig(**spec.get("curriculum_runner", {}))
                require(
                    spec["curriculum"] == limits["curriculum"],
                    "human curriculum phase caps differ from actual source config",
                )
            else:
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
                "{stage_dir}/fresh-sm120",
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
    cli.add_argument("--materialize-admission-plan", type=Path)
    args = cli.parse_args()
    if args.materialize_admission_plan is not None:
        result = materialize_admission_plan(
            json.loads(args.inputs.read_text()),
            args.materialize_admission_plan,
            inspect_draft=args.inspect_draft,
        )
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0)
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
