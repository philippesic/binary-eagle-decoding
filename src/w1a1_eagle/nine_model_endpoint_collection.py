"""Immutable imports of independently frozen lanes; never a CUDA readiness grant.

The production endpoint/checkpoint validators live with the staged exporter.
Tests may inject that boundary to exercise this model-free collection contract.
No training receipt is rewritten and no collection hash is a training hash.
"""

from __future__ import annotations

import json
from pathlib import Path

from .nine_model_pipeline import CANDIDATES, FAMILIES, Files, require, validate_bundle

SCHEMA = "nine_model_endpoint_collection_v1"
REQUIRED = ("candidates", "evaluation_source", "controls", "hardware_stage")
CANDIDATE_FIELDS = {
    "frozen_lane",
    "lane_state",
    "supervisor_state",
    "training_receipt",
    "export_receipt",
    "heldout_admission",
}
RUNTIME_PENDING = [
    "explicit full-nine evaluation authorization/scope",
    "fresh collection GPU lease and supervised continuation",
    "fresh native load/dispatch on each selected model",
    "actual owned process/context release and resource return per cell",
]


def read(files, record):
    return json.loads(files.check(record).read_text())


def missing_inputs(value):
    missing = [key for key in REQUIRED if not value.get(key)]
    for name in CANDIDATES:
        selected = value.get("candidates", {}).get(name, {})
        missing.extend(f"{name}/{key}" for key in sorted(CANDIDATE_FIELDS) if not selected.get(key))
    for family in FAMILIES:
        if not value.get("controls", {}).get(family):
            missing.append(f"controls/{family}")
    return missing


def source_context(locator, files):
    """Import a real existing evaluation source without choosing a new protocol."""
    source = read(files, locator)
    if source.get("schema") == "nine_model_campaign_bundle_v1":
        source, _ = validate_bundle(Path(locator["path"]))
        return {
            "origin": source,
            "target": source["inputs"]["target"],
            "protocol": source["inputs"]["protocol"],
            "prompts": source["inputs"]["prompts"],
            "runtime": {"binary": source["inputs"]["binary"]},
            "source": source["source"],
            "controls": source["controls"],
            "target_policy": source["target_policy"],
            "final_set_authorized": source.get("final_set_authorized", False),
        }
    require(source.get("schema") == "nine_model_lane_endpoint_plan_v1", "unknown evaluation source")
    # Existing endpoint validator is source-only; its live continuation is separate.
    from run_nine_model_lane_endpoint import validate_plan

    source, _, _, _ = validate_plan(Path(locator["path"]), locator["sha256"])
    return {
        "origin": source,
        "target": source["target"],
        "protocol": source["protocol"],
        "prompts": source["prompts"],
        "runtime": source["runtime"],
        "source": source["source"],
        "controls": {"eagle": source["control"]},
        "target_policy": source["target_policy"],
        "final_set_authorized": False,
    }


def validate_control(family, control, files, source):
    require(family in source["controls"], "original control source PENDING: " + family)
    require(
        control.get("family") == family
        and control.get("precision") == "Q4_0"
        and control.get("frozen_original") is True,
        "exact original Q4 control declaration required: " + family,
    )
    files.check(control["model"])
    reference = read(files, control["provenance"])
    require(
        reference.get("schema") == "nine_model_original_q4_reference_v1"
        and reference.get("family") == family
        and reference.get("precision") == "Q4_0"
        and reference.get("frozen_original") is True
        and reference.get("model") == control["model"]
        and reference.get("evidence"),
        "original Q4 producer/historical provenance differs: " + family,
    )
    for record in reference["evidence"]:
        files.check(record)
    if family in source["controls"]:
        require(
            control["model"] == source["controls"][family]["model"], "original control relabelled"
        )
        if source["controls"][family].get("deployment_coverage") is not None:
            require(
                control["deployment_coverage"] == source["controls"][family]["deployment_coverage"],
                "original control coverage relabelled",
            )
    if family == "eagle":
        from prepare_nine_model_lane_endpoint import ORIGINAL_EAGLE_Q4_SHA256

        require(
            control["model"]["sha256"] == ORIGINAL_EAGLE_Q4_SHA256, "original EAGLE hash differs"
        )
    require(
        isinstance(control.get("deployment_coverage"), dict) and control["deployment_coverage"],
        "original control coverage/exceptions missing",
    )
    require(
        reference.get("deployment_coverage", control["deployment_coverage"])
        == control["deployment_coverage"],
        "control coverage does not join original provenance",
    )


def endpoint_validators():
    from export_nine_model_lane_candidate import validate_endpoint, validate_export

    return validate_endpoint, validate_export


def validate_collection(value, *, files=None, validators=None, source_loader=None):
    require(value.get("schema") == SCHEMA, "collection schema differs")
    require(
        set(value) == {"schema", "artifact_kind", "campaign_complete", *REQUIRED},
        "exact collection fields required; no replacement budget or readiness metadata",
    )
    require(
        value.get("artifact_kind") == "production" and value.get("campaign_complete") is False,
        "fixture/complete claim cannot import production endpoints",
    )
    require(
        not missing_inputs(value), "collection inputs PENDING: " + "; ".join(missing_inputs(value))
    )
    require(set(value["candidates"]) == set(CANDIDATES), "exact six candidate inventory required")
    require(set(value["controls"]) == set(FAMILIES), "exact three original controls required")
    files = files or Files()
    source = (source_loader or source_context)(value["evaluation_source"], files)
    require(
        source["target_policy"] == {"immutable": True, "weights": "f16", "kv": "f16"},
        "target/verifier/F16 KV policy differs",
    )
    protocol = read(files, source["protocol"])
    require(
        protocol.get("schema") == "nine_model_native_protocol_v1"
        and protocol.get("split") in {"development", "final"}
        and type(protocol.get("repetitions")) is int
        and protocol["repetitions"] >= 5
        and type(protocol.get("warmups_per_cell")) is int
        and protocol["warmups_per_cell"] >= 2
        and protocol.get("diagnostic_pass_separate_from_timing") is True,
        "paired native protocol declaration differs",
    )
    # Preparation never reads sealed prompt bytes. File shape/path may be checked.
    if protocol["split"] == "final":
        files.opaque(source["prompts"])
    else:
        files.check(source["prompts"])
    files.check(source["target"])
    for record in (*source["runtime"].values(), *source["source"].values()):
        files.check(record)
    stage = value["hardware_stage"]
    require(
        set(stage) == {"gpu_uuid", "compute_capability", "runtime_gate"}
        and isinstance(stage["gpu_uuid"], str)
        and stage["gpu_uuid"]
        and stage["compute_capability"] == [12, 0]
        and stage["runtime_gate"] == "PENDING_fresh_lease_load_dispatch_release",
        "declared SM120 stage cannot grant runtime readiness",
    )
    validate_endpoint, validate_export = validators or endpoint_validators()
    contexts, exports = {}, {}
    for name in CANDIDATES:
        selected = value["candidates"][name]
        require(set(selected) == CANDIDATE_FIELDS, "exact candidate import associations required")
        context = validate_endpoint(
            selected["frozen_lane"],
            selected["lane_state"],
            selected["supervisor_state"],
            selected["training_receipt"],
            files=files,
        )
        require(context["lane"]["candidate"] == name, "candidate lane relabelled")
        require(
            context["target"] == source["target"]
            and context["lane"]["target_policy"] == source["target_policy"]
            and context["lane"]["gpu_uuid"] == stage["gpu_uuid"],
            "mixed target/precision/hardware lane import",
        )
        exported = validate_export(context, selected["export_receipt"], files=files)
        require(exported["candidate"] == name, "export candidate relabelled")
        admission = read(files, selected["heldout_admission"])
        require(
            admission.get("schema") == "nine_model_lane_development_admission_v1"
            and protocol["split"] == "development"
            and admission.get("split") == "development"
            and admission.get("training_disjoint") is True
            and admission.get("prompts") == source["prompts"]
            and admission.get("frozen_training_source") == context["source_bindings"]
            and admission.get("evidence"),
            "held-out prompts not bound/disjoint from original lane source",
        )
        for record in admission["evidence"]:
            files.check(record)
        contexts[name], exports[name] = context, exported
    for family, control in value["controls"].items():
        validate_control(family, control, files, source)
    require(
        len({(e["model"]["path"], e["model"]["sha256"]) for e in exports.values()}) == 6,
        "candidate model locators reused/relabelled",
    )
    return {
        "collection": value,
        "files": files,
        "source": source,
        "protocol": protocol,
        "contexts": contexts,
        "exports": exports,
    }


def inspection(context=None, *, missing=()):
    return {
        "schema": "nine_model_endpoint_collection_inspection_v1",
        "status": "PENDING" if missing else "VALIDATED_IMPORTS_RUNTIME_PENDING",
        "missing": list(missing),
        "runtime_pending": RUNTIME_PENDING,
        "production_ready": False,
        "execution_allowed": False,
        "gpu_queried": False,
        "training_changed": False,
        "campaign_complete": False,
        "training_bundle_sha256": {
            name: selected["frozen_lane"]["sha256"]
            for name, selected in context["collection"]["candidates"].items()
        }
        if context
        else {},
    }


def evaluation_view(context):
    """Read-only argument view for the existing native command builder.

    This is not a campaign bundle: its hash cannot be passed as a lane hash.
    Runtime execution needs a separate supervised collection adapter.
    """
    source = context["source"]
    models = {name: exported["model"] for name, exported in context["exports"].items()}
    models.update(
        {
            family + "_q4": control["model"]
            for family, control in context["collection"]["controls"].items()
        }
    )
    return {"inputs": {"target": source["target"], "binary": source["runtime"]["binary"]}}, models
