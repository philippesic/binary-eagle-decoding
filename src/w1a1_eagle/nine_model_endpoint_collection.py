"""Immutable imports of independently frozen lanes; never a CUDA readiness grant.

The production endpoint/checkpoint validators live with the staged exporter.
Tests may inject that boundary to exercise this model-free collection contract.
No training receipt is rewritten and no collection hash is a training hash.
"""

from __future__ import annotations

import json
import re
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
            "final_selection": source.get("heldout_selection"),
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


def opaque_rows(files, manifest_locator, split):
    """Read only original manifest/index metadata; never hash prompt payloads."""
    manifest = read(files, manifest_locator)
    require(
        manifest.get("schema") in {"w1a_data_manifest_v1", "continuous_w1ax_prompt_manifest_v1"},
        "original opaque corpus manifest PENDING",
    )
    entry = manifest.get("files", {}).get(split)
    require(isinstance(entry, dict), "original corpus split PENDING: " + split)
    shards = entry.get("shards", [entry])
    require(isinstance(shards, list) and shards, "original corpus shards absent")
    result = []
    parent = Path(manifest_locator["path"]).parent
    for shard in shards:
        records = {
            key: {"path": str((parent / shard[key]).resolve()), "sha256": shard[key + "_sha256"]}
            for key in ("prompts", "index")
        }
        require(
            all(parent in Path(r["path"]).parents for r in records.values()),
            "corpus shard escapes original manifest directory",
        )
        files.opaque(records["prompts"])
        rows = [
            json.loads(line)
            for line in files.check(records["index"]).read_text().splitlines()
            if line.strip()
        ]
        require(
            type(shard.get("prompts_count")) is int
            and len(rows) == shard["prompts_count"]
            and rows,
            "opaque index count differs",
        )
        for row in rows:
            require(
                isinstance(row, dict)
                and set(row)
                <= {
                    "id",
                    "group",
                    "source_id",
                    "source_row_id",
                    "content_sha256",
                    "domain",
                    "topic",
                    "characters",
                    "word_count",
                    "input_tokens",
                    "category",
                    "split",
                    "source_split",
                    "role",
                }
                and all(v is None or type(v) in (str, int) for v in row.values()),
                "opaque index contains unsupported fields/prompt payload",
            )
            for key in ("id", "group", "source_id", "source_row_id"):
                require(
                    isinstance(row.get(key), str) and row[key],
                    "opaque source identity PENDING: " + key,
                )
            require(
                isinstance(row.get("content_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["content_sha256"]),
                "opaque content hash absent",
            )
            for key in ("split", "source_split", "role"):
                require(
                    key not in row
                    or (isinstance(row[key], str) and row[key].lower() == split.lower()),
                    "opaque source role differs",
                )
        require(len({r["id"] for r in rows}) == len(rows), "duplicate opaque IDs")
        result.append((records, rows))
    return result


def final_origins(value, contexts, exports):
    return {
        name: {
            **{key: record for key, record in selected.items() if key != "heldout_admission"},
            "config": contexts[name]["lane"]["config"],
            "frozen_training_source": contexts[name]["source_bindings"],
            "model": exports[name]["model"],
        }
        for name, selected in value["candidates"].items()
    }


def selected_train_rows(context, files, corpus_locator, corpus_rows):
    """Join actual frozen trainer membership to original opaque TRAIN identities."""
    spec = context["spec"]
    if context["lane"]["candidate"].startswith("eagle_"):
        continuous = read(files, spec["eagle_config"])
        require(
            continuous.get("schema") == "continuous_w1ax_experiment_v1",
            "original EAGLE config differs",
        )
        stages = continuous.get("stages", {})
        require(
            stages.get("corpus_manifest") == corpus_locator,
            "original EAGLE corpus anchor PENDING/differs",
        )
        rows = []
        captures = [c for c in stages.get("captures", []) if c.get("split") == "train"]
        require(captures, "original TRAIN capture membership PENDING")
        provider_locator = {
            "path": str(
                (Path(spec["prepared"]["run_dir"]) / "stages/train-providers.json").resolve()
            ),
            "sha256": context["source_bindings"]["execution_manifest_sha256"],
        }
        provider = read(files, provider_locator)
        require(
            provider.get("schema") == "w1ax_streaming_train_v2"
            and provider.get("split") == "train"
            and provider.get("training_eligible") is True
            and len(provider.get("shards", [])) == len(captures),
            "actual frozen TRAIN provider membership differs",
        )
        for ordinal, capture in enumerate(captures):
            record = provider["shards"][ordinal]
            child = read(
                files,
                {
                    "path": str(Path(record["provider_manifest"]).resolve()),
                    "sha256": record["provider_manifest_sha256"],
                },
            )
            require(
                child.get("schema") == "w1ax_native_train_provider_v2"
                and child.get("split") == "train"
                and child.get("training_eligible") is True
                and child.get("sha256", {}).get("prompts") == capture.get("prompts_sha256")
                and child.get("prompt_count") == capture.get("prompt_count"),
                "actual TRAIN provider/capture differs",
            )
            require(
                capture.get("source_corpus_manifest_sha256") == corpus_locator["sha256"],
                "TRAIN capture corpus differs",
            )
            matches = [
                (records, index)
                for records, index in corpus_rows
                if records["prompts"]["sha256"] == capture.get("source_prompts_sha256")
                and records["index"]["sha256"] == capture.get("source_index_sha256")
            ]
            require(len(matches) == 1, "TRAIN source index anchor differs")
            positions = capture.get("source_positions")
            index = matches[0][1]
            require(
                isinstance(positions, list)
                and positions
                and len(positions) == capture.get("prompt_count")
                and len(set(positions)) == len(positions)
                and all(type(i) is int and 0 <= i < len(index) for i in positions),
                "original TRAIN source positions differ",
            )
            rows.extend(index[i] for i in positions)
        return rows
    locator = {key: spec["data"][key] for key in ("path", "sha256")}
    require(
        locator["sha256"] == context["source_bindings"].get("data_manifest_sha256"),
        "actual block TRAIN source differs",
    )
    manifest = read(files, locator)
    inventory_locator = manifest["train_inventory"]
    if not Path(inventory_locator["path"]).is_absolute():
        inventory_locator = dict(
            inventory_locator,
            path=str((Path(locator["path"]).parent / inventory_locator["path"]).resolve()),
        )
    inventory = read(files, inventory_locator)
    require(inventory.get("schema") == "block_train_inventory_v1", "TRAIN inventory differs")
    by_id = {row["id"]: row for _, rows in corpus_rows for row in rows}
    require(len(by_id) == sum(len(rows) for _, rows in corpus_rows), "duplicate corpus TRAIN IDs")
    selected = []
    for chain in manifest.get("chains", []):
        require(
            chain.get("split") in {"train", "calibration_fit", "calibration_validation"},
            "block source role is not TRAIN-derived",
        )
        prompt_id = chain.get("prompt_id")
        row = by_id.get(prompt_id)
        original = inventory.get("prompts", {}).get(prompt_id, {})
        require(
            row is not None
            and original.get("split") == "TRAIN"
            and row["content_sha256"] == original.get("sha256") == chain.get("prompt_sha256"),
            "block actual TRAIN ID/content/index join differs",
        )
        selected.append(row)
    require(selected, "actual block TRAIN membership PENDING")
    return selected


def validate_final_authority(authority, files):
    """Keep standing instruction evidence distinct from agent-frozen scope.

    This is a conservative text consistency gate, not proof of human authorship.
    The original evaluation source remains the authenticated provenance root.
    Ambiguous/negative/training-only instructions cannot unseal final here.
    """
    require(
        isinstance(authority, dict)
        and set(authority) == {"record", "instruction", "phase"}
        and authority["phase"] == "final"
        and isinstance(authority["instruction"], str)
        and authority["instruction"].strip(),
        "original explicit final authority PENDING/differs",
    )
    instruction = authority["instruction"].strip()
    evidence = files.check(authority["record"]).read_text()
    require(instruction in evidence, "final instruction does not join original authority record")
    words = instruction.casefold()
    require(
        re.search(r"\b(final|sealed|reserved)\b", words)
        and re.search(r"\b(evaluate|evaluation|compare|comparison)\b", words)
        and not re.search(r"\b(not|never|pause|prohibit|forbid|don't|cannot)\b", words),
        "unambiguous standing final evaluation authority PENDING",
    )


def final_admissions(value, source, contexts, exports, files, *, check_admissions=True):
    selection = source.get("final_selection")
    require(
        source.get("final_set_authorized") is True and isinstance(selection, dict),
        "authenticated frozen final selection PENDING; boolean alone is insufficient",
    )
    require(
        set(selection)
        == {
            "phase",
            "corpus_manifest",
            "split",
            "shard",
            "prompts",
            "protocol",
            "target",
            "origins",
            "authorization",
            "selection_provenance",
            "human_selected",
        },
        "exact final selection scope required",
    )
    require(
        selection["phase"] == "final"
        and selection["selection_provenance"] == "agent_selected"
        and selection["human_selected"] is False
        and selection["split"] in {"final", "sealed_test"}
        and selection["prompts"] == source["prompts"]
        and selection["protocol"] == source["protocol"]
        and selection["target"] == source["target"]
        and selection["origins"] == final_origins(value, contexts, exports),
        "final selected protocol/target/recipe/model/origins differ",
    )
    # The human authorizes final generally; the source freezes concrete scope
    # later under that unchanged instruction. Future hashes need no new approval.
    authority = source["origin"].get("evaluation", {}).get("authorization")
    require(
        selection["authorization"] == authority,
        "standing authority differs from original evaluation source",
    )
    validate_final_authority(authority, files)
    shards = opaque_rows(files, selection["corpus_manifest"], selection["split"])
    shard = selection["shard"]
    require(
        type(shard) is int
        and 0 <= shard < len(shards)
        and shards[shard][0]["prompts"] == source["prompts"],
        "selected final shard/prompt locator differs",
    )
    final_rows = shards[shard][1]  # Entire explicitly selected shard; no caller subset.
    train = (
        opaque_rows(files, selection["corpus_manifest"], "train")
        if selection["split"] == "sealed_test"
        else opaque_rows(files, selection["corpus_manifest"], "train_large")
    )
    for name, context in contexts.items():
        selected = selected_train_rows(context, files, selection["corpus_manifest"], train)
        for key in ("id", "group", "content_sha256"):
            require(
                not ({r[key] for r in final_rows} & {r[key] for r in selected}),
                "final/TRAIN " + key + " overlap: " + name,
            )
        require(
            not (
                {(r["source_id"], r["source_row_id"]) for r in final_rows}
                & {(r["source_id"], r["source_row_id"]) for r in selected}
            ),
            "final/TRAIN original source row overlap: " + name,
        )
        if check_admissions:
            admission = read(files, value["candidates"][name]["heldout_admission"])
            require(
                admission == final_admission(value, source, contexts, exports, name),
                "final lane admission stale/incomplete/forged: " + name,
            )
    return {
        "split": "final",
        "selection": selection,
        "sealed_prompts": True,
        "prompt_count": len(final_rows),
    }


def final_admission(value, source, contexts, exports, name):
    """Deterministic association receipt; all evidence is revalidated on import."""
    return {
        "schema": "nine_model_lane_final_admission_v1",
        "split": "final",
        "evaluation_source": value["evaluation_source"],
        "selection": source["final_selection"],
        "candidate": name,
        "origin": final_origins(value, contexts, exports)[name],
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
        if protocol["split"] == "final":
            contexts[name], exports[name] = context, exported
            continue
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
    heldout = (
        final_admissions(value, source, contexts, exports, files)
        if protocol["split"] == "final"
        else {"split": "development"}
    )
    for family, control in value["controls"].items():
        validate_control(family, control, files, source)
    require(
        len({(e["model"]["path"], e["model"]["sha256"]) for e in exports.values()}) == 6,
        "candidate model locators reused/relabelled",
    )
    return {
        "collection": value,
        "heldout": heldout,
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
