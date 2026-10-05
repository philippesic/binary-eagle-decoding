#!/usr/bin/env python3
"""CPU export adapter for one original frozen production lane.

CLI inspection grants neither release nor execution. The runtime calls
export_endpoint with fresh release and CPU resource callbacks; this module never
queries a GPU, constructs a drafter, or loads target weights. Original producer
paths and hashes stay authoritative, including historical EAGLE lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_nine_model_lane import validate_lane  # noqa: E402

from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    Files,
    atomic_json,
    clean_environment,
    require,
    sha256,
)


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_checkpoint(path):
    # Same hash-bound local pickle contract as the actual trainers. CPU mmap
    # avoids an eager second full checkpoint copy; never restores CUDA RNG.
    import torch

    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _rng(value):
    import random

    import numpy as np
    import torch

    require(
        isinstance(value, dict) and {"python", "numpy", "torch"} <= set(value), "saved RNG absent"
    )
    # Validate in isolated CPU generators; leave the caller's RNG unchanged and
    # never call any CUDA API even when a CUDA RNG byte state was captured.
    random.Random().setstate(value["python"])
    np.random.RandomState().set_state(value["numpy"])
    torch.Generator(device="cpu").set_state(value["torch"])
    if "cuda" in value:
        require(
            isinstance(value["cuda"], torch.Tensor)
            and value["cuda"].device.type == "cpu"
            and value["cuda"].dtype == torch.uint8
            and value["cuda"].numel() > 0,
            "serialized CUDA RNG bytes absent",
        )


def _optimizer(value, updates):
    import torch

    require(
        isinstance(value, dict) and value.get("state") and value.get("param_groups"),
        "trained optimizer absent",
    )
    ids = [key for group in value["param_groups"] for key in group["params"]]
    require(
        len(ids) == len(set(ids)) and set(ids) == set(value["state"]),
        "serialized optimizer ownership differs",
    )
    for state in value["state"].values():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "trained Adam moments absent")
        step = state["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and bool(torch.isfinite(step).all())
            and 0 < float(step) <= updates
            and float(step).is_integer(),
            "trained Adam update counter differs",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            require(
                isinstance(moment, torch.Tensor)
                and moment.dtype == torch.float32
                and moment.device.type == "cpu"
                and bool(torch.isfinite(moment).all()),
                "serialized Adam moment invalid",
            )
        require(
            state["exp_avg"].shape == state["exp_avg_sq"].shape
            and bool((state["exp_avg_sq"] >= 0).all()),
            "serialized Adam moment shape/variance differs",
        )


def validate_endpoint(
    lane_locator,
    lane_state_locator,
    supervisor_locator,
    training_locator,
    *,
    files=None,
    checkpoint_loader=None,
):
    """Validate original typed endpoint and serialized state; no release claim."""
    files = Files() if files is None else files
    lane_path = files.check(lane_locator)
    lane, _, plan = validate_lane(lane_path, lane_locator["sha256"])
    state_path = files.check(lane_state_locator)
    state = json.loads(state_path.read_text())
    supervisor = json.loads(files.check(supervisor_locator).read_text())
    require(
        supervisor.get("status") == "finished"
        and type(supervisor.get("exit_code")) is int
        and supervisor["exit_code"] == 0
        and supervisor.get("received_signal") is None,
        "natural successful supervisor required",
    )
    require(
        state.get("schema") == "nine_model_lane_state_v1"
        and state.get("status") == "training_complete"
        and state.get("candidate") == lane["candidate"]
        and state.get("bundle_sha256") == lane_locator["sha256"]
        and state.get("campaign_complete") is False
        and state.get("owned_release", {}).get("owned_process_groups_absent") is True
        and state.get("train_receipt") == training_locator,
        "original successful released lane endpoint required",
    )
    receipt = json.loads(files.check(training_locator).read_text())
    counters = receipt.get("counters", {})
    require(
        receipt.get("schema") == "nine_model_stage_receipt_v1"
        and receipt.get("stage") == lane["candidate"] + "/train"
        and receipt.get("artifact_kind") == "production"
        and receipt.get("status") == "PASS"
        and receipt.get("committed") is True
        and receipt.get("completion_reason") == "approved_budget_complete"
        and receipt.get("bundle_sha256") == lane_locator["sha256"]
        and receipt.get("config_sha256") == lane["config"]["sha256"]
        and type(counters.get("step")) is int
        and counters["step"] > 0,
        "positive committed exact-config production endpoint required",
    )
    require(
        receipt.get("hardware", {}).get("gpu_uuid") == lane["gpu_uuid"]
        and receipt["hardware"].get("compute_capability") == [12, 0],
        "original training hardware differs",
    )
    spec = json.loads(files.check(lane["config"]).read_text())
    name = lane["candidate"]
    family, precision = name.split("_")
    bits = int(precision[1:])
    require(
        spec.get("candidate") == name
        and spec.get("family") == family
        and spec.get("precision_stage", "direct") == "direct",
        "direct staged config required",
    )
    budget = json.loads(files.check(lane["budget"]).read_text())
    limits = budget["candidates"][name]["training_limits"]
    elapsed = counters.get("elapsed_seconds")
    require(
        type(limits.get("max_seconds")) in (int, float)
        and math.isfinite(limits["max_seconds"])
        and limits["max_seconds"] > 0
        and type(elapsed) in (int, float)
        and math.isfinite(elapsed)
        and elapsed >= limits["max_seconds"],
        "full cumulative training budget required",
    )
    selected = plan["candidates"][name]
    source = selected["source_bindings"]
    require(source and "bundle_sha256" not in source, "original training source bindings required")
    target = plan["target"]
    files.check(target)  # streaming hash only, never target loading
    initial_audit = json.loads(files.check(selected["export"]).read_text())
    base = (
        initial_audit["base_gguf"]
        if family == "eagle"
        else {key: spec["model"][key] for key in ("path", "sha256")}
    )
    files.check(base)
    require(set(receipt.get("exports", {})) == {name}, "single-candidate trained export required")
    exported = receipt["exports"][name]
    require(exported.get("base_gguf_sha256") == base["sha256"], "base export ancestry differs")
    checkpoint = files.check(receipt["checkpoint"])
    run = state_path.parent / "training"
    require(run / "checkpoints" in checkpoint.parents, "checkpoint outside original trainer")
    for record in (exported["checkpoint"], exported["manifest"]):
        require(run in files.check(record).parents, "export outside original trainer")
    joint = json.loads(Path(exported["manifest"]["path"]).read_text())
    require(
        joint.get("checkpoint_sha256") == exported["checkpoint"]["sha256"]
        and joint.get("base_gguf_sha256") == base["sha256"]
        and joint.get("activation_bits") == bits,
        "serializer manifest ancestry/precision differs",
    )
    saved = (load_checkpoint if checkpoint_loader is None else checkpoint_loader)(checkpoint)
    context = dict(
        lane=lane,
        admission_plan=plan,
        spec=spec,
        base_model=base,
        target=target,
        source_bindings=source,
        training_limits=limits,
        receipt=receipt,
        train_receipt=training_locator,
        frozen_lane=lane_locator,
        lane_state=lane_state_locator,
        supervisor_state=supervisor_locator,
        checkpoint=receipt["checkpoint"],
        exported=exported,
    )
    if family == "eagle":
        from w1a1_eagle.continuous_qat import ContinuousConfig, immutable_config
        from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH

        lane_name = "A" + str(bits)
        outer_path = checkpoint.parent / "manifest.json"
        outer = json.loads(outer_path.read_text())
        continuous_spec = json.loads(files.check(spec["eagle_config"]).read_text())
        require(
            continuous_spec.get("schema") == "continuous_w1ax_experiment_v1",
            "original continuous config required",
        )
        continuous = continuous_spec["training"]
        configured = dict(continuous)
        configured["seeds"] = tuple(configured["seeds"])
        configured["activation_bits"] = tuple(configured["activation_bits"])
        configured = ContinuousConfig(**configured)
        require(
            digest(immutable_config(saved["config"]))
            == digest(immutable_config(asdict(configured))),
            "serialized EAGLE immutable config differs from original",
        )
        # Trainer config is normalized at load time; compare serialized immutable
        # config to its own authenticated manifest and the actual pinned config's
        # activation/lifecycle/budget semantics (wrapper config remains hash-bound).
        require(
            continuous.get("activation_bits") == [bits]
            and continuous.get("development_lifecycle") == "standalone"
            and all(continuous.get(key) == value for key, value in limits.items()),
            "EAGLE config budget/precision differs",
        )
        ready_path = Path(spec["prepared"]["run_dir"]).resolve() / "preparation-ready.json"
        ready = json.loads(
            files.check(
                {"path": str(ready_path), "sha256": spec["prepared"]["ready_sha256"]}
            ).read_text()
        )
        require(
            ready.get("schema") == "continuous_w1ax_preparation_ready_v1"
            and ready.get("preparation_complete") is True
            and ready.get("optimization_started") is False
            and ready.get("teacher_coverage", {}).get("source") == source,
            "original prepared TRAIN source differs",
        )
        source_hash = digest(source)
        require(
            source["common_source_sha256"]["target_gguf"] == target["sha256"]
            and source["common_source_sha256"]["base_draft_gguf"] == base["sha256"],
            "native teacher/base source differs",
        )
        require(
            outer.get("schema") == saved.get("schema") == "continuous_joint_w1ax_v1"
            and outer.get("source_sha256") == saved.get("source") == source_hash
            and outer.get("sha256") == receipt["checkpoint"]["sha256"]
            and outer.get("optimizer_rng_cursor_exact") is True
            and outer.get("immutable_config") == immutable_config(saved["config"])
            and outer.get("training_runtime") == saved.get("training_runtime"),
            "serialized continuous checkpoint contract differs",
        )
        for key in ("step", "epoch", "cursor"):
            require(
                type(counters.get(key)) is int
                and counters[key] >= 0
                and saved.get(key) == outer.get(key) == counters[key],
                "serialized EAGLE cursor differs",
            )
        require(
            type(saved.get("elapsed_seconds")) in (int, float)
            and math.isfinite(saved["elapsed_seconds"])
            and 0 <= saved["elapsed_seconds"] <= elapsed
            and saved.get("tokens") == counters.get("supervised_tokens")
            and set(saved["lanes"]) == set(outer["exports"]) == {lane_name},
            "serialized EAGLE counters/inventory differs",
        )
        for role, filename in (("checkpoint", "joint.npz"), ("manifest", "joint.json")):
            require(
                exported[role]
                == {
                    "path": str(checkpoint.parent / lane_name / filename),
                    "sha256": outer["exports"][lane_name][filename],
                },
                "EAGLE export/outer manifest join differs",
            )
        lane_saved = saved["lanes"][lane_name]
        _rng(saved.get("global_rng"))
        _rng(lane_saved.get("rng"))
        _optimizer(lane_saved.get("optimizer"), counters["step"])
        linears = lane_saved["linears"]
        mapping = CANDIDATE_D_BASE_TO_PATH
        require(
            set(joint.get("projections", {})) == set(mapping)
            and set(linears) == set(mapping.values()),
            "all-nine EAGLE projections required",
        )
        context["outer_manifest"] = pin(outer_path)
    else:
        from w1a1_eagle.block_qat import BlockQATConfig, block_contract

        require(spec["limits"] == limits, "block config training limits differ")
        data_locator = {key: spec["data"][key] for key in ("path", "sha256")}
        files.check(data_locator)
        require(
            source.get("data_manifest_sha256") == data_locator["sha256"],
            "block TRAIN manifest ancestry differs",
        )
        from w1a1_eagle.block_training import BlockCursor

        cursor = BlockCursor(**counters)
        require(
            cursor.stage == "direct"
            and cursor.stage_updates == cursor.step
            and isinstance(cursor.data_cursor, dict)
            and cursor.data_cursor.get("dataset_sha256") == source["data_manifest_sha256"]
            and cursor.data_cursor.get("split") == "train"
            and cursor.data_cursor.get("seed") == spec["qat"].get("seed", 8101),
            "serialized block TRAIN cursor ancestry differs",
        )
        require(
            source.get("synthetic") is False
            and source.get("target_sha256") == target["sha256"]
            and source.get("base_gguf_sha256") == base["sha256"],
            "production block teacher/base differs",
        )
        side_path = checkpoint.with_suffix(".receipt.json")
        # Older non-retention checkpoints have only latest.json; it must still
        # point to this exact serialized endpoint, never an intermediate file.
        if not side_path.exists():
            side_path = checkpoint.parent / "latest.json"
        side = json.loads(side_path.read_text())
        config = BlockQATConfig(**spec["qat"])
        require(
            config.family == family
            and config.activation_bits == bits
            and joint.get("family") == family
            and joint.get("profile") == config.profile,
            "block family/profile/precision differs",
        )
        expected_source = {**source, "bundle_sha256": lane_locator["sha256"]}
        require(
            saved.get("schema") == side.get("schema") == "block_qat_checkpoint_v1"
            and saved.get("source") == expected_source
            and saved.get("contract") == block_contract(config)
            and side.get("source_sha256") == digest(expected_source)
            and side.get("contract_sha256") == digest(saved["contract"])
            and side.get("committed") is True
            and {key: side[key] for key in ("path", "sha256")} == receipt["checkpoint"]
            and saved.get("cursor") == side.get("cursor") == counters,
            "serialized block source/config/cursor differs",
        )
        _rng(saved.get("rng"))
        _optimizer(saved.get("optimizer"), counters["step"])
        linears = saved["linears"]
        mapping = {
            f"blk.{i}.ffn_{part}": f"blk.{i}.ffn_{part}"
            for i in range(5)
            for part in ("gate", "up", "down")
        }
        if config.profile == "ffn15_fusion":
            mapping["fc"] = "fc"
        require(
            set(joint.get("projections", {})) == set(mapping) and set(linears) == set(mapping),
            "block selected projection inventory differs",
        )
        context["checkpoint_sidecar"] = pin(side_path)
    # Prove the serializer arrays actually derive from the committed model.
    import numpy as np

    with np.load(exported["checkpoint"]["path"], allow_pickle=False) as arrays:
        for base_name, saved_name in mapping.items():
            entry = joint["projections"][base_name]
            prefix = entry["checkpoint_name"]
            weights = linears[saved_name]
            latent = weights["latent_sign"].detach().cpu().numpy()
            scale = (
                (weights["initial_scale"] + weights["scale_offset"])
                .clamp_min(0)
                .detach()
                .cpu()
                .numpy()
            )
            require(
                list(latent.shape) == entry["shape"]
                and np.array_equal(arrays[prefix + ".latent"], latent)
                and np.array_equal(arrays[prefix + ".scale"], scale),
                "export arrays differ from serialized trained weights",
            )
    serializer_name = (
        "scripts/export_recurrent_binary.py"
        if family == "eagle"
        else "scripts/export_block_binary.py"
    )
    context["serializer_source"] = lane["source"][serializer_name]
    files.check(context["serializer_source"])
    context["serializer_python"] = plan["python"]
    files.check(plan["python"])
    invocation = Path(plan["python_invocation"])
    require(
        invocation.is_absolute() and invocation.resolve() == Path(plan["python"]["path"]),
        "original serializer Python differs",
    )
    return context


def validate_export(context, receipt_locator, files=None):
    """Accept original EAGLE or new lane export without rewriting either."""
    files = Files() if files is None else files
    result = json.loads(files.check(receipt_locator).read_text())
    require(
        result.get("schema") == "nine_model_lane_endpoint_export_v1"
        and result.get("artifact_kind") == "production"
        and result.get("candidate") == context["lane"]["candidate"]
        and result.get("frozen_lane") == context["frozen_lane"]
        and result.get("training_receipt") == context["train_receipt"]
        and result.get("checkpoint") == context["checkpoint"]
        and result.get("campaign_complete") is False,
        "original lane export join differs",
    )
    audit = json.loads(files.check(result["audit"]).read_text())
    files.check(result["model"])
    family = result["candidate"].split("_")[0]
    expected = context["exported"]
    require(
        audit.get("serialization_audit_passed") is True
        and audit.get("activation_bits") == int(result["candidate"].split("_a")[1])
        and set(audit.get("projections", {}))
        == set(json.loads(files.check(expected["manifest"]).read_text())["projections"])
        and audit.get("output") == result["model"],
        "serializer audit/packed inventory differs",
    )
    if family == "eagle":
        require(
            audit.get("base_gguf") == context["base_model"]
            and audit.get("checkpoint") == expected["checkpoint"]
            and audit.get("checkpoint_manifest") == expected["manifest"],
            "EAGLE serializer input ancestry differs",
        )
    else:
        require(
            audit.get("schema") == "block_binary_export_v1"
            and audit.get("family") == family
            and audit.get("profile") == context["spec"]["qat"]["profile"]
            and audit.get("base_gguf") == {"sha256": context["base_model"]["sha256"]}
            and audit.get("checkpoint") == {"sha256": expected["checkpoint"]["sha256"]}
            and audit.get("manifest") == {"sha256": expected["manifest"]["sha256"]},
            "block serializer input ancestry differs",
        )
    for key in (
        "bundle_sha256",
        "serializer_source",
        "base_model",
        "target",
        "target_policy",
        "training_source",
    ):
        if key in result:
            expected_value = {
                "bundle_sha256": context["frozen_lane"]["sha256"],
                "training_source": context["source_bindings"],
                "target_policy": context["lane"]["target_policy"],
            }.get(key, context.get(key))
            require(result[key] == expected_value, "export provenance differs: " + key)
    return result


def export_endpoint(
    context, directory, *, release_check, cpu_admission, wall_seconds=3600, run=subprocess.run
):
    """Execute pinned CPU serializer after fresh external runtime admissions.

    Callbacks are required trusted code, not JSON booleans. release_check must
    inspect actual owned identities/descendants/device holders and kernel locks;
    cpu_admission must check fresh host RAM/disk for serialized and export peaks.
    Neither callback may stop a healthy trainer. Failed output directories stay.
    """
    require(callable(release_check) and callable(cpu_admission), "fresh runtime callbacks required")
    require(
        type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and wall_seconds > 0,
        "finite CPU export wall bound required",
    )
    require(release_check(context) is True, "fresh actual runtime release required")
    require(
        cpu_admission(context, Path(directory)) is True, "fresh CPU RAM/disk admission required"
    )
    # Re-read and rehash all mutable endpoint inputs immediately before execution.
    context = validate_endpoint(
        context["frozen_lane"],
        context["lane_state"],
        context["supervisor_state"],
        context["train_receipt"],
    )
    directory = Path(directory).absolute()
    require(
        directory == directory.resolve() and not directory.exists() and not directory.is_symlink(),
        "new canonical immutable output directory required",
    )
    directory.mkdir(parents=True, exist_ok=False)
    exported = context["exported"]
    command = [
        context["admission_plan"]["python_invocation"],
        context["serializer_source"]["path"],
        "--base",
        context["base_model"]["path"],
        "--checkpoint",
        exported["checkpoint"]["path"],
        "--manifest",
        exported["manifest"]["path"],
        "--output",
        str(directory / "trained.gguf"),
        "--audit",
        str(directory / "export-audit.json"),
    ]
    env = clean_environment()
    env.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"
    )
    run(command, check=True, stdin=subprocess.DEVNULL, timeout=wall_seconds, env=env)
    context = validate_endpoint(
        context["frozen_lane"],
        context["lane_state"],
        context["supervisor_state"],
        context["train_receipt"],
    )
    require(release_check(context) is True, "actual runtime release changed during CPU export")
    result = {
        "schema": "nine_model_lane_endpoint_export_v1",
        "artifact_kind": "production",
        "candidate": context["lane"]["candidate"],
        "frozen_lane": context["frozen_lane"],
        "training_receipt": context["train_receipt"],
        "checkpoint": context["checkpoint"],
        "model": pin(directory / "trained.gguf"),
        "audit": pin(directory / "export-audit.json"),
        "campaign_complete": False,
        "bundle_sha256": context["frozen_lane"]["sha256"],
        "serializer_source": context["serializer_source"],
        "export_source": pin(__file__),
        "base_model": context["base_model"],
        "target": context["target"],
        "target_policy": context["lane"]["target_policy"],
        "training_source": context["source_bindings"],
        "config": context["lane"]["config"],
        "lane_state": context["lane_state"],
        "supervisor_state": context["supervisor_state"],
        "native_runtime_gate": "PENDING_fresh_load_graph_dispatch",
        "runtime_dense_fallback_checked": False,
    }
    for key in ("outer_manifest", "checkpoint_sidecar"):
        if key in context:
            result[key] = context[key]
    # Validate before committing PASS; a failed serializer/audit leaves its raw
    # files and validation-input.json, never an export receipt.
    validation_path = directory / "validation-input.json"
    atomic_json(validation_path, result)
    validate_export(context, pin(validation_path))
    atomic_json(directory / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lane", "lane-state", "supervisor-state", "train-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    context = validate_endpoint(
        *(
            {"path": str(getattr(args, key)), "sha256": getattr(args, key + "_sha256")}
            for key in ("lane", "lane_state", "supervisor_state", "train_receipt")
        )
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "candidate": context["lane"]["candidate"],
                "execution": False,
                "gpu_queried": False,
                "fresh_release": "PENDING_runtime_callback",
                "native_runtime_gate": "PENDING_fresh_load_graph_dispatch",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
