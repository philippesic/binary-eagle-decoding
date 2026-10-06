#!/usr/bin/env python3
"""Stage one DSpark/DFlash source lane; metadata modes never load a model/GPU.

An explicit future export-initial producer restores the unchanged CUDA checkpoint
under both project GPU locks. Packets stay PENDING for real native admission.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_nine_model_bundle as builder  # noqa: E402
from capture_nine_model_train_data import validate_replay  # noqa: E402
from check_block_capture_portability import NativeCaptureGoldens  # noqa: E402

from w1a1_eagle.block_data import (  # noqa: E402
    DOMAINS,
    SPLITS,
    TAPS,
    BlockDataset,
    crop_decode_history,
    identity,
)
from w1a1_eagle.block_fusion import (  # noqa: E402
    ARITHMETIC,
    FusionFitConfig,
    norm_epsilon_bits,
    validate_norm_descriptor,
)
from w1a1_eagle.block_qat import BlockQATConfig, block_contract  # noqa: E402
from w1a1_eagle.block_training import BlockCursor  # noqa: E402
from w1a1_eagle.nine_model_pipeline import (  # noqa: E402
    CANDIDATES,
    Files,
    LinuxResources,
    atomic_json,
    require,
    require_available,
    resource_gate,
)

INPUT_SCHEMA = "block_lane_packet_inputs_v1"
REQUIRED = {
    "candidate",
    "runtime",
    "budget",
    "data",
    "data_admission",
    "coverage_policy",
    "reference",
    "initializer",
    "calibration_policy",
    "objective",
    "conditioning",
    "checkpoint_every",
    "resource_floors",
}
PROJECTIONS = {"fc"} | {f"blk.{i}.ffn_{part}" for i in range(5) for part in ("gate", "up", "down")}
PRODUCERS = (
    "prepare_block_lane_packet.py",
    "train_nine_model_qat.py",
    "fit_block_fusion.py",
    "extract_block_fusion_reference.py",
    "export_block_binary.py",
    "capture_block_qat_teacher.py",
    "check_block_capture_portability.py",
)
TIMED_PRODUCERS = (
    "evaluate_nine_model_timed_checkpoint.py",
    "export_nine_model_lane_candidate.py",
    "evaluate_nine_model_native.py",
    "run_nine_model_lane_endpoint.py",
    "run_nine_model_lane.py",
)


def read(files, locator):
    return json.loads(files.check(locator).read_text())


def artifact(locator):
    return {key: locator[key] for key in ("path", "sha256")}


def inspect_inputs(value):
    """A draft inspection cannot create a production readiness claim."""
    require(value.get("schema") == INPUT_SCHEMA, "block packet inputs schema differs")
    pending = ["missing " + key for key in sorted(REQUIRED - set(value))]
    for key in (
        "runtime",
        "budget",
        "data",
        "data_admission",
        "coverage_policy",
        "calibration_policy",
    ):
        if key in value and not Path(value[key].get("path", "")).is_file():
            pending.append("missing real artifact: " + key)
    for group, keys in (
        ("reference", ("weights", "norm", "metadata")),
        ("initializer", ("fit_report", "artifact")),
    ):
        for key in keys:
            if not Path(value.get(group, {}).get(key, {}).get("path", "")).is_file():
                pending.append("missing real artifact: " + group + ":" + key)
    pending.extend(
        (
            "actual zero-update CUDA checkpoint/export pending",
            "native five-tap goldens/independent portable QA pending",
            "fresh released-slot native admission pending",
        )
    )
    return {
        "schema": "block_lane_packet_inspection_v1",
        "status": "PENDING",
        "candidate": value.get("candidate"),
        "pending": sorted(set(pending)),
        "model_loaded": False,
        "gpu_queried": False,
        "optimizer_updates": 0,
        "production_ready": False,
        "campaign_complete": False,
        "remaining_candidates": [name for name in CANDIDATES if name != value.get("candidate")],
    }


def authorization(files, value):
    require(
        value.get("kind") == "human_delegated_operational_settings"
        and isinstance(value.get("instruction"), str)
        and value["instruction"].strip(),
        "explicit supplied operational authorization required",
    )
    files.check(value["record"])


def load_data(files, inputs, runtime):
    manifest = read(files, inputs["data"])
    admission = read(files, inputs["data_admission"])
    require(
        admission.get("schema") == "block_data_completed_admission_v1"
        and admission.get("manifest_sha256") == inputs["data"]["sha256"],
        "completed current-host native data admission required",
    )
    dataset = BlockDataset(
        inputs["data"]["path"],
        expected_sha256=inputs["data"]["sha256"],
        admission_path=inputs["data_admission"]["path"],
        admission_sha256=inputs["data_admission"]["sha256"],
        allow_synthetic=False,
    )
    family = inputs["candidate"].split("_")[0]
    require(
        manifest["family"] == family
        and manifest["producer"]["kind"] == "native_target_only"
        and manifest["producer"]["target_precision"] == "F16"
        and manifest["producer"]["target_sha256"] == runtime["inputs"]["target"]["sha256"],
        "original native F16 target/family data required",
    )
    policy = read(files, inputs["coverage_policy"])
    require(
        policy.get("schema") == "block_lane_coverage_policy_v1"
        and policy.get("source_role") == "original_TRAIN",
        "explicit serious original TRAIN coverage policy required",
    )
    authorization(files, policy["authorization"])
    # This floor only excludes the preserved nine-chain development pilot. The
    # adequacy of the supplied larger extent is a root/user research decision.
    minimums = (policy.get("min_unique_train_prompts"), policy.get("min_train_blocks"))
    require(
        all(type(n) is int and n > 9 for n in minimums),
        "coverage policy must explicitly exceed development nine-chain pilot",
    )
    train = [chain for chain in dataset.chains.values() if chain["split"] == "train"]
    extent = {
        "unique_train_prompts": len({chain["prompt_id"] for chain in train}),
        "train_blocks": sum(len(chain["anchors"]) for chain in train),
        "train_potential_teacher_rows": sum(len(chain["anchors"]) * 7 for chain in train),
    }
    require(
        extent["unique_train_prompts"] >= minimums[0] and extent["train_blocks"] >= minimums[1],
        "actual TRAIN extent below supplied serious coverage policy; development pilot refused",
    )
    for split in SPLITS:
        require(
            {c["domain"] for c in dataset.chains.values() if c["split"] == split} == set(DOMAINS),
            "each supplied TRAIN-derived role requires three domains",
        )
    for native in dataset.native_receipts.values():
        buffers = native.get("executed_result_buffers", [])
        require(
            buffers
            and all(
                isinstance(b, str) and b.startswith("CUDA") and b[4:].isdigit() for b in buffers
            ),
            "development CPU teacher corpus cannot stage a production block lane",
        )
    if inputs["objective"] == "full_probability_l1" and inputs["conditioning"] == "captured_prefix":
        require(
            all(dataset._arrays[c["chain_id"]][2] is not None for c in train),
            "offline exact probability L1 requires all selected TRAIN anchor teachers",
        )
    return dataset, extent


def validate_reference(files, inputs, runtime):
    refs = inputs["reference"]
    metadata = read(files, refs["metadata"])
    for key in ("weights", "norm"):
        files.check(refs[key])
    epsilon = validate_norm_descriptor(
        metadata,
        family=inputs["candidate"].split("_")[0],
        weights_sha256=refs["weights"]["sha256"],
        norm_sha256=refs["norm"]["sha256"],
        supplied_epsilon=metadata["epsilon"],
    )
    require(
        metadata["model_sha256"] == runtime["base_model"]["sha256"]
        and metadata["fc_source"]["kind"] == "BF16",
        "original BF16 block base/reference FC binding required",
    )
    return metadata, epsilon


def validate_initializer(files, inputs, metadata, dataset):
    bits = int(inputs["candidate"][-1])
    report = read(files, inputs["initializer"]["fit_report"])
    locator = inputs["initializer"]["artifact"]
    files.check(locator)
    require(
        report.get("schema") == "block_fusion_fit_v1"
        and set(report.get("config", {})) == {field.name for field in fields(FusionFitConfig)},
        "actual canonical block fusion fit report required",
    )
    config = FusionFitConfig(**report["config"])
    policy = read(files, inputs["calibration_policy"])
    require(
        policy.get("schema") == "block_lane_calibration_policy_v1",
        "explicit supplied calibration bounds required",
    )
    authorization(files, policy["authorization"])
    for key in ("rows_per_chain", "max_total_rows", "max_array_bytes"):
        require(
            type(policy.get(key)) is int and policy[key] > 0,
            "positive supplied calibration bound required",
        )
    require(
        config.activation_bits == bits
        and config.zero_scale_orientation_rescue is False
        and config.max_coordinate_flips_per_row == 0
        and config.latent_initialization == "preserve_reference_magnitudes"
        and config.reference_kind == "block_source_weight_magnitudes"
        and config.max_seconds == policy["max_seconds"]
        and report["fit"]["events"] == []
        and report["fit"]["arithmetic"] == ARITHMETIC[bits],
        "separate fixed-arithmetic A8/direct A1 scale-only fit required; probes off",
    )
    require(
        report["data_sha256"] == inputs["data"]["sha256"]
        and report["weights_sha256"] == metadata["weights_sha256"]
        and report["norm_sha256"] == metadata["norm_sha256"]
        and report["reference_metadata_sha256"] == inputs["reference"]["metadata"]["sha256"]
        and report["original_model_sha256"] == metadata["model_sha256"]
        and report["norm_epsilon_f32_bits"] == norm_epsilon_bits(metadata["epsilon"])
        and norm_epsilon_bits(report["norm_epsilon"]) == metadata["epsilon_f32_bits"]
        and report["artifact_sha256"] == locator["sha256"],
        "fit original model/data/FC/gamma/exact epsilon/artifact pins differ",
    )
    rows, columns = metadata["fc_source"]["shape"]
    estimated = 4 * rows * columns * 6 + policy["max_total_rows"] * (columns * 32 + rows * 32)
    require(
        report.get("estimated_array_bytes") == estimated
        and estimated <= policy["max_array_bytes"]
        and report.get("teacher_arithmetic")
        == "original_promoted_f32_weight_cpu_blas_raw_features",
        "actual calibration workspace/reference arithmetic differs from supplied policy",
    )
    contract = report["fit"]["latent_initialization"]
    require(
        contract["policy"] == "preserve_reference_magnitudes"
        and contract["reference_kind"] == "block_source_weight_magnitudes"
        and contract["reference_shape"] == metadata["fc_source"]["shape"]
        and contract["reference_sha256_rule"]
        == "contiguous_little_endian_f32_absolute_initializer_bytes",
        "authentic original weight-magnitude initializer contract required",
    )
    for role in ("fit_rows", "validation_rows"):
        require(
            {r["domain"] for r in report[role]} == set(DOMAINS),
            "fit and validation require independently selected TRAIN domains",
        )
    require(
        {r["prompt_sha256"] for r in report["fit_rows"]}.isdisjoint(
            {r["prompt_sha256"] for r in report["validation_rows"]}
        ),
        "fusion fit/validation prompts must remain disjoint",
    )
    for role, split in (
        ("fit_rows", "calibration_fit"),
        ("validation_rows", "calibration_validation"),
    ):
        expected = []
        total = 0
        for cid in sorted(dataset.chains):
            chain = dataset.chains[cid]
            if chain["split"] != split:
                continue
            import numpy as np

            stop = chain["anchors"][-1]
            count = min(stop, policy["rows_per_chain"])
            positions = np.unique(np.linspace(0, stop - 1, count).astype(int)).tolist()
            expected.append(
                {
                    "chain_id": cid,
                    "prompt_id": chain["prompt_id"],
                    "prompt_sha256": chain["prompt_sha256"],
                    "domain": chain["domain"],
                    "positions": positions,
                }
            )
            total += len(positions)
        require(
            total <= policy["max_total_rows"] and report[role] == expected,
            "fit/validation source rows differ from actual calibration selector and supplied caps",
        )
    return locator | {
        "activation_bits": bits,
        "encoding": "policy_latents",
        "latent_initialization": {
            key: contract[key] for key in ("policy", "reference_kind", "reference_sha256")
        },
    }


def live_teacher(files, inputs, runtime):
    if inputs["conditioning"] != "native_greedy":
        require(
            "teacher" not in inputs and "live_teacher_admission" not in inputs,
            "offline captured_prefix cannot silently attach a live teacher",
        )
        return None
    require(
        inputs["candidate"].startswith("dspark_")
        and "teacher" in inputs
        and "live_teacher_admission" in inputs,
        "native_greedy requires explicit admitted current-prefix DSpark teacher",
    )
    teacher = inputs["teacher"]
    require(
        set(teacher)
        == {
            "binary",
            "target",
            "target_sha256",
            "max_tokens",
            "gpu_layers",
            "producer_source_revision",
        },
        "use actual native_teacher source kwargs, without aliases",
    )
    require(
        teacher["binary"] == runtime["inputs"]["teacher_binary"]["path"]
        and teacher["target"] == runtime["inputs"]["target"]["path"]
        and teacher["target_sha256"] == runtime["inputs"]["target"]["sha256"]
        and teacher["producer_source_revision"] == runtime["native_source_revision"]
        and type(teacher["max_tokens"]) is int
        and 0 < teacher["max_tokens"] <= 32768
        and type(teacher["gpu_layers"]) is int
        and 0 < teacher["gpu_layers"] <= 999,
        "explicit current-prefix teacher target/native/token bounds differ",
    )
    gate = read(files, inputs["live_teacher_admission"])
    require(
        gate.get("schema") == "nine_model_capture_portability_v1"
        and gate.get("status") == "PASS"
        and gate.get("artifact_kind") == "production"
        and gate.get("manifest_sha256") == inputs["data"]["sha256"]
        and gate.get("target_sha256") == teacher["target_sha256"]
        and gate.get("native_binary_sha256") == runtime["inputs"]["teacher_binary"]["sha256"]
        and gate.get("native_source_revision") == teacher["producer_source_revision"]
        and gate.get("teacher_client_source_sha256")
        == builder.sha256(ROOT / "scripts/capture_block_qat_teacher.py")
        and gate.get("compute_capability") == [12, 0]
        and gate.get("producer_closed") is True
        and gate.get("producer_returncode") == 0,
        "actual current native teacher portability admission required",
    )
    require(
        gate.get("family") == "dspark"
        and gate.get("tap_ids") == list(TAPS)
        and gate.get("gate_source_sha256")
        == builder.sha256(ROOT / "scripts/check_block_capture_portability.py")
        and gate.get("device", {}).get("uuid") == runtime["gpu_uuid"],
        "current-prefix portable gate family/taps/source/device differs",
    )
    checks = gate.get("numeric_checks", [])
    require(
        {row.get("domain") for row in checks} == set(DOMAINS)
        and all(
            row.get("decision_changed") is False
            and row.get("features", {}).get("status")
            == row.get("full_vocab_logits", {}).get("status")
            == "PASS"
            for row in checks
        ),
        "current-prefix native numeric/decision evidence incomplete",
    )
    return teacher


def prepare(inputs, output):
    output = output.resolve()
    require(
        inputs.get("schema") == INPUT_SCHEMA and REQUIRED <= set(inputs),
        "complete explicit block packet inputs required",
    )
    require(not output.exists(), "preserve packet history")
    name = inputs["candidate"]
    require(
        name in {f"{f}_a{b}" for f in ("dspark", "dflash") for b in (1, 8)},
        "one direct block candidate required",
    )
    family, precision = name.split("_")
    bits = int(precision[1:])
    require(
        inputs["conditioning"] in ("captured_prefix", "native_greedy"),
        "explicit offline captured_prefix or admitted live current-prefix semantics required",
    )
    require(
        inputs["objective"] == ("full_probability_l1" if family == "dspark" else "hard_ce"),
        "DSpark requires explicit full_probability_l1; DFlash source objective is hard_ce",
    )
    files = Files()
    runtime = read(files, inputs["runtime"])
    require(
        runtime.get("schema") == "block_lane_packet_runtime_v1"
        and len(runtime["native_source_revision"]) == 40
        and all(c in "0123456789abcdef" for c in runtime["native_source_revision"]),
        "exact block runtime/native source metadata required",
    )
    for role in ("target", "teacher_binary", "backend_binary", "block_native_binary"):
        files.opaque(runtime["inputs"][role])
    files.opaque(runtime["base_model"])
    budget = read(files, inputs["budget"])
    builder.selected_budget(budget, (name,), staged=True)
    require(
        budget.get("schema") == "nine_model_selected_budget_v1",
        "supplied single-candidate operational budget required",
    )
    execution_controls = builder.selected_execution_controls(
        {**inputs, "profile": "fixed_reference" if bits == 8 else "direct_a1"},
        family,
        budget["candidates"][name]["training_limits"],
        files,
    )
    dataset, extent = load_data(files, inputs, runtime)
    reference, epsilon = validate_reference(files, inputs, runtime)
    require(
        reference["fc_source"]["shape"] == [dataset.target_width, 5 * dataset.target_width],
        "five-tap reference/data width differs",
    )
    initializer = validate_initializer(files, inputs, reference, dataset)
    teacher = live_teacher(files, inputs, runtime)
    if teacher is not None:
        require(
            teacher["max_tokens"]
            >= max(len(n["tokens"]) for n in dataset.native_receipts.values()),
            "current-prefix teacher token bound cannot truncate full native context",
        )
    overrides = copy.deepcopy(inputs.get("qat_overrides", {}))
    forbidden = {
        "family",
        "activation_bits",
        "profile",
        "objective",
        "conditioning",
        "latent_initialization",
        "initialization_encoding",
        "optimizer_backend",
        "norm_eps",
    }
    require(
        not set(overrides) & forbidden,
        "recipe/precision/objective/probe controls cannot be overridden",
    )
    qat = asdict(
        BlockQATConfig(
            **overrides,
            family=family,
            activation_bits=bits,
            profile="ffn15_fusion",
            objective=inputs["objective"],
            conditioning=inputs["conditioning"],
            norm_eps=epsilon,
            latent_initialization="preserve_reference_magnitudes",
            initialization_encoding="policy_latents",
            optimizer_backend="serial",
        )
    )
    require(
        qat["hidden_size"] == dataset.target_width
        and qat["vocab_size"] == dataset.vocab_size
        and qat["mask_token_id"] == dataset.manifest["mask_token_id"],
        "actual released model/config/data geometry differs",
    )
    descriptor = {
        "schema": "nine_model_lane_inputs_v1",
        "budget": inputs["budget"],
        "candidates": {
            name: {
                "profile": "fixed_reference" if bits == 8 else "direct_a1",
                "initialization": initializer,
                "fusion_calibration": artifact(initializer),
                "base_model": runtime["base_model"],
                "data": inputs["data"],
                "data_admission": inputs["data_admission"],
                "deployment_coverage": {"profile": "ffn15_fusion"},
                "qat_overrides": qat,
                "checkpoint_every": inputs["checkpoint_every"],
                "resource_floors": inputs["resource_floors"],
                **execution_controls,
                "native_markers": [
                    "CUDA packed W1A8 INT8 dispatch"
                    if bits == 8
                    else "CUDA packed W1A1 XOR/POPCOUNT dispatch"
                ],
                **({"teacher": teacher} if teacher is not None else {}),
            }
        },
        "inputs": runtime["inputs"],
        "gpu_uuid": runtime["gpu_uuid"],
        "gpu_control_path": runtime["gpu_control_path"],
        "resource_policy": runtime["resource_policy"],
        "environment": runtime.get("environment", {}),
        "preflight": {
            "native_source_revision": runtime["native_source_revision"],
            "portability": {},
            "build_provenance": runtime.get("build_provenance", {}),
        },
    }
    output.mkdir(parents=True)
    atomic_json(output / "packet-inputs.json", inputs)
    builder.materialize_configs(descriptor, output / "configs")
    source = {
        str((ROOT / "scripts" / producer).relative_to(ROOT)): builder.pin(
            ROOT / "scripts" / producer
        )
        for producer in PRODUCERS
    }
    request = {
        "schema": "block_lane_initial_preparation_v1",
        "candidate": name,
        "config": builder.pin(output / f"configs/{name}.json"),
        "source": source,
        "input_descriptor": builder.pin(output / "packet-inputs.json"),
        "optimizer_updates": 0,
    }
    atomic_json(output / "initial-preparation-request.json", request)
    requests, cases = [], []
    maximum = runtime["golden_max_tokens"]
    require(
        type(maximum) is int and 1 <= maximum <= 32768,
        "explicit bounded five-tap golden prefix cap required",
    )
    for domain in DOMAINS:
        cid = next(
            cid
            for cid in sorted(dataset.chains)
            if dataset.chains[cid]["split"] == "train" and dataset.chains[cid]["domain"] == domain
        )
        chain, native = dataset.chains[cid], dataset.native_receipts[cid]
        tokens = native["tokens"][:maximum]
        require(
            chain["prompt_length"] <= len(tokens),
            "golden cap cannot truncate original prompt boundary",
        )
        ancestry = {
            "prompt_id": chain["prompt_id"],
            "prompt_sha256": chain["prompt_sha256"],
            "domain": domain,
            "source_split": "TRAIN",
            "prompt_length": chain["prompt_length"],
        }
        requests.append(
            {
                "tokens": tokens,
                "tap_ids": list(TAPS),
                "logits_mode": "last",
                "chain_ancestry": ancestry,
                "decode_history": crop_decode_history(native["decode_history"], len(tokens)),
            }
        )
        cases.append(
            {
                "chain_id": cid,
                "source_native_receipt": chain["native_receipt"],
                "ancestry": ancestry,
            }
        )
    (output / "golden-replay-requests.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests)
    )
    atomic_json(
        output / "golden-source-joins.json",
        {
            "schema": "block_lane_golden_source_joins_v1",
            "cases": cases,
            "requests": builder.pin(output / "golden-replay-requests.jsonl"),
            "runtime": inputs["runtime"],
            "data": inputs["data"],
            "data_admission": inputs["data_admission"],
        },
    )
    atomic_json(
        output / "effective-controls.json",
        {
            "schema": "block_lane_selected_controls_v1",
            "candidate": name,
            "qat": qat,
            "coverage": extent,
            "coverage_policy": inputs["coverage_policy"],
            "budget": inputs["budget"],
            "calibration_policy": inputs["calibration_policy"],
            **execution_controls,
            "status": "PENDING",
            "production_ready": False,
            "model_loaded": False,
            "gpu_queried": False,
            "optimizer_updates": 0,
            "remaining_candidates": [other for other in CANDIDATES if other != name],
        },
    )
    commands = source_commands(output, inputs, runtime, request)
    atomic_json(output / "commands.json", commands)
    return inspect_inputs(inputs) | {
        "packet": str(output),
        "source_configs": builder.pin(output / "configs/source-configs.json"),
    }


def source_commands(output, inputs, runtime, request):
    name, refs = inputs["candidate"], inputs["reference"]
    calibration = read(Files(), inputs["calibration_policy"])
    python, script = sys.executable, str(ROOT / "scripts/prepare_block_lane_packet.py")
    fit = [python, str(ROOT / "scripts/fit_block_fusion.py")]
    for flag, value in {
        "manifest": inputs["data"]["path"],
        "manifest-sha256": inputs["data"]["sha256"],
        "admission": inputs["data_admission"]["path"],
        "admission-sha256": inputs["data_admission"]["sha256"],
        "weights": refs["weights"]["path"],
        "weights-sha256": refs["weights"]["sha256"],
        "norm": refs["norm"]["path"],
        "norm-sha256": refs["norm"]["sha256"],
        "norm-metadata": refs["metadata"]["path"],
        "norm-metadata-sha256": refs["metadata"]["sha256"],
        "norm-epsilon": read(Files(), refs["metadata"])["epsilon"],
        "activation-bits": name[-1],
        "rows-per-chain": calibration["rows_per_chain"],
        "max-total-rows": calibration["max_total_rows"],
        "max-array-bytes": calibration["max_array_bytes"],
        "max-seconds": calibration["max_seconds"],
        "latent-initialization": "preserve_reference_magnitudes",
        "output-dir": str(Path(inputs["initializer"]["fit_report"]["path"]).parent),
    }.items():
        fit.extend(["--" + flag, str(value)])
    target = runtime["inputs"]["target"]
    return {
        "schema": "block_lane_source_commands_v1",
        "execution_authorized_by_this_file": False,
        "status": "PENDING_released_5080_and_real_inputs",
        "source": request["source"],
        "scale_only_fit": fit,
        "initial_prepare": [
            python,
            str(ROOT / "scripts/train_nine_model_qat.py"),
            "--config",
            request["config"]["path"],
            "--run-dir",
            str(output / "initial-prepare"),
            "--bundle-sha256",
            builder.sha256(output / "initial-preparation-request.json"),
            "--stage-name",
            name + "/initial-prepare",
            "--completion-output",
            str(output / "initial-prepare-receipt.json"),
            "--allow-cuda",
            "--prepare-only",
        ],
        "initial_export": [
            python,
            script,
            "export-initial",
            "--packet",
            str(output),
            "--availability",
            "<fresh-root-provided-5080-lease>",
            "--allow-cuda",
        ],
        "native_goldens": [
            python,
            str(ROOT / "scripts/capture_block_qat_teacher.py"),
            "--binary",
            runtime["inputs"]["teacher_binary"]["path"],
            "--target",
            target["path"],
            "--target-sha256",
            target["sha256"],
            "--producer-source-revision",
            runtime["native_source_revision"],
            "--max-tokens",
            str(runtime["golden_max_tokens"]),
            "--gpu-layers",
            "999",
            "--output-root",
            str(output / "native-goldens"),
            "--requests",
            str(output / "golden-replay-requests.jsonl"),
        ],
        "bind": [
            python,
            script,
            "bind",
            "--packet",
            str(output),
            "--receipts",
            "<actual-native-receipts.jsonl>",
            "--receipts-sha256",
            "<actual-sha256>",
            "--qa-ledger",
            "<independent-selected-lane-qa.json>",
            "--qa-ledger-sha256",
            "<actual-sha256>",
        ],
        "admission_plan": [
            python,
            str(ROOT / "scripts/prepare_nine_model_bundle.py"),
            "--inputs",
            str(output / "production-inputs.json"),
            "--materialize-admission-plan",
            str(output / "admission-plan.json"),
        ],
        "finalize_lane": [
            python,
            script,
            "finalize",
            "--packet",
            str(output),
            "--admission-plan",
            str(output / "admission-plan.json"),
            "--admission-plan-sha256",
            "<actual-materialized-plan-sha256>",
            "--output",
            str(output / "lane.json"),
        ],
        "notes": [
            "fit precedes prepare; existing fit output cannot be overwritten",
            "initial-prepare, export and native commands require sole coordinated released GPU",
            "bind stages only one candidate; actual prelaunch admission/build_lane remain separate",
            "export-initial preserves CUDA runtime; same-device source checkpoint restore required",
        ],
    }


def checkpoint_join(packet, files):
    packet = packet.resolve()
    request_pin = builder.pin(packet / "initial-preparation-request.json")
    request = read(files, request_pin)
    for locator in request["source"].values():
        files.check(locator)
    require(
        request["source"]
        == {"scripts/" + name: builder.pin(ROOT / "scripts" / name) for name in PRODUCERS},
        "initial producer source inventory changed",
    )
    descriptor = read(files, builder.pin(packet / "configs/resolved-inputs.json"))
    require(
        request["input_descriptor"] == builder.pin(packet / "packet-inputs.json"),
        "initial packet inputs changed after source configuration",
    )
    files.check(request["input_descriptor"])
    (candidate,) = builder.selected_candidates(descriptor)
    spec = importlib.import_module("train_nine_model_qat").load_spec(files.check(request["config"]))
    require(
        candidate == request["candidate"] == spec["candidate"]
        and spec.get("precision_stage") == "direct",
        "initial candidate/direct config differs",
    )
    require(
        descriptor["candidates"][candidate]["config"] == request["config"],
        "initial request differs from selected source config",
    )
    config = BlockQATConfig(**spec["qat"])
    source = {
        "base_gguf_sha256": spec["model"]["sha256"],
        "data_manifest_sha256": spec["data"]["sha256"],
        "bundle_sha256": request_pin["sha256"],
        "synthetic": False,
        "target_sha256": descriptor["inputs"]["target"]["sha256"],
    }
    receipt_pin = builder.pin(packet / "initial-prepare-receipt.json")
    receipt = read(files, receipt_pin)
    require(
        receipt.get("schema") == "nine_model_preparation_v1"
        and receipt.get("status") == "PASS"
        and receipt.get("optimizer_updates") == 0,
        "actual successful zero-update block preparation receipt required",
    )
    checkpoint = receipt["checkpoint"]
    cursor = BlockCursor(**checkpoint["cursor"])
    data = spec["data"]
    dataset = BlockDataset(
        data["path"],
        expected_sha256=data["sha256"],
        admission_path=data["admission"]["path"],
        admission_sha256=data["admission"]["sha256"],
    )
    require(
        cursor.data_cursor == dataset.cursor(seed=config.seed).payload(),
        "initial checkpoint exact dataset/order/position cursor differs",
    )
    require(
        cursor.step
        == cursor.epoch
        == cursor.block_index
        == cursor.supervised_tokens
        == cursor.presented_tokens
        == cursor.stage_updates
        == cursor.stage_supervised_tokens
        == 0
        and cursor.elapsed_seconds == 0
        and cursor.stage == "direct"
        and cursor.unique_blocks == (),
        "positive/mixed/charged checkpoint cannot be initial zero-update state",
    )
    require(
        Path(checkpoint["path"]).resolve().parent == packet / "initial-prepare/checkpoints"
        and checkpoint.get("schema") == "block_qat_checkpoint_v1"
        and checkpoint.get("committed") is True
        and checkpoint["contract_sha256"] == identity(block_contract(config))
        and checkpoint["source_sha256"] == identity(source),
        "strict initial checkpoint source/contract/path/commit binding differs",
    )
    files.check(artifact(checkpoint))
    smoke = receipt["smoke"]
    require(
        smoke.get("contract") == block_contract(config)
        and smoke.get("selected_projection_count") == 16
        and smoke.get("hard_forward") is True
        and smoke.get("optimizer_updates") == smoke.get("optimizer_moment_tensors") == 0,
        "actual sixteen-projection zero-state hard-forward smoke required",
    )
    require(
        receipt["smoke_contract"]
        == importlib.import_module("train_nine_model_qat").completed_zero_update_smoke_contract(
            [config.activation_bits], []
        ),
        "actual zero-update arithmetic/optimizer smoke contract differs",
    )
    return spec, source, checkpoint, request_pin, receipt_pin


def protected_names(family):
    names = {"token_embd.weight", "output.weight", "enc.output_norm.weight", "output_norm.weight"}
    names.update(
        f"blk.{i}.{part}.weight"
        for i in range(5)
        for part in (
            "attn_norm",
            "attn_q_norm",
            "attn_k_norm",
            "ffn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
        )
    )
    if family == "dspark":
        names.update(("markov_w1.weight", "markov_w2.weight"))
    return names


def validate_initial_export(
    files, export, spec, source, checkpoint, request_pin, receipt_pin, reference
):
    require(
        export.get("schema") == "block_lane_initial_export_v1"
        and export.get("status") == "PASS"
        and export.get("optimizer_updates") == 0
        and export.get("checkpoint") == artifact(checkpoint)
        and export.get("initial_request") == request_pin
        and export.get("preparation_receipt") == receipt_pin
        and export.get("source") == source
        and export.get("config")
        == builder.pin(Path(request_pin["path"]).parent / f"configs/{spec['candidate']}.json"),
        "initial export/zero checkpoint/config/source joins differ",
    )
    manifest = read(files, export["manifest"])
    serializer = importlib.import_module("export_block_binary")
    serializer.check_manifest(manifest, source["base_gguf_sha256"], export["npz"]["sha256"])
    require(
        manifest["activation_bits"] == spec["qat"]["activation_bits"]
        and manifest["family"] == spec["family"]
        and manifest["profile"] == "ffn15_fusion"
        and set(manifest["projections"]) == PROJECTIONS,
        "exact sixteen-projection native export arithmetic differs",
    )
    h, intermediate = spec["qat"]["hidden_size"], spec["qat"]["intermediate_size"]
    for name, record in manifest["projections"].items():
        expected = (
            [h, 5 * h]
            if name == "fc"
            else [h, intermediate]
            if name.endswith("down")
            else [intermediate, h]
        )
        require(
            record == {"checkpoint_name": name, "shape": expected},
            "actual producer projection names/shapes differ from selected config",
        )
    files.check(export["npz"])
    files.check(export["model"])
    audit = read(files, export["audit"])
    require(
        audit.get("schema") == "block_binary_export_v1"
        and audit.get("serialization_audit_passed") is True
        and audit.get("family") == spec["family"]
        and audit.get("profile") == "ffn15_fusion"
        and audit.get("activation_bits") == spec["qat"]["activation_bits"]
        and audit.get("base_gguf") == {"sha256": source["base_gguf_sha256"]}
        and audit.get("checkpoint") == {"sha256": export["npz"]["sha256"]}
        and audit.get("manifest") == {"sha256": export["manifest"]["sha256"]}
        and audit.get("output") == export["model"]
        and set(audit.get("projections", {})) == PROJECTIONS,
        "actual block serializer audit source/NPZ/manifest/16-projection pins differ",
    )
    for name, row in audit["projections"].items():
        require(
            row.get("shape") == manifest["projections"][name]["shape"]
            and all(
                isinstance(row.get(key), str)
                and len(row[key]) == 64
                and all(c in "0123456789abcdef" for c in row[key])
                for key in ("latent_sha256", "scale_sha256", "packed_sha256")
            ),
            "serializer packed/latent/scale tensor audit differs",
        )
    preserved = audit.get("preserved_tensors", {})
    require(
        protected_names(spec["family"]) <= set(preserved)
        and all(
            row.get("type") in ("F32", "F16", "BF16")
            and isinstance(row.get("raw_sha256"), str)
            and len(row["raw_sha256"]) == 64
            for row in preserved.values()
        )
        and preserved["enc.output_norm.weight"]["raw_sha256"]
        == reference["norm_source"]["payload_sha256"],
        "protected full private tensors/norm gamma payload proof absent",
    )


def bind(args):
    packet = args.packet
    require(not (packet / "production-inputs.json").exists(), "preserve prior selected packet bind")
    files = Files()
    inputs = read(files, builder.pin(packet / "packet-inputs.json"))
    runtime = read(files, inputs["runtime"])
    spec, source, checkpoint, request_pin, receipt_pin = checkpoint_join(packet, files)
    reference, _ = validate_reference(files, inputs, runtime)
    export = read(files, builder.pin(packet / "initial-export-receipt.json"))
    validate_initial_export(
        files, export, spec, source, checkpoint, request_pin, receipt_pin, reference
    )
    joins = read(files, builder.pin(packet / "golden-source-joins.json"))
    require(
        joins["data"] == inputs["data"]
        and joins["data_admission"] == inputs["data_admission"]
        and joins["runtime"] == inputs["runtime"],
        "golden source packet/model/data/runtime join differs",
    )
    dataset, _ = load_data(files, inputs, runtime)
    receipts_pin = {"path": str(args.receipts.resolve()), "sha256": args.receipts_sha256}
    receipts_path = files.check(receipts_pin)
    requests = [
        json.loads(line) for line in files.check(joins["requests"]).read_text().splitlines()
    ]
    receipts = [json.loads(line) for line in receipts_path.read_text().splitlines() if line.strip()]
    require(
        len(receipts) == len(requests) == len(joins["cases"]) == 3,
        "exact one five-tap native golden per original TRAIN domain required",
    )
    native = {
        "binary": runtime["inputs"]["teacher_binary"],
        "target": runtime["inputs"]["target"],
        "source_revision": runtime["native_source_revision"],
        "client_source": builder.pin(ROOT / "scripts/capture_block_qat_teacher.py"),
    }
    for receipt, request, case in zip(receipts, requests, joins["cases"], strict=True):
        require(
            all(receipt.get(key) == value for key, value in request.items())
            and receipt["chain_ancestry"] == case["ancestry"],
            "golden native tokens/absolute history/taps/TRAIN source differs",
        )
        chain = dataset.chains.get(case["chain_id"])
        require(
            chain is not None
            and chain["split"] == "train"
            and chain["native_receipt"] == case["source_native_receipt"],
            "golden parent is not the selected original TRAIN data chain",
        )
        parent = read(files, case["source_native_receipt"])
        prefix = parent["tokens"][: runtime["golden_max_tokens"]]
        require(
            request["tokens"] == prefix
            and request["decode_history"]
            == crop_decode_history(parent["decode_history"], len(prefix))
            and case["ancestry"] == parent["chain_ancestry"],
            "golden exact source parent tokens/history/ancestry changed",
        )
        validate_replay(receipt, request["tokens"], TAPS, native, {"name": runtime["device_name"]})
    qa_pin = {"path": str(args.qa_ledger.resolve()), "sha256": args.qa_ledger_sha256}
    ledger = read(files, qa_pin)
    pending = builder.ledger_pending(ledger, (spec["candidate"],)) + builder.source_pending(ledger)
    require(
        ledger.get("source_files", {}).get("scripts/prepare_block_lane_packet.py")
        == builder.sha256(Path(__file__))
        and not pending,
        "independent selected portable QA/current adapter source pending: " + "; ".join(pending),
    )
    require(
        not any((packet / name).exists() for name in ("train-inventory.json", "block-goldens.json"))
        and not any((packet / f"golden-{i}.json").exists() for i in range(3)),
        "preserve previous golden bind history",
    )
    inventory = {
        "schema": "block_train_inventory_v1",
        "prompts": {
            case["ancestry"]["prompt_id"]: {
                "sha256": case["ancestry"]["prompt_sha256"],
                "domain": case["ancestry"]["domain"],
                "split": "TRAIN",
            }
            for case in joins["cases"]
        },
    }
    atomic_json(packet / "train-inventory.json", inventory)
    cases = []
    for index, receipt in enumerate(receipts):
        path = packet / f"golden-{index}.json"
        atomic_json(path, receipt)
        cases.append(
            {"chain_id": joins["cases"][index]["chain_id"], "native_receipt": builder.pin(path)}
        )
    atomic_json(
        packet / "block-goldens.json",
        {
            "schema": "nine_model_train_capture_goldens_v1",
            "family": spec["family"],
            "tap_ids": list(TAPS),
            "vocab_size": spec["qat"]["vocab_size"],
            "target_width": spec["qat"]["hidden_size"],
            "target_sha256": source["target_sha256"],
            "train_inventory": builder.pin(packet / "train-inventory.json"),
            "cases": cases,
        },
    )
    NativeCaptureGoldens(
        packet / "block-goldens.json", expected_sha256=builder.sha256(packet / "block-goldens.json")
    )
    descriptor = read(files, builder.pin(packet / "configs/resolved-inputs.json"))
    descriptor["candidates"][spec["candidate"]].update(
        initial_model=export["model"], initial_export_audit=export["audit"]
    )
    descriptor["qa_ledger"] = qa_pin
    descriptor["preflight"]["portability"][spec["family"]] = {
        "golden_manifest": builder.pin(packet / "block-goldens.json"),
        "max_tokens": runtime["golden_max_tokens"],
    }
    descriptor["preflight"]["preparation_provenance"] = {
        "input_descriptor": builder.pin(packet / "packet-inputs.json"),
        "initial_request": request_pin,
        "initial_zero_update_receipt": receipt_pin,
        "initial_export_receipt": builder.pin(packet / "initial-export-receipt.json"),
        "original_train_prompt_joins": builder.pin(packet / "golden-source-joins.json"),
        "native_golden_receipts": receipts_pin,
    }
    atomic_json(packet / "production-inputs.json", descriptor)
    return {
        "inputs": builder.pin(packet / "production-inputs.json"),
        "status": "PENDING_fresh_native_admission",
        "production_ready": False,
        "campaign_complete": False,
        "optimizer_updates": 0,
    }


def finalize(args):
    """Reuse the actual single-lane builder; fresh device execution remains pending."""
    files = Files()
    descriptor = read(files, builder.pin(args.packet / "production-inputs.json"))
    plan = {"path": str(args.admission_plan.resolve()), "sha256": args.admission_plan_sha256}
    files.check(plan)
    timed_path = getattr(args, "timed_evaluation_plan", None)
    timed_sha = getattr(args, "timed_evaluation_plan_sha256", None)
    require(
        (timed_path is None) == (timed_sha is None),
        "timed evaluation plan requires both path and SHA256",
    )
    if timed_path is not None:
        timed_locator = {"path": str(timed_path.resolve()), "sha256": timed_sha}
        timed_plan = read(files, timed_locator)
        source = timed_plan.get("source")
        require(isinstance(source, dict), "timed evaluation plan source inventory required")
        for producer in TIMED_PRODUCERS:
            require(
                builder.pin(ROOT / "scripts" / producer) in source.values(),
                "timed evaluation plan lacks current producer source pin: " + producer,
            )
        descriptor["timed_evaluation_plan"] = timed_locator
    require(
        not (args.packet / "lane-inputs.json").exists() and not args.output.exists(),
        "preserve finalized source lane history",
    )
    descriptor["inputs"]["admission_plan"] = plan
    build_lane = importlib.import_module("prepare_nine_model_lane").build_lane
    # build_lane validates current config/model/export/budget/QA/source with the
    # canonical admission plan; no model or accelerator is constructed.
    result = build_lane(descriptor, args.output.resolve())
    atomic_json(args.packet / "lane-inputs.json", descriptor)
    return {
        "status": "PENDING_fresh_native_admission",
        "source_lane": result,
        "production_ready": False,
        "campaign_complete": False,
        "optimizer_updates": 0,
    }


def export_initial(args):
    """Future same-runtime restore: never rewrite a CUDA header/config to CPU."""
    require(
        args.allow_cuda,
        "explicit --allow-cuda required; metadata packet never authorizes execution",
    )
    packet, files = args.packet, Files()
    require(
        not (packet / "initial-export").exists()
        and not (packet / "initial-export-receipt.json").exists(),
        "preserve zero-update export history",
    )
    spec, source, checkpoint, request_pin, receipt_pin = checkpoint_join(packet, files)
    inputs = read(files, builder.pin(packet / "packet-inputs.json"))
    runtime = read(files, inputs["runtime"])
    lease = require_available(args.availability, request_pin["sha256"])
    require(lease["gpu_uuid"] == runtime["gpu_uuid"], "initial export physical device differs")
    api = importlib.import_module("train_continuous_w1ax")
    trainer = importlib.import_module("train_nine_model_qat")
    from w1a1_eagle.block_qat import block_optimizer
    from w1a1_eagle.block_training import export_block_checkpoint, load_block_checkpoint

    # Fixed nonblocking order matches project global serialization first.
    global_path = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with global_path.open("a") as global_lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with api.lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock"):

            def guard():
                control = json.loads(Path(runtime["gpu_control_path"]).read_text())
                require(
                    control.get("rtx5080", {}).get("pause_requested") is False
                    and json.loads(args.availability.read_text()) == lease
                    and not (packet / "STOP").exists(),
                    "initial export pause/ownership/STOP binding changed",
                )
                for locator in read(files, request_pin)["source"].values():
                    files.check(locator)

            guard()
            resources = LinuxResources(runtime["gpu_uuid"])
            released = resources.require_released([], [])
            baseline = resources.snapshot()
            require(
                not released["other_context_pids"] and not baseline["dxg_holders"],
                "released slot requires empty actual CUDA and DXG census before model",
            )
            resource_gate(baseline, baseline, runtime["resource_policy"])
            hardware = trainer.configure_cuda(zero_updates=True)
            require(
                hardware["gpu_uuid"] == runtime["gpu_uuid"]
                and hardware["compute_capability"] == [12, 0],
                "5080 same-runtime export required",
            )
            model = optimizer = _dataset = None
            try:
                model, _dataset, actual_source = trainer.block_inputs(spec, request_pin["sha256"])
                require(actual_source == source, "actual loader original source differs")
                optimizer = block_optimizer(model)
                guard()
                trainer.resources(spec, "before initial checkpoint restore")
                cursor = load_block_checkpoint(model, optimizer, source, checkpoint)
                require(
                    cursor.step == cursor.stage_updates == 0 and not optimizer.state,
                    "initial restore contains optimizer progress/state",
                )
                publication = export_block_checkpoint(model, source, packet / "initial-export")
                npz, manifest = (
                    builder.pin(publication["npz"]),
                    builder.pin(publication["manifest"]),
                )
                serializer = importlib.import_module("export_block_binary")
                guard()
                output = packet / "initial-export/candidate.gguf"
                audit = serializer.export_model(
                    Path(spec["model"]["path"]), Path(npz["path"]), Path(manifest["path"]), output
                )
                atomic_json(packet / "initial-export/export-audit.json", audit)
                receipt = {
                    "schema": "block_lane_initial_export_v1",
                    "status": "PASS",
                    "optimizer_updates": 0,
                    "checkpoint": artifact(checkpoint),
                    "initial_request": request_pin,
                    "preparation_receipt": receipt_pin,
                    "source": source,
                    "config": builder.pin(packet / f"configs/{spec['candidate']}.json"),
                    "npz": npz,
                    "manifest": manifest,
                    "model": builder.pin(output),
                    "audit": builder.pin(packet / "initial-export/export-audit.json"),
                    "hardware": hardware,
                    "native_runtime_gate": "PENDING",
                    "resource_release": "PENDING_parent_process_reap_and_owned_context_return",
                }
                reference, _ = validate_reference(files, inputs, runtime)
                validate_initial_export(
                    files, receipt, spec, source, checkpoint, request_pin, receipt_pin, reference
                )
                atomic_json(packet / "initial-export-receipt.json", receipt)
                return receipt
            finally:
                model = optimizer = _dataset = None
                gc.collect()
                trainer.torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inspect", "prepare"):
        command = sub.add_parser(mode)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--inputs-sha256", required=True)
        if mode == "prepare":
            command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("bind")
    command.add_argument("--packet", type=Path, required=True)
    for name in ("receipts", "qa-ledger"):
        command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--" + name + "-sha256", required=True)
    command = sub.add_parser("finalize")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--admission-plan", type=Path, required=True)
    command.add_argument("--admission-plan-sha256", required=True)
    command.add_argument("--timed-evaluation-plan", type=Path)
    command.add_argument("--timed-evaluation-plan-sha256")
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("export-initial")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--availability", type=Path, required=True)
    command.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if args.mode in ("inspect", "prepare"):
        value = read(Files(), {"path": str(args.inputs.resolve()), "sha256": args.inputs_sha256})
        result = (
            inspect_inputs(value)
            if args.mode == "inspect"
            else prepare(value, args.output.resolve())
        )
    else:
        args.packet = args.packet.resolve()
        result = {"bind": bind, "export-initial": export_initial, "finalize": finalize}[args.mode](
            args
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
