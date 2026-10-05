#!/usr/bin/env python3
"""Metadata-only packet construction for the admitted original EAGLE TRAIN corpus.

Stages config and exact three-domain native capture requests. Existing native
CLIs execute the requests; this helper never constructs a model or GPU context.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_nine_model_bundle as builder  # noqa: E402
from capture_nine_model_train_data import prefix_history, validate_generated  # noqa: E402
from check_block_capture_portability import NativeCaptureGoldens  # noqa: E402

from w1a1_eagle.nine_model_pipeline import Files, atomic_json, require  # noqa: E402

READY_SHA = "bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498"
DOMAINS = ("prose", "code", "reasoning")
FINAL_ONLY_DEVELOPMENT_EVERY = 2**63 - 1


def read(files, path, expected=None):
    locator = {"path": str(Path(path).resolve()), "sha256": expected or builder.sha256(path)}
    return json.loads(files.check(locator).read_text())


def publish_jsonl(path, rows):
    require(not path.exists(), "preserve packet request history")
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def prepare(args):
    files = Files()
    runtime = read(files, args.runtime, args.runtime_sha256)
    require(runtime.get("schema") == "eagle_lane_packet_runtime_v1", "runtime descriptor differs")
    require(not args.output.exists(), "preserve packet history")
    prepared = args.prepared_run_dir.resolve()
    ready = read(files, prepared / "preparation-ready.json", READY_SHA)
    require(
        ready.get("preparation_complete") is True and ready.get("optimization_started") is False,
        "original completed corpus required",
    )
    original = read(files, prepared / "resolved_config.json")
    coverage = ready["teacher_coverage"]
    require(
        original["coverage"]["min_unique_train_prompts"] >= 10000
        and coverage["unique_train_prompts"] == 10000
        and coverage["unique_supervised_rows"] == 3899930,
        "original full TRAIN coverage absent or lowered",
    )
    initializer = read(files, args.initializer_dir / "initializer.json")
    report = read(files, args.initializer_dir / "report.json")
    require(
        report.get("schema") == "eagle_production_fusion_initializer_v1"
        and report["initializer"] == initializer
        and report["prepared_ready_sha256"] == READY_SHA
        and report["authenticated_full_source"] == ready["teacher_coverage"]["source"],
        "production initializer/corpus join differs",
    )
    require(
        report.get("config") == builder.pin(prepared / "resolved_config.json")
        and report.get("prepared_run_dir") == str(prepared),
        "initializer original config/prepared directory differs",
    )
    common = ready["teacher_coverage"]["source"]["common_source_sha256"]
    require(
        runtime["inputs"]["target"]["sha256"] == common["target_gguf"]
        and runtime["base_model"]["sha256"] == common["base_draft_gguf"],
        "packet changes original target/base model",
    )
    require(
        initializer["activation_bits"] == 8
        and report["fit_config"]["orientation_rescue"] is False
        and report["fit_config"]["coordinate_flips"] == 0,
        "fixed A8 scale-only initialization required",
    )
    files.check({key: initializer[key] for key in ("path", "sha256")})
    args.output.mkdir(parents=True)
    authorization = args.output / "authorization.md"
    authorization.write_bytes(args.authorization.read_bytes())
    budget = {
        "schema": "nine_model_selected_budget_v1",
        "human_selected": False,
        "authorization": {
            "kind": "human_delegated_operational_settings",
            "instruction": "Run relevant QAT; preserve healthy runs; healthcheck every 30 min.",
            "record": builder.pin(authorization),
        },
        "candidates": {
            "eagle_a8": {
                "training_limits": {
                    "max_steps": None,
                    "max_tokens": None,
                    "max_seconds": 86400,
                    "max_epochs": None,
                },
                "wall_seconds": 108000,
            }
        },
    }
    atomic_json(args.output / "budget.json", budget)
    template = copy.deepcopy(original)
    template["training"].update(
        checkpoint_every=250,
        keep_checkpoints=3,
        development_every=FINAL_ONLY_DEVELOPMENT_EVERY,
        activation_quantization="fixed",
        a1_computation="reference",
        binary_optimization=None,
        fusion_correction=None,
        affine_weights=None,
        optimize_cache=True,
        optimize_head=True,
        persistent_sign_diagnostics=False,
        optimization_readiness=None,
        min_host_available_bytes=2 * 1024**3,
        min_cuda_free_bytes=1024**3,
        min_free_disk_bytes=8 * 1024**3,
    )
    template.setdefault("native", {})["expected_commit"] = runtime["native_source_revision"]
    atomic_json(args.output / "template.json", template)
    descriptor = {
        "schema": "nine_model_lane_inputs_v1",
        "budget": builder.pin(args.output / "budget.json"),
        "candidates": {
            "eagle_a8": {
                "profile": "fixed_reference",
                "initialization": initializer,
                "fusion_calibration": {key: initializer[key] for key in ("path", "sha256")},
                "base_model": runtime["base_model"],
                "eagle_config_template": builder.pin(args.output / "template.json"),
                "prepared": {"run_dir": str(prepared), "ready_sha256": READY_SHA},
                "data_admission": builder.pin(prepared / "preparation-ready.json"),
                "deployment_coverage": {"profile": "all9"},
                "native_markers": [
                    "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)",
                    "CUDA packed W1A8 INT8 dispatch",
                ],
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
    builder.materialize_configs(descriptor, args.output / "configs")
    atomic_json(
        args.output / "effective-controls.json",
        {
            "schema": "eagle_lane_selected_controls_v1",
            "evidence_scope": "source configuration; actual admission/training pending",
            "config": builder.pin(args.output / "configs/eagle_a8.json"),
            "continuous_config": builder.pin(args.output / "configs/eagle_a8-continuous.json"),
            "full_training_source": ready["teacher_coverage"]["source"],
            "training_iterator": "PreparedProvider.rounds, all original shards in ordinal order",
            "within_shard_order": "sorted (prompt_id, round_index) anchors; full prompt chains",
            "smoke_iterator_only": "PreparedProvider.bounded_rounds, selected TRAIN shard",
            "max_seconds": 86400,
            "other_stop_caps": None,
            "checkpoint_every": 250,
            "keep_checkpoints": 3,
            "development_lifecycle": "standalone",
            "development_every": FINAL_ONLY_DEVELOPMENT_EVERY,
            "cache_head": "requested; current admitted execution required before updates",
            "final_quality_claim": False,
        },
    )
    index = read(
        files,
        prepared / "stages/train-providers.json",
        ready["teacher_coverage"]["source"]["execution_manifest_sha256"],
    )
    requests, joins = [], []
    for domain in DOMAINS:
        selected = next(
            row
            for row in report["selected_prompts"]
            if row["domain"] == domain and row["calibration_split"] == "fit"
        )
        record = index["shards"][selected["shard_ordinal"]]
        provider = read(files, record["provider_manifest"], record["provider_manifest_sha256"])
        prompt_path = files.check(
            {"path": provider["paths"]["prompts"], "sha256": provider["sha256"]["prompts"]}
        )
        prompt = next(
            json.loads(line)
            for line in prompt_path.read_text().splitlines()
            if json.loads(line)["id"] == selected["id"]
        )
        ancestry = {
            "prompt_id": selected["id"],
            "prompt_sha256": selected["content_sha256"],
            "domain": domain,
            "source_split": "TRAIN",
        }
        require(
            hashlib.sha256(
                json.dumps(
                    prompt["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            == ancestry["prompt_sha256"],
            "selected original TRAIN prompt content differs",
        )
        requests.append(
            {
                "messages": prompt["messages"],
                "template_mode": "native_chat",
                "max_new_tokens": 32,
                "max_prompt_tokens": 512,
                "tap_ids": [2, 10, 18, 26, 34],
                "logits_mode": "none",
                "chain_ancestry": ancestry,
            }
        )
        joins.append(
            {
                "domain": domain,
                "original_prompt": prompt,
                "ancestry": ancestry,
                "provider": {
                    "path": record["provider_manifest"],
                    "sha256": record["provider_manifest_sha256"],
                },
            }
        )
    publish_jsonl(args.output / "generation-requests.jsonl", requests)
    atomic_json(
        args.output / "golden-source-joins.json",
        {
            "schema": "eagle_lane_golden_source_joins_v1",
            "runtime": runtime,
            "initializer_report": builder.pin(args.initializer_dir / "report.json"),
            "cases": joins,
        },
    )
    atomic_json(
        args.output / "initial-preparation-request.json",
        {
            "schema": "eagle_lane_initial_preparation_v1",
            "config": builder.pin(args.output / "configs/eagle_a8.json"),
            "source": builder.pin(ROOT / "scripts/train_nine_model_qat.py"),
            "initializer_report": builder.pin(args.initializer_dir / "report.json"),
            "optimizer_updates": 0,
        },
    )
    request_pin = builder.pin(args.output / "initial-preparation-request.json")
    initial = args.output / "initial-prepare"
    checkpoint = initial / "checkpoints/step-000000000000-e000000-r000000000000/A8"
    teacher = runtime["inputs"]["teacher_binary"]["path"]
    target = runtime["inputs"]["target"]
    capture_arguments = [
        "--binary",
        teacher,
        "--target",
        target["path"],
        "--target-sha256",
        target["sha256"],
        "--producer-source-revision",
        runtime["native_source_revision"],
        "--max-tokens",
        "544",
        "--gpu-layers",
        "999",
    ]
    commands = {
        "schema": "eagle_lane_source_commands_v1",
        "execution_authorized_by_this_file": False,
        "source": {
            name: builder.pin(ROOT / "scripts" / name)
            for name in (
                "train_nine_model_qat.py",
                "export_recurrent_binary.py",
                "capture_block_qat_teacher.py",
                "prepare_eagle_lane_packet.py",
            )
        },
        "initial_prepare": [
            sys.executable,
            str(ROOT / "scripts/train_nine_model_qat.py"),
            "--config",
            str(args.output / "configs/eagle_a8.json"),
            "--run-dir",
            str(initial),
            "--bundle-sha256",
            request_pin["sha256"],
            "--stage-name",
            "eagle_a8/initial-prepare",
            "--completion-output",
            str(args.output / "initial-prepare-receipt.json"),
            "--allow-cuda",
            "--prepare-only",
        ],
        "initial_export": [
            sys.executable,
            str(ROOT / "scripts/export_recurrent_binary.py"),
            "--base",
            runtime["base_model"]["path"],
            "--checkpoint",
            str(checkpoint / "joint.npz"),
            "--manifest",
            str(checkpoint / "joint.json"),
            "--output",
            str(args.output / "initial-calibrated-a8.gguf"),
            "--audit",
            str(args.output / "initial-export-audit.json"),
        ],
        "native_generation": [
            sys.executable,
            str(ROOT / "scripts/capture_block_qat_teacher.py"),
            *capture_arguments,
            "--output-root",
            str(args.output / "native-generations"),
            "--requests",
            str(args.output / "generation-requests.jsonl"),
        ],
        "native_replay": [
            sys.executable,
            str(ROOT / "scripts/capture_block_qat_teacher.py"),
            *capture_arguments,
            "--output-root",
            str(args.output / "native-replays"),
            "--requests",
            str(args.output / "replay-requests.jsonl"),
        ],
    }
    atomic_json(args.output / "commands.json", commands)
    print(
        json.dumps(
            {
                "configs": str(args.output / "configs"),
                "initial_preparation_request": builder.pin(
                    args.output / "initial-preparation-request.json"
                ),
                "generation_requests": builder.pin(args.output / "generation-requests.jsonl"),
            }
        )
    )


def replay(args):
    files = Files()
    joins = read(files, args.packet / "golden-source-joins.json")
    receipts = [json.loads(line) for line in args.receipts.read_text().splitlines() if line.strip()]
    require(len(receipts) == 3, "three original native generated chains required")
    requests = []
    for receipt, case in zip(receipts, joins["cases"], strict=True):
        ancestry = dict(case["ancestry"], prompt_length=receipt["prompt_length"])
        require(
            receipt.get("chain_ancestry") == ancestry
            and receipt.get("target_sha256") == joins["runtime"]["inputs"]["target"]["sha256"]
            and receipt.get("producer_source_revision")
            == joins["runtime"]["native_source_revision"]
            and receipt.get("prompt_source_sha256") == ancestry["prompt_sha256"]
            and receipt.get("tokens")
            and receipt["generation"]["mode"] == "native_target_greedy",
            "generated chain ancestry differs",
        )
        tokens = receipt["tokens"]
        native = {
            "binary": joins["runtime"]["inputs"]["teacher_binary"],
            "target": joins["runtime"]["inputs"]["target"],
            "source_revision": joins["runtime"]["native_source_revision"],
            "client_source": builder.pin(ROOT / "scripts/capture_block_qat_teacher.py"),
            "tokenizer_metadata_sha256": receipt["tokenizer_metadata_sha256"],
            "chat_template_sha256": receipt["target_chat_template_sha256"],
        }
        validate_generated(
            receipt,
            case["ancestry"],
            {
                "native": native,
                "vocab_size": 151936,
                "target_width": 2560,
                "objective": "hard_ce",
                "caps": {"max_prompt_tokens": 512, "max_tokens_per_chain": 544},
                "generation": {"max_new_tokens": 32},
                "input_mode": "native_chat",
            },
            {"name": joins["runtime"]["device_name"]},
        )
        require(
            len(tokens) <= 544 and 0 < receipt["prompt_length"] <= 512,
            "native prompt/chain cap exceeded",
        )
        requests.append(
            {
                "tokens": tokens,
                "tap_ids": [2, 18, 33],
                "logits_mode": "last",
                "chain_ancestry": ancestry,
                "decode_history": prefix_history(receipt["decode_history"], len(tokens)),
            }
        )
    publish_jsonl(args.packet / "replay-requests.jsonl", requests)
    atomic_json(
        args.packet / "generation-replay-link.json",
        {
            "source_joins": builder.pin(args.packet / "golden-source-joins.json"),
            "generation_receipts": builder.pin(args.receipts),
            "replay_requests": builder.pin(args.packet / "replay-requests.jsonl"),
            "native": native,
            "requests": requests,
        },
    )
    print(
        json.dumps(
            {
                "generation_replay_link": builder.pin(args.packet / "generation-replay-link.json"),
                "optimizer_updates": 0,
            }
        )
    )


def exact_replay_join(receipts, requests):
    require(len(receipts) == len(requests) == 3, "three exact replay/source joins required")
    domains, ids = set(), set()
    for receipt, request in zip(receipts, requests, strict=True):
        require(
            all(
                receipt.get(key) == request[key]
                for key in ("tokens", "tap_ids", "logits_mode", "chain_ancestry", "decode_history")
            ),
            "replay differs from parent generated tokens/ancestry/decode history",
        )
        ancestry = receipt["chain_ancestry"]
        domains.add(ancestry["domain"])
        ids.add(ancestry["prompt_id"])
    require(
        domains == set(DOMAINS) and len(ids) == 3, "one distinct replay per TRAIN domain required"
    )


def initial_export_join(args, files, runtime):
    request_path = args.packet / "initial-preparation-request.json"
    request = read(files, request_path)
    receipt = read(files, args.packet / "initial-prepare-receipt.json")
    require(
        receipt.get("schema") == "nine_model_preparation_v1"
        and receipt.get("status") == "PASS"
        and receipt.get("optimizer_updates") == 0
        and receipt.get("artifact_kind") == "production"
        and receipt.get("stage") == "eagle_a8/initial-prepare"
        and receipt.get("bundle_sha256") == builder.sha256(request_path)
        and receipt.get("config_sha256") == request["config"]["sha256"],
        "initial prepare request/config/zero-update receipt differs",
    )
    files.check(request["config"])
    initializer_report = read(
        files, request["initializer_report"]["path"], request["initializer_report"]["sha256"]
    )
    spec = read(files, request["config"]["path"], request["config"]["sha256"])
    require(
        spec["initialization"] == initializer_report["initializer"]
        and spec["prepared"]["ready_sha256"]
        == initializer_report["prepared_ready_sha256"]
        == READY_SHA,
        "initial request initializer/prepared source differs",
    )
    ready = read(files, Path(spec["prepared"]["run_dir"]) / "preparation-ready.json", READY_SHA)
    require(
        initializer_report["authenticated_full_source"] == ready["teacher_coverage"]["source"],
        "initial calibration source differs from completed corpus",
    )
    require(
        receipt.get("source") == initializer_report["authenticated_full_source"],
        "initial source differs from calibrated original corpus",
    )
    require(
        request["source"] == builder.pin(ROOT / "scripts/train_nine_model_qat.py"),
        "initial preparation producer source differs",
    )
    checkpoint = files.check({key: receipt["checkpoint"][key] for key in ("path", "sha256")})
    expected_directory = (
        args.packet / "initial-prepare/checkpoints/step-000000000000-e000000-r000000000000"
    )
    require(
        checkpoint == expected_directory / "resume.pt" and receipt["checkpoint"]["step"] == 0,
        "initial checkpoint is not the selected step zero",
    )
    outer = read(files, checkpoint.parent / "manifest.json")
    require(
        outer.get("source_sha256") == initializer_report["source_sha256"],
        "initial checkpoint source digest differs",
    )
    require(
        outer.get("step") == outer.get("epoch") == outer.get("cursor") == 0
        and set(outer["exports"]) == {"A8"}
        and outer["sha256"] == receipt["checkpoint"]["sha256"]
        and outer.get("optimizer_rng_cursor_exact") is True,
        "initial outer checkpoint state differs",
    )
    joint = {
        name: {"path": str(checkpoint.parent / "A8" / name), "sha256": digest}
        for name, digest in outer["exports"]["A8"].items()
    }
    require(set(joint) == {"joint.npz", "joint.json"}, "initial projection publication differs")
    for locator in joint.values():
        files.check(locator)
    audit = read(files, args.export_audit)
    require(
        audit.get("serialization_audit_passed") is True
        and audit.get("activation_bits") == 8
        and len(audit.get("projections", {})) == 9
        and audit.get("base_gguf") == runtime["base_model"]
        and audit.get("output") == builder.pin(args.initial_model)
        and audit.get("checkpoint") == joint["joint.npz"]
        and audit.get("checkpoint_manifest") == joint["joint.json"],
        "initial native export does not join selected step-zero/base/projections",
    )
    return receipt


def bind(args):
    require(not (args.packet / "production-inputs.json").exists(), "preserve production packet")
    require(not any(args.packet.glob("golden-*.json")), "preserve previous golden bind attempt")
    files = Files()
    joins = read(files, args.packet / "golden-source-joins.json")
    runtime = joins["runtime"]
    initial_export_join(args, files, runtime)
    receipts = [json.loads(line) for line in args.receipts.read_text().splitlines() if line.strip()]
    require(len(receipts) == 3, "three replay goldens required")
    link = read(files, args.packet / "generation-replay-link.json", args.generation_link_sha256)
    require(
        link["source_joins"] == builder.pin(args.packet / "golden-source-joins.json"),
        "generation/replay source joins changed",
    )
    files.check(link["generation_receipts"])
    files.check(link["replay_requests"])
    requests = [
        json.loads(line)
        for line in Path(link["replay_requests"]["path"]).read_text().splitlines()
        if line.strip()
    ]
    require(requests == link["requests"], "pinned replay request content differs")
    generated = [
        json.loads(line)
        for line in Path(link["generation_receipts"]["path"]).read_text().splitlines()
        if line.strip()
    ]
    require(
        len(generated) == len(requests) == len(joins["cases"]) == 3,
        "three exact generated parents required",
    )
    for parent, request, case in zip(generated, requests, joins["cases"], strict=True):
        require(
            request["tokens"] == parent["tokens"]
            and request["decode_history"]
            == prefix_history(parent["decode_history"], len(parent["tokens"]))
            and request["chain_ancestry"]
            == dict(case["ancestry"], prompt_length=parent["prompt_length"]),
            "replay request differs from original generated parent/domain",
        )
    exact_replay_join(receipts, requests)
    from capture_nine_model_train_data import validate_replay

    for receipt, request in zip(receipts, requests, strict=True):
        validate_replay(
            receipt,
            request["tokens"],
            [2, 18, 33],
            link["native"],
            {"name": runtime["device_name"]},
        )
    inventory = {
        "schema": "block_train_inventory_v1",
        "prompts": {
            case["ancestry"]["prompt_id"]: {
                "sha256": case["ancestry"]["prompt_sha256"],
                "domain": case["domain"],
                "split": "TRAIN",
            }
            for case in joins["cases"]
        },
    }
    atomic_json(args.packet / "train-inventory.json", inventory)
    cases = []
    for ordinal, receipt in enumerate(receipts):
        path = args.packet / f"golden-{ordinal}.json"
        atomic_json(path, receipt)
        cases.append(
            {"chain_id": f"eagle-native-train-{ordinal}", "native_receipt": builder.pin(path)}
        )
    golden = {
        "schema": "nine_model_train_capture_goldens_v1",
        "family": "eagle",
        "tap_ids": [2, 18, 33],
        "vocab_size": 151936,
        "target_width": 2560,
        "target_sha256": runtime["inputs"]["target"]["sha256"],
        "train_inventory": builder.pin(args.packet / "train-inventory.json"),
        "cases": cases,
    }
    atomic_json(args.packet / "eagle-goldens.json", golden)
    NativeCaptureGoldens(
        args.packet / "eagle-goldens.json",
        expected_sha256=builder.sha256(args.packet / "eagle-goldens.json"),
    )
    prompts = [dict(case["original_prompt"], split="train") for case in joins["cases"]]
    publish_jsonl(args.packet / "native-smoke-prompts.jsonl", prompts)
    atomic_json(
        args.packet / "native-smoke-capture.json",
        {
            "split": "train",
            "target_sha256": golden["target_sha256"],
            "prompts_sha256": builder.sha256(args.packet / "native-smoke-prompts.jsonl"),
            "source": builder.pin(args.packet / "golden-source-joins.json"),
            "goldens": builder.pin(args.packet / "eagle-goldens.json"),
            "generation_replay_link": builder.pin(args.packet / "generation-replay-link.json"),
            "initial_preparation_receipt": builder.pin(
                args.packet / "initial-prepare-receipt.json"
            ),
        },
    )
    descriptor = read(files, args.packet / "configs/resolved-inputs.json")
    descriptor["candidates"]["eagle_a8"].update(
        initial_model=builder.pin(args.initial_model),
        initial_export_audit=builder.pin(args.export_audit),
    )
    descriptor["inputs"].update(
        eagle_smoke_prompts=builder.pin(args.packet / "native-smoke-prompts.jsonl"),
        eagle_smoke_capture=builder.pin(args.packet / "native-smoke-capture.json"),
    )
    descriptor["preflight"]["portability"]["eagle"] = {
        "golden_manifest": builder.pin(args.packet / "eagle-goldens.json"),
        "max_tokens": 544,
    }
    initial_outer = (
        args.packet
        / "initial-prepare/checkpoints/step-000000000000-e000000-r000000000000/manifest.json"
    )
    descriptor["preflight"]["preparation_provenance"] = {
        "generation_replay_link": builder.pin(args.packet / "generation-replay-link.json"),
        "generated_parent_receipts": link["generation_receipts"],
        "exact_replay_requests": link["replay_requests"],
        "original_train_prompt_joins": link["source_joins"],
        "initial_request": builder.pin(args.packet / "initial-preparation-request.json"),
        "initial_zero_update_receipt": builder.pin(args.packet / "initial-prepare-receipt.json"),
        "initial_outer_checkpoint_manifest": builder.pin(initial_outer),
    }
    descriptor["qa_ledger"] = builder.pin(args.qa_ledger)
    atomic_json(args.packet / "production-inputs.json", descriptor)
    print(
        json.dumps(
            {
                "inputs": builder.pin(args.packet / "production-inputs.json"),
                "production_ready": False,
                "optimizer_updates": 0,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    config = sub.add_parser("prepare")
    for name in ("runtime", "prepared-run-dir", "initializer-dir", "authorization", "output"):
        config.add_argument("--" + name, type=Path, required=True)
    config.add_argument("--runtime-sha256", required=True)
    for mode in ("replay", "bind"):
        command = sub.add_parser(mode)
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--receipts", type=Path, required=True)
        if mode == "bind":
            command.add_argument("--generation-link-sha256", required=True)
            for name in ("initial-model", "export-audit", "qa-ledger"):
                command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("packet", "output"):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    {"prepare": prepare, "replay": replay, "bind": bind}[args.mode](args)


if __name__ == "__main__":
    main()
