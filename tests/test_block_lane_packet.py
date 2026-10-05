"""Actual producer schemas/APIs on tiny CPU fixtures; no real model/GPU claims."""
# ruff: noqa: E402

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import export_block_binary as serializer
import prepare_block_lane_packet as packet
import test_block_data as data_fixtures
import test_nine_model_training as model_fixtures
import train_nine_model_qat as trainer
from extract_block_fusion_reference import extract_reference
from gguf import GGMLQuantizationType, GGUFWriter

from w1a1_eagle.block_data import BlockDataset, file_sha256, import_capture_plan
from w1a1_eagle.block_qat import BlockDrafter, block_optimizer
from w1a1_eagle.block_training import (
    BlockCursor,
    export_block_checkpoint,
    load_block_checkpoint,
    save_block_checkpoint,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return packet.builder.pin(path)


def fixture(root, family="dspark", bits=8):
    (root / "target").write_bytes(b"unit fixture source")
    target_sha = file_sha256(root / "target")
    raw = data_fixtures.NativeRawImportTests("test_raw_producer_complete_prefix_full_vocab_memmap")
    raw.setUp()
    # Keep native fixture raw inputs in this fixture's owned temporary directory.
    import shutil

    shutil.copytree(raw.root, root / "raw")
    old = str(raw.root)
    raw.doCleanups()
    rawroot = root / "raw"
    plan = json.loads((rawroot / "plan.json").read_text().replace(old, str(rawroot)))
    plan.update(family=family, target_width=8)
    inventory = {"schema": "block_train_inventory_v1", "prompts": {}}
    sources = []
    for ordinal in range(18):
        domain = packet.DOMAINS[ordinal % 3]
        split = (
            "train"
            if ordinal < 12
            else "calibration_fit"
            if ordinal < 15
            else "calibration_validation"
        )
        source = copy.deepcopy(plan["chains"][ordinal % 3])
        record = source["native_receipt"]
        native = json.loads(Path(record["path"]).read_text())
        for descriptor in native["files"].values():
            descriptor["path"] = descriptor["path"].replace(old, str(rawroot))
        features = native["files"]["features"]
        feature_path = rawroot / f"features-{ordinal}.f32"
        np.arange(18 * 5 * 8, dtype=np.float32).reshape(18, 5, 8).tofile(feature_path)
        features.update(path=str(feature_path), shape=[18, 5, 8], sha256=file_sha256(feature_path))
        prompt = f"original-unit-source-{ordinal}"
        content = packet.identity({"prompt": prompt})
        source.update(
            chain_id=f"chain-{ordinal:02d}",
            prompt_id=prompt,
            prompt_sha256=content,
            domain=domain,
            split=split,
        )
        native.update(
            target_sha256=target_sha,
            features_shape=[18, 5, 8],
            hardware=["Fixture CUDA"],
            executed_result_buffers=["CUDA0"],
            target_storage_buffers={"CUDA0": 1},
            gpu_layers=999,
        )
        native["chain_ancestry"].update(prompt_id=prompt, prompt_sha256=content, domain=domain)
        native_path = rawroot / f"native-{ordinal}.json"
        source["native_receipt"] = write(native_path, native)
        inventory["prompts"][prompt] = {"sha256": content, "domain": domain, "split": "TRAIN"}
        sources.append(source)
    plan["chains"] = sources
    plan["train_inventory"] = write(rawroot / "inventory.json", inventory)
    plan_path = rawroot / "plan.json"
    write(plan_path, plan)
    manifest = import_capture_plan(
        plan_path, expected_sha256=file_sha256(plan_path), output_dir=root / "data"
    )
    cfg, tensors, _ = model_fixtures.block_fixture(family, bits)
    cfg = replace(
        cfg,
        vocab_size=32,
        mask_token_id=31,
        objective="full_probability_l1" if family == "dspark" else "hard_ce",
    )
    torch.manual_seed(17)
    tensors["token_embd.weight"] = torch.randn(32, 8) * 0.2
    tensors["output.weight"] = torch.randn(32, 8) * 0.2
    if family == "dspark":
        tensors["markov_w1.weight"] = torch.randn(32, 3) * 0.1
        tensors["markov_w2.weight"] = torch.randn(32, 3) * 0.1
    base = root / "base.gguf"
    writer = GGUFWriter(base, "dflash")
    writer.add_block_count(5)
    writer.add_block_size(7)
    writer.add_target_layers(list(packet.TAPS))
    writer.add_sample_from_anchor(True)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(12)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_key_length(4)
    writer.add_value_length(4)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_mask_token_id(31)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"unit{i}" for i in range(32)])
    for name, tensor in tensors.items():
        values = tensor.numpy().astype(np.float32)
        bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
        writer.add_tensor(name, bf16, raw_dtype=GGMLQuantizationType.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    extract_reference(
        base, model_sha256=file_sha256(base), family=family, output_dir=root / "reference"
    )
    reference = {
        "weights": packet.builder.pin(root / "reference/fc-weight.npy"),
        "norm": packet.builder.pin(root / "reference/fc-norm.npy"),
        "metadata": packet.builder.pin(root / "reference/reference.json"),
    }
    fit_dir = root / f"fit-a{bits}"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fit_block_fusion.py"),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            file_sha256(manifest),
            "--admission",
            str(root / "data/completed-admission.json"),
            "--admission-sha256",
            file_sha256(root / "data/completed-admission.json"),
            "--weights",
            reference["weights"]["path"],
            "--weights-sha256",
            reference["weights"]["sha256"],
            "--norm",
            reference["norm"]["path"],
            "--norm-sha256",
            reference["norm"]["sha256"],
            "--norm-metadata",
            reference["metadata"]["path"],
            "--norm-metadata-sha256",
            reference["metadata"]["sha256"],
            "--norm-epsilon",
            "1e-6",
            "--activation-bits",
            str(bits),
            "--rows-per-chain",
            "2",
            "--max-total-rows",
            "128",
            "--max-array-bytes",
            "1000000",
            "--max-seconds",
            "5",
            "--output-dir",
            str(fit_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    authorization = root / "authorization.md"
    authorization.write_text(
        "Synthetic test of supplied human operational policy; no GPU authorization."
    )
    auth = {
        "kind": "human_delegated_operational_settings",
        "instruction": "Fixture policy only",
        "record": packet.builder.pin(authorization),
    }
    name = f"{family}_a{bits}"
    budget = write(
        root / "budget.json",
        {
            "schema": "nine_model_selected_budget_v1",
            "human_selected": False,
            "authorization": auth,
            "candidates": {
                name: {
                    "training_limits": {
                        "max_steps": None,
                        "max_supervised_tokens": None,
                        "max_seconds": 15,
                        "max_epochs": None,
                    },
                    "wall_seconds": 30,
                }
            },
        },
    )
    inputs = {}
    for role in ("target", "teacher_binary", "backend_binary", "block_native_binary"):
        path = root / role
        path.write_bytes(b"unit fixture source")
        inputs[role] = packet.builder.pin(path)
    runtime = write(
        root / "runtime.json",
        {
            "schema": "block_lane_packet_runtime_v1",
            "base_model": packet.builder.pin(base),
            "inputs": inputs,
            "native_source_revision": "a" * 40,
            "gpu_uuid": "GPU-unit-fixture",
            "device_name": "Fixture CUDA",
            "gpu_control_path": str(root / "control.json"),
            "resource_policy": {
                "host_floor_bytes": 1,
                "gpu_floor_bytes": 1,
                "host_return_tolerance_bytes": 1,
                "gpu_return_tolerance_bytes": 1,
            },
            "golden_max_tokens": 18,
        },
    )
    write(root / "control.json", {"rtx5080": {"pause_requested": False}})
    descriptor = {
        "schema": packet.INPUT_SCHEMA,
        "candidate": name,
        "runtime": runtime,
        "budget": budget,
        "data": packet.builder.pin(manifest),
        "data_admission": packet.builder.pin(root / "data/completed-admission.json"),
        "coverage_policy": write(
            root / "coverage.json",
            {
                "schema": "block_lane_coverage_policy_v1",
                "source_role": "original_TRAIN",
                "min_unique_train_prompts": 10,
                "min_train_blocks": 10,
                "authorization": auth,
            },
        ),
        "reference": reference,
        "initializer": {
            "fit_report": packet.builder.pin(fit_dir / "fit_report.json"),
            "artifact": packet.builder.pin(fit_dir / "fusion_candidate.npz"),
        },
        "calibration_policy": write(
            root / "calibration.json",
            {
                "schema": "block_lane_calibration_policy_v1",
                "rows_per_chain": 2,
                "max_total_rows": 128,
                "max_array_bytes": 1000000,
                "max_seconds": 5,
                "authorization": auth,
            },
        ),
        "objective": cfg.objective,
        "conditioning": cfg.conditioning,
        "checkpoint_every": 2,
        "resource_floors": {},
        "qat_overrides": {
            key: asdict(cfg)[key]
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "num_kv_heads",
                "head_dim",
                "vocab_size",
                "mask_token_id",
            )
        },
    }
    return descriptor, cfg, tensors


def prepare_fixture(root, family="dspark", bits=8):
    descriptor, cfg, tensors = fixture(root, family, bits)
    output = root / "packet"
    packet.prepare(descriptor, output)
    return descriptor, cfg, tensors, output


def zero_publication(output):
    output = output.resolve()
    files = packet.Files()
    request_pin = packet.builder.pin(output / "initial-preparation-request.json")
    request = packet.read(files, request_pin)
    spec = trainer.load_spec(files.check(request["config"]))
    config = packet.BlockQATConfig(**spec["qat"])
    # Real tiny GGUF loader, initializer loader, checkpoint API and serializer.
    from w1a1_eagle.block_training import load_block_gguf

    model = BlockDrafter(
        load_block_gguf(spec["model"]["path"], spec["model"]["sha256"], config),
        config,
        binary_initializer=trainer.calibration(spec["initialization"], config),
    )
    optimizer = block_optimizer(model)
    dataset = BlockDataset(
        spec["data"]["path"],
        expected_sha256=spec["data"]["sha256"],
        admission_path=spec["data"]["admission"]["path"],
        admission_sha256=spec["data"]["admission"]["sha256"],
    )
    source = {
        "base_gguf_sha256": spec["model"]["sha256"],
        "data_manifest_sha256": spec["data"]["sha256"],
        "bundle_sha256": request_pin["sha256"],
        "synthetic": False,
        "target_sha256": hashlib.sha256(b"unit fixture source").hexdigest(),
    }
    cursor = BlockCursor(data_cursor=dataset.cursor(seed=config.seed).payload())
    checkpoint = save_block_checkpoint(
        model, optimizer, cursor, source, output / "initial-prepare/checkpoints"
    )
    first, _ = dataset.next_block(
        dataset.cursor(seed=config.seed), require_teacher=config.objective == "full_probability_l1"
    )
    smoke = trainer.smoke_block(model, trainer.tensor_batch(first), optimizer)
    write(
        output / "initial-prepare-receipt.json",
        {
            "schema": "nine_model_preparation_v1",
            "status": "PASS",
            "optimizer_updates": 0,
            "checkpoint": checkpoint,
            "smoke": smoke,
            "smoke_contract": trainer.completed_zero_update_smoke_contract(
                [config.activation_bits], [optimizer]
            ),
            "training_memory": {},
            "resources": {},
        },
    )
    # Exact real CPU restore is exercised; production exporter keeps CUDA instead.
    self_cursor = load_block_checkpoint(model, optimizer, source, checkpoint)
    assert self_cursor == cursor and not optimizer.state
    publication = export_block_checkpoint(model, source, output / "initial-export")
    npz, manifest = (
        packet.builder.pin(publication["npz"]),
        packet.builder.pin(publication["manifest"]),
    )
    model_path = output / "initial-export/candidate.gguf"
    audit = serializer.export_model(
        Path(spec["model"]["path"]), Path(npz["path"]), Path(manifest["path"]), model_path
    )
    export = {
        "schema": "block_lane_initial_export_v1",
        "status": "PASS",
        "optimizer_updates": 0,
        "checkpoint": packet.artifact(checkpoint),
        "initial_request": request_pin,
        "preparation_receipt": packet.builder.pin(output / "initial-prepare-receipt.json"),
        "source": source,
        "config": request["config"],
        "npz": npz,
        "manifest": manifest,
        "model": packet.builder.pin(model_path),
        "audit": write(output / "initial-export/export-audit.json", audit),
    }
    write(output / "initial-export-receipt.json", export)
    return spec, source, checkpoint, request_pin, export["preparation_receipt"], export


def golden_bind_args(output):
    output = output.resolve()
    inputs = packet.read(packet.Files(), packet.builder.pin(output / "packet-inputs.json"))
    runtime = packet.read(packet.Files(), inputs["runtime"])
    joins = json.loads((output / "golden-source-joins.json").read_text())
    requests = [
        json.loads(line)
        for line in (output / "golden-replay-requests.jsonl").read_text().splitlines()
    ]
    receipts = []
    for index, (request, case) in enumerate(zip(requests, joins["cases"], strict=True)):
        receipt = packet.read(packet.Files(), case["source_native_receipt"])
        dense = receipt["files"]["logits"]
        row = np.fromfile(dense["path"], dtype=np.float32).reshape(dense["shape"])[-1:]
        path = output / f"native-golden-last-{index}.f32"
        row.tofile(path)
        receipt.update(request)
        receipt.update(
            producer_binary_sha256=runtime["inputs"]["teacher_binary"]["sha256"],
            client_source_sha256=packet.builder.sha256(
                ROOT / "scripts/capture_block_qat_teacher.py"
            ),
            logits_shape=[1, 32],
        )
        receipt["files"]["logits"] = {
            "path": str(path),
            "shape": [1, 32],
            "dtype": "float32",
            "sha256": file_sha256(path),
        }
        receipts.append(receipt)
    receipts_path = output / "actual-schema-fixture-receipts.jsonl"
    receipts_path.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    name = inputs["candidate"]
    profiles = {
        candidate: {"prelaunch_status": "PENDING", "unchanged": True}
        for candidate in packet.CANDIDATES
    }
    profiles[name] = {
        "prelaunch_status": "PASS",
        "prelaunch_requirements": {
            key: {
                "status": "PASS",
                "evidence": [{"scope": "actual producer schema/source contract"}],
            }
            for key in packet.builder.PORTABLE
        },
    }
    ledger = {
        "schema": "nine_model_qa_ledger_v1",
        "profiles": profiles,
        "source_files": {
            source: packet.builder.sha256(ROOT / source)
            for source in (*packet.builder.CRITICAL_SOURCE, "scripts/prepare_block_lane_packet.py")
        },
    }
    qa = write(output / "qa-fixture.json", ledger)
    return SimpleNamespace(
        packet=output,
        receipts=receipts_path,
        receipts_sha256=file_sha256(receipts_path),
        qa_ledger=Path(qa["path"]),
        qa_ledger_sha256=qa["sha256"],
    )


class PacketTests(unittest.TestCase):
    def test_draft_inspection_is_pending_without_models_gpu_or_campaign_pass(self):
        report = packet.inspect_inputs({"schema": packet.INPUT_SCHEMA, "candidate": "dspark_a8"})
        self.assertEqual(report["status"], "PENDING")
        self.assertFalse(
            report["model_loaded"] or report["gpu_queried"] or report["production_ready"]
        )
        self.assertEqual(len(report["remaining_candidates"]), 5)
        self.assertIn("missing data_admission", report["pending"])

    def test_all_four_candidate_configs_use_real_source_dataclasses_and_fits(self):
        for family in ("dspark", "dflash"):
            for bits in (8, 1):
                with tempfile.TemporaryDirectory() as tmp:
                    descriptor, _, _, output = prepare_fixture(Path(tmp), family, bits)
                    config = trainer.load_spec(output / f"configs/{family}_a{bits}.json")
                    self.assertEqual(config["qat"]["profile"], "ffn15_fusion")
                    self.assertEqual(config["qat"]["activation_bits"], bits)
                    self.assertEqual(config["qat"]["conditioning"], "captured_prefix")
                    self.assertEqual(
                        config["qat"]["objective"],
                        "full_probability_l1" if family == "dspark" else "hard_ce",
                    )
                    self.assertEqual(
                        config["qat"]["latent_initialization"], "preserve_reference_magnitudes"
                    )
                    self.assertEqual(
                        config["limits"],
                        packet.read(packet.Files(), descriptor["budget"])["candidates"][
                            descriptor["candidate"]
                        ]["training_limits"],
                    )
                    commands = json.loads((output / "commands.json").read_text())
                    self.assertFalse(commands["execution_authorized_by_this_file"])
                    self.assertIn("--prepare-only", commands["initial_prepare"])
                    self.assertIn("--materialize-admission-plan", commands["admission_plan"])
                    self.assertIn("--inputs", commands["admission_plan"])
                    self.assertNotIn("--help", commands["admission_plan"])
                    self.assertNotIn("--orientation-rescue", commands["scale_only_fit"])
                    self.assertNotIn("--coordinate-flips", commands["scale_only_fit"])
                    self.assertEqual(
                        json.loads((output / "effective-controls.json").read_text())[
                            "remaining_candidates"
                        ],
                        [name for name in packet.CANDIDATES if name != descriptor["candidate"]],
                    )
                    self.assertEqual(
                        json.loads((output / "configs/source-configs.json").read_text())[
                            "production_preparation_ready"
                        ],
                        False,
                    )

    def test_explicit_objective_prefix_probes_precision_and_pilot_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, _, _ = fixture(root)
            for change, reason in (
                ({"objective": "hard_ce"}, "DSpark"),
                ({"conditioning": "native_greedy"}, "explicit admitted"),
                (
                    {"qat_overrides": {"optimizer_backend": "fused_fp32_probe"}},
                    "cannot be overridden",
                ),
            ):
                with self.assertRaisesRegex(ValueError, reason):
                    packet.prepare(descriptor | change, root / "packet")
            policy = packet.read(packet.Files(), descriptor["coverage_policy"])
            policy["min_unique_train_prompts"] = 9
            changed = descriptor | {"coverage_policy": write(root / "coverage.json", policy)}
            with self.assertRaisesRegex(ValueError, "development nine-chain pilot"):
                packet.prepare(changed, root / "packet")
            policy["min_unique_train_prompts"] = 10000
            changed["coverage_policy"] = write(root / "coverage.json", policy)
            with self.assertRaisesRegex(ValueError, "extent below"):
                packet.prepare(changed, root / "packet")

    def test_completed_host_admission_refuses_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, _, _ = fixture(root)
            manifest = packet.read(packet.Files(), descriptor["data"])
            feature = Path(manifest["chains"][0]["features"]["path"])
            with feature.open("ab") as stream:
                stream.write(b"changed after completed audit")
            with self.assertRaisesRegex(ValueError, "identity"):
                packet.prepare(descriptor, root / "packet")
            self.assertFalse((root / "packet").exists())

    def test_reference_epsilon_bf16_and_independent_fit_source_rows_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, _, _ = fixture(root)
            report = packet.read(packet.Files(), descriptor["initializer"]["fit_report"])
            original = copy.deepcopy(report)
            for mutate in (
                lambda r: r["config"].update(activation_bits=1),
                lambda r: r["config"].update(zero_scale_orientation_rescue=True),
                lambda r: r.update(norm_epsilon=1e-5),
                lambda r: r["fit_rows"][0].update(positions=[1]),
                lambda r: r.update(data_sha256="0" * 64),
            ):
                report = copy.deepcopy(original)
                mutate(report)
                altered = descriptor | {
                    "initializer": descriptor["initializer"]
                    | {"fit_report": write(root / "altered-fit.json", report)}
                }
                with self.assertRaises(ValueError):
                    packet.prepare(altered, root / "packet")

    def test_actual_zero_checkpoint_complete_export_and_corruption_joins(self):
        with tempfile.TemporaryDirectory() as tmp:
            descriptor, _, _, output = prepare_fixture(Path(tmp))
            spec, source, checkpoint, request, receipt, export = zero_publication(output)
            self.assertEqual(packet.checkpoint_join(output, packet.Files())[1], source)
            reference = packet.read(packet.Files(), descriptor["reference"]["metadata"])
            packet.validate_initial_export(
                packet.Files(), export, spec, source, checkpoint, request, receipt, reference
            )
            self.assertEqual(
                set(packet.read(packet.Files(), export["manifest"])["projections"]),
                packet.PROJECTIONS,
            )
            for mutate in (
                lambda e: e.update(optimizer_updates=1),
                lambda e: e.update(source=source | {"target_sha256": "0" * 64}),
                lambda e: e.update(checkpoint=export["npz"]),
            ):
                changed = copy.deepcopy(export)
                mutate(changed)
                with self.assertRaises(ValueError):
                    packet.validate_initial_export(
                        packet.Files(),
                        changed,
                        spec,
                        source,
                        checkpoint,
                        request,
                        receipt,
                        reference,
                    )
            initial = json.loads((output / "initial-prepare-receipt.json").read_text())
            for mutate in (
                lambda r: r["checkpoint"]["cursor"].update(step=1),
                lambda r: r["checkpoint"].update(contract_sha256="0" * 64),
                lambda r: r["checkpoint"].update(source_sha256="0" * 64),
                lambda r: r["smoke"].update(selected_projection_count=15),
                lambda r: r["smoke"].update(optimizer_moment_tensors=2),
            ):
                changed = copy.deepcopy(initial)
                mutate(changed)
                write(output / "initial-prepare-receipt.json", changed)
                with self.assertRaises(ValueError):
                    packet.checkpoint_join(output, packet.Files())
            write(output / "initial-prepare-receipt.json", initial)
            audit = packet.read(packet.Files(), export["audit"])
            del audit["preserved_tensors"]["output.weight"]
            changed_export = export | {"audit": write(output / "altered-audit.json", audit)}
            with self.assertRaisesRegex(ValueError, "protected"):
                packet.validate_initial_export(
                    packet.Files(),
                    changed_export,
                    spec,
                    source,
                    checkpoint,
                    request,
                    receipt,
                    reference,
                )

    def test_native_bind_exact_parent_five_taps_and_qa_remains_single_lane_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, output = prepare_fixture(Path(tmp))
            zero_publication(output)
            args = golden_bind_args(output)
            original = args.receipts.read_text()
            changed = [json.loads(row) for row in original.splitlines()]
            changed[0]["tokens"][0] += 1
            args.receipts.write_text("".join(json.dumps(row) + "\n" for row in changed))
            args.receipts_sha256 = file_sha256(args.receipts)
            with self.assertRaisesRegex(ValueError, "tokens/absolute history"):
                packet.bind(args)
            args.receipts.write_text(original)
            args.receipts_sha256 = file_sha256(args.receipts)
            result = packet.bind(args)
            self.assertFalse(result["production_ready"] or result["campaign_complete"])
            self.assertEqual(result["status"], "PENDING_fresh_native_admission")
            descriptor = packet.read(packet.Files(), result["inputs"])
            self.assertEqual(set(descriptor["candidates"]), {"dspark_a8"})
            self.assertEqual(set(descriptor["preflight"]["portability"]), {"dspark"})
            ledger = packet.read(packet.Files(), descriptor["qa_ledger"])
            self.assertTrue(
                all(
                    ledger["profiles"][name]["prelaunch_status"] == "PENDING"
                    for name in packet.CANDIDATES
                    if name != "dspark_a8"
                )
            )
            with self.assertRaisesRegex(ValueError, "prior selected packet"):
                packet.bind(args)

    def test_real_admission_plan_and_single_lane_builder_sequence_is_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, output = prepare_fixture(Path(tmp), family="dflash", bits=1)
            zero_publication(output)
            packet.bind(golden_bind_args(output))
            descriptor = packet.read(
                packet.Files(), packet.builder.pin(output / "production-inputs.json")
            )
            prepared = packet.builder.materialize_admission_plan(
                descriptor, output / "admission-plan.json"
            )
            self.assertFalse(
                prepared["production_ready"] or prepared["model_loaded"] or prepared["gpu_queried"]
            )
            args = SimpleNamespace(
                packet=output.resolve(),
                admission_plan=Path(prepared["plan"]["path"]),
                admission_plan_sha256=prepared["plan"]["sha256"],
                output=output / "lane.json",
            )
            report = packet.finalize(args)
            self.assertEqual(report["status"], "PENDING_fresh_native_admission")
            self.assertFalse(report["production_ready"] or report["campaign_complete"])
            lane = packet.read(packet.Files(), report["source_lane"]["lane"])
            self.assertEqual(lane["candidate"], "dflash_a1")
            self.assertEqual(
                lane["target_policy"], {"immutable": True, "weights": "f16", "kv": "f16"}
            )
            self.assertEqual(len(lane["remaining_candidates"]), 5)
            self.assertEqual(lane["evaluation_status"], "PENDING")
            with self.assertRaisesRegex(ValueError, "history"):
                packet.finalize(args)

    def test_explicit_live_prefix_uses_actual_teacher_and_portability_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, _, _ = fixture(root)
            runtime = packet.read(packet.Files(), descriptor["runtime"])
            gate = {
                "schema": "nine_model_capture_portability_v1",
                "status": "PASS",
                "artifact_kind": "production",
                "family": "dspark",
                "tap_ids": list(packet.TAPS),
                "gate_source_sha256": packet.builder.sha256(
                    ROOT / "scripts/check_block_capture_portability.py"
                ),
                "manifest_sha256": descriptor["data"]["sha256"],
                "target_sha256": hashlib.sha256(b"unit fixture source").hexdigest(),
                "native_binary_sha256": runtime["inputs"]["teacher_binary"]["sha256"],
                "native_source_revision": runtime["native_source_revision"],
                "teacher_client_source_sha256": packet.builder.sha256(
                    ROOT / "scripts/capture_block_qat_teacher.py"
                ),
                "compute_capability": [12, 0],
                "device": {"uuid": runtime["gpu_uuid"]},
                "producer_closed": True,
                "producer_returncode": 0,
                "numeric_checks": [
                    {
                        "domain": domain,
                        "decision_changed": False,
                        "features": {"status": "PASS"},
                        "full_vocab_logits": {"status": "PASS"},
                    }
                    for domain in packet.DOMAINS
                ],
            }
            descriptor.update(
                conditioning="native_greedy",
                teacher={
                    "binary": runtime["inputs"]["teacher_binary"]["path"],
                    "target": runtime["inputs"]["target"]["path"],
                    "target_sha256": hashlib.sha256(b"unit fixture source").hexdigest(),
                    "max_tokens": 32,
                    "gpu_layers": 999,
                    "producer_source_revision": runtime["native_source_revision"],
                },
                live_teacher_admission=write(root / "teacher-gate.json", gate),
            )
            packet.prepare(descriptor, root / "packet")
            spec = trainer.load_spec(root / "packet/configs/dspark_a8.json")
            self.assertEqual(spec["teacher"], descriptor["teacher"])
            self.assertEqual(spec["qat"]["conditioning"], "native_greedy")
            gate["numeric_checks"][0]["decision_changed"] = True
            descriptor["live_teacher_admission"] = write(root / "teacher-gate.json", gate)
            with self.assertRaisesRegex(ValueError, "numeric/decision"):
                packet.prepare(descriptor, root / "bad-packet")

    def test_checkpoint_runtime_cpu_cannot_be_relabeled_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, output = prepare_fixture(Path(tmp))
            spec, source, checkpoint, *_ = zero_publication(output)
            cfg = packet.BlockQATConfig(**spec["qat"])
            from w1a1_eagle.block_training import load_block_gguf

            model = BlockDrafter(
                load_block_gguf(spec["model"]["path"], spec["model"]["sha256"], cfg),
                cfg,
                binary_initializer=trainer.calibration(spec["initialization"], cfg),
            )
            optimizer = block_optimizer(model)
            from w1a1_eagle import block_training

            original = block_training._runtime(model)
            with patch.object(
                block_training, "_runtime", return_value=original | {"device_type": "cuda"}
            ):
                with self.assertRaisesRegex(ValueError, "runtime/cursor"):
                    load_block_checkpoint(model, optimizer, source, checkpoint)
            self.assertFalse(optimizer.state)

    def test_global_then_private_owner_locks_refuse_before_gpu_queries(self):
        import fcntl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, output = prepare_fixture(root)
            zero_publication(output)
            # Remove already completed fixture export: producer must otherwise refuse
            # history before reaching its lock boundary.
            import shutil

            shutil.rmtree(output / "initial-export")
            (output / "initial-export-receipt.json").unlink()
            lease = {"gpu_uuid": "GPU-unit-fixture"}
            args = SimpleNamespace(
                allow_cuda=True, packet=output.resolve(), availability=root / "lease.json"
            )
            home = root / "test-home"
            global_path = home / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
            private_path = home / ".cache/binary-eagle-decoding/cuda-0.owner.lock"
            for path, exception in ((global_path, BlockingIOError), (private_path, RuntimeError)):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as owner:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with (
                        patch.object(packet, "require_available", return_value=lease),
                        patch.object(packet.Path, "home", return_value=home),
                        patch.object(
                            trainer, "configure_cuda", side_effect=AssertionError("GPU queried")
                        ),
                        patch.object(
                            trainer, "block_inputs", side_effect=AssertionError("model loaded")
                        ),
                    ):
                        with self.assertRaises(exception):
                            packet.export_initial(args)
            # Both failed attempts release the global lock so a subsequent owner
            # can obtain it; neither can disturb a held private/global owner.
            with global_path.open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_export_requires_explicit_cuda_before_any_model_or_gpu_call(self):
        with (
            patch.object(trainer, "block_inputs", side_effect=AssertionError("model loaded")),
            patch.object(
                packet, "require_available", side_effect=AssertionError("GPU lease queried")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "allow-cuda"):
                packet.export_initial(SimpleNamespace(allow_cuda=False, packet=Path("/missing")))


if __name__ == "__main__":
    unittest.main()
