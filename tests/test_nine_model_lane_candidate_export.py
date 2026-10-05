"""Tiny CPU metadata/serializer contracts; no production admission claims."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_nine_model_lane_candidate as api

from w1a1_eagle.block_qat import BlockQATConfig, block_contract
from w1a1_eagle.continuous_qat import ContinuousConfig, immutable_config
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH


class LaneExportTests(unittest.TestCase):
    def fixture(self, root, family="eagle", bits=8):
        root = root.resolve()
        name = f"{family}_a{bits}"
        run = root / "run"
        checkpoint_dir = run / "training/checkpoints/endpoint"
        checkpoint_dir.mkdir(parents=True)

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return api.pin(path)

        base = write(root / "base.gguf", {"tiny": True})
        target = write(root / "target.gguf", {"target": True})
        limits = dict(max_steps=None, max_tokens=None, max_seconds=10, max_epochs=None)
        source = (
            {
                "common_source_sha256": {
                    "target_gguf": target["sha256"],
                    "base_draft_gguf": base["sha256"],
                }
            }
            if family == "eagle"
            else {
                "base_gguf_sha256": base["sha256"],
                "target_sha256": target["sha256"],
                "data_manifest_sha256": "a" * 64,
                "synthetic": False,
            }
        )
        spec = {"candidate": name, "family": family, "precision_stage": "direct"}
        if family == "eagle":
            config = ContinuousConfig(
                activation_bits=(bits,),
                seeds=(1, 2),
                max_seconds=10,
                development_lifecycle="standalone",
            )
            prepared = write(
                root / "prepared/preparation-ready.json",
                {
                    "schema": "continuous_w1ax_preparation_ready_v1",
                    "preparation_complete": True,
                    "optimization_started": False,
                    "teacher_coverage": {"source": source},
                },
            )
            spec["prepared"] = {
                "run_dir": str(root / "prepared"),
                "ready_sha256": prepared["sha256"],
            }
            spec["eagle_config"] = write(
                root / "continuous.json",
                {"schema": "continuous_w1ax_experiment_v1", "training": asdict(config)},
            )
            mapping = CANDIDATE_D_BASE_TO_PATH
            counters = {
                "step": 4,
                "epoch": 1,
                "cursor": 3,
                "elapsed_seconds": 10.0,
                "supervised_tokens": 12,
            }
        else:
            config = BlockQATConfig(family=family, activation_bits=bits)
            data = write(root / "data-manifest.json", {"tiny": True})
            source["data_manifest_sha256"] = data["sha256"]
            spec.update(qat=asdict(config), model=base, limits=limits, data=data)
            mapping = {
                f"blk.{i}.ffn_{p}": f"blk.{i}.ffn_{p}"
                for i in range(5)
                for p in ("gate", "up", "down")
            }
            mapping["fc"] = "fc"
            counters = {
                "step": 4,
                "epoch": 1,
                "block_index": 3,
                "data_cursor": {
                    "dataset_sha256": source["data_manifest_sha256"],
                    "split": "train",
                    "seed": config.seed,
                },
                "supervised_tokens": 12,
                "elapsed_seconds": 10.0,
            }
        serializer = write(root / "serializer.py", {"tiny": True})
        lane = {
            "candidate": name,
            "config": write(root / "config.json", spec),
            "budget": write(
                root / "budget.json", {"candidates": {name: {"training_limits": limits}}}
            ),
            "gpu_uuid": "GPU-test",
            "target_policy": {"immutable": True, "weights": "f16", "kv": "f16"},
            "source": {
                "scripts/export_recurrent_binary.py"
                if family == "eagle"
                else "scripts/export_block_binary.py": serializer
            },
        }
        frozen = write(root / "lane.json", lane)
        from w1a1_eagle.continuous_qat import rng_state

        rng = rng_state("cpu")
        optimizer = {
            "state": {
                0: {
                    "step": torch.tensor(4.0),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.zeros(2),
                }
            },
            "param_groups": [{"params": [0]}],
        }
        linears = {
            key: {
                "latent_sign": torch.ones(2, 32),
                "initial_scale": torch.ones(2),
                "scale_offset": torch.zeros(2),
            }
            for key in mapping.values()
        }
        projection_dir = (
            checkpoint_dir / f"A{bits}" if family == "eagle" else run / "training/final-export"
        )
        projection_dir.mkdir(parents=True)
        projections = {
            base_name: {"checkpoint_name": base_name, "shape": [2, 32]} for base_name in mapping
        }
        arrays = {
            base_name + suffix: (
                torch.ones(2, 32) if suffix == ".latent" else torch.ones(2)
            ).numpy()
            for base_name in mapping
            for suffix in (".latent", ".scale")
        }
        npz = projection_dir / "joint.npz"
        np.savez(npz, **arrays)
        manifest = write(
            projection_dir / "joint.json",
            {
                "checkpoint_sha256": api.sha256(npz),
                "base_gguf_sha256": base["sha256"],
                "activation_bits": bits,
                "projections": projections,
                "family": family,
                "profile": "ffn15_fusion",
            },
        )
        checkpoint = checkpoint_dir / "resume.pt"
        if family == "eagle":
            saved = {
                "schema": "continuous_joint_w1ax_v1",
                "source": api.digest(source),
                "config": asdict(config),
                "training_runtime": {"tiny": True},
                **{key: counters[key] for key in ("step", "epoch", "cursor", "elapsed_seconds")},
                "tokens": 12,
                "global_rng": rng,
                "lanes": {f"A{bits}": {"linears": linears, "rng": rng, "optimizer": optimizer}},
            }
            torch.save(saved, checkpoint)
            write(
                checkpoint_dir / "manifest.json",
                {
                    "schema": saved["schema"],
                    "sha256": api.sha256(checkpoint),
                    "source_sha256": saved["source"],
                    "training_runtime": saved["training_runtime"],
                    "immutable_config": immutable_config(saved["config"]),
                    **{key: counters[key] for key in ("step", "epoch", "cursor")},
                    "optimizer_rng_cursor_exact": True,
                    "exports": {
                        f"A{bits}": {"joint.npz": api.sha256(npz), "joint.json": manifest["sha256"]}
                    },
                },
            )
        else:
            saved = {
                "schema": "block_qat_checkpoint_v1",
                "source": {**source, "bundle_sha256": frozen["sha256"]},
                "contract": block_contract(config),
                "cursor": counters,
                "linears": linears,
                "rng": rng,
                "optimizer": optimizer,
            }
            torch.save(saved, checkpoint)
            write(
                checkpoint.with_suffix(".receipt.json"),
                {
                    "schema": saved["schema"],
                    **api.pin(checkpoint),
                    "contract_sha256": api.digest(saved["contract"]),
                    "source_sha256": api.digest(saved["source"]),
                    "cursor": counters,
                    "committed": True,
                },
            )
        receipt = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": name + "/train",
            "status": "PASS",
            "artifact_kind": "production",
            "committed": True,
            "completion_reason": "approved_budget_complete",
            "bundle_sha256": frozen["sha256"],
            "config_sha256": lane["config"]["sha256"],
            "checkpoint": api.pin(checkpoint),
            "exports": {
                name: {
                    "checkpoint": api.pin(npz),
                    "manifest": manifest,
                    "base_gguf_sha256": base["sha256"],
                }
            },
            "counters": counters,
            "hardware": {"gpu_uuid": "GPU-test", "compute_capability": [12, 0]},
        }
        train = write(run / "train-receipt.json", receipt)
        state = write(
            run / "state.json",
            {
                "schema": "nine_model_lane_state_v1",
                "status": "training_complete",
                "candidate": name,
                "bundle_sha256": frozen["sha256"],
                "campaign_complete": False,
                "owned_release": {"owned_process_groups_absent": True},
                "train_receipt": train,
            },
        )
        supervisor = write(
            root / "supervisor.json",
            {"status": "finished", "exit_code": 0, "received_signal": None},
        )
        python = api.pin(Path(sys.executable).resolve())
        plan = {
            "candidates": {
                name: {
                    "source_bindings": source,
                    "export": write(root / "initial-audit.json", {"base_gguf": base}),
                }
            },
            "target": target,
            "python": python,
            "python_invocation": str(Path(sys.executable).resolve()),
        }
        return dict(
            locators=(frozen, state, supervisor, train),
            lane=lane,
            plan=plan,
            saved=saved,
            receipt=receipt,
            write=write,
        )

    def validate(self, fixture):
        with patch.object(
            api, "validate_lane", return_value=(fixture["lane"], api.Files(), fixture["plan"])
        ):
            return api.validate_endpoint(*fixture["locators"])

    def test_six_original_lane_hashes_and_real_cpu_serialized_joins(self):
        for family in ("eagle", "dspark", "dflash"):
            for bits in (8, 1):
                with self.subTest(family=family, bits=bits), tempfile.TemporaryDirectory() as temp:
                    f = self.fixture(Path(temp), family, bits)
                    context = self.validate(f)
                    self.assertEqual(
                        context["receipt"]["bundle_sha256"], context["frozen_lane"]["sha256"]
                    )
                    self.assertEqual(
                        context["source_bindings"],
                        f["plan"]["candidates"][f["lane"]["candidate"]]["source_bindings"],
                    )

    def test_rejects_zero_partial_failed_fixture_foreign_config_or_hash(self):
        mutations = (
            ("artifact_kind", "synthetic"),
            ("artifact_kind", "fixture"),
            ("status", "FAIL"),
            ("completion_reason", "STOP"),
            ("committed", False),
            ("bundle_sha256", "a" * 64),
            ("config_sha256", "b" * 64),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temp:
                f = self.fixture(Path(temp))
                f["receipt"][key] = value
                self.replace_receipt(f)
                with self.assertRaises(ValueError):
                    self.validate(f)
        for key, value in (
            ("step", 0),
            ("step", True),
            ("elapsed_seconds", 9),
            ("elapsed_seconds", float("inf")),
        ):
            with self.subTest(counter=key), tempfile.TemporaryDirectory() as temp:
                f = self.fixture(Path(temp))
                f["receipt"]["counters"][key] = value
                self.replace_receipt(f)
                with self.assertRaises(ValueError):
                    self.validate(f)

    def replace_receipt(self, f):
        locators = list(f["locators"])
        locators[3] = f["write"](Path(locators[3]["path"]), f["receipt"])
        state = json.loads(Path(locators[1]["path"]).read_text())
        state["train_receipt"] = locators[3]
        locators[1] = f["write"](Path(locators[1]["path"]), state)
        f["locators"] = tuple(locators)

    def test_rejects_signal_or_missing_release(self):
        for index, key, value in (
            (2, "received_signal", 15),
            (2, "status", "running"),
            (1, "owned_release", {}),
        ):
            with tempfile.TemporaryDirectory() as temp:
                f = self.fixture(Path(temp))
                locators = list(f["locators"])
                path = Path(locators[index]["path"])
                data = json.loads(path.read_text())
                data[key] = value
                locators[index] = f["write"](path, data)
                f["locators"] = tuple(locators)
                with self.assertRaises(ValueError):
                    self.validate(f)

    def test_serialized_rng_optimizer_cursor_and_arrays_are_not_sidecar_claims(self):
        for family in ("eagle", "dspark"):
            with tempfile.TemporaryDirectory() as temp:
                f = self.fixture(Path(temp), family)
                saved = copy.deepcopy(f["saved"])
                if family == "eagle":
                    saved["lanes"]["A8"]["optimizer"] = {}
                else:
                    saved["rng"] = {}
                with (
                    patch.object(
                        api, "validate_lane", return_value=(f["lane"], api.Files(), f["plan"])
                    ),
                    self.assertRaises(ValueError),
                ):
                    api.validate_endpoint(*f["locators"], checkpoint_loader=lambda _: saved)
                saved = copy.deepcopy(f["saved"])
                linears = saved["lanes"]["A8"]["linears"] if family == "eagle" else saved["linears"]
                next(iter(linears.values()))["latent_sign"][0, 0] = -1
                with (
                    patch.object(
                        api, "validate_lane", return_value=(f["lane"], api.Files(), f["plan"])
                    ),
                    self.assertRaisesRegex(ValueError, "export arrays"),
                ):
                    api.validate_endpoint(*f["locators"], checkpoint_loader=lambda _: saved)

    def audit(self, context, directory):
        directory.mkdir(exist_ok=True)
        (directory / "trained.gguf").write_bytes(b"tiny model")
        exported = context["exported"]
        result = {
            "serialization_audit_passed": True,
            "activation_bits": int(context["lane"]["candidate"].split("_a")[1]),
            "projections": json.loads(Path(exported["manifest"]["path"]).read_text())[
                "projections"
            ],
            "output": api.pin(directory / "trained.gguf"),
        }
        if context["spec"]["family"] == "eagle":
            result.update(
                base_gguf=context["base_model"],
                checkpoint=exported["checkpoint"],
                checkpoint_manifest=exported["manifest"],
            )
        else:
            result.update(
                schema="block_binary_export_v1",
                family=context["spec"]["family"],
                profile="ffn15_fusion",
                base_gguf={"sha256": context["base_model"]["sha256"]},
                checkpoint={"sha256": exported["checkpoint"]["sha256"]},
                manifest={"sha256": exported["manifest"]["sha256"]},
            )
        api.atomic_json(directory / "export-audit.json", result)

    def test_cpu_export_callbacks_pins_hidden_cuda_immutable_failure_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(Path(temp))
            context = self.validate(f)
            directory = Path(temp).resolve() / "export"
            calls = []

            def serializer(command, **kwargs):
                self.assertEqual(command[1], context["serializer_source"]["path"])
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.audit(context, directory)

            with patch.object(api, "validate_endpoint", return_value=context):
                result = api.export_endpoint(
                    context,
                    directory,
                    release_check=lambda _: calls.append("release") or True,
                    cpu_admission=lambda *_: calls.append("cpu") or True,
                    run=serializer,
                )
                self.assertEqual(calls, ["release", "cpu", "release"])
                self.assertEqual(result["bundle_sha256"], f["locators"][0]["sha256"])
                self.assertFalse(result["runtime_dense_fallback_checked"])
                with self.assertRaises(ValueError):
                    api.export_endpoint(
                        context,
                        directory,
                        release_check=lambda _: True,
                        cpu_admission=lambda *_: True,
                        run=serializer,
                    )
                failed = Path(temp).resolve() / "failed"

                def fail(command, **kwargs):
                    (failed / "raw.txt").write_text("failure retained")
                    raise RuntimeError("failed serializer")

                with self.assertRaises(RuntimeError):
                    api.export_endpoint(
                        context,
                        failed,
                        release_check=lambda _: True,
                        cpu_admission=lambda *_: True,
                        run=fail,
                    )
                self.assertTrue((failed / "raw.txt").exists())
                self.assertFalse((failed / "receipt.json").exists())
                with self.assertRaises(ValueError):
                    api.export_endpoint(
                        context,
                        Path(temp).resolve() / "blocked",
                        release_check=lambda _: False,
                        cpu_admission=lambda *_: True,
                        run=serializer,
                    )
                self.assertFalse((Path(temp).resolve() / "blocked").exists())

    def test_original_eagle_export_receipt_imports_without_new_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(Path(temp))
            context = self.validate(f)
            directory = Path(temp).resolve() / "export"
            self.audit(context, directory)
            result = {
                "schema": "nine_model_lane_endpoint_export_v1",
                "artifact_kind": "production",
                "candidate": context["lane"]["candidate"],
                "frozen_lane": context["frozen_lane"],
                "training_receipt": context["train_receipt"],
                "checkpoint": context["checkpoint"],
                "model": api.pin(directory / "trained.gguf"),
                "audit": api.pin(directory / "export-audit.json"),
                "campaign_complete": False,
            }
            locator = f["write"](directory / "old-receipt.json", result)
            self.assertEqual(api.validate_export(context, locator), result)

    def test_actual_family_serializer_audit_interfaces_for_all_six(self):
        from export_block_binary import export_model as block_export
        from export_recurrent_binary import SOURCE_NAMES
        from export_recurrent_binary import export_model as eagle_export
        from test_block_binary_export import BlockExportTests
        from test_recurrent_binary_export import RecurrentBinaryExportTests

        from w1a1_eagle.recurrent_qat import (
            JointQATConfig,
            RowBinaryLinear,
            W1AxContract,
            save_joint_checkpoint,
        )

        for family in ("eagle", "dspark", "dflash"):
            for bits in (8, 1):
                with self.subTest(family=family, bits=bits), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp).resolve()
                    f = self.fixture(root / "metadata", family, bits)
                    context = self.validate(f)
                    if family == "eagle":
                        helper = RecurrentBinaryExportTests()
                        helper.setUp()
                        self.addCleanup(helper.doCleanups)
                        base, checkpoint, manifest = (
                            path.resolve()
                            for path in (helper.base, helper.checkpoint, helper.manifest)
                        )
                        linears = {
                            CANDIDATE_D_BASE_TO_PATH[name]: RowBinaryLinear(
                                torch.from_numpy(helper.arrays[key + ".latent"]),
                                torch.from_numpy(helper.arrays[key + ".scale"][:, 0].copy()),
                                W1AxContract(bits),
                            )
                            for name, key in SOURCE_NAMES.items()
                        }
                        checkpoint.unlink()
                        manifest.unlink()
                        save_joint_checkpoint(
                            linears,
                            JointQATConfig(W1AxContract(bits)),
                            api.sha256(base),
                            checkpoint,
                            manifest,
                        )
                        serializer = eagle_export
                    else:
                        source = root / "tiny-gguf"
                        source.mkdir()
                        base, checkpoint, manifest, _ = BlockExportTests().fixture(
                            source, family, bits=bits
                        )
                        serializer = block_export
                    directory = root / "actual-export"
                    directory.mkdir()
                    model = directory / "trained.gguf"
                    audit = serializer(base, checkpoint, manifest, model)
                    api.atomic_json(directory / "audit.json", audit)
                    context["base_model"] = api.pin(base)
                    context["exported"] = {
                        "checkpoint": api.pin(checkpoint),
                        "manifest": api.pin(manifest),
                    }
                    result = {
                        "schema": "nine_model_lane_endpoint_export_v1",
                        "artifact_kind": "production",
                        "candidate": context["lane"]["candidate"],
                        "frozen_lane": context["frozen_lane"],
                        "training_receipt": context["train_receipt"],
                        "checkpoint": context["checkpoint"],
                        "model": api.pin(model),
                        "audit": api.pin(directory / "audit.json"),
                        "campaign_complete": False,
                    }
                    locator = f["write"](directory / "receipt.json", result)
                    self.assertEqual(api.validate_export(context, locator), result)
