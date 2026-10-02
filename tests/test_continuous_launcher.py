"""CPU filesystem fixtures for manual restart classification; no GPU discovery."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_continuous_w1ax.py"
SPEC = importlib.util.spec_from_file_location("continuous_launcher_cpu_tests", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

HEALTH_SPEC = importlib.util.spec_from_file_location(
    "launcher_health_cpu_tests", SCRIPT.with_name("check_continuous_w1ax_health.py")
)
health = importlib.util.module_from_spec(HEALTH_SPEC)
HEALTH_SPEC.loader.exec_module(health)


class LauncherPreparationTests(unittest.TestCase):
    """Exercise main end to end with CPU state and forbidden accelerator/update calls."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "run"
        self.config_path = self.root / "config.json"
        self.spec = json.loads((launcher.ROOT / "configs/continuous_w1ax.json").read_text())
        self.spec.update(stages={"fixture": True}, development={"fixture": True})
        self.spec["coverage"] = {"min_unique_train_prompts": 1, "min_unique_supervised_rows": 2}
        self.config_path.write_text(json.dumps(self.spec))
        self.events = []
        self.batch = types.SimpleNamespace(
            anchor=types.SimpleNamespace(prompt_id="train-fixture"),
            prefix_token_ids=(1, 2), rows=(1, 2),
        )
        self.provider = types.SimpleNamespace(
            allowed_prompt_ids={"train-fixture"}, source_metadata={"fixture": "audited"},
            rounds=lambda: iter([self.batch]),
        )
        self.parameters = [torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0])]
        self.lanes = [
            types.SimpleNamespace(
                name=name,
                optimizer=types.SimpleNamespace(
                    step=Mock(side_effect=AssertionError("forbidden optimizer update")),
                    state={"parameter": {
                        "step": torch.tensor(0.0), "exp_avg": torch.zeros(2),
                        "exp_avg_sq": torch.zeros(2),
                    }},
                ),
            ) for name in ("A8", "A1")
        ]
        self.before = [parameter.clone() for parameter in self.parameters]
        self.before_optimizer = [
            {key: value.clone() for key, value in lane.optimizer.state["parameter"].items()}
            for lane in self.lanes
        ]
        self.resource_snapshot = {
            "disk_free_bytes": 40 * 1024**3, "host_available_bytes": 3 * 1024**3,
            "cuda_free_bytes": 2 * 1024**3, "cuda_reserved_bytes": 8 * 1024**3,
            "cuda_peak_bytes": 9 * 1024**3, "cuda_allocated_bytes": 7 * 1024**3,
        }
        self.smoke_error = None
        self.smoke_optimizer_step = 0
        self.restored_step = 0
        self.restored_lane_step = 0
        self.restored_optimizer_step = 0
        self.saved_step = 0
        fixture = self

        class FixtureTrainer:
            def __init__(self, provider, lanes, config, run_dir, development_evaluator):
                fixture.events.append("trainer")
                fixture.trainer = self
                self.step = 0
                self.lanes = lanes
                self.metrics = {name: {"step": 0} for name in ("A8", "A1")}
                self.source = "a" * 64
                self.runtime_identity = {"math_source_sha256": {"fixture": "b" * 64}}
                self.smoke_passed = False
                self.checkpoint = None
                self.evaluator = development_evaluator

                def run_training():
                    fixture.events.append("run")
                    for lane in self.lanes:
                        lane.optimizer.step()

                self.run = Mock(side_effect=run_training)

            def resume(self):
                fixture.events.append("resume")
                self.step = fixture.restored_step
                self.metrics["A8"]["step"] = fixture.restored_lane_step
                self.lanes[0].optimizer.state["parameter"]["step"].fill_(
                    fixture.restored_optimizer_step
                )

            def smoke(self, batch):
                fixture.events.append("smoke")
                if fixture.smoke_error:
                    raise fixture.smoke_error
                # Stand in for the paired CUDA backward boundary without running it.
                self.lanes[0].optimizer.state["parameter"]["step"].fill_(
                    fixture.smoke_optimizer_step
                )
                self.smoke_passed = True
                launcher.atomic_json(fixture.run_dir / "dual_smoke.json", {"passed": True})

            def save(self):
                fixture.events.append("save")
                path = fixture.run_dir / "checkpoint.pt"
                path.write_bytes(b"paired initial checkpoint fixture")
                self.checkpoint = {
                    "path": str(path), "sha256": launcher.sha256(path),
                    "step": fixture.saved_step,
                }
                launcher.atomic_json(fixture.run_dir / "latest.json", self.checkpoint)

            def resources(self):
                fixture.events.append("resources")
                return fixture.resource_snapshot

            def status(self, status, **extra):
                fixture.events.append("status:" + status)
                launcher.atomic_json(fixture.run_dir / "status.json", {
                    "schema": "continuous_joint_w1ax_v1", "status": status,
                    "step": self.step, "models": self.metrics,
                    "checkpoint": self.checkpoint, **extra,
                })

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.cuda_available = self.stack.enter_context(
            patch.object(torch.cuda, "is_available", return_value=True)
        )
        self.stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=(
            types.SimpleNamespace(name=self.spec["hardware"]["device_name"], major=12,
                                  minor=0, total_memory=16 * 1024**3)
        )))
        for name in ("mem_get_info", "memory_reserved", "memory_allocated",
                     "max_memory_reserved", "synchronize", "empty_cache", "init"):
            self.stack.enter_context(patch.object(
                torch.cuda, name, side_effect=AssertionError("real CUDA boundary: " + name)
            ))
        self.stack.enter_context(patch.object(Path, "home", return_value=self.root))
        self.stack.enter_context(patch.object(
            launcher, "training_runtime_identity", return_value={}
        ))
        self.provider_pair = self.stack.enter_context(patch.object(
            launcher, "provider_pair", side_effect=lambda *args: self.record("provider", [
                self.provider, self.provider
            ])
        ))
        self.stack.enter_context(patch.object(
            launcher, "audit_provider_round", side_effect=lambda *args: self.record(
                "audit", types.SimpleNamespace(ce_mask=(True, True))
            )
        ))
        self.build = self.stack.enter_context(patch.object(
            launcher, "build_lanes", side_effect=lambda *args: self.record("build", self.lanes)
        ))
        self.stack.enter_context(patch.object(launcher, "ContinuousTrainer", FixtureTrainer))
        stages = types.ModuleType("w1ax_continuous_stages")
        stages.run_stages = Mock(side_effect=lambda *args: self.record(
            "capture_audit_readiness", self.root / "audited-provider.json"
        ))
        stages.evaluate_development = Mock(side_effect=AssertionError("forbidden evaluation"))
        self.stages = stages
        self.stack.enter_context(patch.dict(sys.modules, {"w1ax_continuous_stages": stages}))

    def record(self, event, result):
        self.events.append(event)
        return result

    def launch(self, *options):
        argv = [str(SCRIPT), "--start", "--allow-cuda", "--config", str(self.config_path),
                "--run-dir", str(self.run_dir), *options]
        with patch.object(sys, "argv", argv):
            launcher.main()

    def resume_fixture(self, step=0):
        self.run_dir.mkdir()
        launcher.atomic_json(self.run_dir / "resolved_config.json", self.spec)
        launcher.atomic_json(self.run_dir / "status.json", {
            "schema": "continuous_joint_w1ax_v1", "status": "stopped", "step": step,
            "models": {name: {"step": step} for name in ("A8", "A1")},
        })
        directory = self.run_dir / "checkpoints/step-fixture"
        directory.mkdir(parents=True)
        launcher.atomic_json(directory / "manifest.json", {"step": step})

    def test_preparation_runs_all_gates_and_saves_but_cannot_update_or_evaluate(self):
        self.launch("--prepare-only")
        self.assertEqual(self.events, [
            "capture_audit_readiness", "provider", "audit", "build", "trainer",
            "status:smoke", "smoke", "smoke", "save", "resources", "status:stopped",
        ])
        self.trainer.run.assert_not_called()
        self.stages.evaluate_development.assert_not_called()
        for ordinal, lane in enumerate(self.lanes):
            lane.optimizer.step.assert_not_called()
            self.assertTrue(torch.equal(self.parameters[ordinal], self.before[ordinal]))
            for key, before in self.before_optimizer[ordinal].items():
                self.assertTrue(torch.equal(lane.optimizer.state["parameter"][key], before))
        report = json.loads((self.run_dir / "preparation-ready.json").read_text())
        status = json.loads((self.run_dir / "status.json").read_text())
        for record in (report, status):
            self.assertIs(record["preparation_complete"], True)
            self.assertIs(record["optimization_started"], False)
            self.assertEqual(record["stop_reason"], "prepare_only")
            self.assertEqual(record["step"], 0)
            self.assertEqual(record["models"], {"A8": {"step": 0}, "A1": {"step": 0}})
            self.assertEqual(record["checkpoint"]["step"], 0)
        self.assertEqual(report["resources"], self.resource_snapshot)
        self.assertEqual(report["teacher_coverage"]["unique_supervised_rows"], 2)
        self.assertEqual(status["preparation_report_sha256"], launcher.sha256(
            self.run_dir / "preparation-ready.json"
        ))
        result = health.check_health(self.run_dir / "status.json")
        self.assertTrue(result["healthy"], result["failures"])
        self.assertTrue(result["terminal"])
        self.assertEqual(json.loads((self.run_dir / "resolved_config.json").read_text()), self.spec)
        # Both owner locks are released on the successful early return.
        for path in (self.run_dir / ".owner.lock",
                     self.root / ".cache/binary-eagle-decoding/cuda-0.owner.lock"):
            stream = launcher.lock(path)
            stream.close()

    def test_default_start_still_enters_training_after_smoke_and_save(self):
        for lane in self.lanes:
            lane.optimizer.step.side_effect = None
        self.launch()
        self.trainer.run.assert_called_once_with()
        for lane in self.lanes:
            lane.optimizer.step.assert_called_once_with()
        self.assertEqual(self.events[-4:], ["smoke", "smoke", "save", "run"])
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_failed_native_readiness_cannot_publish_ready_or_construct_models(self):
        self.stages.run_stages.side_effect = ValueError("required A1 native gate failed")
        with self.assertRaisesRegex(ValueError, "A1 native gate"):
            self.launch("--prepare-only")
        self.build.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())
        self.assertEqual(json.loads((self.run_dir / "status.json").read_text())["status"], "failed")

    def test_failed_real_coverage_gate_cannot_publish_ready_or_construct_models(self):
        self.provider.allowed_prompt_ids = set()
        with self.assertRaisesRegex(ValueError, "too few independent train prompts"):
            self.launch("--prepare-only")
        self.build.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_failed_backward_smoke_cannot_save_publish_ready_or_run(self):
        self.smoke_error = ValueError("required A8 later K/V gradient gate failed")
        with self.assertRaisesRegex(ValueError, "gradient gate"):
            self.launch("--prepare-only")
        self.assertNotIn("save", self.events)
        self.trainer.run.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_preparation_can_resume_checkpoint_zero(self):
        self.resume_fixture()
        self.launch("--resume", "--prepare-only")
        self.assertLess(self.events.index("resume"), self.events.index("smoke"))
        self.trainer.run.assert_not_called()
        self.assertTrue((self.run_dir / "preparation-ready.json").is_file())

    def test_interrupted_capture_can_resume_without_checkpoint(self):
        self.run_dir.mkdir()
        launcher.atomic_json(self.run_dir / "resolved_config.json", self.spec)
        launcher.atomic_json(self.run_dir / "status.json", {
            "schema": "continuous_joint_w1ax_v1", "status": "stopped", "models": {},
            "optimization_started": False,
        })
        self.launch("--resume", "--prepare-only")
        self.assertNotIn("resume", self.events)
        self.assertTrue((self.run_dir / "preparation-ready.json").is_file())
        self.trainer.run.assert_not_called()

    def test_trained_resume_rejected_before_capture_cuda_and_config_observation(self):
        self.resume_fixture(step=1)
        with self.assertRaises(SystemExit) as error:
            self.launch("--resume", "--prepare-only")
        self.assertEqual(error.exception.code, 2)
        self.cuda_available.assert_not_called()
        self.stages.run_stages.assert_not_called()
        self.assertFalse(list(self.run_dir.glob("resume-request-*")))
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_trained_orphan_checkpoint_rejected_despite_zero_status(self):
        self.resume_fixture()
        launcher.atomic_json(self.run_dir / "checkpoints/step-fixture/manifest.json", {"step": 1})
        with self.assertRaises(SystemExit):
            self.launch("--resume", "--prepare-only")
        self.cuda_available.assert_not_called()
        self.stages.run_stages.assert_not_called()

    def test_optimizer_log_rejected_despite_zero_checkpoint_and_status(self):
        self.resume_fixture()
        (self.run_dir / "metrics.jsonl.1").write_text('{"step": 1}\n')
        with self.assertRaises(SystemExit):
            self.launch("--resume", "--prepare-only")
        self.cuda_available.assert_not_called()
        self.stages.run_stages.assert_not_called()

    def test_hidden_restored_optimizer_progress_rejected_before_smoke(self):
        self.resume_fixture()
        self.restored_optimizer_step = 1
        with self.assertRaisesRegex(ValueError, "A8 optimizer progress"):
            self.launch("--resume", "--prepare-only")
        self.assertNotIn("smoke", self.events)
        self.trainer.run.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_hidden_restored_global_progress_rejected_before_smoke(self):
        self.resume_fixture()
        self.restored_step = 1
        with self.assertRaisesRegex(ValueError, "global and A8/A1"):
            self.launch("--resume", "--prepare-only")
        self.assertNotIn("smoke", self.events)
        self.trainer.run.assert_not_called()

    def test_hidden_restored_lane_progress_rejected_before_smoke(self):
        self.resume_fixture()
        self.restored_lane_step = 1
        with self.assertRaisesRegex(ValueError, "global and A8/A1"):
            self.launch("--resume", "--prepare-only")
        self.assertNotIn("smoke", self.events)
        self.trainer.run.assert_not_called()

    def test_checkpoint_zero_is_required_for_readiness(self):
        self.saved_step = 1
        with self.assertRaisesRegex(ValueError, "checkpoint zero"):
            self.launch("--prepare-only")
        self.trainer.run.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_optimizer_progress_during_smoke_cannot_publish_ready_or_run(self):
        self.smoke_optimizer_step = 1
        with self.assertRaisesRegex(ValueError, "A8 optimizer progress"):
            self.launch("--prepare-only")
        self.assertIn("save", self.events)
        self.trainer.run.assert_not_called()
        self.assertFalse((self.run_dir / "preparation-ready.json").exists())

    def test_prepare_only_invalid_with_other_actions(self):
        for action in ("--estimate", "--status", "--stop"):
            with self.subTest(action=action), patch.object(
                sys, "argv", [str(SCRIPT), action, "--prepare-only"]
            ), self.assertRaises(SystemExit) as error:
                launcher.main()
            self.assertEqual(error.exception.code, 2)
        self.cuda_available.assert_not_called()


class PreparedCorpusReuseTests(unittest.TestCase):
    """Tiny metadata contracts only: fixtures never establish real data readiness."""

    setUp = LauncherPreparationTests.setUp
    record = LauncherPreparationTests.record
    launch = LauncherPreparationTests.launch

    def completed_fixture(self):
        from w1a1_eagle.continuous_qat import immutable_config

        old = self.root.resolve() / "prepared"
        stage = old / "stages"
        stage.mkdir(parents=True)
        common = {key: str(i) * 64 for i, key in enumerate((
            "target_gguf", "candidate_d_gguf", "base_draft_gguf", "absolute_d2t",
            "model_snapshot_manifest"), 1)}
        sources = {"sha256": dict(common, binary="6" * 64), "native_runtime": {"fixture": True}}
        entries = (("train", "train"), ("development", "development"),
                   ("development", "development-second"))
        prompts = {}
        for _, label in entries:
            prompts[label] = stage / (label + "-prompts.jsonl")
            prompts[label].write_text(json.dumps({"id": label + "-fixture"}) + "\n")
        native_prompts = stage / "native-development-prompts.jsonl"
        native_prompts.write_text("".join(prompts[label].read_text()
                                         for split, label in entries if split == "development"))
        self.spec["stages"] = {
            "sources": sources,
            "captures": [{"split": split, "prompt_count": 1,
                          "prompts": str(prompts[label]),
                          "prompts_sha256": launcher.sha256(prompts[label])}
                         for split, label in entries],
            "development_prompts": str(native_prompts),
            "development_prompts_sha256": launcher.sha256(native_prompts),
        }
        self.spec["development"] = {"from_stages": True}
        self.config_path.write_text(json.dumps(self.spec))
        readiness_path = stage / "readiness.json"
        capture_paths = []
        for split, label in entries:
            capture = stage / (label + "-labels-manifest.json")
            capture.write_text(json.dumps({"split": split, "fixture": label}))
            capture_paths.append(capture)
        launcher.atomic_json(readiness_path, {
            "schema": "w1ax_continuous_readiness_v1", "training_eligible": True,
            "native_binary_sha256": sources["sha256"]["binary"],
            "native_runtime": sources["native_runtime"],
            "teacher_capture_manifest_sha256": [launcher.sha256(p) for p in capture_paths],
        })
        readiness = {"path": str(readiness_path), "sha256": launcher.sha256(readiness_path)}
        attached = {"train": [], "development": []}
        for (split, label), capture in zip(entries, capture_paths):
            child = stage / (label + "-provider.json")
            launcher.atomic_json(child, {
                "schema": "w1ax_native_train_provider_v2", "split": split,
                "training_eligible": split == "train", "prompt_count": 1,
                "paths": {"prompts": str(prompts[label]), "capture_manifest": str(capture)},
                "sha256": dict(common, prompts=launcher.sha256(prompts[label]),
                               capture_manifest=launcher.sha256(capture)),
                "continuous_readiness": readiness,
            })
            attached[split].append({"ordinal": len(attached[split]),
                                    "provider_manifest": str(child),
                                    "provider_manifest_sha256": launcher.sha256(child)})
        for split, records in attached.items():
            launcher.atomic_json(stage / (split + "-providers.json"), {
                "schema": "w1ax_streaming_train_v2", "split": split,
                "training_eligible": split == "train", "shards": records,
                "continuous_readiness": readiness,
            })
        subset = stage / "development-subset-providers.json"
        launcher.atomic_json(subset, json.loads((stage / "development-providers.json").read_text()))
        launcher.atomic_json(stage / "development.json", {
            "schema": "w1ax_continuous_development_v1", "split": "development",
            "sources": sources, "providers_manifest": str(subset),
            "full_pool_manifest": {"path": str(stage / "development-providers.json"),
                                   "sha256": launcher.sha256(stage / "development-providers.json")},
            "full_pool_prompt_count": 2,
            "native_prompts": str(native_prompts),
            "native_prompts_sha256": launcher.sha256(native_prompts),
        })
        source = {"execution_manifest_sha256": launcher.sha256(stage / "train-providers.json"),
                  "common_source_sha256": common, "fixture": "no real corpus loaded"}
        self.provider.source_metadata = source
        source_digest = launcher.prepared_digest(source)
        coverage = {"source": source, "unique_train_prompts": 1, "unique_supervised_rows": 2}
        launcher.atomic_json(old / "teacher_coverage.json", coverage)
        launcher.atomic_json(old / "resolved_config.json", self.spec)
        runtime = {"fixture": "old runtime evidence"}
        launcher.atomic_json(old / "runtime_environment.json", {"training_runtime": runtime})
        checkpoint_dir = old / "checkpoints/step-000000000000-e000000-r000000000000"
        checkpoint_dir.mkdir(parents=True)
        resume = checkpoint_dir / "resume.pt"
        resume.write_bytes(b"evidence only: never deserialized or optimizer resumed")
        exports = {}
        for lane in ("A8", "A1"):
            (checkpoint_dir / lane).mkdir()
            exports[lane] = {}
            for filename in ("joint.npz", "joint.json"):
                path = checkpoint_dir / lane / filename
                path.write_bytes(b"tiny fixture export")
                exports[lane][filename] = launcher.sha256(path)
        checkpoint = {"path": str(resume), "sha256": launcher.sha256(resume), "step": 0}
        _, config = launcher.load_config(self.config_path)
        launcher.atomic_json(checkpoint_dir / "manifest.json", {
            "schema": "continuous_joint_w1ax_v1", "step": 0, "epoch": 0, "cursor": 0,
            "sha256": checkpoint["sha256"], "source_sha256": source_digest,
            "training_runtime": runtime, "optimizer_rng_cursor_exact": True,
            "immutable_config": immutable_config(asdict(config)), "exports": exports,
        })
        launcher.atomic_json(old / "latest.json", checkpoint)
        launcher.atomic_json(old / "dual_smoke.json", {
            lane: {"loss": 1.0, "later_state_gradient_norm": 1.0,
                   "later_k_gradient_norm": 1.0, "later_v_gradient_norm": 1.0}
            for lane in ("A8", "A1")
        })
        ready = {
            "schema": "continuous_w1ax_preparation_ready_v1", "preparation_complete": True,
            "optimization_started": False, "stop_reason": "prepare_only", "step": 0,
            "models": {"A8": {"step": 0}, "A1": {"step": 0}}, "checkpoint": checkpoint,
            "source_sha256": source_digest, "training_runtime": runtime,
            "teacher_coverage": coverage,
            "dual_smoke_sha256": launcher.sha256(old / "dual_smoke.json"),
        }
        launcher.atomic_json(old / "preparation-ready.json", ready)
        ready_sha = launcher.sha256(old / "preparation-ready.json")
        launcher.atomic_json(old / "status.json", dict(ready, schema="continuous_joint_w1ax_v1",
            status="stopped", preparation_report=str(old / "preparation-ready.json"),
            preparation_report_sha256=ready_sha))
        self.stages.validate_readiness = Mock(return_value={})
        self.old, self.ready_sha = old, ready_sha
        return old, ready_sha

    def reuse(self, *extra):
        self.launch("--prepared-run-dir", str(self.old), "--prepared-ready-sha256",
                    self.ready_sha, *extra)

    def test_new_run_reuses_exact_provider_with_no_old_writes_or_old_resume(self):
        self.completed_fixture()
        before = {str(p): p.read_bytes() for p in self.old.rglob("*") if p.is_file()}
        self.reuse("--prepare-only")
        self.stages.run_stages.assert_not_called()
        self.assertNotIn("resume", self.events)
        self.assertEqual(self.provider_pair.call_args.args[0]["manifest"],
                         str(self.old / "stages/train-providers.json"))
        self.assertEqual(self.stages.validate_readiness.call_count, 2)
        after = {str(p): p.read_bytes() for p in self.old.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        binding = json.loads((self.run_dir / "prepared-corpus.json").read_text())
        self.assertEqual(binding["source_sha256"],
                         launcher.prepared_digest(self.provider.source_metadata))
        self.assertEqual(binding["prepared_ready_sha256"], self.ready_sha)
        self.assertTrue((self.run_dir / "checkpoint.pt").exists())
        self.assertIsNotNone(self.trainer.evaluator)

    def test_missing_complete_receipt_rejected_before_cuda_or_output(self):
        self.completed_fixture()
        (self.old / "preparation-ready.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.reuse()
        self.cuda_available.assert_not_called()
        self.assertFalse(self.run_dir.exists())

    def test_changed_receipt_bytes_rejected_before_cuda(self):
        self.completed_fixture()
        (self.old / "preparation-ready.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.reuse()
        self.cuda_available.assert_not_called()

    def test_unfinished_or_nonzero_status_is_never_promoted(self):
        self.completed_fixture()
        path = self.old / "status.json"
        original = json.loads(path.read_text())
        for changes in ({"preparation_complete": False}, {"status": "running"},
                        {"models": {"A8": {"step": 1}, "A1": {"step": 0}}},
                        {"preparation_report_sha256": "f" * 64}):
            with self.subTest(changes=changes):
                launcher.atomic_json(path, dict(original, **changes))
                with self.assertRaises(ValueError):
                    self.reuse()
        self.cuda_available.assert_not_called()

    def test_unjoined_source_and_checkpoint_progress_rejected(self):
        self.completed_fixture()
        checkpoint_path = Path(json.loads((self.old / "latest.json").read_text())["path"])
        manifest_path = checkpoint_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for changes in ({"source_sha256": "f" * 64}, {"step": 1},
                        {"exports": {"A8": manifest["exports"]["A8"]}}):
            with self.subTest(changes=changes):
                launcher.atomic_json(manifest_path, dict(manifest, **changes))
                with self.assertRaises(ValueError):
                    self.reuse()
        self.cuda_available.assert_not_called()

    def test_stage_config_and_model_metadata_must_match(self):
        self.completed_fixture()
        self.spec["stages"]["sources"]["sha256"]["absolute_d2t"] = "f" * 64
        self.config_path.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "immutable stages"):
            self.reuse()
        self.cuda_available.assert_not_called()

    def test_frozen_preparation_accepts_current_native_actor_baseline_metadata(self):
        self.completed_fixture()
        self.spec["native"] = {
            "expected_commit": "9e2c7a90051e738751aab7d7bd7c2d8201fb76e3"
        }
        self.spec["optimization_profile"] = "baseline"
        self.config_path.write_text(json.dumps(self.spec))
        self.reuse("--prepare-only")
        self.stages.run_stages.assert_not_called()
        resolved = json.loads((self.run_dir / "resolved_config.json").read_text())
        self.assertEqual(resolved["native"], self.spec["native"])
        self.assertEqual(resolved["optimization_profile"], "baseline")

    def test_actor_metadata_exception_cannot_change_frozen_data_or_unknown_keys(self):
        self.completed_fixture()
        original = json.loads(json.dumps(self.spec))
        for changes in (
            {"coverage": {"min_unique_train_prompts": 0, "min_unique_supervised_rows": 0}},
            {"hardware": {"device_name": "changed", "compute_capability": [12, 0]}},
            {"provider": {"factory": "different:create", "manifest": None}},
            {"development": {"changed": True}},
            {"model_shapes": [[1, 2]]},
            {"unknown_metadata": {"must_not_be_ignored": True}},
            {"native": {"expected_commit": "9" * 40, "unknown_native_data": True}},
        ):
            with self.subTest(changes=changes):
                self.spec = dict(original, native={"expected_commit": "9" * 40},
                                 optimization_profile="baseline")
                self.spec.update(changes)
                self.config_path.write_text(json.dumps(self.spec))
                with self.assertRaisesRegex(ValueError, "immutable stages"):
                    self.reuse()
        self.cuda_available.assert_not_called()

    def test_actor_metadata_exception_requires_full_commit_and_named_profile(self):
        self.completed_fixture()
        original = json.loads(json.dumps(self.spec))
        for changes in ({"native": {"expected_commit": "9e2"}},
                        {"native": {"expected_commit": "g" * 40}},
                        {"optimization_profile": {}}, {"optimization_profile": ""}):
            with self.subTest(changes=changes):
                self.spec = dict(original, **changes)
                self.config_path.write_text(json.dumps(self.spec))
                with self.assertRaises(ValueError):
                    self.reuse()
        self.cuda_available.assert_not_called()

    def test_changed_provider_artifact_rejected_without_recapture(self):
        self.completed_fixture()
        (self.old / "stages/train-provider.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.reuse()
        self.stages.run_stages.assert_not_called()
        self.build.assert_not_called()

    def test_incomplete_frozen_capture_plan_rejected(self):
        self.completed_fixture()
        path = self.old / "stages/development-providers.json"
        record = json.loads(path.read_text())
        record["shards"] = []
        launcher.atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, "complete frozen capture plan"):
            self.reuse()
        self.cuda_available.assert_not_called()

    def test_development_subset_cannot_omit_a_required_frozen_shard(self):
        self.completed_fixture()
        path = self.old / "stages/development-subset-providers.json"
        subset = json.loads(path.read_text())
        self.assertEqual(len(subset["shards"]), 2)
        subset["shards"] = subset["shards"][:1]
        launcher.atomic_json(path, subset)
        with self.assertRaisesRegex(ValueError, "development subset has unmatched shard ancestry"):
            self.reuse()
        self.cuda_available.assert_not_called()

    def test_new_run_collision_and_ancestry_overlap_rejected(self):
        self.completed_fixture()
        for path in (self.old, self.old / "new", self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "must not overlap"):
                launcher.prepared_corpus_inputs(self.spec, path, self.old, self.ready_sha)
        self.run_dir.mkdir()
        (self.run_dir / "unrelated-output").write_text("preserve")
        with self.assertRaises(SystemExit):
            self.reuse()
        self.assertEqual((self.run_dir / "unrelated-output").read_text(), "preserve")
        self.cuda_available.assert_not_called()

    def test_audited_provider_source_must_equal_original_full_provider(self):
        self.completed_fixture()
        self.provider.source_metadata = {"wrong": "gate-prompt-only provider"}
        with self.assertRaisesRegex(ValueError, "full provider source differs"):
            self.reuse()
        self.build.assert_not_called()
        self.stages.run_stages.assert_not_called()

    def test_new_run_resume_retains_binding_without_repeating_cli(self):
        self.completed_fixture()
        self.reuse("--prepare-only")
        binding_bytes = (self.run_dir / "prepared-corpus.json").read_bytes()
        self.launch("--resume", "--prepare-only")
        self.assertEqual((self.run_dir / "prepared-corpus.json").read_bytes(), binding_bytes)
        self.stages.run_stages.assert_not_called()
        self.assertIn("resume", self.events)
        (self.run_dir / "prepared-corpus.json").unlink()
        with self.assertRaises(SystemExit):
            self.launch("--resume", "--prepare-only")

    def test_resume_rejects_changed_development_artifact(self):
        self.completed_fixture()
        self.reuse("--prepare-only")
        path = self.old / "stages/development.json"
        value = json.loads(path.read_text())
        value["max_wall_seconds"] = 3
        launcher.atomic_json(path, value)
        with self.assertRaisesRegex(SystemExit, "2"):
            self.launch("--resume", "--prepare-only")

    def test_provider_pair_independently_requires_both_eligible_and_identical(self):
        config = launcher.load_config(self.config_path)[1]
        # Use the real provider-pair function for a direct contract check.
        create = types.ModuleType("fixture_prepared_provider")
        a8 = types.SimpleNamespace(source_metadata={"same": True}, training_eligible=True,
                                   full_body_qat_eligible=True)
        a1 = types.SimpleNamespace(source_metadata={"same": True}, training_eligible=True,
                                   full_body_qat_eligible=False)
        create.create = Mock(side_effect=[a8, a1])
        # Read the original definition through a separate module, without invoking main/CUDA.
        module_spec = importlib.util.spec_from_file_location("prepared_provider_contract", SCRIPT)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with patch.dict(sys.modules, {"fixture_prepared_provider": create}):
            with self.assertRaisesRegex(ValueError, "independently for A8 and A1"):
                module.provider_pair(
                    {"factory": "fixture_prepared_provider:create", "manifest": "fixture"}, config
                )
            a1.full_body_qat_eligible = True
            a1.source_metadata = {"different": True}
            create.create.side_effect = [a8, a1]
            with self.assertRaisesRegex(ValueError, "identical immutable"):
                module.provider_pair(
                    {"factory": "fixture_prepared_provider:create", "manifest": "fixture"}, config
                )


class LauncherResumeTests(unittest.TestCase):
    def test_stopped_preoptimization_stage_can_be_retried_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "resolved_config.json").write_text("{}")
            (root / "status.json").write_text(
                json.dumps(
                    {"schema": "continuous_joint_w1ax_v1", "status": "stopped", "models": {}}
                )
            )
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
                self.assertEqual(launcher.resume_kind(root), "preparation")
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "schema": "continuous_joint_w1ax_v1",
                        "status": "failed",
                        "models": {"A8": {"step": 0}, "A1": {"step": 0}},
                    }
                )
            )
            self.assertEqual(launcher.resume_kind(root), "preparation")

    def test_progress_without_durable_checkpoint_is_never_fresh_initialized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "resolved_config.json").write_text("{}")
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "schema": "continuous_joint_w1ax_v1",
                        "status": "failed",
                        "models": {"A8": {"step": 1}, "A1": {"step": 0}},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "optimization evidence"):
                launcher.resume_kind(root)
            (root / "status.json").write_text(
                json.dumps(
                    {"schema": "continuous_joint_w1ax_v1", "status": "stopped", "models": {}}
                )
            )
            (root / "metrics.jsonl.1").write_text('{"step": 1}\n')
            with self.assertRaisesRegex(ValueError, "optimization log"):
                launcher.resume_kind(root)

    def test_orphan_directory_selects_checkpoint_recovery_without_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "checkpoints/step-000000000000-e000000-r000000000000"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{}")
            self.assertEqual(launcher.resume_kind(root), "checkpoint")
            # Detailed directory/hash validation belongs to trainer.resume().

    def test_runtime_startup_is_preserved_and_resume_facts_are_appended(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = {
                "device_name": "fixture",
                "compute_capability": [0, 0],
                "torch_version": "CPU_fixture",
                "cuda_version": None,
                "training_runtime": {"math_source_sha256": {"fixture": "a" * 64}},
            }
            launcher.record_runtime_observation(root, identity)
            original = (root / "runtime_environment.json").read_bytes()
            launcher.record_runtime_observation(root, identity)
            self.assertEqual((root / "runtime_environment.json").read_bytes(), original)
            self.assertEqual(len(list(root.glob("runtime-resume-observation-*"))), 1)
            changed = dict(identity, torch_version="changed_fixture")
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                launcher.record_runtime_observation(root, changed)
            self.assertEqual((root / "runtime_environment.json").read_bytes(), original)
            self.assertEqual(len(list(root.glob("runtime-resume-observation-*"))), 2)

    def test_missing_project_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "resolved config"):
                launcher.resume_kind(Path(folder))


if __name__ == "__main__":
    unittest.main()
