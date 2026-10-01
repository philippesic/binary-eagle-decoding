"""CPU filesystem fixtures for manual restart classification; no GPU discovery."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
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
