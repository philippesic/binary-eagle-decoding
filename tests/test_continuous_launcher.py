"""CPU filesystem fixtures for manual restart classification; no GPU discovery."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_continuous_w1ax.py"
SPEC = importlib.util.spec_from_file_location("continuous_launcher_cpu_tests", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


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

    def test_missing_project_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "resolved config"):
                launcher.resume_kind(Path(folder))


if __name__ == "__main__":
    unittest.main()
