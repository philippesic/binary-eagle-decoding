import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "health", Path(__file__).resolve().parents[1] / "scripts/check_continuous_w1ax_health.py"
)
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "status.json"
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"paired optimizer checkpoint")
        self.state = {
            "status": "running", "heartbeat_unix": 1000,
            "disk_free_bytes": 40 * 1024**3,
            "models": {name: {"step": 4, "loss": 1.2, "grad_finite": True,
                              "heartbeat_unix": 1000} for name in ("A8", "A1")},
            "checkpoint": {"path": "checkpoint.pt", "step": 4,
                           "sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()},
        }

    def check(self, **kwargs):
        self.path.write_text(json.dumps(self.state))
        return health.check_health(self.path, now=1001, **kwargs)

    def test_healthy_and_single_interleaved_step(self):
        self.state["models"]["A8"]["step"] = 5
        self.assertTrue(self.check()["healthy"])

    def test_stall_on_one_model_despite_process_heartbeat(self):
        self.state["models"]["A1"]["heartbeat_unix"] = -2000
        self.assertFalse(self.check()["healthy"])

    def test_nonfinite_and_imbalance(self):
        self.state["models"]["A8"]["loss"] = float("nan")
        self.state["models"]["A1"]["step"] = 1
        result = self.check()
        self.assertEqual(len(result["failures"]), 2)

    def test_checkpoint_corruption_and_external_path_rejected(self):
        self.checkpoint.write_bytes(b"torn")
        self.assertIn("checkpoint unhealthy: checkpoint hash mismatch", self.check()["failures"])
        self.state["checkpoint"]["path"] = "/etc/passwd"
        self.assertFalse(self.check()["healthy"])

    def test_terminal_checkpoint_required_and_no_stall(self):
        self.state["status"] = "stopped"
        self.state["heartbeat_unix"] = 0
        self.assertTrue(self.check()["healthy"])
        self.state["checkpoint"] = None
        self.assertFalse(self.check()["healthy"])

    def test_preparing_phase_has_stage_heartbeat_without_models(self):
        self.state.update(status="preparing", models={}, phase="teacher_capture",
                          optimization_started=False, captures_done=3, captures_total=20)
        self.assertTrue(self.check()["healthy"])
        self.state["heartbeat_unix"] = -2000
        self.assertFalse(self.check()["healthy"])
        self.state.update(status="stopped", heartbeat_unix=0)
        self.assertTrue(self.check()["healthy"])
        self.state["models"] = {"A8": {"step": 1}}
        self.assertFalse(self.check()["healthy"])

    def test_declared_bounded_development_phase_is_healthy(self):
        self.state["status"] = "development_evaluation"
        self.assertTrue(self.check()["healthy"])

    def test_recorded_host_and_cuda_headroom(self):
        self.state["host_available_bytes"] = 3 * 1024**3
        self.state["cuda_free_bytes"] = 2 * 1024**3
        self.assertTrue(self.check()["healthy"])
        self.state["host_available_bytes"] = 1
        self.state["cuda_free_bytes"] = 1
        self.assertEqual(len(self.check()["failures"]), 2)

    def test_malformed_models(self):
        self.state["models"] = None
        self.assertFalse(self.check()["healthy"])

    def test_supervisor_failure(self):
        supervisor = self.root / "state.json"
        supervisor.write_text(json.dumps({"status": "finished", "exit_code": 137}))
        self.assertFalse(self.check(supervisor_path=supervisor)["healthy"])

    def test_supervisor_log_failure_even_after_graceful_training_stop(self):
        self.state["status"] = "stopped"
        supervisor = self.root / "state.json"
        supervisor.write_text(json.dumps({"status": "failed", "exit_code": 74,
                                          "log_error": "disk write failed"}))
        self.assertFalse(self.check(supervisor_path=supervisor)["healthy"])

    def test_missing_status_and_low_disk(self):
        self.assertFalse(health.check_health(self.path)["healthy"])
        self.state["disk_free_bytes"] = 1
        self.assertFalse(self.check()["healthy"])


if __name__ == "__main__":
    unittest.main()
