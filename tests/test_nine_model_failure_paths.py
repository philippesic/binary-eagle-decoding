"""Independent lifecycle failure and cleanup tests for campaign orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from w1a1_eagle.nine_model_pipeline import (
    CANDIDATES,
    CELLS,
    GATES,
    Campaign,
    DiagnosticMemorySampler,
    Files,
    SubprocessRunner,
    atomic_json,
    require,
    sha256,
)


class _Resources:
    def __init__(self):
        self.samples = 0
        self.active = False
        self.fail_release = False

    def snapshot(self):
        self.samples += 1
        return {
            "host_available_bytes": 4096,
            "gpu_free_bytes": 8192,
            "gpu_total_bytes": 16384,
            "gpu_uuid": "fixture",
        }

    def require_released(self, pgids, identities=()):
        if self.fail_release or self.active:
            raise ValueError("owned CUDA context still active")
        return {"owned_process_groups_absent": True, "owned_cuda_pids_absent": True}


class _Stages:
    def __init__(self, root: Path):
        self.root = root
        self.calls: list[str] = []
        self.process_groups: list[int] = []
        self.process_identities: list[dict] = []
        self.fail = None
        self.stop = None
        self.export_valid = True
        self.training_complete = True
        self.candidate_admissions = {}
        self.durable = root / "checkpoint.bin"
        self.durable.write_bytes(b"committed fixture checkpoint")

    def run(self, argv, *, directory, stop_path, wall_seconds):
        stage = argv[1]
        self.calls.append(stage)
        if stage == self.stop:
            stop_path.touch()
            raise InterruptedError("campaign STOP requested")
        if stage == self.fail:
            raise RuntimeError("simulated stage failure")
        receipt = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": stage,
            "status": "PASS",
            "artifact_kind": "fixture",
            "bundle_sha256": "e" * 64,
        }
        if stage == "admission":
            receipt.update(
                gates=dict.fromkeys(sorted(GATES), "PASS"),
                optimizer_updates=0,
                compute_capability=[12, 0],
                candidate_admissions=self.candidate_admissions,
            )
        elif stage.endswith("/train"):
            receipt.update(
                committed=self.training_complete,
                completion_reason=(
                    "approved_budget_complete" if self.training_complete else "STOP"
                ),
                counters={"step": 1 if self.training_complete else 0},
                checkpoint={"path": str(self.durable), "sha256": sha256(self.durable)},
            )
        elif stage.endswith("/export"):
            receipt.update(
                serialization_audit_passed=self.export_valid,
                selected_weights_packed=self.export_valid,
                model={"path": str(self.durable), "sha256": sha256(self.durable)},
            )
        elif stage == "evaluation":
            receipt.update(
                native=True,
                cells=list(CELLS),
                target_only_diagnostic=True,
                report={"path": str(self.durable), "sha256": sha256(self.durable)},
            )
        atomic_json(Path(directory) / "receipt.json", receipt)


class CampaignFailurePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        source = self.root / "source.py"
        source.write_text("pinned fixture source")
        pin = {"path": str(source), "sha256": sha256(source)}

        def stage(name):
            return {
                "argv": [sys.executable, name, "{receipt}"],
                "producer": pin,
                "wall_seconds": 10,
            }

        self.candidate_admissions = {}
        candidates = {}
        for candidate in CANDIDATES:
            config_path = self.root / f"{candidate}.json"
            config_path.write_text(json.dumps({"candidate": candidate}))
            config = {"path": str(config_path), "sha256": sha256(config_path)}
            admission_path = self.root / f"{candidate}-admission.json"
            admission = {
                "schema": "nine_model_training_admission_v1",
                "bundle_sha256": "e" * 64,
                "config_sha256": config["sha256"],
                "status": "PASS",
            }
            atomic_json(admission_path, admission)
            self.candidate_admissions[candidate] = {
                "path": str(admission_path),
                "sha256": sha256(admission_path),
            }
            candidates[candidate] = {
                "config": config,
                "stages": {
                    "train": stage(candidate + "/train"),
                    "export": stage(candidate + "/export"),
                },
            }

        self.bundle = {
            "source": {"campaign": pin},
            "inputs": {"target": pin},
            "controls": {},
            "admission": stage("admission"),
            "evaluation": stage("evaluation"),
            "resource_policy": {
                "host_floor_bytes": 1024,
                "gpu_floor_bytes": 1024,
                "host_return_tolerance_bytes": 256,
                "gpu_return_tolerance_bytes": 256,
            },
            "candidates": candidates,
        }
        self.runner = _Stages(self.root)
        self.runner.candidate_admissions = self.candidate_admissions
        self.resources = _Resources()

    def campaign(self, *, authorization=None):
        return Campaign(
            self.bundle,
            "e" * 64,
            Files(),
            self.root,
            self.root / "campaign-run",
            self.runner,
            self.resources,
            fixture=True,
            authorization=authorization,
        )

    def test_success_orders_all_exports_before_fresh_evaluation(self):
        state = self.campaign().execute()
        expected = ["admission"]
        for candidate in CANDIDATES:
            expected.extend((candidate + "/train", candidate + "/export"))
        expected.append("evaluation")
        self.assertEqual(self.runner.calls, expected)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["artifact_kind"], "fixture")
        self.assertTrue(state["owned_release"]["owned_cuda_pids_absent"])

    def test_stop_keeps_checkpoint_and_never_starts_evaluation(self):
        self.runner.stop = CANDIDATES[0] + "/train"
        with self.assertRaises(InterruptedError):
            self.campaign().execute()
        self.assertTrue(self.runner.durable.is_file())
        state = json.loads((self.root / "campaign-run/state.json").read_text())
        self.assertEqual(state["status"], "stopped")
        self.assertFalse(state["checkpoints_deleted"])
        self.assertNotIn("evaluation", self.runner.calls)

    def test_failed_export_and_incomplete_training_block_evaluation(self):
        self.runner.export_valid = False
        with self.assertRaisesRegex(ValueError, "export contract"):
            self.campaign().execute()
        self.assertTrue(self.runner.durable.is_file())
        self.assertNotIn("evaluation", self.runner.calls)

        import shutil

        shutil.rmtree(self.root / "campaign-run")
        self.runner.calls.clear()
        self.runner.export_valid = True
        self.runner.training_complete = False
        with self.assertRaisesRegex(ValueError, "committed budget"):
            self.campaign().execute()
        self.assertNotIn("evaluation", self.runner.calls)

    def test_resume_reuses_committed_train_receipt_and_retries_export(self):
        self.runner.fail = CANDIDATES[0] + "/export"
        with self.assertRaisesRegex(RuntimeError, "stage failure"):
            self.campaign().execute()
        self.runner.fail = None
        self.runner.calls.clear()
        self.campaign().execute(resume=True)
        self.assertNotIn(CANDIDATES[0] + "/train", self.runner.calls)
        self.assertIn(CANDIDATES[0] + "/export", self.runner.calls)

    def test_resource_or_context_failure_prevents_next_stage(self):
        self.resources.fail_release = True
        with self.assertRaisesRegex(ValueError, "owned CUDA context"):
            self.campaign().execute()
        self.assertEqual(self.runner.calls, ["admission"])
        self.assertNotIn("evaluation", self.runner.calls)

    def test_campaign_keeps_ownership_checks_after_initial_availability_window(self):
        checks = []
        grant_time = 100
        now = [grant_time]

        def authorization():
            now[0] += 60
            checks.append(now[0])
            require(not (self.root / "PAUSED").exists(), "host paused")

        campaign = self.campaign(authorization=authorization)
        campaign.fixture = False
        state = campaign.execute()
        self.assertEqual(state["status"], "complete")
        self.assertGreater(now[0], grant_time + 300)
        self.assertGreater(len(checks), 1)

    def test_subprocess_stop_and_wall_cap_kill_owned_process_group(self):
        runner = SubprocessRunner(self.root)
        marker = self.root / "started"
        stop_file = self.root / "STOP"
        code = (
            "import pathlib,subprocess,sys,time; "
            "pathlib.Path(sys.argv[1]).write_text('ready'); "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "time.sleep(60)"
        )

        def request_stop():
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            stop_file.touch()

        stopper = threading.Thread(target=request_stop)
        stopper.start()
        with self.assertRaises(InterruptedError):
            runner.run(
                [sys.executable, "-c", code, str(marker)],
                directory=self.root / "stop-test",
                stop_path=stop_file,
                wall_seconds=15,
            )
        stopper.join(timeout=2)
        pgid = runner.process_groups[-1]
        self.assertEqual(
            subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode, 1
        )

    @unittest.skipUnless(sys.platform == "linux", "Linux process ownership is the release gate")
    def test_process_leader_exit_still_cleans_descendant(self):
        runner = SubprocessRunner(self.root)
        code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
        )
        runner.run(
            [sys.executable, "-c", code],
            directory=self.root / "orphan-test",
            stop_path=self.root / "NEVER_STOP",
            wall_seconds=10,
        )
        pgid = runner.process_groups[-1]
        self.assertEqual(
            subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode, 1
        )


class DiagnosticMemorySamplerIndependentTests(unittest.TestCase):
    def test_sampled_diagnostics_report_cadence_and_separate_rss_scope(self):
        resources = _Resources()
        times = iter((10.0, 10.002, 10.3, 10.31))
        with (
            patch("w1a1_eagle.nine_model_pipeline.descendant_identities", return_value=[]),
            patch("w1a1_eagle.nine_model_pipeline.time.monotonic", side_effect=times),
        ):
            sampler = DiagnosticMemorySampler(resources, 987, interval_seconds=0.1, max_samples=2)
            sampler.sample()
            sampler.sample()
            report = sampler.stop()

        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["clean_timing_instrumented"])
        self.assertTrue(report["sample_limit_reached"])
        self.assertEqual(report["observed_start_gap_seconds"]["count"], 1)
        self.assertAlmostEqual(report["observed_start_gap_seconds"]["mean"], 0.3)
        self.assertAlmostEqual(report["longest_observer_duration_seconds"], 0.01)
        self.assertEqual(report["evaluator_and_descendant_process_rss_maxima"], [])
        self.assertIn("may overlap", report["scope"])
        self.assertNotIn("per_kernel_process_rss_maxima", report)


if __name__ == "__main__":
    unittest.main()
