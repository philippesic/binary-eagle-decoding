"""CPU fixtures exercise lifecycle failures without certifying native readiness."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w1a1_eagle.nine_model_pipeline import (
    CANDIDATES,
    CELLS,
    GATES,
    Campaign,
    Files,
    SubprocessRunner,
    atomic_json,
    clean_environment,
    require_available,
    resource_gate,
    sha256,
    validate_bundle,
)
from w1a1_eagle.nine_model_report import aggregate


class ResourceFixture:
    def __init__(self):
        self.calls = 0
        self.fail_after = None
        self.owned_active = False

    def require_released(self, pgids, identities=()):
        if self.owned_active:
            raise ValueError("owned CUDA context still active")
        return {"fixture": True}

    def snapshot(self):
        self.calls += 1
        return {
            "host_available_bytes": 1000 if self.calls != self.fail_after else 0,
            "gpu_free_bytes": 1000,
            "gpu_uuid": "fixture",
        }


class StageFixture:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.process_groups = []
        self.process_identities = []
        self.fail_stage = None
        self.stop_stage = None
        self.export_bad = False
        self.partial = False

    def run(self, argv, *, directory, stop_path, wall_seconds):
        stage = argv[1]
        self.calls.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError("fixture child failed")
        if self.stop_stage == stage:
            stop_path.touch()
            raise InterruptedError("fixture STOP")
        receipt = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": stage,
            "status": "PASS",
            "artifact_kind": "fixture",
            "bundle_sha256": "a" * 64,
        }
        checkpoint = self.root / "checkpoint"
        if not checkpoint.exists():
            checkpoint.write_text("actual fixture checkpoint bytes")
        if stage == "admission":
            receipt.update(
                gates=dict.fromkeys(sorted(GATES), "PASS"),
                optimizer_updates=0,
                compute_capability=[12, 0],
            )
        elif stage.endswith("train"):
            receipt.update(
                committed=True,
                completion_reason="STOP" if self.partial else "approved_budget_complete",
                checkpoint={"path": str(checkpoint), "sha256": sha256(checkpoint)},
            )
        elif stage.endswith("export"):
            receipt.update(
                serialization_audit_passed=not self.export_bad,
                no_dense_fallback=True,
                model={"path": str(checkpoint), "sha256": sha256(checkpoint)},
            )
        elif stage == "evaluation":
            receipt.update(
                native=True,
                cells=list(CELLS),
                target_only_diagnostic=True,
                report={"path": str(checkpoint), "sha256": sha256(checkpoint)},
            )
        atomic_json(directory / "receipt.json", receipt)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "source.py"
        source.write_text("fixture source")
        pin = {"path": str(source), "sha256": sha256(source)}

        def stage(name):
            return {
                "argv": [str(source), name, "{receipt}", "{resume}"],
                "producer": pin,
                "wall_seconds": 10,
            }

        self.bundle = {
            "source": {"source": pin},
            "inputs": {"target": pin},
            "controls": {},
            "admission": stage("admission"),
            "evaluation": stage("evaluation"),
            "resource_policy": {
                "host_floor_bytes": 500,
                "gpu_floor_bytes": 500,
                "host_return_tolerance_bytes": 10,
                "gpu_return_tolerance_bytes": 10,
            },
            "candidates": {
                c: {"stages": {"train": stage(c + "/train"), "export": stage(c + "/export")}}
                for c in CANDIDATES
            },
        }
        self.runner = StageFixture(self.root)
        self.resources = ResourceFixture()

    def campaign(self):
        return Campaign(
            self.bundle,
            "a" * 64,
            Files(),
            self.root,
            self.root / "run",
            self.runner,
            self.resources,
            fixture=True,
        )

    def test_success_is_checkpoint_export_release_fresh_eval(self):
        state = self.campaign().execute()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(
            self.runner.calls,
            [
                "admission",
                *(stage for c in CANDIDATES for stage in (c + "/train", c + "/export")),
                "evaluation",
            ],
        )
        self.assertEqual(state["artifact_kind"], "fixture")
        self.assertGreaterEqual(self.resources.calls, 9)

    def test_failed_export_retains_checkpoint_and_no_eval(self):
        self.runner.export_bad = True
        with self.assertRaisesRegex(ValueError, "export contract"):
            self.campaign().execute()
        self.assertTrue((self.root / "checkpoint").is_file())
        self.assertNotIn("evaluation", self.runner.calls)
        self.assertEqual(json.loads((self.root / "run/state.json").read_text())["status"], "failed")

    def test_stop_and_budget_incomplete_never_evaluate(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                if (self.root / "run").exists():
                    import shutil

                    shutil.rmtree(self.root / "run")
                self.runner.stop_stage = None if partial else CANDIDATES[0] + "/train"
                self.runner.partial = partial
                with self.assertRaises((InterruptedError, ValueError)):
                    self.campaign().execute()
                self.assertNotIn("evaluation", self.runner.calls)

    def test_resource_return_failure_blocks_next_train_and_eval(self):
        self.resources.fail_after = 2
        with self.assertRaisesRegex(ValueError, "host resource return"):
            self.campaign().execute()
        self.assertEqual(self.runner.calls[-1], "admission")
        self.assertNotIn("evaluation", self.runner.calls)

    def test_source_changed_before_launch_refuses(self):
        Path(self.bundle["source"]["source"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.campaign().execute()
        self.assertEqual(self.runner.calls, [])

    def test_owned_context_blocks_eval_even_after_memory_return(self):
        self.resources.owned_active = True
        with self.assertRaisesRegex(ValueError, "owned CUDA context"):
            self.campaign().execute()
        self.assertNotIn("evaluation", self.runner.calls)

    def test_new_dxg_holder_blocks_return(self):
        baseline = self.resources.snapshot()
        observed = dict(
            baseline, dxg_holders=[{"pid": 123, "start_ticks": 99, "boot_id": "fixture"}]
        )
        with self.assertRaisesRegex(ValueError, "dxg context"):
            resource_gate(observed, baseline, self.bundle["resource_policy"])

    def test_virtual_long_campaign_chains_after_initial_lease_window(self):
        # Initial authorization expires as a freshness observation, but durable
        # ownership authorization persists and STOP/pause checks still apply.
        with patch("w1a1_eagle.nine_model_pipeline.time.time", side_effect=lambda: 999999):
            state = self.campaign().execute()
        self.assertEqual(state["status"], "complete")

    def test_resume_skips_committed_training_but_rechecks_export(self):
        self.runner.fail_stage = CANDIDATES[0] + "/export"
        with self.assertRaises(RuntimeError):
            self.campaign().execute()
        self.runner.fail_stage = None
        self.runner.calls.clear()
        self.campaign().execute(resume=True)
        self.assertNotIn(CANDIDATES[0] + "/train", self.runner.calls)
        self.assertIn(CANDIDATES[0] + "/export", self.runner.calls)

    def test_resume_changed_checkpoint_is_not_reinitialized(self):
        self.runner.fail_stage = CANDIDATES[0] + "/export"
        with self.assertRaises(RuntimeError):
            self.campaign().execute()
        (self.root / "checkpoint").write_text("mutated")
        self.runner.fail_stage = None
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.campaign().execute(resume=True)

    def test_fixture_bundle_never_grants_portable_production_readiness(self):
        path = self.root / "bundle.json"
        atomic_json(path, {"schema": "nine_model_campaign_bundle_v1", "artifact_kind": "fixture"})
        with self.assertRaisesRegex(ValueError, "fixture"):
            validate_bundle(path)

    def test_ambient_flags_cleared(self):
        with patch.dict(
            os.environ,
            {
                "GGML_CUDA_FORCE_CUBLAS": "1",
                "DSPARK_PROFILE": "1",
                "W1AX_DENSE": "1",
                "LD_PRELOAD": "surprise",
            },
        ):
            env = clean_environment()
        self.assertNotIn("GGML_CUDA_FORCE_CUBLAS", env)
        self.assertNotIn("DSPARK_PROFILE", env)
        self.assertNotIn("LD_PRELOAD", env)
        with self.assertRaisesRegex(ValueError, "experimental"):
            clean_environment({"GGML_DENSE": "1"})

    def test_stale_paused_unannounced_lease_refuses(self):
        path = self.root / "lease.json"
        valid = {
            "schema": "nine_model_gpu_lease_v1",
            "host": "rtx5080",
            "sole_owner": True,
            "user_announced_available": True,
            "pause_requested": False,
            "bundle_sha256": "a" * 64,
            "gpu_uuid": "GPU-fixture",
            "granted_unix": 100,
            "expires_unix": 200,
        }
        for changes in (
            {"pause_requested": True},
            {"user_announced_available": False},
            {"expires_unix": 110},
            {"host": "rtx2080ti"},
        ):
            atomic_json(path, dict(valid, **changes))
            with self.assertRaises(ValueError):
                require_available(path, "a" * 64, now=150)
        atomic_json(path, valid)
        require_available(path, "a" * 64, now=150)

    def test_gpu_memory_recovery_required_even_without_processes(self):
        with self.assertRaisesRegex(ValueError, "gpu resource"):
            resource_gate(
                {"host_available_bytes": 1000, "gpu_free_bytes": 500, "gpu_uuid": "fixture"},
                self.resources.snapshot(),
                self.bundle["resource_policy"],
            )


class PortableProcessTests(unittest.TestCase):
    def test_child_signal_mask_is_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            runner = SubprocessRunner(root)
            code = (
                "import signal; blocked=signal.pthread_sigmask(signal.SIG_BLOCK, []); "
                "assert signal.SIGTERM not in blocked; print('unblocked')"
            )
            runner.run(
                [sys.executable, "-c", code],
                directory=root / "mask",
                stop_path=root / "STOP",
                wall_seconds=10,
            )
            self.assertIn("unblocked", (root / "mask/stdout.log").read_text())

    def test_pause_callback_stops_active_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()

            def paused():
                raise InterruptedError("user paused")

            runner = SubprocessRunner(root, authorization=paused)
            with self.assertRaisesRegex(InterruptedError, "user paused"):
                runner.run(
                    [sys.executable, "-c", "import time; time.sleep(20)"],
                    directory=root / "paused",
                    stop_path=root / "STOP",
                    wall_seconds=10,
                )
            self.assertTrue((root / "paused/process-exit.json").is_file())


@unittest.skipUnless(sys.platform == "linux", "owned /proc cleanup is Linux-only")
class ProcessTests(unittest.TestCase):
    def test_child_failure_and_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            runner = SubprocessRunner(root)
            with self.assertRaisesRegex(ValueError, "exit code 3"):
                runner.run(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    directory=root / "failure",
                    stop_path=root / "STOP",
                    wall_seconds=10,
                )
            with self.assertRaises(TimeoutError):
                runner.run(
                    [sys.executable, "-c", "import time; time.sleep(20)"],
                    directory=root / "timeout",
                    stop_path=root / "STOP",
                    wall_seconds=0.1,
                )
            self.assertTrue(
                json.loads((root / "timeout/process-exit.json").read_text())["owned_group_cleaned"]
            )

    def test_leader_exit_descendant_cleaned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            runner = SubprocessRunner(root)
            code = (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])"
            )
            runner.run(
                [sys.executable, "-c", code],
                directory=root / "children",
                stop_path=root / "STOP",
                wall_seconds=10,
            )
            for pgid in runner.process_groups:
                self.assertNotEqual(
                    subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode, 0
                )


class ReportTests(unittest.TestCase):
    def measurement(self):
        return {
            "schema": "nine_model_native_measurements_v1",
            "native": True,
            "instrumentation_in_clean_timing": False,
            "compute_capability": [12, 0],
            "hardware": "fixture",
            "protocol": {},
            "target": {},
            "model_ancestry": {},
            "records": [
                {
                    "cell": c,
                    "repetition": rep,
                    "prompt_id": "opaque-id",
                    "output_tokens": 10,
                    "latency_s": 2 if c == "dspark_q4" else 1,
                }
                for rep in range(5)
                for c in (*CELLS, "target_only")
            ],
        }

    def test_family_q4_is_primary_denominator(self):
        result = aggregate(self.measurement())
        self.assertEqual(result["cells"]["dspark_a8"]["speed_vs_family_q4"], 2)
        self.assertEqual(result["cells"]["dspark_a8"]["speed_vs_eagle_q4"], 1)

    def test_missing_pairs_repeats_or_instrumentation_rejected(self):
        for case in ("pair", "repeat", "instrumentation"):
            data = self.measurement()
            if case == "pair":
                data["records"].pop()
            elif case == "repeat":
                data["records"] = [r for r in data["records"] if r["repetition"] < 4]
            else:
                data["instrumentation_in_clean_timing"] = True
            with self.assertRaises(ValueError):
                aggregate(data)


if __name__ == "__main__":
    unittest.main()
