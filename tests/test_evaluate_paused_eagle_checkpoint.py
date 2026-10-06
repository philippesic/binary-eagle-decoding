"""Paused evaluation joins/lifecycle fixtures; no models, CUDA or SSH."""

import fcntl
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_paused_eagle_checkpoint as paused  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return paused.endpoint.pin(path)


class PausedTests(unittest.TestCase):
    def fixture(self, root):
        training = root / "original-lane/training"
        checkpoint = training / "checkpoints/step-000000000002-e000000-r000000000002"
        base = write(root / "base.gguf", {"fixture": "F16 base bytes"})
        plan = {
            "candidate": "eagle_a8",
            "training_run_dir": str(training.parent),
            "training_source_sha256": "f" * 64,
            "base_model": base,
            "training_limits": {"max_seconds": 86400},
            "source": {
                name: paused.endpoint.pin(ROOT / name)
                for name in (
                    "scripts/run_nine_model_lane_endpoint.py",
                    "scripts/benchmark_native_eagle.py",
                    "scripts/benchmark_dspark_screen.py",
                    "src/w1a1_eagle/nine_model_pipeline.py",
                )
            },
            "gpu_control_path": str(root / "control.json"),
            "gpu_uuid": "GPU-fixture",
            "controller_identity": {"pid": 10, "start_ticks": 1, "boot_id": "old"},
            "supervisor_identity": {"pid": 11, "start_ticks": 1, "boot_id": "old"},
            "runtime": {},
            "environment": {},
            "resource_policy": {},
            "serializer_python_invocation": sys.executable,
            "serializer": paused.endpoint.pin(ROOT / "scripts/export_recurrent_binary.py"),
        }

        def publish(directory, step):
            resume = write(
                directory / "resume.pt", {"fixture": "optimizer RNG cursor", "step": step}
            )
            npz = write(directory / "A8/joint.npz", {"fixture": "all-nine NPZ", "step": step})
            joint = write(
                directory / "A8/joint.json",
                {
                    "checkpoint_sha256": npz["sha256"],
                    "base_gguf_sha256": base["sha256"],
                    "activation_bits": 8,
                    "scale_layout": "row",
                    "projections": {
                        key: {
                            "checkpoint_name": name + ".weight",
                            "shape": paused.shapes.PINNED_SHAPES[name],
                        }
                        for key, name in paused.SOURCE_NAMES.items()
                    },
                },
            )
            write(
                directory / "manifest.json",
                {
                    "schema": "continuous_joint_w1ax_v1",
                    "step": step,
                    "epoch": 0,
                    "cursor": step,
                    "sha256": resume["sha256"],
                    "source_sha256": "f" * 64,
                    "optimizer_rng_cursor_exact": True,
                    "exports": {"A8": {"joint.npz": npz["sha256"], "joint.json": joint["sha256"]}},
                },
            )
            return resume, npz, joint

        resume, _, _ = publish(checkpoint, 2)
        write(
            training / "status.json",
            {
                "status": "stopped",
                "step": 2,
                "checkpoint": {**resume, "step": 2},
                "training_elapsed_seconds": 51950.921720,
            },
        )
        write(
            training / "budget-used.json",
            {
                "schema": "continuous_training_budget_v1",
                "source_sha256": "f" * 64,
                "active_attempt": None,
                "max_seconds": 86400,
                "training_seconds": 51950.921720,
            },
        )
        _, zero_npz, zero_joint = publish(root / "initial/checkpoints/step-zero", 0)
        initial = write(root / "initial.gguf", {"fixture": "exact calibrated zero model"})
        audit = write(
            root / "initial-audit.json",
            {
                "serialization_audit_passed": True,
                "output": initial,
                "base_gguf": base,
                "activation_bits": 8,
                "checkpoint": zero_npz,
                "checkpoint_manifest": zero_joint,
                "projections": {name: {} for name in paused.SOURCE_NAMES},
            },
        )
        write(root / "control.json", {"rtx5080": {"pause_requested": False}})
        args = SimpleNamespace(
            helper_checkout=ROOT,
            plan=root / "plan.json",
            plan_sha256="a" * 64,
            checkpoint_dir=checkpoint,
            expected_resume_sha=resume["sha256"],
            output_root=root / "new-evaluation",
            control_path=root / "control.json",
            initial_model=Path(initial["path"]),
            initial_model_sha256=initial["sha256"],
            initial_audit=Path(audit["path"]),
            initial_audit_sha256=audit["sha256"],
            start=False,
            availability=root / "fresh-availability.json",
            supervisor_state=root / "fresh-supervisor.json",
        )
        policy = {"export_wall_seconds": 600, "evaluation_wall_seconds": 1200}
        protocol = {
            "repetitions": 5,
            "warmups_per_cell": 2,
            "max_output_tokens": 128,
            "split": "development",
            "port": 18290,
        }
        return args, plan, policy, protocol

    def inspect(self, args, plan, policy, protocol):
        with (
            patch.object(paused, "INITIAL_SHA256", args.initial_model_sha256),
            patch.object(
                paused.endpoint,
                "validate_plan",
                return_value=(plan, paused.endpoint.Files(), policy, protocol),
            ),
        ):
            return paused.inspect(args)

    def test_inspection_preserves_interrupted_budget_and_all_original_files(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, policy, protocol = self.fixture(Path(temp).resolve())
            before = {p: p.read_bytes() for p in Path(temp).rglob("*") if p.is_file()}
            result = self.inspect(args, plan, policy, protocol)
            self.assertEqual(result[-1]["training_seconds"], 51950.921720)
            self.assertEqual(result[-1]["max_seconds"], 86400)
            self.assertEqual(result[-2]["optimizer_updates"], 0)
            self.assertFalse(args.output_root.exists())
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_source_resume_joint_and_accounting_rejections(self):
        for mutation, expected in (
            ("source", "publication/source"),
            ("resume", "resume hash"),
            ("npz", "artifact changed"),
            ("shape", "all-nine"),
            ("budget", "accounting"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                args, plan, policy, protocol = self.fixture(Path(temp).resolve())
                outer = paused.read(args.checkpoint_dir / "manifest.json")
                if mutation == "source":
                    outer["source_sha256"] = "d" * 64
                    write(args.checkpoint_dir / "manifest.json", outer)
                elif mutation == "resume":
                    args.expected_resume_sha = "d" * 64
                elif mutation == "npz":
                    (args.checkpoint_dir / "A8/joint.npz").write_text("mutated")
                elif mutation == "shape":
                    joint = paused.read(args.checkpoint_dir / "A8/joint.json")
                    joint["projections"]["fc"]["shape"][0] = 1
                    pin = write(args.checkpoint_dir / "A8/joint.json", joint)
                    outer["exports"]["A8"]["joint.json"] = pin["sha256"]
                    write(args.checkpoint_dir / "manifest.json", outer)
                elif mutation == "budget":
                    budget = paused.read(
                        Path(plan["training_run_dir"]) / "training/budget-used.json"
                    )
                    budget["active_attempt"] = {"live": True}
                    write(Path(plan["training_run_dir"]) / "training/budget-used.json", budget)
                with self.assertRaisesRegex(ValueError, expected):
                    self.inspect(args, plan, policy, protocol)

    def test_exact_initial_model_and_zero_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, policy, protocol = self.fixture(Path(temp).resolve())
            with patch.object(
                paused.endpoint,
                "validate_plan",
                return_value=(plan, paused.endpoint.Files(), policy, protocol),
            ):
                with self.assertRaisesRegex(ValueError, "exact original"):
                    paused.inspect(args)
            directory = Path(temp) / "initial/checkpoints/step-zero"
            outer = paused.read(directory / "manifest.json")
            outer["step"] = 1
            write(directory / "manifest.json", outer)
            with self.assertRaisesRegex(ValueError, "snapshot kind"):
                self.inspect(args, plan, policy, protocol)

    def test_pause_and_lock_contention_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control = root / "control.json"
            write(control, {"rtx5080": {"pause_requested": True}})
            with self.assertRaisesRegex(ValueError, "paused"):
                paused.require_unpaused(control)
            lock = root / "owner.lock"
            owner = {"pid": 123, "start_ticks": 1, "boot_id": "fixture"}
            with paused.owner_lock(lock, owner):
                with self.assertRaises(BlockingIOError):
                    with paused.owner_lock(lock, {**owner, "pid": 456}):
                        self.fail("competing owner acquired exclusive GPU")
                self.assertEqual(paused.read(lock), owner)
            with lock.open() as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_supervisor_must_be_fresh_actual_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(supervisor_state=Path(temp) / "supervisor.json")
            write(
                args.supervisor_state,
                {
                    "pid": os.getpid() + 1,
                    "supervisor_pid": os.getppid(),
                    "status": "running",
                    "stop_grace_seconds": 90,
                },
            )
            with patch.dict(os.environ, {"TMUX": "fixture"}):
                with self.assertRaisesRegex(ValueError, "fresh detached"):
                    paused.supervised(args)

    def test_fresh_lease_rejects_stale_or_other_plan(self):
        import time

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "availability.json"
            now = time.time()
            lease = {
                "schema": "nine_model_gpu_lease_v1",
                "host": "rtx5080",
                "user_announced_available": True,
                "sole_owner": True,
                "pause_requested": False,
                "bundle_sha256": "a" * 64,
                "gpu_uuid": "GPU-fixture",
                "granted_unix": now - 10,
                "expires_unix": now + 10,
            }
            write(path, lease)
            self.assertEqual(paused.require_available(path, "a" * 64), lease)
            with self.assertRaisesRegex(ValueError, "lease absent"):
                paused.require_available(path, "b" * 64)
            lease["expires_unix"] = now - 1
            write(path, lease)
            with self.assertRaisesRegex(ValueError, "stale"):
                paused.require_available(path, "a" * 64)

    def test_aggregation_uses_native_counts_and_token_time_ratios(self):
        report = {
            "cells": {
                "eagle_a8": {
                    "output_tokens": 20,
                    "request_seconds": 4,
                    "native_counts": {"accepted": 3, "proposed": 10, "rounds": 2},
                },
                "eagle_q4": {
                    "output_tokens": 20,
                    "request_seconds": 2,
                    "native_counts": {"accepted": 8, "proposed": 12, "rounds": 4},
                },
                "target_only": {"output_tokens": 20, "request_seconds": 5},
            }
        }
        result = paused.summarize(report)
        self.assertEqual(result["eagle_a8"]["acceptance_rate"], 0.3)
        self.assertEqual(result["eagle_a8"]["accepted_per_round"], 1.5)
        self.assertEqual(result["eagle_a8"]["request_tokens_per_second"], 5)
        self.assertEqual(result["eagle_a8"]["request_speed_vs_q4"], 0.5)
        self.assertNotIn("acceptance_rate", result["target_only"])
        report["cells"]["eagle_a8"]["native_counts"]["proposed"] = 0
        with self.assertRaisesRegex(ValueError, "denominators"):
            paused.summarize(report)

    def test_wall_cap_restores_handler(self):
        import signal

        before = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, "wall cap"):
            with paused.wall_cap(0.01):
                signal.pause()
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def lifecycle(self, *, fail=False):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, policy, protocol = self.fixture(Path(temp).resolve())
            args.start = True
            write(args.plan, {"fixture": "frozen plan"})
            owner = {"pid": 101, "start_ticks": 1, "boot_id": "fresh"}
            supervisor = {"pid": 102, "start_ticks": 2, "boot_id": "fresh"}
            write(
                args.supervisor_state,
                {"pid": owner["pid"], "supervisor_pid": supervisor["pid"], "status": "running"},
            )
            write(args.availability, {"gpu_uuid": plan["gpu_uuid"]})
            calls = []

            class Observer:
                def __init__(self, _uuid):
                    pass

                def snapshot(self):
                    return {"dxg_holders": []}

                def require_released(self, *_args):
                    return {"other_context_pids": []}

            class Runner:
                process_groups = []
                process_identities = []

                def __init__(self, *_args):
                    pass

                def run(self, argv, **_kwargs):
                    self.authorization()
                    calls.append("export")
                    output = Path(argv[argv.index("--output") + 1])
                    audit = Path(argv[argv.index("--audit") + 1])
                    model = write(output, {"fixture": "trained GGUF"})
                    write(
                        audit,
                        {
                            "serialization_audit_passed": True,
                            "base_gguf": plan["base_model"],
                            "checkpoint": paused.endpoint.pin(args.checkpoint_dir / "A8/joint.npz"),
                            "checkpoint_manifest": paused.endpoint.pin(
                                args.checkpoint_dir / "A8/joint.json"
                            ),
                            "output": model,
                            "activation_bits": 8,
                            "projections": {name: {} for name in paused.SOURCE_NAMES},
                        },
                    )

            def evaluate(_plan, _files, _protocol, exported, directory, _stop, authorize):
                authorize()
                calls.append(exported["checkpoint_kind"])
                if fail:
                    raise RuntimeError("actual evaluator failure preserved")
                report = {
                    "cells": {
                        "eagle_a8": {
                            "output_tokens": 20,
                            "request_seconds": 4,
                            "native_counts": {"accepted": 3, "proposed": 10, "rounds": 2},
                        },
                        "eagle_q4": {
                            "output_tokens": 20,
                            "request_seconds": 2,
                            "native_counts": {"accepted": 8, "proposed": 12, "rounds": 4},
                        },
                        "target_only": {"output_tokens": 20, "request_seconds": 5},
                    }
                }
                return {"report": write(directory / "report.json", report)}

            with (
                patch.object(paused, "INITIAL_SHA256", args.initial_model_sha256),
                patch.object(
                    paused.endpoint,
                    "validate_plan",
                    return_value=(plan, paused.endpoint.Files(), policy, protocol),
                ),
                patch.object(paused, "supervised", return_value=(owner, supervisor)),
                patch.object(
                    paused, "require_available", return_value=paused.read(args.availability)
                ),
                patch.object(paused, "LinuxResources", Observer),
                patch.object(paused, "SubprocessRunner", Runner),
                patch.object(paused.endpoint, "owned_training_processes", return_value=([], [])),
                patch.object(
                    paused,
                    "identity_active",
                    side_effect=lambda value: value in (owner, supervisor),
                ),
                patch.object(paused, "resource_gate"),
                patch.object(paused.endpoint, "available_port", return_value=True),
                patch.object(paused.endpoint, "evaluate_endpoint", side_effect=evaluate),
                patch.object(Path, "home", return_value=Path(temp)),
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "failure preserved"):
                        paused.run(args)
                    result = paused.read(args.output_root / "failure.json")
                else:
                    result = paused.run(args)
            if fail:
                self.assertEqual(calls, ["export", "paused_trained_snapshot"])
                self.assertEqual(result["owned_release"]["status"], "PASS")
                self.assertFalse(result["original_training_budget_complete"])
                self.assertEqual(result["error_type"], "RuntimeError")
                self.assertFalse((args.output_root / "result.json").exists())
                self.assertTrue((args.checkpoint_dir / "resume.pt").exists())
                return
            self.assertEqual(calls, ["export", "paused_trained_snapshot", "calibrated_zero_update"])
            self.assertEqual(result["accounted_training_seconds"], 51950.921720)
            self.assertFalse(result["original_training_budget_complete"])
            self.assertFalse(result["campaign_complete"])
            self.assertEqual(result["owned_release"]["status"], "PASS")
            self.assertEqual(len(result["comparisons"]), 2)

    def test_mocked_lifecycle_exports_then_two_independent_comparisons(self):
        self.lifecycle()

    def test_failed_helper_preserves_error_checkpoint_and_release_receipt(self):
        self.lifecycle(fail=True)


if __name__ == "__main__":
    unittest.main()
