"""Endpoint software fixtures; no released models, device queries or trainer actions."""

import copy
import fcntl
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_nine_model_lane_endpoint as prepare  # noqa: E402
import run_nine_model_lane_endpoint as endpoint  # noqa: E402

from w1a1_eagle.nine_model_pipeline import Files  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prepare.pin(path)


class EndpointTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "original-lane"
        config = write(root / "original-config.json", {"fixture": "unchanged config"})
        lane = write(root / "original-lane.json", {"config": config})
        checkpoint = write(
            run / "training/checkpoints/step-2/resume.pt",
            {"fixture": "unchanged optimizer RNG cursor"},
        )
        joint = write(
            run / "training/checkpoints/step-2/A8/joint.npz", {"fixture": "no tensors or models"}
        )
        base = write(root / "original-f16-base", {"fixture": "not a GGUF"})
        manifest = write(
            run / "training/checkpoints/step-2/A8/joint.json",
            {
                "checkpoint_sha256": joint["sha256"],
                "base_gguf_sha256": base["sha256"],
                "activation_bits": 8,
                "projections": {str(i): {} for i in range(9)},
            },
        )
        write(
            run / "training/checkpoints/step-2/manifest.json",
            {
                "schema": "continuous_joint_w1ax_v1",
                "step": 2,
                "epoch": 0,
                "cursor": 2,
                "sha256": checkpoint["sha256"],
                "source_sha256": "f" * 64,
                "optimizer_rng_cursor_exact": True,
                "exports": {"A8": {"joint.npz": joint["sha256"], "joint.json": manifest["sha256"]}},
            },
        )
        receipt_path = run / "attempts/original/train-receipt.json"
        receipt = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": "eagle_a8/train",
            "status": "PASS",
            "artifact_kind": "production",
            "committed": True,
            "completion_reason": "approved_budget_complete",
            "bundle_sha256": lane["sha256"],
            "config_sha256": config["sha256"],
            "counters": {"step": 2, "epoch": 0, "cursor": 2, "elapsed_seconds": 10},
            "hardware": {"gpu_uuid": "fixture", "compute_capability": [12, 0]},
            "checkpoint": checkpoint,
            "exports": {
                "eagle_a8": {
                    "checkpoint": joint,
                    "manifest": manifest,
                    "base_gguf_sha256": base["sha256"],
                }
            },
        }
        receipt_pin = write(receipt_path, receipt)
        state = {
            "schema": "nine_model_lane_state_v1",
            "status": "training_complete",
            "candidate": "eagle_a8",
            "bundle_sha256": lane["sha256"],
            "campaign_complete": False,
            "owned_release": {"owned_process_groups_absent": True},
            "train_receipt": receipt_pin,
        }
        write(run / "state.json", state)
        supervisor_path = root / "original-supervisor.json"
        supervisor = {
            "status": "finished",
            "pid": 10,
            "supervisor_pid": 11,
            "exit_code": 0,
            "received_signal": None,
        }
        write(supervisor_path, supervisor)

        def identity(pid):
            return {"pid": pid, "start_ticks": 100, "boot_id": "fixture-boot"}

        plan = {
            "candidate": "eagle_a8",
            "frozen_lane": lane,
            "training_run_dir": str(run),
            "training_supervisor_state": str(supervisor_path),
            "controller_identity": identity(10),
            "supervisor_identity": identity(11),
            "gpu_uuid": "fixture",
            "base_model": base,
            "training_source_sha256": "f" * 64,
            "training_limits": {"max_seconds": 10},
        }
        return plan, receipt_path, receipt, supervisor_path, supervisor, state

    def test_draft_reports_pending_and_proposed_protocol_is_not_authorization(self):
        report = prepare.prepare(
            {"schema": "nine_model_lane_endpoint_inputs_v1"}, None, inspect_draft=True
        )
        self.assertEqual(report["status"], "PENDING")
        self.assertFalse(report["gpu_queried"])
        self.assertFalse(report["training_changed"])
        protocol = json.loads((ROOT / "configs/nine-model/protocol.json").read_text())
        with self.assertRaisesRegex(ValueError, "not evaluation authorization"):
            prepare.validate_protocol(
                protocol, {"evaluation_wall_seconds": protocol["evaluation_wall_seconds"]}
            )
        protocol["selection_status"] = "SELECTED_operational_under_overnight_delegation"
        prepare.validate_protocol(
            protocol, {"evaluation_wall_seconds": protocol["evaluation_wall_seconds"]}
        )

    def test_live_training_waits_without_reading_endpoint_or_touching_trainer(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, _, supervisor_path, supervisor, _ = self.fixture(Path(temp).resolve())
            supervisor.update(status="running", exit_code=None)
            write(supervisor_path, supervisor)
            run = Path(plan["training_run_dir"])
            before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
            self.assertIsNone(endpoint.training_endpoint(plan, Files(), active=lambda _: True))
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            self.assertFalse((run / "STOP").exists())
            with self.assertRaisesRegex(ValueError, "handle absent"):
                endpoint.training_endpoint(plan, Files(), active=lambda _: False)

    def test_natural_positive_cap_and_exact_optimizer_export_join(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, receipt_path, receipt, *_ = self.fixture(Path(temp).resolve())
            result = endpoint.training_endpoint(plan, Files(), active=lambda _: False)
            self.assertEqual(result["receipt"]["checkpoint"], receipt["checkpoint"])
            self.assertEqual(result["train_receipt"], prepare.pin(receipt_path))
            self.assertFalse(result["state"]["campaign_complete"])
            self.assertIsNone(endpoint.training_endpoint(plan, Files(), active=lambda _: True))

    def test_stop_partial_zero_wrong_config_hardware_and_source_are_not_endpoints(self):
        changes = (
            lambda r: r.update(committed=False),
            lambda r: r.update(completion_reason="STOP"),
            lambda r: r["counters"].update(step=0),
            lambda r: r["counters"].update(elapsed_seconds=9),
            lambda r: r.update(config_sha256="e" * 64),
            lambda r: r["hardware"].update(gpu_uuid="another-device"),
        )
        for change in changes:
            with tempfile.TemporaryDirectory() as temp:
                plan, path, original, _, _, state = self.fixture(Path(temp).resolve())
                receipt = copy.deepcopy(original)
                change(receipt)
                state["train_receipt"] = write(path, receipt)
                write(Path(plan["training_run_dir"]) / "state.json", state)
                with self.assertRaises(ValueError):
                    endpoint.training_endpoint(plan, Files(), active=lambda _: False)
        with tempfile.TemporaryDirectory() as temp:
            plan, _, _, path, supervisor, _ = self.fixture(Path(temp).resolve())
            supervisor.update(received_signal={"number": 15})
            write(path, supervisor)
            with self.assertRaisesRegex(ValueError, "natural endpoint"):
                endpoint.training_endpoint(plan, Files(), active=lambda _: False)
            supervisor["received_signal"] = None
            write(path, supervisor)
            outer = Path(plan["training_run_dir"]) / "training/checkpoints/step-2/manifest.json"
            record = json.loads(outer.read_text())
            record["source_sha256"] = "a" * 64
            write(outer, record)
            with self.assertRaisesRegex(ValueError, "source differs"):
                endpoint.training_endpoint(plan, Files(), active=lambda _: False)

    def test_real_outer_schema_and_nonboolean_epoch_cursor_are_required(self):
        for change in ({"schema": "other"}, {"epoch": False}, {"cursor": True}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                plan, *_ = self.fixture(Path(temp).resolve())
                path = Path(plan["training_run_dir"]) / "training/checkpoints/step-2/manifest.json"
                outer = json.loads(path.read_text())
                outer.update(change)
                write(path, outer)
                with self.assertRaisesRegex(ValueError, "schema/integer"):
                    endpoint.training_endpoint(plan, Files(), active=lambda _: False)

    def test_matched_native_report_uses_q4_primary_and_rejects_unpaired_or_changed_outputs(self):
        plan = {
            "candidate": "eagle_a8",
            "target_policy": {"immutable": True, "weights": "f16", "kv": "f16"},
            "frozen_lane": {"sha256": "a" * 64},
            "protocol": {"sha256": "b" * 64},
            "remaining_candidates": ["other-five-PENDING"],
        }
        records = [
            {
                "cell": cell,
                "repetition": rep,
                "prompt_id": "development-fixture",
                "output_tokens": 10,
                "latency_s": 1.0 if cell == "eagle_a8" else 2.0,
                "generated_token_ids": [1, 2],
                "speculative": {
                    "proposed": 10,
                    "accepted": 4 if cell == "eagle_a8" else 2,
                    "rounds": 2,
                },
            }
            for rep in range(5)
            for cell in ("eagle_a8", "eagle_q4", "target_only")
        ]
        report = endpoint.aggregate(
            plan, records, [], {"fixture": "not actual hardware"}, fixture=True
        )
        self.assertFalse(report["campaign_complete"])
        self.assertEqual(report["cells"]["eagle_a8"]["speed_vs_original_eagle_q4"], 2)
        self.assertEqual(report["cells"]["eagle_a8"]["acceptance_vs_original_eagle_q4"], 2)
        for altered in (
            records[:-1],
            records + [records[0]],
            [
                dict(row, generated_token_ids=[9]) if row["cell"] == "eagle_a8" else row
                for row in records
            ],
        ):
            with self.assertRaises(ValueError):
                endpoint.aggregate(plan, altered, [], {}, fixture=True)

    def test_separate_serializer_exports_only_exact_committed_joint_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan, *_ = self.fixture(root)
            committed = endpoint.training_endpoint(plan, Files(), active=lambda _: False)
            checkpoint = Path(committed["receipt"]["checkpoint"]["path"])
            before = checkpoint.read_bytes()
            joint = committed["receipt"]["exports"][plan["candidate"]]
            calls = []
            original_script = write(
                root / "original-source/export_recurrent_binary.py",
                {"fixture": "original serializer"},
            )
            plan.update(
                serializer=original_script,
                serializer_python=prepare.pin(Path(sys.executable).resolve()),
                serializer_python_invocation=sys.executable,
            )

            def serialize(argv, **kwargs):
                self.assertEqual(argv[:2], [sys.executable, original_script["path"]])
                self.assertEqual(kwargs["timeout"], 30)
                base, weights, manifest, output, audit_path = (
                    Path(argv[argv.index(flag) + 1])
                    for flag in ("--base", "--checkpoint", "--manifest", "--output", "--audit")
                )
                calls.append((base, weights, manifest))
                output.write_bytes(b"synthetic serializer output; no native model")
                write(
                    audit_path,
                    {
                        "serialization_audit_passed": True,
                        "activation_bits": 8,
                        "projections": {str(i): {} for i in range(9)},
                        "base_gguf": prepare.pin(base),
                        "checkpoint": prepare.pin(weights),
                        "checkpoint_manifest": prepare.pin(manifest),
                        "output": prepare.pin(output),
                    },
                )

            with patch.object(
                endpoint.subprocess,
                "run",
                side_effect=serialize,
            ):
                result = endpoint.export_endpoint(
                    plan, Files(), committed, root / "new-export", wall_seconds=30
                )
            self.assertEqual(
                calls,
                [
                    (
                        Path(plan["base_model"]["path"]),
                        Path(joint["checkpoint"]["path"]),
                        Path(joint["manifest"]["path"]),
                    )
                ],
            )
            self.assertFalse(result["campaign_complete"])
            self.assertEqual(checkpoint.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "preserve endpoint export"):
                endpoint.export_endpoint(
                    plan, Files(), committed, root / "new-export", wall_seconds=30
                )

    def test_clean_and_diagnostic_native_commands_differ_only_in_declared_verbosity(self):
        plan = {
            "runtime": {"binary": {"path": "/fixture/server"}},
            "target": {"path": "/fixture/target"},
        }
        protocol = {
            "context_tokens": 2048,
            "batch_tokens": 32,
            "microbatch_tokens": 32,
            "draft_lengths": {"eagle": 5},
        }
        for cell in ("eagle_a8", "eagle_q4", "target_only"):
            argv = endpoint.native_command(plan, protocol, cell, {"path": "/fixture/draft"}, 18990)
            diagnostic = endpoint.native_command(
                plan, protocol, cell, {"path": "/fixture/draft"}, 18990, diagnostic=True
            )
            self.assertEqual(argv[argv.index("--log-verbosity") + 1], "3")
            self.assertEqual(diagnostic[diagnostic.index("--log-verbosity") + 1], "4")
            diagnostic[diagnostic.index("--log-verbosity") + 1] = "3"
            self.assertEqual(argv, diagnostic)
            self.assertEqual(argv[argv.index("--cache-type-k") + 1], "f16")
            self.assertEqual(argv[argv.index("--cache-type-v") + 1], "f16")
            self.assertNotIn("GGML_W1AX_ADMISSION_TRACE", argv)
        for rep in range(5):
            self.assertEqual(
                set(endpoint.cell_order("eagle_a8", rep)), {"eagle_a8", "eagle_q4", "target_only"}
            )

    def test_fresh_typed_continuation_uses_unchanged_standing_lease_actual_lock_and_empty_census(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan, *_ = self.fixture(root)
            committed = endpoint.training_endpoint(plan, Files(), active=lambda _: False)
            authorization = write(
                root / "standing-authorization.json", {"fixture": "existing human instruction"}
            )
            plan["policy"] = write(
                root / "endpoint-policy.json", {"authorization": {"record": authorization}}
            )
            plan["gpu_control_path"] = str(root / "gpu-control.json")
            write(Path(plan["gpu_control_path"]), {"rtx5080": {"pause_requested": False}})
            origin = write(
                root / "original-startup-lease.json",
                {
                    "issued_unix": 1,
                    "expires_unix": 2,
                    "fixture": "expired startup window; standing consent unchanged",
                },
            )
            original_bytes = Path(origin["path"]).read_bytes()
            resources = {
                "gpu_uuid": "fixture",
                "boot_id": "fixture-boot",
                "host_available_bytes": 100,
                "gpu_free_bytes": 100,
                "dxg_holders": [],
            }
            observer = SimpleNamespace(
                require_released=lambda *_: {
                    "other_context_pids": [],
                    "owned_process_groups_absent": True,
                    "owned_cuda_pids_absent": True,
                },
                snapshot=lambda: dict(resources),
            )
            plan["resource_policy"] = {
                "host_floor_bytes": 1,
                "gpu_floor_bytes": 1,
                "host_return_tolerance_bytes": 0,
                "gpu_return_tolerance_bytes": 0,
            }
            owner = {"pid": 123, "start_ticks": 99, "boot_id": "fixture-boot"}
            lock_path = root / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
            write(lock_path, owner)
            with (
                patch.object(Path, "home", return_value=root),
                patch.object(endpoint, "process_identity", return_value=owner),
                patch.object(endpoint, "identity_active", return_value=True),
                lock_path.open("a") as lock,
            ):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fresh = endpoint.issue_continuation(
                    plan,
                    "a" * 64,
                    origin,
                    committed,
                    root,
                    "export",
                    observer,
                    [],
                    [],
                    resources,
                    now=100,
                )
                proof = endpoint.verify_continuation(
                    plan, "a" * 64, origin, fresh, "export", Files(), now=101, parent_pid=123
                )
                self.assertFalse(proof["new_human_announcement"])
                self.assertEqual(Path(origin["path"]).read_bytes(), original_bytes)
                with self.assertRaisesRegex(ValueError, "freshness"):
                    endpoint.verify_continuation(
                        plan, "a" * 64, origin, fresh, "export", Files(), now=400, parent_pid=123
                    )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    endpoint.verify_continuation(
                        plan, "a" * 64, origin, fresh, "evaluate", Files(), now=101, parent_pid=123
                    )
                fcntl.flock(lock, fcntl.LOCK_UN)
                with self.assertRaisesRegex(ValueError, "actually held"):
                    endpoint.verify_continuation(
                        plan, "a" * 64, origin, fresh, "export", Files(), now=101, parent_pid=123
                    )
            for which in ("cuda", "dxg", "missing"):
                sample = dict(resources)
                contexts = [456] if which == "cuda" else []
                if which == "dxg":
                    sample["dxg_holders"] = [{"pid": 456}]
                elif which == "missing":
                    sample.pop("dxg_holders")
                failed = SimpleNamespace(
                    require_released=lambda *_: {"other_context_pids": contexts},
                    snapshot=lambda: sample,
                )
                with (
                    patch.object(Path, "home", return_value=root),
                    patch.object(endpoint, "process_identity", return_value=owner),
                    self.assertRaises(ValueError),
                ):
                    endpoint.issue_continuation(
                        plan,
                        "a" * 64,
                        origin,
                        committed,
                        root,
                        "evaluate",
                        failed,
                        [],
                        [],
                        resources,
                        now=101,
                    )


if __name__ == "__main__":
    unittest.main()
