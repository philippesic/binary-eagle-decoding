"""Filesystem handoff checks without models, Torch, CUDA or remote operations."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_nine_model_lane_handoff as handoff


def write(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in value) if jsonl else json.dumps(value))
    return handoff.pin(path)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        checkout = self.root / "original"
        names = (
            "train_nine_model_qat.py",
            "export_recurrent_binary.py",
            "prepare_eagle_lane_packet.py",
            "capture_block_qat_teacher.py",
        )
        source = {}
        for name in names:
            path = checkout / "scripts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# pinned original " + name)
            source["scripts/" + name] = handoff.pin(path)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "Original fixture",
            ],
            check=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        auth = write(self.root / "authorization.md", {"instruction": "standing delegation"})
        authorization = {"kind": "human_delegated_operational_settings", "record": auth}
        budget = write(
            self.root / "packet/budget.json",
            {
                "schema": "nine_model_selected_budget_v1",
                "human_selected": False,
                "authorization": authorization,
                "candidates": {
                    "eagle_a1": {"training_limits": {"max_seconds": 86400}, "wall_seconds": 108000}
                },
            },
        )
        initializer = {
            "activation_bits": 1,
            "path": "/not-read/initializer.npz",
            "sha256": "a" * 64,
        }
        calibration = write(
            self.root / "calibration.json",
            {
                "schema": "eagle_production_fusion_initializer_v1",
                "initializer": initializer,
                "authenticated_full_source": {"fixture": "original full TRAIN"},
            },
        )
        continuous = write(
            self.root / "packet/configs/eagle_a1-continuous.json",
            {"training": {"activation_bits": [1], "max_seconds": 86400}},
        )
        config = write(
            self.root / "packet/configs/eagle_a1.json",
            {
                "candidate": "eagle_a1",
                "family": "eagle",
                "initialization": initializer,
                "eagle_config": continuous,
            },
        )
        resolved = write(
            self.root / "packet/configs/resolved-inputs.json",
            {
                "budget": budget,
                "candidates": {"eagle_a1": {"config": config, "profile": "direct_a1"}},
            },
        )
        request = write(
            self.root / "packet/initial-preparation-request.json",
            {
                "schema": "eagle_lane_initial_preparation_v1",
                "optimizer_updates": 0,
                "config": config,
                "source": source["scripts/train_nine_model_qat.py"],
                "initializer_report": calibration,
            },
        )
        native = {
            "binary": {"path": "/not-read/native", "sha256": "b" * 64},
            "target": {"path": "/not-read/target.gguf", "sha256": "c" * 64},
            "source_revision": "d" * 40,
            "client_source": source["scripts/capture_block_qat_teacher.py"],
        }
        cases, generated, requests, replays = [], [], [], []
        history = [
            {"offset": 0, "count": 2, "phase": "prefill", "kv_reused_from_same_chain": False}
        ]
        for i, domain in enumerate(("prose", "code", "reasoning")):
            ancestry = {
                "domain": domain,
                "prompt_id": str(i),
                "prompt_sha256": str(i) * 64,
                "source_split": "TRAIN",
            }
            cases.append({"ancestry": ancestry})
            req = {
                "tokens": [i, 42],
                "tap_ids": [2, 18, 33],
                "logits_mode": "last",
                "chain_ancestry": ancestry | {"prompt_length": 2},
                "decode_history": history,
            }
            requests.append(req)
            producer = {
                "client_source_sha256": native["client_source"]["sha256"],
                "producer_binary_sha256": native["binary"]["sha256"],
                "producer_source_revision": native["source_revision"],
                "target_sha256": native["target"]["sha256"],
            }
            generated.append(
                req
                | producer
                | {
                    "prompt_length": 2,
                    "prompt_source_sha256": ancestry["prompt_sha256"],
                    "generation": {"mode": "native_target_greedy"},
                }
            )
            replays.append(req | producer)
        joins = write(
            self.root / "packet/golden-source-joins.json",
            {
                "cases": cases,
                "initializer_report": calibration,
                "runtime": {
                    "base_model": {"path": "/not-read/base.gguf", "sha256": "e" * 64},
                    "inputs": {"teacher_binary": native["binary"], "target": native["target"]},
                    "native_source_revision": native["source_revision"],
                },
            },
        )
        gen = write(self.root / "original-generation.log", generated, True)
        replay_req = write(self.root / "packet/replay-requests.jsonl", requests, True)
        replay = write(self.root / "original-replay.log", replays, True)
        link = write(
            self.root / "packet/generation-replay-link.json",
            {
                "native": native,
                "source_joins": joins,
                "generation_receipts": gen,
                "replay_requests": replay_req,
                "requests": requests,
            },
        )
        p = self.root / "packet"
        joint = p / "initial-prepare/checkpoints/step-000000000000-e000000-r000000000000/A1"
        commands = write(
            p / "commands.json",
            {
                "schema": "eagle_lane_source_commands_v1",
                "execution_authorized_by_this_file": False,
                "source": {k.split("/")[1]: v for k, v in source.items()},
                "initial_prepare": [
                    sys.executable,
                    str(checkout / "scripts/train_nine_model_qat.py"),
                    "--config",
                    config["path"],
                    "--run-dir",
                    str(p / "initial-prepare"),
                    "--bundle-sha256",
                    request["sha256"],
                    "--stage-name",
                    "eagle_a1/initial-prepare",
                    "--completion-output",
                    str(p / "initial-prepare-receipt.json"),
                    "--allow-cuda",
                    "--prepare-only",
                ],
                "initial_export": [
                    sys.executable,
                    str(checkout / "scripts/export_recurrent_binary.py"),
                    "--base",
                    "/not-read/base.gguf",
                    "--checkpoint",
                    str(joint / "joint.npz"),
                    "--manifest",
                    str(joint / "joint.json"),
                    "--output",
                    str(p / "initial-calibrated-a1.gguf"),
                    "--audit",
                    str(p / "initial-export-audit.json"),
                ],
            },
        )

        def identity(pid):
            return {"pid": pid, "start_ticks": 12, "boot_id": "original-boot"}

        oldconfig = write(self.root / "a8-config.json", {})
        lane = write(self.root / "a8-lane.json", {"config": oldconfig})
        endpoint = write(
            self.root / "a8-endpoint.json",
            {
                "schema": "nine_model_lane_endpoint_plan_v1",
                "candidate": "eagle_a8",
                "artifact_kind": "production",
                "campaign_complete": False,
                "source": source,
                "controller_identity": identity(10),
                "supervisor_identity": identity(11),
                "training_supervisor_state": str(self.root / "supervisor.json"),
                "training_run_dir": str(self.root / "a8-run"),
                "frozen_lane": lane,
                "training_limits": {"max_seconds": 86400},
                "training_source": {"fixture": "original full TRAIN"},
                "training_source_sha256": "f" * 64,
                "target": native["target"],
                "base_model": {"path": "/not-read/base.gguf", "sha256": "e" * 64},
                "gpu_uuid": "GPU-fixture",
            },
        )
        policy = write(
            self.root / "policy.json",
            {
                "schema": "nine_model_lane_endpoint_policy_v1",
                "authorization": authorization,
                "wait_wall_seconds": 108000,
                "export_wall_seconds": 300,
                "evaluation_wall_seconds": 3600,
                "poll_seconds": 1800,
            },
        )
        self.descriptor = {
            "schema": "nine_model_lane_handoff_inputs_v1",
            "upstream_endpoint_plan": endpoint,
            "source_checkout": str(checkout),
            "source_revision": revision,
            "source": source,
            "policy": policy,
            "packet": dict(
                zip(
                    handoff.PACKET_KEYS,
                    (commands, request, config, resolved, budget, joins, link, replay),
                )
            ),
            "outputs": {
                k: {"path": str(self.root / "future" / (k + ".json")), "status": "PENDING"}
                for k in handoff.OUTPUTS
            },
        }
        self.descriptor["outputs"]["initial_receipt"]["path"] = str(
            p / "initial-prepare-receipt.json"
        )
        self.descriptor["outputs"]["export_audit"]["path"] = str(p / "initial-export-audit.json")
        self.descriptor["outputs"]["production_inputs"]["path"] = str(p / "production-inputs.json")

    def mutate(self, key, transform):
        locator = self.descriptor["packet"][key]
        value = json.loads(Path(locator["path"]).read_text())
        transform(value)
        self.descriptor["packet"][key] = write(Path(locator["path"]), value)

    def test_freeze_inspect_truthful_and_exclusive(self):
        plan = handoff.build(self.descriptor)
        self.assertFalse(plan["execution_authorized_by_this_file"])
        self.assertTrue(plan["fresh_live_verification_required"])
        self.assertTrue(all(s["status"] == "PENDING" for s in plan["stages"]))
        self.assertNotIn("native_replay", [s["name"] for s in plan["stages"]])
        output = self.root / "plan.json"
        handoff.publish(output, plan)
        with self.assertRaises(FileExistsError):
            handoff.publish(output, plan)
        self.assertEqual(handoff.build(json.loads(output.read_text())["inputs"]), plan)

    def test_original_source_current_mismatch(self):
        locator = self.descriptor["source"]["scripts/train_nine_model_qat.py"]
        Path(locator["path"]).write_text("# current retention edits")
        with self.assertRaises(ValueError):
            handoff.build(self.descriptor)

    def test_repin_current_source_cannot_replace_old_request(self):
        locator = self.descriptor["source"]["scripts/train_nine_model_qat.py"]
        Path(locator["path"]).write_text("# current retention edits")
        self.descriptor["source"]["scripts/train_nine_model_qat.py"] = handoff.pin(locator["path"])
        with self.assertRaises(ValueError):
            handoff.build(self.descriptor)

    def test_command_changed_even_after_repin_rejected(self):
        self.mutate("commands", lambda c: c["initial_prepare"].remove("--prepare-only"))
        with self.assertRaisesRegex(ValueError, "argv"):
            handoff.build(self.descriptor)

    def test_link_or_packet_hash_change_rejected(self):
        Path(self.descriptor["packet"]["generation_replay_link"]["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            handoff.build(self.descriptor)

    def test_forged_generation_ancestry_rejected_after_repin(self):
        link_path = Path(self.descriptor["packet"]["generation_replay_link"]["path"])
        link = json.loads(link_path.read_text())
        parents = handoff.lines(handoff.Files(), link["generation_receipts"])
        parents[0]["chain_ancestry"]["source_split"] = "development"
        link["generation_receipts"] = write(
            Path(link["generation_receipts"]["path"]), parents, True
        )
        self.descriptor["packet"]["generation_replay_link"] = write(link_path, link)
        with self.assertRaisesRegex(ValueError, "ancestry"):
            handoff.build(self.descriptor)

    def test_forged_replay_producer_rejected_after_repin(self):
        locator = self.descriptor["packet"]["replay_receipts"]
        rows = handoff.lines(handoff.Files(), locator)
        rows[0]["client_source_sha256"] = "0" * 64
        self.descriptor["packet"]["replay_receipts"] = write(Path(locator["path"]), rows, True)
        with self.assertRaisesRegex(ValueError, "producer"):
            handoff.build(self.descriptor)

    def test_no_missing_or_promoted_output(self):
        for variant in ("missing", "promoted", "published"):
            with self.subTest(variant=variant):
                d = copy.deepcopy(self.descriptor)
                if variant == "missing":
                    del d["outputs"]["lane"]
                elif variant == "promoted":
                    d["outputs"]["lane"]["status"] = "PASS"
                else:
                    Path(d["outputs"]["lane"]["path"]).parent.mkdir(parents=True)
                    Path(d["outputs"]["lane"]["path"]).write_text("{}")
                with self.assertRaises(ValueError):
                    handoff.build(d)

    def test_human_selected_forgery(self):
        self.mutate("budget", lambda b: b.update(human_selected=True))
        with self.assertRaises(ValueError):
            handoff.build(self.descriptor)

    def snapshot(self):
        plan = handoff.read(handoff.Files(), self.descriptor["upstream_endpoint_plan"])
        lane = handoff.read(handoff.Files(), plan["frozen_lane"])
        receipt = {
            "schema": "nine_model_stage_receipt_v1",
            "stage": "eagle_a8/train",
            "status": "PASS",
            "artifact_kind": "production",
            "committed": True,
            "completion_reason": "approved_budget_complete",
            "bundle_sha256": plan["frozen_lane"]["sha256"],
            "config_sha256": lane["config"]["sha256"],
            "counters": {"step": 10, "epoch": 0, "cursor": 10, "elapsed_seconds": 86400},
            "checkpoint": {
                "path": str(self.root / "a8-run/training/checkpoints/step-10/resume.pt"),
                "sha256": "a" * 64,
            },
            "hardware": {"gpu_uuid": "GPU-fixture", "compute_capability": [12, 0]},
        }
        write(
            self.root / "a8-run/training/checkpoints/step-10/manifest.json",
            {
                "schema": "continuous_joint_w1ax_v1",
                "step": 10,
                "epoch": 0,
                "cursor": 10,
                "sha256": "a" * 64,
                "source_sha256": "f" * 64,
                "optimizer_rng_cursor_exact": True,
                "exports": {"A8": {}},
            },
        )
        receipt_pin = write(self.root / "a8-run/attempts/original/train-receipt.json", receipt)
        state = {
            "schema": "nine_model_lane_state_v1",
            "attempt": str(self.root / "a8-run/attempts/original"),
            "status": "training_complete",
            "candidate": "eagle_a8",
            "bundle_sha256": plan["frozen_lane"]["sha256"],
            "train_receipt": receipt_pin,
            "owned_release": {"owned_process_groups_absent": True},
        }
        report = write(
            self.root / "endpoint-run/evaluation/report.json",
            {
                "schema": "nine_model_lane_matched_report_v1",
                "candidate": "eagle_a8",
                "artifact_kind": "production",
                "frozen_lane": plan["frozen_lane"],
                "trained_export": {"training_receipt": receipt_pin},
            },
        )
        self.descriptor["upstream_snapshot"] = {
            "supervisor": write(
                self.root / "supervisor.json",
                {
                    "pid": 10,
                    "supervisor_pid": 11,
                    "status": "finished",
                    "exit_code": 0,
                    "received_signal": None,
                },
            ),
            "lane_state": write(self.root / "a8-run/state.json", state),
            "train_receipt": receipt_pin,
            "endpoint_state": write(
                self.root / "endpoint-run/state.json",
                {"status": "endpoint_complete", "campaign_complete": False, "report": report},
            ),
        }

    def test_saved_complete_never_grants_execution(self):
        self.snapshot()
        plan = handoff.build(self.descriptor)
        self.assertIn("fresh_live", plan["upstream_saved_metadata"])
        self.assertFalse(plan["execution_authorized_by_this_file"])

    def test_stopped_wrong_parent_and_premature_receipts_rejected(self):
        for key, field, value in (
            ("supervisor", "received_signal", 2),
            ("supervisor", "pid", 999),
            ("supervisor", "status", "running"),
            ("train_receipt", "completion_reason", "STOP"),
            ("train_receipt", "bundle_sha256", "f" * 64),
            ("endpoint_state", "status", "waiting_for_natural_endpoint"),
        ):
            with self.subTest(field=field, value=value):
                self.snapshot()
                locator = self.descriptor["upstream_snapshot"][key]
                row = json.loads(Path(locator["path"]).read_text())
                row[field] = value
                self.descriptor["upstream_snapshot"][key] = write(Path(locator["path"]), row)
                with self.assertRaises(ValueError):
                    handoff.build(self.descriptor)

    def test_premature_budget_and_wrong_report_parent(self):
        self.snapshot()
        locator = self.descriptor["upstream_snapshot"]["train_receipt"]
        row = json.loads(Path(locator["path"]).read_text())
        row["counters"]["elapsed_seconds"] = 86399
        self.descriptor["upstream_snapshot"]["train_receipt"] = write(Path(locator["path"]), row)
        with self.assertRaises(ValueError):
            handoff.build(self.descriptor)
        self.snapshot()
        result_locator = self.descriptor["upstream_snapshot"]["endpoint_state"]
        result = json.loads(Path(result_locator["path"]).read_text())
        report = json.loads(Path(result["report"]["path"]).read_text())
        report["frozen_lane"]["sha256"] = "f" * 64
        result["report"] = write(Path(result["report"]["path"]), report)
        self.descriptor["upstream_snapshot"]["endpoint_state"] = write(
            Path(result_locator["path"]), result
        )
        with self.assertRaisesRegex(ValueError, "report candidate"):
            handoff.build(self.descriptor)

    def test_fully_consistent_foreign_parent_receipt_rejected(self):
        self.snapshot()
        snapshot = self.descriptor["upstream_snapshot"]
        receipt = json.loads(Path(snapshot["train_receipt"]["path"]).read_text())
        snapshot["train_receipt"] = write(self.root / "foreign-parent/train-receipt.json", receipt)
        state = json.loads(Path(snapshot["lane_state"]["path"]).read_text())
        state["train_receipt"] = snapshot["train_receipt"]
        snapshot["lane_state"] = write(Path(snapshot["lane_state"]["path"]), state)
        result = json.loads(Path(snapshot["endpoint_state"]["path"]).read_text())
        report = json.loads(Path(result["report"]["path"]).read_text())
        report["trained_export"]["training_receipt"] = snapshot["train_receipt"]
        result["report"] = write(Path(result["report"]["path"]), report)
        snapshot["endpoint_state"] = write(Path(snapshot["endpoint_state"]["path"]), result)
        with self.assertRaisesRegex(ValueError, "outside original parent"):
            handoff.build(self.descriptor)

    def test_failed_serialization_publishes_nothing(self):
        path = self.root / "invalid-plan.json"
        with self.assertRaises(ValueError):
            handoff.publish(path, {"invalid": float("nan")})
        self.assertFalse(path.exists())

    def test_cli_freeze_and_immutable_inspect(self):
        inputs = self.root / "inputs.json"
        output = self.root / "plan.json"
        write(inputs, self.descriptor)
        command = [sys.executable, str(handoff.ROOT / "scripts/prepare_nine_model_lane_handoff.py")]
        subprocess.run(
            command + ["--inputs", str(inputs), "--output", str(output)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            command + ["--plan", str(output), "--plan-sha256", handoff.sha256(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertFalse(json.loads(result.stdout)["execution_authorized_by_this_file"])
        previous = output.read_bytes()
        refused = subprocess.run(
            command + ["--inputs", str(inputs), "--output", str(output)], capture_output=True
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(output.read_bytes(), previous)

    def test_changed_dependency_revision_rejected(self):
        checkout = self.descriptor["source_checkout"]
        subprocess.run(
            [
                "git",
                "-C",
                checkout,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "Different checkout",
            ],
            check=True,
        )
        with self.assertRaisesRegex(ValueError, "HEAD differs"):
            handoff.build(self.descriptor)

    def test_repinned_dependency_bytes_reject_against_original_git(self):
        inventory = copy.deepcopy(self.descriptor["source"])
        locator = inventory["scripts/capture_block_qat_teacher.py"]
        Path(locator["path"]).write_text("# altered dependency; unchanged trainer entrypoint")
        inventory["scripts/capture_block_qat_teacher.py"] = handoff.pin(locator["path"])
        with self.assertRaisesRegex(ValueError, "dependency bytes"):
            handoff.sources(
                handoff.Files(),
                Path(self.descriptor["source_checkout"]),
                inventory,
                self.descriptor["source_revision"],
            )

    def test_exact_checkpoint_manifest_metadata_rejects_tampering(self):
        for field, value in (
            ("source_sha256", "0" * 64),
            ("optimizer_rng_cursor_exact", False),
            ("epoch", 1),
            ("cursor", 11),
            ("sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                self.snapshot()
                path = self.root / "a8-run/training/checkpoints/step-10/manifest.json"
                row = json.loads(path.read_text())
                row[field] = value
                write(path, row)
                with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
                    handoff.build(self.descriptor)

    def test_stale_lease_uses_existing_contract(self):
        path = self.root / "lease.json"
        write(
            path,
            {
                "schema": "nine_model_gpu_lease_v1",
                "host": "rtx5080",
                "user_announced_available": True,
                "sole_owner": True,
                "pause_requested": False,
                "bundle_sha256": "a" * 64,
                "granted_unix": 10,
                "expires_unix": 310,
                "gpu_uuid": "GPU-fixture",
            },
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            handoff.require_available(path, "a" * 64, now=311)

    def test_import_has_no_model_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import prepare_nine_model_lane_handoff; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], cwd=handoff.ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
