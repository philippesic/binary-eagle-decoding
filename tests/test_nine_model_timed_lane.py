"""CPU-only fake child lifecycle and tiny actual serialized checkpoint checks."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import evaluate_nine_model_timed_checkpoint as evaluation  # noqa: E402
import export_nine_model_lane_candidate as exporter  # noqa: E402
import run_nine_model_lane as lane_api  # noqa: E402
import test_nine_model_lane_candidate_export as export_fixtures  # noqa: E402

from w1a1_eagle.continuous_budget import TimedEvaluation  # noqa: E402
from w1a1_eagle.nine_model_pipeline import Files, atomic_json  # noqa: E402


class TimedLaneTests(unittest.TestCase):
    def fixture(self, root):
        root = root.resolve()
        run = root / "run"
        run.mkdir()
        config = root / "config.json"
        config.write_text("{}")
        frozen = root / "lane.json"
        frozen.write_text("{}")
        protocol = root / "protocol.json"
        protocol.write_text("{}")
        lane = {
            "candidate": "dspark_a8",
            "config": exporter.pin(config),
            "gpu_uuid": "GPU-fixture",
            "timed_evaluation_plan": {"path": "plan", "sha256": "a" * 64},
            "admission_plan": {"path": "admission"},
            "training_wall_seconds": 45000,
        }
        state = {}
        events = []
        controller = root / "supervisor.json"
        controller.write_text("{}")
        proof = {
            "owned_process_groups_absent": True,
            "owned_cuda_pids_absent": True,
            "other_context_pids": [],
        }
        parent = self

        class Runner:
            calls = []
            alive = False
            index = 0

            def run(self, argv, **kwargs):
                parent.assertFalse(self.alive)
                self.alive = True
                self.calls.append(argv)
                self.index += 1
                events.append("train" + str(self.index))
                milestone = 14400 * self.index
                checkpoint = run / "training/checkpoints" / str(self.index) / "resume.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(str(self.index))
                latest = run / "training/checkpoints/latest.json"
                latest.write_text("{}")
                request_path = checkpoint.parent / "request.json"
                request = {
                    "schema": "nine_model_timed_evaluation_request_v1",
                    "checkpoint": exporter.pin(checkpoint),
                    "protocol": exporter.pin(protocol),
                    "elapsed_seconds": milestone + 0.2,
                    "milestone_seconds": milestone,
                    "final_training_complete": self.index == 3,
                }
                atomic_json(request_path, request)
                endpoint = {
                    "schema": "nine_model_stage_receipt_v1",
                    "stage": "dspark_a8/train",
                    "artifact_kind": "production",
                    "status": "PASS",
                    "committed": True,
                    "completion_reason": "approved_budget_complete"
                    if self.index == 3
                    else "timed_evaluation_boundary",
                    "bundle_sha256": exporter.pin(frozen)["sha256"],
                    "config_sha256": lane["config"]["sha256"],
                    "hardware": {"gpu_uuid": lane["gpu_uuid"], "compute_capability": [12, 0]},
                    "counters": {"step": self.index, "elapsed_seconds": milestone + 0.2},
                    "checkpoint": exporter.pin(checkpoint),
                    "timed_evaluation_request": exporter.pin(request_path),
                }
                receipt = Path(argv[argv.index("--completion-output") + 1])
                atomic_json(receipt, endpoint)
                self.alive = False

        runner = Runner()
        runner.calls = []

        def release():
            parent.assertFalse(runner.alive, "training/evaluation overlap")
            events.append("release")
            state["owned_release"] = proof
            return proof

        def authorize():
            if (run / "STOP").exists():
                raise InterruptedError("STOP")

        def evaluate(*args, **kwargs):
            parent.assertFalse(runner.alive)
            parent.assertEqual(events[-1], "release")
            events.append("eval")
            training = args[3]
            endpoint = json.loads(Path(training["path"]).read_text())
            request = json.loads(Path(endpoint["timed_evaluation_request"]["path"]).read_text())
            dest = Path(args[5])
            dest.mkdir(parents=True, exist_ok=False)
            report = dest / "report.json"
            report.write_text('{"native":true}')
            receipt = {
                "schema": "nine_model_timed_evaluation_receipt_v1",
                "status": "PASS",
                "completed": True,
                "candidate": lane["candidate"],
                "bundle_sha256": exporter.pin(frozen)["sha256"],
                "config_sha256": lane["config"]["sha256"],
                "request": endpoint["timed_evaluation_request"],
                "checkpoint": endpoint["checkpoint"],
                "protocol": request["protocol"],
                "evaluation": exporter.pin(report),
                "owned_release": proof,
            }
            atomic_json(dest / "receipt.json", receipt)
            return exporter.pin(dest / "receipt.json")

        def invoke(evaluator=evaluate, resume=False):
            with (
                patch.object(lane_api, "validate_plan", return_value=({}, "pin")),
                patch.object(evaluation, "validate_plan", return_value=({}, {})),
            ):
                return lane_api.run_training_lifecycle(
                    lane,
                    exporter.pin(frozen),
                    run,
                    root / ("attempt-" + os.urandom(4).hex()),
                    "admitted",
                    controller,
                    state,
                    run / "state.json",
                    Files(),
                    runner,
                    None,
                    release,
                    authorize,
                    resume=resume,
                    evaluator=evaluator,
                )

        return lane, state, runner, events, evaluate, invoke, run

    def test_three_boundaries_release_before_eval_and_exact_receipt_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, runner, events, _, invoke, _ = self.fixture(Path(tmp))
            invoke()
            self.assertEqual(len(runner.calls), 3)
            self.assertEqual(events.count("eval"), 3)
            self.assertEqual(state["status"], "training_complete")
            self.assertEqual(state["evaluation_status"], "PASS")
            self.assertIsNone(state["pending_training"])
            self.assertNotIn("--resume", runner.calls[0])
            for argv in runner.calls[1:]:
                self.assertIn("--resume", argv)
                path = argv[argv.index("--evaluation-receipt") + 1]
                self.assertEqual(
                    exporter.pin(path)["sha256"],
                    argv[argv.index("--evaluation-receipt-sha256") + 1],
                )

    def test_failed_eval_never_resumes_and_crash_replay_evaluates_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, runner, _, evaluate, invoke, _ = self.fixture(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "evaluation crashed"):
                invoke(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("evaluation crashed")))
            pending = copy.deepcopy(state["pending_training"])
            self.assertEqual(len(runner.calls), 1)
            observed = []

            def replay(*args, **kwargs):
                observed.append(args[3])
                return evaluate(*args, **kwargs)

            invoke(replay, resume=True)
            self.assertEqual(observed[0], pending)
            self.assertEqual(len(runner.calls), 3)
            self.assertEqual(state["status"], "training_complete")

    def test_stop_after_eval_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, runner, _, evaluate, invoke, run = self.fixture(Path(tmp))

            def stop(*args, **kwargs):
                result = evaluate(*args, **kwargs)
                (run / "STOP").touch()
                return result

            with self.assertRaises(InterruptedError):
                invoke(stop)
            self.assertEqual(len(runner.calls), 1)
            self.assertIsNotNone(state["pending_training"])

    def test_wrong_checkpoint_or_failed_release_receipt_blocks_resume(self):
        for key, value in (
            ("checkpoint", {"path": "other", "sha256": "b" * 64}),
            ("status", "FAIL"),
            ("owned_release", {"owned_process_groups_absent": False}),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                _, _, runner, _, evaluate, invoke, _ = self.fixture(Path(tmp))

                def invalid(*args, **kwargs):
                    locator = evaluate(*args, **kwargs)
                    data = json.loads(Path(locator["path"]).read_text())
                    data[key] = value
                    atomic_json(Path(locator["path"]), data)
                    return exporter.pin(locator["path"])

                with self.assertRaisesRegex(ValueError, "authenticated"):
                    invoke(invalid)
                self.assertEqual(len(runner.calls), 1)

    def test_mutated_receipt_hash_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, runner, _, evaluate, invoke, _ = self.fixture(Path(tmp))

            def mutation(*args, **kwargs):
                locator = evaluate(*args, **kwargs)
                Path(locator["path"]).write_text("{}")
                return locator

            with self.assertRaises(ValueError):
                invoke(mutation)
            self.assertEqual(len(runner.calls), 1)

    def test_four_cell_pairing_and_token_mismatch_are_rejected(self):
        plan = {
            "candidate": "dspark_a8",
            "gpu_uuid": "GPU-fixture",
            "target_policy": {},
            "frozen_lane": {},
            "protocol": {},
            "remaining_candidates": [],
            "initial": {},
        }
        records = [
            {
                "cell": cell,
                "repetition": rep,
                "prompt_id": "p",
                "output_tokens": 2,
                "latency_s": 1.0,
                "generated_token_ids": [1, 2],
                "speculative": {"accepted": 1, "proposed": 2, "rounds": 1},
            }
            for rep in range(5)
            for cell in evaluation.cell_order("dspark_a8", rep)
        ]
        report = evaluation.matched_report(plan, records, [], {}, fixture=True)
        self.assertEqual(set(report["cells"]), {"dspark_a8", "initial", "eagle_q4", "target_only"})
        self.assertEqual(report["cells"]["initial"]["acceptance_rate"], 0.5)
        records[1]["generated_token_ids"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "token sequence"):
            evaluation.matched_report(plan, records, [], {}, fixture=True)

    def recovery_fixture(self, root, family, outer_receipt, orphan_request=False):
        root = root.resolve()
        fixtures = export_fixtures.LaneExportTests()
        f = fixtures.fixture(root, family)
        lane = f["lane"]
        lane["admission_plan"] = f["write"](root / "plan.json", f["plan"])
        protocol = f["write"](root / "protocol.json", {})
        spec = json.loads(Path(lane["config"]["path"]).read_text())
        spec.update(evaluation_milestones_seconds=[4, 8, 10], evaluation_protocol=protocol)
        lane["config"] = f["write"](Path(lane["config"]["path"]), spec)
        lane["timed_evaluation_plan"] = {"path": "test-plan", "sha256": "a" * 64}
        lane["training_wall_seconds"] = 30
        run = root / "run"
        training = run / "training"
        source = f["plan"]["candidates"][lane["candidate"]]["source_bindings"]
        ledger = {
            "schema": "continuous_training_budget_v1",
            "source_sha256": exporter.digest(source)
            if family == "eagle"
            else f["locators"][0]["sha256"],
            "max_seconds": 10,
            "active_attempt": None,
            "training_seconds": 10.0,
        }
        ledger_pin = f["write"](training / "budget-used.json", ledger)
        boundary = TimedEvaluation(
            training,
            [4, 8, 10],
            10,
            candidate=lane["candidate"],
            bundle_sha256=f["locators"][0]["sha256"],
            config_sha256=lane["config"]["sha256"],
            protocol=protocol,
            atomic_write=atomic_json,
        )
        request = boundary.publish(
            f["receipt"]["checkpoint"], 10.0, ledger_pin, f["receipt"]["counters"]
        )
        if orphan_request:
            boundary.path.unlink()
        latest_path = training / ("latest.json" if family == "eagle" else "checkpoints/latest.json")
        f["write"](latest_path, f["receipt"]["checkpoint"])
        if family != "eagle":
            original = f["receipt"]["exports"][lane["candidate"]]
            destination = Path(request["path"]).parent / "export-original"
            destination.mkdir()
            shutil.copyfile(original["checkpoint"]["path"], destination / "binary.npz")
            shutil.copyfile(original["manifest"]["path"], destination / "binary.json")
            original["checkpoint"] = exporter.pin(destination / "binary.npz")
            original["manifest"] = exporter.pin(destination / "binary.json")
        f["receipt"].update(
            completion_reason="timed_evaluation_boundary",
            config_sha256=lane["config"]["sha256"],
            timed_evaluation_request=request,
        )
        if outer_receipt:
            f["write"](
                run / "attempts/original/training-attempts/original-child/train-receipt.json",
                f["receipt"],
            )
        admitted = {
            "schema": "nine_model_training_admission_v1",
            "status": "PASS",
            "artifact_kind": "production",
            "candidate": lane["candidate"],
            "bundle_sha256": f["locators"][0]["sha256"],
            "config_sha256": lane["config"]["sha256"],
            "gpu_uuid": lane["gpu_uuid"],
            "compute_capability": [12, 0],
        }
        admission = f["write"](root / "admission.json", admitted)
        return f, run, request, Path(admission["path"])

    def test_publication_crash_matrix_recovers_same_boundary_without_trainer_launch(self):
        for family in ("eagle", "dspark", "dflash"):
            for outer, orphan in ((False, True), (False, False), (True, False)):
                with (
                    self.subTest(family=family, outer=outer, orphan=orphan),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp).resolve()
                    f, run, request, admitted = self.recovery_fixture(root, family, outer, orphan)
                    state_path = run / "state.json"
                    state = {
                        "schema": "nine_model_lane_state_v1",
                        "status": "training",
                        "candidate": f["lane"]["candidate"],
                        "bundle_sha256": f["locators"][0]["sha256"],
                        "campaign_complete": False,
                    }
                    supervisor = root / "supervisor.json"
                    atomic_json(
                        supervisor,
                        {"status": "running", "pid": os.getpid(), "supervisor_pid": os.getppid()},
                    )
                    calls = []

                    class NoTrainer:
                        def run(self, *a, **k):
                            raise AssertionError("published boundary relaunched trainer")

                    def released():
                        state["owned_release"] = {
                            "owned_process_groups_absent": True,
                            "owned_cuda_pids_absent": True,
                        }
                        return state["owned_release"]

                    def evaluate_boundary(*args, **kwargs):
                        calls.append(args[3])
                        context = exporter.validate_endpoint(*args[:4], checkpoint_mode="timed")
                        self.assertEqual(
                            context["checkpoint"],
                            json.loads(Path(request["path"]).read_text())["checkpoint"],
                        )
                        self.assertEqual(context["receipt"]["counters"], f["receipt"]["counters"])
                        raise RuntimeError("evaluated same boundary")

                    with (
                        patch.object(lane_api, "validate_plan", return_value=(f["plan"], Files())),
                        patch.object(evaluation, "validate_plan", return_value=({}, {})),
                        patch.object(
                            exporter, "validate_lane", return_value=(f["lane"], Files(), f["plan"])
                        ),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "evaluated same boundary"):
                            lane_api.run_training_lifecycle(
                                f["lane"],
                                f["locators"][0],
                                run,
                                run / "attempts/recovery",
                                admitted,
                                supervisor,
                                state,
                                state_path,
                                Files(),
                                NoTrainer(),
                                None,
                                released,
                                lambda: None,
                                resume=True,
                                evaluator=evaluate_boundary,
                            )
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(state["pending_training"], calls[0])
                    recovered = json.loads(Path(calls[0]["path"]).read_text())
                    if outer:
                        self.assertIn("original-child", calls[0]["path"])
                    else:
                        self.assertEqual(recovered["recovery"]["optimizer_updates_performed"], 0)
                    gate = json.loads((run / "training/timed-evaluation-state.json").read_text())
                    self.assertEqual(gate["pending"], request)
                    self.assertEqual(
                        json.loads(Path(request["path"]).read_text())["elapsed_seconds"], 10.0
                    )

    def test_recovery_rejects_changed_ledger_counter_and_export_hash(self):
        for corruption in ("ledger", "counter", "export"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                f, run, request, admitted = self.recovery_fixture(root, "dspark", False)
                if corruption == "ledger":
                    path = run / "training/budget-used.json"
                    value = json.loads(path.read_text())
                    value["training_seconds"] += 1
                    atomic_json(path, value)
                elif corruption == "counter":
                    path = Path(f["receipt"]["checkpoint"]["path"]).with_suffix(".receipt.json")
                    value = json.loads(path.read_text())
                    value["cursor"]["step"] += 1
                    atomic_json(path, value)
                else:
                    path = Path(f["receipt"]["exports"][f["lane"]["candidate"]]["manifest"]["path"])
                    value = json.loads(path.read_text())
                    value["checkpoint_sha256"] = "b" * 64
                    atomic_json(path, value)
                with self.assertRaises(ValueError):
                    evaluation.recover_boundary_receipt(
                        f["lane"],
                        f["locators"][0],
                        run,
                        run / "attempts/recovery",
                        admitted,
                        {},
                        Files(),
                    )
                self.assertFalse((run / "attempts/recovery").exists())

    def test_native_loop_rejects_explicit_train_and_unknown_prompt_split_before_launch(self):
        for split in ("train", "heldout", "unknown"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                path = root / "prompts.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "id": "p",
                            "domain": "code",
                            "split": split,
                            "messages": [{"role": "user", "content": "tiny"}],
                        }
                    )
                )
                with patch.object(evaluation.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "unique admitted development prompts"):
                        evaluation.evaluate(
                            {"prompts": exporter.pin(path)},
                            Files(),
                            {},
                            {},
                            root / "evaluation",
                            root / "STOP",
                            lambda: None,
                            None,
                            None,
                        )
                    launch.assert_not_called()

    def test_native_loop_stops_each_owned_server_and_preserves_failed_identity(self):
        for crash in (False, True):
            with self.subTest(crash=crash), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()

                def artifact(name, data):
                    path = root / name
                    atomic_json(path, data)
                    return exporter.pin(path)

                prompts_path = root / "prompts.jsonl"
                prompts_path.write_text(
                    json.dumps(
                        {
                            "id": "p",
                            "domain": "code",
                            "messages": [{"role": "user", "content": "tiny"}],
                        }
                    )
                )
                prompts = exporter.pin(prompts_path)
                model = artifact("model.gguf", {})
                audit = artifact("audit.json", {"projections": {}})
                plan = {
                    "candidate": "dspark_a8",
                    "gpu_uuid": "GPU-fixture",
                    "prompts": prompts,
                    "initial": {"model": model, "audit": audit},
                    "control": {"model": model},
                    "runtime": {"binary": model},
                    "target": model,
                    "environment": {},
                    "resource_policy": {},
                    "target_policy": {},
                    "frozen_lane": {},
                    "protocol": {},
                    "remaining_candidates": [],
                    "development_admission": {},
                }
                protocol = {
                    "repetitions": 5,
                    "warmups_per_cell": 2,
                    "max_output_tokens": 2,
                    "seed": 1,
                    "context_tokens": 32,
                    "batch_tokens": 16,
                    "microbatch_tokens": 16,
                    "port": 8000,
                    "startup_wall_seconds": 1,
                    "evaluation_wall_seconds": 30,
                    "draft_lengths": {"dspark": 7, "eagle": 7},
                }
                alive = set()
                processes = []

                class Process:
                    def __init__(self, *args, **kwargs):
                        self.pid = 1000 + len(processes)
                        self.returncode = None
                        processes.append(self)
                        self.assert_no_overlap = not alive
                        alive.add(self.pid)

                class Runner:
                    process_groups = []
                    process_identities = []

                runner = Runner()
                runner.process_groups, runner.process_identities = [], []

                class Observer:
                    releases = 0

                    def snapshot(self):
                        return {
                            "gpu_uuid": "GPU-fixture",
                            "compute_capability": [12, 0],
                            "dxg_holders": [],
                        }

                    def require_released(self, groups, identities):
                        if alive:
                            raise AssertionError("owned native server remains live")
                        self.releases += 1
                        return {
                            "owned_process_groups_absent": True,
                            "owned_cuda_pids_absent": True,
                            "other_context_pids": [],
                        }

                observer = Observer()

                def stop(proc):
                    alive.remove(proc.pid)
                    proc.returncode = 0

                def request(_url, _body, _timeout, destination):
                    if crash:
                        raise RuntimeError("request failed")
                    artifact_path = Path(destination) / "measurement.json"
                    atomic_json(artifact_path, {})
                    return {
                        "generated_token_ids": [1, 2],
                        "completion_tokens": 2,
                        "request_wall_s": 1.0,
                        "speculative": {"accepted": 1, "proposed": 2, "rounds": 1},
                    }

                with (
                    patch.object(evaluation.subprocess, "Popen", Process),
                    patch.object(
                        evaluation,
                        "process_identity",
                        side_effect=lambda pid: {"pid": pid, "boot_id": "boot", "start_ticks": 1},
                    ),
                    patch.object(evaluation, "descendant_identities", return_value=[]),
                    patch.object(evaluation, "cleanup_descendants"),
                    patch.object(evaluation, "stop_owned_server", side_effect=stop),
                    patch.object(evaluation, "wait_ready"),
                    patch.object(evaluation, "available_port", return_value=True),
                    patch.object(evaluation, "execute_request", side_effect=request),
                    patch.object(evaluation, "request_body", return_value={}),
                    patch.object(evaluation, "resource_gate"),
                    patch.object(evaluation, "round_summary", return_value={"rounds": 1}),
                    patch.object(
                        evaluation, "validate_cuda_dispatch", return_value={"status": "PASS"}
                    ),
                ):
                    if crash:
                        with self.assertRaisesRegex(RuntimeError, "request failed"):
                            evaluation.evaluate(
                                plan,
                                Files(),
                                protocol,
                                {"model": model, "audit": audit},
                                root / "eval",
                                root / "STOP",
                                lambda: None,
                                observer,
                                runner,
                            )
                        self.assertEqual(len(runner.process_identities), 1)
                        self.assertTrue((root / "eval/rep-00/dspark_a8/process.json").is_file())
                    else:
                        report = evaluation.evaluate(
                            plan,
                            Files(),
                            protocol,
                            {"model": model, "audit": audit},
                            root / "eval",
                            root / "STOP",
                            lambda: None,
                            observer,
                            runner,
                        )
                        self.assertEqual(len(processes), 24)
                        self.assertEqual(observer.releases, 24)
                        self.assertEqual(
                            len(json.loads(Path(report["path"]).read_text())["cells"]), 4
                        )
                self.assertFalse(alive)
                self.assertTrue(all(p.assert_no_overlap for p in processes))

    def test_actual_tiny_checkpoint_timed_mode_preserves_default_final_gate(self):
        for family in ("eagle", "dspark", "dflash"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                fixtures = export_fixtures.LaneExportTests()
                f = fixtures.fixture(Path(tmp), family)
                spec = json.loads(Path(f["lane"]["config"]["path"]).read_text())
                protocol = f["write"](Path(tmp) / "protocol.json", {})
                spec.update(evaluation_milestones_seconds=[4, 8, 10], evaluation_protocol=protocol)
                f["lane"]["config"] = f["write"](Path(f["lane"]["config"]["path"]), spec)
                f["receipt"].update(
                    completion_reason="timed_evaluation_boundary",
                    config_sha256=f["lane"]["config"]["sha256"],
                )
                request = {
                    "schema": "nine_model_timed_evaluation_request_v1",
                    "candidate": f["lane"]["candidate"],
                    "bundle_sha256": f["receipt"]["bundle_sha256"],
                    "config_sha256": f["receipt"]["config_sha256"],
                    "checkpoint": f["receipt"]["checkpoint"],
                    "protocol": protocol,
                    "elapsed_seconds": 10.0,
                    "milestone_seconds": 8,
                    "milestone_index": 1,
                    "final_training_complete": False,
                    "budget_ledger": f["lane"]["budget"],
                }
                f["receipt"]["timed_evaluation_request"] = f["write"](
                    Path(tmp) / "request.json", request
                )
                fixtures.replace_receipt(f)
                locators = list(f["locators"])
                state = json.loads(Path(locators[1]["path"]).read_text())
                state["status"] = "awaiting_evaluation"
                locators[1] = f["write"](Path(locators[1]["path"]), state)
                locators[2] = f["write"](
                    Path(locators[2]["path"]),
                    {"status": "running", "pid": os.getpid(), "supervisor_pid": os.getppid()},
                )
                with patch.object(
                    exporter, "validate_lane", return_value=(f["lane"], Files(), f["plan"])
                ):
                    with self.assertRaises(ValueError):
                        exporter.validate_endpoint(*locators)
                    context = exporter.validate_endpoint(*locators, checkpoint_mode="timed")
                    self.assertEqual(context["checkpoint"], request["checkpoint"])
                    request["checkpoint"]["sha256"] = "a" * 64
                    f["write"](Path(tmp) / "request.json", request)
                    with self.assertRaises(ValueError):
                        exporter.validate_endpoint(*locators, checkpoint_mode="timed")


if __name__ == "__main__":
    unittest.main()
