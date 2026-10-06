"""CPU-only uneven-clock training/evaluation/resume acceptance checks."""

import contextlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_nine_model_training as fixtures
import torch
import torch._dynamo  # Initialize before the temporary provider module fixture.
import train_nine_model_qat as launcher
from test_continuous_qat import config, make
from test_nine_model_training import block_fixture

from w1a1_eagle.block_qat import BlockDrafter, block_train_step
from w1a1_eagle.continuous_budget import (
    TimedEvaluation,
    TrainingBudget,
    artifact_locator,
    validate_evaluation_milestones_seconds,
)
from w1a1_eagle.continuous_qat import atomic_json


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def gate(root, milestones=(3, 6, 9), protocol=None, writer=atomic_json):
    if protocol is None:
        protocol_path = root / "protocol.json"
        if not protocol_path.exists():
            atomic_json(protocol_path, {"fixed": "CPU fixture, no prompts"})
        protocol = artifact_locator(protocol_path)
    return TimedEvaluation(
        root,
        milestones,
        milestones[-1],
        candidate="eagle_a1",
        bundle_sha256="a" * 64,
        config_sha256="b" * 64,
        protocol=protocol,
        atomic_write=writer,
    )


def result_for(root, request_locator, suffix=""):
    request = json.loads(Path(request_locator["path"]).read_text())
    report = root / f"evaluation-{request['milestone_index']}{suffix}.json"
    atomic_json(report, {"fixture": "no native runtime claim"})
    receipt = root / f"evaluation-receipt-{request['milestone_index']}{suffix}.json"
    atomic_json(
        receipt,
        {
            "schema": "nine_model_timed_evaluation_receipt_v1",
            "status": "PASS",
            "completed": True,
            **{
                key: request[key]
                for key in ("candidate", "bundle_sha256", "config_sha256", "checkpoint", "protocol")
            },
            "request": request_locator,
            "evaluation": artifact_locator(report),
            "owned_release": {"owned_process_groups_absent": True, "owned_cuda_pids_absent": True},
        },
    )
    return artifact_locator(receipt)


class TimedGateTests(unittest.TestCase):
    def publish(self, root, helper, elapsed=3.4, writer=atomic_json):
        checkpoint = root / "resume.pt"
        checkpoint.write_bytes(b"exact optimizer rng cursor fixture")
        ledger = root / "budget.json"
        writer(
            ledger,
            {
                "schema": "continuous_training_budget_v1",
                "max_seconds": 9,
                "active_attempt": None,
                "training_seconds": elapsed,
            },
        )
        return helper.publish(
            artifact_locator(checkpoint),
            elapsed,
            artifact_locator(ledger),
            {"step": 2, "epoch": 0, "cursor": 2},
        )

    def test_schedule_rejects_bool_duplicates_unsorted_and_missing_final(self):
        for values in ([True, 6, 9], [3, 3, 9], [6, 3, 9], [3, 6], [], [3, float("nan"), 9]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_evaluation_milestones_seconds(values, 9)
        self.assertEqual(
            validate_evaluation_milestones_seconds([14400, 28800, 43200], 43200),
            (14400, 28800, 43200),
        )

    def test_pending_fails_closed_and_exact_result_unlocks_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper = gate(root)
            request = self.publish(root, helper)
            helper = gate(root)
            with self.assertRaisesRegex(ValueError, "explicit authenticated"):
                helper.authorize_resume()
            with self.assertRaisesRegex(ValueError, "forbids optimizer"):
                helper.due(4)
            receipt = result_for(root, request)
            helper.authorize_resume(receipt)
            self.assertFalse(helper.due(3.4))
            helper.require_checkpoint(
                json.loads(Path(request["path"]).read_text())["checkpoint"],
                {"step": 2, "epoch": 0, "cursor": 2},
            )
            with self.assertRaisesRegex(ValueError, "restored checkpoint"):
                helper.require_checkpoint({"path": "wrong", "sha256": "bad"})
            self.assertTrue(gate(root).due(6))
            # Idempotent same-result recovery, but it cannot authorize the next boundary.
            gate(root).authorize_resume(receipt)
            next_request = self.publish(root, helper, 6.2)
            self.assertNotEqual(request, next_request)
            with self.assertRaisesRegex(ValueError, "receipt/checkpoint/protocol"):
                helper.authorize_resume(receipt)

    def test_rejects_changed_protocol_wrong_checkpoint_failure_and_unreleased_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper = gate(root)
            request = self.publish(root, helper)
            good = result_for(root, request)
            original = json.loads(Path(good["path"]).read_text())
            for key, value in [
                ("status", "FAIL"),
                ("checkpoint", {"path": "other", "sha256": "0" * 64}),
                ("protocol", {"path": "other", "sha256": "0" * 64}),
                ("owned_release", {"owned_process_groups_absent": True}),
            ]:
                bad = root / "bad.json"
                atomic_json(bad, original | {key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    helper.authorize_resume(artifact_locator(bad))
            Path(original["protocol"]["path"]).write_text("{}")
            with self.assertRaises(ValueError):
                gate(root)

    def test_request_rename_state_failure_recovers_original_clock_and_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def fail_state(path, value):
                if Path(path).name == "timed-evaluation-state.json":
                    raise OSError("fixture state publication failed")
                atomic_json(path, value)

            helper = gate(root, writer=fail_state)
            with self.assertRaises(OSError):
                self.publish(root, helper)
            helper = gate(root)
            request = self.publish(root, helper, elapsed=3.9)
            saved = json.loads(Path(request["path"]).read_text())
            self.assertEqual(saved["elapsed_seconds"], 3.4)
            ledger = json.loads(Path(saved["budget_ledger"]["path"]).read_text())
            self.assertEqual(ledger["training_seconds"], 3.4)
            self.assertIsNone(ledger["active_attempt"])

    def test_budget_startup_eval_and_reconstruction_excluded_without_refund(self):
        with tempfile.TemporaryDirectory() as folder, patch("time.monotonic") as clock:
            path = Path(folder) / "budget.json"
            clock.return_value = 100
            first = TrainingBudget(path, "frozen-source", 43200, atomic_json)
            first.begin(0)
            clock.return_value = 14510
            self.assertEqual(first.finish(), 14410)
            clock.return_value = 20000
            second = TrainingBudget(path, "frozen-source", 43200, atomic_json)
            self.assertEqual(second.load(14400), 14410)
            second.begin(14400)
            clock.return_value = 34390
            self.assertEqual(second.finish(), 28800)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                TrainingBudget(path, "frozen-source", 28800, atomic_json).load(28800)

    def test_normal_final_update_overage_is_not_refunded(self):
        with tempfile.TemporaryDirectory() as folder, patch("time.monotonic") as clock:
            path = Path(folder) / "budget.json"
            clock.return_value = 100
            budget = TrainingBudget(path, "source", 43200, atomic_json)
            budget.begin(43199)
            clock.return_value = 102.7
            self.assertAlmostEqual(budget.finish(), 43201.7)
            self.assertEqual(json.loads(path.read_text())["max_seconds"], 43200)
            clock.return_value = 1000
            self.assertAlmostEqual(
                TrainingBudget(path, "source", 43200, atomic_json).load(43200), 43201.7
            )


class ContinuousTimedTests(unittest.TestCase):
    durations = [1.4, 1.9, 0.2, 2.8, 0.4, 1.6, 1.7]

    def trainer(self, root, clock, helper=None, max_steps=None):
        cfg = replace(
            config(max_steps=max_steps, max_seconds=9 if helper else None),
            activation_bits=(1,),
            development_lifecycle="standalone",
            keep_checkpoints=1,
        )
        trainer = make(root, cfg)
        trainer.timed_evaluation = helper
        original = trainer.lanes[0].optimizer.step

        def timed_step(*args, **kwargs):
            answer = original(*args, **kwargs)
            clock.advance(self.durations[trainer.step])
            return answer

        trainer.lanes[0].optimizer.step = timed_step
        return trainer

    def test_three_uneven_boundaries_equal_uninterrupted_state_and_preserve_milestones(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clock = Clock()
            with patch("time.monotonic", clock):
                full = self.trainer(root / "full", clock, max_steps=7)
                full.run(require_smoke=False)
                lane_root = root / "segments"
                lane_root.mkdir()
                protected = []
                previous = None
                for index, expected_step in enumerate((2, 4, 7)):
                    helper = gate(lane_root)
                    if index:
                        with self.assertRaisesRegex(ValueError, "explicit authenticated"):
                            helper.authorize_resume()
                        helper.authorize_resume(result_for(lane_root, previous))
                    clock.advance(2000)  # reconstruction/native eval gap, ledger inactive
                    trainer = self.trainer(lane_root, clock, helper)
                    if index:
                        trainer.resume()
                    trainer.run(require_smoke=False)
                    self.assertEqual(trainer.step, expected_step)
                    previous = trainer.timed_evaluation_request
                    request = json.loads(Path(previous["path"]).read_text())
                    self.assertEqual(request["milestone_seconds"], (3, 6, 9)[index])
                    self.assertEqual(request["final_training_complete"], index == 2)
                    self.assertGreaterEqual(request["elapsed_seconds"], (3, 6, 9)[index])
                    self.assertEqual(request["checkpoint"]["sha256"], trainer.checkpoint["sha256"])
                    protected.append(Path(trainer.checkpoint["path"]))
                for checkpoint in protected:
                    self.assertTrue(checkpoint.is_file())
                    self.assertTrue(
                        json.loads((checkpoint.parent / "manifest.json").read_text())[
                            "protected_milestone"
                        ]
                    )
                self.assertEqual(trainer.cursor, full.cursor)
                self.assertEqual(trainer.epoch, full.epoch)
                self.assertEqual(trainer.tokens, full.tokens)
                for name, module in full.lanes[0].linears.items():
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(
                            trainer.lanes[0].linears[name].state_dict()[key], value, rtol=0, atol=0
                        )
                for expected, actual in zip(
                    full.lanes[0].optimizer.state.values(),
                    trainer.lanes[0].optimizer.state.values(),
                ):
                    for key in expected:
                        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
                torch.testing.assert_close(
                    full.lanes[0].rng["torch"], trainer.lanes[0].rng["torch"], rtol=0, atol=0
                )
                ledger = json.loads((lane_root / "budget-used.json").read_text())
                self.assertAlmostEqual(ledger["training_seconds"], 10)
                self.assertEqual(ledger["max_seconds"], 9)
                self.assertIsNone(ledger["active_attempt"])

    def test_pending_gate_blocks_updates_without_explicit_result(self):
        with tempfile.TemporaryDirectory() as folder, patch("time.monotonic", Clock()) as _:
            root = Path(folder)
            clock = Clock()
            with patch("time.monotonic", clock):
                trainer = self.trainer(root, clock, gate(root))
                trainer.run(require_smoke=False)
                resumed = self.trainer(root, clock, gate(root))
                resumed.resume()
                with self.assertRaisesRegex(ValueError, "explicit authenticated"):
                    resumed.run(require_smoke=False)
                self.assertEqual(resumed.step, 2)

    def test_later_periodic_checkpoint_crash_recovers_with_last_acknowledged_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root, clock = Path(folder), Clock()
            with patch("time.monotonic", clock):
                first = self.trainer(root, clock, gate(root))
                first.run(require_smoke=False)
                receipt = result_for(root, first.timed_evaluation_request)
                helper = gate(root)
                helper.authorize_resume(receipt)
                segment = self.trainer(root, clock, helper)
                segment.resume()
                original = segment.should_publish_running_status

                def fail_after_committed_periodic(step):
                    if step == 3:
                        raise RuntimeError("fixture crash after periodic checkpoint")
                    return original(step)

                segment.should_publish_running_status = fail_after_committed_periodic
                with self.assertRaisesRegex(RuntimeError, "after periodic checkpoint"):
                    segment.run(require_smoke=False)
                helper = gate(root)
                helper.authorize_resume(receipt)  # controller keeps the last acknowledged result
                with self.assertRaisesRegex(ValueError, "rolls back"):
                    helper.require_checkpoint(first.checkpoint, {"step": 1})
                clock.advance(1000)
                recovered = self.trainer(root, clock, helper)
                recovered.resume()
                self.assertEqual(recovered.step, 3)
                self.assertNotEqual(recovered.checkpoint, first.checkpoint)
                recovered.run(require_smoke=False)
                self.assertEqual(recovered.step, 4)
                request = json.loads(Path(recovered.timed_evaluation_request["path"]).read_text())
                self.assertEqual(request["milestone_index"], 1)
                self.assertAlmostEqual(request["elapsed_seconds"], 6.3)


class BlockTimedTests(unittest.TestCase):
    def transaction(
        self, root, clock, family, *, resume=False, receipt=None, stop=False, timed=True
    ):
        cfg, tensors, batch = block_fixture(family, 1)
        model = BlockDrafter(tensors, cfg)
        batch.chain_id, batch.block_index = "chain", 0

        class DataCursor:
            def __init__(self, epoch=0):
                self.epoch = epoch

            def payload(self):
                return {"epoch": self.epoch}

        class Dataset:
            def cursor(self, **_):
                return DataCursor()

            def next_block(self, cursor, **_):
                return batch, DataCursor(cursor.epoch + 1)

        protocol_path = root / "protocol.json"
        config_path = root / "config.json"
        if not config_path.exists():
            atomic_json(config_path, {"fixture": "frozen block config"})
            atomic_json(protocol_path, {"fixture": "frozen protocol"})
        args = SimpleNamespace(
            config=config_path,
            run_dir=root,
            bundle_sha256="c" * 64,
            stage_name=family + "_a1/train",
            resume=resume,
            smoke_zero_updates=False,
            prepare_only=False,
            evaluation_receipt=None if receipt is None else Path(receipt["path"]),
            evaluation_receipt_sha256=None if receipt is None else receipt["sha256"],
        )
        spec = {
            "candidate": family + "_a1",
            "family": family,
            "checkpoint_every": 1,
            "limits": {"max_seconds": 9},
            "evaluation_milestones_seconds": [3, 6, 9],
            "evaluation_protocol": artifact_locator(protocol_path),
            "checkpoint_retention": {"keep_recent": 1, "max_checkpoints": 8, "max_bytes": 10**8},
        }
        if not timed:
            spec.pop("evaluation_milestones_seconds")
            spec.pop("evaluation_protocol")
        durations = [1.4, 1.9, 0.2, 2.8, 0.4, 1.6, 1.7]

        def step(student, optimizer, batch, **kwargs):
            metrics, output = block_train_step(student, optimizer, batch, **kwargs)
            index = int(next(iter(optimizer.state.values()))["step"]) - 1
            clock.advance(durations[index])
            if stop:
                (root / "STOP").touch()
            return metrics, output

        @contextlib.contextmanager
        def teacher(*_):
            yield None

        with (
            patch.dict(
                sys.modules, {"w1a1_eagle.block_data": SimpleNamespace(BlockCursor=DataCursor)}
            ),
            patch.object(
                launcher,
                "block_inputs",
                return_value=(model, Dataset(), dict(fixtures.BlockCheckpointTests.source)),
            ),
            patch.object(launcher, "resources", return_value={"CPU fixture": True}),
            patch.object(launcher, "native_teacher", teacher),
            patch.object(launcher, "block_train_step", side_effect=step),
        ):
            return launcher.run_block(args, spec, {"CPU fixture": True})

    def test_both_families_three_boundaries_exact_exports_retention_and_receipt_gate(self):
        for family in ("dspark", "dflash"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                root, clock = Path(folder).resolve(), Clock()
                with patch("time.monotonic", clock):
                    full_root = root / "full"
                    full_root.mkdir()
                    full_result = self.transaction(full_root, clock, family, timed=False)
                    full = torch.load(full_result["checkpoint"]["path"], weights_only=False)
                    previous = None
                    checkpoints = []
                    for index, expected_step in enumerate((2, 4, 7)):
                        receipt = None
                        if index:
                            with self.assertRaisesRegex(ValueError, "explicit authenticated"):
                                self.transaction(root, clock, family, resume=True)
                            receipt = result_for(root, previous)
                        clock.advance(1000)
                        result = self.transaction(
                            root, clock, family, resume=bool(index), receipt=receipt
                        )
                        self.assertEqual(result["counters"]["step"], expected_step)
                        self.assertEqual(
                            result["completion_reason"],
                            "approved_budget_complete"
                            if index == 2
                            else "timed_evaluation_boundary",
                        )
                        self.assertEqual(result["counters"]["data_cursor"]["epoch"], expected_step)
                        checkpoints.append(Path(result["checkpoint"]["path"]))
                        previous = result["timed_evaluation_request"]
                        payload = torch.load(checkpoints[-1], weights_only=False)
                        self.assertEqual(
                            payload["cursor"], json.loads(json.dumps(result["counters"]))
                        )
                        for optimizer_state in payload["optimizer"]["state"].values():
                            self.assertEqual(float(optimizer_state["step"]), expected_step)
                        for export in result["exports"][family + "_a1"].values():
                            if isinstance(export, dict):
                                self.assertTrue(Path(export["path"]).is_file())
                    for name, expected in full["linears"].items():
                        for key, value in expected.items():
                            torch.testing.assert_close(
                                payload["linears"][name][key], value, rtol=0, atol=0
                            )
                    for key, expected in full["optimizer"]["state"].items():
                        for field, value in expected.items():
                            torch.testing.assert_close(
                                payload["optimizer"]["state"][key][field], value, rtol=0, atol=0
                            )
                    torch.testing.assert_close(
                        payload["rng"]["torch"], full["rng"]["torch"], rtol=0, atol=0
                    )
                    for path in checkpoints:
                        self.assertTrue(path.is_file())
                        sidecar = json.loads(path.with_suffix(".receipt.json").read_text())
                        self.assertIn("milestone", sidecar["retention"]["protected"])
                    self.assertLessEqual(len(list((root / "checkpoints").glob("*.pt"))), 5)
                    ledger = json.loads((root / "budget-used.json").read_text())
                    self.assertAlmostEqual(ledger["training_seconds"], 10)
                    self.assertIsNone(ledger["active_attempt"])

    def test_stop_without_pending_milestone_resumes_without_eval_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root, clock = Path(folder).resolve(), Clock()
            with patch("time.monotonic", clock):
                with self.assertRaisesRegex(InterruptedError, "STOP"):
                    self.transaction(root, clock, "dspark", stop=True)
                self.assertFalse((root / "timed-evaluation-state.json").exists())
                (root / "STOP").unlink()
                result = self.transaction(root, clock, "dspark", resume=True)
                self.assertEqual(result["counters"]["step"], 2)
                self.assertEqual(result["completion_reason"], "timed_evaluation_boundary")

    def test_later_block_progress_repair_accepts_last_receipt_without_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            root, clock = Path(folder).resolve(), Clock()
            with patch("time.monotonic", clock):
                first = self.transaction(root, clock, "dspark")
                receipt = result_for(root, first["timed_evaluation_request"])
                with self.assertRaisesRegex(InterruptedError, "STOP"):
                    self.transaction(root, clock, "dspark", resume=True, receipt=receipt, stop=True)
                latest = json.loads((root / "checkpoints/latest.json").read_text())
                self.assertEqual(latest["cursor"]["step"], 3)
                self.assertNotEqual(latest["sha256"], first["checkpoint"]["sha256"])
                (root / "STOP").unlink()
                clock.advance(1000)
                repaired = self.transaction(root, clock, "dspark", resume=True, receipt=receipt)
                self.assertEqual(repaired["counters"]["step"], 4)
                self.assertAlmostEqual(repaired["counters"]["elapsed_seconds"], 6.3)


if __name__ == "__main__":
    unittest.main()
