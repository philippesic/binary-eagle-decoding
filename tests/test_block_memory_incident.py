"""CPU state/incident evidence; CUDA snapshots are mocks, never SM120 admission."""

import contextlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import train_nine_model_qat as launcher
from test_block_checkpoint_contract import SOURCE
from test_block_recipe import fixture

from w1a1_eagle.block_qat import BlockDrafter, block_optimizer, block_train_batch
from w1a1_eagle.block_recipe import BlockRunPolicy
from w1a1_eagle.block_training import load_block_checkpoint


def assert_tree_equal(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(type(left), type(right))
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class MemoryReleaseTests(unittest.TestCase):
    def test_two_effective_updates_preserve_parameters_moments_rng_and_teachers(self):
        cfg, tensors, full, tail = fixture()
        original, repaired = BlockDrafter(tensors, cfg), BlockDrafter(tensors, cfg)
        a, b = block_optimizer(original), block_optimizer(repaired)
        teachers = [full.teacher_logits.clone(), tail.teacher_logits.clone()]
        with patch.object(launcher, "resources", return_value={"fixture_only": True}):
            for _ in range(2):
                before = torch.get_rng_state().clone()
                launcher.block_transaction_resources({}, repaired, b)
                self.assertTrue(all(p.grad is None for p in repaired.parameters()))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                expected, _ = block_train_batch(original, a, [full, tail], return_outputs=False)
                actual, _ = block_train_batch(repaired, b, [full, tail], return_outputs=False)
                self.assertEqual(actual, expected)
                assert_tree_equal(self, original.state_dict(), repaired.state_dict())
                assert_tree_equal(self, a.state_dict(), b.state_dict())
        assert_tree_equal(self, teachers, [full.teacher_logits, tail.teacher_logits])

    def test_healthy_cuda_cache_stays_warm_without_sync_or_release(self):
        model = SimpleNamespace(token_embd=SimpleNamespace(device=SimpleNamespace(type="cuda")))
        optimizer = SimpleNamespace(
            zero_grad=lambda **kwargs: self.assertEqual(kwargs, {"set_to_none": True})
        )
        measured = {"cuda_allocated_bytes": 9, "cuda_reserved_bytes": 11, "cuda_free_bytes": 3}
        with (
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
            patch.object(launcher, "resources", return_value=measured) as guard,
        ):
            result = launcher.block_transaction_resources({}, model, optimizer)
        synchronize.assert_not_called()
        empty_cache.assert_not_called()
        guard.assert_called_once_with({}, "block optimizer update")
        self.assertFalse(result["cleanup_performed"])
        self.assertEqual(result["before_cache_release"], measured)
        for key, value in measured.items():
            self.assertEqual(result[key], value)

    def test_failed_cuda_gate_trims_once_and_rechecks_strictly(self):
        events = []
        model = SimpleNamespace(token_embd=SimpleNamespace(device=SimpleNamespace(type="cuda")))
        optimizer = SimpleNamespace(zero_grad=lambda **kwargs: events.append(("zero_grad", kwargs)))
        before = {"cuda_allocated_bytes": 9, "cuda_reserved_bytes": 15, "cuda_free_bytes": 1}
        after = {"cuda_allocated_bytes": 8, "cuda_reserved_bytes": 13, "cuda_free_bytes": 2}
        initial_failure = launcher.ResourceLimitError("block optimizer update", before)
        failure = launcher.ResourceLimitError("block optimizer update", after)
        results = iter((initial_failure, failure))

        def fail_guard(*args):
            events.append("guard")
            raise next(results)

        with (
            patch.object(torch.cuda, "synchronize", side_effect=lambda *_: events.append("sync")),
            patch.object(torch.cuda, "empty_cache", side_effect=lambda: events.append("empty")),
            patch.object(launcher, "resources", side_effect=fail_guard),
        ):
            with self.assertRaises(launcher.ResourceLimitError) as caught:
                launcher.block_transaction_resources({}, model, optimizer)
        self.assertIs(caught.exception, failure)
        self.assertEqual(
            events, [("zero_grad", {"set_to_none": True}), "guard", "sync", "empty", "guard"]
        )
        self.assertTrue(failure.resources["cleanup_performed"])
        self.assertEqual(failure.resources["before_cache_release"], before)
        self.assertEqual(failure.resources["cuda_reserved_bytes"], 13)

    def test_conditional_cuda_trim_returns_recovered_before_and_after_metrics(self):
        model = SimpleNamespace(token_embd=SimpleNamespace(device=SimpleNamespace(type="cuda")))
        optimizer = SimpleNamespace(zero_grad=lambda **_: None)
        before = {"cuda_allocated_bytes": 9, "cuda_reserved_bytes": 15, "cuda_free_bytes": 1}
        after = {"cuda_allocated_bytes": 8, "cuda_reserved_bytes": 10, "cuda_free_bytes": 3}
        with (
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
            patch.object(
                launcher,
                "resources",
                side_effect=[launcher.ResourceLimitError("guard", before), after],
            ) as guard,
        ):
            result = launcher.block_transaction_resources({}, model, optimizer)
        synchronize.assert_called_once_with("cuda:0")
        empty_cache.assert_called_once_with()
        self.assertEqual(guard.call_count, 2)
        self.assertTrue(result["cleanup_performed"])
        self.assertEqual(result["before_cache_release"], before)
        for key, value in after.items():
            self.assertEqual(result[key], value)

    def test_exact_reserved_cap_and_free_floor_are_preserved(self):
        cap, floor = 12 * 1024**3, 1024**3
        spec = {"resource_floors": {"cuda_reserved_bytes": cap, "cuda_free_bytes": floor}}
        for reserved, free, fails in (
            (cap, floor, False),
            (cap + 1, floor, True),
            (cap, floor - 1, True),
        ):
            measured = {
                "cuda_reserved_bytes": reserved,
                "cuda_free_bytes": free,
                "cuda_allocated_bytes": cap - 2,
            }
            with (
                patch.object(launcher, "linux_host_memory", return_value={}),
                patch.object(launcher, "require_host_memory", return_value={}),
                patch.object(launcher, "cuda_memory_snapshot", return_value=measured),
            ):
                if fails:
                    with self.assertRaises(launcher.ResourceLimitError) as caught:
                        launcher.resources(spec, "test gate")
                    self.assertEqual(caught.exception.resources["limits"], spec["resource_floors"])
                    for key, value in measured.items():
                        self.assertEqual(caught.exception.resources[key], value)
                else:
                    self.assertEqual(
                        launcher.resources(spec, "test gate")["limits"], spec["resource_floors"]
                    )


class IncidentCheckpointTests(unittest.TestCase):
    def run_incident(self, root, *, mid_update=False, checkpoint_failure=False):
        cfg, tensors, full, tail = fixture()
        full.chain_id, tail.chain_id = "chain-a", "chain-b"
        full.block_index, tail.block_index = 0, 0
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        clock, updates = [100.0], [0]
        failure = (
            RuntimeError("partial optimizer failure")
            if mid_update
            else launcher.ResourceLimitError(
                "block optimizer update",
                {"cuda_allocated_bytes": 10, "cuda_reserved_bytes": 13, "cuda_free_bytes": 2},
            )
        )

        class ProviderCursor:
            def __init__(self, offset=0):
                self.offset, self.epoch = offset, 0

            def payload(self):
                return {"offset": self.offset}

        class Dataset:
            def cursor(self, **kwargs):
                return ProviderCursor()

            def restore_cursor(self, payload):
                return ProviderCursor(**payload)

            def reserve_batch(self, cursor, size, reservation=None):
                return [full, tail], ProviderCursor(cursor.offset + 2)

            def group_id(self, batch):
                return batch.chain_id

        @contextlib.contextmanager
        def teacher(*args):
            yield None

        def resource(*args):
            if not mid_update and updates[0] == 2:
                raise failure
            return {"cuda_allocated_bytes": 9, "cuda_reserved_bytes": 11, "cuda_free_bytes": 3}

        real_step = optimizer.step

        def partial_step(*args, **kwargs):
            real_step(*args, **kwargs)
            raise failure

        def train(*args, **kwargs):
            updates[0] += 1
            clock[0] += 2.0
            if mid_update and updates[0] == 2:
                with patch.object(optimizer, "step", side_effect=partial_step):
                    return block_train_batch(*args, **kwargs)
            return block_train_batch(*args, **kwargs)

        real_save = launcher.save_block_checkpoint

        def save(*args, **kwargs):
            if checkpoint_failure and args[2].step > 0:
                raise OSError("synthetic checkpoint disk failure")
            return real_save(*args, **kwargs)

        config = root / "config.json"
        config.write_text("{}")
        args = SimpleNamespace(
            config=config,
            run_dir=root,
            bundle_sha256="c" * 64,
            stage_name="dspark_a8/train",
            resume=False,
            smoke_zero_updates=False,
            prepare_only=False,
        )
        spec = {
            "candidate": "dspark_a8",
            "family": "dspark",
            "qat": asdict(cfg),
            "limits": {"max_seconds": 43200.0},
            "block_run_policy": asdict(BlockRunPolicy()),
            "checkpoint_retention": {
                "keep_recent": 3,
                "max_checkpoints": 10,
                "max_bytes": 56 * 1024**3,
            },
        }
        with (
            patch.object(launcher, "block_inputs", return_value=(model, Dataset(), SOURCE)),
            patch.object(launcher, "block_optimizer", return_value=optimizer),
            patch.object(launcher, "resources", side_effect=resource),
            patch.object(launcher, "native_teacher", teacher),
            patch.object(launcher, "smoke_with_training_memory", return_value=({}, {})),
            patch.object(launcher, "block_train_batch", side_effect=train),
            patch.object(launcher, "save_block_checkpoint", side_effect=save),
            patch("w1a1_eagle.continuous_budget.time.monotonic", side_effect=lambda: clock[0]),
        ):
            with self.assertRaises(RuntimeError) as caught:
                launcher.run_block(args, spec, {"device": "synthetic CPU"})
        self.assertIs(caught.exception, failure)
        self.assertFalse((root / "final-export").exists())
        ledger = json.loads((root / "budget-used.json").read_text())
        self.assertIsNone(ledger["active_attempt"])
        self.assertEqual(ledger["training_seconds"], 4.0)
        incident_paths = list((root / "incidents").glob("*.json"))
        self.assertEqual(len(incident_paths), 1)
        incident = json.loads(incident_paths[0].read_text())
        self.assertEqual(incident["training_seconds"], 4.0)
        self.assertEqual(incident["error"], f"{type(failure).__name__}: {failure}")
        self.assertFalse(incident["evaluation_permitted"])
        self.assertEqual(json.loads((root / "status.json").read_text())["status"], "FAIL")
        return cfg, tensors, model, optimizer, incident, failure

    def test_pretransaction_guard_failure_saves_two_completed_updates_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cfg, tensors, model, optimizer, incident, _ = self.run_incident(root)
            receipt = json.loads((root / "checkpoints/latest.json").read_text())
            self.assertTrue(incident["optimizer_transaction_complete"])
            self.assertEqual(incident["checkpoint"], receipt)
            self.assertEqual(receipt["cursor"]["step"], 2)
            self.assertEqual(receipt["cursor"]["supervised_tokens"], 18)
            self.assertEqual(receipt["cursor"]["data_cursor"], {"offset": 4})
            self.assertEqual(receipt["cursor"]["elapsed_seconds"], 4.0)
            self.assertIn("stop", receipt["retention"]["protected"])
            restored = BlockDrafter(tensors, cfg)
            restored_optimizer = block_optimizer(restored)
            before = torch.get_rng_state().clone()
            cursor = load_block_checkpoint(restored, restored_optimizer, SOURCE, receipt)
            self.assertEqual(cursor.step, 2)
            assert_tree_equal(self, model.state_dict(), restored.state_dict())
            assert_tree_equal(self, optimizer.state_dict(), restored_optimizer.state_dict())
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(len(list((root / "checkpoints").glob("*.pt"))), 2)

    def test_partial_second_optimizer_update_never_publishes_incident_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            *_, incident, _ = self.run_incident(root, mid_update=True)
            self.assertFalse(incident["optimizer_transaction_complete"])
            self.assertIsNone(incident["checkpoint"])
            self.assertEqual(incident["last_completed_cursor"]["step"], 1)
            latest = json.loads((root / "checkpoints/latest.json").read_text())
            self.assertEqual(latest["cursor"]["step"], 0)
            self.assertEqual(len(list((root / "checkpoints").glob("*.pt"))), 1)

    def test_checkpoint_failure_preserves_original_guard_error_and_paid_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            *_, incident, failure = self.run_incident(root, checkpoint_failure=True)
            self.assertIsNone(incident["checkpoint"])
            self.assertIn("synthetic checkpoint disk failure", incident["checkpoint_error"])
            self.assertIn("incident checkpoint failed", failure.__notes__[0])
            self.assertEqual(
                json.loads((root / "checkpoints/latest.json").read_text())["cursor"]["step"], 0
            )


if __name__ == "__main__":
    unittest.main()
