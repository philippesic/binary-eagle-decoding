"""Approved DSpark objective, F32 Adam time schedule and CPU transaction evidence."""

import copy
import json
import math
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from test_block_checkpoint_contract import SOURCE
from test_nine_model_training import block_fixture

from w1a1_eagle.block_qat import BlockDrafter, block_loss, block_optimizer, block_train_batch
from w1a1_eagle.block_recipe import (
    BlockRunPolicy,
    apply_schedule,
    schedule_ratio,
    schedule_state,
    seed_all,
    update_history,
)
from w1a1_eagle.block_training import BlockCursor, load_block_checkpoint, save_block_checkpoint
from w1a1_eagle.continuous_qat import sha256


def fixture():
    cfg, tensors, full = block_fixture()
    cfg = replace(
        cfg,
        objective="full_probability_l1",
        cross_entropy_weight=0.1,
        probability_l1_weight=0.9,
        depth_decay=math.exp(-0.25),
    )
    tail = copy.deepcopy(full)
    tail.loss_mask[2:] = False
    tail.labels[2:] = -1
    tail.predecessor_ids[2:] = -1
    tail.teacher_logits[2:] = 0
    return cfg, tensors, full, tail


class RecipeLossTests(unittest.TestCase):
    def test_exact_point_one_ce_point_nine_full_vocab_l1(self):
        cfg, tensors, full, tail = fixture()
        model = BlockDrafter(tensors, cfg)
        result = model(tail)
        loss, metrics = block_loss(result, tail, cfg)
        weights = torch.exp(-torch.arange(2).float() / 4)
        ce = torch.nn.functional.cross_entropy(result.logits[:2], tail.labels[:2], reduction="none")
        l1 = (result.logits[:2].softmax(-1) - tail.teacher_logits[:2].softmax(-1)).abs().sum(-1)
        expected = ((0.1 * ce + 0.9 * l1) * weights).sum() / weights.sum()
        torch.testing.assert_close(loss, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(metrics["supervised_tokens"], 2)
        self.assertEqual(result.logits.shape, (7, 19))
        self.assertTrue(((result.predecessor_ids >= 0) & (result.predecessor_ids < 19)).all())

    def test_accumulation_matches_combined_weighted_loss_and_all_gradients(self):
        cfg, tensors, full, tail = fixture()
        # Exclude clipping from this comparison; check the exact accumulated VJP.
        cfg = replace(cfg, max_grad_norm=1e6)
        originals = {name: value.clone() for name, value in tensors.items()}
        teachers = [full.teacher_logits.clone(), tail.teacher_logits.clone()]
        student, reference = BlockDrafter(tensors, cfg), BlockDrafter(tensors, cfg)
        metrics, _ = block_train_batch(
            student, block_optimizer(student), [full, tail], update=False
        )
        losses, denominators = [], []
        for batch in (full, tail):
            output = reference(batch)
            loss, counts = block_loss(output, batch, cfg)
            losses.append(loss)
            denominators.append(counts["weighted_denominator"])
        loss = sum(v * d for v, d in zip(losses, denominators)) / sum(denominators)
        loss.backward()
        self.assertAlmostEqual(metrics["loss"], float(loss.detach()), places=6)
        self.assertEqual(metrics["supervised_tokens"], 9)
        self.assertAlmostEqual(metrics["weighted_denominator"], sum(denominators), places=6)
        for left, right in zip(student.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)
        self.assertTrue((student.fc.latent_sign.grad != 0).any())
        self.assertTrue((student.layers[0].ffn_gate.latent_sign.grad != 0).any())
        self.assertTrue(all(not b.requires_grad for b in student.buffers()))
        self.assertFalse(full.teacher_logits.requires_grad)
        self.assertTrue(all(torch.equal(tensors[name], value) for name, value in originals.items()))
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(teachers, [full.teacher_logits, tail.teacher_logits])
            )
        )

    def test_invalid_masked_predecessor_not_counted(self):
        cfg, tensors, full, tail = fixture()
        full.predecessor_ids[3] = -1
        model = BlockDrafter(tensors, cfg)
        metrics, _ = block_train_batch(model, block_optimizer(model), [full], update=False)
        self.assertEqual(metrics["supervised_tokens"], 6)
        self.assertEqual(metrics["prefix_mismatch_tokens"], 1)

    def test_projection_clipping_and_legacy_coefficient_default(self):
        cfg, tensors, full, tail = fixture()
        cfg = replace(cfg, max_grad_norm=1e-5)
        model = BlockDrafter(tensors, cfg)
        metrics, _ = block_train_batch(model, block_optimizer(model), [full, tail])
        self.assertTrue(metrics["clipped"])
        self.assertEqual(metrics["effective_batch_blocks"], 2)
        self.assertTrue(
            all(
                (p.abs() <= 1).all()
                for m in model.binary_linears().values()
                for p in [m.latent_sign]
            )
        )
        self.assertTrue(
            all((m.effective_scales() >= 0).all() for m in model.binary_linears().values())
        )
        legacy = replace(cfg, cross_entropy_weight=1.0, probability_l1_weight=1.0, depth_decay=1.0)
        out = model(full)
        loss, _ = block_loss(out, full, legacy)
        expected = (
            torch.nn.functional.cross_entropy(out.logits, full.labels)
            + (out.logits.softmax(-1) - full.teacher_logits.softmax(-1)).abs().sum(-1).mean()
        )
        torch.testing.assert_close(loss, expected)


class RecipeScheduleTests(unittest.TestCase):
    def test_warmup_cosine_floor_at_exact_times_and_overage(self):
        p = BlockRunPolicy()
        self.assertEqual(schedule_ratio(p, 0), 0)
        self.assertEqual(schedule_ratio(p, 432), 0.5)
        self.assertEqual(schedule_ratio(p, 864), 1)
        self.assertAlmostEqual(schedule_ratio(p, (43200 + 864) / 2), 0.55)
        self.assertEqual(schedule_ratio(p, 43200), 0.1)
        self.assertEqual(schedule_ratio(p, 44000), 0.1)
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                schedule_ratio(p, bad)

    def test_exact_adam_rng_scheduler_reservation_resume(self):
        cfg, tensors, full, tail = fixture()
        policy = BlockRunPolicy()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        seed_all(8101)
        state = schedule_state(policy, 1234.0)
        apply_schedule(optimizer, cfg, state, 1234.0)
        metrics, _ = block_train_batch(model, optimizer, [full, tail])
        history = update_history(
            None, metrics, step=1, elapsed=1235.0, chains=["a", "b"], groups=["g1", "g2"]
        )
        cursor = BlockCursor(
            step=1,
            block_index=2,
            elapsed_seconds=1235.0,
            scheduler_state=state,
            data_cursor={"global_sha": "b" * 64, "offset": 2},
            batch_reservation={"kind": "batch", "reserved": [2, 3]},
            telemetry=history,
        )
        with tempfile.TemporaryDirectory() as tmp:
            receipt = save_block_checkpoint(model, optimizer, cursor, SOURCE, Path(tmp))
            draws = (random.random(), np.random.random(), torch.rand(4))
            fresh = BlockDrafter(tensors, cfg)
            other = block_optimizer(fresh)
            seed_all(20)
            recovered = load_block_checkpoint(fresh, other, SOURCE, receipt)
            self.assertEqual(recovered, cursor)
            self.assertEqual(draws[0], random.random())
            self.assertEqual(draws[1], np.random.random())
            self.assertTrue(torch.equal(draws[2], torch.rand(4)))
            next_state = schedule_state(policy, 1500.0)
            for opt in (optimizer, other):
                apply_schedule(opt, cfg, next_state, 1500.0)
            for m, opt in ((model, optimizer), (fresh, other)):
                block_train_batch(m, opt, [full, tail])
            for a, b in zip(model.parameters(), fresh.parameters()):
                self.assertTrue(torch.equal(a, b))
            a, b = optimizer.state_dict(), other.state_dict()
            self.assertEqual(a["param_groups"], b["param_groups"])
            for key, state in a["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, b["state"][key][name]))

    def test_failure_during_second_microbatch_replays_last_complete_transaction(self):
        cfg, tensors, full, tail = fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        state = schedule_state(BlockRunPolicy(), 1000.0)
        apply_schedule(optimizer, cfg, state, 1000.0)
        cursor = BlockCursor(
            elapsed_seconds=1000.0, scheduler_state=state, data_cursor={"offset": 0}
        )
        with tempfile.TemporaryDirectory() as tmp:
            receipt = save_block_checkpoint(model, optimizer, cursor, SOURCE, tmp)
            bad = copy.deepcopy(tail)
            bad.teacher_logits = None
            before = [p.clone() for p in model.parameters()]
            with self.assertRaisesRegex(ValueError, "full-vocabulary"):
                block_train_batch(model, optimizer, [full, bad])
            self.assertFalse(optimizer.state)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            self.assertEqual(
                json.loads((Path(tmp) / "latest.json").read_text())["cursor"]["step"], 0
            )
            fresh = BlockDrafter(tensors, cfg)
            other = block_optimizer(fresh)
            self.assertEqual(load_block_checkpoint(fresh, other, SOURCE, receipt), cursor)
            block_train_batch(model, optimizer, [full, tail])
            block_train_batch(fresh, other, [full, tail])
            for a, b in zip(model.parameters(), fresh.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_dynamic_lr_mismatch_or_schedule_tampering_rejected_before_mutation(self):
        cfg, tensors, full, tail = fixture()
        model = BlockDrafter(tensors, cfg)
        opt = block_optimizer(model)
        state = schedule_state(BlockRunPolicy(), 1000.0)
        apply_schedule(opt, cfg, state, 1000.0)
        cursor = BlockCursor(elapsed_seconds=1000.0, scheduler_state=state)
        with tempfile.TemporaryDirectory() as tmp:
            opt.param_groups[0]["lr"] *= 0.5
            with self.assertRaisesRegex(ValueError, "optimizer"):
                save_block_checkpoint(model, opt, cursor, SOURCE, tmp)
            self.assertFalse((Path(tmp) / "latest.json").exists())
            apply_schedule(opt, cfg, state, 1000.0)
            receipt = save_block_checkpoint(model, opt, cursor, SOURCE, tmp)
            saved = torch.load(receipt["path"], weights_only=False)
            saved["optimizer"]["param_groups"][0]["lr"] *= 0.5
            torch.save(saved, receipt["path"])
            receipt["sha256"] = sha256(Path(receipt["path"]))
            fresh = BlockDrafter(tensors, cfg)
            other = block_optimizer(fresh)
            before = [p.clone() for p in fresh.parameters()]
            with self.assertRaisesRegex(ValueError, "optimizer"):
                load_block_checkpoint(fresh, other, SOURCE, receipt)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, fresh.parameters())))

    def test_persistent_compact_history_keeps_exact_totals_and_coverage(self):
        cfg, tensors, full, tail = fixture()
        m = BlockDrafter(tensors, cfg)
        metrics, _ = block_train_batch(m, block_optimizer(m), [full, tail], update=False)
        history = None
        for step in range(1, 102):
            history = update_history(
                history,
                metrics,
                step=step,
                elapsed=step * 60.0,
                chains=["a", "b"],
                groups=["g1", "g2"],
            )
        self.assertEqual(history["totals"]["updates"], 101)
        self.assertEqual(history["unique_groups"], ["g1", "g2"])
        self.assertEqual(len(history["recent"]), 60)
        self.assertEqual(history["recent"][-1]["step"], 101)
        self.assertAlmostEqual(
            history["totals"]["weighted_denominator"], 101 * metrics["weighted_denominator"]
        )
        self.assertTrue(len(history["samples"]) < 25)
        json.dumps(history, allow_nan=False)


class RecipeRunloopTests(unittest.TestCase):
    def transaction(self, root, *, resume=False, missing_at_start=False):
        import contextlib
        import sys

        import train_nine_model_qat as launcher

        cfg, tensors, full, tail = fixture()
        full.chain_id, tail.chain_id = "chain-a", "chain-b"
        full.block_index, tail.block_index = 0, 0
        model = BlockDrafter(tensors, cfg)
        clock = [100.0]

        class ProviderCursor:
            def __init__(self, offset=0):
                self.offset = offset
                self.epoch = 0

            def payload(self):
                return {"offset": self.offset}

        class ShardRequired(Exception):
            def __init__(self):
                self.shard_ids = ("shard-1",)
                self.reservation = {"kind": "batch", "slots": [2, 3]}

        class Dataset:
            sha256 = "b" * 64

            def cursor(self, **kwargs):
                return ProviderCursor()

            def restore_cursor(self, payload):
                return ProviderCursor(**payload)

            def group_id(self, batch):
                return batch.chain_id

            def reserve_batch(self, cursor, size, reservation=None):
                self_case.assertEqual(size, 2)
                if missing_at_start or cursor.offset >= (4 if resume else 2):
                    clock[0] += 100
                    raise ShardRequired()
                if resume:
                    self_case.assertEqual(reservation, {"kind": "batch", "slots": [2, 3]})
                return [full, tail], ProviderCursor(cursor.offset + 2)

        self_case = self

        @contextlib.contextmanager
        def teacher(*args):
            yield None

        config = root / "config.json"
        if not config.exists():
            config.write_text("{}")
        args = SimpleNamespace(
            config=config,
            run_dir=root,
            bundle_sha256="c" * 64,
            stage_name="dspark_a8/train",
            resume=resume,
            smoke_zero_updates=False,
            prepare_only=False,
        )
        spec = {
            "candidate": "dspark_a8",
            "family": "dspark",
            "qat": asdict(cfg),
            "limits": {"max_seconds": 43200.0},
            "block_run_policy": asdict(BlockRunPolicy()),
        }
        real_train = launcher.block_train_batch

        def train(*args, **kwargs):
            out = real_train(*args, **kwargs)
            clock[0] += 901.0
            return out

        with (
            patch.dict(
                sys.modules,
                {"w1a1_eagle.block_shard_lifecycle": SimpleNamespace(ShardRequired=ShardRequired)},
            ),
            patch.object(launcher, "block_inputs", return_value=(model, Dataset(), SOURCE)),
            patch.object(launcher, "resources", return_value={}),
            patch.object(launcher, "native_teacher", teacher),
            patch.object(launcher, "block_train_batch", side_effect=train),
            patch("w1a1_eagle.continuous_budget.time.monotonic", side_effect=lambda: clock[0]),
        ):
            result = launcher.run_block(args, spec, {"device": "synthetic CPU"})
        return result, model

    def test_invalid_or_reused_budget_fails_before_actual_gradients_or_publication(self):
        import train_nine_model_qat as launcher

        valid = {
            "schema": "continuous_training_budget_v1",
            "source_sha256": "c" * 64,
            "max_seconds": 43200.0,
            "training_seconds": 1.0,
            "active_attempt": None,
        }
        cases = (
            ("corrupt", {**valid, "source_sha256": "d" * 64}, "contract differs"),
            ("invalid", {**valid, "training_seconds": -1.0}, "budget is invalid"),
            ("reused", valid, "requires exact checkpoint resume"),
        )
        for label, ledger, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "budget-used.json").write_text(json.dumps(ledger))
                with (
                    patch.object(launcher, "smoke_block", wraps=launcher.smoke_block) as smoke,
                    patch.object(
                        launcher, "save_block_checkpoint", wraps=launcher.save_block_checkpoint
                    ) as publish,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        self.transaction(root)
                    smoke.assert_not_called()
                    publish.assert_not_called()
                self.assertFalse((root / "checkpoints").exists())

    def test_expired_resumed_budget_fails_before_gradients_or_new_checkpoint(self):
        import train_nine_model_qat as launcher

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.transaction(root)
            ledger = json.loads((root / "budget-used.json").read_text())
            ledger["training_seconds"] = 43200.0
            (root / "budget-used.json").write_text(json.dumps(ledger))
            previous = (root / "checkpoints/latest.json").read_bytes()
            with (
                patch.object(launcher, "smoke_block", wraps=launcher.smoke_block) as smoke,
                patch.object(
                    launcher, "save_block_checkpoint", wraps=launcher.save_block_checkpoint
                ) as publish,
            ):
                with self.assertRaisesRegex(ValueError, "allocation exhausted"):
                    self.transaction(root, resume=True)
                smoke.assert_not_called()
                publish.assert_not_called()
            self.assertEqual((root / "checkpoints/latest.json").read_bytes(), previous)

    def test_early_crash_charge_is_retained_and_startup_stays_unbilled(self):
        import train_nine_model_qat as launcher

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.transaction(root)
            ledger = json.loads((root / "budget-used.json").read_text())
            # Same-boot dead attempt: the helper's monotonic clock begins at100.
            ledger["active_attempt"] = {
                "boot_id": "test-boot",
                "pid": 999999,
                "process_birth": "old",
                "started_monotonic": 50.0,
                "reserved_seconds": 1000.0,
            }
            (root / "budget-used.json").write_text(json.dumps(ledger))
            real_smoke = launcher.smoke_block

            def startup(*args, **kwargs):
                settled = json.loads((root / "budget-used.json").read_text())
                self.assertIsNone(settled["active_attempt"])
                self.assertEqual(settled["training_seconds"], 1051.0)
                return real_smoke(*args, **kwargs)

            with (
                patch("w1a1_eagle.continuous_budget.boot_identity", return_value="test-boot"),
                patch("w1a1_eagle.continuous_budget.process_birth", return_value=None),
                patch.object(launcher, "smoke_block", side_effect=startup),
            ):
                result, _ = self.transaction(root, resume=True)
            self.assertEqual(result["counters"]["elapsed_seconds"], 2052.0)

    def test_shard_boundary_keeps_completed_pair_pending_reservation_and_paid_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, model = self.transaction(root)
            self.assertEqual(result["completion_reason"], "shard_required")
            cursor = result["counters"]
            self.assertEqual(cursor["step"], 1)
            self.assertEqual(cursor["block_index"], 2)
            self.assertEqual(cursor["data_cursor"], {"offset": 2})
            self.assertEqual(cursor["batch_reservation"], {"kind": "batch", "slots": [2, 3]})
            self.assertEqual(cursor["last_checkpoint_seconds"], 901.0)
            self.assertEqual(cursor["elapsed_seconds"], 1001.0)
            self.assertEqual(cursor["supervised_tokens"], 9)
            self.assertEqual(result["shard_request"]["plan_sha256"], "b" * 64)
            ledger = json.loads((root / "budget-used.json").read_text())
            self.assertIsNone(ledger["active_attempt"])
            self.assertEqual(ledger["training_seconds"], 1001.0)
            self.assertFalse((root / "final-export").exists())
            # One initial, one 900-second ordinary checkpoint, one shard boundary.
            self.assertEqual(len(list((root / "checkpoints").glob("*.pt"))), 3)
            continued, _ = self.transaction(root, resume=True)
            self.assertEqual(continued["counters"]["step"], 2)
            self.assertEqual(continued["counters"]["data_cursor"], {"offset": 4})
            self.assertEqual(continued["counters"]["supervised_tokens"], 18)
            self.assertEqual(continued["counters"]["telemetry"]["totals"]["updates"], 2)
            self.assertEqual(continued["counters"]["elapsed_seconds"], 2002.0)
            self.assertGreater(continued["counters"]["scheduler_state"]["applied_seconds"], 864.0)

    def test_step_zero_shard_boundary_has_no_update_or_paid_bulk_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            result, model = self.transaction(Path(folder), missing_at_start=True)
            self.assertEqual(result["counters"]["step"], 0)
            self.assertEqual(result["counters"]["elapsed_seconds"], 0.0)
            self.assertEqual(result["counters"]["data_cursor"], {"offset": 0})
            saved = torch.load(result["checkpoint"]["path"], weights_only=False)
            self.assertEqual(saved["optimizer"]["state"], {})
            self.assertEqual(
                saved["cursor"]["batch_reservation"], {"kind": "batch", "slots": [2, 3]}
            )


if __name__ == "__main__":
    unittest.main()
