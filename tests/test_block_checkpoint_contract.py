"""Independent corruption paths for exact stage-local Adam publication/recovery."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from test_nine_model_training import block_fixture

from w1a1_eagle.block_qat import BlockDrafter, block_optimizer, block_train_step
from w1a1_eagle.block_training import (
    BlockCursor,
    load_block_checkpoint,
    save_block_checkpoint,
    transition_a8_to_a1,
)
from w1a1_eagle.continuous_qat import sha256

SOURCE = {
    "base_gguf_sha256": "a" * 64,
    "data_manifest_sha256": "b" * 64,
    "bundle_sha256": "c" * 64,
    "synthetic": True,
}


def corrupt(states, kind):
    key = next(iter(states))
    if kind == "drop_all":
        states.clear()
    elif kind == "drop_owned_state":
        del states[key]
    elif kind == "stale":
        states[key]["step"].fill_(0)
    elif kind == "mixed":
        states[key]["step"].fill_(2)
    elif kind == "step_dtype":
        states[key]["step"] = states[key]["step"].to(torch.int64)
    elif kind == "step_shape":
        states[key]["step"] = states[key]["step"].reshape(1)
    elif kind == "moment_dtype":
        states[key]["exp_avg"] = states[key]["exp_avg"].double()


class BlockCheckpointContractTests(unittest.TestCase):
    def test_save_rejects_missing_stale_mixed_or_untyped_state_before_publication(self):
        for kind in (
            "drop_all",
            "drop_owned_state",
            "stale",
            "mixed",
            "step_dtype",
            "step_shape",
            "moment_dtype",
        ):
            cfg, tensors, batch = block_fixture()
            model = BlockDrafter(tensors, cfg)
            optimizer = block_optimizer(model)
            block_train_step(model, optimizer, batch)
            corrupt(optimizer.state, kind)
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    save_block_checkpoint(
                        model, optimizer, BlockCursor(step=1), SOURCE, Path(folder)
                    )
                self.assertFalse((Path(folder) / "latest.json").exists())

    def test_load_rejects_same_cases_before_parameter_mutation(self):
        for kind in (
            "drop_all",
            "drop_owned_state",
            "stale",
            "mixed",
            "step_dtype",
            "step_shape",
            "moment_dtype",
        ):
            cfg, tensors, batch = block_fixture()
            model = BlockDrafter(tensors, cfg)
            optimizer = block_optimizer(model)
            block_train_step(model, optimizer, batch)
            with tempfile.TemporaryDirectory() as folder:
                receipt = save_block_checkpoint(
                    model, optimizer, BlockCursor(step=1), SOURCE, Path(folder)
                )
                payload = torch.load(receipt["path"], weights_only=False)
                corrupt(payload["optimizer"]["state"], kind)
                torch.save(payload, receipt["path"])
                receipt["sha256"] = sha256(Path(receipt["path"]))
                fresh = BlockDrafter(tensors, cfg)
                fresh_opt = block_optimizer(fresh)
                before = [p.detach().clone() for p in fresh.parameters()]
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    load_block_checkpoint(fresh, fresh_opt, SOURCE, receipt)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, fresh.parameters())))
                self.assertFalse(fresh_opt.state)

    def test_reset_requires_empty_optimizer_and_next_a1_step_resumes_exactly(self):
        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        block_train_step(model, optimizer, batch)
        old = copy.deepcopy(optimizer.state_dict())
        final, reset, _ = transition_a8_to_a1(
            model, source_checkpoint_sha256="d" * 64, in_place=True
        )
        cursor = BlockCursor(step=1, stage="a1_final", stage_updates=0)
        reset.load_state_dict(old)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "empty optimizer"):
                save_block_checkpoint(final, reset, cursor, SOURCE, Path(folder))
            reset = block_optimizer(final)
            first = save_block_checkpoint(final, reset, cursor, SOURCE, Path(folder))
            clone = copy.deepcopy(final)
            restored = block_optimizer(clone)
            actual = load_block_checkpoint(clone, restored, SOURCE, first)
            self.assertEqual(actual.stage_updates, 0)
            block_train_step(final, reset, batch)
            block_train_step(clone, restored, batch)
            self.assertTrue(
                all(torch.equal(a, b) for a, b in zip(final.parameters(), clone.parameters()))
            )
            endpoint = save_block_checkpoint(
                final,
                reset,
                BlockCursor(step=2, stage="a1_final", stage_updates=1),
                SOURCE,
                Path(folder),
            )
            self.assertEqual(endpoint["cursor"]["stage_updates"], 1)
