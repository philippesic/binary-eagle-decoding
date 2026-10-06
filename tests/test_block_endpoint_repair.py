"""Final export recovery with actual CPU transactions and an artificial clock."""

import contextlib
import copy
import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import train_nine_model_qat as launcher
from test_block_checkpoint_contract import SOURCE
from test_block_recipe import fixture
from test_timed_training import result_for

from w1a1_eagle.block_qat import BlockDrafter
from w1a1_eagle.block_recipe import BlockRunPolicy
from w1a1_eagle.continuous_budget import TrainingBudget, artifact_locator
from w1a1_eagle.continuous_qat import atomic_json, rng_state


class BlockEndpointRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importlib.import_module("torch._dynamo")

    def exercise(self, root, *, reject_changed_ledger=False):
        cfg, tensors, full, tail = fixture()
        full.chain_id, tail.chain_id = "chain-a", "chain-b"
        full.block_index = tail.block_index = 0
        clock = [100.0]

        class Cursor:
            epoch = 0

            def __init__(self, offset=0):
                self.offset = offset

            def payload(self):
                return {"offset": self.offset}

        class Dataset:
            sha256 = "b" * 64

            def cursor(self, **kwargs):
                return Cursor()

            def restore_cursor(self, payload):
                return Cursor(**payload)

            def group_id(self, batch):
                return batch.chain_id

            def reserve_batch(self, cursor, size, reservation=None):
                self_test.assertEqual(size, 2)
                return [full, tail], Cursor(cursor.offset + 2)

        class ShardRequired(Exception):
            pass

        self_test = self

        @contextlib.contextmanager
        def teacher(*args):
            yield None

        real_train = launcher.block_train_batch
        real_export = launcher.export_block_checkpoint
        real_optimizer = launcher.block_optimizer
        atomic_json(root / "config.json", {"fixture": "approved endpoint repair"})
        atomic_json(root / "protocol.json", {"fixture": "CPU native receipt"})
        spec = {
            "candidate": "dspark_a8",
            "family": "dspark",
            "qat": asdict(cfg),
            "limits": {"max_seconds": 43200.0},
            "block_run_policy": asdict(BlockRunPolicy()),
            "evaluation_milestones_seconds": [14400.0, 28800.0, 43200.0],
            "evaluation_protocol": artifact_locator(root / "protocol.json"),
        }
        optimizer_refs = []

        def optimizer(model):
            value = real_optimizer(model)
            optimizer_refs.append(value)
            return value

        def train(*args, **kwargs):
            result = real_train(*args, **kwargs)
            clock[0] += 7201.0
            return result

        def invoke(evaluation=None, *, resume=False, failure=False, repair=False):
            model = BlockDrafter(tensors, cfg)
            args = SimpleNamespace(
                config=root / "config.json",
                run_dir=root,
                bundle_sha256="c" * 64,
                stage_name="dspark_a8/train",
                resume=resume,
                smoke_zero_updates=False,
                prepare_only=False,
                evaluation_receipt=None if evaluation is None else Path(evaluation["path"]),
                evaluation_receipt_sha256=None if evaluation is None else evaluation["sha256"],
            )

            def export(*args):
                if failure:
                    Path(args[2]).mkdir(parents=True)
                    (Path(args[2]) / "partial.bin").write_bytes(b"preserved failure")
                    raise OSError("injected final serializer failure")
                # Wall time in repair cannot alter the already settled ledger.
                if repair:
                    clock[0] += 10000.0
                return real_export(*args)

            with (
                patch.dict(
                    sys.modules,
                    {
                        "w1a1_eagle.block_shard_lifecycle": SimpleNamespace(
                            ShardRequired=ShardRequired
                        )
                    },
                ),
                patch.object(launcher, "block_inputs", return_value=(model, Dataset(), SOURCE)),
                patch.object(launcher, "block_optimizer", side_effect=optimizer),
                patch.object(launcher, "resources", return_value={}),
                patch.object(launcher, "native_teacher", teacher),
                patch.object(
                    launcher,
                    "block_train_batch",
                    side_effect=AssertionError("repair update") if repair else train,
                ) as updates,
                patch.object(
                    launcher,
                    "smoke_block",
                    side_effect=AssertionError("repair gradient")
                    if repair
                    else launcher.smoke_block,
                ) as smoke,
                patch.object(launcher, "export_block_checkpoint", side_effect=export),
                patch("w1a1_eagle.continuous_budget.time.monotonic", side_effect=lambda: clock[0]),
            ):
                if repair:
                    with (
                        patch.object(
                            TrainingBudget,
                            "begin",
                            side_effect=AssertionError("repair billed time"),
                        ) as begin,
                        patch(
                            "w1a1_eagle.block_training._restore_block_rng",
                            side_effect=AssertionError("repair RNG restoration"),
                        ) as restore,
                    ):
                        try:
                            output = launcher.run_block(args, spec, {"device": "CPU fixture"})
                        finally:
                            updates.assert_not_called()
                            smoke.assert_not_called()
                            begin.assert_not_called()
                            restore.assert_not_called()
                else:
                    output = launcher.run_block(args, spec, {"device": "CPU fixture"})
            return output, model

        evaluation = None
        for milestone in (1, 2):
            output, _ = invoke(evaluation, resume=milestone > 1)
            self.assertEqual(output["counters"]["elapsed_seconds"], milestone * 14402.0)
            evaluation = result_for(root, output["timed_evaluation_request"])
        with self.assertRaisesRegex(OSError, "injected final serializer"):
            invoke(evaluation, resume=True, failure=True)
        latest = json.loads((root / "checkpoints/latest.json").read_text())
        payload = torch.load(latest["path"], weights_only=False)
        self.assertEqual(payload["cursor"]["elapsed_seconds"], 43206.0)
        self.assertEqual(payload["cursor"]["step"], 6)
        self.assertEqual(
            payload["cursor"]["completed_endpoint"]["budget_ledger"],
            artifact_locator(root / "budget-used.json"),
        )
        ledger_before = (root / "budget-used.json").read_bytes()
        checkpoint_before = Path(latest["path"]).read_bytes()
        latest_before = (root / "checkpoints/latest.json").read_bytes()
        caller_rng = copy.deepcopy(rng_state("cpu"))
        if reject_changed_ledger:
            altered = json.loads(ledger_before)
            altered["training_seconds"] += 1.0
            atomic_json(root / "budget-used.json", altered)
            with self.assertRaisesRegex(ValueError, "allocation exhausted"):
                invoke(evaluation, resume=True, repair=True)
            self.assertFalse((root / "timed-evaluation/milestone-02/request.json").exists())
            return
        output, repaired_model = invoke(evaluation, resume=True, repair=True)
        self.assertEqual(output["completion_reason"], "approved_budget_complete")
        self.assertEqual(json.loads(json.dumps(output["counters"])), latest["cursor"])
        self.assertEqual(output["checkpoint"], {k: latest[k] for k in ("path", "sha256")})
        self.assertEqual((root / "budget-used.json").read_bytes(), ledger_before)
        self.assertEqual(Path(latest["path"]).read_bytes(), checkpoint_before)
        self.assertEqual((root / "checkpoints/latest.json").read_bytes(), latest_before)
        after_rng = rng_state("cpu")
        self.assertEqual(caller_rng["python"], after_rng["python"])
        self.assertEqual(caller_rng["numpy"][0], after_rng["numpy"][0])
        self.assertTrue(np.array_equal(caller_rng["numpy"][1], after_rng["numpy"][1]))
        self.assertEqual(caller_rng["numpy"][2:], after_rng["numpy"][2:])
        self.assertTrue(torch.equal(caller_rng["torch"], after_rng["torch"]))
        actual = optimizer_refs[-1].state_dict()
        self.assertEqual(actual["param_groups"], payload["optimizer"]["param_groups"])
        for key, values in actual["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, payload["optimizer"]["state"][key][name]))
        for name, module in repaired_model.binary_linears().items():
            for key, value in module.state_dict().items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, payload["linears"][name][key]))
        request = json.loads(Path(output["timed_evaluation_request"]["path"]).read_text())
        self.assertEqual(request["milestone_seconds"], 43200.0)
        self.assertEqual(request["elapsed_seconds"], 43206.0)
        self.assertTrue(request["final_training_complete"])
        self.assertTrue(list((root / "timed-evaluation/milestone-02").glob("export-*/partial.bin")))

    def test_complete_endpoint_repairs_failed_export_without_training_or_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp))

    def test_changed_settled_ledger_rejects_endpoint_repair_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), reject_changed_ledger=True)


if __name__ == "__main__":
    unittest.main()
