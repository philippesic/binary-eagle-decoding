"""Bounded telemetry equivalence; CPU evidence, no native acceptance claim."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from test_block_checkpoint_contract import SOURCE
from test_block_recipe import fixture

from w1a1_eagle.block_diagnostics import (
    SAMPLE_SIZE,
    BlockDiagnosticSession,
    exposure_metadata,
    fixed_indices,
    slot_metrics,
)
from w1a1_eagle.block_qat import BlockDrafter, block_optimizer, block_train_batch
from w1a1_eagle.block_recipe import update_history
from w1a1_eagle.block_training import BlockCursor, load_block_checkpoint, save_block_checkpoint
from w1a1_eagle.continuous_qat import rng_state


class BlockDiagnosticTests(unittest.TestCase):
    def check_rng(self, before, after):
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"][0], after["numpy"][0])
        self.assertTrue(np.array_equal(before["numpy"][1], after["numpy"][1]))
        self.assertEqual(before["numpy"][2:], after["numpy"][2:])
        self.assertTrue(torch.equal(before["torch"], after["torch"]))

    def test_diagnostics_on_off_exact_gradients_parameters_moments_loss_and_rng(self):
        cfg, tensors, full, tail = fixture()
        model, other = BlockDrafter(tensors, cfg), BlockDrafter(tensors, cfg)
        optimizer, reference = block_optimizer(model), block_optimizer(other)
        before = copy.deepcopy(rng_state("cpu"))
        session = BlockDiagnosticSession(model, None, 100)
        with (
            session.capture(model),
            patch.object(model, "forward", wraps=model.forward) as forwards,
        ):
            observed, outputs = block_train_batch(
                model, optimizer, [full, tail], diagnostics=True, return_outputs=False
            )
        self.assertEqual(forwards.call_count, 2)
        report = session.finish(model)
        baseline, _ = block_train_batch(other, reference, [full, tail], return_outputs=False)
        self.assertEqual(outputs, [])
        for key, value in baseline.items():
            self.assertEqual(observed[key], value)
        for a, b in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(a.grad, b.grad))
        a, b = optimizer.state_dict(), reference.state_dict()
        self.assertEqual(a["param_groups"], b["param_groups"])
        for key, state in a["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, b["state"][key][name]))
        self.check_rng(before, rng_state("cpu"))
        self.assertEqual(observed["slot_supported"], [2, 2, 1, 1, 1, 1, 1])
        self.assertEqual(len(report["sample"]), 16)
        for layer in report["sample"].values():
            self.assertLessEqual(layer["latent_samples"], SAMPLE_SIZE)
            self.assertLessEqual(layer["scale_samples"], SAMPLE_SIZE)
            self.assertEqual(layer["a8_status"], "measured_fixed_coordinate_sample")
            self.assertLessEqual(
                layer["a8_hard_code_endpoint_occupancy"]["sampled_codes"], 2 * SAMPLE_SIZE
            )
        self.assertTrue(
            all(getattr(m, "_diagnostic_a8", None) is None for m in model.binary_linears().values())
        )

    def test_top1_and_survival_use_valid_partial_slots_and_existing_logits(self):
        cfg, _, _, tail = fixture()
        logits = torch.full((7, cfg.vocab_size), -100.0)
        logits[0, tail.labels[0]] = 100
        logits[1, 0] = 100
        result = slot_metrics(SimpleNamespace(logits=logits), tail, cfg)
        self.assertEqual(result["slot_supported"], [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(result["teacher_forced_top1_hits"], [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result["captured_prefix_survival_hits"], [1, 0, 0, 0, 0, 0, 0])
        self.assertIn("not_native_acceptance", result["diagnostic_semantics"])

    def test_declared_domain_length_and_measured_context_exposure_are_truthful(self):
        cfg, tensors, full, tail = fixture()
        full.chain_id = "known"
        tail.chain_id = "missing"
        metadata = exposure_metadata(
            SimpleNamespace(chains={"known": {"domain": "code", "input_tokens": 700}}), [full, tail]
        )
        self.assertEqual(metadata[0]["domain_declared"], "code")
        self.assertIsNone(metadata[1]["domain_declared"])
        self.assertEqual(metadata[0]["source_input_tokens_declared"], 700)
        self.assertEqual(metadata[0]["context_tokens_measured"], 3)
        model = BlockDrafter(tensors, cfg)
        metrics, _ = block_train_batch(
            model, block_optimizer(model), [full, tail], diagnostics=True, update=False
        )
        history = update_history(
            None,
            metrics,
            step=1,
            elapsed=1.0,
            chains=["known", "missing"],
            groups=[],
            exposure=metadata,
        )
        self.assertEqual(history["exposure"]["domain_declared"]["code"]["blocks"], 1)
        self.assertEqual(
            history["exposure"]["domain_declared"]["unavailable"]["supervised_slots"], 2
        )
        self.assertEqual(
            history["exposure"]["source_input_tokens_declared"]["over_512"]["blocks"], 1
        )
        self.assertEqual(history["slot_totals"]["slot_supported"], [2, 2, 1, 1, 1, 1, 1])

    def test_interval_flip_back_state_and_metric_history_resume_exactly(self):
        cfg, tensors, full, tail = fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        initial = BlockDiagnosticSession(model, None, 1)
        history = {"diagnostic_sample_state": initial.state}
        for module in model.binary_linears().values():
            with torch.no_grad():
                module.latent_sign.mul_(-1)
        session = BlockDiagnosticSession(model, history, 100)
        report = session.finish(model)
        history["diagnostic_sample_state"] = report["state"]
        cursor = BlockCursor(telemetry=history)
        with tempfile.TemporaryDirectory() as tmp:
            receipt = save_block_checkpoint(model, optimizer, cursor, SOURCE, Path(tmp))
            fresh = BlockDrafter(tensors, cfg)
            other = block_optimizer(fresh)
            restored = load_block_checkpoint(fresh, other, SOURCE, receipt)
            reports = []
            for current, saved in ((model, history), (fresh, restored.telemetry)):
                for module in current.binary_linears().values():
                    with torch.no_grad():
                        module.latent_sign.mul_(-1)
                reports.append(BlockDiagnosticSession(current, saved, 200).finish(current))
            self.assertEqual(reports[0], reports[1])
            for layer in reports[0]["sample"].values():
                self.assertEqual(layer["sampled_endpoint_flip_backs"], layer["latent_samples"])
                self.assertEqual(layer["signs_equal_initial"], layer["latent_samples"])
            self.assertLess(len(json.dumps(reports[0])), 100000)
            self.assertLessEqual(fixed_indices(1000000, "cpu").numel(), SAMPLE_SIZE)

    def test_sampler_exception_cleanup_and_history_window_size(self):
        cfg, tensors, full, tail = fixture()
        model = BlockDrafter(tensors, cfg)
        session = BlockDiagnosticSession(model, None, 100)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with session.capture(model):
                raise RuntimeError("failure")
        self.assertTrue(
            all(getattr(m, "_diagnostic_a8", None) is None for m in model.binary_linears().values())
        )
        metrics, _ = block_train_batch(
            model, block_optimizer(model), [full, tail], diagnostics=True, update=False
        )
        report = session.finish(model)
        history = None
        for step in range(1, 101):
            history = update_history(
                history,
                metrics,
                step=step,
                elapsed=step * 300.0,
                chains=["a", "b"],
                groups=["g1", "g2"],
                diagnostics=report,
            )
        self.assertEqual(len(history["layer_samples"]), 32)
        self.assertEqual(len(history["recent"]), 60)
        self.assertLess(len(json.dumps(history)), 500000)


if __name__ == "__main__":
    unittest.main()
