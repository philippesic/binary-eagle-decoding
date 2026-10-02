"""Bounded acceptance checks, no production-shape allocations."""

import unittest
from unittest.mock import patch

import torch

from research.parallel20261002.step_bookkeeping.reference.audit import (
    comparison,
    configured_bytes,
    fixture,
)
from research.parallel20261002.step_bookkeeping.reference.snapshot_step import (
    joint_train_step_without_snapshot_clone,
)
from scripts.train_joint_w1ax import tiny_rollout
from w1a1_eagle.recurrent_qat import joint_optimizer, joint_train_step


class SnapshotStepTests(unittest.TestCase):
    def test_fixed_actual_step_exact(self):
        comparison()

    def test_learned_affine_actual_step_exact(self):
        comparison(True)

    def test_nan_gradient_fails_before_step(self):
        comparison(nan_gradient=True)

    def test_ownership_error_precedes_objective_and_zero_grad(self):
        for step in (joint_train_step, joint_train_step_without_snapshot_clone):
            linears, trace, config = fixture()
            optimizer = joint_optimizer(linears, config)
            optimizer.param_groups[0]["params"].pop()
            logits, _, _ = tiny_rollout(linears, "cpu")
            with patch.object(optimizer, "zero_grad") as zero_grad, patch.object(
                optimizer, "step"
            ) as update:
                with self.assertRaisesRegex(ValueError, "optimizer must own exactly"):
                    step(linears, logits, trace, optimizer, config, teacher={})
                zero_grad.assert_not_called()
                update.assert_not_called()

    def test_objective_error_after_zero_grad_before_update(self):
        for step in (joint_train_step, joint_train_step_without_snapshot_clone):
            linears, trace, config = fixture()
            optimizer = joint_optimizer(linears, config)
            logits, _, _ = tiny_rollout(linears, "cpu")
            with patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero_grad:
                with patch.object(optimizer, "step") as update:
                    with self.assertRaisesRegex(ValueError, "hard CE does not take"):
                        step(linears, logits, trace, optimizer, config, teacher={})
                    zero_grad.assert_called_once_with(set_to_none=True)
                    update.assert_not_called()

    def test_bool_snapshot_survives_master_mutation(self):
        latent = torch.tensor([-1.0, -0.0, 0.0, 1.0, float("nan")])
        before = latent.detach() < 0
        self.assertNotEqual(
            before.untyped_storage().data_ptr(), latent.untyped_storage().data_ptr()
        )
        latent.fill_(-1)
        torch.testing.assert_close(before, torch.tensor([True, False, False, False, False]))

    def test_full_shape_exact_arithmetic(self):
        result = configured_bytes()
        self.assertEqual(result["weights_per_lane"], 218234880)
        self.assertEqual(result["f32_copy_payload_bytes_per_lane_step"], 872939520)
        self.assertEqual(result["nominal_read_plus_write_bytes_per_lane_step"], 1745879040)
        self.assertEqual(result["largest_f32_clone_bytes"], 327680000)
        self.assertEqual(result["retained_bool_snapshot_bytes_per_lane_step"], 218234880)


if __name__ == "__main__":
    unittest.main()
