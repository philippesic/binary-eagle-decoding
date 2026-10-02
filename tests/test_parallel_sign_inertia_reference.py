"""Decision-level acceptance gates for the bounded synthetic inertia audit."""

import importlib.util
import unittest
from pathlib import Path

import torch

from w1a1_eagle.qat_optimization import (
    BinaryOptimizationConfig,
    make_binary_optimizer,
    project_binary_parameters_,
    transform_binary_gradients_,
)
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

SPEC = importlib.util.spec_from_file_location(
    "sign_inertia_audit",
    Path(__file__).parents[1] / "research/parallel20261002/sign_inertia/reference/audit.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SignInertiaReferenceTests(unittest.TestCase):
    def test_constant_gradient_with_production_100_step_warmup(self):
        for magnitude, expected_crossing in ((0.5, None), (0.02, 63)):
            module = RowBinaryLinear(torch.tensor([[magnitude]]), torch.ones(1), W1AxContract(1))
            modules = {"fixture": module}
            config = BinaryOptimizationConfig()
            optimizer = make_binary_optimizer(modules, config)
            first = None
            for step in range(1, 65):
                optimizer.zero_grad(set_to_none=True)
                module.latent_sign.grad = torch.ones_like(module.latent_sign)
                # Exact ContinuousConfig default schedule, without launching a trainer.
                optimizer.param_groups[0]["lr"] = config.sign_lr * min(1, step / 100)
                transform_binary_gradients_(modules, config)
                optimizer.step()
                project_binary_parameters_(modules)
                if module.latent_sign.item() < 0 and first is None:
                    first = step
            self.assertEqual(first, expected_crossing)

    def test_capacity_oracle_matches_all_recurrent_hard_logits(self):
        x, _ = AUDIT.fixture("recurrent")
        for scale in (0.01, 1, 100):
            capacity = AUDIT.enumerate_capacity("recurrent", scale)
            self.assertEqual(capacity["feasible_states"], 0 if scale == 0.01 else 1)
            for row in capacity["states"]:
                module = RowBinaryLinear(
                    torch.tensor([row["signs"]], dtype=torch.float32),
                    torch.tensor([scale]),
                    W1AxContract(1),
                    bias=torch.tensor([0.15]),
                )
                torch.testing.assert_close(
                    AUDIT.logits(module, "recurrent", x),
                    torch.tensor(row["logits"]),
                    rtol=1e-6,
                    atol=1e-5,
                )

    def test_same_reachable_capacity_but_budget_stalls_high_inertia(self):
        self.assertEqual(AUDIT.enumerate_capacity("linear", 1)["feasible_states"], 1)
        stalled = AUDIT.run_case("linear", 1, 0.5, "frozen", {})["summary"]
        moving = AUDIT.run_case("linear", 1, 0.02, "frozen", {})["summary"]
        self.assertIsNone(stalled["first_useful_flip"])
        self.assertEqual(stalled["final"]["correct"], 2)
        self.assertEqual(moving["first_useful_flip"], 20)
        self.assertEqual(moving["final"]["correct"], 4)
        self.assertGreater(moving["final"]["min_margin"], 0)
        self.assertEqual(moving["tail16_full_correct_updates"], 16)

    def test_recurrent_useful_first_flip_does_not_imply_persistence(self):
        result = AUDIT.run_case("recurrent", 1, 0.02, "frozen", {})["summary"]
        self.assertEqual(result["first_useful_flip"], 20)
        self.assertEqual(result["final"]["correct"], 12)
        self.assertEqual(result["final"]["cumulative_sign_flips"], 7)
        self.assertEqual(result["final"]["cumulative_flip_backs"], 3)
        self.assertEqual(result["wrong_flip_events"], 3)
        self.assertLess(result["tail16_full_correct_updates"], 16)

    def test_infeasible_control_can_flip_without_useful_learning(self):
        self.assertEqual(AUDIT.enumerate_capacity("xor", 1)["feasible_states"], 0)
        result = AUDIT.run_case("xor", 1, 0.02, "frozen", {})["summary"]
        self.assertIsNone(result["first_useful_flip"])
        self.assertGreater(result["final"]["cumulative_sign_flips"], 0)
        self.assertGreater(result["final"]["cumulative_flip_backs"], 0)
        self.assertLessEqual(result["final"]["correct"], 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
