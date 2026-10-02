"""Independent CPU checks of sign capacity and optimizer crossing mechanics."""

from __future__ import annotations

import itertools
import math
import unittest

import torch

from w1a1_eagle.qat_optimization import (
    BinaryOptimizationConfig,
    SignFlipDiagnostics,
    make_binary_optimizer,
    project_binary_parameters_,
    transform_binary_gradients_,
)
from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract


class SignInertiaValidationTests(unittest.TestCase):
    def _linear(self, value: float, *, scale: float = 1.0) -> dict[str, RowBinaryLinear]:
        return {
            "fc": RowBinaryLinear(
                torch.tensor([[value]], dtype=torch.float32),
                torch.tensor([scale], dtype=torch.float32),
                W1AxContract(1),
            )
        }

    def _constant_gradient_steps(
        self,
        *,
        optimizer_name: str,
        momentum: float,
        initial: float,
        lr: float,
        steps: int,
    ) -> list[tuple[float, float]]:
        linears = self._linear(initial)
        config = BinaryOptimizationConfig(
            optimizer=optimizer_name,
            momentum=momentum,
            sign_lr=lr,
            scale_lr=lr,
            max_grad_norm=10.0,
        )
        optimizer = make_binary_optimizer(linears, config)
        trajectory = []
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            linears["fc"].latent_sign.grad = torch.ones_like(linears["fc"].latent_sign)
            transform_binary_gradients_(linears, config)
            optimizer.step()
            project_binary_parameters_(linears)
            latent = float(linears["fc"].latent_sign.item())
            hard = float(hard_sign_ste(linears["fc"].latent_sign).item())
            trajectory.append((latent, hard))
        return trajectory

    def test_exact_hard_sign_capacity_on_four_input_rows(self):
        x = torch.tensor(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            dtype=torch.int64,
        )
        desired = torch.tensor([1, -1, 1], dtype=torch.int64)
        labels = torch.sign(x @ desired)
        self.assertFalse(bool((x @ desired == 0).any()))
        matching = []
        strict = []
        for candidate in itertools.product((-1, 1), repeat=3):
            weight = torch.tensor(candidate, dtype=torch.int64)
            margins = x @ weight
            strict.append(candidate if bool((margins != 0).all()) else None)
            if bool((margins != 0).all()) and bool(torch.equal(torch.sign(margins), labels)):
                matching.append(candidate)
        self.assertEqual(matching, [(1, -1, 1)])
        self.assertEqual(sum(candidate is not None for candidate in strict), 8)

    def test_xor_is_infeasible_for_all_hard_sign_rows_without_bias(self):
        x = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.int64)
        labels = torch.tensor([1, -1, -1, 1], dtype=torch.int64)
        exact = []
        for candidate in itertools.product((-1, 1), repeat=2):
            margins = x @ torch.tensor(candidate, dtype=torch.int64)
            if bool((margins != 0).all()) and bool(torch.equal(torch.sign(margins), labels)):
                exact.append(candidate)
        self.assertEqual(exact, [])
        # Every positive row scale leaves each nonzero row's hard sign unchanged.
        for scale in (0.001, 0.5, 17.0):
            for candidate in itertools.product((-1, 1), repeat=2):
                margins = scale * (x.float() @ torch.tensor(candidate, dtype=torch.float32))
                if bool((margins != 0).all()):
                    self.assertFalse(bool(torch.equal(torch.sign(margins), labels.float())))

    def test_recurrent_sign_enumeration_matches_a1_module_decisions(self):
        # Independent scalar oracle for the fixture used by the reference audit.
        sequences = ((1, 1, -1), (1, -1, -1), (-1, 1, 1), (-1, -1, 1))

        def scalar(signs, scale, sequence):
            hidden = 0.25
            outputs = []
            for value in sequence:
                alpha = (abs(value) + abs(hidden)) / 2
                z = (
                    scale * alpha * (signs[0] * value + signs[1] * (1 if hidden >= 0 else -1))
                    + 0.15
                )
                outputs.append(z)
                hidden = math.tanh(z)
            return outputs

        # Build labels on the fixed scale-1 desired-sign recurrence.
        labels = torch.tensor(
            [[1 if value > 0 else -1 for value in scalar((1, -1), 1, seq)] for seq in sequences]
        )
        expected_feasible = {0.01: 0, 1.0: 1, 100.0: 1}
        for scale, feasible_count in expected_feasible.items():
            scalar_feasible = 0
            module_feasible = 0
            for signs in itertools.product((-1, 1), repeat=2):
                reference_rows = [scalar(signs, scale, seq) for seq in sequences]
                module = RowBinaryLinear(
                    torch.tensor([signs], dtype=torch.float32),
                    torch.tensor([scale], dtype=torch.float32),
                    W1AxContract(1),
                    bias=torch.tensor([0.15]),
                )
                module_rows = []
                with torch.no_grad():
                    for sequence in sequences:
                        hidden = torch.tensor([0.25])
                        outputs = []
                        for value in sequence:
                            output = module(torch.tensor([[float(value), float(hidden.item())]]))[0]
                            outputs.append(float(output))
                            hidden = torch.tanh(output)
                        module_rows.append(outputs)
                ref_correct = (torch.tensor(reference_rows) * labels > 0).all()
                module_correct = (torch.tensor(module_rows) * labels > 0).all()
                scalar_feasible += int(ref_correct)
                module_feasible += int(module_correct)
                # Compare the scalar recurrence with A1's mean-absolute scale
                # against the production layer outputs, not only their signs.
                torch.testing.assert_close(
                    torch.tensor(reference_rows),
                    torch.tensor(module_rows),
                    rtol=1e-6,
                    atol=1e-5,
                )
                self.assertTrue(
                    torch.equal(
                        torch.tensor(reference_rows).sign(), torch.tensor(module_rows).sign()
                    )
                )
            self.assertEqual(scalar_feasible, feasible_count)
            self.assertEqual(module_feasible, feasible_count)

    def test_latent_magnitude_changes_leave_hard_forward_unchanged(self):
        x = torch.tensor([[-1.0, 2.0], [0.25, 0.5]])
        sign_vectors = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        forwards = []
        for magnitude in (0.01, 0.1, 0.5, 1.0):
            module = RowBinaryLinear(
                sign_vectors * magnitude,
                torch.tensor([0.25, 0.75]),
                W1AxContract(1),
            )
            forwards.append(module(x).detach())
        for actual in forwards[1:]:
            torch.testing.assert_close(actual, forwards[0], rtol=0, atol=0)

    def test_zero_scale_weight_unit_rule_does_not_revive_sign(self):
        linears = self._linear(-0.25, scale=0.0)
        config = BinaryOptimizationConfig(
            optimizer="sgd",
            sign_gradient_rule="weight_unit",
            sign_lr=0.25,
            max_grad_norm=10.0,
        )
        optimizer = make_binary_optimizer(linears, config)
        before = linears["fc"].latent_sign.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        linears["fc"].latent_sign.grad = torch.ones_like(linears["fc"].latent_sign)
        transform_binary_gradients_(linears, config)
        self.assertEqual(float(linears["fc"].latent_sign.grad.item()), 0.0)
        optimizer.step()
        project_binary_parameters_(linears)
        torch.testing.assert_close(linears["fc"].latent_sign, before, rtol=0, atol=0)
        self.assertEqual(float(hard_sign_ste(linears["fc"].latent_sign).item()), -1.0)

    def test_constant_gradient_crossings_match_hand_derived_times(self):
        # SGD without momentum moves by 1/8 per update. AdamW follows the same
        # trajectory within float32/epsilon rounding. At zero, hard sign is +1.
        for optimizer_name in ("sgd", "adamw"):
            with self.subTest(optimizer=optimizer_name):
                trajectory = self._constant_gradient_steps(
                    optimizer_name=optimizer_name,
                    momentum=0.0,
                    initial=0.25,
                    lr=0.125,
                    steps=3,
                )
                self.assertEqual([hard for _, hard in trajectory], [1.0, 1.0, -1.0])
                if optimizer_name == "sgd":
                    self.assertEqual(trajectory[1][0], 0.0)
                else:
                    self.assertGreaterEqual(trajectory[1][0], 0.0)
                    self.assertLess(trajectory[1][0], 3e-8)
                self.assertLess(trajectory[2][0], 0.0)

        # Momentum=1/2 gives displacements 1/8, 3/16, 7/32; from 5/16
        # the second step lands at zero and the third crosses negative.
        trajectory = self._constant_gradient_steps(
            optimizer_name="sgd", momentum=0.5, initial=0.3125, lr=0.125, steps=3
        )
        self.assertEqual([hard for _, hard in trajectory], [1.0, 1.0, -1.0])
        self.assertEqual(trajectory[1][0], 0.0)
        self.assertLess(trajectory[2][0], 0.0)

    def test_every_other_observation_can_hide_alternating_sign_chatter(self):
        linears = self._linear(0.25)
        tracker = SignFlipDiagnostics(linears, contract={"fixture": "alternating"})
        observed = []
        for step, value in enumerate((0.25, -0.25, 0.25, -0.25, 0.25), start=0):
            if step:
                with torch.no_grad():
                    linears["fc"].latent_sign.fill_(value)
            if step in (2, 4):
                observed.append(tracker.observe(linears, step=step))
        self.assertEqual([entry["sign_flips"] for entry in observed], [0, 0])
        self.assertEqual([entry["observation_gap_steps"] for entry in observed], [2, 2])
        # Four actual changes occurred between the initial state and step 4.
        self.assertEqual(sum(a != b for a, b in zip((1, -1, 1, -1), (-1, 1, -1, 1))), 4)


if __name__ == "__main__":
    unittest.main()
