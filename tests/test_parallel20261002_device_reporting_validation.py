"""Independent CPU validation of the device-aware reporting fast path."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from unittest.mock import Mock, patch

import torch
from torch import Tensor

from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout
from w1a1_eagle import recurrent_qat
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract, joint_optimizer, joint_train_step


def _pinned_grouped_reporting(
    metrics: Mapping[str, Tensor | float | int],
) -> dict[str, float | int]:
    """Independent copy of the grouped conversion pinned at 93a2ff1."""
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[str, Tensor]]] = {}
    values: dict[str, float | int] = {}
    for name, value in metrics.items():
        if isinstance(value, Tensor):
            groups.setdefault((value.device, value.dtype), []).append((name, value.detach()))
        else:
            values[name] = value
    for group in groups.values():
        scalars = torch.stack([value for _, value in group]).cpu().tolist()
        values.update((name, scalar) for (name, _), scalar in zip(group, scalars))
    return {name: values[name] for name in metrics}


def _run_steps(*, decay: float, reference_reporter: bool):
    config = JointQATConfig(
        W1AxContract(16),
        seed=8675309,
        sign_lr=0.01,
        scale_lr=0.01,
        depth_loss_decay=decay,
    )
    linears, trace = tiny_joint_fixture(config)
    optimizer = joint_optimizer(linears, config)
    metrics = []
    patcher = (
        patch.object(recurrent_qat, "_reporting_scalars", side_effect=_pinned_grouped_reporting)
        if reference_reporter
        else patch.object(
            recurrent_qat, "_reporting_scalars", wraps=recurrent_qat._reporting_scalars
        )
    )
    with patcher:
        for _ in range(3):
            logits, _, _ = tiny_rollout(linears, "cpu")
            metrics.append(joint_train_step(linears, logits, trace, optimizer, config))
    return linears, optimizer, metrics


def _optimizer_snapshot(optimizer):
    return [
        {
            key: value.detach().clone() if isinstance(value, Tensor) else value
            for key, value in optimizer.state[param].items()
        }
        for group in optimizer.param_groups
        for param in group["params"]
    ]


def _assert_models_and_optimizer_equal(testcase, left, right):
    linears_a, optimizer_a, metrics_a = left
    linears_b, optimizer_b, metrics_b = right
    testcase.assertEqual(metrics_a, metrics_b)
    for name in linears_a:
        testcase.assertEqual(
            linears_a[name].state_dict().keys(), linears_b[name].state_dict().keys()
        )
        for key, value in linears_a[name].state_dict().items():
            torch.testing.assert_close(value, linears_b[name].state_dict()[key], rtol=0, atol=0)
    states_a, states_b = _optimizer_snapshot(optimizer_a), _optimizer_snapshot(optimizer_b)
    testcase.assertEqual(len(states_a), len(states_b))
    for state_a, state_b in zip(states_a, states_b):
        testcase.assertEqual(state_a.keys(), state_b.keys())
        for key, value_a in state_a.items():
            value_b = state_b[key]
            if isinstance(value_a, Tensor):
                torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)
            else:
                testcase.assertEqual(value_a, value_b)


class DeviceAwareReportingValidation(unittest.TestCase):
    def test_cpu_scalar_values_types_and_order_match_independent_grouped_reference(self):
        exact_int64 = 2**53 + 137
        metrics = {
            "bool": torch.tensor(True, dtype=torch.bool),
            "int64": torch.tensor(exact_int64, dtype=torch.int64),
            "float16": torch.tensor(0.5, dtype=torch.float16),
            "bfloat16": torch.tensor(1.25, dtype=torch.bfloat16),
            "float32": torch.tensor(2.5, dtype=torch.float32),
            "float64": torch.tensor(3.75 + 2**-40, dtype=torch.float64),
            "native_int": 7,
            "native_float": 0.25,
        }
        actual = recurrent_qat._reporting_scalars(metrics)
        expected = _pinned_grouped_reporting(metrics)
        self.assertEqual(list(actual.items()), list(expected.items()))
        self.assertIs(type(actual["bool"]), bool)
        self.assertIs(type(actual["int64"]), int)
        self.assertEqual(actual["int64"], exact_int64)
        for key in ("float16", "bfloat16", "float32", "float64"):
            self.assertIs(type(actual[key]), float)
            self.assertEqual(actual[key], expected[key])
        self.assertIs(type(actual["native_int"]), int)
        self.assertIs(type(actual["native_float"]), float)

    def test_actual_trainer_and_depth_curriculum_match_pinned_reporting_path(self):
        for decay in (1.0, 0.8):
            with self.subTest(depth_loss_decay=decay):
                fast = _run_steps(decay=decay, reference_reporter=False)
                pinned = _run_steps(decay=decay, reference_reporter=True)
                _assert_models_and_optimizer_equal(self, fast, pinned)
                self.assertEqual(len(fast[2]), 3)
                for record in fast[2]:
                    self.assertEqual(
                        set(record),
                        {
                            "loss",
                            "token_loss",
                            "midpoint_regularization",
                            "gradient_tensors",
                            "gradient_norm",
                            "sign_flips",
                            "scale_l1_movement",
                            "latent_outside_clip",
                            "saturation_mean",
                        },
                    )
                    for key in ("gradient_tensors", "sign_flips", "latent_outside_clip"):
                        self.assertIs(type(record[key]), int)
                    for key in set(record) - {
                        "gradient_tensors",
                        "sign_flips",
                        "latent_outside_clip",
                    }:
                        self.assertIs(type(record[key]), float)

    def test_optimizer_ownership_precedes_objective_validation_and_mutation(self):
        config = JointQATConfig(W1AxContract(16), seed=719)
        linears, trace = tiny_joint_fixture(config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        unrelated = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([unrelated], lr=0.1)
        zero_grad = Mock(wraps=optimizer.zero_grad)
        optimizer.zero_grad = zero_grad
        with self.assertRaisesRegex(ValueError, "optimizer must own exactly"):
            joint_train_step(
                linears, logits, trace, optimizer, config, teacher={"bad": torch.ones(1)}
            )
        zero_grad.assert_not_called()
        self.assertEqual(optimizer.state, {})

        optimizer = joint_optimizer(linears, config)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in params]
        step = Mock(wraps=optimizer.step)
        optimizer.step = step
        with self.assertRaisesRegex(ValueError, "hard CE does not take compact teacher"):
            joint_train_step(
                linears, logits, trace, optimizer, config, teacher={"bad": torch.ones(1)}
            )
        step.assert_not_called()
        self.assertEqual(optimizer.state, {})
        for param, expected in zip(params, before):
            torch.testing.assert_close(param, expected, rtol=0, atol=0)

    def test_nonfinite_gradient_rejects_before_step_and_postmutation_check_precedes_reporting(self):
        config = JointQATConfig(W1AxContract(16), seed=887)
        linears, trace = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        params = [p for group in optimizer.param_groups for p in group["params"]]
        hook = params[0].register_hook(lambda grad: torch.full_like(grad, float("nan")))
        step = Mock(wraps=optimizer.step)
        optimizer.step = step
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite joint QAT gradient"):
                joint_train_step(linears, logits, trace, optimizer, config)
        finally:
            hook.remove()
        step.assert_not_called()
        self.assertEqual(optimizer.state, {})

        linears, trace = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        real_step = optimizer.step

        def step_then_poison(*args, **kwargs):
            result = real_step(*args, **kwargs)
            with torch.no_grad():
                linears["fc"].latent_sign[0, 0] = float("nan")
            return result

        optimizer.step = Mock(side_effect=step_then_poison)
        with patch.object(
            recurrent_qat, "_reporting_scalars", wraps=recurrent_qat._reporting_scalars
        ) as report:
            with self.assertRaisesRegex(
                ValueError, "joint QAT update produced nonfinite parameters"
            ):
                joint_train_step(linears, logits, trace, optimizer, config)
        report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
