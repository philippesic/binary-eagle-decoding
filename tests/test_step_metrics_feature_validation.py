"""Independent CPU checks for joint-step scalar reporting compatibility."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    _reporting_scalars,
    joint_optimizer,
    joint_train_step,
)

EXPECTED_KEYS = {
    "loss",
    "token_loss",
    "midpoint_regularization",
    "gradient_tensors",
    "gradient_norm",
    "sign_flips",
    "scale_l1_movement",
    "latent_outside_clip",
    "saturation_mean",
}


def run_seeded_steps():
    config = JointQATConfig(
        W1AxContract(16), seed=8675309, sign_lr=0.01, scale_lr=0.01
    )
    linears, trace = tiny_joint_fixture(config)
    optimizer = joint_optimizer(linears, config)
    records = []
    for _ in range(2):
        logits, _, _ = tiny_rollout(linears, "cpu")
        records.append(joint_train_step(linears, logits, trace, optimizer, config))
    return linears, optimizer, records


class StepMetricsFeatureValidation(unittest.TestCase):
    def test_reporting_scalars_preserves_int64_exactness_and_schema_order(self):
        exact_count = 2**53 + 137
        result = _reporting_scalars(
            {
                "wide_count": torch.tensor(exact_count, dtype=torch.int64),
                "float16": torch.tensor(0.5, dtype=torch.float16),
                "bfloat16": torch.tensor(1.25, dtype=torch.bfloat16),
                "float32": torch.tensor(2.5, dtype=torch.float32),
                "float64": torch.tensor(3.75 + 2**-40, dtype=torch.float64),
                "native_int": 7,
                "native_float": 0.25,
            }
        )
        self.assertEqual(
            list(result),
            [
                "wide_count",
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "native_int",
                "native_float",
            ],
        )
        self.assertEqual(result["wide_count"], exact_count)
        self.assertIs(type(result["wide_count"]), int)
        for key, expected in (
            ("float16", 0.5),
            ("bfloat16", 1.25),
            ("float32", 2.5),
            ("float64", 3.75 + 2**-40),
        ):
            self.assertEqual(result[key], expected)
            self.assertIs(type(result[key]), float)
        self.assertEqual(result["native_int"], 7)
        self.assertEqual(result["native_float"], 0.25)

    def test_seeded_optimizer_moments_and_checkpoint_metrics_are_stable(self):
        linears_a, optimizer_a, records_a = run_seeded_steps()
        linears_b, optimizer_b, records_b = run_seeded_steps()
        for record in records_a + records_b:
            self.assertEqual(set(record), EXPECTED_KEYS)
        self.assertEqual(records_a, records_b)
        for key in ("gradient_tensors", "sign_flips", "latent_outside_clip"):
            for record in records_a:
                self.assertIs(type(record[key]), int)
        for key in EXPECTED_KEYS - {"gradient_tensors", "sign_flips", "latent_outside_clip"}:
            for record in records_a:
                self.assertIs(type(record[key]), float)

        # Continuous-run checkpoints and metrics.jsonl use JSON-compatible
        # values from this same step result. Verify serialization keeps counts
        # as integers after seeded AdamW moments have been populated.
        checkpoint_metrics = {"A16": records_a[-1]}
        restored_metrics = json.loads(json.dumps(checkpoint_metrics))
        self.assertEqual(restored_metrics, checkpoint_metrics)
        self.assertIs(type(restored_metrics["A16"]["sign_flips"]), int)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "metrics.pt"
            torch.save({"metrics": checkpoint_metrics}, checkpoint_path)
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint_payload["metrics"], checkpoint_metrics)
        for name in linears_a:
            for key, value in linears_a[name].state_dict().items():
                torch.testing.assert_close(value, linears_b[name].state_dict()[key], rtol=0, atol=0)
        for param_a, param_b in zip(
            [p for group in optimizer_a.param_groups for p in group["params"]],
            [p for group in optimizer_b.param_groups for p in group["params"]],
        ):
            state_a, state_b = optimizer_a.state[param_a], optimizer_b.state[param_b]
            self.assertEqual(state_a.keys(), state_b.keys())
            for key in state_a:
                if isinstance(state_a[key], torch.Tensor):
                    torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=0)
                else:
                    self.assertEqual(state_a[key], state_b[key])

    def test_rejected_step_keeps_precedence_and_does_not_update_parameters_or_moments(self):
        config = JointQATConfig(W1AxContract(16), seed=919, sign_lr=0.01, scale_lr=0.01)
        linears, trace = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        joint_train_step(linears, logits, trace, optimizer, config)

        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        parameter_snapshot = [p.detach().clone() for p in parameters]
        state_snapshot = {
            p: {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.state[p].items()
            }
            for p in parameters
        }
        logits, _, _ = tiny_rollout(linears, "cpu")
        with self.assertRaisesRegex(ValueError, "hard CE does not take compact teacher"):
            joint_train_step(
                linears,
                logits,
                trace,
                optimizer,
                config,
                teacher={"malformed": torch.ones(1)},
            )
        for param, snapshot in zip(parameters, parameter_snapshot):
            torch.testing.assert_close(param, snapshot, rtol=0, atol=0)
            for key, value in optimizer.state[param].items():
                expected = state_snapshot[param][key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    self.assertEqual(value, expected)

    def test_nonfinite_gradient_rejects_before_optimizer_step(self):
        config = JointQATConfig(W1AxContract(16), seed=113, sign_lr=0.01, scale_lr=0.01)
        linears, trace = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        joint_train_step(linears, logits, trace, optimizer, config)

        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        parameter_snapshot = [p.detach().clone() for p in parameters]
        state_snapshot = {
            p: {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.state[p].items()
            }
            for p in parameters
        }
        hook = parameters[0].register_hook(lambda grad: torch.full_like(grad, float("nan")))
        logits, _, _ = tiny_rollout(linears, "cpu")
        step = Mock(wraps=optimizer.step)
        optimizer.step = step
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite joint QAT gradient"):
                joint_train_step(linears, logits, trace, optimizer, config)
        finally:
            hook.remove()
        step.assert_not_called()
        for param, snapshot in zip(parameters, parameter_snapshot):
            torch.testing.assert_close(param, snapshot, rtol=0, atol=0)
            for key, value in optimizer.state[param].items():
                expected = state_snapshot[param][key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    self.assertEqual(value, expected)

    def test_optimizer_ownership_error_precedes_invalid_teacher_and_zero_grad(self):
        config = JointQATConfig(W1AxContract(16), seed=127)
        linears, trace = tiny_joint_fixture(config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        unrelated = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([unrelated], lr=0.1)
        zero_grad = Mock(wraps=optimizer.zero_grad)
        optimizer.zero_grad = zero_grad
        with self.assertRaisesRegex(
            ValueError,
            "optimizer must own exactly declared binary/activation/fusion parameters",
        ):
            joint_train_step(
                linears,
                logits,
                trace,
                optimizer,
                config,
                teacher={"invalid": torch.ones(1)},
            )
        zero_grad.assert_not_called()

    def test_nonfinite_updated_parameters_reject_before_reporting_scalar_extraction(self):
        config = JointQATConfig(W1AxContract(16), seed=139)
        linears, trace = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        logits, _, _ = tiny_rollout(linears, "cpu")
        real_step = optimizer.step

        def step_then_poison_parameter(*args, **kwargs):
            result = real_step(*args, **kwargs)
            with torch.no_grad():
                linears["fc"].latent_sign[0, 0] = float("nan")
            return result

        step = Mock(side_effect=step_then_poison_parameter)
        optimizer.step = step
        with patch(
            "w1a1_eagle.recurrent_qat._reporting_scalars", wraps=_reporting_scalars
        ) as reporting:
            with self.assertRaisesRegex(
                ValueError, "joint QAT update produced nonfinite parameters"
            ):
                joint_train_step(linears, logits, trace, optimizer, config)
        step.assert_called_once()
        reporting.assert_not_called()
        self.assertTrue(torch.isnan(linears["fc"].latent_sign[0, 0]))


if __name__ == "__main__":
    unittest.main()
