"""Independent CPU checks for joint-step scalar reporting compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

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
                "ratio": torch.tensor(0.125, dtype=torch.float32),
                "native_int": 7,
                "native_float": 0.25,
            }
        )
        self.assertEqual(list(result), ["wide_count", "ratio", "native_int", "native_float"])
        self.assertEqual(result["wide_count"], exact_count)
        self.assertIs(type(result["wide_count"]), int)
        self.assertEqual(result["ratio"], 0.125)
        self.assertIs(type(result["ratio"]), float)
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


if __name__ == "__main__":
    unittest.main()
