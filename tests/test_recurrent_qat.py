"""CPU checks for the row W1Ax contract and joint training boundary."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    compact_probability_loss,
    hard_activation,
    joint_optimizer,
    joint_train_step,
    save_joint_checkpoint,
)
from w1a1_eagle.recurrent_trace import TraceAudit


def audit() -> TraceAudit:
    return TraceAudit(
        draft_labels=(-1, -1, 1),
        valid_mask=(True, True, True),
        supported_mask=(False, False, True),
        ce_mask=(False, False, True),
        denominator_mask=(True, True, True),
        target_to_draft=(0, 1, 2, 3),
        counts={},
        per_depth={},
    )


class RecurrentQATTests(unittest.TestCase):
    def test_activation_contracts_and_zero_gradient(self):
        x = torch.tensor([[0.0, -0.0, 0.0], [1.0, -0.5, 0.0]], requires_grad=True)
        values, scales, _ = hard_activation(x, 1)
        torch.testing.assert_close(values[0], torch.zeros(3))
        torch.testing.assert_close(scales[1], torch.tensor([0.5]))
        torch.testing.assert_close(values[1], torch.tensor([0.5, -0.5, 0.5]))
        values.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        for bits, qmax in ((4, 7), (8, 127)):
            x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, -1.0]], requires_grad=True)
            values, scales, saturated = hard_activation(x, bits)
            torch.testing.assert_close(values[0], torch.zeros(3))
            self.assertAlmostEqual(float(scales[1]), 1 / qmax, places=7)
            self.assertTrue(bool(saturated[1, 0]))
            values.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
        a16, _, _ = hard_activation(torch.tensor([1.0001, -0.0]), 16)
        self.assertEqual(a16[0].item(), 1.0)

    def test_contract_guards(self):
        with self.assertRaisesRegex(ValueError, "A16 only"):
            W1AxContract(4, "group128")
        with self.assertRaisesRegex(ValueError, "explicit allow_accelerator"):
            JointQATConfig(W1AxContract(4), device="cuda")

    def test_row_129_tail_and_hard_quantized_forward(self):
        x = torch.zeros(129)
        x[0], x[127], x[128] = 1.0, -0.5, 0.25
        weight = torch.ones((2, 129))
        weight[1, 128] = -1
        scales = torch.tensor([0.5, 0.25])
        for bits in (1, 4, 8, 16):
            module = RowBinaryLinear(weight, scales, W1AxContract(bits))
            quantized, _, _ = hard_activation(x, bits)
            expected = torch.mv(torch.where(weight < 0, -1.0, 1.0), quantized) * scales
            torch.testing.assert_close(module(x), expected)

    def test_last_loss_reaches_earlier_state_and_kv(self):
        from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout

        for bits in (1, 4, 8, 16):
            config = JointQATConfig(W1AxContract(bits), sign_lr=0.01, scale_lr=0.01)
            linears, trace = tiny_joint_fixture(config)
            optimizer = joint_optimizer(linears, config)
            logits, cache, states = tiny_rollout(linears, "cpu")
            first_key, first_value = cache[0]
            metrics = joint_train_step(linears, logits, trace, optimizer, config)
            self.assertGreater(metrics["loss"], 0)
            self.assertGreater(float(first_key.grad.abs().sum()), 0)
            self.assertGreater(float(first_value.grad.abs().sum()), 0)
            self.assertGreater(float(states[0].grad.abs().sum()), 0)
            self.assertGreater(metrics["gradient_tensors"], 0)
            self.assertEqual(metrics["latent_outside_clip"], 0)

    def test_sign_flip_and_row_checkpoint_is_not_group_export(self):
        from scripts.train_joint_w1ax import tiny_joint_fixture

        config = JointQATConfig(W1AxContract(16), sign_lr=1.0, scale_lr=0.01)
        linears, trace = tiny_joint_fixture(config)
        for module in linears.values():
            module.latent_sign.data.fill_(0.001)
        optimizer = joint_optimizer(linears, config)
        # One controlled current-student graph with a requested opposite sign.
        fc = linears["fc"]
        output = fc(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        logits = torch.stack((output, output, output))
        metrics = joint_train_step(linears, logits, trace, optimizer, config)
        self.assertGreater(metrics["sign_flips"], 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest = Path(directory) / "joint.npz", Path(directory) / "joint.json"
            saved = save_joint_checkpoint(linears, config, "a" * 64, checkpoint, manifest)
            self.assertEqual(len(saved["checkpoint_sha256"]), 64)
            contents = json.loads(manifest.read_text())
            self.assertEqual(contents["scale_layout"], "row")
            self.assertEqual(contents["schema_version"], 2)
            self.assertEqual(contents["export_status"], "row_w1ax_requires_native_validation")

    def test_compact_teacher_conditional_mass(self):
        logits = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 3, requires_grad=True)
        teacher = {
            "draft_topk_ids": torch.tensor([[1, 0]] * 3, dtype=torch.int32),
            "draft_topk_probs": torch.tensor([[0.5, 0.2]] * 3),
            "draft_tail_mass": torch.tensor([0.2] * 3),
            "outside_draft_mass": torch.tensor([0.1] * 3),
        }
        loss = compact_probability_loss(logits, audit(), teacher)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertGreater(float(logits.grad[2].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        teacher["outside_draft_mass"] = torch.tensor([0.2] * 3)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            compact_probability_loss(logits, audit(), teacher)


if __name__ == "__main__":
    unittest.main()
