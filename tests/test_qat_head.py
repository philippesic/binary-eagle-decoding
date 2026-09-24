"""CPU checks for the bounded head-only W1A1 training arithmetic."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.fake_binary import W1A1Config, fake_binary_linear  # noqa: E402
from w1a1_eagle.qat_head import TrainableW1A1Head, _WeightSignSTE  # noqa: E402


class TrainableW1A1HeadTests(unittest.TestCase):
    def test_forward_is_bitwise_equal_for_random_bf16_inputs(self) -> None:
        torch.manual_seed(17)
        weight = torch.randn(7, 11, dtype=torch.bfloat16)
        inputs = torch.randn(2, 3, 11, dtype=torch.bfloat16)
        for config in (W1A1Config(), W1A1Config(-1, "rms", "unit")):
            with self.subTest(config=config):
                head = TrainableW1A1Head(weight, config)
                actual = head(inputs)
                reference = fake_binary_linear(inputs, weight, config=config)
                self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(head.latent_weight.dtype, torch.float32)

    def test_forward_is_bitwise_equal_with_zeros_and_near_zeros(self) -> None:
        weight = torch.tensor(
            [[0.0, -0.0, 1e-40, -1e-40], [1e-5, -1e-5, 0.0, -0.0]],
            dtype=torch.bfloat16,
        )
        inputs = torch.tensor(
            [[0.0, -0.0, 1e-40, -1e-40], [-1e-5, 1e-5, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        for zero_sign in (-1, 1):
            config = W1A1Config(zero_sign=zero_sign)
            head = TrainableW1A1Head(weight, config)
            self.assertTrue(
                torch.equal(head(inputs), fake_binary_linear(inputs, weight, config=config))
            )
            self.assertTrue(torch.equal(head.export_bf16_weight(), weight))

    def test_clipped_scale_normalized_ste_gradient(self) -> None:
        latent = torch.tensor([[0.25, 0.5, 0.0, -0.25]], requires_grad=True)
        row_scale = torch.tensor([[0.25]], dtype=torch.bfloat16)
        _WeightSignSTE.apply(latent, 1, row_scale, 1.0, 1e-3).float().sum().backward()
        torch.testing.assert_close(latent.grad, torch.tensor([[4.0, 0.0, 4.0, 4.0]]))

    def test_finite_nonzero_gradient_and_optimizer_step(self) -> None:
        weight = torch.tensor(
            [[0.25, -0.5, 0.75, 0.0], [-0.25, 0.5, -0.75, 0.125]],
            dtype=torch.bfloat16,
        )
        inputs = torch.tensor(
            [[1.0, -0.5, 0.25, 0.0], [-0.25, 0.75, 0.5, -1.0]], dtype=torch.bfloat16
        )
        head = TrainableW1A1Head(weight)
        optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)
        original = head.latent_weight.detach().clone()
        head(inputs).float().square().sum().backward()
        gradient = head.latent_weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertLess(gradient.abs().max().item(), 1e5)
        optimizer.step()
        self.assertFalse(torch.equal(head.latent_weight.detach(), original))
        exported = head.export_bf16_weight()
        self.assertTrue(torch.equal(head(inputs), fake_binary_linear(inputs, exported)))
        self.assertEqual(exported.dtype, torch.bfloat16)
        self.assertFalse(exported.requires_grad)
        self.assertTrue(torch.equal(exported, head.latent_weight.detach().to(torch.bfloat16)))
        exported[0, 0] = 0
        self.assertNotEqual(head.export_bf16_weight()[0, 0].item(), 0)

    def test_rejects_non_bf16_or_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            TrainableW1A1Head(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            TrainableW1A1Head(torch.ones(3, dtype=torch.bfloat16))
        head = TrainableW1A1Head(torch.ones(2, 3, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            head(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            head(torch.ones(1, 2, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
