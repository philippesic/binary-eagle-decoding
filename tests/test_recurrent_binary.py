"""CPU-only numerical and gradient checks for recurrent W1A16 binary linears."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.recurrent_binary import (  # noqa: E402
    CANDIDATE_D_BASE_TO_PATH,
    GroupedBinaryLinear,
    install_candidate_d_linears,
    pack_signs,
    replay_packed_linear,
    unpack_signs,
)


class RecurrentBinaryTests(unittest.TestCase):
    @staticmethod
    def candidate_fixture():
        drafter = nn.Module()
        drafter.config = SimpleNamespace(pretraining_tp=1)
        drafter.fc = nn.Linear(2, 3, bias=True, device="cpu")
        drafter.midlayer = nn.Module()
        drafter.midlayer.self_attn = nn.Module()
        drafter.midlayer.mlp = nn.Module()
        for name, rows in (("q_proj", 64), ("k_proj", 16), ("v_proj", 3), ("o_proj", 3)):
            setattr(drafter.midlayer.self_attn, name, nn.Linear(2, rows, device="cpu"))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(drafter.midlayer.mlp, name, nn.Linear(2, 3, device="cpu"))
        drafter.lm_head = nn.Linear(2, 3, device="cpu")
        candidate, originals = {}, {}
        for base, path in CANDIDATE_D_BASE_TO_PATH.items():
            linear = drafter.get_submodule(path)
            originals[path] = linear
            rows = linear.out_features
            signs = np.ones((rows, 2), dtype=np.float32)
            signs[np.arange(rows) % 2 == 1, 0] = -1
            signs[np.arange(rows) % 3 == 0, 1] = -1
            scales = (np.arange(rows, dtype=np.float32)[:, None] + 1) / 8
            if base in ("blk.0.attn_q", "blk.0.attn_k"):
                heads = 32 if base.endswith("_q") else 8
                signs = (
                    signs.reshape(heads, 2, rows // heads // 2, 2).swapaxes(1, 2).reshape(rows, 2)
                )
                scales = (
                    scales.reshape(heads, 2, rows // heads // 2, 1).swapaxes(1, 2).reshape(rows, 1)
                )
            candidate[base] = (pack_signs(signs), scales)
        return drafter, candidate, originals

    def test_hard_forward_zero_sign_and_group_boundary(self):
        weight = torch.ones((2, 129), dtype=torch.float32, device="cpu")
        weight[0, 0] = 0
        weight[0, 127] = -0.0
        weight[0, 128] = -0.2
        weight[1, :128] = -0.3
        scales = torch.tensor([[0.5, 3.0], [2.0, 0.25]], device="cpu")
        layer = GroupedBinaryLinear(weight, scales)
        x = torch.ones((129,), dtype=torch.float32, device="cpu")
        x[0] = 0.1  # explicit F32 -> F16 -> F32 boundary
        y = layer(x)
        postcast = float(torch.tensor(0.1, device="cpu").half().float())
        expected_0 = (127.0 + postcast) * float(layer.effective_scales()[0, 0].detach()) - 3.0
        expected_1 = (-127.0 - postcast) * float(layer.effective_scales()[1, 0].detach()) + 0.25
        torch.testing.assert_close(y, torch.tensor([expected_0, expected_1]), rtol=1e-6, atol=1e-6)
        packed, exported_scales = layer.export_arrays()
        decoded = unpack_signs(packed, 129)
        self.assertEqual(decoded[0, 0], 1)
        self.assertEqual(decoded[0, 127], 1)
        self.assertEqual(decoded[0, 128], -1)
        self.assertEqual(packed.shape, (2, 5))
        self.assertEqual(exported_scales.shape, (2, 2))

    def test_sign_ste_and_nonnegative_scale_gradients(self):
        weight = torch.tensor([[0.0, -0.5, 2.0]], dtype=torch.float32, device="cpu")
        layer = GroupedBinaryLinear(weight, torch.tensor([[2.0, 3.0]], device="cpu"), group_size=2)
        x = torch.tensor([2.0, 1.0, 4.0], device="cpu", requires_grad=True)
        y = layer(x)
        self.assertEqual(float(y.detach()), 14.0)  # (2 - 1) * 2 + 4 * 3
        y.backward()
        # Clipped-identity sign STE: the third latent weight is outside [-1, 1].
        torch.testing.assert_close(layer.latent_sign.grad, torch.tensor([[4.0, 2.0, 0.0]]))
        torch.testing.assert_close(x.grad, torch.tensor([2.0, -2.0, 3.0]))
        # Scale offset has the exact group-sum gradient above zero.
        torch.testing.assert_close(layer.scale_offset.grad, torch.tensor([[1.0, 4.0]]))
        self.assertTrue(bool((layer.effective_scales() > 0).all()))
        with torch.no_grad():
            layer.latent_sign[0, 0] = 0.8
        self.assertEqual(float(layer(x).detach()), 14.0)  # magnitudes never enter forward

    def test_two_step_unroll_reaches_earlier_binary_body(self):
        body = GroupedBinaryLinear(
            torch.tensor([[0.2, -0.3]], device="cpu"),
            torch.tensor([[1.0]], device="cpu"),
            group_size=2,
        )
        head = GroupedBinaryLinear(
            torch.tensor([[0.4]], device="cpu"),
            torch.tensor([[2.0]], device="cpu"),
            group_size=1,
        )
        x = torch.tensor([0.5, 0.25], device="cpu")
        first = body(x)
        first.retain_grad()
        later_input = torch.stack((first.squeeze(), x[1]))
        later_loss = head(body(later_input)).sum()
        later_loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertGreater(abs(float(first.grad[0])), 0)
        self.assertIsNotNone(body.latent_sign.grad)
        self.assertGreater(abs(float(body.scale_offset.grad[0, 0])), 0)

    def test_packed_roundtrip_and_scalar_replay(self):
        rng = np.random.default_rng(14)
        signs = rng.choice([-1.0, 1.0], size=(3, 131)).astype(np.float32)
        packed = pack_signs(signs)
        self.assertEqual(packed.dtype, np.dtype("<i4"))
        np.testing.assert_array_equal(unpack_signs(packed, 131), signs)
        scales = np.array([[0.25, 1.5], [2.0, 0.75], [0.625, 3.0]], dtype=np.float32)
        layer = GroupedBinaryLinear(
            torch.tensor(signs, dtype=torch.float32, device="cpu"),
            torch.tensor(scales, dtype=torch.float32, device="cpu"),
        )
        inputs = rng.normal(size=(2, 3, 131)).astype(np.float32)
        inputs[0, 0, 0] = np.float32(0.10001)
        output = layer(torch.tensor(inputs, device="cpu")).detach().numpy()
        exported_packed, exported_scales = layer.export_arrays()
        np.testing.assert_array_equal(exported_scales, scales)
        restored = GroupedBinaryLinear.from_packed(
            exported_packed, exported_scales, in_features=131
        )
        np.testing.assert_array_equal(restored.export_arrays()[0], exported_packed)
        np.testing.assert_array_equal(restored.export_arrays()[1], exported_scales)
        np.testing.assert_array_equal(
            restored(torch.tensor(inputs, device="cpu")).detach().numpy(), output
        )
        replay = replay_packed_linear(inputs, exported_packed, exported_scales, 131)
        np.testing.assert_array_equal(output, replay)
        np.testing.assert_array_equal(exported_packed, packed)
        with self.assertRaisesRegex(ValueError, "tail bits"):
            invalid = packed.copy()
            invalid[0, -1] |= np.int32(-2147483648)
            unpack_signs(invalid, 131)

    def test_exact_zero_scale_can_train_and_export(self):
        weight = torch.ones((1, 2), device="cpu")
        layer = GroupedBinaryLinear(weight, torch.zeros((1, 1), device="cpu"))
        x = torch.ones(2, device="cpu")
        self.assertEqual(float(layer(x).detach()), 0)
        (-layer(x).sum()).backward()
        self.assertEqual(float(layer.scale_offset.grad[0, 0]), -2)
        with torch.no_grad():
            layer.scale_offset -= 0.1 * layer.scale_offset.grad
        self.assertAlmostEqual(float(layer(x).detach()), 0.4, places=6)
        packed, scales = layer.export_arrays()
        np.testing.assert_array_equal(scales, np.array([[0.2]], dtype=np.float32))
        self.assertEqual(
            float(replay_packed_linear(np.ones(2, dtype=np.float32), packed, scales, 2).item()),
            float(layer(x).detach()),
        )
        with torch.no_grad():
            layer.scale_offset.fill_(-1)
        layer.project_scales_()
        self.assertEqual(float(layer.effective_scales()[0, 0].detach()), 0)

    def test_rejects_negative_scale(self):
        weight = torch.ones((1, 2), device="cpu")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            GroupedBinaryLinear(weight, -torch.ones((1, 1), device="cpu"))
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            GroupedBinaryLinear(weight, torch.ones((1, 1), device="cpu"), arithmetic="dense")

    def test_explicit_group_matmul_keeps_hard_signs_and_gradients(self):
        rng = np.random.default_rng(82)
        weights = rng.normal(size=(3, 257)).astype(np.float32)
        scales = rng.uniform(0.1, 1.5, size=(3, 3)).astype(np.float32)
        inputs = rng.normal(size=(4, 257)).astype(np.float32)
        exact = GroupedBinaryLinear(
            torch.from_numpy(weights), torch.from_numpy(scales), arithmetic="native_order"
        )
        grouped = GroupedBinaryLinear(
            torch.from_numpy(weights), torch.from_numpy(scales), arithmetic="group_matmul"
        )
        exact_input = torch.tensor(inputs, device="cpu", requires_grad=True)
        grouped_input = torch.tensor(inputs, device="cpu", requires_grad=True)
        exact_output = exact(exact_input)
        grouped_output = grouped(grouped_input)
        torch.testing.assert_close(grouped_output, exact_output, rtol=2e-5, atol=2e-5)
        exact_output.sum().backward()
        grouped_output.sum().backward()
        torch.testing.assert_close(grouped.latent_sign.grad, exact.latent_sign.grad)
        torch.testing.assert_close(
            grouped.scale_offset.grad, exact.scale_offset.grad, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(grouped_input.grad, exact_input.grad)
        np.testing.assert_array_equal(grouped.export_arrays()[0], exact.export_arrays()[0])

    def test_nine_linear_install_restores_qk_row_order_and_preserves_bias(self):
        drafter, candidate, originals = self.candidate_fixture()
        target = nn.Linear(2, 2, device="cpu")
        frozen_target = target.weight.detach().clone()
        replacements = install_candidate_d_linears(drafter, candidate, target=target)
        self.assertEqual(set(replacements), set(CANDIDATE_D_BASE_TO_PATH.values()))
        x = torch.tensor([0.5, 0.25], device="cpu")
        for base, path in CANDIDATE_D_BASE_TO_PATH.items():
            layer = drafter.get_submodule(path)
            self.assertIs(layer, replacements[path])
            signs = unpack_signs(candidate[base][0], 2)
            scales = candidate[base][1]
            if base in ("blk.0.attn_q", "blk.0.attn_k"):
                rows = len(signs)
                heads = 32 if base.endswith("_q") else 8
                signs = (
                    signs.reshape(heads, rows // heads // 2, 2, 2).swapaxes(1, 2).reshape(rows, 2)
                )
                scales = (
                    scales.reshape(heads, rows // heads // 2, 2, 1).swapaxes(1, 2).reshape(rows, 1)
                )
            latent, training_scales = layer.training_arrays()
            np.testing.assert_array_equal(latent, signs)
            np.testing.assert_array_equal(training_scales, scales)
            latent[0, 0] = 123
            self.assertNotEqual(float(layer.latent_sign[0, 0].detach()), 123)
            expected = (0.5 * signs[:, 0] + 0.25 * signs[:, 1]) * scales[:, 0]
            if originals[path].bias is not None:
                expected += originals[path].bias.detach().numpy()
                torch.testing.assert_close(layer.bias, originals[path].bias.detach())
            np.testing.assert_array_equal(layer(x).detach().numpy(), expected)
        torch.testing.assert_close(target.weight.detach(), frozen_target, rtol=0, atol=0)

    def test_install_rejects_target_alias_before_mutation(self):
        drafter, candidate, originals = self.candidate_fixture()
        target = nn.Module()
        target.shared = drafter.fc
        with self.assertRaisesRegex(ValueError, "target-owned"):
            install_candidate_d_linears(drafter, candidate, target=target)
        for path, original in originals.items():
            self.assertIs(drafter.get_submodule(path), original)


if __name__ == "__main__":
    unittest.main()
