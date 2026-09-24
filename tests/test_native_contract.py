"""CPU checks for the head-only native packed arithmetic simulation."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

from kernels.binary_reference import binary_dot, binary_linear, pack_activations, pack_weights
from scripts.evaluate_pytorch_w1a1 import variant_quantization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.native_contract import (  # noqa: E402
    NativeContractHead,
    activation_scale_f64_to_f32,
    install_native_contract_head,
    integer_sign_dot,
    native_contract_linear,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeContractTests(unittest.TestCase):
    def test_activation_scale_uses_f64_sum_before_f32_rounding(self):
        values = torch.tensor([[2**20, 2**-4, 2**-4]], dtype=torch.bfloat16)
        f32_mean = values.float().abs().mean(dim=-1, keepdim=True)
        native_mean = activation_scale_f64_to_f32(values)
        self.assertEqual(f32_mean.item(), 349525.34375)
        self.assertEqual(native_mean.item(), 349525.375)

    def test_integer_sign_dot_matches_packed_reference_for_tails_and_signed_zero(self):
        for k in (1, 31, 32, 33, 63, 64, 65, 2560):
            with self.subTest(k=k):
                inputs = torch.tensor(
                    [
                        [(-0.0 if i % 11 == 0 else (i % 5) - 2) for i in range(k)],
                        [(0.0 if i % 7 == 0 else 2 - (i % 5)) for i in range(k)],
                    ],
                    dtype=torch.bfloat16,
                )
                weights = torch.tensor(
                    [
                        [(0.0 if i % 13 == 0 else (i % 3) - 1) for i in range(k)],
                        [(-0.0 if i % 17 == 0 else 1 - (i % 3)) for i in range(k)],
                    ],
                    dtype=torch.bfloat16,
                )
                packed_inputs = pack_activations(inputs.tolist())
                packed_weights = pack_weights(weights.tolist())
                actual = integer_sign_dot(inputs, weights)
                for row in range(2):
                    for column in range(2):
                        expected = binary_dot(
                            packed_inputs.words[row], packed_weights.words[column], k
                        )
                        self.assertEqual(actual[row, column].item(), expected)

    def test_output_matches_packed_f32_reference(self):
        for k in (1, 31, 33, 65):
            with self.subTest(k=k):
                values = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0]
                inputs = torch.tensor(
                    [[values[(i + offset) % len(values)] for i in range(k)] for offset in (0, 3)],
                    dtype=torch.bfloat16,
                )
                weights = torch.tensor(
                    [
                        [values[(i * 3 + offset) % len(values)] for i in range(k)]
                        for offset in (0, 1, 2)
                    ],
                    dtype=torch.bfloat16,
                )
                bias = torch.tensor([0.0, 0.5, -0.5], dtype=torch.bfloat16)
                expected = binary_linear(
                    pack_activations(inputs.tolist()),
                    pack_weights(
                        weights.tolist(),
                        scales=weights.float().abs().mean(dim=-1).tolist(),
                    ),
                    bias=bias.tolist(),
                )
                expected_f32 = torch.tensor(expected, dtype=torch.float32)
                self.assertTrue(
                    torch.equal(native_contract_linear(inputs, weights, bias), expected_f32)
                )

    def test_f32_logits_flow_through_angelslim_topk_and_d2t_operations(self):
        inputs = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)
        weights = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=torch.bfloat16,
        )
        logits = native_contract_linear(inputs, weights)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertNotEqual(logits[0, 0].item(), logits.bfloat16()[0, 0].item())
        # Pinned AngelSlim _get_topk_tokens applies LogSoftmax then topk;
        # topK_genrate maps the selected indices with its integer d2t buffer.
        scores = nn.LogSoftmax(dim=-1)(logits)
        topk = torch.topk(scores, 2, dim=-1)
        d2t = torch.tensor([0, 1, 2], dtype=torch.long)
        mapped = topk.indices + d2t[topk.indices]
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(topk.values.dtype, torch.float32)
        self.assertEqual(mapped.dtype, torch.long)

    def test_wrapper_disabled_and_uninstall_preserve_ordinary_head(self):
        drafter = nn.Module()
        drafter.lm_head = nn.Linear(33, 3, bias=False, dtype=torch.bfloat16)
        original = drafter.lm_head
        inputs = torch.tensor([[1.0, -1.0, 0.0] * 11], dtype=torch.bfloat16)
        ordinary = original(inputs)
        handle = install_native_contract_head(drafter, enabled=False)
        self.assertIsInstance(drafter.lm_head, NativeContractHead)
        self.assertTrue(torch.equal(drafter.lm_head(inputs), ordinary))
        self.assertEqual(drafter.lm_head(inputs).dtype, torch.bfloat16)
        handle.set_enabled(True)
        self.assertEqual(drafter.lm_head(inputs).dtype, torch.float32)
        self.assertTrue(
            torch.equal(drafter.lm_head(inputs), native_contract_linear(inputs, original.weight))
        )
        handle.uninstall()
        self.assertIs(drafter.lm_head, original)

    def test_rejects_non_bf16_and_wrong_shapes(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            native_contract_linear(torch.ones(2, 3), torch.ones(4, 3))
        with self.assertRaisesRegex(ValueError, "expected input"):
            native_contract_linear(
                torch.ones(2, 3, dtype=torch.bfloat16),
                torch.ones(4, 2, dtype=torch.bfloat16),
            )

    def test_variant_contract_and_dry_run(self):
        quant = {
            "mode": "native_packed_binary",
            "zero_sign": 1,
            "weight_scale": "mean_abs_f32_per_row",
            "activation_scale": "mean_abs_f64_sum_f32_per_token",
            "integer_dot": "k_minus_2_popcount_xor",
            "scale_order": "dot_weight_activation_f32",
            "output_dtype": "float32",
        }
        self.assertEqual(variant_quantization(quant, {"groups": []}), {"mode": "ordinary"})
        self.assertEqual(
            variant_quantization(quant, {"groups": ["lm_head"]})["mode"],
            "native_packed_binary",
        )
        with self.assertRaisesRegex(ValueError, "only the lm_head"):
            variant_quantization(quant, {"groups": ["ffn"]})
        with self.assertRaisesRegex(ValueError, "zero_sign=1"):
            variant_quantization({**quant, "zero_sign": -1}, {"groups": ["lm_head"]})
        output = subprocess.check_output(
            [
                sys.executable,
                "scripts/evaluate_pytorch_w1a1.py",
                "--config",
                "configs/pytorch_w1a1_native_head.toml",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
        )
        self.assertEqual(json.loads(output)["variants"], ["ordinary", "head"])


if __name__ == "__main__":
    unittest.main()
