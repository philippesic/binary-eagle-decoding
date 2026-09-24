"""Greedy parity checks permit tree-round overshoot but detect missing tokens."""

import unittest

from scripts.evaluate_pytorch_w1a1 import first_greedy_mismatch, variant_quantization


class GreedyPrefixTests(unittest.TestCase):
    def test_exact_and_round_overshoot(self):
        self.assertIsNone(first_greedy_mismatch([2, 3, 4], [2, 3, 4]))
        self.assertIsNone(first_greedy_mismatch([2, 3, 4], [2, 3, 4, 5]))

    def test_missing_or_different_token(self):
        self.assertEqual(first_greedy_mismatch([2, 3, 4], [2, 3]), 2)
        self.assertEqual(first_greedy_mismatch([2, 3, 4], [2, 8, 4]), 1)

    def test_low_bit_protocol_is_explicit(self):
        protocol = {
            "mode": "symmetric_uniform",
            "scale": "absmax_per_row_and_token",
            "rounding": "nearest_even",
            "accumulation": "fp32_simulation",
            "output_dtype": "drafter_dtype",
        }
        variant = {
            "name": "int4_w4a4",
            "groups": ["lm_head"],
            "weight_bits": 4,
            "activation_bits": 4,
        }
        spec = variant_quantization(protocol, variant)
        self.assertEqual((spec["weight_bits"], spec["activation_bits"]), (4, 4))
        weight_only = variant_quantization(protocol, {**variant, "activation_bits": None})
        self.assertIsNone(weight_only["activation_bits"])
        self.assertEqual(variant_quantization(protocol, {"groups": []}), {"mode": "ordinary"})
        with self.assertRaisesRegex(ValueError, "4 or 8 weight bits"):
            variant_quantization(protocol, {**variant, "weight_bits": 3})
        with self.assertRaisesRegex(ValueError, "rounding=nearest_even"):
            variant_quantization({**protocol, "rounding": "floor"}, variant)


if __name__ == "__main__":
    unittest.main()
