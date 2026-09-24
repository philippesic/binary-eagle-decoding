"""Independent scalar checks for the CPU packed W1A1 contract."""

import math
import random
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.binary_reference import (  # noqa: E402
    PackedRows,
    binary_dot,
    binary_linear,
    pack_activations,
    pack_weights,
)


def floating_sign_dot(left: list[float], right: list[float]) -> int:
    """Use arithmetic signs and scalar multiplication, not packed XOR."""
    return sum(
        (1 if a >= 0 else -1) * (1 if b >= 0 else -1) for a, b in zip(left, right, strict=True)
    )


def scale(row: list[float], rule: str) -> float:
    if rule == "mean_abs":
        return math.fsum(abs(value) for value in row) / len(row)
    if rule == "rms":
        return math.sqrt(math.fsum(value * value for value in row) / len(row))
    return 1.0


class BinaryReferenceTests(unittest.TestCase):
    def test_known_encoding_layout_and_zero_sign(self) -> None:
        packed = pack_activations(
            [[0.0, -0.0, 2.0, -3.0, -0.125], [-1.0, 0.0, -0.0, 4.0, 5.0]],
            scale_rule="unit",
        )
        self.assertEqual(packed.k, 5)
        self.assertEqual(packed.words, ((0b00111,), (0b11110,)))
        self.assertEqual(packed.scales, (1.0, 1.0))
        self.assertEqual(binary_dot(*packed.words, packed.k), -1)

    def test_full_and_partial_words_match_floating_sign_dot(self) -> None:
        rng = random.Random(750)
        for k in (1, 2, 31, 32, 33, 63, 64, 65, 97, 255):
            with self.subTest(k=k):
                left = [rng.choice((-2.0, -0.0, 0.0, 0.25, 4.0)) for _ in range(k)]
                right = [rng.choice((-3.0, -0.0, 0.0, 0.5, 5.0)) for _ in range(k)]
                packed_left = pack_activations([left], scale_rule="unit")
                packed_right = pack_weights([right], scale_rule="unit")
                self.assertEqual(
                    binary_dot(packed_left.words[0], packed_right.words[0], k),
                    floating_sign_dot(left, right),
                )
                self.assertEqual(len(packed_left.words[0]), (k + 31) // 32)
                self.assertEqual(
                    binary_linear(packed_left, packed_right)[0][0], floating_sign_dot(left, right)
                )

    def test_tail_mask_ignores_dirty_unused_bits(self) -> None:
        left = pack_activations([[1.0] * 33], scale_rule="unit")
        right = pack_weights([[-1.0] * 33], scale_rule="unit")
        self.assertEqual(left.words[0], (0xFFFFFFFF, 1))
        self.assertEqual(right.words[0], (0, 0))
        # A backend may leave arbitrary data in padding. K still controls the dot.
        dirty = replace(left, words=((0xFFFFFFFF, 0xFFFFFFFF),))
        self.assertEqual(binary_dot(dirty.words[0], right.words[0], 33), -33)
        self.assertEqual(binary_linear(dirty, right), ((-33.0,),))

    def test_rows_tokens_bias_and_scale_rules(self) -> None:
        tokens = [[0.0, -2.0, 1.0], [-1.0, 5.0, 0.0], [0.0, 0.0, 0.0]]
        weights = [[0.0, 2.0, -3.0], [-4.0, 0.0, 1.0]]
        bias = [0.25, -1.0]
        for weight_rule in ("mean_abs", "rms", "unit"):
            for token_rule in ("mean_abs", "rms", "unit"):
                with self.subTest(weight_rule=weight_rule, token_rule=token_rule):
                    packed_tokens = pack_activations(tokens, scale_rule=token_rule)
                    packed_weights = pack_weights(weights, scale_rule=weight_rule)
                    actual = binary_linear(packed_tokens, packed_weights, bias=bias)
                    self.assertEqual((len(actual), len(actual[0])), (3, 2))
                    for i, token in enumerate(tokens):
                        for j, weight in enumerate(weights):
                            expected = (
                                floating_sign_dot(token, weight)
                                * scale(token, token_rule)
                                * scale(weight, weight_rule)
                                + bias[j]
                            )
                            self.assertAlmostEqual(actual[i][j], expected, delta=2e-6)

    def test_explicit_scale_scalar_and_per_row_broadcast(self) -> None:
        tokens = pack_activations([[1.0, -1.0], [0.0, -2.0]], scales=0.5)
        weights = pack_weights([[1.0, 1.0], [-1.0, -1.0]], scales=[1.25, 2.5])
        self.assertEqual(tokens.scales, (0.5, 0.5))
        self.assertEqual(weights.scales, (1.25, 2.5))
        self.assertEqual(binary_linear(tokens, weights), ((0.0, 0.0), (0.0, 0.0)))
        all_positive = pack_activations([[1.0, 1.0], [0.0, 0.0]], scales=[0.5, 1.0])
        self.assertEqual(binary_linear(all_positive, weights), ((1.25, -2.5), (2.5, -5.0)))

    def test_scales_are_float32_and_zero_rows_remain_zero(self) -> None:
        packed = pack_activations([[0.0] * 7, [0.1] * 7])
        expected_f32 = struct.unpack("<f", struct.pack("<f", 0.1))[0]
        self.assertEqual(packed.scales, (0.0, expected_f32))
        weights = pack_weights([[1.0] * 7], scales=1.0)
        self.assertEqual(binary_linear(packed, weights)[0], (0.0,))

    def test_audited_eagle_reduction_shapes_fit_layout(self) -> None:
        # (out, K) from experiments/eagle3-graph-audit.md. Sample one output row
        # per layer; this checks real reduction widths without loading 218M weights.
        shapes = (
            (2560, 7680),
            (4096, 5120),
            (1024, 5120),
            (1024, 5120),
            (2560, 4096),
            (9728, 2560),
            (9728, 2560),
            (2560, 9728),
            (32000, 2560),
        )
        for output_features, k in shapes:
            with self.subTest(out=output_features, k=k):
                sample = [
                    0.0 if index % 7 == 0 else (-1.0 if index % 3 == 0 else 1.0)
                    for index in range(k)
                ]
                packed = pack_weights([sample], scale_rule="unit")
                self.assertEqual(len(packed.words[0]), (k + 31) // 32)
                self.assertEqual(binary_dot(packed.words[0], packed.words[0], k), k)
                self.assertEqual(
                    binary_linear(pack_activations([sample], scale_rule="unit"), packed),
                    ((float(k),),),
                )

    def test_invalid_shape_and_values(self) -> None:
        for rows in ([], [[]], [[1.0], [1.0, 2.0]], [[math.nan]], [[math.inf]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pack_weights(rows)
        for rule in ("bad", ""):
            with self.assertRaises(ValueError):
                pack_activations([[1.0]], scale_rule=rule)
        with self.assertRaises(ValueError):
            pack_weights([[1.0], [2.0]], scales=[1.0])
        with self.assertRaises(ValueError):
            pack_weights([[1.0]], scales=math.inf)
        with self.assertRaises(ValueError):
            PackedRows(((0x100000000,),), 1, (1.0,))
        with self.assertRaises(ValueError):
            binary_dot((1,), (1,), 33)
        with self.assertRaises(ValueError):
            binary_linear(pack_activations([[1.0]]), pack_weights([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            binary_linear(pack_activations([[1.0]]), pack_weights([[1.0]]), bias=[0, 1])


if __name__ == "__main__":
    unittest.main()
