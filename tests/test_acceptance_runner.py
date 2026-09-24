"""Greedy parity checks permit tree-round overshoot but detect missing tokens."""

import unittest

from scripts.evaluate_pytorch_w1a1 import first_greedy_mismatch


class GreedyPrefixTests(unittest.TestCase):
    def test_exact_and_round_overshoot(self):
        self.assertIsNone(first_greedy_mismatch([2, 3, 4], [2, 3, 4]))
        self.assertIsNone(first_greedy_mismatch([2, 3, 4], [2, 3, 4, 5]))

    def test_missing_or_different_token(self):
        self.assertEqual(first_greedy_mismatch([2, 3, 4], [2, 3]), 2)
        self.assertEqual(first_greedy_mismatch([2, 3, 4], [2, 8, 4]), 1)


if __name__ == "__main__":
    unittest.main()
