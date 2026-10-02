"""Bounded composed-export checks, independent of real model artifacts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.parallel20261002.export_function.reference.audit import (
    evaluate,
    original_rows,
    quantize,
    run,
)


class ExportedFunctionTests(unittest.TestCase):
    def test_actual_checkpoint_export_all_nine_composed_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory))
            self.assertEqual(result["comparisons"], 81)
            self.assertEqual(result["max_abs_error"], 0.0)
            self.assertEqual({c["bits"] for c in result["cases"]}, {1, 4, 8})
            self.assertEqual({c["width"] for c in result["cases"]}, {33, 65, 130})
            self.assertEqual({c["rank"] for c in result["cases"]}, {1, 4})
            for case in result["cases"]:
                self.assertGreater(case["negative_control_max_delta"], 0.01)
                self.assertGreater(case["factor_rounding_max_delta"], 1e-7)
                # Decoder receives only serialized bytes plus raw x.
                with np.load(case["expected"]) as expected:
                    for item in case["comparisons"]:
                        base = item["base"]
                        np.testing.assert_array_equal(
                            evaluate(case["gguf"], base, expected["x::" + base]),
                            expected["y::" + base],
                        )

    def test_inverse_qk_permutation_is_observable(self):
        for base, heads in (("blk.0.attn_q", 32), ("blk.0.attn_k", 8)):
            rows = heads * 4
            native = np.arange(rows).reshape(rows, 1)
            expected = np.concatenate(
                [np.asarray([4 * h, 4 * h + 2, 4 * h + 1, 4 * h + 3]) for h in range(heads)]
            ).reshape(rows, 1)
            np.testing.assert_array_equal(original_rows(native, base), expected)
            self.assertFalse(np.array_equal(native, expected))

    def test_signed_zeros_ties_and_subnormal_quantizer_codes(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        x = np.asarray([[-0.0, 0.0, -tiny, tiny]], dtype=np.float32)
        codes, beta = quantize(x, 1, 0, 1)
        np.testing.assert_array_equal(codes, [[1, 1, -1, 1]])
        self.assertEqual(beta.dtype, np.float32)
        # x=1, beta=1, delta=1 gives an exact positive threshold tie.
        codes, _ = quantize(np.ones((1, 4), dtype=np.float32), 1, 1, 1)
        np.testing.assert_array_equal(codes, np.ones((1, 4), dtype=np.int32))
        for bits in (4, 8):
            codes, beta = quantize(x, bits, 0, 0.75)
            qmax = (1 << (bits - 1)) - 1
            np.testing.assert_array_equal(codes, [[0, 0, -qmax, qmax]])
            np.testing.assert_array_equal(beta, np.zeros((1, 1), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
