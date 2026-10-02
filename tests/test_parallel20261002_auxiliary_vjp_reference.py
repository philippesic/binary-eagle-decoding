"""All-family actual-adapter composition gate, synthetic CPU only."""

import unittest

import torch

from research.parallel20261002.auxiliary_vjp.reference.audit import (
    mode_matrix,
    negative_controls,
    one_update,
)


class AuxiliaryVJPReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_unique_vjps_match_independent_local_algebra(self):
        self.assertEqual(len(mode_matrix()), 48)

    def test_one_joint_clipped_update_matches(self):
        for bits in (1, 4, 8):
            self.assertLessEqual(one_update(bits)["clipped_norm"], 0.070001)

    def test_detach_and_quantizer_alias_negative_controls(self):
        controls = negative_controls()
        self.assertGreater(controls["state"], 1e-4)
        self.assertGreater(controls["cache"], 1e-4)
        self.assertTrue(controls["split_tied_quantizer_rejected"])
        self.assertTrue(controls["cross_boundary_alias_rejected"])


if __name__ == "__main__":
    unittest.main()
