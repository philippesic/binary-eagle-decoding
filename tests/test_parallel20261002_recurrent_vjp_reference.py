"""Composed production-API CPU contract; no model or capture loading."""

import unittest

import torch

from research.parallel20261002.recurrent_vjp.reference.audit import (
    mode_matrix,
    rollback_rebuild_control,
    stale_cache_control,
    two_step_resume,
)


class RecurrentVJPReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_serial_topology_matches_all_supported_optimization_controls(self):
        self.assertEqual(len(mode_matrix()), 216)

    def test_second_update_matches_serial_and_restored_optimizer(self):
        for bits in (1, 4, 8):
            with self.subTest(bits=bits):
                self.assertEqual(two_step_resume(bits)["parameter_families"], 18)

    def test_stale_context_cache_after_weight_change_is_detected(self):
        self.assertGreater(stale_cache_control()["max_stale_logit_error"], 1e-4)

    def test_next_round_discards_proposal_cache_and_rebuilds_accepted_prefix(self):
        self.assertTrue(rollback_rebuild_control()["wrong_position_rejected"])


if __name__ == "__main__":
    unittest.main()
