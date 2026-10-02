"""Owner checks; independent exhaustive vertices live in validation/."""

import importlib.util
import math
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "certificate", Path(__file__).with_name("certificate.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def case(logits, bounds, **changes):
    args = dict(
        token_ids=list(range(len(logits))),
        identity="synthetic-fixed-state",
        provenance="synthetic external coordinatewise maximum error",
        contract=dict(
            domain="complete_processed",
            normalization_complete=True,
            temperature=1.0,
            tie_rule="first_in_order",
            mapping_stable=True,
            processor_state_stable=True,
            processor_identity="synthetic identity processor",
        ),
        numeric_allowance=dict(
            logit_absolute=0.0,
            probability_absolute=1e-14,
            provenance="synthetic CPU validation tolerance",
        ),
    )
    args.update(changes)
    return MODULE.certify(logits, bounds, **args)


class GateChecks(unittest.TestCase):
    def test_stable_and_ambiguous(self):
        self.assertEqual(case([2.0, 0.0], [0.1, 0.1])["status"], "stable")
        self.assertEqual(case([1e-5, 0.0], [0.0, 2e-5])["status"], "ambiguous")

    def test_first_tie_order_and_mapping(self):
        row = case([0.0, 0.0], [0.0, 0.0], token_ids=[99, 1])
        self.assertEqual(row["status"], "stable")
        self.assertEqual(row["winner_token"], 99)
        self.assertEqual(case([0.0, 0.0], [0.0, 0.0], token_ids=[1, 1])["status"], "invalid")

    def test_missing_max_bound(self):
        self.assertEqual(case([0.0], None)["status"], "missing_bound")
        self.assertEqual(case([0.0], [0.0], provenance="")["status"], "missing_bound")

    def test_external_numeric_allowance_required(self):
        self.assertEqual(case([0.0], [0.0], numeric_allowance=None)["status"], "invalid")
        self.assertEqual(case([0.0], [0.0], numeric_allowance={})["status"], "invalid")

    def test_nonfinite_and_overflow_fail_closed(self):
        for value in [math.nan, math.inf, -math.inf, True]:
            self.assertEqual(case([value], [0.0])["status"], "invalid")
        self.assertEqual(case([1e308], [1e308])["status"], "invalid")

    def test_large_offsets_and_extreme_separation(self):
        for logits in [[1e300, 1e300], [1e308, -1e308]]:
            row = case(logits, [0.0, 0.0])
            self.assertEqual(row["status"], "stable")
            self.assertTrue(all(0 <= lo <= hi <= 1 for lo, hi in row["probability_bounds"]))
        self.assertGreater(case([1e308, -1e308], [0.0, 0.0])["probability_bounds"][0][0], 0.999)

    def test_temperature_and_processor_contract(self):
        base = case([1.0, 0.0], [0.0, 0.0])["processor_contract"]
        for value in [0.0, -1.0, math.nan, math.inf]:
            self.assertEqual(
                case([1.0, 0.0], [0.0, 0.0], contract=dict(base, temperature=value))["status"],
                "invalid",
            )
        self.assertEqual(
            case([1.0, 0.0], [0.0, 0.0], contract=dict(base, domain="raw_top_k"))["status"],
            "processor_contract",
        )
        row = case([1e308, -1e308], [0.0, 0.0], contract=dict(base, temperature=1e308))
        exact = 1.0 / (1.0 + math.exp(-2.0))
        self.assertLessEqual(row["probability_bounds"][0][0], exact)
        self.assertGreaterEqual(row["probability_bounds"][0][1], exact)

    def test_radius_addition_rounds_outward_before_cancellation(self):
        row = case(
            [1.0],
            [1.0],
            numeric_allowance=dict(
                logit_absolute=2**-53,
                probability_absolute=1e-14,
                provenance="synthetic rounding fixture",
            ),
        )
        self.assertLess(row["lower_logits"][0], -(2**-53))

    def test_rms_not_margin_proof(self):
        row = MODULE.rms_counterexample()
        self.assertLess(row["rms_error"], row["reference_margin"])
        self.assertGreater(row["maximum_coordinate_error"], row["reference_margin"])
        self.assertNotEqual(row["reference_winner"], row["candidate_winner"])


if __name__ == "__main__":
    unittest.main()
