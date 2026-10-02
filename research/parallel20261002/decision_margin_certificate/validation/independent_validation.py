"""Independent tiny-box validation for the decision-margin certificate.

This oracle deliberately enumerates box vertices and computes softmax directly;
it does not reuse the certificate's interval formulas.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import random
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
REFERENCE = REPOSITORY / "research/parallel20261002/decision_margin_certificate/reference"
CERTIFICATE_SPEC = importlib.util.spec_from_file_location(
    "decision_margin_certificate_reference", REFERENCE / "certificate.py"
)
if CERTIFICATE_SPEC is None or CERTIFICATE_SPEC.loader is None:
    raise ImportError(f"cannot load reference certificate from {REFERENCE}")
CERTIFICATE_MODULE = importlib.util.module_from_spec(CERTIFICATE_SPEC)
CERTIFICATE_SPEC.loader.exec_module(CERTIFICATE_MODULE)
certify = CERTIFICATE_MODULE.certify


IDENTITY = "fixture:model:tokenizer:processor:v1"
PROVENANCE = "independent tiny fixture with a finite coordinatewise bound"
ALLOWANCE = {
    "logit_absolute": 0.0,
    "probability_absolute": 0.0,
    "provenance": "zero allowance for exact F64 oracle fixtures",
}


def contract(temperature: float = 1.0) -> dict[str, object]:
    return {
        "domain": "complete_processed",
        "temperature": temperature,
        "tie_rule": "first_in_order",
        "processor_identity": "fixture-processor-v1",
        "normalization_complete": True,
        "mapping_stable": True,
        "processor_state_stable": True,
    }


def softmax_at(logits: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    scaled = tuple(value / temperature for value in logits)
    peak = max(scaled)
    weights = tuple(math.exp(value - peak) for value in scaled)
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def all_vertices(
    logits: tuple[float, ...], bounds: tuple[float, ...], allowance: float = 0.0
) -> tuple[tuple[float, ...], ...]:
    """Enumerate all corners after applying the externally supplied logit allowance."""
    radii = tuple(bound + allowance for bound in bounds)
    return tuple(
        tuple(value + sign * radius for value, sign, radius in zip(logits, signs, radii))
        for signs in itertools.product((-1.0, 1.0), repeat=len(logits))
    )


def first_winner(values: tuple[float, ...]) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def call_certificate(
    logits: tuple[float, ...],
    bounds: tuple[float, ...] | None,
    token_ids: tuple[int, ...] | None = None,
    temperature: float = 1.0,
    numeric_allowance: dict[str, object] | None = None,
    evidence: str = PROVENANCE,
    identity: str = IDENTITY,
    processor_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    return certify(
        logits=logits,
        bounds=bounds,
        token_ids=token_ids
        if token_ids is not None
        else tuple(101 + i for i in range(len(logits))),
        identity=identity,
        provenance=evidence,
        contract=processor_contract if processor_contract is not None else contract(temperature),
        numeric_allowance=numeric_allowance if numeric_allowance is not None else ALLOWANCE,
    )


class DecisionMarginCertificateValidation(unittest.TestCase):
    def test_exhaustive_vertex_probability_bounds_and_greedy(self) -> None:
        rng = random.Random(20261002)
        cases = [
            ((0.2, -0.4), (0.0, 0.0), 0.7),
            ((0.5, -0.3, 0.1), (0.15, 0.2, 0.05), 1.4),
            ((1.0, 0.2, -0.5, 0.0), (0.1, 0.1, 0.15, 0.05), 0.8),
        ]
        for dimension in (2, 3, 4):
            for _ in range(5):
                logits = tuple(rng.uniform(-0.8, 0.8) for _ in range(dimension))
                bounds = tuple(rng.uniform(0.0, 0.12) for _ in range(dimension))
                cases.append((logits, bounds, rng.uniform(0.4, 1.8)))

        for logits, bounds, temperature in cases:
            with self.subTest(logits=logits, bounds=bounds, temperature=temperature):
                result = call_certificate(logits, bounds, temperature=temperature)
                vertices = all_vertices(logits, bounds)
                probabilities = [softmax_at(vertex, temperature) for vertex in vertices]
                expected = [
                    (
                        min(row[index] for row in probabilities),
                        max(row[index] for row in probabilities),
                    )
                    for index in range(len(logits))
                ]
                actual = result["probability_bounds"]
                self.assertEqual(len(actual), len(expected))
                for observed, oracle in zip(actual, expected):
                    self.assertAlmostEqual(observed[0], oracle[0], delta=3e-14)
                    self.assertAlmostEqual(observed[1], oracle[1], delta=3e-14)

                vertex_winners = {
                    first_winner(softmax_at(vertex, temperature)) for vertex in vertices
                }
                self.assertEqual(
                    result["status"], "stable" if len(vertex_winners) == 1 else "ambiguous"
                )
                if len(vertex_winners) == 1:
                    winner = next(iter(vertex_winners))
                    self.assertEqual(result["winner_index"], winner)
                    self.assertEqual(result["winner_token"], 101 + winner)
                else:
                    # Ambiguous results retain the nominal reference argmax;
                    # it is not a robust candidate-decision guarantee.
                    self.assertEqual(result["winner_index"], first_winner(logits))

    def test_ordered_exact_tie_is_stable_only_at_zero_allowance(self) -> None:
        exact = call_certificate((2.0, 2.0, 1.0), (0.0, 0.0, 0.0))
        self.assertEqual(exact["status"], "stable")
        self.assertEqual(exact["winner_index"], 0)
        self.assertEqual(exact["winner_token"], 101)

        positive = dict(ALLOWANCE)
        positive["logit_absolute"] = 1e-9
        positive["provenance"] = "nonzero outward logit allowance"
        uncertain = call_certificate((2.0, 2.0, 1.0), (0.0, 0.0, 0.0), numeric_allowance=positive)
        self.assertEqual(uncertain["status"], "ambiguous")
        self.assertEqual(uncertain["winner_index"], 0)

    def test_probability_allowance_outward_expands_endpoints(self) -> None:
        allowance = dict(ALLOWANCE)
        allowance["probability_absolute"] = 1e-6
        allowance["provenance"] = "specified absolute endpoint allowance"
        result = call_certificate((1.0, -1.0), (0.0, 0.0), numeric_allowance=allowance)
        for low, high in result["probability_bounds"]:
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertAlmostEqual(high - low, 2e-6, delta=2e-12)

    def test_radius_rounding_is_outward_at_a_tie_boundary(self) -> None:
        # The nominal first coordinate exceeds the second by one ULP. A radius
        # of that same size must make the first-in-order tie boundary reachable.
        logits = (1.0, math.nextafter(1.0, 0.0))
        allowance = dict(ALLOWANCE)
        allowance["logit_absolute"] = math.ulp(1.0) / 2.0
        allowance["provenance"] = "half-ULP externally supplied radius control"
        result = call_certificate(logits, (0.0, 0.0), numeric_allowance=allowance)
        self.assertEqual(result["status"], "ambiguous")

    def test_softmax_handles_opposite_sign_extremes_and_large_temperature(self) -> None:
        logits = (1e308, -1e308)
        temperature = 1e308
        result = call_certificate(
            logits,
            (0.0, 0.0),
            temperature=temperature,
        )
        expected = softmax_at(logits, temperature)
        self.assertEqual(result["status"], "stable")
        for observed, oracle in zip(result["probability_bounds"], expected):
            self.assertAlmostEqual(observed[0], oracle, delta=3e-15)
            self.assertAlmostEqual(observed[1], oracle, delta=3e-15)

    def test_constructive_rms_bound_does_not_certify_a_maximum_error(self) -> None:
        # A single 0.1 error among 100 coordinates has RMS 0.01 but exceeds it
        # by 10x. Treating that RMS as a coordinatewise maximum falsely certifies
        # the target winner even though the actual approximate winner is flipped.
        target_logits = (0.05, 0.0) + (0.0,) * 98
        approximate_logits = (-0.05, 0.0) + (0.0,) * 98
        error = tuple(a - b for a, b in zip(target_logits, approximate_logits))
        rms = math.sqrt(math.fsum(value * value for value in error) / len(error))
        self.assertAlmostEqual(rms, 0.01)
        self.assertEqual(first_winner(target_logits), 0)
        self.assertEqual(first_winner(approximate_logits), 1)
        alleged_rms_as_max = (rms,) + (0.0,) * (len(target_logits) - 1)
        certificate = call_certificate(
            target_logits,
            alleged_rms_as_max,
            evidence="deliberately invalid control: RMS mislabeled as coordinatewise maximum",
        )
        self.assertEqual(certificate["status"], "stable")

    def test_identity_and_mapping_are_bound_and_order_is_preserved(self) -> None:
        ordered = call_certificate((3.0, 1.0), (0.0, 0.0), token_ids=(901, 17))
        reversed_order = call_certificate((1.0, 3.0), (0.0, 0.0), token_ids=(17, 901))
        self.assertEqual(ordered["winner_token"], 901)
        self.assertEqual(reversed_order["winner_token"], 901)
        self.assertNotEqual(ordered["mapping_sha256"], reversed_order["mapping_sha256"])
        self.assertNotEqual(ordered["coordinate_sha256"], reversed_order["coordinate_sha256"])

        changed_identity = call_certificate((3.0, 1.0), (0.0, 0.0), identity="different-runtime")
        self.assertEqual(ordered["coordinate_sha256"], changed_identity["coordinate_sha256"])
        self.assertNotEqual(ordered["identity"], changed_identity["identity"])

    def test_missing_or_invalid_coordinate_bounds_fail_closed(self) -> None:
        missing = call_certificate((1.0, 0.0), None)
        self.assertEqual(missing["status"], "missing_bound")

        for bad_logits, bad_bounds in (
            ((math.nan, 0.0), (0.1, 0.1)),
            ((math.inf, 0.0), (0.1, 0.1)),
            ((1.0, 0.0), (0.1,)),
            ((1.0, 0.0), (0.1, math.nan)),
            ((1.0, 0.0), (0.1, math.inf)),
            ((1.0, 0.0), (0.1, -0.1)),
        ):
            with self.subTest(logits=bad_logits, bounds=bad_bounds):
                result = call_certificate(bad_logits, bad_bounds)
                self.assertEqual(result["status"], "invalid")
                self.assertNotIn("winner_index", result)
                self.assertNotIn("probability_bounds", result)

    def test_contract_is_checked_before_missing_coordinate_bounds(self) -> None:
        bad = contract()
        bad["normalization_complete"] = False
        result = call_certificate((1.0, 0.0), None, processor_contract=bad)
        self.assertEqual(result["status"], "processor_contract")

        for temperature in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature):
                result = call_certificate((1.0, 0.0), (0.0, 0.0), temperature=temperature)
                self.assertEqual(result["status"], "invalid")

    def test_invalid_identity_provenance_mapping_and_allowances(self) -> None:
        for kwargs in (
            {"identity": ""},
            {"evidence": ""},
            {"token_ids": (7, 7)},
        ):
            with self.subTest(kwargs=kwargs):
                result = call_certificate((1.0, 0.0), (0.0, 0.0), **kwargs)
                self.assertIn(result["status"], {"invalid", "processor_contract", "missing_bound"})
                self.assertNotIn("winner_index", result)
                self.assertNotIn("probability_bounds", result)

        for bad_allowance in (
            {"logit_absolute": -1.0, "probability_absolute": 0.0, "provenance": "bad"},
            {"logit_absolute": math.nan, "probability_absolute": 0.0, "provenance": "bad"},
            {"logit_absolute": 0.0, "probability_absolute": math.inf, "provenance": "bad"},
            {"logit_absolute": 0.0, "probability_absolute": 0.0, "provenance": ""},
            {"logit_absolute": 0.0, "provenance": "missing probability"},
        ):
            with self.subTest(allowance=bad_allowance):
                result = call_certificate((1.0, 0.0), (0.0, 0.0), numeric_allowance=bad_allowance)
                self.assertEqual(result["status"], "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
