"""Conditional F64 logit-box gate; no backend error bound is inferred here."""

from __future__ import annotations

import hashlib
import json
import math


def _hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _difference(left, right, temperature):
    difference = left - right
    # Center before scaling, with overflow-safe opposite-sign subtraction.
    return (
        (left / 2 - right / 2) / temperature * 2
        if not math.isfinite(difference)
        else difference / temperature
    )


def _excluding_normalizers(values, temperature):
    """O(n) normalizers without subtracting nearly equal mass totals."""
    if len(values) == 1:
        return [(values[0], 0.0)]
    maximum_index = max(range(len(values)), key=values.__getitem__)
    maximum = values[maximum_index]
    weights = [math.exp(_difference(value, maximum, temperature)) for value in values]
    prefix, suffix = [0.0], [0.0] * (len(values) + 1)
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    for i in range(len(values) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]
    normalizers = [(maximum, prefix[i] + suffix[i + 1]) for i in range(len(values))]
    # Removing the largest coordinate needs the second largest anchor: otherwise
    # all surviving weights might underflow to zero despite a finite ratio.
    others = values[:maximum_index] + values[maximum_index + 1 :]
    second = max(others)
    normalizers[maximum_index] = (
        second,
        math.fsum(math.exp(_difference(value, second, temperature)) for value in others),
    )
    return normalizers


def _probability(own, normalizer, temperature):
    maximum, weight_sum = normalizer
    if weight_sum == 0.0:
        return 1.0
    log_odds = _difference(maximum, own, temperature) + math.log(weight_sum)
    if log_odds >= 0:
        weight = math.exp(-log_odds)
        return weight / (1.0 + weight)
    return 1.0 / (1.0 + math.exp(log_odds))


def certify(logits, bounds=None, *, token_ids, identity, provenance, contract, numeric_allowance):
    """Certify one fixed ordered, complete processed-logit domain.

    Bounds are EXTERNALLY supplied coordinatewise absolute maximum errors.
    numeric_allowance has separate logit_absolute and probability_absolute
    fields, plus provenance. The caller warrants these enclose arithmetic error;
    this executable does not establish libm or CUDA rounding guarantees.
    """
    result = {
        "status": "invalid",
        "arithmetic": "python_binary64",
        "identity": identity,
        "error_provenance": provenance,
        "numeric_allowance": numeric_allowance,
        "scope": "conditional_fixed_processed_logit_box",
    }

    def fail(status, reason):
        result.update(status=status, reason=reason)
        return result

    if not isinstance(identity, str) or not identity.strip():
        return fail("invalid", "missing reference/candidate/state identity")
    if not isinstance(logits, (list, tuple)) or not logits:
        return fail("invalid", "empty or malformed complete logits")
    try:
        if not all(_finite(x) for x in logits):
            return fail("invalid", "nonfinite or nonnumeric logits")
        values = [float(x) for x in logits]
        if not isinstance(token_ids, (list, tuple)) or len(token_ids) != len(values):
            return fail("invalid", "mapping length mismatch")
        if not all(type(t) is int and t >= 0 for t in token_ids):
            return fail("invalid", "invalid token mapping")
        if len(set(token_ids)) != len(token_ids):
            return fail("invalid", "duplicate token mapping")
        result.update(coordinate_sha256=_hash(values), mapping_sha256=_hash(token_ids))
        if not isinstance(contract, dict) or any(
            [
                contract.get("domain") != "complete_processed",
                contract.get("normalization_complete") is not True,
                contract.get("tie_rule") != "first_in_order",
                contract.get("mapping_stable") is not True,
                contract.get("processor_state_stable") is not True,
                not isinstance(contract.get("processor_identity"), str),
                not bool(contract.get("processor_identity")),
            ]
        ):
            return fail(
                "processor_contract", "need complete fixed processed domain, state and order"
            )
        temperature = contract.get("temperature")
        if not _finite(temperature) or temperature <= 0:
            return fail("invalid", "temperature must be finite and strictly positive")
        temperature = float(temperature)
        result["processor_contract"] = dict(contract)
        if bounds is None or not isinstance(provenance, str) or not provenance.strip():
            return fail("missing_bound", "need externally supplied maximum errors and provenance")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != len(values):
            return fail("invalid", "error bound length mismatch")
        if not all(_finite(e) and e >= 0 for e in bounds):
            return fail("invalid", "bounds must be finite nonnegative coordinatewise maxima")
        if not isinstance(numeric_allowance, dict):
            return fail("invalid", "external numeric allowance required")
        a = numeric_allowance.get("logit_absolute")
        p_allowance = numeric_allowance.get("probability_absolute")
        if not all(_finite(x) and x >= 0 for x in (a, p_allowance)):
            return fail("invalid", "invalid numeric allowance")
        if (
            not isinstance(numeric_allowance.get("provenance"), str)
            or not numeric_allowance["provenance"].strip()
        ):
            return fail("invalid", "numeric allowance provenance required")
        lower, upper = [], []
        for value, error in zip(values, bounds):
            radius = float(error) + float(a)
            if radius > 0:
                radius = math.nextafter(radius, math.inf)
            if not math.isfinite(radius):
                return fail("invalid", "effective radius overflow")
            lo, hi = value - radius, value + radius
            if radius > 0:
                lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
            if not math.isfinite(lo) or not math.isfinite(hi):
                return fail("invalid", "logit box endpoint overflow")
            lower.append(lo)
            upper.append(hi)
        winner = max(range(len(values)), key=values.__getitem__)
        blockers = [
            i
            for i in range(len(values))
            if i != winner
            and (lower[winner] <= upper[i] if i < winner else lower[winner] < upper[i])
        ]
        intervals = []
        upper_normalizers = _excluding_normalizers(upper, temperature)
        lower_normalizers = _excluding_normalizers(lower, temperature)
        for i in range(len(values)):
            lo = _probability(lower[i], upper_normalizers[i], temperature)
            hi = _probability(upper[i], lower_normalizers[i], temperature)
            intervals.append(
                [
                    max(0.0, math.nextafter(lo - p_allowance, -math.inf)),
                    min(1.0, math.nextafter(hi + p_allowance, math.inf)),
                ]
            )
        result.update(
            status="ambiguous" if blockers else "stable",
            reason="competitor box overlaps winner"
            if blockers
            else "ordered argmax fixed throughout box",
            winner_index=winner,
            winner_token=token_ids[winner],
            blockers=blockers,
            lower_logits=lower,
            upper_logits=upper,
            supplied_max_errors=list(bounds),
            probability_bounds=intervals,
            probability_bound_semantics="coordinatewise extrema; distinct attaining vertices",
            backend_guarantee="none; external bound and arithmetic allowance must hold",
        )
        return result
    except (OverflowError, ValueError, TypeError) as exc:
        return fail("invalid", "arithmetic/input failure: " + str(exc))


def rms_counterexample(size=10000):
    """A tiny RMS perturbation flips a near tie; RMS is not a max-error bound."""
    if type(size) is not int or size < 2:
        raise ValueError("size must be an integer >=2")
    gap, shift = 1e-5, 2e-5
    return {
        "size": size,
        "reference_top_two": [gap, 0.0],
        "candidate_top_two": [gap, shift],
        "rms_error": shift / math.sqrt(size),
        "maximum_coordinate_error": shift,
        "reference_winner": 0,
        "candidate_winner": 1,
        "reference_margin": gap,
        "lesson": "RMS cannot be substituted for coordinatewise maximum error",
    }
