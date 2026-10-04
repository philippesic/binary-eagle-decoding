"""Bounded TRAIN-only fusion calibration at fixed deployed A8/A1 arithmetic.

Scale-only signs are retained as a control. Optional zero-scale orientation
rescue negates an entire negatively correlated row, re-solves its nonnegative
scale, and admits it only against the separately rounded exported F32 objective.
This is a calibration initializer, not a model-quality or native admission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

ARITHMETIC = {
    8: "f32_absmax_reciprocal_rint127_integer_dot_scale_f32_token_scale_f32",
    1: "f64_meanabs_to_f32_zero_positive_sign_integer_dot_scale_f32_token_scale_f32",
}


def quantize(raw, bits):
    x = np.asarray(raw)
    if x.dtype != np.float32 or x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("fusion inputs require finite nonempty 2D F32")
    if type(bits) is not int or bits not in ARITHMETIC:
        raise ValueError("only deployed fixed A8/A1 arithmetic supported")
    if bits == 1:
        beta = np.mean(np.abs(x).astype(np.float64), axis=1).astype(np.float32)
        codes = np.where(x < 0, -1, 1).astype(np.int16)
    else:
        limit = np.max(np.abs(x), axis=1)
        beta = np.divide(limit, np.float32(127), dtype=np.float32)
        inv = np.divide(np.float32(127), limit, out=np.zeros_like(limit), where=limit > 0)
        if not np.isfinite(inv).all():
            raise ValueError("native fixed A8 reciprocal overflow; no substitution")
        codes = np.clip(np.rint(np.multiply(x, inv[:, None], dtype=np.float32)), -127, 127).astype(
            np.int16
        )
    if x.shape[1] * (127 if bits == 8 else 1) >= 2**24:
        raise ValueError("integer dot exceeds exact F32 range")
    return codes, beta


def project(codes, beta, signs, scales):
    if (
        codes.ndim != 2
        or signs.ndim != 2
        or codes.shape[1] != signs.shape[1]
        or beta.shape != (len(codes),)
        or scales.shape != (len(signs),)
        or not np.all((signs == -1) | (signs == 1))
        or not np.isfinite(scales).all()
        or np.any(scales < 0)
        or not np.isfinite(beta).all()
        or np.any(beta < 0)
        or codes.dtype.kind not in "iu"
        or np.any(np.abs(codes.astype(np.int64)) > 127)
        or codes.shape[1] * 127 >= 2**24
    ):
        raise ValueError("invalid deployed projection operands")
    dots = codes.astype(np.float32) @ signs.T.astype(np.float32)
    return np.multiply(np.multiply(dots, scales, dtype=np.float32), beta[:, None], dtype=np.float32)


def _output(dots, scale, beta):
    return np.multiply(
        np.multiply(dots.astype(np.float32), np.float32(scale), dtype=np.float32),
        beta,
        dtype=np.float32,
    )


def _sse(prediction, teacher):
    residual = prediction.astype(np.float64) - teacher.astype(np.float64)
    return float(np.sum(residual * residual))


def _solve_scale(dots, beta, teacher, previous):
    design = dots.astype(np.float64) * beta.astype(np.float64)
    correlation = float(design @ teacher.astype(np.float64))
    denominator = float(design @ design)
    optimum = max(0.0, correlation / denominator) if denominator else 0.0
    if not np.isfinite(optimum) or optimum > np.finfo(np.float32).max:
        raise ValueError("nonfinite exported scale optimum")
    candidates = (np.float32(optimum), np.float32(previous), np.float32(0))
    best = min(candidates, key=lambda a: _sse(_output(dots, a, beta), teacher))
    loss = _sse(_output(dots, best, beta), teacher)
    for _ in range(32):
        neighbors = (np.nextafter(best, np.float32(0)), np.nextafter(best, np.float32(np.inf)))
        changed = False
        for candidate in neighbors:
            if np.isfinite(candidate) and candidate >= 0:
                candidate_loss = _sse(_output(dots, candidate, beta), teacher)
                if candidate_loss < loss:
                    best, loss, changed = candidate, candidate_loss, True
        if not changed:
            return best, loss, correlation
    raise ValueError("exported F32 scale neighboring objective did not settle in bound")


@dataclass(frozen=True)
class FusionFitConfig:
    activation_bits: int
    zero_scale_orientation_rescue: bool = False
    max_coordinate_flips_per_row: int = 0
    max_seconds: float = 60.0

    def __post_init__(self):
        if type(self.activation_bits) is not int or self.activation_bits not in ARITHMETIC:
            raise ValueError("fusion activation bits must be 1 or 8")
        if type(self.zero_scale_orientation_rescue) is not bool:
            raise ValueError("orientation rescue must be explicit boolean")
        if (
            type(self.max_coordinate_flips_per_row) is not int
            or not 0 <= self.max_coordinate_flips_per_row <= 32
        ):
            raise ValueError("coordinate flips bound must be in [0,32]")
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not np.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("finite positive calibration time bound required")


def fit_fusion(raw, teacher, reference_weight, config: FusionFitConfig):
    """Fit only admitted calibration-fit rows; caller owns independent validation.

    Optional coordinate search tries the best continuous estimated coordinate,
    re-solves its scale and checks exact finite arithmetic before accepting.
    No sign-search optimality claim is made. Defaults are scale-only control.
    """
    started = time.monotonic()
    codes, beta = quantize(raw, config.activation_bits)
    weight, y = np.asarray(reference_weight), np.asarray(teacher)
    if (
        weight.dtype != np.float32
        or weight.ndim != 2
        or weight.shape[1] != raw.shape[1]
        or y.dtype != np.float32
        or y.shape != (len(raw), len(weight))
        or not np.isfinite(weight).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("fusion teacher/original weights require exact finite F32 shapes")
    original_signs = np.where(weight < 0, -1, 1).astype(np.int8)
    signs = original_signs.copy()
    initializer = np.mean(np.abs(weight), axis=1, dtype=np.float32)
    scales = np.zeros(len(weight), dtype=np.float32)
    control = np.zeros_like(scales)
    events = []
    negative_zero = []
    design = codes.astype(np.float64) * beta[:, None].astype(np.float64)
    diagonal = np.sum(design * design, axis=0)

    def budget():
        if time.monotonic() - started > config.max_seconds:
            raise TimeoutError("fusion calibration time bound exhausted")

    for row in range(len(weight)):
        budget()
        dots = codes.astype(np.float32) @ signs[row].astype(np.float32)
        scales[row], loss, correlation = _solve_scale(dots, beta, y[:, row], initializer[row])
        control[row] = scales[row]
        if scales[row] == 0 and correlation < 0:
            negative_zero.append(row)
            if config.zero_scale_orientation_rescue:
                trial_scale, trial_loss, _ = _solve_scale(-dots, beta, y[:, row], 0)
                if trial_loss < loss:
                    events.append(
                        {
                            "kind": "row_orientation",
                            "row": row,
                            "before_sse": loss,
                            "after_sse": trial_loss,
                        }
                    )
                    signs[row] *= -1
                    dots, scales[row], loss = -dots, trial_scale, trial_loss
        for _ in range(config.max_coordinate_flips_per_row):
            budget()
            if scales[row] == 0:
                break
            residual = _output(dots, scales[row], beta).astype(np.float64) - y[:, row]
            delta = -2 * float(scales[row]) * signs[row]
            estimates = 2 * delta * (residual @ design) + delta * delta * diagonal
            proposals = np.argsort(estimates, kind="stable")[:16]
            accepted = False
            for col in proposals:
                if estimates[col] >= 0:
                    break
                trial_dots = dots - 2 * int(signs[row, col]) * codes[:, col].astype(np.float32)
                trial_scale, trial_loss, _ = _solve_scale(trial_dots, beta, y[:, row], scales[row])
                if trial_loss < loss:
                    events.append(
                        {
                            "kind": "coordinate",
                            "row": row,
                            "column": int(col),
                            "before_sse": loss,
                            "after_sse": trial_loss,
                        }
                    )
                    signs[row, col] *= -1
                    dots, scales[row], loss = trial_dots, trial_scale, trial_loss
                    accepted = True
                    break
            if not accepted:
                break
    prediction = project(codes, beta, signs, scales)
    control_prediction = project(codes, beta, original_signs, control)
    if _sse(prediction, y) > _sse(control_prediction, y):
        raise ValueError("frozen calibration candidate worsens scale-only exported objective")
    return {
        "latent": signs.astype(np.float32),
        "scale": scales,
        "control_latent": original_signs.astype(np.float32),
        "control_scale": control,
        "report": {
            "arithmetic": ARITHMETIC[config.activation_bits],
            "negative_correlation_zero_scale_rows": negative_zero,
            "events": events,
            "fit_seconds": time.monotonic() - started,
            "candidate": diagnostics(prediction, y),
            "scale_only": diagnostics(control_prediction, y),
            "admission": "requires_independent_native_trajectory_and_quality_gates",
        },
    }


def diagnostics(prediction, teacher):
    p, y = np.asarray(prediction), np.asarray(teacher)
    if (
        p.shape != y.shape
        or p.ndim != 2
        or not p.size
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("diagnostics require finite matching nonempty matrices")
    energy = float(np.sum(y.astype(np.float64) ** 2))
    denominator = np.linalg.norm(p.astype(np.float64), axis=1) * np.linalg.norm(
        y.astype(np.float64), axis=1
    )
    valid = denominator > 0
    cosine = np.sum(p.astype(np.float64) * y, axis=1)[valid] / denominator[valid]
    return {
        "sse": _sse(p, y),
        "relative_squared_error": _sse(p, y) / energy if energy else None,
        "row_cosine": float(np.mean(cosine)) if len(cosine) else None,
        "coordinate_sign_agreement": float(np.mean((p >= 0) == (y >= 0))),
        "largest_coordinate_agreement": float(
            np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1))
        ),
        "scope": "fusion coordinates only; not native acceptance or throughput",
    }


def rms_norm_reference(raw, gamma, epsilon):
    """Portable F32 diagnostic; native norm reduction equivalence needs its gate."""
    x, g = np.asarray(raw), np.asarray(gamma)
    if (
        x.dtype != np.float32
        or x.ndim != 2
        or g.dtype != np.float32
        or g.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(g).all()
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(
            "post-fusion RMS diagnostic requires finite F32 values and positive epsilon"
        )
    variance = np.mean(np.multiply(x, x, dtype=np.float32), axis=1, keepdims=True, dtype=np.float32)
    return np.multiply(
        np.multiply(x, 1 / np.sqrt(variance + np.float32(epsilon)), dtype=np.float32),
        g,
        dtype=np.float32,
    )
