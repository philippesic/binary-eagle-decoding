"""CPU-only numerical reference for packed W1A1 drafter linears.

This module defines an interchange contract for a later native kernel, not a
performance implementation. Rows are contiguous and each row has ``ceil(K/32)``
little-bit-order uint32 words: bit ``k % 32`` of word ``k // 32`` is 1 for
``value >= 0`` and 0 for ``value < 0``. Thus positive and negative zero are +1.
Unused high bits in the final word are zero when packed and always masked when
read. The logical reduction length, never padded length, enters the dot.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ScaleRule = Literal["mean_abs", "rms", "unit"]
WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1


def _f32(value: float) -> float:
    """Round a Python value to IEEE-754 binary32, including scale arithmetic."""
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("value is outside finite float32 range") from exc
    if not math.isfinite(result):
        raise ValueError("value is outside finite float32 range")
    return result


def _row_scale(row: Sequence[float], rule: ScaleRule) -> float:
    if rule == "unit":
        return 1.0
    if rule == "mean_abs":
        return _f32(math.fsum(abs(value) for value in row) / len(row))
    if rule == "rms":
        return _f32(math.sqrt(math.fsum(value * value for value in row) / len(row)))
    raise ValueError("scale rule must be mean_abs, rms, or unit")


def _scales(
    rows: Sequence[Sequence[float]],
    rule: ScaleRule,
    scales: float | Sequence[float] | None,
) -> tuple[float, ...]:
    if scales is None:
        return tuple(_row_scale(row, rule) for row in rows)
    if isinstance(scales, (int, float)):
        return (_f32(float(scales)),) * len(rows)
    if len(scales) != len(rows):
        raise ValueError("per-row scales must match the number of rows")
    return tuple(_f32(float(scale)) for scale in scales)


@dataclass(frozen=True)
class PackedRows:
    """A row-major matrix with one float32 scale per logical row.

    Weight rows are output features; activation rows are independent tokens.
    The original input values are not retained. A caller flattens batch/time
    dimensions into activation rows and restores the output shape afterward.
    """

    words: tuple[tuple[int, ...], ...]
    k: int
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.k <= 0 or not self.words or len(self.words) != len(self.scales):
            raise ValueError("packed rows need positive K and one scale per row")
        expected_words = (self.k + WORD_BITS - 1) // WORD_BITS
        for row in self.words:
            if len(row) != expected_words or any(word < 0 or word > WORD_MASK for word in row):
                raise ValueError("each packed row needs ceil(K/32) uint32 words")
        object.__setattr__(self, "scales", tuple(_f32(float(scale)) for scale in self.scales))

    @property
    def nrows(self) -> int:
        return len(self.words)


def _pack(
    values: Sequence[Sequence[float]],
    *,
    scale_rule: ScaleRule,
    scales: float | Sequence[float] | None,
) -> PackedRows:
    if scale_rule not in ("mean_abs", "rms", "unit"):
        raise ValueError("scale rule must be mean_abs, rms, or unit")
    if not values or not values[0]:
        raise ValueError("expected a nonempty row-major matrix")
    k = len(values[0])
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if any(len(row) != k for row in rows):
        raise ValueError("rows must have the same positive K")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("input values must be finite")
    packed = []
    for row in rows:
        words = [0] * ((k + WORD_BITS - 1) // WORD_BITS)
        for index, value in enumerate(row):
            if value >= 0:
                words[index // WORD_BITS] |= 1 << (index % WORD_BITS)
        packed.append(tuple(words))
    return PackedRows(tuple(packed), k, _scales(rows, scale_rule, scales))


def pack_weights(
    weights: Sequence[Sequence[float]],
    *,
    scale_rule: ScaleRule = "mean_abs",
    scales: float | Sequence[float] | None = None,
) -> PackedRows:
    """Pack an ``(output_features, K)`` weight matrix.

    Default scales are per output row, matching the inference simulation.
    Pass a scalar or one value per row to override them. All scales are float32.
    """
    return _pack(weights, scale_rule=scale_rule, scales=scales)


def pack_activations(
    activations: Sequence[Sequence[float]],
    *,
    scale_rule: ScaleRule = "mean_abs",
    scales: float | Sequence[float] | None = None,
) -> PackedRows:
    """Pack ``(tokens, K)`` activations, with one scale per token by default."""
    return _pack(activations, scale_rule=scale_rule, scales=scales)


def binary_dot(lhs: Sequence[int], rhs: Sequence[int], k: int) -> int:
    """Return ``K - 2*popcount(XOR)`` over exactly K logical bits."""
    if k <= 0 or len(lhs) != (k + WORD_BITS - 1) // WORD_BITS or len(rhs) != len(lhs):
        raise ValueError("word count must equal ceil(positive K/32)")
    mismatches = 0
    for index, (a, b) in enumerate(zip(lhs, rhs, strict=True)):
        if not (0 <= a <= WORD_MASK and 0 <= b <= WORD_MASK):
            raise ValueError("words must be uint32")
        remaining = k - index * WORD_BITS
        mask = WORD_MASK if remaining >= WORD_BITS else (1 << remaining) - 1
        mismatches += ((a ^ b) & mask).bit_count()
    return k - 2 * mismatches


def binary_linear(
    activations: PackedRows,
    weights: PackedRows,
    *,
    bias: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Compute ``(tokens, output_features)`` in float32 from packed operands.

    For each dot, float32 rounding occurs after the integer-to-float conversion,
    after multiplication by the weight scale, after the activation scale, and
    after optional bias addition. This is a numerical contract for comparison;
    BF16 PyTorch simulation may round at different points.
    """
    if activations.k != weights.k:
        raise ValueError("activation and weight K must match")
    if bias is not None and len(bias) != weights.nrows:
        raise ValueError("bias length must equal output features")
    rounded_bias = None if bias is None else tuple(_f32(float(value)) for value in bias)
    result = []
    for token_words, token_scale in zip(activations.words, activations.scales, strict=True):
        output_row = []
        for feature, (weight_words, weight_scale) in enumerate(
            zip(weights.words, weights.scales, strict=True)
        ):
            value = _f32(float(binary_dot(token_words, weight_words, weights.k)))
            value = _f32(value * weight_scale)
            value = _f32(value * token_scale)
            if rounded_bias is not None:
                value = _f32(value + rounded_bias[feature])
            output_row.append(value)
        result.append(tuple(output_row))
    return tuple(result)
