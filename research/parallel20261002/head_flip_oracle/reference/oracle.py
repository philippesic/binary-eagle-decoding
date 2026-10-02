"""Source-bound CPU diagnostic for <=32 independent, single-row flip sets.

All inputs are captured and detached. This evaluates native-order F32 logits,
then F64 hard CE and rank telemetry; it neither selects nor applies an update.
Ties use the declared lowest draft-row convention, before the absolute d2t map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from w1a1_eagle.affine_binary import AffineBinaryMidpoint
from w1a1_eagle.learned_activation import LearnedActivationQuantizer, learned_activation_reference
from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import RowBinaryLinear, _HardActivationSTE

F32_EPS = torch.finfo(torch.float32).eps


@dataclass(frozen=True)
class FlipCandidate:
    row: int
    columns: tuple[int, ...]


@dataclass(frozen=True)
class HeadSnapshot:
    codes: Tensor
    signs: Tensor
    alpha: Tensor
    beta: Tensor
    midpoint: Tensor | None
    bias: Tensor | None
    baseline_logits: Tensor
    module_logits: Tensor
    dots: Tensor
    input_sum: Tensor
    d2t: Tensor
    target_labels: Tensor
    label_rows: Tensor
    activation_bits: int
    rule: str
    baseline_max_drift: float


@dataclass(frozen=True)
class CandidateResult:
    candidate: FlipCandidate
    dots: Tensor
    logits: Tensor
    ce_delta: Tensor
    top1_rows: Tensor
    top1_target_ids: Tensor
    label_margin: Tensor
    top1_margin: Tensor
    rounding_tie: Tensor
    avoided_candidate_head_gemms: int = 1


def _epilogue(dot, alpha, beta, input_sum, midpoint, bias):
    output = (dot.float() * alpha) * beta
    if midpoint is not None:
        output = output + (input_sum.float() * midpoint) * beta
    if bias is not None:
        output = output + bias
    # Match RowBinaryLinear's reference attachment's positive-zero addition.
    return output + 0.0


@torch.no_grad()
def capture_head(
    module: RowBinaryLinear, inputs: Tensor, d2t: Tensor, target_labels: Tensor
) -> HeadSnapshot:
    """Capture one batch of fixed head states; d2t contains absolute target IDs.

    Fixed symmetric A4/A8 trainer logits have a different dense-dot order; both
    baselines are retained. No bitwise equivalence is asserted for that path.
    """
    if type(module) is not RowBinaryLinear or module.contract.activation_bits not in (1, 4, 8):
        raise ValueError("requires an actual row A1/A4/A8 head; A16 is excluded")
    if module.latent_sign.device.type != "cpu" or inputs.device.type != "cpu":
        raise ValueError("this isolated diagnostic is CPU only")
    if getattr(module, "fusion_correction", None) is not None:
        raise ValueError("head FC correction is outside this diagnostic")
    if module._round_hard_signs is not None:
        raise ValueError("capture outside an active shared-sign training round")
    if inputs.ndim != 2 or inputs.shape[1] != module.in_features or inputs.shape[0] < 1:
        raise ValueError("inputs must have nonempty token by feature shape")
    if not inputs.is_floating_point() or not bool(torch.isfinite(inputs.float()).all()):
        raise ValueError("head inputs must be finite after F32 cast")
    if (
        d2t.dtype != torch.int64 or d2t.shape != (module.out_features,)
        or d2t.device.type != "cpu" or bool((d2t < 0).any())
        or d2t.unique().numel() != d2t.numel()
    ):
        raise ValueError("absolute d2t must be unique nonnegative CPU int64 row IDs")
    if (
        target_labels.dtype != torch.int64 or target_labels.shape != (inputs.shape[0],)
        or target_labels.device.type != "cpu" or bool((target_labels < 0).any())
    ):
        raise ValueError("target labels need nonnegative CPU int64 token shape")
    bits = module.contract.activation_bits
    quantizer = getattr(module, "activation_quantizer", None)
    affine = getattr(module, "affine_binary", None)
    if not bool(torch.isfinite(module.latent_sign).all()):
        raise ValueError("head latent signs must be finite")
    if quantizer is not None and type(quantizer) is not LearnedActivationQuantizer:
        raise ValueError("unknown source quantizer")
    if affine is not None and type(affine) is not AffineBinaryMidpoint:
        raise ValueError("unknown source affine midpoint")
    if quantizer is not None and quantizer.bits != bits:
        raise ValueError("quantizer precision differs from head")
    if quantizer is not None or affine is not None:
        parameter = (
            quantizer.parameter.detach().clone() if quantizer is not None
            else torch.tensor(0.0 if bits == 1 else 1.0)
        )
        result = learned_activation_reference(inputs, bits, parameter)
        codes, beta = result.codes, result.scale
        rule = "learned" if quantizer is not None else "fixed_affine_v1"
    else:
        _, beta, _, codes = _HardActivationSTE.apply(inputs, bits)
        rule = "fixed_symmetric"
    qmax = 1 if bits == 1 else (1 << (bits - 1)) - 1
    if not bool(torch.isfinite(codes).all()) or bool((codes.abs() > qmax).any()):
        raise ValueError("legacy quantizer produced invalid/nonfinite codes")
    codes = codes.to(torch.int64).clone()
    if module.in_features * qmax >= 2**24:
        raise ValueError("integer dot exceeds exact F32 accumulation bound")
    signs = hard_sign_ste(module.latent_sign).to(torch.int64).clone()
    alpha = module.effective_scales().detach().clone()
    midpoint = None if affine is None else affine.midpoint.detach().clone()
    bias = None if module.frozen_bias is None else module.frozen_bias.detach().clone()
    for value in (alpha, beta, midpoint, bias):
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError("captured epilogue parameters must be finite")
    dots = codes @ signs.T  # One baseline head projection, never one per candidate.
    input_sum = codes.sum(-1, keepdim=True)
    baseline = _epilogue(dots, alpha, beta, input_sum, midpoint, bias)
    # Preserve telemetry mutated by the source forward; no parameter/state update.
    saturation = module.last_saturation_fraction
    try:
        module_logits = module(inputs).detach().clone()
    finally:
        module.last_saturation_fraction = saturation
    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(module_logits).all()):
        raise ValueError("projection overflowed F32")
    labels = torch.full_like(target_labels, -1)
    for row, target in enumerate(d2t.tolist()):
        labels[target_labels == target] = row
    return HeadSnapshot(
        codes, signs, alpha, beta.detach().clone(), midpoint, bias,
        baseline.clone(), module_logits, dots, input_sum, d2t.clone(),
        target_labels.clone(), labels, bits, rule,
        float((baseline - module_logits).abs().max()),
    )


def _lse_parts(logits: Tensor):
    maximum = logits.max(-1).values
    relative = torch.log(torch.exp(logits - maximum[:, None]).sum(-1))
    return maximum, relative


@torch.no_grad()
def evaluate_candidates(snapshot: HeadSnapshot, candidates: list[FlipCandidate]):
    """Evaluate every nomination independently against the same cached baseline.

    No candidate head GEMM occurs: exact I64 dot update, one F32 row epilogue,
    cached unchanged logits. Unsupported labels retain NaN CE/margin and still
    have top1 telemetry; never drop them from an acceptance denominator.
    """
    if len(candidates) > 32:
        raise ValueError("at most 32 nominated candidates are allowed")
    tokens, vocab = snapshot.baseline_logits.shape
    if vocab < 2:
        raise ValueError("rank telemetry requires at least two draft rows")
    base = snapshot.baseline_logits.double()
    old_max, old_relative = _lse_parts(base)
    supported = snapshot.label_rows >= 0
    indices = snapshot.label_rows.clamp_min(0)
    old_label = base[torch.arange(tokens), indices]
    competitors = {}
    outputs = []
    for candidate in candidates:
        row, columns = candidate.row, candidate.columns
        if type(row) is not int or not 0 <= row < vocab:
            raise ValueError("candidate row outside head")
        if (
            not columns or len(set(columns)) != len(columns)
            or any(type(c) is not int or not 0 <= c < snapshot.codes.shape[1] for c in columns)
        ):
            raise ValueError("flip coordinates must be unique nonempty valid integers")
        dot = snapshot.dots[:, row] - 2 * (
            snapshot.codes[:, columns] * snapshot.signs[row, columns]
        ).sum(-1)
        new_row = _epilogue(
            dot, snapshot.alpha[row], snapshot.beta[:, 0], snapshot.input_sum[:, 0],
            None if snapshot.midpoint is None else snapshot.midpoint[row],
            None if snapshot.bias is None else snapshot.bias[row],
        )
        if not bool(torch.isfinite(new_row).all()):
            raise ValueError("candidate projection overflowed F32")
        if row not in competitors:
            other = base.clone()
            other[:, row] = -torch.inf
            competitors[row] = (other.max(-1), _lse_parts(other))
        (best_value, best_index), (other_max, other_relative) = competitors[row]
        new = new_row.double()
        new_max = torch.maximum(new, other_max)
        new_relative = torch.logaddexp(
            new - new_max, (other_max - new_max) + other_relative
        )
        label_delta = torch.where(indices == row, new - base[:, row], 0.0)
        ce_delta = ((new_max - old_max) - label_delta) + (new_relative - old_relative)
        ce_delta = torch.where(new == base[:, row], 0.0, ce_delta)
        ce_delta = torch.where(supported, ce_delta, torch.nan)
        choose_row = (new > best_value) | ((new == best_value) & (row < best_index))
        winner = torch.where(choose_row, row, best_index)
        # Reuse baseline rows for exact rank/margin telemetry, without projection.
        changed = base.clone()
        changed[:, row] = new
        sorted_values = changed.topk(2, dim=-1).values
        top_margin = sorted_values[:, 0] - sorted_values[:, 1]
        rivals = changed.clone()
        rivals[torch.arange(tokens), indices] = -torch.inf
        label_margin = old_label + label_delta - rivals.max(-1).values
        label_margin = torch.where(supported, label_margin, torch.nan)
        gate = 8 * F32_EPS * torch.maximum(torch.ones_like(new_max), new_max.abs())
        outputs.append(CandidateResult(
            candidate, dot, new_row, ce_delta, winner, snapshot.d2t[winner],
            label_margin, top_margin, top_margin <= gate,
        ))
    return outputs


def overshoot_counterexample():
    """Correct local relaxed derivative, harmful complete binary sign flip."""
    signs = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1])
    logits = torch.stack((2 * signs.expand(4), torch.zeros(4)), dim=-1)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    derivative, = torch.autograd.grad(loss, signs)
    flipped = torch.stack((-2 * torch.ones(4), torch.zeros(4)), dim=-1).double()
    new_loss = torch.nn.functional.cross_entropy(flipped, labels)
    return {
        "old_ce": float(loss.detach()), "gradient": float(derivative),
        "linear_prediction_delta": float(-2 * derivative),
        "new_ce": float(new_loss), "actual_delta": float(new_loss - loss.detach()),
        "old_accuracy": 0.75, "new_accuracy": 0.25,
    }
