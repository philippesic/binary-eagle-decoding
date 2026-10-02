"""Dense, independent CPU reference for changed-row head flip candidates."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
INITIAL_RESET_UNIX = 1791049896


def check_control() -> dict:
    """Fail closed when the shared research allowance says to stop."""
    control = json.loads(CONTROL.read_text())
    remaining = 100.0 - float(control["last_weekly_used_percent"])
    if (
        control["research_stop"]
        or control["reset_observed"]
        or control["last_reset_unix"] != INITIAL_RESET_UNIX
        or remaining <= float(control.get("threshold_remaining_percent", 1))
    ):
        raise RuntimeError(f"parallel research control stopped: {control}")
    return control


def _f32(value: Tensor) -> Tensor:
    return value.to(dtype=torch.float32, device="cpu")


def dense_reference(snapshot, candidate) -> dict[str, Tensor]:
    """Reconstruct one candidate from a full flipped sign matrix and dense dots.

    This intentionally does not use candidate deltas, model methods, or helper
    functions from the implementation under test.
    """
    check_control()
    codes = snapshot.codes.to(dtype=torch.int64, device="cpu")
    signs = snapshot.signs.to(dtype=torch.int64, device="cpu").clone()
    if codes.ndim != 2 or signs.ndim != 2 or signs.shape[1] != codes.shape[1]:
        raise AssertionError("snapshot codes/signs must be [T,K] and [V,K]")
    row = int(candidate.row)
    columns = tuple(int(c) for c in candidate.columns)
    if row < 0 or row >= signs.shape[0] or any(c < 0 or c >= signs.shape[1] for c in columns):
        raise AssertionError("candidate row or column outside the dense head")
    if len(columns) != len(set(columns)):
        raise AssertionError("candidate contains duplicate columns")
    if columns:
        signs[row, torch.tensor(columns, dtype=torch.long)] *= -1

    # Explicit dense integer GEMM is the independent head oracle.
    integer_dots = codes @ signs.T
    alpha = _f32(snapshot.alpha)
    beta = _f32(snapshot.beta)
    logits = (integer_dots.to(torch.float32) * alpha.unsqueeze(0)) * beta
    midpoint = getattr(snapshot, "midpoint", None)
    if midpoint is not None:
        mid = codes.sum(dim=1, dtype=torch.int64).to(torch.float32).unsqueeze(1) * _f32(
            midpoint
        ).unsqueeze(0)
        logits = logits + mid * beta
    bias = getattr(snapshot, "bias", None)
    if bias is not None:
        logits = logits + _f32(bias).unsqueeze(0)
    logits = logits + 0.0

    labels = snapshot.target_labels.to(dtype=torch.long, device="cpu")
    label_rows = snapshot.label_rows.to(dtype=torch.long, device="cpu")
    if labels.ndim != 1 or labels.shape[0] != codes.shape[0]:
        raise AssertionError("target labels must have one absolute token id per row")
    if label_rows.shape != labels.shape:
        raise AssertionError("label_rows must map each target id to a head row")
    baseline = _f32(snapshot.baseline_logits)
    if baseline.shape != logits.shape:
        raise AssertionError("baseline logits do not cover the dense output head")

    # Float64 log-sum-exp gives a stable independent CE comparison.
    base64, candidate64 = baseline.to(torch.float64), logits.to(torch.float64)
    base_lse = torch.logsumexp(base64, dim=1)
    candidate_lse = torch.logsumexp(candidate64, dim=1)
    safe_rows = label_rows.clamp_min(0)
    base_ce = base_lse - base64.gather(1, safe_rows[:, None]).squeeze(1)
    candidate_ce = candidate_lse - candidate64.gather(1, safe_rows[:, None]).squeeze(1)
    base_ce = torch.where(label_rows >= 0, base_ce, torch.nan)
    candidate_ce = torch.where(label_rows >= 0, candidate_ce, torch.nan)
    top1_rows = logits.argmax(dim=1)
    top1_target_ids = snapshot.d2t.to(dtype=torch.long, device="cpu")[top1_rows]
    top1_margin = logits.topk(2, dim=1).values.diff(dim=1).squeeze(1).abs()
    target_logit = logits.gather(1, safe_rows[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, safe_rows[:, None], -torch.inf)
    label_margin = target_logit - masked.max(dim=1).values
    label_margin = torch.where(label_rows >= 0, label_margin, torch.nan)
    return {
        "dots": integer_dots[:, row],
        "logits": logits[:, row],
        "full_logits": logits,
        "ce_delta": candidate_ce - base_ce,
        "top1_rows": top1_rows,
        "top1_target_ids": top1_target_ids,
        "top1_margin": top1_margin,
        "label_margin": label_margin,
    }
