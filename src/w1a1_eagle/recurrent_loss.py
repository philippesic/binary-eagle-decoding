"""Masked exact-prefix objective for a recurrent binary drafter unroll.

The caller supplies logits from its *current* hard-binary student forward and
a separately audited trace. This module does not construct the EAGLE model,
capture target labels, or change any student cache state.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .recurrent_trace import TraceAudit


def supported_prefix_rows(logits: Tensor, audit: TraceAudit) -> tuple[Tensor, Tensor]:
    """Mean CE over valid mapped labels without detaching any student logits.

    Unsupported labels remain in ``audit.denominator_mask`` for separate
    quality accounting. An unsupported earlier row must not sever autograd
    between its student state and a supported later row.
    """
    if logits.ndim != 2 or logits.shape != (
        len(audit.draft_labels),
        sum(label >= 0 for label in audit.target_to_draft),
    ):
        raise ValueError("logits must have one row per trace row and one column per draft token")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    if len(audit.ce_mask) != len(audit.draft_labels) or not any(audit.ce_mask):
        raise ValueError("trace has no supported valid label")
    if any(mask != (label >= 0) for mask, label in zip(audit.ce_mask, audit.draft_labels)):
        raise ValueError("CE mask disagrees with audited labels")
    mask = torch.tensor(audit.ce_mask, dtype=torch.bool, device=logits.device)
    labels = torch.tensor(audit.draft_labels, dtype=torch.long, device=logits.device)
    return logits[mask].float(), labels[mask]


def supported_prefix_ce(logits: Tensor, audit: TraceAudit) -> Tensor:
    """Mean CE on audited supported rows; every current-student graph stays attached."""
    rows, labels = supported_prefix_rows(logits, audit)
    return F.cross_entropy(rows, labels, reduction="mean")
