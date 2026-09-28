"""Differentiable CPU reference for a captured native EAGLE proposal chain.

The feature encoder and decoder step are supplied by the caller. A step must
return the decoder's pre-norm state for the next proposal, its head logits,
and the updated draft cache. The caller must construct and audit the starting
cache from the current student; cached normalized head states are not valid
recurrence inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class DraftStep:
    logits: Tensor
    pre_norm: Tensor
    cache: Any


def rollout_captured_prefix(
    rows: Sequence[Mapping[str, object]],
    raw_target_features: Tensor,
    *,
    encode_feature: Callable[[Tensor], Tensor],
    decode_step: Callable[[int, Tensor, int, Any], DraftStep],
    initial_cache: Any,
    draft_vocab_size: int,
) -> Tensor:
    """Unroll one exact-prefix chain without detaching student states or cache.

    ``rows`` must first pass ``validate_recurrent_trace`` with its independent
    round anchor and verifier vocabulary map. This function checks the local
    recurrence and fills only a terminal invalid row with zero, ignored logits
    so its output aligns with the trace CE mask.
    """
    if not rows or draft_vocab_size < 1 or raw_target_features.device.type != "cpu":
        raise ValueError("one nonempty CPU round and positive draft vocabulary are required")
    feature = encode_feature(raw_target_features)
    if feature.device.type != "cpu" or feature.ndim != 1 or not torch.isfinite(feature).all():
        raise ValueError("feature encoder must return one finite CPU state")
    cache = initial_cache
    logits_rows = []
    first = rows[0]
    identity = (first["prompt_id"], first["round_index"], first["parent_position"])
    token = first["input_token_id"]
    if type(token) is not int:
        raise ValueError("seed token must be an integer")
    previous_proposal = None
    for depth, row in enumerate(rows):
        if (
            (row.get("prompt_id"), row.get("round_index"), row.get("parent_position")) != identity
            or row.get("depth") != depth
            or row.get("input_position") != identity[2] + depth + 1
            or row.get("input_token_id") != token
        ):
            raise ValueError("proposal chain position, identity or input token mismatch")
        if depth > 0 and row.get("input_token_id") != previous_proposal:
            raise ValueError("proposal chain does not follow the previous proposal")
        if row.get("valid") is not True:
            if depth != len(rows) - 1:
                raise ValueError("invalid proposal row must be terminal")
            logits_rows.append(torch.zeros(draft_vocab_size, dtype=torch.float32, device="cpu"))
            continue
        result = decode_step(token, feature, row["input_position"], cache)
        if not isinstance(result, DraftStep):
            raise TypeError("decoder must return DraftStep")
        if (
            result.logits.device.type != "cpu"
            or result.logits.shape != (draft_vocab_size,)
            or not torch.isfinite(result.logits).all()
            or result.pre_norm.device.type != "cpu"
            or result.pre_norm.shape != feature.shape
            or not torch.isfinite(result.pre_norm).all()
        ):
            raise ValueError("decoder returned invalid logits or pre-norm state")
        logits_rows.append(result.logits)
        feature, cache = result.pre_norm, result.cache
        previous_proposal = row.get("proposed_token_id")
        if type(previous_proposal) is not int:
            raise ValueError("valid proposal row needs a proposed token")
        token = previous_proposal
    return torch.stack(logits_rows)
