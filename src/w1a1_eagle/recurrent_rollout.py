"""Differentiable Torch rollout for a captured native EAGLE proposal chain.

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


@dataclass(frozen=True)
class RebuiltPrefix:
    cache: Any
    seed_token: int
    seed_raw_features: Tensor
    decoder_position: int


def rebuild_prefix_cache(
    prefix_token_ids: Sequence[int],
    raw_target_features: Tensor,
    feature_positions: Sequence[int],
    *,
    parent_position: int,
    encode_feature: Callable[[Tensor], Tensor],
    decode_context: Callable[[int, Tensor, int, Any], DraftStep],
    new_cache: Callable[[], Any],
    build_context_cache: Callable[[Sequence[int], Tensor], Any] | None = None,
) -> RebuiltPrefix:
    """Recompute current-student cache from accepted-prefix target features.

    For every context position ``j < P``, native ``process()`` decodes
    ``(token[j+1], encode_feature(target_features[j]))`` at memory position
    ``j``. The final feature row ``target_features[P]`` is deferred for the
    draft seed ``token[P+1]`` at position ``P``. Context reconstruction is a
    declared truncated-gradient boundary; the proposal unroll retains its
    own student-state and cache gradients. An optional architecture-validated
    builder receives only shifted context tokens and raw context rows under
    no_grad; the deferred seed never enters that batch.
    """
    if type(parent_position) is not int or parent_position < 0:
        raise ValueError("parent_position must be nonnegative")
    if len(prefix_token_ids) != parent_position + 2 or any(
        type(token) is not int or token < 0 for token in prefix_token_ids
    ):
        raise ValueError("accepted prefix tokens do not match parent position")
    if (
        raw_target_features.ndim != 2
        or raw_target_features.shape[0] != parent_position + 1
        or (
            raw_target_features.device.type == "cpu"
            and not torch.isfinite(raw_target_features).all()
        )
    ):
        raise ValueError("raw target features must cover every context and deferred row")
    if tuple(feature_positions) != tuple(range(parent_position + 1)):
        raise ValueError("raw target feature positions are missing or misordered")
    cache = new_cache()
    with torch.no_grad():
        if build_context_cache is not None:
            cache = build_context_cache(
                prefix_token_ids[1 : parent_position + 1], raw_target_features[:parent_position]
            )
        for position in (range(parent_position) if build_context_cache is None else ()):
            feature = encode_feature(raw_target_features[position])
            if (
                feature.device != raw_target_features.device
                or feature.ndim != 1
                or (feature.device.type == "cpu" and not torch.isfinite(feature).all())
            ):
                raise ValueError("feature encoder returned an invalid context row")
            result = decode_context(prefix_token_ids[position + 1], feature, position, cache)
            if not isinstance(result, DraftStep):
                raise TypeError("context decoder must return DraftStep")
            cache = result.cache
    return RebuiltPrefix(
        cache, prefix_token_ids[-1], raw_target_features[parent_position], parent_position
    )


def rollout_captured_prefix(
    rows: Sequence[Mapping[str, object]],
    raw_target_features: Tensor,
    *,
    encode_feature: Callable[[Tensor], Tensor],
    decode_step: Callable[[int, Tensor, int, Any], DraftStep],
    initial_cache: Any,
    draft_vocab_size: int,
    decode_head: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Unroll one exact-prefix chain without detaching student states or cache.

    ``rows`` must first pass ``validate_recurrent_trace`` with its independent
    round anchor and verifier vocabulary map. This function checks the local
    recurrence and fills only a terminal invalid row with zero, ignored logits
    so its output aligns with the trace CE mask. With ``decode_head``, the step
    callback must omit logits (return an empty row); valid attached pre-norm
    states are stacked for one head call after serial recurrent decoding.
    """
    if not rows or draft_vocab_size < 1:
        raise ValueError("one nonempty round and positive draft vocabulary are required")
    feature = encode_feature(raw_target_features)
    if (
        feature.device != raw_target_features.device
        or feature.ndim != 1
        or (feature.device.type == "cpu" and not torch.isfinite(feature).all())
    ):
        raise ValueError("feature encoder must return one valid state on the input device")
    cache = initial_cache
    logits_rows = []
    states = []
    valid_indices = []
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
            logits_rows.append(
                torch.zeros(draft_vocab_size, dtype=torch.float32, device=feature.device)
            )
            continue
        # Native decoder memory/RoPE position is one before the shifted token.
        result = decode_step(token, feature, row["input_position"] - 1, cache)
        if not isinstance(result, DraftStep):
            raise TypeError("decoder must return DraftStep")
        expected_logits = (draft_vocab_size,) if decode_head is None else (0,)
        if (
            result.logits.device != feature.device
            or result.logits.shape != expected_logits
            or (feature.device.type == "cpu" and not torch.isfinite(result.logits).all())
            or result.pre_norm.device != feature.device
            or result.pre_norm.shape != feature.shape
            or (feature.device.type == "cpu" and not torch.isfinite(result.pre_norm).all())
        ):
            raise ValueError("decoder returned invalid logits or pre-norm state")
        if decode_head is None:
            logits_rows.append(result.logits)
        else:
            valid_indices.append(len(logits_rows))
            states.append(result.pre_norm)
            logits_rows.append(None)
        feature, cache = result.pre_norm, result.cache
        previous_proposal = row.get("proposed_token_id")
        if type(previous_proposal) is not int:
            raise ValueError("valid proposal row needs a proposed token")
        token = previous_proposal
    if states:
        logits = decode_head(torch.stack(states))
        if (
            not isinstance(logits, Tensor)
            or logits.device != feature.device
            or logits.shape != (len(states), draft_vocab_size)
            or (feature.device.type == "cpu" and not torch.isfinite(logits).all())
        ):
            raise ValueError("batched head returned invalid logits")
        for index, value in zip(valid_indices, logits.unbind(0)):
            logits_rows[index] = value
    return torch.stack(logits_rows)
