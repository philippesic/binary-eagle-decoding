"""CPU-only contract for exact-prefix recurrent EAGLE training traces.

This audits metadata emitted by a native verifier capture; it does not produce
verifier labels or prove that the native capture used the correct target model.
Every round needs an independently recorded anchor so a row cannot borrow a
prefix from another round or prompt. Invalid rows are terminal: synthetic
padding after a missing proposal has no exact proposal ancestry to validate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

LABEL_SOURCE = "cloned_native_verifier_sampler_at_actual_proposal_prefix"
VERIFIER_LOGITS_SOURCE = "raw_target_verifier_at_exact_proposal_prefix"
INVALID_REASONS = frozenset({"padding", "pruned", "eos", "unreached"})


@dataclass(frozen=True)
class RoundAnchor:
    prompt_id: str
    split: str
    round_index: int
    prefix_token_ids: tuple[int, ...]
    seed_token_id: int


@dataclass(frozen=True)
class TraceAudit:
    # Labels are draft indices; -1 means unsupported or invalid.
    draft_labels: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    supported_mask: tuple[bool, ...]
    ce_mask: tuple[bool, ...]
    # The accuracy/coverage denominator includes unsupported valid labels.
    denominator_mask: tuple[bool, ...]
    target_to_draft: tuple[int, ...]
    counts: Mapping[str, int]
    per_depth: Mapping[int, Mapping[str, int]]
    mapped_probability_mass: tuple[float | None, ...] = ()
    reached_mask: tuple[bool, ...] = ()


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _token(value: object, name: str, size: int) -> int:
    token = _integer(value, name)
    if token >= size:
        raise ValueError(f"{name} outside target vocabulary")
    return token


def _prefix(value: object, name: str, size: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty token sequence")
    return tuple(_token(token, name, size) for token in value)


def validate_offset_d2t(
    offsets: Sequence[int], target_vocab_size: int, draft_vocab_size: int = 32_000
) -> tuple[int, ...]:
    """Return inverse map after verifying native ``target_id = i + d2t[i]``.

    The native d2t tensor stores offsets, unlike preparation manifests that
    may store absolute target IDs. Callers must pass the native offsets here.
    ``draft_vocab_size`` can be reduced only for small synthetic tests.
    """
    _integer(target_vocab_size, "target_vocab_size", 1)
    _integer(draft_vocab_size, "draft_vocab_size", 1)
    if len(offsets) != draft_vocab_size:
        raise ValueError("d2t length differs from draft vocabulary size")
    reverse = [-1] * target_vocab_size
    for draft_id, offset in enumerate(offsets):
        if isinstance(offset, bool) or not isinstance(offset, Integral):
            raise ValueError("d2t offsets must be integers")
        target_id = draft_id + offset
        if not 0 <= target_id < target_vocab_size:
            raise ValueError("d2t target ID outside target vocabulary")
        if reverse[target_id] != -1:
            raise ValueError("d2t target IDs are not unique")
        reverse[target_id] = draft_id
    return tuple(reverse)


def validate_recurrent_trace(
    rows: Sequence[Mapping[str, object]],
    anchors: Sequence[RoundAnchor],
    *,
    offsets: Sequence[int],
    target_vocab_size: int,
    allowed_prompt_ids: Collection[str],
    split: str,
    draft_vocab_size: int = 32_000,
    max_depth: int = 5,
) -> TraceAudit:
    """Audit native-style proposal rows and return training masks and labels.

    Rows may be interleaved across rounds but must be ordered by depth within
    each round. Each round starts at depth zero and its first prefix is exactly
    ``anchor.prefix_token_ids + (anchor.seed_token_id,)``. The next depth must
    append the previous *proposal*, including when its verifier label differs.
    A terminal invalid row has no label or proposal and cannot have a child.
    """
    reverse = validate_offset_d2t(offsets, target_vocab_size, draft_vocab_size)
    _integer(max_depth, "max_depth", 1)
    if not split or not allowed_prompt_ids:
        raise ValueError("split and allowed_prompt_ids must be explicit")
    allowed = set(allowed_prompt_ids)
    if len(allowed) != len(allowed_prompt_ids):
        raise ValueError("duplicate allowed prompt IDs")
    by_key: dict[tuple[str, int], RoundAnchor] = {}
    for anchor in anchors:
        if anchor.split != split or anchor.prompt_id not in allowed:
            raise ValueError("anchor prompt or split outside requested capture")
        round_index = _integer(anchor.round_index, "anchor round_index")
        prefix = _prefix(anchor.prefix_token_ids, "anchor prefix", target_vocab_size)
        _token(anchor.seed_token_id, "anchor seed", target_vocab_size)
        key = (anchor.prompt_id, round_index)
        if key in by_key:
            raise ValueError("duplicate round anchor")
        by_key[key] = RoundAnchor(
            anchor.prompt_id, split, round_index, prefix, anchor.seed_token_id
        )
    if not rows or not by_key:
        raise ValueError("trace rows and round anchors must be nonempty")

    previous: dict[tuple[str, int], Mapping[str, object]] = {}
    seen_keys: set[tuple[str, int, int]] = set()
    labels: list[int] = []
    valid_mask: list[bool] = []
    supported_mask: list[bool] = []
    reached_mask: list[bool] = []
    depth_counts: dict[int, Counter[str]] = {}
    totals: Counter[str] = Counter()
    mapped_mass: list[float | None] = []
    for row in rows:
        prompt, row_split = row.get("prompt_id"), row.get("split")
        if not isinstance(prompt, str) or prompt not in allowed or row_split != split:
            raise ValueError("row prompt or split outside requested capture")
        round_index = _integer(row.get("round_index"), "round_index")
        depth = _integer(row.get("depth"), "depth")
        if depth >= max_depth:
            raise ValueError("depth outside recurrent horizon")
        key = (prompt, round_index)
        anchor = by_key.get(key)
        if anchor is None:
            raise ValueError("row has no matching round anchor")
        row_key = (*key, depth)
        if row_key in seen_keys:
            raise ValueError("duplicate prompt/round/depth row")
        seen_keys.add(row_key)

        parent = _integer(row.get("parent_position"), "parent_position")
        expected_parent = len(anchor.prefix_token_ids) - 1
        if parent != expected_parent:
            raise ValueError("parent position differs from round anchor")
        if (
            row.get("alignment_valid") is not True
            or row.get("is_bonus") is not False
            or row.get("forced", False) is not False
            or _integer(row.get("verifier_row"), "verifier_row") != depth
            or _integer(row.get("input_position"), "input_position") != parent + depth + 1
            or _integer(row.get("label_position"), "label_position") != parent + depth + 2
        ):
            raise ValueError("state/verifier position or alignment mismatch")
        prefix = _prefix(row.get("prefix_token_ids"), "row prefix", target_vocab_size)
        if (
            len(prefix) != row["label_position"]
            or _token(row.get("input_token_id"), "input_token_id", target_vocab_size) != prefix[-1]
        ):
            raise ValueError("row prefix/input/label position mismatch")
        prior = previous.get(key)
        if prior is None:
            if depth != 0 or prefix != anchor.prefix_token_ids + (anchor.seed_token_id,):
                raise ValueError("round starts at wrong depth or prefix")
        else:
            if depth != prior["depth"] + 1:
                raise ValueError("round depths are not contiguous")
            prior_proposal = prior.get("proposed_token_id")
            if prior.get("valid") is not True or not isinstance(prior_proposal, Integral):
                raise ValueError("row follows invalid or proposal-free ancestor")
            if prefix != tuple(prior["prefix_token_ids"]) + (prior_proposal,):
                raise ValueError("row prefix does not follow proposed-token ancestry")
        previous[key] = row

        valid = row.get("valid")
        if type(valid) is not bool:
            raise ValueError("valid mask must be explicit bool")
        reached = row.get("verifier_reached")
        if type(reached) is not bool:
            raise ValueError("verifier_reached mask must be explicit bool")
        reason = row.get("invalid_reason")
        if valid:
            if reason is not None:
                raise ValueError("valid row has invalid_reason")
            if row.get("label_source") != LABEL_SOURCE:
                raise ValueError("label is not from exact-prefix native verifier")
            target_id = _token(row.get("verifier_token_id"), "verifier_token_id", target_vocab_size)
            _token(row.get("proposed_token_id"), "proposed_token_id", target_vocab_size)
            draft_label = reverse[target_id]
            supported = draft_label >= 0
            if row.get("label_supported") is not supported:
                raise ValueError("label_supported conflicts with d2t inverse")
        else:
            if reason not in INVALID_REASONS:
                raise ValueError("invalid row needs padding/pruned/eos/unreached reason")
            if (
                row.get("verifier_token_id") is not None
                or row.get("proposed_token_id") is not None
                or row.get("label_supported") is not False
            ):
                raise ValueError("invalid row must have no label or proposal")
            draft_label, supported = -1, False

        logits = row.get("verifier_logits")
        probability_mass = None
        if logits is not None:
            if row.get("verifier_logits_source") != VERIFIER_LOGITS_SOURCE:
                raise ValueError("full logits are not raw target verifier logits")
            if (
                not valid
                or not isinstance(logits, (list, tuple))
                or len(logits) != target_vocab_size
            ):
                raise ValueError("verifier logits require a valid full target-vocabulary row")
            if any(
                isinstance(x, bool) or not isinstance(x, Real) or math.isnan(x) or x == math.inf
                for x in logits
            ):
                raise ValueError("invalid verifier logits")
            maximum = max(logits)
            if maximum == -math.inf:
                raise ValueError("verifier logits have zero probability mass")
            exponentials = [math.exp(value - maximum) for value in logits]
            total_mass = sum(exponentials)
            probability_mass = (
                sum(
                    exponentials[token_id]
                    for token_id, draft_id in enumerate(reverse)
                    if draft_id >= 0
                )
                / total_mass
            )
            totals["logit_rows"] += 1
        mapped_mass.append(probability_mass)
        labels.append(draft_label)
        valid_mask.append(valid)
        supported_mask.append(supported)
        reached_mask.append(reached)
        bucket = depth_counts.setdefault(depth, Counter())
        for counter in (totals, bucket):
            counter["total"] += 1
            if reached:
                counter["verifier_reached"] += 1
            if valid:
                counter["valid"] += 1
                counter["supported" if supported else "unsupported"] += 1
            else:
                counter["invalid"] += 1
                counter[f"invalid_{reason}"] += 1

    if set(previous) != set(by_key):
        raise ValueError("round anchor has no trace rows")
    ce_mask = tuple(valid and supported for valid, supported in zip(valid_mask, supported_mask))
    return TraceAudit(
        tuple(labels),
        tuple(valid_mask),
        tuple(supported_mask),
        ce_mask,
        tuple(valid_mask),
        reverse,
        dict(totals),
        {depth: dict(counts) for depth, counts in sorted(depth_counts.items())},
        tuple(mapped_mass),
        tuple(reached_mask),
    )
