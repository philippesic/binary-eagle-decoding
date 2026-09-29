"""Audited native-prefix provider boundary for joint EAGLE W1Ax training.

Providers own model/capture loading, file hash verification and source policy.
This module never reads a model or capture on import. The Torch path reconstructs
the current student cache and unrolls exact proposal prefixes without detaching
the proposal state or K/V graph. Accelerator rollout needs a separate provider
implementation and is deliberately not exercised by Phase 1A checks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from .recurrent_qat import JointQATConfig, install_joint_linears, joint_optimizer, joint_train_step
from .recurrent_rollout import rebuild_prefix_cache, rollout_captured_prefix
from .recurrent_trace import RoundAnchor, TraceAudit, validate_recurrent_trace

TEACHER_FIELDS = ("draft_topk_ids", "draft_topk_probs", "draft_tail_mass", "outside_draft_mass")


@dataclass(frozen=True)
class ProviderRound:
    """One accepted-prefix round and optionally its verified compact teacher.

    `teacher_metadata` and `teacher_arrays` must originate from
    `iter_verified_shards(..., expected_prefixes=...)`; the provider binds its
    immutable manifest hashes before yielding. Each valid row has one teacher
    ID in current row order. Invalid terminal rows have ID None and no teacher.
    """

    anchor: RoundAnchor
    rows: tuple[Mapping[str, object], ...]
    prefix_token_ids: tuple[int, ...]
    raw_target_features: Tensor
    feature_positions: tuple[int, ...]
    capture_id: str
    teacher_row_ids: tuple[str | None, ...] = ()
    teacher_metadata: tuple[Mapping[str, object], ...] = ()
    teacher_arrays: Mapping[str, np.ndarray] | None = None
    shard_ordinal: int | None = None


class StepAdapter(Protocol):
    linears: Mapping[str, nn.Module]

    def new_cache(self) -> Any: ...

    def encode_feature(self, raw: Tensor) -> Tensor: ...

    def decode_context(self, token: int, feature: Tensor, position: int, cache: Any) -> Any: ...

    def decode_step(self, token: int, feature: Tensor, position: int, cache: Any) -> Any: ...


class JointTrainingProvider(Protocol):
    """Injected loading seam for official drafter, target and audited captures."""

    split: str
    training_eligible: bool
    allowed_prompt_ids: set[str]
    target_vocab_size: int
    draft_vocab_size: int
    max_depth: int
    d2t_offsets: Sequence[int]
    base_gguf_sha256: str
    candidate_d: Mapping[str, tuple[np.ndarray, np.ndarray]] | None

    def load_models(self) -> tuple[nn.Module, nn.Module]: ...

    def make_step_adapter(self, drafter: nn.Module) -> StepAdapter: ...

    def rounds(self) -> Iterable[ProviderRound]: ...


def bind_teacher_rows(batch: ProviderRound, audit: TraceAudit) -> dict[str, Tensor]:
    """Bind verified compact probabilities to *current* exact-prefix rows.

    Hash verification belongs to the provider's data reader. This second gate
    prevents mixing a verified shard with the wrong prompt, capture or changed
    proposal prefix. Invalid rows receive zero mapped mass and are masked out.
    """
    if batch.teacher_arrays is None or set(batch.teacher_arrays) < set(TEACHER_FIELDS):
        raise ValueError("compact teacher arrays are missing")
    if len(batch.teacher_row_ids) != len(batch.rows):
        raise ValueError("teacher row IDs must align with audited trace rows")
    valid = [i for i, flag in enumerate(audit.valid_mask) if flag]
    if len(batch.teacher_metadata) != len(valid):
        raise ValueError("one verified teacher metadata row is required per valid row")
    if not isinstance(batch.capture_id, str) or not batch.capture_id:
        raise ValueError("captured teacher source ID is required")
    arrays = batch.teacher_arrays
    if (
        not isinstance(arrays["draft_topk_ids"], np.ndarray)
        or arrays["draft_topk_ids"].dtype != np.int32
        or arrays["draft_topk_ids"].ndim != 2
        or arrays["draft_topk_ids"].shape[0] != len(valid)
        or not isinstance(arrays["draft_topk_probs"], np.ndarray)
        or arrays["draft_topk_probs"].dtype != np.float32
        or arrays["draft_topk_probs"].shape != arrays["draft_topk_ids"].shape
        or any(
            not isinstance(arrays[key], np.ndarray)
            or arrays[key].dtype != np.float32
            or arrays[key].shape != (len(valid),)
            for key in ("draft_tail_mass", "outside_draft_mass")
        )
    ):
        raise ValueError("compact teacher arrays violate pinned NPZ dtypes or shapes")
    if "next_target_id" in arrays and len(arrays["next_target_id"]) != len(valid):
        raise ValueError("teacher label rows differ from valid proposals")
    topk = arrays["draft_topk_ids"].shape[1]
    if topk < 1:
        raise ValueError("compact teacher needs positive top-k")
    ids_seen = set()
    for teacher_index, row_index in enumerate(valid):
        current = batch.rows[row_index]
        metadata = batch.teacher_metadata[teacher_index]
        row_id = batch.teacher_row_ids[row_index]
        prefix = current["prefix_token_ids"]
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in ids_seen
            or metadata.get("id") != row_id
            or metadata.get("prompt_id") != current["prompt_id"]
            or metadata.get("capture_id") != batch.capture_id
            or metadata.get("prefix_token_ids") != list(prefix)
            or (
                "target_logits_row" in current
                and metadata.get("logits_row") != current["target_logits_row"]
            )
            or (
                "next_target_id" in arrays
                and int(arrays["next_target_id"][teacher_index]) != current["verifier_token_id"]
            )
        ):
            raise ValueError("teacher row ID, label, prompt, capture or exact prefix mismatch")
        ids_seen.add(row_id)
    if any(
        batch.teacher_row_ids[i] is not None for i, flag in enumerate(audit.valid_mask) if not flag
    ):
        raise ValueError("invalid proposal row cannot borrow teacher mass")
    n = len(batch.rows)
    result = {
        "draft_topk_ids": torch.zeros((n, topk), dtype=torch.int32),
        "draft_topk_probs": torch.zeros((n, topk), dtype=torch.float32),
        "draft_tail_mass": torch.zeros(n, dtype=torch.float32),
        "outside_draft_mass": torch.ones(n, dtype=torch.float32),
    }
    for key in TEACHER_FIELDS:
        source = torch.as_tensor(arrays[key])
        result[key][valid] = source.to(result[key].dtype)
    return result


def audit_provider_round(batch: ProviderRound, provider: JointTrainingProvider) -> TraceAudit:
    """Repeat structural trace audit at training time, before a student step."""
    if not isinstance(batch, ProviderRound) or not batch.rows:
        raise ValueError("provider needs one nonempty captured round")
    if tuple(batch.prefix_token_ids) != (
        *batch.anchor.prefix_token_ids,
        batch.anchor.seed_token_id,
    ):
        raise ValueError("accepted-prefix tokens differ from round anchor")
    if any(
        row.get("prompt_id") != batch.anchor.prompt_id
        or row.get("round_index") != batch.anchor.round_index
        for row in batch.rows
    ):
        raise ValueError("provider round crosses prompt or round boundary")
    return validate_recurrent_trace(
        batch.rows,
        [batch.anchor],
        offsets=provider.d2t_offsets,
        target_vocab_size=provider.target_vocab_size,
        allowed_prompt_ids=provider.allowed_prompt_ids,
        split=provider.split,
        draft_vocab_size=provider.draft_vocab_size,
        max_depth=provider.max_depth,
    )


def forward_torch_round(
    batch: ProviderRound, adapter: StepAdapter, draft_vocab_size: int
) -> Tensor:
    """Rebuild context cache, then retain proposal-state/K/V autograd links."""
    rebuilt = rebuild_prefix_cache(
        batch.prefix_token_ids,
        batch.raw_target_features,
        batch.feature_positions,
        parent_position=len(batch.anchor.prefix_token_ids) - 1,
        encode_feature=adapter.encode_feature,
        decode_context=adapter.decode_context,
        new_cache=adapter.new_cache,
    )
    return rollout_captured_prefix(
        batch.rows,
        rebuilt.seed_raw_features,
        encode_feature=adapter.encode_feature,
        decode_step=adapter.decode_step,
        initial_cache=rebuilt.cache,
        draft_vocab_size=draft_vocab_size,
    )


def train_from_provider(
    provider: JointTrainingProvider, config: JointQATConfig, *, max_rounds: int
) -> tuple[Mapping[str, nn.Module], list[dict[str, float | int | str | None]]]:
    """Install nine linears and train on injected audited native-prefix rounds.

    Accelerator execution still requires `JointQATConfig.allow_accelerator`.
    The provider chooses the official model device; this function moves only
    audited native feature rows to that device. Importing does no model work.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if provider.training_eligible is not True:
        raise ValueError("provider capture is not approved for substantive training")
    validate_budget = getattr(provider, "validate_training_budget", None)
    if callable(validate_budget):
        validate_budget(config, max_rounds)
    if provider.split != "train" or not provider.allowed_prompt_ids:
        raise ValueError("provider must declare an eligible training split and prompt set")
    drafter, target = provider.load_models()
    linears = install_joint_linears(drafter, target, config, candidate_d=provider.candidate_d)
    adapter = provider.make_step_adapter(drafter)
    if set(adapter.linears) != set(linears) or any(
        adapter.linears[path] is not linears[path] for path in linears
    ):
        raise ValueError("step adapter does not use installed joint linears")
    optimizer = joint_optimizer(linears, config)
    metrics = []
    for batch in provider.rounds():
        audit = audit_provider_round(batch, provider)
        teacher = (
            bind_teacher_rows(batch, audit) if config.objective == "compact_probability" else None
        )
        device_batch = replace(
            batch, raw_target_features=batch.raw_target_features.to(config.device)
        )
        logits = forward_torch_round(device_batch, adapter, provider.draft_vocab_size)
        item = joint_train_step(linears, logits, audit, optimizer, config, teacher=teacher)
        item.update(
            {
                "prompt_id": batch.anchor.prompt_id,
                "round_index": batch.anchor.round_index,
                "capture_id": batch.capture_id,
                "shard_ordinal": batch.shard_ordinal,
            }
        )
        metrics.append(item)
        if len(metrics) >= max_rounds:
            break
    if not metrics:
        raise ValueError("provider yielded no audited training rounds")
    return linears, metrics
