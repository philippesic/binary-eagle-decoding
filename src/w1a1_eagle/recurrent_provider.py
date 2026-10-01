"""Audited native-prefix provider boundary for joint EAGLE W1Ax training.

Providers own model/capture loading, file hash verification and source policy.
This module never reads a model or capture on import. The Torch path reconstructs
the current student cache and unrolls exact proposal prefixes without detaching
the proposal state or K/V graph. Accelerator rollout needs a separate provider
implementation and is deliberately not exercised by Phase 1A checks.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
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
CALIBRATION_ONLY_SCOPE = "row_a16_hard_ce_100_steps"


def calibration_measurement_metadata(device: str) -> dict[str, object]:
    """Describe the actual runtime source for calibration-only measurements."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration instrumentation requires an available CUDA device")
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_name = torch.cuda.get_device_name(index)
        capability = [int(properties.major), int(properties.minor)]
    else:
        device_name = "CPU"
        capability = None
    return {
        "scope": CALIBRATION_ONLY_SCOPE,
        "device": str(torch_device),
        "device_name": device_name,
        "compute_capability": capability,
        "cuda_memory_source": (
            "torch CUDA allocator peak allocated/reserved bytes since per-step reset; "
            "not whole-device usage"
        ),
        "process_memory_source": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss process-lifetime high-water mark"
        ),
        "process_memory_units": "bytes",
        "process_memory_raw_unit_conversion": (
            "ru_maxrss is KiB on Linux and bytes on macOS; unsupported platforms report null"
        ),
        "timing_scope": "feature transfer, prefix rebuild, forward, backward, and optimizer step",
        "timing_synchronization": (
            "CUDA synchronized immediately before timing and after optimizer step"
        ),
    }


def _process_peak_rss_bytes() -> int | None:
    """Return process-lifetime peak RSS in bytes across Linux and macOS."""
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    system = platform.system()
    if system == "Linux":
        return peak * 1024
    if system == "Darwin":
        return peak
    return None


def _sync_calibration_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    batch: ProviderRound,
    adapter: StepAdapter,
    draft_vocab_size: int,
    *,
    optimize_cache: bool = False,
    optimize_head: bool = False,
    context_chunk_size: int = 64,
) -> Tensor:
    """Rebuild context and retain proposal-state/K/V autograd links.

    Optional capabilities enable single-layer K/V-only reconstruction and one
    head call over a captured chain. Generic/diagnostic adapters fall back to
    serial execution. Reference controls change no optimizer/update cadence.
    A wrapper's decode_step remains the proposal observation boundary.
    """
    if not batch.rows:
        raise ValueError("one nonempty captured round is required")
    if type(optimize_cache) is not bool or type(optimize_head) is not bool:
        raise ValueError("optimization controls must be boolean")
    if type(context_chunk_size) is not int or context_chunk_size < 1:
        raise ValueError("context chunk size must be positive")
    build_context = getattr(adapter, "build_context_cache", None)
    if not (
        optimize_cache
        and getattr(adapter, "supports_context_cache", False)
        and callable(build_context)
    ):
        build_context = None
    context_builder = (
        (lambda tokens, raw: build_context(tokens, raw, chunk_size=context_chunk_size))
        if build_context is not None
        else None
    )
    head = getattr(adapter, "decode_head", None)
    if not (optimize_head and getattr(adapter, "supports_batched_head", False) and callable(head)):
        head = None
    decode_step = adapter.decode_step
    if head is not None:
        # Call the public wrapper, not an underlying decode-body method: an
        # ObservedAdapter must still see the attached first state and K/V.
        def decode_step(*args):
            return adapter.decode_step(*args, compute_logits=False)

    rebuilt = rebuild_prefix_cache(
        batch.prefix_token_ids,
        batch.raw_target_features,
        batch.feature_positions,
        parent_position=len(batch.anchor.prefix_token_ids) - 1,
        encode_feature=adapter.encode_feature,
        decode_context=adapter.decode_context,
        new_cache=adapter.new_cache,
        build_context_cache=context_builder,
    )
    if (
        batch.rows[0].get("input_token_id") != rebuilt.seed_token
        or batch.rows[0].get("parent_position") != rebuilt.decoder_position
    ):
        raise ValueError("proposal seed token/position differs from rebuilt accepted prefix")
    return rollout_captured_prefix(
        batch.rows,
        rebuilt.seed_raw_features,
        encode_feature=adapter.encode_feature,
        decode_step=decode_step,
        initial_cache=rebuilt.cache,
        draft_vocab_size=draft_vocab_size,
        decode_head=head,
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
    from .qat_readiness import optimization_requires_receipt, require_measured_cuda_readiness

    if optimization_requires_receipt(config):
        if getattr(provider, "readiness_scope", None) == CALIBRATION_ONLY_SCOPE:
            raise ValueError("frozen A16 calibration cannot admit a new optimization recipe")
        if torch.device(config.device).type == "cuda":
            from .continuous_runtime import training_runtime_identity

            source = hashlib.sha256(
                json.dumps(provider.source_metadata, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            require_measured_cuda_readiness(
                config,
                source_sha256=source,
                runtime_identity=training_runtime_identity(config.device),
            )
    drafter, target = provider.load_models()
    linears = install_joint_linears(drafter, target, config, candidate_d=provider.candidate_d)
    adapter = provider.make_step_adapter(drafter)
    if set(adapter.linears) != set(linears) or any(
        adapter.linears[path] is not linears[path] for path in linears
    ):
        raise ValueError("step adapter does not use installed joint linears")
    optimizer = joint_optimizer(linears, config)
    calibration_metrics = getattr(provider, "readiness_scope", None) == CALIBRATION_ONLY_SCOPE
    measurement_device = torch.device(config.device)
    metrics = []
    for batch in provider.rounds():
        audit = audit_provider_round(batch, provider)
        teacher = (
            bind_teacher_rows(batch, audit) if config.objective == "compact_probability" else None
        )
        if calibration_metrics:
            _sync_calibration_device(measurement_device)
            if measurement_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(measurement_device)
            started = time.perf_counter()
        device_batch = replace(
            batch, raw_target_features=batch.raw_target_features.to(config.device)
        )
        logits = forward_torch_round(
            device_batch,
            adapter,
            provider.draft_vocab_size,
            optimize_cache=config.optimize_cache,
            optimize_head=config.optimize_head,
            context_chunk_size=config.context_chunk_size,
        )
        item = joint_train_step(linears, logits, audit, optimizer, config, teacher=teacher)
        if calibration_metrics:
            _sync_calibration_device(measurement_device)
            item.update(
                {
                    "calibration_step_wall_seconds": time.perf_counter() - started,
                    "calibration_cuda_allocator_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(measurement_device))
                        if measurement_device.type == "cuda"
                        else None
                    ),
                    "calibration_cuda_allocator_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(measurement_device))
                        if measurement_device.type == "cuda"
                        else None
                    ),
                    "calibration_process_peak_rss_bytes": _process_peak_rss_bytes(),
                }
            )
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
