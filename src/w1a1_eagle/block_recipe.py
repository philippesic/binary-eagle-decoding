"""Explicit DSpark time schedule, transaction reservations and compact history."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BlockRunPolicy:
    effective_batch_blocks: int = 2
    microbatch_blocks: int = 1
    warmup_seconds: float = 864.0
    total_seconds: float = 43200.0
    final_lr_ratio: float = 0.1
    checkpoint_seconds: float = 900.0

    def __post_init__(self):
        if self.effective_batch_blocks != 2 or self.microbatch_blocks != 1:
            raise ValueError("approved block recipe requires microbatch1 accumulation2")
        if (
            self.warmup_seconds,
            self.total_seconds,
            self.final_lr_ratio,
            self.checkpoint_seconds,
        ) != (864.0, 43200.0, 0.1, 900.0):
            raise ValueError("approved 12h schedule/cadence differs")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def schedule_ratio(policy, elapsed):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("schedule requires finite cumulative trainer seconds")
    if elapsed < policy.warmup_seconds:
        return elapsed / policy.warmup_seconds
    progress = min(
        1.0, (elapsed - policy.warmup_seconds) / (policy.total_seconds - policy.warmup_seconds)
    )
    return (
        policy.final_lr_ratio + (1 - policy.final_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


def schedule_state(policy, elapsed):
    return {
        "schema": "block_time_schedule_v1",
        "policy": asdict(policy),
        "applied_seconds": float(elapsed),
        "ratio": schedule_ratio(policy, elapsed),
    }


def validate_schedule(state, elapsed):
    if not isinstance(state, dict) or set(state) != {
        "schema",
        "policy",
        "applied_seconds",
        "ratio",
    }:
        raise ValueError("exact persisted optimizer schedule required")
    policy = BlockRunPolicy(**state["policy"])
    if (
        state != schedule_state(policy, state["applied_seconds"])
        or state["applied_seconds"] > elapsed
    ):
        raise ValueError("persisted optimizer schedule differs from trainer clock")
    return state["ratio"]


def apply_schedule(optimizer, config, state, elapsed):
    ratio = validate_schedule(state, elapsed)
    for group in optimizer.param_groups:
        group["lr"] = {"sign": config.sign_lr, "scale": config.scale_lr}[group["family"]] * ratio


def update_history(history, metrics, *, step, elapsed, chains, groups):
    """Persist bounded rolling observations and exact cumulative components."""
    value = copy.deepcopy(history or {})
    totals = value.setdefault(
        "totals",
        {
            "updates": 0,
            "weighted_denominator": 0.0,
            "ce_numerator": 0.0,
            "l1_numerator": 0.0,
            "loss_numerator": 0.0,
            "clipped_updates": 0,
            "sign_flips": 0,
        },
    )
    totals["updates"] += 1
    denominator = metrics["weighted_denominator"]
    totals["weighted_denominator"] += denominator
    for field, metric in (
        ("ce_numerator", "ce"),
        ("l1_numerator", "probability_l1"),
        ("loss_numerator", "loss"),
    ):
        totals[field] += metrics[metric] * denominator
    totals["clipped_updates"] += int(metrics["clipped"])
    totals["sign_flips"] += metrics["sign_flips"]
    value["unique_chains"] = sorted(set(value.get("unique_chains", [])) | set(chains))
    value["unique_groups"] = sorted(set(value.get("unique_groups", [])) | set(groups))
    record = {"step": step, "elapsed_seconds": elapsed, **metrics}
    value["recent"] = [*value.get("recent", []), record][-60:]
    samples = value.setdefault("samples", [])
    if not samples or int(elapsed / 300) > int(samples[-1]["elapsed_seconds"] / 300):
        samples.append(record)
        value["samples"] = samples[-160:]
    return value
