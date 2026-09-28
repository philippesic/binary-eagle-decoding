#!/usr/bin/env python3
"""Probe whether real depth-1 CE reaches depth-0 student state and K/V.

This CPU-only diagnostic uses an audited one-prompt capture and frozen
candidate-D initialization. It takes no optimizer step, writes no checkpoint,
and makes no trained-model or numerical-parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import load_audited_capture, sha256  # noqa: E402
from check_recurrent_real_step import _build_drafter  # noqa: E402
from run_binary_head_capture import (  # noqa: E402
    DRAFT_D_D2T_SHA256,
    DRAFT_D_SHA256,
    TARGET_F16_SHA256,
)

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_step import NativeStepCache  # noqa: E402
from w1a1_eagle.recurrent_rollout import (  # noqa: E402
    DraftStep,
    rebuild_prefix_cache,
    rollout_captured_prefix,
)
from w1a1_eagle.recurrent_trace import TraceAudit, validate_recurrent_trace  # noqa: E402

DIRECT_CE_MASK = (False, True)
ARITHMETIC = "group_matmul"


def depth_one_label(rows: tuple[dict, ...], trace: TraceAudit) -> int:
    """Require two exact-prefix, supported proposals; return the later label."""
    if len(rows) != 2 or tuple(row.get("depth") for row in rows) != (0, 1):
        raise ValueError("diagnostic requires exactly depth-0 and depth-1 proposal rows")
    if (
        len(trace.draft_labels) != 2
        or trace.valid_mask != (True, True)
        or trace.supported_mask != (True, True)
        or trace.ce_mask != (True, True)
        or any(type(label) is not int or label < 0 for label in trace.draft_labels)
    ):
        raise ValueError("first two proposal labels must be valid and supported")
    if DIRECT_CE_MASK != (False, True):
        raise ValueError("earlier direct-loss mask must be zero")
    return trace.draft_labels[1]


def depth_one_ce(logits: Tensor, label: int, direct_mask: tuple[bool, bool]) -> Tensor:
    """CE on the later row only, independent of the trace's trainable mask."""
    if direct_mask != (False, True):
        raise ValueError("depth-0 direct loss is prohibited")
    if (
        logits.ndim != 2
        or logits.shape[0] != 2
        or logits.device.type != "cpu"
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
        or type(label) is not int
        or not 0 <= label < logits.shape[1]
    ):
        raise ValueError("later CE requires two finite CPU logit rows and a supported label")
    return F.cross_entropy(logits[1:2].float(), torch.tensor([label], device="cpu"))


def _gradient_stats(grad: Tensor | None, name: str, *, require_nonzero: bool) -> dict:
    if grad is None or not bool(torch.isfinite(grad).all()):
        raise ValueError(f"{name} gradient is missing or nonfinite")
    nonzero = int(torch.count_nonzero(grad))
    if require_nonzero and not nonzero:
        raise ValueError(f"{name} gradient is zero")
    grad64 = grad.detach().to(torch.float64)
    return {
        "shape": list(grad.shape),
        "nonzero": nonzero,
        "l2": float(torch.linalg.vector_norm(grad64)),
        "max_abs": float(grad64.abs().max()),
    }


def _tensor_gradient(tensor: Tensor, name: str, *, require_nonzero: bool) -> dict:
    return _gradient_stats(tensor.grad, name, require_nonzero=require_nonzero)


def later_only_backward(
    rows: tuple[dict, dict],
    seed_raw_features: Tensor,
    adapter: Any,
    initial_cache: NativeStepCache,
    label: int,
    *,
    draft_vocab_size: int,
) -> dict:
    """Backpropagate depth-1 CE and audit the depth-0 causal derivatives."""
    first: DraftStep | None = None
    first_cache_length = initial_cache.key.shape[1]

    def watched_decode(token: int, feature: Tensor, position: int, cache: Any) -> DraftStep:
        nonlocal first
        result = adapter.decode_step(token, feature, position, cache)
        if first is None:
            if not isinstance(result.cache, NativeStepCache):
                raise ValueError("depth-0 decoder must return a native K/V cache")
            if (
                result.cache.key.shape[1] != first_cache_length + 1
                or result.cache.value.shape[1] != first_cache_length + 1
            ):
                raise ValueError("depth-0 decoder did not append exactly one K/V row")
            for name, value in (
                ("pre_norm", result.pre_norm),
                ("key cache", result.cache.key),
                ("value cache", result.cache.value),
                ("logits", result.logits),
            ):
                if not value.requires_grad:
                    raise ValueError(f"depth-0 {name} has no autograd path")
                value.retain_grad()
            first = result
        return result

    logits = rollout_captured_prefix(
        rows,
        seed_raw_features,
        encode_feature=adapter.encode_feature,
        decode_step=watched_decode,
        initial_cache=initial_cache,
        draft_vocab_size=draft_vocab_size,
    )
    if first is None:
        raise ValueError("depth-0 decoder was not called")
    loss = depth_one_ce(logits, label, DIRECT_CE_MASK)
    if not math.isfinite(float(loss.detach())):
        raise ValueError("later CE is nonfinite")
    loss.backward()
    first_logit_gradient = first.logits.grad
    if first_logit_gradient is not None and bool(torch.count_nonzero(first_logit_gradient)):
        raise ValueError("depth-0 logits unexpectedly received direct-loss gradient")
    if first_logit_gradient is not None and not bool(torch.isfinite(first_logit_gradient).all()):
        raise ValueError("depth-0 logit gradient is nonfinite")
    cache = first.cache
    return {
        "loss": float(loss.detach()),
        "direct_ce_mask": list(DIRECT_CE_MASK),
        "depth_0_logits_gradient_nonzero": 0,
        "cache_positions_before": first_cache_length,
        "cache_positions_after_depth_0": cache.key.shape[1],
        "depth_0_pre_norm": _tensor_gradient(
            first.pre_norm, "depth-0 pre_norm", require_nonzero=True
        ),
        "depth_0_key_cache": _tensor_gradient(cache.key, "depth-0 K cache", require_nonzero=True),
        "depth_0_value_cache": _tensor_gradient(
            cache.value, "depth-0 V cache", require_nonzero=True
        ),
        "depth_0_appended_key": _gradient_stats(
            cache.key.grad[:, -1, :], "depth-0 appended K", require_nonzero=True
        ),
        "depth_0_appended_value": _gradient_stats(
            cache.value.grad[:, -1, :], "depth-0 appended V", require_nonzero=True
        ),
    }


def _linear_gradients(linears: Any) -> dict:
    if len(linears) != 9:
        raise ValueError("diagnostic needs all nine binary projections")
    gradients = {}
    for name, module in linears.items():
        gradients[name] = {
            "latent_sign": _tensor_gradient(
                module.latent_sign, f"{name} sign", require_nonzero=False
            ),
            "scale_offset": _tensor_gradient(
                module.scale_offset, f"{name} scale", require_nonzero=False
            ),
        }
    return gradients


def _pinned_absolute_map_hash(offsets: np.ndarray) -> str:
    if offsets.shape != (32_000,) or offsets.dtype.kind not in "iu":
        raise ValueError("capture D offsets must be a 32,000-element integer vector")
    absolute = np.arange(32_000, dtype="<i8") + offsets.astype("<i8")
    digest = hashlib.sha256(absolute.tobytes()).hexdigest()
    if digest != DRAFT_D_D2T_SHA256:
        raise ValueError("capture vocabulary map differs from pinned candidate D")
    return digest


def _cpu_hardware() -> dict[str, str]:
    name = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            name = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
                or name
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"processor": name, "machine": platform.machine(), "platform": platform.platform()}


def diagnose(
    bundle_manifest: Path,
    train_prompts: Path,
    target_gguf: Path,
    candidate_d: Path,
    config: Path,
    report_path: Path,
) -> dict:
    if report_path.exists():
        raise ValueError("diagnostic report path must be new")
    start = time.monotonic()
    sources = {
        "bundle_manifest": sha256(bundle_manifest),
        "train_prompts": sha256(train_prompts),
        "target_gguf": sha256(target_gguf),
        "candidate_d": sha256(candidate_d),
        "drafter_config": sha256(config),
    }
    if sources["target_gguf"] != TARGET_F16_SHA256 or sources["candidate_d"] != DRAFT_D_SHA256:
        raise ValueError("target and candidate D must match pinned FP16/D hashes")
    capture = load_audited_capture(
        bundle_manifest, train_prompts, sources["train_prompts"], expected_prompt_count=1
    )
    if len({key[0] for key in capture.anchors}) != 1:
        raise ValueError("diagnostic requires one captured training prompt")
    prompt_id = next(iter(capture.anchors))[0]
    if not prompt_id.startswith("qat-revisit-train-") or (prompt_id, 0) not in capture.anchors:
        raise ValueError("diagnostic requires the first frozen training round")
    round0 = capture.round_inputs(prompt_id, 0)
    rows = round0.rows[:2]
    if len(rows) != 2:
        raise ValueError("diagnostic requires two proposal positions")
    manifest = json.loads(bundle_manifest.read_text())
    offsets_record = manifest["offsets"]
    offsets = np.load(bundle_manifest.parent / offsets_record["path"], allow_pickle=False)
    map_hash = _pinned_absolute_map_hash(offsets)
    trace = validate_recurrent_trace(
        rows,
        [round0.anchor],
        offsets=offsets,
        target_vocab_size=151_936,
        allowed_prompt_ids={prompt_id},
        split="train",
        draft_vocab_size=32_000,
        max_depth=5,
    )
    label = depth_one_label(rows, trace)
    operands = FrozenOperands(target_gguf, candidate_d)
    adapter = _build_drafter(config, candidate_d, operands, ARITHMETIC)
    rebuilt = rebuild_prefix_cache(
        round0.prefix_token_ids,
        round0.raw_target_features,
        round0.feature_positions,
        parent_position=len(round0.anchor.prefix_token_ids) - 1,
        encode_feature=adapter.encode_feature,
        decode_context=adapter.decode_context,
        new_cache=adapter.new_cache,
    )
    causal = later_only_backward(
        rows,
        rebuilt.seed_raw_features,
        adapter,
        rebuilt.cache,
        label,
        draft_vocab_size=32_000,
    )
    report = {
        "schema": "recurrent_real_later_only_cpu_gradient_v1",
        "status": "later_only_causal_gradient_verified",
        "training_eligible": False,
        "device": "cpu",
        "hardware": _cpu_hardware(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "arithmetic": ARITHMETIC,
        "cache_write_dtype": "float16_rounded_in_float32_storage",
        "optimizer_steps": 0,
        "prompt_id": prompt_id,
        "round_index": 0,
        "positions": 2,
        "draft_labels": list(trace.draft_labels),
        "verifier_target_labels": [row["verifier_token_id"] for row in rows],
        "input_tokens": [row["input_token_id"] for row in rows],
        "verifier_reached_mask": list(trace.reached_mask),
        "valid_mask": list(trace.valid_mask),
        "supported_mask": list(trace.supported_mask),
        "audited_ce_mask": list(trace.ce_mask),
        "causal": causal,
        "nine_linear_gradients": _linear_gradients(adapter.linears),
        "source_sha256": {**sources, **capture.report["source_sha256"]},
        "candidate_d_absolute_map_sha256": map_hash,
        "elapsed_seconds": time.monotonic() - start,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.bundle_manifest,
        args.train_prompts,
        args.target_gguf,
        args.candidate_d,
        args.config,
        args.report,
    )
    print(json.dumps({"status": report["status"], "loss": report["causal"]["loss"]}))


if __name__ == "__main__":
    main()
