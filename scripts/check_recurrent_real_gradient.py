#!/usr/bin/env python3
"""Run one CPU diagnostic SGD step on two real captured EAGLE positions.

This is an integration gate for recurrent gradients and checkpoint/export
plumbing. It is not an approved training recipe or a quality experiment.
Output remains outside Git and explicitly ineligible for training claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import load_audited_capture, sha256  # noqa: E402
from check_recurrent_real_step import _build_drafter  # noqa: E402

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.recurrent_rollout import (  # noqa: E402
    rebuild_prefix_cache,
    rollout_captured_prefix,
)
from w1a1_eagle.recurrent_trace import validate_recurrent_trace  # noqa: E402
from w1a1_eagle.recurrent_training import (  # noqa: E402
    save_training_checkpoint,
    train_step,
)


def diagnose(
    bundle_manifest: Path,
    train_prompts: Path,
    target_gguf: Path,
    candidate_d: Path,
    config: Path,
    base_f16_gguf: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise ValueError("diagnostic output directory must be new")
    start = time.monotonic()
    prompt_hash = sha256(train_prompts)
    capture = load_audited_capture(
        bundle_manifest, train_prompts, prompt_hash, expected_prompt_count=1
    )
    if len({key[0] for key in capture.anchors}) != 1:
        raise ValueError("diagnostic requires one captured training prompt")
    prompt_id = next(iter(capture.anchors))[0]
    if not prompt_id.startswith("qat-revisit-train-") or (prompt_id, 0) not in capture.anchors:
        raise ValueError("diagnostic requires the first frozen training round")
    round0 = capture.round_inputs(prompt_id, 0)
    if len(round0.rows) < 2:
        raise ValueError("diagnostic requires two proposal positions")
    rows = round0.rows[:2]
    offsets = np.load(bundle_manifest.parent / "offsets.npy", allow_pickle=False)
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
    operands = FrozenOperands(target_gguf, candidate_d)
    adapter = _build_drafter(config, candidate_d, operands, "group_matmul")
    rebuilt = rebuild_prefix_cache(
        round0.prefix_token_ids,
        round0.raw_target_features,
        round0.feature_positions,
        parent_position=len(round0.anchor.prefix_token_ids) - 1,
        encode_feature=adapter.encode_feature,
        decode_context=adapter.decode_context,
        new_cache=adapter.new_cache,
    )
    logits = rollout_captured_prefix(
        rows,
        rebuilt.seed_raw_features,
        encode_feature=adapter.encode_feature,
        decode_step=adapter.decode_step,
        initial_cache=rebuilt.cache,
        draft_vocab_size=32_000,
    )
    optimizer = torch.optim.SGD(
        [
            parameter
            for module in adapter.linears.values()
            for parameter in (module.latent_sign, module.scale_offset)
        ],
        lr=1e-5,
    )
    loss = train_step(adapter.linears, logits, trace, optimizer)
    gradients = {
        name: {
            "sign_nonzero": int(torch.count_nonzero(module.latent_sign.grad)),
            "scale_nonzero": int(torch.count_nonzero(module.scale_offset.grad)),
            "sign_max_abs": float(module.latent_sign.grad.abs().max()),
            "scale_max_abs": float(module.scale_offset.grad.abs().max()),
        }
        for name, module in adapter.linears.items()
    }
    if not all(
        item["sign_nonzero"] > 0 and item["scale_nonzero"] > 0 for item in gradients.values()
    ):
        raise ValueError("a binary body or head projection received zero gradient")
    output_dir.mkdir(parents=True)
    checkpoint = save_training_checkpoint(
        adapter.linears,
        sha256(base_f16_gguf),
        output_dir / "diagnostic_step.npz",
        output_dir / "checkpoint_manifest.json",
    )
    report = {
        "schema": "recurrent_real_two_step_cpu_gradient_v1",
        "execution_device": "cpu",
        "status": "diagnostic_gradient_and_checkpoint_verified",
        "training_eligible": False,
        "arithmetic": "group_matmul",
        "optimizer": {"type": "SGD", "lr": 1e-5, "steps": 1},
        "prompt_id": prompt_id,
        "positions": 2,
        "loss": loss,
        "draft_labels": list(trace.draft_labels),
        "ce_mask": list(trace.ce_mask),
        "gradients": gradients,
        "checkpoint": checkpoint,
        "bundle_manifest_sha256": sha256(bundle_manifest),
        "elapsed_seconds": time.monotonic() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-f16-gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.bundle_manifest,
        args.train_prompts,
        args.target_gguf,
        args.candidate_d,
        args.config,
        args.base_f16_gguf,
        args.output_dir,
    )
    print(json.dumps({"status": report["status"], "loss": report["loss"]}))


if __name__ == "__main__":
    main()
