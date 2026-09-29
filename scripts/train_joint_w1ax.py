#!/usr/bin/env python3
"""Joint W1Ax trainer with injected audited provider or tiny CPU fixture.

The provider loads official models and audited native-prefix captures only
when --provider is selected. The default fixture checks the optimizer,
quantizer and checkpoint boundary without opening models or final prompts.
Accelerator execution requires an explicit opt-in flag; Phase 1A checks use CPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear  # noqa: E402
from w1a1_eagle.recurrent_provider import train_from_provider  # noqa: E402
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    joint_optimizer,
    joint_train_step,
    save_joint_checkpoint,
)
from w1a1_eagle.recurrent_trace import TraceAudit  # noqa: E402
from w1a1_eagle.recurrent_training import save_training_checkpoint  # noqa: E402


def tiny_joint_fixture(config: JointQATConfig):
    """Three-position toy graph; each of the eight body paths and head is used."""
    torch.manual_seed(config.seed)
    width = 4
    linears = {}
    for path in CANDIDATE_D_BASE_TO_PATH.values():
        weight = (torch.randn(width, width) * 0.2).float()
        scale = torch.full((width,), 0.25)
        if config.contract.scale_layout == "row":
            module = RowBinaryLinear(weight, scale, config.contract)
        else:
            module = GroupedBinaryLinear(weight, scale[:, None], arithmetic="group_matmul")
        linears[path] = module.to(config.device)
    audit = TraceAudit(
        draft_labels=(-1, -1, 1),
        valid_mask=(True, True, True),
        supported_mask=(False, False, True),
        ce_mask=(False, False, True),
        denominator_mask=(True, True, True),
        target_to_draft=(0, 1, 2, 3),
        counts={},
        per_depth={},
    )
    return linears, audit


def tiny_rollout(linears, device: str):
    """Attached cache terms make the last-position loss reach earlier K/V."""
    fc = linears["fc"]
    q = linears["midlayer.self_attn.q_proj"]
    k = linears["midlayer.self_attn.k_proj"]
    v = linears["midlayer.self_attn.v_proj"]
    o = linears["midlayer.self_attn.o_proj"]
    gate = linears["midlayer.mlp.gate_proj"]
    up = linears["midlayer.mlp.up_proj"]
    down = linears["midlayer.mlp.down_proj"]
    head = linears["lm_head"]
    state = torch.tensor([0.1, -0.2, 0.3, -0.1], device=device)
    cache = []
    logits = []
    states = []
    for pos in range(3):
        fused = fc(state + (pos + 1) * 0.02)
        query, key, value = q(fused), k(fused), v(fused)
        key.retain_grad()
        value.retain_grad()
        cache.append((key, value))
        attention = sum((query * old_key).sum() * old_value for old_key, old_value in cache)
        state = torch.tanh(fused + o(attention / (4 * len(cache))) + down(gate(fused) * up(fused)))
        state.retain_grad()
        states.append(state)
        logits.append(head(state))
    return torch.stack(logits), cache, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-bits", type=int, choices=(1, 4, 8, 16), default=16)
    parser.add_argument("--scale-layout", choices=("row", "group128"), default="row")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-accelerator", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--objective", choices=("hard_ce", "compact_probability"), default="hard_ce"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--provider", help="importable MODULE:FACTORY returning a training provider"
    )
    parser.add_argument("--provider-manifest", type=Path)
    parser.add_argument("--base-gguf-sha256", default="0" * 64)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.provider_manifest is not None and args.provider is None:
        parser.error("--provider-manifest requires --provider")
    if args.all_rounds and args.provider is None:
        parser.error("--all-rounds requires --provider")
    config = JointQATConfig(
        W1AxContract(args.activation_bits, args.scale_layout),
        device=args.device,
        allow_accelerator=args.allow_accelerator,
        objective=args.objective,
        seed=args.seed,
    )
    if args.provider:
        module_name, separator, factory_name = args.provider.partition(":")
        if not separator or not module_name or not factory_name:
            parser.error("--provider must be importable MODULE:FACTORY")
        factory = getattr(importlib.import_module(module_name), factory_name)
        provider = (
            factory(config, args.provider_manifest) if args.provider_manifest else factory(config)
        )
        max_rounds = provider.total_rounds if args.all_rounds else args.steps
        validate_budget = getattr(provider, "validate_training_budget", None)
        if callable(validate_budget):
            validate_budget(config, max_rounds, all_rounds=args.all_rounds)
        linears, metrics = train_from_provider(provider, config, max_rounds=max_rounds)
        if args.require_complete and len(metrics) != getattr(provider, "total_rounds", None):
            raise ValueError("provider run did not consume every audited training round")
        training_complete = len(metrics) == getattr(provider, "total_rounds", None)
        base_hash = provider.base_gguf_sha256
        execution = "audited_provider_rounds"
        source = getattr(provider, "source_metadata", None) or {
            "factory": args.provider,
            "split": provider.split,
            "base_gguf_sha256": base_hash,
        }
    else:
        linears, audit = tiny_joint_fixture(config)
        optimizer = joint_optimizer(linears, config)
        metrics = []
        for _ in range(args.steps):
            logits, _, _ = tiny_rollout(linears, config.device)
            teacher = None
            if config.objective == "compact_probability":
                # Synthetic unconditional target masses for interface smoke only.
                teacher = {
                    "draft_topk_ids": torch.tensor([[1, 0]] * 3, dtype=torch.int32),
                    "draft_topk_probs": torch.tensor([[0.5, 0.2]] * 3),
                    "draft_tail_mass": torch.tensor([0.2] * 3),
                    "outside_draft_mass": torch.tensor([0.1] * 3),
                }
            metrics.append(
                joint_train_step(linears, logits, audit, optimizer, config, teacher=teacher)
            )
        base_hash = args.base_gguf_sha256
        execution = "synthetic_fixture"
        source = {"kind": "deterministic_tiny_fixture"}
        max_rounds = args.steps
        training_complete = False
    result = {
        "contract": vars(config.contract),
        "steps": len(metrics),
        "max_rounds_requested": max_rounds,
        "training_complete": training_complete,
        "execution": execution,
        "source": source,
        "metrics": metrics,
        "checkpoint_kind": config.contract.export_status,
        "hardware": "CPU F32 hard-quant simulation" if config.device == "cpu" else config.device,
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint, manifest = args.output_dir / "joint.npz", args.output_dir / "joint.json"
        report = args.output_dir / "training_run.json"
        if report.exists():
            raise FileExistsError(report)
        if args.scale_layout == "row":
            result["saved"] = save_joint_checkpoint(
                linears, config, base_hash, checkpoint, manifest
            )
        else:
            result["saved"] = save_training_checkpoint(linears, base_hash, checkpoint, manifest)
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
