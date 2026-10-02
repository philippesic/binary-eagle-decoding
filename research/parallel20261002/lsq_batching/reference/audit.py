"""Synthetic CPU production-path audit; no model/capture or live source edits."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.recurrent_provider import ProviderRound, forward_torch_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    joint_optimizer,
    shared_round_hard_signs,
)
from w1a1_eagle.recurrent_rollout import DraftStep
from w1a1_eagle.recurrent_trace import RoundAnchor

ROOT = Path(__file__).resolve().parents[4]
CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
ATOL, RTOL = 2e-6, 2e-5


def check_control():
    control = json.loads(CONTROL.read_text())
    if (
        control["research_stop"]
        or control.get("reset_observed")
        or control["last_reset_unix"] != control["initial_reset_unix"]
        or control["last_weekly_used_percent"] >= 99
    ):
        raise RuntimeError("Research stop: checkpoint without running further experiments")


def tiny_native(bits, computation="single_forward"):
    """Reduced native EAGLE geometry, independently initialized synthetic weights."""
    g = torch.Generator().manual_seed(17)
    d = nn.Module()
    d.config = SimpleNamespace(
        pretraining_tp=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=5,
        head_dim=2,
        max_position_embeddings=16,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
    )
    d.early_stop_method, d.tree_mask = None, None
    d.midlayer = nn.Module()
    d.midlayer.self_attn, d.midlayer.mlp = nn.Module(), nn.Module()
    d.embed_tokens = nn.Embedding(5, 4, dtype=torch.float16)
    with torch.no_grad():
        d.embed_tokens.weight.copy_(torch.randn((5, 4), generator=g).half() * 0.3)
    d.embed_tokens.weight.requires_grad_(False)
    shapes = {
        "fc": (4, 12),
        "midlayer.self_attn.q_proj": (4, 8),
        "midlayer.self_attn.k_proj": (2, 8),
        "midlayer.self_attn.v_proj": (2, 8),
        "midlayer.self_attn.o_proj": (4, 4),
        "midlayer.mlp.gate_proj": (5, 4),
        "midlayer.mlp.up_proj": (5, 4),
        "midlayer.mlp.down_proj": (4, 5),
        "lm_head": (3, 4),
    }
    for path in (
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
        "norm",
    ):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(4))
        module.variance_epsilon = 1e-5
        parent, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent) if parent else d, name, module)
    for path, shape in shapes.items():
        module = nn.Linear(shape[1], shape[0], bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.randn(shape, generator=g) * 0.3)
        parent, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent) if parent else d, name, module)
    cfg = JointQATConfig(
        contract=W1AxContract(bits),
        activation_quantization="learned",
        a1_computation=computation,
        activation_lr=0.05,
        binary_optimization={"optimizer": "sgd", "sign_lr": 0.01, "scale_lr": 0.01},
    )
    linears = install_joint_linears(d, nn.Linear(1, 1), cfg)
    with torch.no_grad():
        for q in LearnedActivationBank.from_attached(linears).quantizers.values():
            q.parameter.fill_(0.2 if bits == 1 else 0.63)
    return NativeStepAdapter(d), cfg


def batch(depth, invalid=False, index=0):
    rows = []
    for i in range(depth + int(invalid)):
        valid = i < depth
        rows.append(
            dict(
                prompt_id=f"synthetic-{index}",
                round_index=index,
                parent_position=0,
                depth=i,
                input_position=i + 1,
                input_token_id=1,
                proposed_token_id=1 if valid else None,
                valid=valid,
            )
        )
    raw = torch.tensor(
        [[0.8, -0.12, 0.31, -0.7, 0.2, -0.55, 0.9, -0.41, 0.13, -0.29, 0.66, -0.8]],
        requires_grad=True,
    )
    return ProviderRound(
        RoundAnchor(f"synthetic-{index}", "train", index, (0,), 1),
        tuple(rows),
        (0, 1),
        raw,
        (0,),
        "synthetic-no-capture",
    )


class ControlledAdapter:
    """Actual provider/rollout with repeated attached states to expose exact algebra."""

    supports_batched_head = True

    def __init__(self, native):
        self.native = native
        self.linears = native.linears

    def new_cache(self):
        return ()

    def encode_feature(self, raw):
        # All four coordinates are attached to independently measurable raw features.
        return raw[:4] * torch.tensor([1.25, 0.8333333, 0.7419355, 0.5857143])

    def decode_context(self, token, feature, position, cache):
        return DraftStep(torch.zeros(3), feature, cache)

    def decode_step(self, token, feature, position, cache, *, compute_logits=True):
        return DraftStep(
            self.decode_head(feature[None])[0] if compute_logits else feature.new_empty(0),
            feature,
            (*cache, position),
        )

    def decode_head(self, states):
        return self.native.decode_head(states)


class HeadPolicy:
    """Reference-only dispatch policies. No production source/recipe is mutated."""

    def __init__(self, adapter, policy, valid_depth):
        self.adapter, self.policy, self.valid_depth = adapter, policy, valid_depth
        self.linears = adapter.linears

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def decode_step(self, *args, **kwargs):
        return self.adapter.decode_step(*args, **kwargs)

    def decode_head(self, states):
        if self.policy in ("serial_fallback", "chunk2"):
            size = 1 if self.policy == "serial_fallback" else 2
            return torch.cat([self.adapter.decode_head(s) for s in states.split(size)])
        return self.adapter.decode_head(states)

    def install_chain_domain(self):
        # Existing public low-level API supports the domain. This shim forwards it
        # only in the isolated research object, including serial/chunk invocations.
        quantizer = self.linears["lm_head"].activation_quantizer
        original = quantizer.forward
        count = self.valid_depth * self.linears["lm_head"].in_features

        def forward(input, **kwargs):
            return original(input, normalization_count=count, **kwargs)

        quantizer.forward = forward


def run_case(
    bits,
    depth,
    invalid,
    *,
    controlled=False,
    computation="single_forward",
    policy="serial",
    chain_domain=False,
):
    check_control()
    native, cfg = tiny_native(bits, computation)
    adapter = ControlledAdapter(native) if controlled else native
    b = batch(depth, invalid)
    adapter = HeadPolicy(adapter, policy, depth)
    if chain_domain:
        adapter.install_chain_domain()
    optimizer = joint_optimizer(native.linears, cfg)
    with shared_round_hard_signs(native.linears):
        logits = forward_torch_round(b, adapter, 3, optimize_head=policy != "serial")
        valid = torch.tensor([True] * depth + [False] * int(invalid))
        labels = torch.full((depth,), 2, dtype=torch.long)
        loss = F.cross_entropy(logits[valid], labels, reduction="mean")
        loss.backward()
    grads = {
        name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in native.drafter.named_parameters()
        if p.requires_grad
    }
    # One actual joint_optimizer SGD step: no optimizer state reset or real data.
    optimizer.step()
    params = {
        name: p.detach().clone() for name, p in native.drafter.named_parameters() if p.requires_grad
    }
    return dict(
        logits=logits.detach(),
        loss=loss.detach(),
        raw_vjp=b.raw_target_features.grad,
        grads=grads,
        params=params,
    )


def max_error(a, b):
    return float((a - b).abs().max())


def compare(a, b):
    head = "lm_head.activation_quantizer."
    head_key = next(k for k in a["grads"] if k.startswith(head))
    others = [k for k in a["grads"] if k != head_key]
    ga, gb = float(a["grads"][head_key]), float(b["grads"][head_key])
    return {
        "logits_max_abs": max_error(a["logits"], b["logits"]),
        "loss_abs": max_error(a["loss"], b["loss"]),
        "input_vjp_max_abs": max_error(a["raw_vjp"], b["raw_vjp"]),
        "sign_vjp_max_abs": max(
            max_error(a["grads"][k], b["grads"][k]) for k in others if k.endswith("latent_sign")
        ),
        "scale_vjp_max_abs": max(
            max_error(a["grads"][k], b["grads"][k]) for k in others if k.endswith("scale_offset")
        ),
        "other_activation_vjp_max_abs": max(
            [max_error(a["grads"][k], b["grads"][k]) for k in others if "activation_quantizer" in k]
            or [0.0]
        ),
        "head_gradient_reference": ga,
        "head_gradient_candidate": gb,
        "head_gradient_ratio": ga / gb if abs(gb) > 1e-10 else None,
        "head_parameter_step_abs": max_error(a["params"][head_key], b["params"][head_key]),
        "other_parameter_step_max_abs": max(
            max_error(a["params"][k], b["params"][k]) for k in others
        ),
    }


def run_audit():
    check_control()
    torch.set_num_threads(1)
    results = []
    for controlled in (True, False):
        for computation in ("reference", "single_forward"):
            for bits in (1, 4, 8):
                for depth in (1, 2, 4):
                    for invalid in (False, True):
                        serial = run_case(
                            bits, depth, invalid, controlled=controlled, computation=computation
                        )
                        entry = dict(
                            fixture="repeat" if controlled else "native",
                            bits=bits,
                            depth=depth,
                            invalid_terminal=invalid,
                            computation=computation,
                        )
                        for policy in ("batched", "chunk2", "serial_fallback"):
                            other = run_case(
                                bits,
                                depth,
                                invalid,
                                controlled=controlled,
                                computation=computation,
                                policy=policy,
                            )
                            entry[policy] = compare(serial, other)
                        # Declared alternative domain, using the existing larger-N API.
                        chain_serial = run_case(
                            bits,
                            depth,
                            invalid,
                            controlled=controlled,
                            computation=computation,
                            chain_domain=True,
                        )
                        for policy in ("batched", "chunk2"):
                            other = run_case(
                                bits,
                                depth,
                                invalid,
                                controlled=controlled,
                                computation=computation,
                                policy=policy,
                                chain_domain=True,
                            )
                            entry["chain_domain_" + policy] = compare(chain_serial, other)
                        results.append(entry)
    sources = [
        "learned_activation.py",
        "recurrent_qat.py",
        "recurrent_rollout.py",
        "recurrent_provider.py",
        "native_step.py",
        "qat_optimization.py",
    ]
    source_paths = {
        s: Path(importlib.import_module("w1a1_eagle." + Path(s).stem).__file__).resolve()
        for s in sources
    }
    return {
        "device": "CPU",
        "hardware": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "precision": "F32 projections/STE, F16 native cache/embedding",
        "cpu_threads": torch.get_num_threads(),
        "seed": 17,
        "atol": ATOL,
        "rtol": RTOL,
        "source_paths": {s: str(p) for s, p in source_paths.items()},
        "source_sha256": {
            s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in source_paths.items()
        },
        "cases": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed {len(result['cases'])} synthetic production-path cases on CPU")


def run_ragged(bits, policy, depths=(1, 2, 4)):
    """One fixed mean loss over variable valid chains; preserve per-chain dispatch."""
    check_control()
    native, cfg = tiny_native(bits)
    optimizer = joint_optimizer(native.linears, cfg)
    rounds, outputs = [], []
    with shared_round_hard_signs(native.linears):
        for i, depth in enumerate(depths):
            b = batch(depth, invalid=True, index=i)
            rounds.append(b)
            adapter = HeadPolicy(native, policy, depth)
            outputs.append(forward_torch_round(b, adapter, 3, optimize_head=policy != "serial"))
        valid_outputs = torch.cat([x[:d] for x, d in zip(outputs, depths)])
        loss = F.cross_entropy(valid_outputs, torch.full((sum(depths),), 2, dtype=torch.long))
        loss.backward()
    grads = {
        name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in native.drafter.named_parameters()
        if p.requires_grad
    }
    optimizer.step()
    return dict(
        logits=torch.cat(outputs).detach(),
        loss=loss.detach(),
        raw_vjp=torch.cat([b.raw_target_features.grad for b in rounds]),
        grads=grads,
        params={
            name: p.detach().clone()
            for name, p in native.drafter.named_parameters()
            if p.requires_grad
        },
    )
