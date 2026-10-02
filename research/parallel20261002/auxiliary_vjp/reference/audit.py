"""Synthetic CPU all-family recurrent VJP gate; independent local backward algebra."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from research.parallel20261002.recurrent_vjp.reference.audit import (
    batch,
    check_control,
    explicit_serial,
    model,
)
from w1a1_eagle.affine_binary import AffineBinaryConfig, install_affine_binary
from w1a1_eagle.fusion_correction import FusionCorrectionConfig, install_fusion_correction
from w1a1_eagle.learned_activation import LearnedActivationBank, learned_activation_reference
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.recurrent_provider import forward_torch_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_optimizer,
    joint_parameter_families,
    joint_train_step,
    shared_round_hard_signs,
)
from w1a1_eagle.recurrent_trace import TraceAudit

ROOT = Path(__file__).resolve().parents[4]
ATOL, RTOL = 2e-6, 5e-5


class AlgebraicProjection(torch.autograd.Function):
    """Hand VJP: W=alpha*S+mu, followed by the declared activation STE.

    Only hard values/codes come from the no-grad production hard reference.
    No production affine, sign, or learned-activation backward is used here.
    """

    @staticmethod
    def forward(ctx, x, latent, raw_alpha, parameter, midpoint, bias, bits):
        hard = learned_activation_reference(x, bits, parameter)
        signs = torch.where(latent < 0, -1.0, 1.0)
        alpha = raw_alpha.clamp_min(0)
        binary = (F.linear(hard.codes.float(), signs) * alpha) * hard.scale
        affine = (hard.codes.float().sum(-1, keepdim=True) * midpoint) * hard.scale
        output = binary + affine
        if bias is not None:
            output = output + bias
        if bits == 1:
            shifted = x if float(parameter) == 0 else x - parameter * hard.scale
            support = (shifted.abs() <= hard.scale) | (hard.scale == 0)
            derivative = -hard.scale * support / math.sqrt(x.numel())
        else:
            support = x.abs() <= x.abs().amax(-1, keepdim=True) * parameter
            derivative = torch.where(
                support, (hard.values - x) / parameter, hard.values / parameter
            ) / math.sqrt(x.numel() * ((1 << (bits - 1)) - 1))
        ctx.save_for_backward(
            hard.values, signs, alpha, midpoint, support, derivative, latent, raw_alpha
        )
        return output

    @staticmethod
    def backward(ctx, upstream):
        values, signs, alpha, midpoint, support, derivative, latent, raw_alpha = ctx.saved_tensors
        g = upstream.reshape(-1, signs.shape[0])
        q = values.reshape(-1, signs.shape[1])
        # Quantized-value VJP includes every row's affine contribution.
        dq = ((g * alpha) @ signs + (g * midpoint).sum(-1, keepdim=True)).reshape_as(values)
        contraction = g.T @ q
        dlatent = contraction * alpha[:, None] * (latent.abs() <= 1)
        dalpha = (contraction * signs).sum(-1) * (raw_alpha >= 0)
        dmidpoint = (g * q.sum(-1, keepdim=True)).sum(0)
        return dq * support, dlatent, dalpha, (dq * derivative).sum(), dmidpoint, None, None


class AlgebraicCorrection(torch.autograd.Function):
    """D=U16(V16*x); F16 factor casts have identity backward."""

    @staticmethod
    def forward(ctx, x, u, v, bias, bound):
        uh, vh = u.half().float(), v.half().float()
        z = F.linear(x, vh)
        ctx.save_for_backward(x, z, uh, vh, bias)
        ctx.bound = bound
        return F.linear(z, uh) + bias.clamp(-bound, bound)

    @staticmethod
    def backward(ctx, upstream):
        x, z, u, v, bias = ctx.saved_tensors
        g = upstream.reshape(-1, u.shape[0])
        dz = g @ u
        return (
            (dz @ v).reshape_as(x),
            g.T @ z.reshape(-1, u.shape[1]),
            dz.T @ x.reshape(-1, v.shape[1]),
            g.sum(0) * (bias.abs() <= ctx.bound),
            None,
        )


def oracle_forward(module, x):
    result = AlgebraicProjection.apply(
        x,
        module.latent_sign,
        module.initial_scale + module.scale_offset,
        module.activation_quantizer.parameter,
        module.affine_binary.midpoint,
        module.frozen_bias,
        module.contract.activation_bits,
    )
    correction = getattr(module, "fusion_correction", None)
    if correction is not None:
        result = result + AlgebraicCorrection.apply(
            x, correction.u, correction.v, correction.output_bias, correction.config.bias_bound
        )
    return result


def combined(bits=8, computation="reference", *, oracle=False):
    d = model(bits, computation)
    linears = {p: m for p, m in d.named_modules() if hasattr(m, "latent_sign")}
    bank = LearnedActivationBank(bits, {p: m.in_features for p, m in linears.items()})
    bank.attach(linears)
    affine = AffineBinaryConfig(enabled=True, coverage="all", mild_l2=0.0)
    install_affine_binary(linears, target=nn.Module(), config=affine)
    config = FusionCorrectionConfig(enabled=True, rank=1, output_bias=True)
    correction = install_fusion_correction(linears["fc"], target=nn.Module(), config=config)
    generator = torch.Generator(device="cpu").manual_seed(1062)
    with torch.no_grad():
        for index, quantizer in enumerate(bank.quantizers.values()):
            quantizer.parameter.fill_(0.13 + index * 0.01 if bits == 1 else 0.71 + index * 0.02)
        for module in linears.values():
            module.affine_binary.midpoint.copy_(torch.linspace(-0.027, 0.031, module.out_features))
            module.frozen_bias = torch.linspace(-0.01, 0.014, module.out_features)
        correction.u.copy_(torch.randn(correction.u.shape, generator=generator) * 0.09)
        correction.v.copy_(torch.randn(correction.v.shape, generator=generator) * 0.12)
        correction.output_bias.copy_(torch.linspace(-0.03, 0.023, correction.out_features))
    if oracle:
        for module in linears.values():
            module._forward_hooks.clear()
            module.forward = MethodType(oracle_forward, module)
    return NativeStepAdapter(d)


def parameters(adapter):
    """Canonical names; preserve each tied activation parameter exactly once."""
    named, seen = {}, set()
    for path, module in adapter.linears.items():
        for suffix, parameter in module.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                named[f"{path}.{suffix}"] = parameter
                seen.add(id(parameter))
    assert len(named) == 36
    assert [len(v) for v in joint_parameter_families(adapter.linears).values()] == [9, 9, 6, 3, 9]
    return named


def observe(adapter, *, detach_state=False, detach_cache=False):
    """Capture the first attached proposal input and append-only K/V outputs."""
    original = adapter.decode_step
    seen = []

    def step(token, state, position, cache, **kwargs):
        result = original(token, state, position, cache, **kwargs)
        if torch.is_grad_enabled():
            if not seen:
                seen.extend((state, result.cache.key, result.cache.value))
            if detach_state:
                result = replace(result, pre_norm=result.pre_norm.detach())
            if detach_cache:
                result = replace(
                    result,
                    cache=replace(
                        result.cache,
                        key=result.cache.key.detach(),
                        value=result.cache.value.detach(),
                    ),
                )
        return result

    adapter.decode_step = step
    return seen


def snapshot(
    b,
    adapter,
    *,
    serial=False,
    cache=False,
    chunk=2,
    shared=False,
    detach_state=False,
    detach_cache=False,
):
    b = replace(b, raw_target_features=b.raw_target_features.clone().requires_grad_())
    named = parameters(adapter)
    seen = observe(adapter, detach_state=detach_state, detach_cache=detach_cache)
    with shared_round_hard_signs(adapter.linears) if shared else nullcontext():
        logits = (
            explicit_serial(b, adapter)
            if serial
            else forward_torch_round(
                b, adapter, 7, optimize_cache=cache, optimize_head=False, context_chunk_size=chunk
            )
        )
        depth = sum(r["valid"] for r in b.rows)
        loss = F.cross_entropy(logits[depth - 1 : depth], torch.tensor([3]))
        tensors = (*named.values(), b.raw_target_features, *seen)
        gradients = torch.autograd.grad(loss, tensors, allow_unused=True)
    labels = (*named, "raw", "first_state", "first_cache_key", "first_cache_value")
    return {
        "logits": logits.detach(),
        "loss": loss.detach(),
        **{
            name: torch.zeros_like(tensor) if grad is None else grad.detach()
            for name, tensor, grad in zip(labels, tensors, gradients, strict=True)
        },
    }


def compare(actual, expected):
    table = {}
    for name, wanted in expected.items():
        torch.testing.assert_close(
            actual[name], wanted, atol=ATOL, rtol=RTOL, msg=lambda message: f"{name}: {message}"
        )
        table[name] = {
            "max_abs_error": float((actual[name] - wanted).abs().max()),
            "reference_norm": float(wanted.norm()),
            "actual_norm": float(actual[name].norm()),
            "elements": wanted.numel(),
        }
    return table


def mode_matrix():
    rows = []
    for bits in (1, 4, 8):
        check_control()
        for depth in (1, 3):
            b = batch(parent=2, depth=depth, terminal=True)
            expected = snapshot(b, combined(bits, oracle=True), serial=True)
            assert expected["raw"][:-1].count_nonzero() == 0
            assert expected["raw"][-1].norm() > 0
            for computation in ("reference", "single_forward"):
                for cache, chunk, shared in (
                    (False, 2, False),
                    (False, 2, True),
                    (True, 1, False),
                    (True, 2, True),
                ):
                    actual = snapshot(
                        b, combined(bits, computation), cache=cache, chunk=chunk, shared=shared
                    )
                    table = compare(actual, expected)
                    rows.append(
                        dict(
                            bits=bits,
                            depth=depth,
                            computation=computation,
                            optimize_cache=cache,
                            chunk=chunk,
                            shared=shared,
                            optimize_head=False,
                            loss=float(actual["loss"]),
                            vjps=table,
                        )
                    )
    return rows


def one_update(bits):
    check_control()
    actual, oracle = combined(bits, "single_forward"), combined(bits, oracle=True)
    cfg = JointQATConfig(
        W1AxContract(bits),
        activation_quantization="learned",
        a1_computation="single_forward",
        max_grad_norm=0.07,
        fusion_correction=actual.linears["fc"].fusion_correction.config,
        affine_weights=actual.linears["fc"].affine_binary.config,
    )
    b = batch(parent=2, depth=3, terminal=True)
    mask = (True, True, True, False)
    audit = TraceAudit(
        (3, 3, 3, -1),
        (True, True, True, False),
        (True, True, True, False),
        mask,
        (True, True, True, False),
        tuple(range(7)),
        {},
        {},
    )
    outputs = []
    for adapter, serial in ((actual, False), (oracle, True)):
        optimizer = joint_optimizer(adapter.linears, cfg)
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(owned) == len({id(p) for p in owned}) == 36
        with shared_round_hard_signs(adapter.linears) if not serial else nullcontext():
            logits = (
                explicit_serial(b, adapter)
                if serial
                else forward_torch_round(
                    b, adapter, 7, optimize_cache=True, optimize_head=False, context_chunk_size=2
                )
            )
            metrics = joint_train_step(adapter.linears, logits, audit, optimizer, cfg)
        outputs.append(
            (
                {k: p.detach().clone() for k, p in parameters(adapter).items()},
                {k: p.grad.detach().clone() for k, p in parameters(adapter).items()},
                metrics,
            )
        )
    updates = compare(outputs[0][0], outputs[1][0])
    clipped = compare(outputs[0][1], outputs[1][1])
    norm = math.sqrt(sum(float(g.square().sum()) for g in outputs[0][1].values()))
    assert norm <= cfg.max_grad_norm + 1e-6
    assert norm > cfg.max_grad_norm * 0.99
    return dict(
        bits=bits,
        update_table=updates,
        clipped_gradient_table=clipped,
        clipped_norm=norm,
        max_grad_norm=cfg.max_grad_norm,
        metrics=outputs[0][2],
    )


def negative_controls():
    check_control()
    b = batch(parent=2, depth=3, terminal=True)
    expected = snapshot(b, combined(8), serial=True)
    deltas = {}
    for kind in ("state", "cache"):
        actual = snapshot(
            b, combined(8), detach_state=kind == "state", detach_cache=kind == "cache"
        )
        torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
        try:
            compare(actual, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("Detached graph must fail the VJP checker")
        deltas[kind] = max(float((actual[k] - expected[k]).abs().max()) for k in expected)
    bad = combined(8)
    bad.linears["midlayer.self_attn.k_proj"].activation_quantizer = copy.deepcopy(
        bad.linears["midlayer.self_attn.q_proj"].activation_quantizer
    )
    try:
        LearnedActivationBank.from_attached(bad.linears)
    except ValueError:
        deltas["split_tied_quantizer_rejected"] = True
    else:
        raise AssertionError("Split QKV ownership must reject")
    bad = combined(8)
    bad.linears["lm_head"].activation_quantizer.clip_ratio = bad.linears[
        "fc"
    ].activation_quantizer.clip_ratio
    try:
        LearnedActivationBank.from_attached(bad.linears)
    except ValueError:
        deltas["cross_boundary_alias_rejected"] = True
    else:
        raise AssertionError("Different activation boundaries must reject parameter aliasing")
    return deltas


def run():
    torch.set_num_threads(1)
    sources = [
        f"src/w1a1_eagle/{n}.py"
        for n in (
            "recurrent_qat",
            "learned_activation",
            "affine_binary",
            "fusion_correction",
            "native_step",
            "recurrent_provider",
            "recurrent_rollout",
            "recurrent_binary",
            "recurrent_loss",
            "recurrent_trace",
        )
    ]
    sources += [
        "research/parallel20261002/recurrent_vjp/reference/audit.py",
        "research/parallel20261002/auxiliary_vjp/reference/audit.py",
    ]
    return dict(
        schema="combined_auxiliary_vjp_cpu_v1",
        device="CPU",
        hardware=platform.machine(),
        torch=torch.__version__,
        precision="F32 masters/arithmetic, F16 K/V/factors",
        atol=ATOL,
        rtol=RTOL,
        matrix=mode_matrix(),
        one_update=[one_update(bits) for bits in (1, 4, 8)],
        negative_controls=negative_controls(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    )


def summarize(result, full_result_sha256):
    """Keep six aggregate VJP tables in Git, full mode tables in ignored runs."""
    compact = {k: v for k, v in result.items() if k not in ("matrix", "one_update")}
    compact["comparisons"] = len(result["matrix"])
    compact["tables"] = []
    for bits in (1, 4, 8):
        for depth in (1, 3):
            group = [x for x in result["matrix"] if x["bits"] == bits and x["depth"] == depth]
            table = {
                k: {**v, "max_abs_error": max(x["vjps"][k]["max_abs_error"] for x in group)}
                for k, v in group[0]["vjps"].items()
            }
            compact["tables"].append(
                dict(bits=bits, depth=depth, loss=group[0]["loss"], vjps=table)
            )
    compact["modes"] = [
        {k: v for k, v in x.items() if k not in ("vjps", "loss", "bits", "depth")}
        for x in result["matrix"][:8]
    ]
    compact["one_update"] = [
        dict(
            bits=x["bits"],
            clipped_norm=x["clipped_norm"],
            max_grad_norm=x["max_grad_norm"],
            metrics=x["metrics"],
            max_parameter_error=max(v["max_abs_error"] for v in x["update_table"].values()),
            max_clipped_gradient_error=max(
                v["max_abs_error"] for v in x["clipped_gradient_table"].values()
            ),
        )
        for x in result["one_update"]
    ]
    compact["full_result_sha256"] = full_result_sha256
    compact["raw_path"] = "runs/parallel20261002/auxiliary-vjp/results.json"
    return compact


if __name__ == "__main__":
    result = run()
    output = ROOT / "runs/parallel20261002/auxiliary-vjp/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    summary = ROOT / "experiments/parallel20261002/auxiliary_vjp/summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(summarize(result, hashlib.sha256(output.read_bytes()).hexdigest()), indent=2)
        + "\n"
    )
    print(f"{len(result['matrix'])} composed comparisons and 3 clipped updates passed")
