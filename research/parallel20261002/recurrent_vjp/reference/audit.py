"""Bounded source-bound CPU audit of composed current-student recurrence."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.qat_curriculum_runner import CurriculumRunner
from w1a1_eagle.recurrent_provider import ProviderRound, forward_torch_round
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, shared_round_hard_signs
from w1a1_eagle.recurrent_trace import RoundAnchor

ROOT = Path(__file__).resolve().parents[4]
CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
ATOL, RTOL = 2e-6, 5e-5


def check_control():
    if CONTROL.exists():
        state = json.loads(CONTROL.read_text())
        used = state.get("last_weekly_used_percent", 0)
        reset_changed = state.get("last_reset_unix") != state.get("initial_reset_unix")
        if (
            state["research_stop"]
            or state.get("reset_observed")
            or reset_changed
            or used >= 100 - state.get("threshold_remaining_percent", 1)
        ):
            raise RuntimeError("Research stopped by supervisor control")


def model(bits=8, computation="reference", hidden=8):
    """Reduced real nine-projection EAGLE; fixed activation contract."""
    g = torch.Generator(device="cpu").manual_seed(811)
    d = nn.Module()
    d.config = SimpleNamespace(
        pretraining_tp=1,
        num_hidden_layers=1,
        hidden_size=hidden,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=hidden + 5,
        head_dim=hidden // 2,
        max_position_embeddings=32,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
    )
    d.early_stop_method = d.tree_mask = None
    d.embed_tokens = nn.Embedding(11, hidden, dtype=torch.float16)
    with torch.no_grad():
        d.embed_tokens.weight.copy_(torch.randn(11, hidden, generator=g).half())
    d.midlayer = nn.Module()
    d.midlayer.self_attn, d.midlayer.mlp = nn.Module(), nn.Module()
    shapes = {
        "fc": (hidden, 3 * hidden),
        "midlayer.self_attn.q_proj": (hidden, 2 * hidden),
        "midlayer.self_attn.k_proj": (hidden // 2, 2 * hidden),
        "midlayer.self_attn.v_proj": (hidden // 2, 2 * hidden),
        "midlayer.self_attn.o_proj": (hidden, hidden),
        "midlayer.mlp.gate_proj": (hidden + 5, hidden),
        "midlayer.mlp.up_proj": (hidden + 5, hidden),
        "midlayer.mlp.down_proj": (hidden, hidden + 5),
        "lm_head": (7, hidden),
    }
    for path, (out_width, in_width) in shapes.items():
        module = RowBinaryLinear(
            torch.randn(out_width, in_width, generator=g) * 0.4,
            0.08 + torch.rand(out_width, generator=g) * 0.04,
            W1AxContract(bits, "row"),
            a1_computation=computation,
        )
        parent, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent), name, module)
    for path in (
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
        "norm",
    ):
        norm = nn.Module()
        norm.weight = nn.Parameter(0.9 + torch.rand(hidden, generator=g) * 0.2)
        norm.variance_epsilon = 1e-5
        parent, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent), name, norm)
    return d


def batch(parent=3, depth=3, terminal=True, hidden=8):
    tokens = tuple(i % 11 for i in range(parent + 2))
    token, rows = tokens[-1], []
    for index in range(depth + int(terminal)):
        valid = index < depth
        proposal = (token + 1) % 11 if valid else None
        rows.append(
            dict(
                prompt_id="synthetic",
                round_index=0,
                parent_position=parent,
                depth=index,
                input_position=parent + index + 1,
                input_token_id=token,
                proposed_token_id=proposal,
                valid=valid,
            )
        )
        token = proposal
    raw = torch.randn(
        parent + 1, hidden * 3, generator=torch.Generator(device="cpu").manual_seed(912)
    )
    return ProviderRound(
        RoundAnchor("synthetic", "train", 0, tokens[:-1], tokens[-1]),
        tuple(rows),
        tokens,
        raw,
        tuple(range(parent + 1)),
        "synthetic-only",
    )


def explicit_serial(b, adapter, *, initial_cache=None):
    """Independent topology: does not call provider, rebuild, or rollout helpers.

    Accepted context is intentionally truncated. Each proposal inherits the
    previous pre-norm state and its attached append-only cache. Real adapter
    arithmetic (including F16 K/V) is common to both sides of this graph audit.
    """
    parent = len(b.prefix_token_ids) - 2
    if initial_cache is None:
        cache = adapter.new_cache()
        with torch.no_grad():
            for j in range(parent):
                state = adapter.encode_feature(b.raw_target_features[j])
                cache = adapter.decode_context(b.prefix_token_ids[j + 1], state, j, cache).cache
    else:
        cache = initial_cache
    state = adapter.encode_feature(b.raw_target_features[parent])
    logits = []
    for row in b.rows:
        if row["valid"]:
            result = adapter.decode_step(
                row["input_token_id"], state, row["input_position"] - 1, cache
            )
            state, cache = result.pre_norm, result.cache
            logits.append(result.logits)
        else:
            logits.append(torch.zeros(7))
    return torch.stack(logits)


def parameters(adapter):
    return {
        f"{path}.{family}": getattr(module, family)
        for path, module in adapter.linears.items()
        for family in ("latent_sign", "scale_offset")
    }


def later_loss(logits, b):
    depth = sum(row["valid"] for row in b.rows)
    return F.cross_entropy(logits[depth - 1 : depth], torch.tensor([3]))


def snapshot(b, adapter, *, serial=False, cache=False, head=False, chunk=2, shared=True):
    b = replace(b, raw_target_features=b.raw_target_features.detach().clone().requires_grad_())
    named = parameters(adapter)
    context = shared_round_hard_signs(adapter.linears) if shared else nullcontext()
    with context:
        logits = (
            explicit_serial(b, adapter)
            if serial
            else forward_torch_round(
                b, adapter, 7, optimize_cache=cache, optimize_head=head, context_chunk_size=chunk
            )
        )
        loss = later_loss(logits, b)
        gradients = torch.autograd.grad(loss, (*named.values(), b.raw_target_features))
    return {
        "logits": logits.detach(),
        "loss": loss.detach(),
        **{name: grad.detach() for name, grad in zip(named, gradients[:-1], strict=True)},
        "raw": gradients[-1].detach(),
    }


def compare(actual, expected):
    errors = {}
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], atol=ATOL, rtol=RTOL, msg=lambda msg: f"{name}: {msg}"
        )
        errors[name] = float((actual[name] - expected[name]).abs().max())
    return errors


def mode_matrix():
    rows = []
    for bits in (1, 4, 8):
        check_control()
        for parent in (0, 1, 3):
            for depth in (1, 3):
                for terminal in (False, True):
                    b = batch(parent, depth, terminal)
                    oracle = snapshot(b, NativeStepAdapter(model(bits)), serial=True, shared=False)
                    assert not oracle["raw"][:-1].count_nonzero()
                    assert oracle["raw"][-1].norm() > 0
                    for cache, head, chunk, shared, computation in (
                        (False, False, 2, True, "reference"),
                        (True, False, 1, True, "reference"),
                        (False, True, 2, True, "reference"),
                        (True, True, 2, True, "reference"),
                        (True, True, 64, False, "reference"),
                        (True, True, 2, True, "single_forward"),
                    ):
                        actual = snapshot(
                            b,
                            NativeStepAdapter(model(bits, computation)),
                            cache=cache,
                            head=head,
                            chunk=chunk,
                            shared=shared,
                        )
                        errors = compare(actual, oracle)
                        if terminal:
                            assert torch.equal(actual["logits"][-1], torch.zeros(7))
                        rows.append(
                            dict(
                                bits=bits,
                                parent=parent,
                                depth=depth,
                                terminal=terminal,
                                cache=cache,
                                head=head,
                                chunk=chunk,
                                shared=shared,
                                computation=computation,
                                max_logit_error=errors["logits"],
                                max_gradient_error=max(
                                    v for k, v in errors.items() if k not in ("logits", "loss")
                                ),
                                loss=float(actual["loss"]),
                            )
                        )
    return rows


def runner_stub(adapter):
    return SimpleNamespace(
        device=torch.device("cpu"),
        linears=adapter.linears,
        adapter=adapter,
        provider=SimpleNamespace(draft_vocab_size=7),
        qat=SimpleNamespace(optimize_cache=True, optimize_head=True, context_chunk_size=2),
    )


def training_step(adapter, optimizer, b, *, serial=False):
    optimizer.zero_grad(set_to_none=True)
    if serial:
        with shared_round_hard_signs(adapter.linears):
            logits = explicit_serial(b, adapter)
            loss = later_loss(logits, b)
            loss.backward()
    else:
        logits = CurriculumRunner._forward(runner_stub(adapter), b)
        loss = later_loss(logits, b)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(list(parameters(adapter).values()), 0.7)
    grads = {k: p.grad.detach().clone() for k, p in parameters(adapter).items()}
    optimizer.step()
    for module in adapter.linears.values():
        module.project_scales_()
    return {"logits": logits.detach(), **grads}


def two_step_resume(bits=8):
    check_control()
    b = batch()
    actual, oracle = NativeStepAdapter(model(bits)), NativeStepAdapter(model(bits))

    def optim(a):
        return torch.optim.AdamW(list(parameters(a).values()), lr=1e-3)

    left, right = optim(actual), optim(oracle)
    first = compare(training_step(actual, left, b), training_step(oracle, right, b, serial=True))
    compare(
        {k: p.detach() for k, p in parameters(actual).items()},
        {k: p.detach() for k, p in parameters(oracle).items()},
    )
    for path, module in actual.linears.items():
        assert torch.equal(module.latent_sign.sign(), oracle.linears[path].latent_sign.sign())
    payload = copy.deepcopy((actual.drafter.state_dict(), left.state_dict()))
    resumed = NativeStepAdapter(model(bits))
    resumed.drafter.load_state_dict(payload[0])
    resume_optimizer = optim(resumed)
    resume_optimizer.load_state_dict(payload[1])
    second_actual = training_step(actual, left, b)
    second_oracle = training_step(oracle, right, b, serial=True)
    second_resume = training_step(resumed, resume_optimizer, b)
    second = compare(second_actual, second_oracle)
    resume = compare(second_resume, second_actual)
    compare(
        {k: p.detach() for k, p in parameters(actual).items()},
        {k: p.detach() for k, p in parameters(oracle).items()},
    )
    compare(
        {k: p.detach() for k, p in parameters(resumed).items()},
        {k: p.detach() for k, p in parameters(actual).items()},
    )
    for optimizer in (left, right, resume_optimizer):
        assert all(int(s["step"]) == 2 for s in optimizer.state.values())
    for path, module in actual.linears.items():
        assert torch.equal(module.latent_sign.sign(), oracle.linears[path].latent_sign.sign())
        assert torch.equal(module.latent_sign.sign(), resumed.linears[path].latent_sign.sign())
    for resumed_state, actual_state in zip(
        resume_optimizer.state.values(), left.state.values(), strict=True
    ):
        for name in actual_state:
            torch.testing.assert_close(resumed_state[name], actual_state[name], atol=0, rtol=0)
    return dict(
        bits=bits,
        first_max=max(first.values()),
        second_max=max(second.values()),
        resume_max=max(resume.values()),
        parameter_families=len(parameters(actual)),
    )


def stale_cache_control():
    check_control()
    adapter, b = NativeStepAdapter(model(8)), batch()
    stale = adapter.build_context_cache(b.prefix_token_ids[1:-1], b.raw_target_features[:-1])
    # Large controlled synthetic scale perturbation isolates lifetime semantics;
    # it is not a recipe/optimizer recommendation.
    with torch.no_grad():
        adapter.linears["midlayer.self_attn.v_proj"].scale_offset.add_(0.08)
    fresh = forward_torch_round(b, adapter, 7, optimize_cache=True, optimize_head=True)
    wrong = explicit_serial(b, adapter, initial_cache=stale)
    delta = float((fresh - wrong).detach().abs().max())
    assert delta > 1e-4, "Fixture must detect stale accepted-context values after weight change"
    rebuilt = adapter.build_context_cache(b.prefix_token_ids[1:-1], b.raw_target_features[:-1])
    assert not torch.equal(stale.value, rebuilt.value)
    return dict(
        max_stale_logit_error=delta,
        max_stale_value_error=float((stale.value - rebuilt.value).abs().max()),
    )


def rollback_rebuild_control():
    """Rebuild accepted context, excluding proposals from the previous chain.

    No verifier/branch selection is simulated: the supported representation is
    one contiguous exact-prefix chain per ProviderRound.
    """
    check_control()
    adapter, b = NativeStepAdapter(model(8)), batch(parent=1)
    recorded = []
    original_step = adapter.decode_step

    def record(token, feature, position, cache, **kwargs):
        result = original_step(token, feature, position, cache, **kwargs)
        recorded.append((position, cache, result.cache))
        return result

    adapter.decode_step = record
    first = forward_torch_round(b, adapter, 7, optimize_cache=True, optimize_head=True)
    first_seed_cache, rejected_chain_cache = recorded[0][1], recorded[-1][2]
    assert rejected_chain_cache.key.shape[1] == 4
    recorded.clear()
    rebuilt = forward_torch_round(b, adapter, 7, optimize_cache=True, optimize_head=True)
    second_seed_cache = recorded[0][1]
    torch.testing.assert_close(first, rebuilt, atol=0, rtol=0)
    assert second_seed_cache is not first_seed_cache
    assert second_seed_cache.key.shape[1] == 1
    assert not second_seed_cache.key.requires_grad
    try:
        explicit_serial(b, adapter, initial_cache=rejected_chain_cache)
    except ValueError:
        pass
    else:
        raise AssertionError("A proposal cache at another position must be rejected")
    return dict(
        rejected_chain_length=4,
        rebuilt_accepted_length=1,
        logits_exact=True,
        wrong_position_rejected=True,
    )


def run():
    torch.set_num_threads(1)
    matrix = mode_matrix()
    sources = (
        "native_step.py",
        "recurrent_rollout.py",
        "recurrent_provider.py",
        "qat_curriculum_runner.py",
        "recurrent_qat.py",
        "recurrent_binary.py",
    )
    return dict(
        schema="recurrent_vjp_cpu_v1",
        device="CPU",
        hardware=platform.machine(),
        torch=torch.__version__,
        precision="F32 arithmetic; F16 K/V roundtrip; fixed A1/A4/A8",
        atol=ATOL,
        rtol=RTOL,
        matrix=matrix,
        two_step_resume=[two_step_resume(bits) for bits in (1, 4, 8)],
        stale_cache=stale_cache_control(),
        rollback_rebuild=rollback_rebuild_control(),
        source_sha256={
            name: hashlib.sha256((ROOT / "src/w1a1_eagle" / name).read_bytes()).hexdigest()
            for name in sources
        },
    )


if __name__ == "__main__":
    result = run()
    output = ROOT / "experiments/parallel20261002/recurrent_vjp/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "matrix"}, indent=2))
    print(f"Matrix: {len(result['matrix'])} comparisons passed")
