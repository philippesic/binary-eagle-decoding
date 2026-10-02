"""Bounded actual NativeStep and invalidation acceptance gates."""

import unittest

import torch

from research.parallel20261002.activation_reuse.reference.prototype import (
    immediate_group,
    install_reuse,
    quantize,
)
from research.parallel20261002.auxiliary_vjp.reference.audit import (
    batch,
    check_control,
    combined,
    compare,
    parameters,
    snapshot,
)
from w1a1_eagle.learned_activation import LearnedActivationQuantizer
from w1a1_eagle.recurrent_provider import forward_torch_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_optimizer,
    joint_train_step,
    shared_round_hard_signs,
)
from w1a1_eagle.recurrent_trace import TraceAudit


def count_quantizers(adapter):
    counts, seen = {}, set()
    for module in adapter.linears.values():
        q = module.activation_quantizer
        if id(q) in seen:
            continue
        seen.add(id(q))
        original = q.forward

        def counted(input, original=original, boundary=q.boundary, **kwargs):
            counts[boundary] = counts.get(boundary, 0) + 1
            return original(input, **kwargs)

        q.forward = counted
    return counts


def graph_gate(bits):
    check_control()
    b = batch(parent=2, depth=3, terminal=True)
    baseline = combined(bits, "single_forward")
    reuse = combined(bits, "single_forward")
    baseline_calls, reuse_calls = count_quantizers(baseline), count_quantizers(reuse)
    install_reuse(reuse)
    hard_outputs = [[], []]
    for adapter, outputs in zip((baseline, reuse), hard_outputs, strict=True):
        for path, module in adapter.linears.items():
            def capture(module, args, result, path=path, outputs=outputs):
                outputs.append((path, result.detach().clone()))

            module.register_forward_hook(capture)
    expected = snapshot(b, baseline, cache=True, chunk=2, shared=True)
    actual = snapshot(b, reuse, cache=True, chunk=2, shared=True)
    assert len(hard_outputs[0]) == len(hard_outputs[1])
    for (path, wanted), (actual_path, got) in zip(*hard_outputs, strict=True):
        assert path == actual_path
        torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
    errors = compare(actual, expected)
    events = reuse.reuse_events
    assert events
    assert all(e["misses"] == 1 and e["retained"] == 0 for e in events)
    assert all(e["hits"] == (2 if e["label"] == "qkv" else 1) for e in events)
    # Optimized detached context uses K/V-only chunks; the three attached
    # proposals are the all-nine decode graph and the sole reuse scope here.
    assert len([e for e in events if e["label"] == "qkv"]) == 3
    assert baseline_calls["qkv"] - reuse_calls["qkv"] == 6
    assert baseline_calls["gate_up"] == 6 and reuse_calls["gate_up"] == 3
    return dict(bits=bits, groups=len(events), events=events, vjps=errors,
                baseline_calls=baseline_calls, reuse_calls=reuse_calls,
                exact_projection_outputs=len(hard_outputs[0]))


def update_gate(bits):
    check_control()
    b = batch(parent=2, depth=3, terminal=True)
    audit = TraceAudit(
        (3, 3, 3, -1), (True, True, True, False), (True, True, True, False),
        (True, True, True, False), (True, True, True, False), tuple(range(7)), {}, {},
    )
    outputs = []
    for use_reuse in (False, True):
        adapter = combined(bits, "single_forward")
        if use_reuse:
            install_reuse(adapter)
        cfg = JointQATConfig(
            W1AxContract(bits), activation_quantization="learned",
            a1_computation="single_forward", max_grad_norm=0.07,
            fusion_correction=adapter.linears["fc"].fusion_correction.config,
            affine_weights=adapter.linears["fc"].affine_binary.config,
        )
        optimizer = joint_optimizer(adapter.linears, cfg)
        with shared_round_hard_signs(adapter.linears):
            logits = forward_torch_round(
                b, adapter, 7, optimize_cache=True, optimize_head=False, context_chunk_size=2
            )
            metrics = joint_train_step(adapter.linears, logits, audit, optimizer, cfg)
        outputs.append((
            {k: p.detach().clone() for k, p in parameters(adapter).items()},
            {k: p.grad.detach().clone() for k, p in parameters(adapter).items()}, metrics,
        ))
    update = compare(outputs[1][0], outputs[0][0])
    clipped = compare(outputs[1][1], outputs[0][1])
    norm = sum(float(g.square().sum()) for g in outputs[1][1].values()) ** 0.5
    assert 0.069 < norm <= 0.070001
    return dict(bits=bits, updates=update, clipped=clipped, clipped_norm=norm)


def invalidation_gate():
    check_control()
    q = LearnedActivationQuantizer(4, "qkv", 4)
    x = torch.tensor([[-0.8, -0.1, 0.3, 0.9]], requires_grad=True)
    mask = torch.tensor([True])
    with immediate_group("counterexamples") as group:
        first = quantize(q, x)
        assert quantize(q, x) is first
        with torch.no_grad():
            x.add_(0.2)
        changed = quantize(q, x)
        assert changed is not first
        with torch.no_grad():
            q.parameter.fill_(0.7)
        updated = quantize(q, x)
        assert updated is not changed
        view = x.view_as(x)
        assert quantize(q, view) is not updated
        masked = quantize(q, x, valid_mask=mask)
        with torch.no_grad():
            mask.fill_(False)
        assert quantize(q, x, valid_mask=mask) is not masked
        normal = quantize(q, x, normalization_count=4)
        assert quantize(q, x, normalization_count=8) is not normal
        try:
            quantize(q, x, normalization_count=4.0)
        except ValueError:
            pass
        else:
            raise AssertionError("Equal-valued invalid N must preserve validation error")
        with torch.no_grad():
            detached = quantize(q, x)
        attached = quantize(q, x)
        assert attached is not detached and attached.values.requires_grad
        # Failed body also reaches finally and clears strong references.
        assert group.entries
    assert group.entries == {}
    with immediate_group("fresh"):
        assert quantize(q, x) is not attached
    with torch.inference_mode(), immediate_group("inference") as inference:
        z = torch.ones(1, 4)
        assert quantize(q, z) is not quantize(q, z)
        assert inference.entries == {}
    return dict(hits=group.hits, misses=group.misses, retained=len(group.entries))


def test_native_graph_and_updates():
    torch.set_num_threads(1)
    for bits in (1, 4, 8):
        graph_gate(bits)
        update_gate(bits)


def test_invalidation():
    invalidation_gate()


def test_exception_scope_cleanup():
    q = LearnedActivationQuantizer(1, "qkv", 4)
    group = None
    try:
        with immediate_group("exception") as group:
            quantize(q, torch.ones(4))
            raise RuntimeError("synthetic body error")
    except RuntimeError:
        pass
    assert group.entries == {}
    with immediate_group("next"):
        pass


class ReuseTests(unittest.TestCase):
    test_native_graph_and_updates = staticmethod(test_native_graph_and_updates)
    test_invalidation = staticmethod(test_invalidation)
    test_exception_scope_cleanup = staticmethod(test_exception_scope_cleanup)


if __name__ == "__main__":
    unittest.main()
