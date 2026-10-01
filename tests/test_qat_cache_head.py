"""Synthetic CPU cache/head contracts and an opt-in bounded timing helper.

No weights, captures, accelerator discovery or optimizer updates on real data.
Run ``python tests/test_qat_cache_head.py --benchmark`` for tiny CPU timings.
"""

import copy
import json
import platform
import resource
import statistics
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from w1a1_eagle.continuous_qat import ObservedAdapter
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_loss import supported_prefix_ce
from w1a1_eagle.recurrent_provider import ProviderRound, forward_torch_round
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, shared_round_hard_signs
from w1a1_eagle.recurrent_rollout import DraftStep, rebuild_prefix_cache
from w1a1_eagle.recurrent_trace import RoundAnchor, TraceAudit


def drafter(bits=8, *, hidden=16):
    generator = torch.Generator(device="cpu").manual_seed(811)
    d = nn.Module()
    d.config = SimpleNamespace(
        pretraining_tp=1,
        num_hidden_layers=1,
        hidden_size=hidden,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=hidden + 5,
        head_dim=hidden // 2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
    )
    d.early_stop_method = None
    d.tree_mask = None
    d.embed_tokens = nn.Embedding(11, hidden, dtype=torch.float16)
    with torch.no_grad():
        d.embed_tokens.weight.copy_(torch.randn(11, hidden, generator=generator).half())
    d.midlayer = nn.Module()
    d.midlayer.self_attn = nn.Module()
    d.midlayer.mlp = nn.Module()
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
    for path, (out_features, in_features) in shapes.items():
        module = RowBinaryLinear(
            torch.randn(out_features, in_features, generator=generator) * 0.4,
            0.08 + torch.rand(out_features, generator=generator) * 0.04,
            W1AxContract(bits, "row"),
        )
        parent_path, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent_path), name, module)
    for path in (
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
        "norm",
    ):
        norm = nn.Module()
        norm.weight = nn.Parameter(0.9 + torch.rand(hidden, generator=generator) * 0.2)
        norm.variance_epsilon = 1e-5
        parent_path, _, name = path.rpartition(".")
        setattr(d.get_submodule(parent_path), name, norm)
    return d


def batch(parent=3, *, hidden=16, terminal=True):
    tokens = tuple(index % 11 for index in range(parent + 2))
    seed = tokens[-1]
    rows = []
    for depth in range(3 + int(terminal)):
        valid = depth < 3
        proposal = (seed + 1) % 11 if valid else None
        rows.append(
            {
                "prompt_id": "synthetic",
                "round_index": 0,
                "parent_position": parent,
                "depth": depth,
                "input_position": parent + depth + 1,
                "input_token_id": seed,
                "proposed_token_id": proposal,
                "valid": valid,
            }
        )
        seed = proposal
    raw = torch.randn(
        parent + 1, 3 * hidden, generator=torch.Generator(device="cpu").manual_seed(912)
    )
    return ProviderRound(
        RoundAnchor("synthetic", "train", 0, tokens[:-1], tokens[-1]),
        tuple(rows),
        tokens,
        raw,
        tuple(range(parent + 1)),
        "synthetic-only",
    )


def audit(terminal=True):
    valid = (True, True, True) + ((False,) if terminal else ())
    ce = (False, False, True) + ((False,) if terminal else ())
    labels = (-1, -1, 3) + ((-1,) if terminal else ())
    return TraceAudit(labels, valid, ce, ce, valid, tuple(range(7)), {}, {})


class QATCacheHeadTests(unittest.TestCase):
    def test_chunked_cache_equals_serial_and_is_detached_f16_exact(self):
        for bits in (1, 8):
            adapter = NativeStepAdapter(drafter(bits))
            for parent in (0, 1, 3, 7):
                b = batch(parent)
                b.raw_target_features.requires_grad_()
                reference = rebuild_prefix_cache(
                    b.prefix_token_ids,
                    b.raw_target_features,
                    b.feature_positions,
                    parent_position=parent,
                    encode_feature=adapter.encode_feature,
                    decode_context=adapter.decode_context,
                    new_cache=adapter.new_cache,
                )
                for chunk in (1, 2, 64):
                    with self.subTest(bits=bits, parent=parent, chunk=chunk):
                        cache = adapter.build_context_cache(
                            b.prefix_token_ids[1:-1], b.raw_target_features[:-1], chunk_size=chunk
                        )
                        for name in ("key", "value"):
                            value = getattr(cache, name)
                            self.assertTrue(torch.equal(value, getattr(reference.cache, name)))
                            self.assertTrue(torch.equal(value, value.half().float()))
                            self.assertFalse(value.requires_grad)
                            self.assertIsNone(value.grad_fn)
                        self.assertTrue(reference.seed_raw_features.requires_grad)

    def test_prefix_omits_q_attention_o_ffn_and_head_and_bounds_chunks(self):
        adapter = NativeStepAdapter(drafter())
        b = batch(7)
        forbidden = [
            path
            for path in CANDIDATE_D_BASE_TO_PATH.values()
            if path not in ("fc", "midlayer.self_attn.k_proj", "midlayer.self_attn.v_proj")
        ]
        hooks = [
            adapter.linears[path].register_forward_pre_hook(
                lambda *_: self.fail("context used an unnecessary projection")
            )
            for path in forbidden
        ]
        calls = []
        hook = adapter.drafter.fc.register_forward_pre_hook(
            lambda _, args: calls.append(tuple(args[0].shape))
        )
        try:
            with patch("w1a1_eagle.native_step.F.softmax", side_effect=AssertionError("attention")):
                adapter.build_context_cache(
                    b.prefix_token_ids[1:-1], b.raw_target_features[:-1], chunk_size=2
                )
        finally:
            for handle in hooks + [hook]:
                handle.remove()
        self.assertEqual(calls, [(2, 48), (2, 48), (2, 48), (1, 48)])

    def test_logits_later_gradients_all_nine_and_adamw_update_match(self):
        # F32 GEMM/GEMV tolerances, not a bitwise backend parity requirement.
        for bits in (1, 8):
            with self.subTest(bits=bits):
                b = batch()
                b.raw_target_features.requires_grad_()
                left = NativeStepAdapter(drafter(bits))
                right = NativeStepAdapter(copy.deepcopy(left.drafter))
                observations = []
                results = []
                gradients = []
                for adapter, optimized in ((left, False), (right, True)):
                    observer = ObservedAdapter(adapter)
                    parameters = [
                        p for m in adapter.linears.values() for p in (m.latent_sign, m.scale_offset)
                    ]
                    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
                    with shared_round_hard_signs(adapter.linears):
                        logits = forward_torch_round(
                            b,
                            observer,
                            7,
                            optimize_cache=optimized,
                            optimize_head=optimized,
                            context_chunk_size=2,
                        )
                        loss = supported_prefix_ce(logits, audit())
                        first = observer.first
                        seed_and_cache = (first.pre_norm, first.cache.key, first.cache.value)
                        attached_grads = torch.autograd.grad(
                            loss, seed_and_cache, retain_graph=True
                        )
                        for g in attached_grads:
                            self.assertGreater(float(g.norm()), 0)
                        raw_grad = torch.autograd.grad(
                            loss, b.raw_target_features, retain_graph=True
                        )[0]
                        self.assertEqual(float(raw_grad[:-1].abs().sum()), 0)
                        self.assertGreater(float(raw_grad[-1].norm()), 0)
                        loss.backward()
                    for path, module in adapter.linears.items():
                        for parameter in (module.latent_sign, module.scale_offset):
                            self.assertIsNotNone(parameter.grad, path)
                            self.assertGreater(float(parameter.grad.norm()), 0, path)
                    gradients.append([p.grad.detach().clone() for p in parameters])
                    optimizer.step()
                    self.assertTrue(
                        all(int(state["step"]) == 1 for state in optimizer.state.values())
                    )
                    self.assertFalse(adapter.drafter.embed_tokens.weight.requires_grad)
                    self.assertIsNone(adapter.drafter.norm.weight.grad)
                    observations.append(attached_grads)
                    results.append(
                        (logits.detach(), loss.detach(), [p.detach() for p in parameters])
                    )
                for a, z in zip(results[0][:2], results[1][:2]):
                    torch.testing.assert_close(a, z, rtol=2e-5, atol=2e-6)
                self.assertTrue(torch.equal(results[1][0][-1], torch.zeros(7)))
                for a, z in zip(gradients[0], gradients[1]):
                    torch.testing.assert_close(a, z, rtol=5e-5, atol=2e-6)
                for a, z in zip(results[0][2], results[1][2]):
                    torch.testing.assert_close(a, z, rtol=2e-5, atol=2e-6)
                self.assertEqual(right.head_saturation_scope, "valid_chain_mean")

    def test_one_head_call_valid_states_and_absolute_proposal_positions(self):
        adapter = NativeStepAdapter(drafter())
        observer = ObservedAdapter(adapter)
        b = batch(7)
        steps = []
        heads = []
        original = observer.decode_step

        def record(token, feature, position, cache, **kwargs):
            steps.append((token, position, kwargs.get("compute_logits")))
            return original(token, feature, position, cache, **kwargs)

        observer.decode_step = record
        handle = adapter.drafter.lm_head.register_forward_pre_hook(
            lambda _, args: heads.append(tuple(args[0].shape))
        )
        try:
            logits = forward_torch_round(b, observer, 7, optimize_cache=True, optimize_head=True)
        finally:
            handle.remove()
        self.assertEqual(steps, [(8, 7, False), (9, 8, False), (10, 9, False)])
        self.assertEqual(heads, [(3, 16)])
        self.assertEqual(logits.shape, (4, 7))
        self.assertEqual(observer.first.cache.key.shape, (1, 8, 8))

    def test_generic_and_diagnostic_capability_fallback(self):
        class Generic:
            new_cache = staticmethod(tuple)
            encode_feature = staticmethod(lambda raw: raw[:2])

            def decode_context(self, token, feature, position, cache):
                return DraftStep(torch.empty(0), feature, (*cache, position))

            def decode_step(self, token, feature, position, cache):
                return DraftStep(torch.arange(7).float(), feature, (*cache, position))

        result = forward_torch_round(batch(), Generic(), 7, optimize_cache=True, optimize_head=True)
        self.assertTrue(torch.equal(result[:3], torch.arange(7).float().expand(3, -1)))
        adapter = NativeStepAdapter(
            drafter(),
            attention_mode="native_forward_f32_backward",
            native_attention_oracle=lambda *_: torch.zeros(2, 8),
        )
        self.assertFalse(adapter.supports_context_cache)
        self.assertFalse(adapter.supports_batched_head)
        with (
            patch.object(adapter, "build_context_cache", side_effect=AssertionError("fast cache")),
            patch.object(adapter, "decode_head", side_effect=AssertionError("fast head")),
            patch(
                "w1a1_eagle.native_step.native_forward_f32_backward", return_value=torch.zeros(2, 8)
            ),
            torch.no_grad(),
        ):
            self.assertEqual(
                forward_torch_round(
                    batch(), adapter, 7, optimize_cache=True, optimize_head=True
                ).shape,
                (4, 7),
            )

    def test_invalid_terminal_has_no_child_and_all_invalid_skips_head(self):
        adapter = NativeStepAdapter(drafter())
        b = batch()
        invalid = ({**b.rows[0], "valid": False, "proposed_token_id": None},)
        from dataclasses import replace

        with (
            patch.object(adapter, "decode_step", side_effect=AssertionError("invalid body")),
            patch.object(adapter, "decode_head", side_effect=AssertionError("invalid head")),
        ):
            self.assertTrue(
                torch.equal(
                    forward_torch_round(
                        replace(b, rows=invalid),
                        adapter,
                        7,
                        optimize_cache=True,
                        optimize_head=True,
                    ),
                    torch.zeros(1, 7),
                )
            )
        with self.assertRaisesRegex(ValueError, "terminal"):
            forward_torch_round(replace(b, rows=invalid + b.rows[1:]), adapter, 7)

    def test_fail_closed_shapes_positions_and_architecture(self):
        d = drafter()
        d.config.num_hidden_layers = 2
        with self.assertRaisesRegex(ValueError, "single-layer"):
            NativeStepAdapter(d)
        adapter = NativeStepAdapter(drafter())
        b = batch()
        for size in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "chunk size"):
                forward_torch_round(b, adapter, 7, context_chunk_size=size)
        with self.assertRaisesRegex(ValueError, "aligned"):
            adapter.build_context_cache([1, 2], torch.ones(1, 48))
        with self.assertRaisesRegex(ValueError, "embedding"):
            adapter.build_context_cache([99], torch.ones(1, 48))
        with self.assertRaisesRegex(ValueError, "misordered"):
            rebuild_prefix_cache(
                b.prefix_token_ids,
                b.raw_target_features,
                (0, 2, 1, 3),
                parent_position=3,
                encode_feature=adapter.encode_feature,
                decode_context=adapter.decode_context,
                new_cache=adapter.new_cache,
                build_context_cache=adapter.build_context_cache,
            )
        with self.assertRaisesRegex(ValueError, "seed token/position"):
            from dataclasses import replace

            bad = ({**b.rows[0], "input_token_id": 9},) + b.rows[1:]
            forward_torch_round(replace(b, rows=bad), adapter, 7)

    def test_unknown_attachments_rejected_and_unowned_parameters_frozen(self):
        for attachment in ("activation_quantizer", "fusion_correction"):
            d = drafter()
            child = nn.Module()
            child.extra = nn.Parameter(torch.ones(()))
            setattr(d.fc, attachment, child)
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                NativeStepAdapter(d)
        d = drafter()
        d.unused = nn.Parameter(torch.ones(2))
        NativeStepAdapter(d)
        self.assertFalse(d.unused.requires_grad)

    def test_independent_round_graphs_have_no_cross_example_leakage(self):
        from dataclasses import replace

        adapter = NativeStepAdapter(drafter())
        left = batch(1)
        right = batch(7)
        first = forward_torch_round(left, adapter, 7)
        second = forward_torch_round(right, adapter, 7)
        changed = replace(right, raw_target_features=right.raw_target_features * -3)
        changed_logits = forward_torch_round(changed, adapter, 7)
        again = forward_torch_round(left, adapter, 7)
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(second[:3], changed_logits[:3]))
        # Same-snapshot grouped-round loss uses an explicit equal-round mean.
        # This is a graph/mask fixture, not an enabled minibatch optimizer.
        parameters = list(adapter.drafter.lm_head.parameters())
        grads_left = torch.autograd.grad(
            supported_prefix_ce(first, audit()), parameters, retain_graph=True
        )
        grads_right = torch.autograd.grad(
            supported_prefix_ce(second, audit()), parameters, retain_graph=True
        )
        loss = (supported_prefix_ce(first, audit()) + supported_prefix_ce(second, audit())) / 2
        grads_mean = torch.autograd.grad(loss, parameters)
        for left, right, actual in zip(grads_left, grads_right, grads_mean):
            torch.testing.assert_close(actual, (left + right) / 2, rtol=2e-5, atol=2e-6)


def benchmark():
    """Tiny CPU stage timings, three warmed repeats, no accelerator queries.

    The final-cache bytes are exact tensor size; larger batch estimates below
    are storage arithmetic only and do not establish allocator admission.
    """
    torch.set_num_threads(1)
    output = []
    for bits in (1, 8):
        adapter = NativeStepAdapter(drafter(bits, hidden=32))
        for parent in (64, 256):
            b = batch(parent, hidden=32)
            for cache_opt, head_opt in ((False, False), (True, False), (True, True)):

                def run():
                    adapter.zero_grad(set_to_none=True)
                    with shared_round_hard_signs(adapter.linears):
                        logits = forward_torch_round(
                            b,
                            adapter,
                            7,
                            optimize_cache=cache_opt,
                            optimize_head=head_opt,
                            context_chunk_size=64,
                        )
                        supported_prefix_ce(logits, audit()).backward()

                run()
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    run()
                    timings.append(time.perf_counter() - start)
                output.append(
                    {
                        "hardware": "CPU",
                        "activation_bits": bits,
                        "hidden": 32,
                        "prefix_rows": parent,
                        "valid_depth": 3,
                        "cache_optimized": cache_opt,
                        "head_optimized": head_opt,
                        "warm_repeats": 3,
                        "median_forward_backward_seconds": statistics.median(timings),
                        "cache_f32_bytes": 2 * (32 // 2) * parent * 4,
                        "raw_f32_bytes": (parent + 1) * 96 * 4,
                    }
                )
            # Simulate 2/4 live independent round graphs at one parameter
            # snapshot. No body batching or optimizer update is implemented.
            for group in (2, 4):

                def run_group():
                    adapter.zero_grad(set_to_none=True)
                    with shared_round_hard_signs(adapter.linears):
                        losses = [
                            supported_prefix_ce(
                                forward_torch_round(b, adapter, 7, context_chunk_size=64), audit()
                            )
                            for _ in range(group)
                        ]
                        torch.stack(losses).mean().backward()

                run_group()
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    run_group()
                    timings.append(time.perf_counter() - start)
                rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                output.append(
                    {
                        "hardware": "CPU",
                        "activation_bits": bits,
                        "hidden": 32,
                        "prefix_rows": parent,
                        "valid_depth": 3,
                        "grouped_graph_probe": group,
                        "body_batching": False,
                        "optimizer_updates": 0,
                        "weighting": "equal_round_mean",
                        "warm_repeats": 3,
                        "median_forward_backward_seconds": statistics.median(timings),
                        "total_cache_f32_bytes": group * 2 * (32 // 2) * parent * 4,
                        "shared_raw_f32_bytes": (parent + 1) * 96 * 4,
                        "process_lifetime_peak_rss_bytes": rss
                        * (1024 if platform.system() == "Linux" else 1),
                    }
                )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--benchmark"]:
        benchmark()
    else:
        torch.set_num_threads(1)
        unittest.main()
