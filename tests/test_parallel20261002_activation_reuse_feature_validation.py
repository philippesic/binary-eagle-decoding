"""Independent CPU integration checks for NativeStep activation reuse."""

from __future__ import annotations

import copy
import gc
import sys
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_qat_cache_head import batch as provider_batch  # noqa: E402

from w1a1_eagle.learned_activation import BOUNDARY_PATHS, LearnedActivationBank  # noqa: E402
from w1a1_eagle.native_step import NativeStepAdapter  # noqa: E402
from w1a1_eagle.recurrent_provider import forward_torch_round  # noqa: E402
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract  # noqa: E402


def _drafter(seed: int = 2301, bits: int = 4) -> nn.Module:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden, intermediate = 8, 11
    drafter = nn.Module()
    drafter.config = SimpleNamespace(
        pretraining_tp=1,
        num_hidden_layers=1,
        hidden_size=hidden,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=intermediate,
        head_dim=hidden // 2,
        max_position_embeddings=32,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
    )
    drafter.early_stop_method = None
    drafter.tree_mask = None
    drafter.embed_tokens = nn.Embedding(13, hidden, dtype=torch.float16)
    with torch.no_grad():
        drafter.embed_tokens.weight.copy_(
            (torch.randn(13, hidden, generator=generator) * 0.4).half()
        )

    shapes = {
        "fc": (hidden, 3 * hidden),
        "midlayer.self_attn.q_proj": (hidden, 2 * hidden),
        "midlayer.self_attn.k_proj": (hidden // 2, 2 * hidden),
        "midlayer.self_attn.v_proj": (hidden // 2, 2 * hidden),
        "midlayer.self_attn.o_proj": (hidden, hidden),
        "midlayer.mlp.gate_proj": (intermediate, hidden),
        "midlayer.mlp.up_proj": (intermediate, hidden),
        "midlayer.mlp.down_proj": (hidden, intermediate),
        "lm_head": (7, hidden),
    }
    drafter.midlayer = nn.Module()
    drafter.midlayer.self_attn = nn.Module()
    drafter.midlayer.mlp = nn.Module()
    for path, (out_features, in_features) in shapes.items():
        projection = RowBinaryLinear(
            torch.randn(out_features, in_features, generator=generator) * 0.35,
            0.07 + torch.rand(out_features, generator=generator) * 0.04,
            W1AxContract(bits, "row"),
        )
        parent, _, leaf = path.rpartition(".")
        setattr(drafter.get_submodule(parent), leaf, projection)

    for path in (
        "midlayer.input_layernorm",
        "midlayer.hidden_norm",
        "midlayer.post_attention_layernorm",
        "norm",
    ):
        norm = nn.Module()
        norm.weight = nn.Parameter(0.9 + torch.rand(hidden, generator=generator) * 0.2)
        norm.variance_epsilon = 1e-5
        parent, _, leaf = path.rpartition(".")
        setattr(drafter.get_submodule(parent), leaf, norm)

    linears = {
        path: drafter.get_submodule(path) for paths in BOUNDARY_PATHS.values() for path in paths
    }
    LearnedActivationBank(
        bits, {path: module.in_features for path, module in linears.items()}
    ).attach(linears)
    return drafter


def _adapters(bits: int = 4) -> tuple[NativeStepAdapter, NativeStepAdapter]:
    source = _drafter(bits=bits)
    return (
        NativeStepAdapter(copy.deepcopy(source), activation_reuse=False),
        NativeStepAdapter(copy.deepcopy(source), activation_reuse=True),
    )


def _step(adapter: NativeStepAdapter, index: int, feature: torch.Tensor | None = None):
    if feature is None:
        feature = torch.tensor(
            [0.2, -0.4, 0.1, 0.5, -0.3, 0.7, -0.2, 0.6],
            dtype=torch.float32,
            requires_grad=True,
        )
    cache = adapter.new_cache()
    for position in range(index + 1):
        result = adapter.decode_step(2 + position, feature, position, cache)
        cache = result.cache
    return result, feature


def _assert_tree_close(test: unittest.TestCase, left, right, *, atol=2e-6, rtol=5e-5):
    test.assertEqual(left.keys(), right.keys())
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=atol, rtol=rtol, msg=name)


def _assert_state_equal(test: unittest.TestCase, left, right):
    test.assertEqual(left.keys(), right.keys())
    for name in left:
        if isinstance(left[name], torch.Tensor):
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0, msg=name)
        else:
            test.assertEqual(left[name], right[name], name)


class ActivationReuseIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_actual_native_graph_vjp_and_sgd_update_parity(self):
        for bits in (1, 8):
            with self.subTest(bits=bits):
                baseline, reused = _adapters(bits)
                outputs = []
                for adapter in (baseline, reused):
                    feature = torch.tensor(
                        [0.2, -0.4, 0.1, 0.5, -0.3, 0.7, -0.2, 0.6],
                        dtype=torch.float32,
                        requires_grad=True,
                    )
                    first = adapter.decode_step(3, feature, 0, adapter.new_cache())
                    second = adapter.decode_step(4, first.pre_norm, 1, first.cache)
                    weight = torch.linspace(-0.7, 0.8, second.logits.numel())
                    loss = (second.logits * weight).sum() + 0.03 * second.cache.key.square().sum()
                    loss.backward()
                    named = {
                        name: parameter
                        for name, parameter in adapter.named_parameters()
                        if parameter.requires_grad
                    }
                    missing_gradients = {
                        name for name, parameter in named.items() if parameter.grad is None
                    }
                    gradients = {
                        name: parameter.grad.detach().clone()
                        for name, parameter in named.items()
                        if parameter.grad is not None
                    }
                    with torch.no_grad():
                        for parameter in named.values():
                            if parameter.grad is not None:
                                parameter.add_(parameter.grad, alpha=-0.01)
                    outputs.append(
                        {
                            "logits": second.logits.detach().clone(),
                            "pre_norm": second.pre_norm.detach().clone(),
                            "key": second.cache.key.detach().clone(),
                            "value": second.cache.value.detach().clone(),
                            "loss": loss.detach().clone(),
                            "feature_gradient": feature.grad.detach().clone(),
                            "gradients": gradients,
                            "missing_gradients": missing_gradients,
                            "updated": {name: p.detach().clone() for name, p in named.items()},
                        }
                    )
                for key in (
                    "logits",
                    "pre_norm",
                    "key",
                    "value",
                    "loss",
                    "feature_gradient",
                ):
                    torch.testing.assert_close(
                        outputs[0][key], outputs[1][key], atol=2e-6, rtol=5e-5
                    )
                _assert_tree_close(self, outputs[0]["gradients"], outputs[1]["gradients"])
                self.assertEqual(outputs[0]["missing_gradients"], outputs[1]["missing_gradients"])
                _assert_tree_close(self, outputs[0]["updated"], outputs[1]["updated"])

    def test_only_adjacent_groups_reduce_quantizer_calls_each_step(self):
        observed = []
        from w1a1_eagle import learned_activation as learned_module

        original = learned_module.learned_activation
        for adapter in _adapters(8):
            counts: dict[str, int] = {boundary: 0 for boundary in BOUNDARY_PATHS}
            parameter_boundaries = {
                id(adapter.linears[paths[0]].activation_quantizer.parameter): boundary
                for boundary, paths in BOUNDARY_PATHS.items()
            }

            def counted(input, bits, parameter, *, valid_mask=None, normalization_count=None):
                counts[parameter_boundaries[id(parameter)]] += 1
                return original(
                    input,
                    bits,
                    parameter,
                    valid_mask=valid_mask,
                    normalization_count=normalization_count,
                )

            with patch.object(learned_module, "learned_activation", counted):
                _step(adapter, 1)
            observed.append(counts.copy())
            events = adapter.activation_reuse_state()["last_decode_events"]
            if adapter.activation_reuse:
                self.assertEqual(
                    [
                        (row["boundary"], row["hits"], row["misses"], row["retained_entries"])
                        for row in events
                    ],
                    [("qkv", 2, 1, 0), ("gate_up", 1, 1, 0)],
                )
            else:
                self.assertEqual(events, [])
        base, optimized = observed
        for boundary in BOUNDARY_PATHS:
            expected_delta = {"qkv": 4, "gate_up": 2}.get(boundary, 0)
            self.assertEqual(base[boundary] - optimized[boundary], expected_delta, boundary)
        self.assertEqual((base["qkv"], optimized["qkv"]), (6, 2))
        self.assertEqual((base["gate_up"], optimized["gate_up"]), (4, 2))
        for boundary in ("fc",):
            self.assertEqual((base[boundary], optimized[boundary]), (0, 0), boundary)
        for boundary in ("attn_output", "down", "head"):
            self.assertEqual((base[boundary], optimized[boundary]), (2, 2), boundary)

    def test_default_state_dict_and_config_remain_unchanged(self):
        source = _drafter()
        implicit = NativeStepAdapter(copy.deepcopy(source))
        explicit_off = NativeStepAdapter(copy.deepcopy(source), activation_reuse=False)
        enabled = NativeStepAdapter(copy.deepcopy(source), activation_reuse=True)
        self.assertEqual(implicit.state_dict().keys(), explicit_off.state_dict().keys())
        self.assertEqual(implicit.state_dict().keys(), enabled.state_dict().keys())
        _assert_state_equal(self, implicit.state_dict(), explicit_off.state_dict())
        _assert_state_equal(self, implicit.state_dict(), enabled.state_dict())

    def test_result_references_are_released_between_steps_and_on_exception(self):
        _baseline, adapter = _adapters()
        quantizer = adapter.linears["midlayer.self_attn.q_proj"].activation_quantizer
        results: list[weakref.ReferenceType] = []
        hook = quantizer.register_forward_hook(
            lambda _module, _args, result: results.append(weakref.ref(result))
        )
        failing = adapter.drafter.midlayer.self_attn.k_proj.register_forward_pre_hook(
            lambda *_: (_ for _ in ()).throw(RuntimeError("injected QKV failure"))
        )
        feature = torch.tensor([0.2, -0.4, 0.1, 0.5, -0.3, 0.7, -0.2, 0.6])
        try:
            with self.assertRaisesRegex(RuntimeError, "injected QKV failure"):
                adapter.decode_step(3, feature, 0, adapter.new_cache())
        finally:
            failing.remove()
        gc.collect()
        self.assertTrue(results)
        self.assertTrue(all(reference() is None for reference in results))
        results.clear()
        first = adapter.decode_step(3, feature, 0, adapter.new_cache())
        gc.collect()
        self.assertTrue(all(reference() is None for reference in results))
        second = adapter.decode_step(4, first.pre_norm, 1, first.cache)
        self.assertIsNotNone(second.logits.grad_fn)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in results))
        hook.remove()

    def test_serial_learned_head_path_is_unchanged_by_reuse_flag(self):
        baseline, reused = _adapters(8)
        round_data = provider_batch(parent=1, hidden=8, terminal=True)
        results = []
        head_calls = []
        hooks = [
            adapter.drafter.lm_head.register_forward_pre_hook(
                lambda _module, args: head_calls.append(tuple(args[0].shape))
            )
            for adapter in (baseline, reused)
        ]
        try:
            for adapter in (baseline, reused):
                raw = round_data.raw_target_features.detach().clone().requires_grad_()
                candidate = replace(round_data, raw_target_features=raw)
                logits = forward_torch_round(candidate, adapter, 7)
                logits.square().sum().backward()
                results.append(
                    (
                        logits.detach().clone(),
                        raw.grad.detach().clone(),
                        adapter.linears["lm_head"]
                        .activation_quantizer.parameter.grad.detach()
                        .clone(),
                    )
                )
        finally:
            for hook in hooks:
                hook.remove()
        for left, right in zip(results[0], results[1], strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=5e-5)
        self.assertEqual(len(head_calls), 6)
        self.assertEqual(head_calls[:3], head_calls[3:])


if __name__ == "__main__":
    unittest.main()
