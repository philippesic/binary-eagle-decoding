"""Independent tiny CPU VJP check for NativeStepAdapter and recurrent rollout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.native_step import NativeStepAdapter, NativeStepCache  # noqa: E402
from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402
from w1a1_eagle.recurrent_rollout import (  # noqa: E402
    DraftStep,
    rebuild_prefix_cache,
    rollout_captured_prefix,
)


class TinyNorm(nn.Module):
    def __init__(self, width: int, seed: float):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.8, 1.2, width) * seed)
        self.variance_epsilon = 1e-5


def make_adapter() -> NativeStepAdapter:
    generator = torch.Generator(device="cpu").manual_seed(43021)
    drafter = nn.Module()
    drafter.config = SimpleNamespace(
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
    drafter.early_stop_method = None
    drafter.tree_mask = None
    drafter.embed_tokens = nn.Embedding(9, 4, dtype=torch.float16)
    with torch.no_grad():
        drafter.embed_tokens.weight.copy_(
            torch.randn(9, 4, generator=generator, dtype=torch.float32).mul_(0.4).half()
        )

    def binary(out_features: int, in_features: int) -> GroupedBinaryLinear:
        weights = torch.randn(out_features, in_features, generator=generator) * 0.5
        scales = 0.07 + torch.rand(out_features, 1, generator=generator) * 0.03
        return GroupedBinaryLinear(weights, scales, group_size=128)

    drafter.fc = binary(4, 12)
    drafter.midlayer = nn.Module()
    layer = drafter.midlayer
    layer.input_layernorm = TinyNorm(4, 1.0)
    layer.hidden_norm = TinyNorm(4, 0.9)
    layer.post_attention_layernorm = TinyNorm(4, 1.1)
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = binary(4, 8)
    layer.self_attn.k_proj = binary(2, 8)
    layer.self_attn.v_proj = binary(2, 8)
    layer.self_attn.o_proj = binary(4, 4)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = binary(5, 4)
    layer.mlp.up_proj = binary(5, 4)
    layer.mlp.down_proj = binary(4, 5)
    drafter.norm = TinyNorm(4, 1.0)
    drafter.lm_head = binary(6, 4)
    return NativeStepAdapter(drafter)


def chain(invalid_terminal: bool = False) -> list[dict[str, object]]:
    return [
        {
            "prompt_id": "cpu-fixture",
            "round_index": 0,
            "parent_position": 1,
            "depth": 0,
            "input_position": 2,
            "input_token_id": 3,
            "proposed_token_id": 4,
            "valid": True,
        },
        {
            "prompt_id": "cpu-fixture",
            "round_index": 0,
            "parent_position": 1,
            "depth": 1,
            "input_position": 3,
            "input_token_id": 4,
            "proposed_token_id": None if invalid_terminal else 2,
            "valid": not invalid_terminal,
        },
    ]


def run_case(*, detach_state: bool = False, detach_cache: bool = False, invalid: bool = False):
    torch.manual_seed(0)
    adapter = make_adapter()
    context_observations: list[tuple[bool, bool, bool]] = []
    context_raw = torch.tensor(
        [
            [0.15, -0.20, 0.35, 0.11, -0.27, 0.42, 0.19, -0.31, 0.25, 0.07, -0.13, 0.29],
            [-0.22, 0.17, 0.09, 0.31, 0.14, -0.28, 0.37, 0.05, -0.19, 0.26, 0.12, -0.33],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    def encode_context(row: torch.Tensor) -> torch.Tensor:
        encoded = adapter.encode_feature(row)
        context_observations.append(
            (torch.is_grad_enabled(), encoded.requires_grad, row.requires_grad)
        )
        return encoded

    def decode_context(
        token: int, feature: torch.Tensor, position: int, cache: NativeStepCache
    ) -> DraftStep:
        result = adapter.decode_context(token, feature, position, cache)
        context_observations.append(
            (
                torch.is_grad_enabled(),
                result.cache.key.requires_grad,
                result.cache.value.requires_grad,
            )
        )
        return result

    prefix = rebuild_prefix_cache(
        [0, 3, 4],
        context_raw,
        [0, 1],
        parent_position=1,
        encode_feature=encode_context,
        decode_context=decode_context,
        new_cache=adapter.new_cache,
    )
    assert context_observations and all(
        not enabled and not grad for enabled, grad, _ in context_observations
    )
    assert not prefix.cache.key.requires_grad and not prefix.cache.value.requires_grad

    observations: dict[str, torch.Tensor] = {}
    calls: list[int] = []

    def decode(
        token: int, feature: torch.Tensor, position: int, cache: NativeStepCache
    ) -> DraftStep:
        calls.append(position)
        result = adapter.decode_step(token, feature, position, cache)
        if position == 1:
            result.pre_norm.retain_grad()
            result.cache.key.retain_grad()
            result.cache.value.retain_grad()
            observations.update(
                state=result.pre_norm, key=result.cache.key, value=result.cache.value
            )
            if detach_state:
                result = DraftStep(result.logits, result.pre_norm.detach(), result.cache)
            if detach_cache:
                result = DraftStep(
                    result.logits,
                    result.pre_norm,
                    NativeStepCache(result.cache.key.detach(), result.cache.value.detach()),
                )
        return result

    logits = rollout_captured_prefix(
        chain(invalid),
        prefix.seed_raw_features,
        encode_feature=adapter.encode_feature,
        decode_step=decode,
        initial_cache=prefix.cache,
        draft_vocab_size=6,
    )
    scalar = (logits[1] * torch.tensor([0.4, -0.7, 0.2, 0.9, -0.3, 0.6])).sum()
    if scalar.requires_grad:
        scalar.backward()
    return {
        "adapter": adapter,
        "logits": logits.detach(),
        "scalar": scalar.detach(),
        "calls": calls,
        "raw_feature_grad": context_raw.grad.detach().clone()
        if context_raw.grad is not None
        else None,
        "obs": {
            name: value.grad.detach().clone() if value.grad is not None else None
            for name, value in observations.items()
        },
        "trainable_grads": {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in adapter.named_parameters()
        },
    }


class RecurrentVJPValidation(unittest.TestCase):
    def test_later_depth_loss_reaches_early_state_and_kv(self):
        result = run_case()
        for name in ("state", "key", "value"):
            grad = result["obs"][name]
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().max()), 1e-8, name)

    def test_detaching_state_or_cache_breaks_corresponding_vjp(self):
        baseline = run_case()
        state_cut = run_case(detach_state=True)
        cache_cut = run_case(detach_cache=True)
        torch.testing.assert_close(baseline["logits"], state_cut["logits"], atol=0, rtol=0)
        torch.testing.assert_close(baseline["logits"], cache_cut["logits"], atol=0, rtol=0)
        self.assertGreater(float(baseline["obs"]["state"].abs().max()), 1e-8)
        self.assertLessEqual(float(state_cut["obs"]["state"].abs().max()), 1e-8)
        self.assertGreater(float(state_cut["obs"]["key"].abs().max()), 1e-8)
        self.assertGreater(float(state_cut["obs"]["value"].abs().max()), 1e-8)
        self.assertGreater(float(cache_cut["obs"]["state"].abs().max()), 1e-8)
        for name in ("key", "value"):
            # Detaching returned cache blocks the later-attention contribution,
            # while this step's own attention still supplies an early-cache VJP.
            delta = (baseline["obs"][name] - cache_cut["obs"][name]).abs().max()
            self.assertGreater(float(delta), 1e-8, name)
        self.assertEqual(float(baseline["raw_feature_grad"][0].abs().max()), 0.0)
        self.assertGreater(float(baseline["raw_feature_grad"][1].abs().max()), 1e-8)

    def test_invalid_terminal_row_skips_decoder_and_has_no_loss_gradient(self):
        invalid = run_case(invalid=True)
        one_depth = run_case()
        self.assertEqual(invalid["calls"], [1])
        torch.testing.assert_close(invalid["logits"][1], torch.zeros(6), atol=0, rtol=0)
        torch.testing.assert_close(
            invalid["scalar"], torch.zeros_like(invalid["scalar"]), atol=0, rtol=0
        )
        for name, grad in invalid["trainable_grads"].items():
            if grad is not None:
                self.assertLessEqual(float(grad.abs().max()), 0.0, name)
        self.assertGreater(float(one_depth["scalar"].abs()), 1e-8)


if __name__ == "__main__":
    unittest.main()
