"""Small CPU-only structural checks for the one-step binary EAGLE adapter."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.native_step import NativeStepAdapter, NativeStepCache  # noqa: E402
from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402
from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache, rollout_captured_prefix  # noqa: E402
from w1a1_eagle.recurrent_trace import TraceAudit  # noqa: E402
from w1a1_eagle.recurrent_training import train_step  # noqa: E402


class TinyNorm(nn.Module):
    def __init__(self, width, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.variance_epsilon = 1e-5


def _drafter():
    generator = torch.Generator(device="cpu").manual_seed(114)
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
    drafter.embed_tokens = nn.Embedding(11, 4, dtype=torch.float16, device="cpu")
    with torch.no_grad():
        drafter.embed_tokens.weight.copy_(torch.randn(11, 4, generator=generator).half() * 0.4)

    def binary(out_features, in_features):
        weights = torch.randn(out_features, in_features, generator=generator) * 0.5
        scales = 0.06 + torch.rand(out_features, 1, generator=generator) * 0.04
        return GroupedBinaryLinear(weights, scales, group_size=128)

    drafter.fc = binary(4, 12)
    drafter.midlayer = nn.Module()
    layer = drafter.midlayer
    layer.input_layernorm = TinyNorm(4, torch.tensor([1.0, 1.1, 0.9, 1.2]))
    layer.hidden_norm = TinyNorm(4, torch.tensor([0.8, 1.0, 1.1, 0.9]))
    layer.post_attention_layernorm = TinyNorm(4, torch.tensor([1.0, 0.9, 1.2, 1.1]))
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = binary(4, 8)
    layer.self_attn.k_proj = binary(2, 8)
    layer.self_attn.v_proj = binary(2, 8)
    layer.self_attn.o_proj = binary(4, 4)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = binary(5, 4)
    layer.mlp.up_proj = binary(5, 4)
    layer.mlp.down_proj = binary(4, 5)
    drafter.norm = TinyNorm(4, torch.tensor([0.9, 1.1, 1.0, 0.8]))
    drafter.lm_head = binary(7, 4)
    return drafter


def _manual_linear(module, x):
    """Independent hard-sign matrix reference for this one-group fixture."""
    operand = x.half().float()
    signs = torch.where(module.latent_sign.detach() < 0, -1.0, 1.0)
    return (signs * operand[None, :]).sum(-1) * module.effective_scales().detach()[:, 0]


def _manual_norm(x, norm):
    return x * torch.rsqrt(torch.mean(x * x) + norm.variance_epsilon) * norm.weight.detach()


def _manual_step(drafter, token, feature, position, keys, values):
    """Compute a step without NativeStepAdapter helpers or its cache type."""
    layer = drafter.midlayer
    attn = layer.self_attn
    emb = drafter.embed_tokens.weight.detach()[token].float()
    fused = torch.cat(
        (_manual_norm(emb, layer.input_layernorm), _manual_norm(feature, layer.hidden_norm))
    )
    q = _manual_linear(attn.q_proj, fused).reshape(2, 2)
    k = _manual_linear(attn.k_proj, fused).reshape(1, 2)
    v = _manual_linear(attn.v_proj, fused).reshape(1, 2)
    angle = torch.tensor(float(position))
    c, s = torch.cos(angle), torch.sin(angle)
    q = torch.stack((q[:, 0] * c - q[:, 1] * s, q[:, 1] * c + q[:, 0] * s), -1)
    k = torch.stack((k[:, 0] * c - k[:, 1] * s, k[:, 1] * c + k[:, 0] * s), -1)
    keys = torch.cat((keys, k.half().float()[:, None, :]), dim=1)
    values = torch.cat((values, v.half().float()[:, None, :]), dim=1)
    repeated_keys = keys.repeat_interleave(2, dim=0)
    repeated_values = values.repeat_interleave(2, dim=0)
    scores = (q[:, None, :] * repeated_keys).sum(-1) / math.sqrt(2)
    weighted = (F.softmax(scores, dim=-1)[..., None] * repeated_values).sum(1).reshape(4)
    residual = feature + _manual_linear(attn.o_proj, weighted)
    ff_input = _manual_norm(residual, layer.post_attention_layernorm)
    gate = _manual_linear(layer.mlp.gate_proj, ff_input)
    up = _manual_linear(layer.mlp.up_proj, ff_input)
    pre_norm = residual + _manual_linear(layer.mlp.down_proj, F.silu(gate) * up)
    logits = _manual_linear(drafter.lm_head, _manual_norm(pre_norm, drafter.norm))
    return logits, pre_norm, keys, values


class NativeStepTests(unittest.TestCase):
    def test_prefix_rebuild_and_proposal_rollout_share_current_student(self):
        drafter = _drafter()
        adapter = NativeStepAdapter(drafter)
        raw = torch.tensor(
            [
                [0.6, -0.2, 0.9, -0.3, 0.2, 0.1, 0.4, 0.7, -0.1, 0.8, -0.5, 0.3],
                [0.2, 0.4, -0.1, 0.6, -0.3, 0.8, 0.5, -0.7, 0.9, 0.1, 0.3, -0.2],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        rebuilt = rebuild_prefix_cache(
            [0, 1, 3],
            raw,
            [0, 1],
            parent_position=1,
            encode_feature=adapter.encode_feature,
            decode_context=adapter.decode_step,
            new_cache=adapter.new_cache,
        )
        self.assertEqual(rebuilt.cache.key.shape, (1, 1, 2))
        self.assertFalse(rebuilt.cache.key.requires_grad)
        rows = [
            {
                "prompt_id": "train-0",
                "round_index": 0,
                "parent_position": 1,
                "depth": 0,
                "input_position": 2,
                "input_token_id": 3,
                "proposed_token_id": 5,
                "valid": True,
            },
            {
                "prompt_id": "train-0",
                "round_index": 0,
                "parent_position": 1,
                "depth": 1,
                "input_position": 3,
                "input_token_id": 5,
                "proposed_token_id": 6,
                "valid": True,
            },
        ]
        logits = rollout_captured_prefix(
            rows,
            rebuilt.seed_raw_features,
            encode_feature=adapter.encode_feature,
            decode_step=adapter.decode_step,
            initial_cache=rebuilt.cache,
            draft_vocab_size=7,
        )
        self.assertEqual(logits.shape, (2, 7))
        audit = TraceAudit(
            draft_labels=(-1, 0),
            valid_mask=(True, True),
            supported_mask=(False, True),
            ce_mask=(False, True),
            denominator_mask=(True, True),
            target_to_draft=tuple(range(7)),
            counts={"total": 2},
            per_depth={},
        )
        parameters = [
            parameter
            for linear in adapter.linears.values()
            for parameter in (linear.latent_sign, linear.scale_offset)
        ]
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        loss = train_step(adapter.linears, logits, audit, optimizer)
        self.assertGreater(loss, 0)
        self.assertGreater(float(drafter.fc.scale_offset.grad.abs().sum()), 0)
        self.assertGreater(float(drafter.lm_head.scale_offset.grad.abs().sum()), 0)

    def test_two_steps_match_independent_manual_reference(self):
        drafter = _drafter()
        adapter = NativeStepAdapter(drafter)
        raw = torch.tensor(
            [0.6, -0.2, 0.9, -0.3, 0.2, 0.1, 0.4, 0.7, -0.1, 0.8, -0.5, 0.3],
            dtype=torch.float32,
            device="cpu",
        )
        feature = adapter.encode_feature(raw)
        torch.testing.assert_close(feature, _manual_linear(drafter.fc, raw), rtol=0, atol=2e-7)
        cache = adapter.new_cache()
        keys, values = cache.key, cache.value
        for position, token in ((0, 3), (1, 5)):
            expected_logits, expected_state, keys, values = _manual_step(
                drafter, token, feature.detach(), position, keys, values
            )
            step = adapter.decode_step(token, feature, position, cache)
            torch.testing.assert_close(step.logits, expected_logits, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(step.pre_norm, expected_state, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(step.cache.key, keys, rtol=0, atol=0)
            torch.testing.assert_close(step.cache.value, values, rtol=0, atol=0)
            self.assertTrue(torch.equal(step.cache.key, step.cache.key.half().float()))
            self.assertTrue(torch.equal(step.cache.value, step.cache.value.half().float()))
            cache, feature = step.cache, step.pre_norm

    def test_later_loss_reaches_earlier_kv_and_binary_body(self):
        drafter = _drafter()
        adapter = NativeStepAdapter(drafter)
        first = adapter.decode_step(
            3,
            adapter.encode_feature(
                torch.tensor([0.6, -0.2, 0.9, -0.3, 0.2, 0.1, 0.4, 0.7, -0.1, 0.8, -0.5, 0.3])
            ),
            0,
            adapter.new_cache(),
        )
        first.cache.key.retain_grad()
        first.cache.value.retain_grad()
        first.pre_norm.retain_grad()
        second = adapter.decode_step(5, first.pre_norm, 1, first.cache)
        second.logits.square().sum().backward()
        self.assertGreater(float(first.cache.key.grad.abs().sum()), 0)
        self.assertGreater(float(first.cache.value.grad.abs().sum()), 0)
        self.assertGreater(float(first.pre_norm.grad.abs().sum()), 0)
        self.assertGreater(float(drafter.fc.scale_offset.grad.abs().sum()), 0)
        for module in (
            drafter.embed_tokens,
            drafter.midlayer.input_layernorm,
            drafter.midlayer.hidden_norm,
            drafter.midlayer.post_attention_layernorm,
            drafter.norm,
        ):
            self.assertTrue(
                all(
                    not parameter.requires_grad and parameter.grad is None
                    for parameter in module.parameters()
                )
            )

    def test_rejects_wrong_cache_position_tree_and_shapes(self):
        drafter = _drafter()
        adapter = NativeStepAdapter(drafter)
        cache = adapter.new_cache()
        feature = adapter.encode_feature(torch.ones(12, dtype=torch.float32, device="cpu"))
        with self.assertRaisesRegex(ValueError, "prior rows"):
            adapter.decode_step(1, feature, 1, cache)
        with self.assertRaisesRegex(ValueError, "decoder_position"):
            adapter.decode_step(1, feature, -1, cache)
        with self.assertRaisesRegex(ValueError, "feature"):
            adapter.decode_step(1, feature.half(), 0, cache)
        with self.assertRaisesRegex(ValueError, "raw target feature"):
            adapter.encode_feature(torch.ones((1, 12), dtype=torch.float32, device="cpu"))
        bad = NativeStepCache(torch.ones(1, 0, 2, dtype=torch.float16), cache.value)
        with self.assertRaisesRegex(ValueError, "key cache"):
            adapter.decode_step(1, feature, 0, bad)
        drafter.tree_mask = torch.ones((1, 1, 1, 1), dtype=torch.bool, device="cpu")
        with self.assertRaisesRegex(ValueError, "tree mask"):
            adapter.decode_step(1, feature, 0, cache)

    def test_rejects_unsupported_config_and_nonbinary_projection(self):
        drafter = _drafter()
        drafter.config.pretraining_tp = 2
        with self.assertRaisesRegex(ValueError, "pretraining_tp"):
            NativeStepAdapter(drafter)
        drafter.config.pretraining_tp = 1
        drafter.config.rope_scaling = {"type": "linear", "factor": 2}
        with self.assertRaisesRegex(ValueError, "scaled RoPE"):
            NativeStepAdapter(drafter)
        drafter.config.rope_scaling = None
        drafter.midlayer.self_attn.q_proj = nn.Linear(8, 4, device="cpu")
        with self.assertRaisesRegex(ValueError, "group-128"):
            NativeStepAdapter(drafter)


if __name__ == "__main__":
    unittest.main()
