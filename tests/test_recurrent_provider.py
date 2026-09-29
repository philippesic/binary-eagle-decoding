"""Injected CPU provider checks; no model snapshot or capture is opened."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear, pack_signs
from w1a1_eagle.recurrent_provider import (
    CALIBRATION_ONLY_SCOPE,
    ProviderRound,
    bind_teacher_rows,
    calibration_measurement_metadata,
    forward_torch_round,
    train_from_provider,
)
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    install_joint_linears,
)
from w1a1_eagle.recurrent_rollout import DraftStep
from w1a1_eagle.recurrent_trace import LABEL_SOURCE, RoundAnchor, validate_recurrent_trace


def dense_drafter(*, native_shape=False):
    generator = torch.Generator().manual_seed(17)
    d = nn.Module()
    d.config = SimpleNamespace(pretraining_tp=1)
    d.midlayer = nn.Module()
    d.midlayer.self_attn = nn.Module()
    d.midlayer.mlp = nn.Module()
    shapes = {path: (4, 4) for path in CANDIDATE_D_BASE_TO_PATH.values()}
    if native_shape:
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
        d.early_stop_method = None
        d.tree_mask = None
        d.embed_tokens = nn.Embedding(5, 4, dtype=torch.float16)
        d.embed_tokens.weight.requires_grad_(False)
        shapes.update(
            {
                "fc": (4, 12),
                "midlayer.self_attn.q_proj": (4, 8),
                "midlayer.self_attn.k_proj": (2, 8),
                "midlayer.self_attn.v_proj": (2, 8),
                "midlayer.mlp.gate_proj": (5, 4),
                "midlayer.mlp.up_proj": (5, 4),
                "midlayer.mlp.down_proj": (4, 5),
                "lm_head": (3, 4),
            }
        )
        for path in (
            "midlayer.input_layernorm",
            "midlayer.hidden_norm",
            "midlayer.post_attention_layernorm",
            "norm",
        ):
            module = nn.Module()
            module.weight = nn.Parameter(torch.ones(4))
            module.variance_epsilon = 1e-5
            parent = d
            for piece in path.split(".")[:-1]:
                parent = getattr(parent, piece)
            setattr(parent, path.split(".")[-1], module)
    else:
        shapes["lm_head"] = (3, 4)
    for path, (out_features, in_features) in shapes.items():
        module = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.randn(out_features, in_features, generator=generator) * 0.3)
        parent = d
        for piece in path.split(".")[:-1]:
            parent = getattr(parent, piece)
        setattr(parent, path.split(".")[-1], module)
    return d


def row(depth, prefix):
    return {
        "prompt_id": "p",
        "split": "train",
        "round_index": 0,
        "depth": depth,
        "parent_position": 0,
        "input_position": depth + 1,
        "label_position": depth + 2,
        "verifier_row": depth,
        "prefix_token_ids": prefix,
        "input_token_id": prefix[-1],
        "alignment_valid": True,
        "is_bonus": False,
        "valid": True,
        "verifier_reached": True,
        "invalid_reason": None,
        "label_source": LABEL_SOURCE,
        "verifier_token_id": 2,
        "proposed_token_id": 1,
        "label_supported": True,
    }


def provider_round():
    rows = (row(0, [0, 1]), row(1, [0, 1, 1]))
    return ProviderRound(
        RoundAnchor("p", "train", 0, (0,), 1),
        rows,
        (0, 1),
        torch.tensor([[0.2, -0.1, 0.3, 0.4]]),
        (0,),
        "capture-a",
        ("r0", "r1"),
        (
            {"id": "r0", "prompt_id": "p", "capture_id": "capture-a", "prefix_token_ids": [0, 1]},
            {
                "id": "r1",
                "prompt_id": "p",
                "capture_id": "capture-a",
                "prefix_token_ids": [0, 1, 1],
            },
        ),
        {
            "draft_topk_ids": np.array([[1, 0], [1, 0]], dtype=np.int32),
            "draft_topk_probs": np.array([[0.5, 0.2], [0.5, 0.2]], dtype=np.float32),
            "draft_tail_mass": np.array([0.2, 0.2], dtype=np.float32),
            "outside_draft_mass": np.array([0.1, 0.1], dtype=np.float32),
            "next_target_id": np.array([2, 2], dtype=np.int32),
        },
    )


class TinyStep:
    def __init__(self, drafter):
        self.linears = {
            path: drafter.get_submodule(path) for path in CANDIDATE_D_BASE_TO_PATH.values()
        }
        self.first = None

    def new_cache(self):
        return ()

    def encode_feature(self, raw):
        return self.linears["fc"](raw)

    def decode_context(self, token, feature, position, cache):
        return self.decode_step(token, feature, position, cache)

    def decode_step(self, token, feature, position, cache):
        q = self.linears["midlayer.self_attn.q_proj"](feature)
        k = self.linears["midlayer.self_attn.k_proj"](feature)
        v = self.linears["midlayer.self_attn.v_proj"](feature)
        history = sum(old_k * old_v for old_k, old_v in cache) if cache else 0
        fused = self.linears["midlayer.self_attn.o_proj"](q + k + v + history)
        gate = self.linears["midlayer.mlp.gate_proj"](feature)
        up = self.linears["midlayer.mlp.up_proj"](feature)
        state = torch.tanh(feature + fused + self.linears["midlayer.mlp.down_proj"](gate * up))
        if self.first is None:
            self.first = (state, k, v)
            for item in self.first:
                item.retain_grad()
        return DraftStep(self.linears["lm_head"](state), state, (*cache, (k, v)))


class TinyProvider:
    split = "train"
    training_eligible = True
    allowed_prompt_ids = {"p"}
    target_vocab_size = 4
    draft_vocab_size = 3
    max_depth = 2
    d2t_offsets = (0, 1, 1)
    base_gguf_sha256 = "a" * 64
    candidate_d = None

    def __init__(self):
        self.drafter = dense_drafter()
        self.adapter = None
        self.load_calls = 0

    def load_models(self):
        self.load_calls += 1
        return self.drafter, nn.Linear(4, 4)

    def make_step_adapter(self, drafter):
        self.adapter = TinyStep(drafter)
        return self.adapter

    def rounds(self):
        yield provider_round()


def tiny_factory(config):
    if config.device != "cpu":
        raise ValueError("tiny provider is CPU-only")
    return TinyProvider()


class ProviderTests(unittest.TestCase):
    def test_install_row_from_original_dense_and_native_structure(self):
        drafter = dense_drafter(native_shape=True)
        original = drafter.fc.weight.detach().clone()
        target = nn.Linear(4, 4)
        config = JointQATConfig(W1AxContract(4))
        linears = install_joint_linears(drafter, target, config)
        self.assertEqual(len(linears), 9)
        self.assertTrue(all(isinstance(m, RowBinaryLinear) for m in linears.values()))
        torch.testing.assert_close(linears["fc"].initial_scale, original.abs().mean(dim=1))
        adapter = NativeStepAdapter(drafter)
        self.assertIs(adapter.linears["fc"], linears["fc"])
        self.assertFalse(drafter.embed_tokens.weight.requires_grad)
        self.assertFalse(drafter.norm.weight.requires_grad)

    def test_target_alias_rejected_before_any_install(self):
        drafter = dense_drafter()
        target = nn.Module()
        target.shared = drafter.fc
        original = drafter.fc
        with self.assertRaisesRegex(ValueError, "target-owned"):
            install_joint_linears(drafter, target, JointQATConfig(W1AxContract(8)))
        self.assertIs(drafter.fc, original)

    def test_meta_device_f32_row_step_shapes_without_accelerator(self):
        drafter = dense_drafter(native_shape=True)
        config = JointQATConfig(W1AxContract(4))
        install_joint_linears(drafter, nn.Linear(4, 4), config)
        drafter.to("meta")
        adapter = NativeStepAdapter(drafter)
        feature = adapter.encode_feature(torch.empty(12, device="meta"))
        step = adapter.decode_step(1, feature, 0, adapter.new_cache())
        self.assertEqual(step.logits.shape, (3,))
        self.assertEqual(step.pre_norm.shape, (4,))
        self.assertEqual(step.cache.key.shape, (1, 1, 2))
        batch = replace(provider_round(), raw_target_features=torch.empty((1, 12), device="meta"))
        self.assertEqual(forward_torch_round(batch, adapter, 3).shape, (2, 3))
        invalid = list(batch.rows)
        invalid[1] = dict(invalid[1], valid=False, proposed_token_id=None)
        terminal = forward_torch_round(replace(batch, rows=tuple(invalid)), adapter, 3)
        self.assertEqual(terminal.device.type, "meta")
        self.assertEqual(terminal.shape, (2, 3))

    def test_group_install_requires_explicit_candidate_d_arrays(self):
        drafter = dense_drafter()
        drafter.midlayer.self_attn.q_proj = nn.Linear(4, 64, bias=False)
        drafter.midlayer.self_attn.k_proj = nn.Linear(4, 16, bias=False)
        config = JointQATConfig(W1AxContract(16, "group128"))
        with self.assertRaisesRegex(ValueError, "candidate-D arrays"):
            install_joint_linears(drafter, nn.Linear(4, 4), config)
        arrays = {}
        for base, path in CANDIDATE_D_BASE_TO_PATH.items():
            linear = drafter.get_submodule(path)
            signs = np.where(linear.weight.detach().numpy() < 0, -1, 1)
            arrays[base] = (pack_signs(signs), np.ones((linear.out_features, 1), dtype=np.float32))
        linears = install_joint_linears(drafter, nn.Linear(4, 4), config, candidate_d=arrays)
        self.assertEqual(len(linears), 9)
        self.assertTrue(all(isinstance(m, GroupedBinaryLinear) for m in linears.values()))

    def test_ineligible_capture_rejected_before_model_loading(self):
        provider = TinyProvider()
        provider.training_eligible = False
        with self.assertRaisesRegex(ValueError, "not approved"):
            train_from_provider(provider, JointQATConfig(W1AxContract(4)), max_rounds=1)
        self.assertEqual(provider.load_calls, 0)

    def test_provider_accelerator_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit allow_accelerator"):
            JointQATConfig(W1AxContract(4), device="cuda")
        config = JointQATConfig(W1AxContract(4), device="cuda", allow_accelerator=True)
        self.assertEqual(config.device, "cuda:0")

    def test_provider_training_preserves_first_state_and_cache_gradients(self):
        for objective in ("hard_ce", "compact_probability"):
            provider = TinyProvider()
            config = JointQATConfig(W1AxContract(4), objective=objective)
            linears, metrics = train_from_provider(provider, config, max_rounds=1)
            self.assertEqual(len(metrics), 1)
            self.assertGreater(metrics[0]["loss"], 0)
            self.assertEqual(metrics[0]["gradient_tensors"], 18)
            self.assertEqual(len(linears), 9)
            for item in provider.adapter.first:
                self.assertGreater(float(item.grad.abs().sum()), 0)

    def test_calibration_metrics_are_scoped_and_report_byte_units(self):
        provider = TinyProvider()
        provider.readiness_scope = CALIBRATION_ONLY_SCOPE
        _, metrics = train_from_provider(provider, JointQATConfig(W1AxContract(16)), max_rounds=1)
        item = metrics[0]
        self.assertGreaterEqual(item["calibration_step_wall_seconds"], 0)
        self.assertIsNone(item["calibration_cuda_allocator_peak_allocated_bytes"])
        self.assertIsNone(item["calibration_cuda_allocator_peak_reserved_bytes"])
        self.assertIsInstance(item["calibration_process_peak_rss_bytes"], int)
        self.assertGreater(item["calibration_process_peak_rss_bytes"], 0)

        ordinary_provider = TinyProvider()
        _, ordinary_metrics = train_from_provider(
            ordinary_provider, JointQATConfig(W1AxContract(16)), max_rounds=1
        )
        self.assertFalse(any(key.startswith("calibration_") for key in ordinary_metrics[0]))
        metadata = calibration_measurement_metadata("cpu")
        self.assertEqual(metadata["device_name"], "CPU")
        self.assertIsNone(metadata["compute_capability"])
        self.assertIn("not whole-device usage", metadata["cuda_memory_source"])

    def test_calibration_process_rss_linux_and_macos_unit_conversions(self):
        from w1a1_eagle import recurrent_provider

        usage = SimpleNamespace(ru_maxrss=1234)
        with (
            patch.object(recurrent_provider.resource, "getrusage", return_value=usage),
            patch.object(recurrent_provider.platform, "system", return_value="Linux"),
        ):
            self.assertEqual(recurrent_provider._process_peak_rss_bytes(), 1234 * 1024)
        with (
            patch.object(recurrent_provider.resource, "getrusage", return_value=usage),
            patch.object(recurrent_provider.platform, "system", return_value="Darwin"),
        ):
            self.assertEqual(recurrent_provider._process_peak_rss_bytes(), 1234)

    def test_wrong_teacher_prefix_rejected(self):
        from dataclasses import replace

        batch = provider_round()
        audit = validate_recurrent_trace(
            batch.rows,
            [batch.anchor],
            offsets=(0, 1, 1),
            target_vocab_size=4,
            allowed_prompt_ids={"p"},
            split="train",
            draft_vocab_size=3,
            max_depth=2,
        )
        bad = replace(
            batch,
            teacher_metadata=(
                dict(batch.teacher_metadata[0], prefix_token_ids=[0, 2]),
                batch.teacher_metadata[1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact prefix mismatch"):
            bind_teacher_rows(bad, audit)
        wrong_label = dict(batch.teacher_arrays)
        wrong_label["next_target_id"] = np.array([2, 3], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "label"):
            bind_teacher_rows(replace(batch, teacher_arrays=wrong_label), audit)


if __name__ == "__main__":
    unittest.main()
