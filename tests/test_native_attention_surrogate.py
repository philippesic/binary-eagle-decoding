"""CPU checks for exact native attention forward and F32 surrogate backward."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(ROOT / "tests"))

from test_native_step import _drafter  # noqa: E402

from w1a1_eagle.native_attention_oracle import (  # noqa: E402
    NATIVE_ATTENTION_SLOTS,
    NativeCPUAttentionOracle,
    f32_attention_surrogate,
    native_forward_f32_backward,
)
from w1a1_eagle.native_step import NativeStepAdapter  # noqa: E402
from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402


def _drafter_native_width():
    drafter = _drafter()
    drafter.config.head_dim = 128
    generator = torch.Generator(device="cpu").manual_seed(291)

    def binary(out_features, in_features):
        weights = torch.randn(out_features, in_features, generator=generator) * 0.5
        scales = torch.full((out_features, (in_features + 127) // 128), 0.08)
        return GroupedBinaryLinear(weights, scales, group_size=128)

    attention = drafter.midlayer.self_attn
    attention.q_proj = binary(256, 8)
    attention.k_proj = binary(128, 8)
    attention.v_proj = binary(128, 8)
    attention.o_proj = binary(4, 256)
    return drafter


class NativeAttentionSurrogateTests(unittest.TestCase):
    def test_native_forward_is_returned_exactly_and_padded_mask_is_causal(self):
        query = (torch.arange(256, dtype=torch.float32).reshape(2, 128) / 256).requires_grad_()
        key_rows = torch.arange(128, dtype=torch.float32) / 128
        keys = torch.stack((key_rows, -key_rows))[None].requires_grad_()
        values = torch.stack((key_rows / 2, key_rows / 4))[None].requires_grad_()
        native = torch.arange(256, dtype=torch.float32).reshape(2, 128) / 512
        calls = []

        def oracle(q, k, v, mask):
            calls.append((q, k, v, mask))
            return native.clone()

        output = native_forward_f32_backward(query, keys, values, oracle)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(output, native))
        q, k, v, mask = calls[0]
        self.assertEqual(q.shape, (2, 128))
        self.assertEqual(k.shape, (1, NATIVE_ATTENTION_SLOTS, 128))
        self.assertEqual(v.shape, k.shape)
        self.assertEqual(mask.shape, (1, NATIVE_ATTENTION_SLOTS))
        self.assertEqual(mask.dtype, torch.float16)
        expected_q = torch.stack((query[:, :64], query[:, 64:]), dim=-1).reshape(2, 128)
        expected_k = torch.stack((keys[..., :64], keys[..., 64:]), dim=-1).reshape(1, 2, 128)
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(k[:, :2], expected_k)
        self.assertEqual(q[0, :4].tolist(), [0.0, 0.25, 1 / 256, 65 / 256])
        torch.testing.assert_close(v[:, :2], values)
        self.assertTrue(torch.equal(k[:, 2:], torch.zeros_like(k[:, 2:])))
        self.assertTrue(torch.equal(v[:, 2:], torch.zeros_like(v[:, 2:])))
        self.assertTrue(torch.equal(mask[0, :2], torch.zeros(2, dtype=torch.float16)))
        self.assertTrue(torch.isneginf(mask[0, 2:]).all())

    def test_backward_equals_original_f32_attention_surrogate(self):
        generator = torch.Generator(device="cpu").manual_seed(292)
        query = (torch.randn(2, 128, generator=generator) / 8).requires_grad_()
        keys = (torch.randn(1, 2, 128, generator=generator) / 8).half().float().requires_grad_()
        values = (torch.randn(1, 2, 128, generator=generator) / 8).half().float().requires_grad_()
        incoming = torch.randn(2, 128, generator=generator)
        native = native_forward_f32_backward(
            query, keys, values, lambda *_: torch.full((2, 128), 13.0)
        )
        self.assertTrue(torch.equal(native, torch.full((2, 128), 13.0)))
        native_grad = torch.autograd.grad(native, (query, keys, values), incoming)
        reference = f32_attention_surrogate(query, keys, values)
        reference_grad = torch.autograd.grad(reference, (query, keys, values), incoming)
        for actual, expected in zip(native_grad, reference_grad, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(float(native_grad[0].abs().sum()), 0)
        self.assertGreater(float(native_grad[1].abs().sum()), 0)
        self.assertGreater(float(native_grad[2].abs().sum()), 0)

    def test_default_adapter_keeps_f32_attention_and_rejects_unpaired_oracle(self):
        drafter = _drafter()
        adapter = NativeStepAdapter(drafter)
        self.assertEqual(adapter.attention_mode, "f32")
        self.assertIsNone(adapter.native_attention_oracle)
        feature = adapter.encode_feature(torch.ones(12))
        normal = adapter.decode_step(1, feature, 0, adapter.new_cache())
        explicit = NativeStepAdapter(_drafter(), attention_mode="f32")
        other = explicit.decode_step(
            1, explicit.encode_feature(torch.ones(12)), 0, explicit.new_cache()
        )
        torch.testing.assert_close(normal.logits, other.logits, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "requires native attention mode"):
            NativeStepAdapter(_drafter(), native_attention_oracle=lambda *_: torch.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, "callable oracle"):
            NativeStepAdapter(_drafter(), attention_mode="native_forward_f32_backward")

    def test_rejects_nonfinite_shape_and_prefix_overflow(self):
        query = torch.zeros(2, 128)
        keys = torch.zeros(1, 1, 128)
        values = torch.zeros_like(keys)

        def oracle(*_):
            return torch.zeros(2, 128)

        with self.assertRaisesRegex(ValueError, "query"):
            native_forward_f32_backward(torch.full((2, 128), float("nan")), keys, values, oracle)
        with self.assertRaisesRegex(ValueError, "head_dim=128"):
            native_forward_f32_backward(
                torch.zeros(2, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), oracle
            )
        with self.assertRaisesRegex(ValueError, "head geometry"):
            native_forward_f32_backward(query, torch.zeros(3, 1, 128), values, oracle)
        with self.assertRaisesRegex(ValueError, "prefix"):
            native_forward_f32_backward(
                query,
                torch.zeros(1, NATIVE_ATTENTION_SLOTS + 1, 128),
                torch.zeros(1, NATIVE_ATTENTION_SLOTS + 1, 128),
                oracle,
            )
        with self.assertRaisesRegex(ValueError, "F16-exact"):
            native_forward_f32_backward(query, torch.full((1, 1, 128), 0.1), values, oracle)
        with self.assertRaisesRegex(ValueError, "native attention output"):
            native_forward_f32_backward(query, keys, values, lambda *_: torch.zeros(2, 127))
        drafter = _drafter_native_width()
        drafter.config.max_position_embeddings = 512
        adapter = NativeStepAdapter(
            drafter, attention_mode="native_forward_f32_backward", native_attention_oracle=oracle
        )
        with self.assertRaisesRegex(ValueError, "positions 0..255"):
            adapter.decode_step(1, torch.zeros(4), 256, adapter.new_cache())

    def test_rejects_wrong_helper_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "native-recurrent-attention"
            helper.write_bytes(b"wrong ggml helper")
            with self.assertRaisesRegex(ValueError, "helper identity"):
                NativeCPUAttentionOracle(helper)

    def test_two_step_later_loss_reaches_prior_state_and_kv(self):
        calls = []

        def oracle(q, k, v, mask):
            prefix = int((mask[0] == 0).sum())
            calls.append(prefix)
            return f32_attention_surrogate(q, k[:, :prefix], v[:, :prefix]).detach()

        drafter = _drafter_native_width()
        adapter = NativeStepAdapter(
            drafter,
            attention_mode="native_forward_f32_backward",
            native_attention_oracle=oracle,
        )
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
        self.assertEqual(calls, [1, 2])
        self.assertGreater(float(first.cache.key.grad.abs().sum()), 0)
        self.assertGreater(float(first.cache.value.grad.abs().sum()), 0)
        self.assertGreater(float(first.pre_norm.grad.abs().sum()), 0)
        self.assertGreater(float(drafter.fc.scale_offset.grad.abs().sum()), 0)
        q_gradient = drafter.midlayer.self_attn.q_proj.scale_offset.grad
        self.assertGreater(float(q_gradient.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
