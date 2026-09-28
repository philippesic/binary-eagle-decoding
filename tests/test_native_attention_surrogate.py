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


class NativeAttentionSurrogateTests(unittest.TestCase):
    def test_native_forward_is_returned_exactly_and_padded_mask_is_causal(self):
        query = torch.tensor([[0.25, -0.5], [0.5, 0.125]], requires_grad=True)
        keys = torch.tensor([[[0.5, 0.25], [0.125, -0.5]]], requires_grad=True)
        values = torch.tensor([[[0.25, 0.5], [0.75, 0.125]]], requires_grad=True)
        native = torch.tensor([[0.03125, 0.0625], [-0.125, 0.25]], dtype=torch.float32)
        calls = []

        def oracle(q, k, v, mask):
            calls.append((q, k, v, mask))
            return native.clone()

        output = native_forward_f32_backward(query, keys, values, oracle)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(output, native))
        q, k, v, mask = calls[0]
        self.assertEqual(q.shape, (2, 2))
        self.assertEqual(k.shape, (1, NATIVE_ATTENTION_SLOTS, 2))
        self.assertEqual(v.shape, k.shape)
        self.assertEqual(mask.shape, (1, NATIVE_ATTENTION_SLOTS))
        self.assertEqual(mask.dtype, torch.float16)
        torch.testing.assert_close(q, query)
        torch.testing.assert_close(k[:, :2], keys)
        torch.testing.assert_close(v[:, :2], values)
        self.assertTrue(torch.equal(k[:, 2:], torch.zeros_like(k[:, 2:])))
        self.assertTrue(torch.equal(v[:, 2:], torch.zeros_like(v[:, 2:])))
        self.assertTrue(torch.equal(mask[0, :2], torch.zeros(2, dtype=torch.float16)))
        self.assertTrue(torch.isneginf(mask[0, 2:]).all())

    def test_backward_equals_original_f32_attention_surrogate(self):
        query = torch.tensor([[0.25, -0.5], [0.5, 0.125]], requires_grad=True)
        keys = torch.tensor([[[0.5, 0.25], [0.125, -0.5]]], requires_grad=True)
        values = torch.tensor([[[0.25, 0.5], [0.75, 0.125]]], requires_grad=True)
        incoming = torch.tensor([[0.25, -0.5], [-0.25, 0.75]])
        native = native_forward_f32_backward(
            query, keys, values, lambda *_: torch.full((2, 2), 13.0)
        )
        self.assertTrue(torch.equal(native, torch.full((2, 2), 13.0)))
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
        query = torch.zeros(2, 2)
        keys = torch.zeros(1, 1, 2)
        values = torch.zeros_like(keys)

        def oracle(*_):
            return torch.zeros(2, 2)

        with self.assertRaisesRegex(ValueError, "query"):
            native_forward_f32_backward(torch.full((2, 2), float("nan")), keys, values, oracle)
        with self.assertRaisesRegex(ValueError, "head geometry"):
            native_forward_f32_backward(query, torch.zeros(3, 1, 2), values, oracle)
        with self.assertRaisesRegex(ValueError, "prefix"):
            native_forward_f32_backward(
                query,
                torch.zeros(1, NATIVE_ATTENTION_SLOTS + 1, 2),
                torch.zeros(1, NATIVE_ATTENTION_SLOTS + 1, 2),
                oracle,
            )
        with self.assertRaisesRegex(ValueError, "F16-exact"):
            native_forward_f32_backward(query, torch.full((1, 1, 2), 0.1), values, oracle)
        with self.assertRaisesRegex(ValueError, "native attention output"):
            native_forward_f32_backward(query, keys, values, lambda *_: torch.zeros(2, 3))
        drafter = _drafter()
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

        drafter = _drafter()
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
