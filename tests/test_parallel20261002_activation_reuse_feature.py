"""Focused scoped invalidation and actual NativeStep operation/storage gates."""

import gc
import unittest
import weakref
from collections import Counter, defaultdict
from unittest.mock import patch

import torch

import w1a1_eagle.learned_activation as learned
from research.parallel20261002.auxiliary_vjp.reference.audit import (
    batch,
    combined,
    compare,
    snapshot,
)
from w1a1_eagle.activation_reuse import activation_reuse_group
from w1a1_eagle.learned_activation import LearnedActivationQuantizer
from w1a1_eagle.native_step import NativeStepAdapter


def scope(q, x, *, enabled=True, events=None):
    return activation_reuse_group("qkv", x, (q, q, q), enabled=enabled, events=events)


class ScopedActivationReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.q = LearnedActivationQuantizer(4, "qkv", 4)
        self.x = torch.tensor([[-0.8, -0.1, 0.3, 0.9]], requires_grad=True)

    def test_only_exact_declared_input_and_quantizer_share(self):
        with scope(self.q, self.x) as state:
            first = self.q(self.x)
            self.assertIs(self.q(self.x), first)
            self.assertIsNot(self.q(self.x.view_as(self.x)), first)
            other = LearnedActivationQuantizer(4, "qkv", 4)
            self.assertIsNot(other(self.x), first)
            self.assertEqual((state.hits, state.misses, len(state.entries)), (1, 1, 1))
        self.assertEqual(state.entries, {})
        self.assertIsNone(state.input)
        self.assertIsNone(state.quantizer)
        self.assertIsNot(self.q(self.x), first)

    def test_input_parameter_and_result_versions_invalidate(self):
        with scope(self.q, self.x) as state:
            first = self.q(self.x)
            with torch.no_grad():
                self.x.add_(0.2)
            changed = self.q(self.x)
            self.assertIsNot(changed, first)
            with torch.no_grad():
                self.q.parameter.fill_(0.7)
            updated = self.q(self.x)
            self.assertIsNot(updated, changed)
            with torch.no_grad():
                updated.codes.zero_()
            self.assertIsNot(self.q(self.x), updated)
            self.assertEqual(len(state.entries), 1)

    def test_mask_and_normalization_identity_and_validation(self):
        mask = torch.tensor([True])
        with scope(self.q, self.x):
            first = self.q(self.x, valid_mask=mask, normalization_count=4)
            self.assertIs(self.q(self.x, valid_mask=mask, normalization_count=4), first)
            self.assertIsNot(self.q(self.x, valid_mask=mask, normalization_count=8), first)
            with torch.no_grad():
                mask.fill_(False)
            self.assertIsNot(self.q(self.x, valid_mask=mask, normalization_count=4), first)
            with self.assertRaises(ValueError):
                self.q(self.x, normalization_count=4.0)
            with self.assertRaises(TypeError):
                self.q(self.x, unsupported=True)

    def test_execution_context_and_quantizer_contract_invalidate(self):
        with scope(self.q, self.x):
            first = self.q(self.x)
            with torch.no_grad():
                self.assertIsNot(self.q(self.x), first)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                autocast = self.q(self.x)
                self.assertIsNot(autocast, first)
                self.assertIs(self.q(self.x), autocast)
            self.q.eval()
            self.assertIsNot(self.q(self.x), autocast)
            normal = self.q(self.x)
            self.q.bits = 4.0
            with self.assertRaises(ValueError):
                self.q(self.x)
            self.q.bits = 4
            self.assertIs(self.q(self.x), normal)

    def test_inference_tensors_fail_closed_without_retained_results(self):
        with torch.inference_mode():
            x = torch.ones(1, 4)
            with scope(self.q, x) as state:
                self.assertIsNot(self.q(x), self.q(x))
                self.assertEqual(state.entries, {})
                self.assertEqual((state.hits, state.misses), (0, 2))

    def test_disabled_unshared_and_wrong_boundary_are_ineffective(self):
        with scope(self.q, self.x, enabled=False) as state:
            self.assertIsNone(state)
            self.assertIsNot(self.q(self.x), self.q(self.x))
        other = LearnedActivationQuantizer(4, "qkv", 4)
        with activation_reuse_group("qkv", self.x, (self.q, other, self.q), enabled=True):
            self.assertIsNot(self.q(self.x), self.q(self.x))
        with self.assertRaises(ValueError):
            with activation_reuse_group("head", self.x, (self.q,), enabled=True):
                pass
        with self.assertRaises(ValueError):
            with scope(self.q, self.x, enabled=1):
                pass

    def test_exception_cleanup_drops_scope_owned_tensors_and_context(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with scope(self.q, self.x, events=events) as state:
                result = self.q(self.x)
                reference = weakref.ref(result)
                del result
                raise RuntimeError("injected")
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(state.entries, {})
        self.assertIsNone(state.input)
        self.assertIsNone(state.quantizer)
        self.assertEqual(events, [dict(boundary="qkv", hits=0, misses=1, retained_entries=0)])
        with scope(self.q, self.x):
            self.q(self.x)


def graph_census(adapter):
    """Track actual computes and unique STE support/derivative storage only."""
    computations = Counter()
    storage = defaultdict(set)
    parameter_boundaries = {
        id(m.activation_quantizer.parameter): m.activation_quantizer.boundary
        for m in adapter.linears.values()
    }
    original = learned.learned_activation

    def counted(input, bits, parameter, **kwargs):
        result = original(input, bits, parameter, **kwargs)
        if torch.is_grad_enabled():
            boundary = parameter_boundaries[id(parameter)]
            computations[boundary] += 1
            storage[boundary].update(
                t.untyped_storage().data_ptr() for t in result.values.grad_fn.saved_tensors
            )
        return result

    with patch.object(learned, "learned_activation", counted):
        actual = snapshot(
            batch(parent=2, depth=3, terminal=True), adapter, cache=True, chunk=2, shared=True
        )
    return actual, computations, {k: len(v) for k, v in storage.items()}


class NativeStepActivationReuseTests(unittest.TestCase):
    def test_combined_graph_preserves_fusion_affine_serial_head_and_reduces_storage(self):
        torch.set_num_threads(1)
        # One representative learned profile, with real raw FC correction and
        # affine sum sharing, decisively exercises the production source path.
        baseline = combined(4, "single_forward")
        optimized = NativeStepAdapter(combined(4, "single_forward").drafter, activation_reuse=True)
        expected, base_calls, base_storage = graph_census(baseline)
        actual, reuse_calls, reuse_storage = graph_census(optimized)
        compare(actual, expected)
        torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
        self.assertEqual((base_calls["qkv"], reuse_calls["qkv"]), (9, 3))
        self.assertEqual((base_calls["gate_up"], reuse_calls["gate_up"]), (6, 3))
        self.assertEqual((base_storage["qkv"], reuse_storage["qkv"]), (18, 6))
        self.assertEqual((base_storage["gate_up"], reuse_storage["gate_up"]), (12, 6))
        self.assertEqual(reuse_calls["fc"], base_calls["fc"])
        self.assertEqual((base_calls["head"], reuse_calls["head"]), (3, 3))
        self.assertEqual(
            optimized.activation_reuse_state()["last_decode_events"],
            [
                dict(boundary="qkv", hits=2, misses=1, retained_entries=0),
                dict(boundary="gate_up", hits=1, misses=1, retained_entries=0),
            ],
        )

    def test_constructor_flag_is_default_disabled_and_not_serialized(self):
        baseline = combined(4, "single_forward")
        optimized = NativeStepAdapter(combined(4, "single_forward").drafter, activation_reuse=True)
        self.assertFalse(baseline.activation_reuse)
        self.assertEqual(baseline.activation_reuse_state()["enabled_groups"], [])
        self.assertEqual(optimized.activation_reuse_state()["enabled_groups"], ["qkv", "gate_up"])
        self.assertEqual(baseline.state_dict().keys(), optimized.state_dict().keys())
        for key, value in baseline.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, optimized.state_dict()[key], atol=0, rtol=0)
            else:
                self.assertEqual(value, optimized.state_dict()[key])
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            NativeStepAdapter(baseline.drafter, activation_reuse=1)


if __name__ == "__main__":
    unittest.main()
