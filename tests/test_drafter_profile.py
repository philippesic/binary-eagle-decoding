"""CPU checks for temporary draft-path instrumentation and accounting."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from scripts.profile_drafter_layers import (
    DrafterEventProfiler,
    aggregate_repetitions,
    eligible_linears,
    summarize_spans,
)


class FakeEvent:
    tick = 0

    def record(self):
        self.time = FakeEvent.tick
        FakeEvent.tick += 1

    def elapsed_time(self, other):
        return float(other.time - self.time)


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(4, 4, bias=False))

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(4, 4, bias=False))

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) + self.up_proj(x))


class MockDrafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pretraining_tp=1)
        self.fc = nn.Linear(4, 4, bias=False)
        self.midlayer = nn.Module()
        self.midlayer.self_attn = MockAttention()
        self.midlayer.mlp = MockMLP()
        self.lm_head = nn.Linear(4, 8, bias=False)

    def topK_genrate(self, x):
        x = self.fc(x)
        x = self.midlayer.self_attn(x)
        x = self.midlayer.mlp(x)
        return self.lm_head(x)


class DrafterProfileTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.tick = 0
        self.drafter = MockDrafter()
        self.x = torch.ones(2, 4)

    def test_observes_all_groups_shapes_and_restores_hooks(self):
        original = self.drafter.topK_genrate.__func__
        with DrafterEventProfiler(self.drafter, None, FakeEvent, lambda: None) as profiler:
            self.drafter.topK_genrate(self.x)
            result = profiler.result()
        self.assertIs(self.drafter.topK_genrate.__func__, original)
        self.assertEqual(result["draft_invocations"], 1)
        self.assertEqual(sum(group["calls"] for group in result["groups"].values()), 9)
        self.assertEqual(result["groups"]["attention"]["calls"], 4)
        self.assertEqual(result["groups"]["ffn"]["calls"], 3)
        shape = result["paths"]["fc"]["observed_shapes"][0]
        self.assertEqual(shape["input"], [2, 4])
        self.assertEqual(shape["weight"], [4, 4])
        self.assertEqual(shape["input_dtype"], "torch.float32")
        self.assertAlmostEqual(
            result["binary_eligible_linear_share"] + result["remaining_graph_share"], 1
        )
        self.assertEqual(len(self.drafter.fc._forward_pre_hooks), 0)
        self.assertEqual(len(self.drafter.fc._forward_hooks), 0)

    def test_linear_outside_draft_scope_is_excluded(self):
        with DrafterEventProfiler(self.drafter, None, FakeEvent, lambda: None) as profiler:
            self.drafter.fc(self.x)
            self.drafter.topK_genrate(self.x)
            result = profiler.result()
        self.assertEqual(result["paths"]["fc"]["calls"], 1)

    def test_aggregate_uses_total_time_and_exception_restores_method(self):
        original = self.drafter.topK_genrate.__func__
        results = []
        for _ in range(2):
            with DrafterEventProfiler(self.drafter, None, FakeEvent, lambda: None) as profiler:
                self.drafter.topK_genrate(self.x)
                results.append(profiler.result())
        aggregate = aggregate_repetitions(results)
        self.assertEqual(aggregate["draft_invocations"], 2)
        self.assertEqual(aggregate["groups"]["lm_head"]["calls"], 2)
        self.assertAlmostEqual(
            aggregate["binary_eligible_linear_share"] + aggregate["remaining_graph_share"],
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            with DrafterEventProfiler(self.drafter, None, FakeEvent, lambda: None):
                raise RuntimeError("expected failure")
        self.assertIs(self.drafter.topK_genrate.__func__, original)
        self.assertEqual(len(self.drafter.fc._forward_hooks), 0)

    def test_structure_and_target_alias_are_rejected(self):
        self.drafter.config.pretraining_tp = 2
        with self.assertRaisesRegex(ValueError, "pretraining_tp"):
            eligible_linears(self.drafter)
        self.drafter.config.pretraining_tp = 1
        target = nn.Module()
        target.fc = self.drafter.fc
        with self.assertRaisesRegex(ValueError, "target-owned"):
            eligible_linears(self.drafter, target)
        self.drafter.midlayer.self_attn.q_proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, "nn.Linear"):
            eligible_linears(self.drafter)

    def test_unknown_path_and_negative_time_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown profiled path"):
            summarize_spans([{"path": "other", "elapsed_ms": 1}], [2])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            summarize_spans([], [-1])


if __name__ == "__main__":
    unittest.main()
