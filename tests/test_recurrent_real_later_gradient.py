"""Depth-1-only CE must travel through the earlier real-style state/cache."""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_recurrent_real_later_gradient as probe  # noqa: E402

from w1a1_eagle.native_step import NativeStepCache  # noqa: E402
from w1a1_eagle.recurrent_rollout import DraftStep  # noqa: E402
from w1a1_eagle.recurrent_trace import TraceAudit  # noqa: E402


def _row(depth: int, token: int, proposal: int) -> dict:
    return {
        "prompt_id": "qat-revisit-train-test",
        "round_index": 0,
        "parent_position": 1,
        "depth": depth,
        "input_position": depth + 2,
        "input_token_id": token,
        "proposed_token_id": proposal,
        "valid": True,
    }


def _trace(*, valid=(True, True), supported=(True, True), labels=(1, 2)) -> TraceAudit:
    return TraceAudit(
        draft_labels=labels,
        valid_mask=valid,
        supported_mask=supported,
        ce_mask=tuple(a and b for a, b in zip(valid, supported)),
        denominator_mask=valid,
        target_to_draft=(0, 1, 2),
        counts={},
        per_depth={},
    )


class TinyCausalAdapter(nn.Module):
    def __init__(self, *, detach_cache: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.2, 0.8]))
        self.detach_cache = detach_cache
        self.calls = []

    def encode_feature(self, raw: torch.Tensor) -> torch.Tensor:
        return raw * self.scale

    def decode_step(
        self, token: int, feature: torch.Tensor, position: int, cache: NativeStepCache
    ) -> DraftStep:
        self.calls.append((token, position))
        k = torch.tanh(feature * torch.tensor([0.7, -0.5])).half().float()[None, None, :]
        v = (feature * torch.tensor([0.3, 0.9])).half().float()[None, None, :]
        keys = torch.cat((cache.key, k), dim=1)
        values = torch.cat((cache.value, v), dim=1)
        if self.detach_cache and len(self.calls) == 1:
            keys = keys.detach()
        next_cache = NativeStepCache(keys, values)
        q = feature * torch.tensor([0.8, -0.4])
        attention = torch.softmax((keys[0] * q).sum(dim=-1), dim=0) @ values[0]
        state = feature + attention * torch.tensor([0.6, 1.1])
        logits = torch.stack((state[0] + state[1], -state[0] + state[1], state[0] - state[1]))
        return DraftStep(logits, state, next_cache)


class RealLaterGradientTests(unittest.TestCase):
    def setUp(self):
        self.rows = (_row(0, 3, 4), _row(1, 4, 5))

    def test_depth_one_only_loss_reaches_depth_zero_state_and_appended_cache(self):
        adapter = TinyCausalAdapter()
        before = adapter.scale.detach().clone()
        initial = NativeStepCache(torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2)))
        result = probe.later_only_backward(
            self.rows,
            torch.tensor([0.5, -0.25]),
            adapter,
            initial,
            2,
            draft_vocab_size=3,
        )
        self.assertEqual(adapter.calls, [(3, 1), (4, 2)])
        self.assertEqual(result["direct_ce_mask"], [False, True])
        self.assertEqual(result["depth_0_logits_gradient_nonzero"], 0)
        for name in ("pre_norm", "appended_key", "appended_value"):
            with self.subTest(name=name):
                self.assertGreater(result[f"depth_0_{name}"]["l2"], 0)
                self.assertGreater(result[f"depth_0_{name}"]["nonzero"], 0)
        self.assertIsNotNone(adapter.scale.grad)
        self.assertGreater(int(torch.count_nonzero(adapter.scale.grad)), 0)
        torch.testing.assert_close(adapter.scale, before)  # backward, never an optimizer update

    def test_detached_earlier_cache_is_rejected(self):
        adapter = TinyCausalAdapter(detach_cache=True)
        initial = NativeStepCache(torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2)))
        with self.assertRaisesRegex(ValueError, "key cache has no autograd path"):
            probe.later_only_backward(
                self.rows,
                torch.tensor([0.5, -0.25]),
                adapter,
                initial,
                2,
                draft_vocab_size=3,
            )

    def test_audited_labels_and_direct_mask_are_strict(self):
        self.assertEqual(probe.depth_one_label(self.rows, _trace()), 2)
        for trace in (
            _trace(valid=(True, False), labels=(1, -1)),
            _trace(supported=(True, False), labels=(1, -1)),
            _trace(supported=(False, True), labels=(-1, 2)),
        ):
            with self.subTest(trace=trace):
                with self.assertRaisesRegex(ValueError, "valid and supported"):
                    probe.depth_one_label(self.rows, trace)
        with self.assertRaisesRegex(ValueError, "exactly depth-0 and depth-1"):
            probe.depth_one_label((self.rows[1], self.rows[0]), _trace())
        with self.assertRaisesRegex(ValueError, "depth-0 direct loss"):
            probe.depth_one_ce(torch.zeros((2, 3)), 2, (True, True))
        with self.assertRaisesRegex(ValueError, "supported label"):
            probe.depth_one_ce(torch.zeros((2, 3)), -1, (False, True))

    def test_nonfinite_and_zero_causal_gradients_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe._gradient_stats(torch.tensor([float("nan")]), "fixture", require_nonzero=True)
        with self.assertRaisesRegex(ValueError, "zero"):
            probe._gradient_stats(torch.zeros(2), "fixture", require_nonzero=True)
        with self.assertRaisesRegex(ValueError, "missing"):
            probe._gradient_stats(None, "fixture", require_nonzero=False)

    def test_absolute_vocabulary_map_must_match_pinned_d(self):
        offsets = np.zeros(32_000, dtype="<i8")
        expected = hashlib.sha256(np.arange(32_000, dtype="<i8").tobytes()).hexdigest()
        with mock.patch.object(probe, "DRAFT_D_D2T_SHA256", expected):
            self.assertEqual(probe._pinned_absolute_map_hash(offsets), expected)
            offsets[9] = 1
            with self.assertRaisesRegex(ValueError, "map differs"):
                probe._pinned_absolute_map_hash(offsets)
            with self.assertRaisesRegex(ValueError, "32,000-element"):
                probe._pinned_absolute_map_hash(offsets[:9])

    def test_cli_requires_explicit_sources_and_a_new_report(self):
        with mock.patch.object(sys, "argv", ["probe"]), self.assertRaises(SystemExit) as error:
            with contextlib.redirect_stderr(io.StringIO()):
                probe.main()
        self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "already.json"
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "must be new"):
                probe.diagnose(report, report, report, report, report, report)


if __name__ == "__main__":
    unittest.main()
