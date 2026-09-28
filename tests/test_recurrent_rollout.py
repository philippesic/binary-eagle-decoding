"""CPU-only student state, causal cache and position unroll checks."""

import math
import unittest

import torch

from w1a1_eagle.recurrent_binary import GroupedBinaryLinear
from w1a1_eagle.recurrent_loss import supported_prefix_ce
from w1a1_eagle.recurrent_rollout import DraftStep, rollout_captured_prefix
from w1a1_eagle.recurrent_trace import TraceAudit


def row(depth, token, proposal, *, valid=True):
    return {
        "prompt_id": "train-0",
        "round_index": 0,
        "parent_position": 1,
        "depth": depth,
        "input_position": depth + 2,
        "input_token_id": token,
        "proposed_token_id": proposal if valid else None,
        "valid": valid,
    }


def layer(weight):
    return GroupedBinaryLinear(
        torch.tensor(weight, dtype=torch.float32, device="cpu"),
        torch.ones((len(weight), 1), dtype=torch.float32, device="cpu"),
    )


class RecurrentRolloutTests(unittest.TestCase):
    def test_later_ce_reaches_first_state_and_cached_key_value(self):
        fusion = layer([[1, -1], [1, 1]])
        query = layer([[1, 1], [1, -1]])
        key = layer([[1, -1], [-1, 1]])
        value = layer([[1, 1], [-1, 1]])
        head = layer([[1, -1], [-1, 1]])
        seen = []
        first = {}

        def decode(token, feature, position, cache):
            state = feature + torch.tensor([token / 100, -token / 100], device="cpu")
            q, k, v = query(state), key(state), value(state)
            if not seen:
                k.retain_grad()
                v.retain_grad()
                feature.retain_grad()
                first.update(key=k, value=v, feature=feature)
            keys = torch.cat((cache["keys"], k.unsqueeze(0)))
            values = torch.cat((cache["values"], v.unsqueeze(0)))
            positions = (*cache["positions"], position)
            self.assertEqual(len(keys), len(positions))
            self.assertTrue(all(p <= position for p in positions))
            weights = torch.softmax(keys @ q / math.sqrt(2), dim=0)
            pre_norm = weights @ values + 0.1 * state
            seen.append(positions)
            return DraftStep(
                head(pre_norm), pre_norm, {"keys": keys, "values": values, "positions": positions}
            )

        rows = [row(0, 3, 4), row(1, 4, 5)]
        cache = {
            "keys": torch.empty((0, 2), device="cpu"),
            "values": torch.empty((0, 2), device="cpu"),
            "positions": (),
        }
        logits = rollout_captured_prefix(
            rows,
            torch.tensor([0.5, 0.25], device="cpu"),
            encode_feature=fusion,
            decode_step=decode,
            initial_cache=cache,
            draft_vocab_size=2,
        )
        audit = TraceAudit(
            draft_labels=(-1, 1),
            valid_mask=(True, True),
            supported_mask=(False, True),
            ce_mask=(False, True),
            denominator_mask=(True, True),
            target_to_draft=(0, 1),
            counts={"total": 2},
            per_depth={},
        )
        supported_prefix_ce(logits, audit).backward()
        self.assertEqual(seen, [(2,), (2, 3)])
        for name in ("key", "value", "feature"):
            self.assertIsNotNone(first[name].grad)
            self.assertGreater(float(first[name].grad.abs().sum()), 0, name)
        self.assertGreater(float(fusion.latent_sign.grad.abs().sum()), 0)
        self.assertGreater(float(head.latent_sign.grad.abs().sum()), 0)

    def test_terminal_invalid_row_has_no_decoder_call(self):
        calls = []

        def decode(token, feature, position, cache):
            calls.append(position)
            return DraftStep(torch.tensor([1.0, 0.0], device="cpu"), feature, cache)

        rows = [row(0, 3, 4), row(1, 4, None, valid=False)]
        logits = rollout_captured_prefix(
            rows,
            torch.ones(2, device="cpu"),
            encode_feature=lambda x: x,
            decode_step=decode,
            initial_cache=None,
            draft_vocab_size=2,
        )
        self.assertEqual(calls, [2])
        torch.testing.assert_close(logits[1], torch.zeros(2, device="cpu"))
        with self.assertRaisesRegex(ValueError, "terminal"):
            rollout_captured_prefix(
                [rows[0], rows[1], row(2, 4, 5)],
                torch.ones(2, device="cpu"),
                encode_feature=lambda x: x,
                decode_step=decode,
                initial_cache=None,
                draft_vocab_size=2,
            )

    def test_wrong_position_or_token_rejected(self):
        rows = [row(0, 3, 4), row(1, 5, 6)]
        with self.assertRaisesRegex(ValueError, "input token"):
            rollout_captured_prefix(
                rows,
                torch.ones(2, device="cpu"),
                encode_feature=lambda x: x,
                decode_step=lambda *args: DraftStep(
                    torch.ones(2, device="cpu"), torch.ones(2, device="cpu"), None
                ),
                initial_cache=None,
                draft_vocab_size=2,
            )


if __name__ == "__main__":
    unittest.main()
