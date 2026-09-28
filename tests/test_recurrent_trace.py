"""Synthetic CPU checks for recurrent native-capture alignment and masks."""

from __future__ import annotations

import unittest

import numpy as np

from w1a1_eagle.recurrent_trace import (
    LABEL_SOURCE,
    RoundAnchor,
    validate_offset_d2t,
    validate_recurrent_trace,
)

OFFSETS = (2, 3, 3)  # Native i+d2t[i] -> target IDs (2, 4, 5).
ANCHOR = RoundAnchor("train-a", "train", 0, (0, 1), 3)


def row(depth: int, prefix: list[int], *, label: int = 4, proposal: int = 6) -> dict:
    return {
        "prompt_id": "train-a",
        "split": "train",
        "round_index": 0,
        "depth": depth,
        "parent_position": 1,
        "input_position": 2 + depth,
        "label_position": 3 + depth,
        "verifier_row": depth,
        "prefix_token_ids": prefix,
        "input_token_id": prefix[-1],
        "alignment_valid": True,
        "is_bonus": False,
        "valid": True,
        "invalid_reason": None,
        "label_source": LABEL_SOURCE,
        "verifier_token_id": label,
        "proposed_token_id": proposal,
        "label_supported": label in (2, 4, 5),
    }


class RecurrentTraceTests(unittest.TestCase):
    def audit(self, rows: list[dict], anchors: list[RoundAnchor] | None = None):
        return validate_recurrent_trace(
            rows,
            [ANCHOR] if anchors is None else anchors,
            offsets=OFFSETS,
            target_vocab_size=8,
            draft_vocab_size=3,
            allowed_prompt_ids={"train-a", "train-b"},
            split="train",
            max_depth=5,
        )

    def test_native_offset_inverse_and_supported_ce_denominator(self):
        rows = [
            row(0, [0, 1, 3], label=4, proposal=6),
            row(1, [0, 1, 3, 6], label=7, proposal=5),
            row(2, [0, 1, 3, 6, 5], label=2, proposal=4),
        ]
        result = self.audit(rows)
        self.assertEqual(result.target_to_draft, (-1, -1, 0, -1, 1, 2, -1, -1))
        self.assertEqual(result.draft_labels, (1, -1, 0))
        self.assertEqual(result.ce_mask, (True, False, True))
        self.assertEqual(result.denominator_mask, (True, True, True))
        self.assertEqual(result.counts["unsupported"], 1)
        self.assertEqual(result.per_depth[1]["valid"], 1)

    def test_terminal_invalid_reason_keeps_row_but_excludes_denominator(self):
        first = row(0, [0, 1, 3], proposal=6)
        terminal = row(1, [0, 1, 3, 6])
        terminal.update(
            valid=False,
            invalid_reason="pruned",
            verifier_token_id=None,
            proposed_token_id=None,
            label_supported=False,
        )
        result = self.audit([first, terminal])
        self.assertEqual(result.draft_labels, (1, -1))
        self.assertEqual(result.valid_mask, (True, False))
        self.assertEqual(result.counts["invalid_pruned"], 1)
        self.assertEqual(result.counts["valid"], 1)
        for reason in ("padding", "eos", "unreached"):
            with self.subTest(reason=reason):
                changed = dict(terminal, invalid_reason=reason)
                self.assertEqual(self.audit([first, changed]).counts[f"invalid_{reason}"], 1)

    def test_rejects_offset_and_inverse_errors(self):
        self.assertEqual(validate_offset_d2t(np.asarray(OFFSETS, dtype=np.int32), 8, 3)[4], 1)
        for offsets in ((2, 2, 0), (2, 1, 0), (2, 3), (2, 3, 7), (2, 3, 3.0)):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                validate_offset_d2t(offsets, 8, 3)

    def test_rejects_off_by_one_and_mismatched_verifier_row(self):
        base = row(0, [0, 1, 3])
        for field, value in (
            ("input_position", 3),
            ("label_position", 4),
            ("parent_position", 0),
            ("verifier_row", 1),
            ("alignment_valid", False),
            ("is_bonus", True),
            ("forced", True),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit([dict(base, **{field: value})])

    def test_rejects_wrong_proposal_ancestry_and_depth_gap(self):
        first = row(0, [0, 1, 3], label=4, proposal=6)
        # Following the verifier label instead of the proposal is leakage.
        wrong = row(1, [0, 1, 3, 4])
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.audit([first, wrong])
        gap = row(2, [0, 1, 3, 6, 5])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.audit([first, gap])

    def test_round_anchors_prevent_cross_round_and_cross_prompt_leakage(self):
        second = RoundAnchor("train-a", "train", 1, (0, 1, 3, 6), 5)
        same_prompt = row(0, [0, 1, 3])
        next_round = row(0, [0, 1, 3, 6, 5])
        next_round.update(round_index=1, parent_position=3, input_position=4, label_position=5)
        self.assertEqual(self.audit([same_prompt, next_round], [ANCHOR, second]).counts["total"], 2)
        with self.assertRaisesRegex(ValueError, "round starts"):
            self.audit(
                [same_prompt, dict(next_round, prefix_token_ids=[0, 1, 3, 6, 4], input_token_id=4)],
                [ANCHOR, second],
            )
        with self.assertRaisesRegex(ValueError, "anchor"):
            self.audit([dict(same_prompt, prompt_id="train-b")])
        with self.assertRaisesRegex(ValueError, "split"):
            self.audit([dict(same_prompt, split="development")])

    def test_invalid_masks_cannot_hide_labels_or_have_descendants(self):
        first = row(0, [0, 1, 3])
        bad = dict(
            first,
            valid=False,
            invalid_reason="padding",
            label_supported=False,
            verifier_token_id=None,
            proposed_token_id=None,
        )
        self.assertEqual(self.audit([bad]).ce_mask, (False,))
        for field, value in (
            ("verifier_token_id", 4),
            ("proposed_token_id", 6),
            ("label_supported", True),
            ("invalid_reason", None),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit([dict(bad, **{field: value})])
        with self.assertRaisesRegex(ValueError, "ancestor"):
            self.audit([bad, row(1, [0, 1, 3, 6])])

    def test_optional_logits_and_exact_label_source_are_checked(self):
        base = row(0, [0, 1, 3])
        good = dict(base, verifier_logits=[0.0] * 8)
        result = self.audit([good])
        self.assertEqual(result.counts["valid"], 1)
        self.assertEqual(result.counts["logit_rows"], 1)
        self.assertAlmostEqual(result.mapped_probability_mass[0], 3 / 8)
        for change in (
            {"verifier_logits": [0.0] * 7},
            {"verifier_logits": [float("nan")] * 8},
            {"verifier_logits": [-float("inf")] * 8},
            {"label_source": "cached_head_label"},
            {"label_supported": False},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit([dict(base, **change)])


if __name__ == "__main__":
    unittest.main()
