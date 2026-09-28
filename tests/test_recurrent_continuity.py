"""CPU-only cross-round accepted-prefix and feature retention checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_recurrent_continuity import audit_internal_continuity  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rounds_path = self.root / "rounds.jsonl"
        self.features_path = self.root / "features.jsonl"
        self.mapping = self.root / "tasks.json"
        self.mapping.write_text(json.dumps({"9": "p-0"}))
        self.rounds = [
            {
                "schema": "eagle_forced_round_v1",
                "task_id": 9,
                "round_index": 0,
                "prefix_token_ids": [0, 1],
                "seed_token_id": 3,
                "pos0": 2,
                "draft_token_ids": [4],
                "accepted_drafts": 0,
                "verifier_token_ids": [5],
            },
            {
                "schema": "eagle_forced_round_v1",
                "task_id": 9,
                "round_index": 1,
                "prefix_token_ids": [0, 1, 3],
                "seed_token_id": 5,
                "pos0": 3,
                "draft_token_ids": [6, 8],
                "accepted_drafts": 1,
                "verifier_token_ids": [6, 7],
            },
        ]
        self.events = []
        self.add_feature(0, [0], "prefill", 0, True)
        self.add_feature(1, [0, 1], "prefill", 1, True)
        self.add_feature(2, [0, 1, 3], "speculative", 2, True, round_index=0, j=0, accepted=0)
        self.add_feature(3, [0, 1, 3, 4], "speculative", 2, False, round_index=0, j=1, accepted=0)
        self.add_feature(4, [0, 1, 3, 5], "speculative", 3, True, round_index=1, j=0, accepted=1)
        self.add_feature(5, [0, 1, 3, 5, 6], "speculative", 3, True, round_index=1, j=1, accepted=1)
        self.add_feature(
            6, [0, 1, 3, 5, 6, 8], "speculative", 3, False, round_index=1, j=2, accepted=1
        )
        self.save()

    def add_feature(
        self, index, prefix, phase, ordinal, kept, *, round_index=None, j=None, accepted=None
    ):
        row = {
            "schema": "eagle_target_feature_v1",
            "event": "decoded_row",
            "feature_row": index,
            "task_id": 9,
            "slot_id": 0,
            "decode_ordinal": ordinal,
            "phase": phase,
            "position": len(prefix) - 1,
            "token_id": prefix[-1],
            "prefix_token_ids": prefix,
        }
        if phase == "speculative":
            row.update(round_index=round_index, spec_input_row=j)
        status = {
            "schema": "eagle_target_feature_v1",
            "event": "disposition",
            "feature_row": index,
            "task_id": 9,
            "slot_id": 0,
            "retained_input": kept,
            "reason": "accepted_prefix" if kept else "rejected_suffix",
            "round_index": round_index,
            "accepted_drafts": accepted,
        }
        self.events.extend((row, status))

    def save(self):
        write_jsonl(self.rounds_path, self.rounds)
        write_jsonl(self.features_path, self.events)

    def audit(self):
        return audit_internal_continuity(self.rounds_path, self.features_path, self.mapping)

    def test_valid_zero_and_later_acceptance_with_terminal_input(self):
        self.add_feature(7, [0, 1, 3, 5, 6, 7], "target_only", 4, True)
        self.save()
        report = self.audit()
        self.assertEqual(report["rounds"], 2)
        self.assertEqual(report["prefill_inputs"], 2)
        self.assertEqual(report["speculative_inputs"], 5)
        self.assertEqual(report["rejected_suffix_inputs"], 2)
        self.assertEqual(report["terminal_emission_and_stop"], "unverified")

    def test_wrong_next_prefix_seed_and_gap_fail(self):
        self.rounds[1]["prefix_token_ids"] = [0, 1, 4]
        self.save()
        with self.assertRaisesRegex(ValueError, "continuity"):
            self.audit()
        self.rounds[1]["prefix_token_ids"] = [0, 1, 3]
        self.rounds[1]["seed_token_id"] = 6
        self.save()
        with self.assertRaisesRegex(ValueError, "continuity"):
            self.audit()
        self.rounds[1]["seed_token_id"] = 5
        self.rounds[1]["round_index"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "gap"):
            self.audit()

    def test_wrong_feature_retention_and_missing_input_fail(self):
        suffix = next(
            row for row in self.events if row["event"] == "disposition" and row["feature_row"] == 3
        )
        suffix["retained_input"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "canonical round"):
            self.audit()
        suffix["retained_input"] = False
        self.events = [row for row in self.events if row["feature_row"] != 6]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            self.audit()

    def test_prefill_gap_slot_change_and_terminal_reorder_fail(self):
        first = next(
            row for row in self.events if row["event"] == "decoded_row" and row["feature_row"] == 1
        )
        first["prefix_token_ids"] = [0, 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "prefill"):
            self.audit()
        first["prefix_token_ids"] = [0, 1]
        first["slot_id"] = 1
        disposition = next(
            row for row in self.events if row["event"] == "disposition" and row["feature_row"] == 1
        )
        disposition["slot_id"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "one slot"):
            self.audit()
        first["slot_id"] = disposition["slot_id"] = 0
        self.add_feature(7, [0, 1, 3, 5, 6, 7], "target_only", 1, True)
        self.save()
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.audit()

    def test_eos_is_not_inferred_as_full_emission(self):
        self.rounds = [
            {
                "schema": "eagle_forced_round_v1",
                "task_id": 9,
                "round_index": 0,
                "prefix_token_ids": [0, 1],
                "seed_token_id": 3,
                "pos0": 2,
                "draft_token_ids": [7, 8],
                "accepted_drafts": 2,
                "verifier_token_ids": [7, 8, 9],
            }
        ]
        self.events = [row for row in self.events if row["feature_row"] < 2]
        self.add_feature(2, [0, 1, 3], "speculative", 2, True, round_index=0, j=0, accepted=2)
        self.add_feature(3, [0, 1, 3, 7], "speculative", 2, True, round_index=0, j=1, accepted=2)
        self.add_feature(4, [0, 1, 3, 7, 8], "speculative", 2, True, round_index=0, j=2, accepted=2)
        self.save()
        report = self.audit()
        self.assertEqual(report["status"], "internal_accepted_prefix_continuity_verified")
        self.assertEqual(report["terminal_emission_and_stop"], "unverified")

    def test_terminal_input_token_must_match_accepted_prefix(self):
        self.add_feature(7, [0, 1, 3, 5, 6, 7], "target_only", 4, True)
        terminal = next(
            row for row in self.events if row["event"] == "decoded_row" and row["feature_row"] == 7
        )
        terminal["token_id"] = 8
        self.save()
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
