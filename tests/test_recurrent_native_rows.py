"""Synthetic native-row preparation gates. No model or accelerator execution."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_recurrent_native_rows import prepare, read_jsonl, sha256, write_prepared  # noqa: E402


def jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class NativeRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prompts = self.root / "train.jsonl"
        jsonl(self.prompts, [{"id": "train-a"}, {"id": "train-b"}])
        self.task_map = self.root / "task-map.json"
        self.task_map.write_text(json.dumps({"9": "train-a"}))
        self.heads = self.root / "heads.jsonl"
        self.rounds = self.root / "forced-rounds.jsonl"
        self.absolute_map = self.root / "absolute.npy"
        np.save(self.absolute_map, np.array([2, 4, 5], dtype=np.int64))
        self.logits = self.root / "target.f32"
        np.zeros(8, dtype="<f4").tofile(self.logits)
        self.offsets = self.root / "offsets.npy"
        self.t2d = self.root / "t2d.npy"
        np.save(self.offsets, np.array([2, 3, 3], dtype=np.int64))
        np.save(self.t2d, np.array([False, False, True, False, True, True, False, False]))
        self.round_records = [
            {
                "schema": "eagle_forced_round_v1",
                "task_id": 9,
                "round_index": 0,
                "prefix_token_ids": [0, 1],
                "seed_token_id": 3,
                "pos0": 2,
                "draft_token_ids": [6, 5],
                "accepted_drafts": 0,
                "verifier_token_ids": [4],
            }
        ]
        base = {
            "schema": "eagle_head_state_v1",
            "task_id": 9,
            "round_index": 0,
            "parent_position": 1,
            "alignment_valid": True,
            "finite": True,
            "valid": True,
            "is_bonus": False,
            "forced": False,
            "state_dim": 2,
            "state_boundary": "native_output_norm_f32_before_head_operand_conversion",
            "state_dtype": "float32_native_endian",
            "label_source": "cloned_native_verifier_sampler_at_actual_proposal_prefix",
            "target_logits_source": "raw_target_verifier_at_exact_proposal_prefix",
            "target_logits_dim": 8,
            "full_logits_source": "draft_head_mapped_target_vocabulary_before_sampler",
        }
        self.head_records = [
            dict(
                base,
                state_row=0,
                depth=0,
                verifier_row=0,
                input_position=2,
                label_position=3,
                prefix_token_ids=[0, 1, 3],
                input_token_id=3,
                proposed_token_id=6,
                verifier_token_id=4,
                label_supported=True,
                verifier_reached=True,
                verifier_sampled_token_id=4,
                target_logits_row=0,
                full_logits_row=0,
            ),
            dict(
                base,
                state_row=1,
                depth=1,
                verifier_row=1,
                input_position=3,
                label_position=4,
                prefix_token_ids=[0, 1, 3, 6],
                input_token_id=6,
                proposed_token_id=5,
                verifier_token_id=7,
                label_supported=False,
                verifier_reached=False,
                verifier_sampled_token_id=None,
                target_logits_row=None,
                full_logits_row=1,
            ),
        ]
        self.cell = self.root / "cell.json"
        self.save()

    def save(self) -> None:
        jsonl(self.heads, self.head_records)
        jsonl(self.rounds, self.round_records)
        self.cell.write_text(
            json.dumps(
                {
                    "schema": "binary_head_capture_cell_v1",
                    "complete": True,
                    "prompts_sha256": sha256(self.prompts),
                    "task_prompt_ids": {"9": "train-a"},
                    "requests": [
                        {
                            "id": "train-a",
                            "task_id": "9",
                            "capture_rows": [0, len(self.head_records)],
                            "forced_round_rows": [0, len(self.round_records)],
                        }
                    ],
                    "files": {
                        self.heads.name: {
                            "sha256": sha256(self.heads),
                            "bytes": self.heads.stat().st_size,
                        },
                        self.rounds.name: {
                            "sha256": sha256(self.rounds),
                            "bytes": self.rounds.stat().st_size,
                        },
                    },
                }
            )
        )

    def prepare(self, *, with_cell: bool = True, with_logits: bool = True):
        return prepare(
            heads_path=self.heads,
            rounds_path=self.rounds,
            task_map_path=self.task_map,
            cell_manifest_path=self.cell if with_cell else None,
            prompts_path=self.prompts,
            absolute_map_path=self.absolute_map,
            target_vocab_size=8,
            target_logits_path=self.logits if with_logits else None,
            offset_path=self.offsets,
            t2d_path=self.t2d,
            expected_prompt_hash=sha256(self.prompts),
            expected_prompt_count=2,
        )

    def test_rejection_after_proposal_keeps_later_teacher_label_and_distinct_logits(self):
        result = self.prepare()
        rows, anchors, offsets, t2d, report = result
        self.assertEqual([row["verifier_reached"] for row in rows], [True, False])
        self.assertEqual([row["valid"] for row in rows], [True, True])
        self.assertEqual([row["label_supported"] for row in rows], [True, False])
        self.assertEqual([row["verifier_token_id"] for row in rows], [4, 7])
        self.assertEqual(rows[1]["prefix_token_ids"], [0, 1, 3, 6])
        self.assertEqual(rows[0]["target_logits_row"], 0)
        self.assertEqual(rows[0]["full_logits_row"], 0)
        self.assertNotEqual(rows[0]["target_logits_source"], rows[0]["full_logits_source"])
        self.assertEqual(anchors[0]["prefix_token_ids"], [0, 1])
        self.assertEqual(offsets.tolist(), [2, 3, 3])
        self.assertEqual(t2d.nonzero()[0].tolist(), [2, 4, 5])
        self.assertEqual(report["trace_counts"]["unsupported"], 1)
        self.assertEqual(
            report["request_prompt_ownership"], "cell_manifest_request_ranges_verified"
        )
        self.assertEqual(report["raw_target_feature_ledger"], "unverified_missing")
        output = self.root / "prepared"
        written = write_prepared(output, result)
        self.assertEqual(read_jsonl(output / "rows.jsonl"), rows)
        self.assertEqual(read_jsonl(output / "anchors.jsonl"), anchors)
        self.assertEqual(written["output_sha256"]["rows.jsonl"], sha256(output / "rows.jsonl"))

    def test_plain_task_map_is_allowed_but_request_ownership_unverified(self):
        report = self.prepare(with_cell=False)[-1]
        self.assertEqual(report["request_prompt_ownership"], "unverified_plain_task_map_only")

    def test_wrong_ancestry_and_position_fail(self):
        original = copy.deepcopy(self.head_records)
        for field, value in (
            ("prefix_token_ids", [0, 1, 3, 4]),
            ("input_position", 4),
            ("proposed_token_id", 4),
        ):
            with self.subTest(field=field):
                self.head_records[1][field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "ancestry|positions|proposal"):
                    self.prepare()
                self.head_records = copy.deepcopy(original)

    def test_wrong_task_mapping_and_request_ranges_fail(self):
        self.task_map.write_text(json.dumps({"9": "train-b"}))
        with self.assertRaisesRegex(ValueError, "task map"):
            self.prepare()
        self.task_map.write_text(json.dumps({"9": "development-a"}))
        with self.assertRaisesRegex(ValueError, "unknown training prompt"):
            self.prepare(with_cell=False)
        self.task_map.write_text(json.dumps({"9": "train-a"}))
        cell = json.loads(self.cell.read_text())
        cell["requests"][0]["capture_rows"] = [0, 1]
        self.cell.write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "unclaimed"):
            self.prepare()

    def test_missing_duplicate_or_forced_rows_fail(self):
        original = copy.deepcopy(self.head_records)
        self.head_records.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "missing, duplicate"):
            self.prepare()
        self.head_records = copy.deepcopy(original)
        self.head_records[1] = copy.deepcopy(self.head_records[0])
        self.head_records[1]["state_row"] = 1
        self.save()
        with self.assertRaises(ValueError):
            self.prepare()
        self.head_records = copy.deepcopy(original)
        self.head_records[1]["forced"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "forced native"):
            self.prepare()

    def test_invalid_mask_and_unsupported_label_disagreement_fail(self):
        for field, value, expected in (
            ("valid", False, "invalid native masked"),
            ("finite", False, "invalid native masked"),
            ("label_supported", True, "supported label"),
        ):
            with self.subTest(field=field):
                original = self.head_records[1][field]
                self.head_records[1][field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, expected):
                    self.prepare()
                self.head_records[1][field] = original

    def test_raw_target_logits_and_map_consistency_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "raw target file"):
            self.prepare(with_logits=False)
        self.head_records[0]["full_logits_source"] = "raw_target_verifier_at_exact_proposal_prefix"
        self.save()
        with self.assertRaisesRegex(ValueError, "declared draft-head logits"):
            self.prepare()
        self.head_records[0]["full_logits_source"] = (
            "draft_head_mapped_target_vocabulary_before_sampler"
        )
        self.head_records[0]["target_logits_source"] = (
            "draft_head_mapped_target_vocabulary_before_sampler"
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "target logits dimension/source"):
            self.prepare()
        self.head_records[0]["target_logits_source"] = (
            "raw_target_verifier_at_exact_proposal_prefix"
        )
        self.save()
        np.save(self.t2d, np.zeros(8, dtype=np.bool_))
        with self.assertRaisesRegex(ValueError, "t2d mask"):
            self.prepare()

    def test_missing_round_and_wrong_live_verifier_label_fail(self):
        self.round_records.append(copy.deepcopy(self.round_records[0]))
        self.round_records[1]["round_index"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "missing all proposal"):
            self.prepare()
        self.round_records.pop()
        self.head_records[0]["verifier_token_id"] = 5
        self.save()
        with self.assertRaisesRegex(ValueError, "cloned native label"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
