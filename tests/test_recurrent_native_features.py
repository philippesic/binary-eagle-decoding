"""CPU-only accepted-prefix selection from native feature/disposition events."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_recurrent_binary_capture import audit_feature_ledger  # noqa: E402
from prepare_recurrent_native_features import (  # noqa: E402
    RAW_BOUNDARY,
    RAW_SCHEMA,
    RAW_SOURCE,
    prepare_native_features,
    sha256,
)

from w1a1_eagle.recurrent_trace import RoundAnchor  # noqa: E402


class RecurrentNativeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.prompts = self.directory / "train.jsonl"
        self.prompts.write_text(
            "".join(json.dumps({"id": f"p-{index}"}) + "\n" for index in range(96))
        )
        self.mapping = self.directory / "task-map.json"
        self.mapping.write_text(json.dumps({"7": "p-0"}))
        self.cell = self.directory / "cell.json"
        self.anchors = self.directory / "anchors.jsonl"
        self.anchor_rows = [
            {
                "prompt_id": "p-0",
                "split": "train",
                "round_index": 0,
                "prefix_token_ids": [0, 1],
                "seed_token_id": 3,
            },
            {
                "prompt_id": "p-0",
                "split": "train",
                "round_index": 1,
                "prefix_token_ids": [0, 1, 3],
                "seed_token_id": 5,
            },
            {
                "prompt_id": "p-0",
                "split": "train",
                "round_index": 2,
                "prefix_token_ids": [0, 1, 3, 5, 6],
                "seed_token_id": 7,
            },
        ]
        self.metadata = self.directory / "native-features.jsonl"
        self.values = self.directory / "native-features.f32"
        self.output = self.directory / "selected"
        # Two prefill rows, then A=0 and A=1 speculative rounds. The
        # proposed suffix rows 3 and 6 must never enter accepted history.
        specs = [
            (0, 0, [0], "prefill", None, None, True),
            (0, 1, [0, 1], "prefill", None, None, True),
            (1, 0, [0, 1, 3], "speculative", 0, 0, True),
            (1, 1, [0, 1, 3, 4], "speculative", 0, 1, False),
            (2, 0, [0, 1, 3, 5], "speculative", 1, 0, True),
            (2, 1, [0, 1, 3, 5, 6], "speculative", 1, 1, True),
            (2, 2, [0, 1, 3, 5, 6, 8], "speculative", 1, 2, False),
        ]
        self.events = []
        for index, (ordinal, local, prefix, phase, round_index, spec_row, retained) in enumerate(
            specs
        ):
            row = {
                "schema": RAW_SCHEMA,
                "event": "decoded_row",
                "feature_row": index,
                "task_id": 7,
                "parent_task_id": -1,
                "slot_id": 0,
                "decode_ordinal": ordinal,
                "batch_row_local": local,
                "batch_row_global": local,
                "position": len(prefix) - 1,
                "token_id": prefix[-1],
                "prefix_token_ids": prefix,
                "target_layer_ids": [2, 18, 33],
                "feature_dim": 7680,
                "boundary": RAW_BOUNDARY,
                "source": RAW_SOURCE,
                "phase": phase,
            }
            if phase == "speculative":
                row.update(round_index=round_index, spec_input_row=spec_row)
            self.events.append(row)
            status = {
                "schema": RAW_SCHEMA,
                "event": "disposition",
                "feature_row": index,
                "task_id": 7,
                "retained_input": retained,
                "reason": "accepted_prefix" if retained else "rejected_suffix",
            }
            if phase == "speculative":
                status.update(round_index=round_index, accepted_drafts=round_index)
            self.events.append(status)
        self.array = np.repeat(
            (np.arange(len(specs), dtype=np.float32) + 0.25)[:, None], 7680, axis=1
        )
        self.save()

    def save(self):
        self.metadata.write_text("".join(json.dumps(row) + "\n" for row in self.events))
        self.values.write_bytes(self.array.astype("<f4").tobytes())
        self.anchors.write_text("".join(json.dumps(row) + "\n" for row in self.anchor_rows))
        self.cell.write_text(
            json.dumps(
                {
                    "schema": "binary_head_capture_cell_v1",
                    "complete": True,
                    "prompts_sha256": sha256(self.prompts),
                    "task_prompt_ids": {"7": "p-0"},
                    "requests": [{"task_id": "7", "id": "p-0"}],
                    "files": {
                        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                        for path in (self.metadata, self.values)
                    },
                }
            )
        )

    def prepare(self, *, with_cell=False):
        return prepare_native_features(
            self.metadata,
            self.values,
            self.anchors,
            self.mapping,
            self.prompts,
            self.output,
            cell_manifest_path=self.cell if with_cell else None,
            expected_prompt_hash=sha256(self.prompts),
            target_vocab_size=16,
        )

    def test_selects_only_retained_prefix_features_across_rejections(self):
        report = self.prepare()
        self.assertEqual(report["decoded_rows"], 7)
        self.assertEqual(report["rejected_suffix_rows"], 2)
        self.assertEqual(report["selected_feature_rows"], 5)
        self.assertEqual(
            report["task_prompt_ownership"], "plain_task_map_unverified_without_request_manifest"
        )
        selected = np.load(self.output / "features.npy", allow_pickle=False)
        np.testing.assert_array_equal(selected[:, 0], [0.25, 1.25, 2.25, 4.25, 5.25])
        anchors = [RoundAnchor(**row) for row in self.anchor_rows]
        audit = audit_feature_ledger(
            self.output / "features.npy",
            self.output / "feature_rows.jsonl",
            anchors,
            {"p-0"},
            16,
        )
        self.assertEqual(audit["rows"], 5)

    def test_cell_manifest_proves_request_join_and_raw_file_hashes(self):
        report = self.prepare(with_cell=True)
        self.assertEqual(
            report["task_prompt_ownership"], "cell_manifest_request_and_file_hashes_verified"
        )
        cell = json.loads(self.cell.read_text())
        cell["requests"][0]["id"] = "p-1"
        self.cell.write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "request/task ownership"):
            prepare_native_features(
                self.metadata,
                self.values,
                self.anchors,
                self.mapping,
                self.prompts,
                self.directory / "other",
                cell_manifest_path=self.cell,
                expected_prompt_hash=sha256(self.prompts),
                target_vocab_size=16,
            )
        self.save()
        self.values.write_bytes(self.values.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "raw feature hash/size"):
            prepare_native_features(
                self.metadata,
                self.values,
                self.anchors,
                self.mapping,
                self.prompts,
                self.directory / "another",
                cell_manifest_path=self.cell,
                expected_prompt_hash=sha256(self.prompts),
                target_vocab_size=16,
            )

    def test_rejects_wrong_retention_or_missing_disposition(self):
        suffix = next(
            row for row in self.events if row["event"] == "disposition" and row["feature_row"] == 3
        )
        suffix["retained_input"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "accepted depth"):
            self.prepare()
        self.assertFalse(self.output.exists())
        suffix["retained_input"] = False
        self.events.remove(suffix)
        self.save()
        with self.assertRaisesRegex(ValueError, "dispositions are missing"):
            self.prepare()

    def test_rejects_wrong_ancestry_nonfinite_and_nontraining_task(self):
        row = next(
            row for row in self.events if row["event"] == "decoded_row" and row["feature_row"] == 4
        )
        row["prefix_token_ids"] = [0, 1, 4, 5]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing for round anchor"):
            self.prepare()
        row["prefix_token_ids"] = [0, 1, 3, 5]
        self.array[4, 0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.prepare()
        self.array[4, 0] = 4.25
        self.mapping.write_text(json.dumps({"7": "not-a-train-prompt"}))
        self.save()
        with self.assertRaisesRegex(ValueError, "nontraining task"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
