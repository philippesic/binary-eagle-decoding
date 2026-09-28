"""Synthetic frozen-source and streamed prefill checks; no model or GPU run."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_recurrent_full_target_features as full  # noqa: E402
from audit_recurrent_binary_capture import sha256  # noqa: E402


class FullTargetFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cell = self.root / "d_d"
        self.cell.mkdir()
        self.prompts = self.root / "train.jsonl"
        self.prompts.write_text("".join(json.dumps({"id": f"train-{i}"}) + "\n" for i in range(2)))
        self.mapping = self.root / "task_prompt_ids.json"
        self.mapping.write_text(json.dumps({"7": "train-0", "9": "train-1"}))
        self.events_path = self.cell / "heads.target_features.jsonl"
        self.values_path = self.cell / "heads.target_features.f32"
        self.rounds_path = self.cell / "forced-rounds.jsonl"
        self.requests = []
        self.events = []
        for index, task in enumerate((7, 9)):
            start = len(self.events)
            for position, token in enumerate((10 + index, 20 + index)):
                row = index * 2 + position
                self.events.append(
                    {
                        "schema": full.RAW_SCHEMA,
                        "event": "decoded_row",
                        "feature_row": row,
                        "task_id": task,
                        "phase": "prefill",
                        "position": position,
                        "batch_row_local": position,
                        "batch_row_global": position,
                        "slot_id": 0,
                        "decode_ordinal": index,
                        "token_id": token,
                        "prefix_token_ids": [10 + index, 20 + index][: position + 1],
                        "target_layer_ids": list(full.TAPS),
                        "feature_dim": full.WIDTH,
                        "boundary": full.RAW_BOUNDARY,
                        "source": full.RAW_SOURCE,
                    }
                )
                self.events.append(
                    {
                        "schema": full.RAW_SCHEMA,
                        "event": "disposition",
                        "feature_row": row,
                        "task_id": task,
                        "retained_input": True,
                    }
                )
            self.requests.append(
                {
                    "id": f"train-{index}",
                    "task_id": str(task),
                    "target_feature_event_rows": [start, len(self.events)],
                    "target_feature_rows": [index * 2, index * 2 + 2],
                }
            )
        self.save_events()
        np.ones((4, full.WIDTH), dtype="<f4").tofile(self.values_path)
        self.rounds_path.write_text(
            "".join(
                json.dumps({"task_id": task, "prefix_token_ids": [10 + i, 20 + i]}) + "\n"
                for i, task in enumerate((7, 9))
            )
        )
        self.write_manifests()

    def save_events(self):
        self.events_path.write_text("".join(json.dumps(event) + "\n" for event in self.events))

    def write_manifests(self):
        cell = {
            "schema": "binary_head_capture_cell_v1",
            "complete": True,
            "prompts_sha256": sha256(self.prompts),
            "task_prompt_ids": {"7": "train-0", "9": "train-1"},
            "requests": self.requests,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (self.events_path, self.values_path, self.rounds_path)
            },
        }
        cell_path = self.cell / "manifest.json"
        cell_path.write_text(json.dumps(cell))
        capture = {
            "schema": "recurrent_binary_native_capture_v1",
            "split": "train",
            "trajectory": "own_history",
            "body": "D",
            "head": "D",
            "complete": True,
            "train_prompts_sha256": sha256(self.prompts),
            "task_prompt_ids_path": self.mapping.name,
            "task_prompt_ids_sha256": sha256(self.mapping),
            "cell_manifest_path": "d_d/manifest.json",
            "cell_manifest_sha256": sha256(cell_path),
            "raw_files": {
                path.name: {"path": f"d_d/{path.name}", **cell["files"][path.name]}
                for path in (self.events_path, self.values_path, self.rounds_path)
            },
            "requests": self.requests,
        }
        (self.root / "capture-manifest.json").write_text(json.dumps(capture))

    def validate(self):
        with (
            mock.patch.object(full, "TRAIN_PROMPTS", 2),
            mock.patch.object(full, "TRAIN_PROMPTS_SHA256", sha256(self.prompts)),
        ):
            return full.validate_sources(self.root, self.prompts)

    def test_selects_all_task_prefill_rows_and_stats(self):
        _, tasks, requests = self.validate()
        self.assertEqual(tasks, {7: "train-0", 9: "train-1"})
        self.assertEqual(
            full.select_prefill_rows(self.events_path, requests),
            {7: ([10, 20], [0, 1]), 9: ([11, 21], [2, 3])},
        )
        full.validate_first_round_prefixes(
            self.rounds_path, requests, full.select_prefill_rows(self.events_path, requests)
        )
        stats = full._stats(np.array([0.0, 0.2]), np.array([[0.0], [1.0]]))
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["relative_l2_max"], 0.2)

    def test_rejects_wrong_ancestry_and_tap_order(self):
        self.events[2]["prefix_token_ids"] = [10, 22]
        self.save_events()
        with self.assertRaisesRegex(ValueError, "ancestry"):
            full.select_prefill_rows(self.events_path, self.requests)
        self.events[2]["prefix_token_ids"] = [10, 20]
        self.events[0]["target_layer_ids"] = [18, 2, 33]
        self.save_events()
        with self.assertRaisesRegex(ValueError, "tap order"):
            full.select_prefill_rows(self.events_path, self.requests)

    def test_rejects_missing_prefill_and_cross_task_row(self):
        self.events[4]["phase"] = "target_only"
        self.save_events()
        with self.assertRaisesRegex(ValueError, "prefill resumes"):
            full.select_prefill_rows(self.events_path, self.requests)
        self.events[4]["phase"] = "prefill"
        self.events[4]["task_id"] = 7
        self.save_events()
        with self.assertRaisesRegex(ValueError, "another request"):
            full.select_prefill_rows(self.events_path, self.requests)

    def test_rejects_capture_hash_and_nontraining_map(self):
        capture = json.loads((self.root / "capture-manifest.json").read_text())
        capture["cell_manifest_sha256"] = "0" * 64
        (self.root / "capture-manifest.json").write_text(json.dumps(capture))
        with self.assertRaisesRegex(ValueError, "cell manifest SHA256"):
            self.validate()
        self.write_manifests()
        self.mapping.write_text(json.dumps({"7": "train-0", "9": "development-1"}))
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "nontraining"):
            self.validate()

    def test_rejects_truncated_prefill_relative_to_first_round(self):
        self.rounds_path.write_text(
            self.rounds_path.read_text().replace("[10, 20]", "[10, 20, 30]")
        )
        self.write_manifests()
        _, _, requests = self.validate()
        selected = full.select_prefill_rows(self.events_path, requests)
        with self.assertRaisesRegex(ValueError, "complete prefill"):
            full.validate_first_round_prefixes(self.rounds_path, requests, selected)

    def test_accepts_microbatched_prefill_local_row_reset(self):
        self.events[2]["decode_ordinal"] = 1
        self.events[2]["batch_row_local"] = 0
        self.events[2]["batch_row_global"] = 0
        self.events[4]["decode_ordinal"] = 2
        self.events[6]["decode_ordinal"] = 3
        self.events[6]["batch_row_local"] = 0
        self.events[6]["batch_row_global"] = 0
        self.save_events()
        selected = full.select_prefill_rows(self.events_path, self.requests)
        self.assertEqual(selected[7][0], [10, 20])
        self.assertEqual(selected[9][1], [2, 3])


if __name__ == "__main__":
    unittest.main()
