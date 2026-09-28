"""Synthetic ladder integrity and metric checks; no model or GPU needed."""

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

import check_target_layer_ladder as ladder  # noqa: E402


class LadderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sealed = self.root / "sealed"
        self.new = self.root / "new"
        self.sealed_cell = self.sealed / "d_d" / "manifest.json"
        self.sealed_cell.parent.mkdir(parents=True)
        self.new.mkdir()
        self.target = self.root / "target.gguf"
        self.draft = self.root / "draft.gguf"
        self.train = self.root / "train.jsonl"
        self.binary = self.root / "llama-server"
        self.cmake_cache = self.root / "CMakeCache.txt"
        for path in (self.target, self.draft, self.train, self.binary, self.cmake_cache):
            path.write_bytes(path.name.encode())
        self.sealed_manifest = self.sealed / "capture-manifest.json"
        self.sealed_manifest.write_text("{}")
        self.prefix = [10, 20, 30, 40]
        self.task = 7
        self.feature = np.zeros((4, ladder.WIDTH), dtype="=f4")
        self.values = np.zeros((4, ladder.LADDER_WIDTH), dtype="=f4")
        for index, tap in enumerate(ladder.TAPS):
            value = np.float32(index + 1)
            self.feature[:, index * ladder.HIDDEN : (index + 1) * ladder.HIDDEN] = value
            layer_index = ladder.LADDER_LAYERS.index(tap)
            self.values[:, layer_index * ladder.HIDDEN : (layer_index + 1) * ladder.HIDDEN] = value
        self.feature.tofile(self.new / "heads.target_features.f32")
        self.feature.tofile(self.sealed_cell.parent / "heads.target_features.f32")
        self.values.tofile(self.new / "heads.target_layer_ladder.f32")
        self.feature_events = []
        self.ladder_rows = []
        for position, token in enumerate(self.prefix):
            event = {
                "schema": ladder.RAW_SCHEMA,
                "event": "decoded_row",
                "feature_row": position,
                "task_id": self.task,
                "phase": "prefill",
                "position": position,
                "token_id": token,
                "prefix_token_ids": self.prefix[: position + 1],
                "target_layer_ids": list(ladder.TAPS),
                "feature_dim": ladder.WIDTH,
                "boundary": ladder.RAW_BOUNDARY,
                "source": ladder.RAW_SOURCE,
                "slot_id": 0,
                "decode_ordinal": 1,
                "batch_row_local": position,
                "batch_row_global": position,
            }
            self.feature_events.append(event)
            self.ladder_rows.append(
                {
                    "schema": ladder.LADDER_SCHEMA,
                    "row": position,
                    "task_id": self.task,
                    "slot_id": 0,
                    "decode_ordinal": 1,
                    "batch_row_local": position,
                    "batch_row_global": position,
                    "position": position,
                    "token_id": token,
                    "layer_ids": list(ladder.LADDER_LAYERS),
                    "hidden": ladder.HIDDEN,
                    "byte_offset": position * ladder.LADDER_ROW_BYTES,
                    "byte_count": ladder.LADDER_ROW_BYTES,
                    "dtype": "float32_native_endian",
                    "boundary": ladder.LADDER_BOUNDARY,
                    "source": ladder.RAW_SOURCE,
                    "target_source": str(self.target.resolve()),
                }
            )
        self.write_rows()
        (self.new / "forced-rounds.jsonl").write_text(
            json.dumps({"task_id": self.task, "prefix_token_ids": self.prefix}) + "\n"
        )
        self.write_manifest()

    def write_rows(self):
        (self.new / "heads.target_features.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.feature_events)
        )
        (self.new / "heads.target_layer_ladder.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.ladder_rows)
        )

    def write_manifest(self):
        names = (
            "heads.target_features.jsonl",
            "heads.target_features.f32",
            "heads.target_layer_ladder.jsonl",
            "heads.target_layer_ladder.f32",
            "forced-rounds.jsonl",
        )
        manifest = {
            "schema": "recurrent_cuda_native_diagnostic_v1",
            "execution_device": "cuda",
            "prompt_id": ladder.PROMPT_ID,
            "training_eligible": False,
            "source_sha256": {
                "target": ladder.sha256(self.target),
                "draft": ladder.sha256(self.draft),
                "frozen_train_prompts": ladder.sha256(self.train),
                "binary": ladder.sha256(self.binary),
                "cmake_cache": ladder.sha256(self.cmake_cache),
            },
            "capture_counts": {"target_feature_rows": 4, "target_ladder_rows": 4},
            "files": {
                name: {
                    "bytes": (self.new / name).stat().st_size,
                    "sha256": ladder.sha256(self.new / name),
                }
                for name in names
            },
        }
        (self.new / "manifest.json").write_text(json.dumps(manifest))

    def validate(self):
        sealed = {
            "target_sha256": ladder.sha256(self.target),
            "draft_sha256": ladder.sha256(self.draft),
            "train_prompts_sha256": ladder.sha256(self.train),
            "cell_manifest_sha256": "c" * 64,
        }
        selected = (sealed, {"id": ladder.PROMPT_ID}, self.prefix, list(range(4)), self.sealed_cell)
        with (
            mock.patch.object(ladder, "SEALED_CAPTURE_SHA256", ladder.sha256(self.sealed_manifest)),
            mock.patch.object(ladder, "select_frozen_prompt", return_value=selected),
        ):
            return ladder.validate_capture(
                self.new,
                self.sealed,
                self.train,
                self.target,
                self.draft,
                self.binary,
                self.cmake_cache,
            )

    def test_accepts_complete_join_and_old_taps(self):
        prefix, values, proof = self.validate()
        self.assertEqual(prefix, self.prefix)
        self.assertEqual(values.shape, (4, ladder.LADDER_WIDTH))
        self.assertEqual(proof["target_source"], str(self.target.resolve()))

    def test_rejects_ladder_hash_corruption(self):
        with (self.new / "heads.target_layer_ladder.f32").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            self.validate()

    def test_rejects_wrong_row_join_even_with_updated_hash(self):
        self.ladder_rows[2]["token_id"] += 1
        self.write_rows()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "does not join"):
            self.validate()

    def test_rejects_changed_old_tap_even_with_updated_hash(self):
        self.values[3, ladder.LADDER_LAYERS.index(18) * ladder.HIDDEN] = np.float32(4.0)
        self.values.tofile(self.new / "heads.target_layer_ladder.f32")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "same-run feature"):
            self.validate()

    def test_rejects_changed_same_run_tap_even_with_updated_hash(self):
        self.feature[3, ladder.TAPS.index(18) * ladder.HIDDEN] = np.float32(4.0)
        self.feature.tofile(self.new / "heads.target_features.f32")
        self.values[3, ladder.LADDER_LAYERS.index(18) * ladder.HIDDEN] = np.float32(4.0)
        self.values.tofile(self.new / "heads.target_layer_ladder.f32")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "sealed full96"):
            self.validate()

    def test_rejects_target_source_path_change(self):
        self.ladder_rows[0]["target_source"] = str(self.draft)
        self.write_rows()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "target source changed"):
            self.validate()

    def test_rejects_coherent_wrong_target_source_path(self):
        for row in self.ladder_rows:
            row["target_source"] = str(self.draft)
        self.write_rows()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source path differs"):
            self.validate()

    def test_rejects_ladder_offset_with_updated_hash(self):
        self.ladder_rows[2]["byte_offset"] += 4
        self.write_rows()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "does not join"):
            self.validate()

    def test_rejects_missing_diagnostic_ladder_count(self):
        manifest_path = self.new / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["capture_counts"]["target_ladder_rows"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "target-ladder count"):
            self.validate()

    def test_rejects_binary_hash_change(self):
        self.binary.write_bytes(b"new binary")
        with self.assertRaisesRegex(ValueError, "source hashes differ"):
            self.validate()

    def test_layer_metrics_and_first_growth(self):
        actual = np.ones((4, ladder.HIDDEN), dtype=np.float32)
        expected = actual.copy()
        expected[3] += 0.5
        metrics = ladder.layer_metrics(expected, actual)
        self.assertAlmostEqual(metrics["positions"][3]["relative_l2"], 0.5)
        self.assertAlmostEqual(metrics["positions"][3]["rms"], 0.5)
        self.assertAlmostEqual(metrics["positions"][3]["max_abs"], 0.5)
        layers = {
            str(layer): {"positions": [{"relative_l2": 0.1} for _ in range(4)]}
            for layer in ladder.LADDER_LAYERS
        }
        layers["2"]["positions"][3]["relative_l2"] = 0.21
        self.assertEqual(ladder.first_sharp_growth(layers)["to_layer"], 2)


if __name__ == "__main__":
    unittest.main()
