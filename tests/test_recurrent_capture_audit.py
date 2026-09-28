"""CPU-only frozen-split capture metadata gate."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from w1a1_eagle.recurrent_trace import LABEL_SOURCE, VERIFIER_LOGITS_SOURCE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from audit_recurrent_binary_capture import (  # noqa: E402
    FEATURE_BOUNDARY,
    FEATURE_SOURCE,
    FEATURE_TAPS,
    FEATURE_WIDTH,
    audit_capture,
    load_audited_capture,
    sha256,
    streamed_mapped_probability_mass,
)

from w1a1_eagle.recurrent_rollout import DraftStep, rebuild_prefix_cache  # noqa: E402


class RecurrentCaptureAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.prompts = self.directory / "train.jsonl"
        self.prompts.write_text(
            "".join(json.dumps({"id": f"p-{index}"}) + "\n" for index in range(96))
        )
        self.prompt_hash = sha256(self.prompts)
        self.rows = self.directory / "rows.jsonl"
        self.anchors = self.directory / "anchors.jsonl"
        self.offsets = self.directory / "offsets.npy"
        self.t2d = self.directory / "t2d.npy"
        self.target_logits = self.directory / "target-logits.f32"
        self.features = self.directory / "features.npy"
        self.feature_rows = self.directory / "feature-rows.jsonl"
        self.manifest = self.directory / "manifest.json"
        self.row = {
            "prompt_id": "p-0",
            "split": "train",
            "round_index": 0,
            "depth": 0,
            "parent_position": 1,
            "input_position": 2,
            "label_position": 3,
            "verifier_row": 0,
            "prefix_token_ids": [0, 1, 3],
            "input_token_id": 3,
            "alignment_valid": True,
            "is_bonus": False,
            "valid": True,
            "verifier_reached": True,
            "invalid_reason": None,
            "label_source": LABEL_SOURCE,
            "verifier_token_id": 4,
            "proposed_token_id": 5,
            "label_supported": True,
            "target_logits_row": 0,
            "target_logits_dim": 8,
            "target_logits_source": VERIFIER_LOGITS_SOURCE,
        }
        self.feature_metadata = [
            {
                "feature_row": index,
                "prompt_id": "p-0",
                "position": index,
                "prefix_token_ids": [0, 1][: index + 1],
                "tap_ids": FEATURE_TAPS,
                "boundary": FEATURE_BOUNDARY,
                "source": FEATURE_SOURCE,
                "accepted_prefix": True,
            }
            for index in range(2)
        ]
        self.save()

    def save(self):
        self.rows.write_text(json.dumps(self.row) + "\n")
        self.anchors.write_text(
            json.dumps(
                {
                    "prompt_id": "p-0",
                    "split": "train",
                    "round_index": 0,
                    "prefix_token_ids": [0, 1],
                    "seed_token_id": 3,
                }
            )
            + "\n"
        )
        np.save(self.offsets, np.array([2, 3, 3], dtype=np.int64))
        np.save(self.t2d, np.array([False, False, True, False, True, True, False, False]))
        self.target_logits.write_bytes(np.zeros(8, dtype="<f4").tobytes())
        features = np.zeros((len(self.feature_metadata), FEATURE_WIDTH), dtype=np.float32)
        features[:, 0] = np.arange(len(features), dtype=np.float32)
        np.save(self.features, features)
        self.feature_rows.write_text(
            "".join(json.dumps(row) + "\n" for row in self.feature_metadata)
        )
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": "recurrent_binary_capture_v1",
                    "split": "train",
                    "prompts_sha256": self.prompt_hash,
                    "target_vocab_size": 8,
                    "draft_vocab_size": 3,
                    "max_depth": 5,
                    **{
                        field: {"path": path.name, "sha256": sha256(path)}
                        for field, path in (
                            ("rows", self.rows),
                            ("anchors", self.anchors),
                            ("offsets", self.offsets),
                            ("t2d", self.t2d),
                            ("target_logits", self.target_logits),
                            ("features", self.features),
                            ("feature_rows", self.feature_rows),
                        )
                    },
                }
            )
        )

    def test_audits_frozen_prompt_hash_alignment_and_mass(self):
        report = audit_capture(self.manifest, self.prompts, self.prompt_hash)
        self.assertEqual(report["counts"]["supported"], 1)
        self.assertAlmostEqual(report["mapped_probability_mass_mean_on_sampled_logit_rows"], 3 / 8)
        self.assertEqual(report["real_model_feature_and_kv_parity"], "unverified")
        self.assertEqual(report["feature_ledger"]["rows"], 2)
        self.assertEqual(report["feature_ledger"]["tap_ids"], FEATURE_TAPS)
        self.assertEqual(report["raw_target_logit_rows"], 1)

    def test_streamed_raw_logits_reject_nonfinite_and_zero_mass_rows(self):
        for value, message in (
            (np.nan, "invalid verifier logits"),
            (np.inf, "invalid verifier logits"),
            (-np.inf, "zero probability mass"),
        ):
            with self.subTest(value=value):
                self.save()
                self.target_logits.write_bytes(np.full(8, value, dtype="<f4").tobytes())
                manifest = json.loads(self.manifest.read_text())
                manifest["target_logits"]["sha256"] = sha256(self.target_logits)
                self.manifest.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_mapped_mass_streams_across_batch_boundary(self):
        np.zeros((17, 8), dtype="<f4").tofile(self.target_logits)
        target_to_draft = (-1, -1, 0, -1, 1, 2, -1, -1)
        self.assertAlmostEqual(
            streamed_mapped_probability_mass(self.target_logits, 17, 8, target_to_draft),
            3 / 8,
        )

    def test_audited_round_loads_prefix_features_for_rebuild(self):
        loaded = load_audited_capture(self.manifest, self.prompts, self.prompt_hash)
        round_input = loaded.round_inputs("p-0", 0)
        self.assertEqual(round_input.prefix_token_ids, (0, 1, 3))
        self.assertEqual(round_input.feature_positions, (0, 1))
        self.assertEqual(len(round_input.rows), 1)
        self.assertEqual(round_input.raw_target_features.shape, (2, FEATURE_WIDTH))
        observed = []

        def step(token, feature, position, cache):
            observed.append((token, position, float(feature[0])))
            return DraftStep(torch.zeros(3, device="cpu"), feature, (*cache, position))

        rebuilt = rebuild_prefix_cache(
            round_input.prefix_token_ids,
            round_input.raw_target_features,
            round_input.feature_positions,
            parent_position=1,
            encode_feature=lambda raw: raw[:2] + 1,
            decode_context=step,
            new_cache=tuple,
        )
        self.assertEqual(observed, [(1, 0, 1.0)])
        self.assertEqual(rebuilt.seed_token, 3)
        self.assertEqual(rebuilt.decoder_position, 1)
        self.assertEqual(float(rebuilt.seed_raw_features[0]), 1.0)
        with self.assertRaises(KeyError):
            loaded.round_inputs("p-0", 1)

    def test_rejects_wrong_split_or_changed_rows(self):
        self.assertRaisesRegex(
            ValueError, "frozen training", audit_capture, self.manifest, self.prompts, "0" * 64
        )
        self.row["input_position"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "position"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_draft_logit_substitution_and_bad_target_join(self):
        self.row["target_logits_source"] = "native_mapped_target_vocabulary_before_sampler"
        self.save()
        with self.assertRaisesRegex(ValueError, "source"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)
        self.row["target_logits_source"] = VERIFIER_LOGITS_SOURCE
        self.row["target_logits_row"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "row"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_missing_or_corrupt_target_logit_file(self):
        manifest = json.loads(self.manifest.read_text())
        del manifest["target_logits"]
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "require the native"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)
        self.save()
        self.target_logits.write_bytes(b"wrong")
        self.save_manifest_hashes_only("target_logits", self.target_logits)
        with self.assertRaisesRegex(ValueError, "payload size"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_file_hash_mismatch(self):
        self.rows.write_text(self.rows.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_inconsistent_inverse_vocabulary_mask(self):
        np.save(self.t2d, np.zeros(8, dtype=bool))
        self.save_manifest_hashes_only("t2d", self.t2d)
        with self.assertRaisesRegex(ValueError, "t2d mask disagrees"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_missing_or_rejected_branch_feature_rows(self):
        self.feature_metadata.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "feature row is missing"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)
        self.feature_metadata.append(
            {
                "feature_row": 1,
                "prompt_id": "p-0",
                "position": 1,
                "prefix_token_ids": [0, 4],
                "tap_ids": FEATURE_TAPS,
                "boundary": FEATURE_BOUNDARY,
                "source": FEATURE_SOURCE,
                "accepted_prefix": False,
            }
        )
        self.save()
        with self.assertRaisesRegex(ValueError, "accepted-prefix provenance"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_wrong_feature_tap_order_or_position(self):
        self.feature_metadata[1]["tap_ids"] = [33, 18, 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "accepted-prefix provenance"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)
        self.feature_metadata[1]["tap_ids"] = FEATURE_TAPS
        self.feature_metadata[1]["position"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "position"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def save_manifest_hashes_only(self, field, path):
        manifest = json.loads(self.manifest.read_text())
        manifest[field]["sha256"] = sha256(path)
        self.manifest.write_text(json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
