"""CPU-only frozen-split capture metadata gate."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from w1a1_eagle.recurrent_trace import LABEL_SOURCE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_recurrent_binary_capture import audit_capture, sha256  # noqa: E402


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
            "invalid_reason": None,
            "label_source": LABEL_SOURCE,
            "verifier_token_id": 4,
            "proposed_token_id": 5,
            "label_supported": True,
            "verifier_logits": [0.0] * 8,
        }
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

    def test_rejects_wrong_split_or_changed_rows(self):
        self.assertRaisesRegex(
            ValueError, "frozen training", audit_capture, self.manifest, self.prompts, "0" * 64
        )
        self.row["input_position"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "position"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_file_hash_mismatch(self):
        self.rows.write_text(self.rows.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def test_rejects_inconsistent_inverse_vocabulary_mask(self):
        np.save(self.t2d, np.zeros(8, dtype=bool))
        self.save_manifest_hashes_only()
        with self.assertRaisesRegex(ValueError, "t2d mask disagrees"):
            audit_capture(self.manifest, self.prompts, self.prompt_hash)

    def save_manifest_hashes_only(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["t2d"]["sha256"] = sha256(self.t2d)
        self.manifest.write_text(json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
