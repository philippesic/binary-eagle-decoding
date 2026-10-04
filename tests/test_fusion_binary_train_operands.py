"""Independent negative gates for saved native fusion operand lineage."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fusion_train", ROOT / "scripts/prepare_fusion_binary_train_operands.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FusionTrainOperandTest(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros(7680, dtype=np.float32)
        self.evidence = {
            "row_id": "train-one:feature:7",
            "feature_row": 7,
            "raw_input_sha256": module.digest(self.x.tobytes()),
            "prompt_token_ids": [3, 9],
            "feature_metadata": {
                "feature_row": 7,
                "prompt_id": "train-one",
                "position": 1,
                "prefix_token_ids": [3, 9],
                "tap_ids": [2, 18, 33],
                "boundary": module.BOUNDARY,
                "source": "native_target_features_on_accepted_prefix",
                "accepted_prefix": True,
                "native_feature_row": 41,
                "native_decode_ordinal": 15,
                "task_id": 2,
            },
            "anchor": {
                "prompt_id": "train-one",
                "prefix_token_ids": [3, 9, 8],
                "seed_token_id": 7,
                "round_index": 4,
                "split": "train",
            },
            "native_events": {
                "decoded_row": {
                    "feature_row": 41,
                    "task_id": 2,
                    "decode_ordinal": 15,
                    "position": 1,
                    "prefix_token_ids": [3, 9],
                    "boundary": "raw_target_layer_input_before_eagle_encoder",
                    "source": "target_verifier",
                    "feature_dtype": "float32_native_endian",
                    "target_layer_ids": [2, 18, 33],
                    "feature_dim": 7680,
                    "phase": "speculative",
                    "round_index": 4,
                    "spec_input_row": 1,
                },
                "disposition": {
                    "feature_row": 41,
                    "task_id": 2,
                    "retained_input": True,
                    "reason": "accepted_prefix",
                    "round_index": 4,
                    "accepted_drafts": 1,
                },
            },
        }

    def test_exact_accepted_raw_operand(self):
        module.validate_row_evidence(self.x, self.evidence)

    def test_rejected_suffix_is_not_training_evidence(self):
        self.evidence["native_events"]["disposition"]["retained_input"] = False
        with self.assertRaisesRegex(ValueError, "retained-input"):
            module.validate_row_evidence(self.x, self.evidence)

    def test_stale_speculative_cache_join_fails(self):
        self.evidence["native_events"]["disposition"]["accepted_drafts"] = 0
        with self.assertRaisesRegex(ValueError, "accepted cache"):
            module.validate_row_evidence(self.x, self.evidence)

    def test_raw_cast_or_target_source_change_fails(self):
        for field, value in (("feature_dtype", "float16"), ("source", "diagnostic")):
            evidence = copy.deepcopy(self.evidence)
            evidence["native_events"]["decoded_row"][field] = value
            with self.assertRaisesRegex(ValueError, "producer/native"):
                module.validate_row_evidence(self.x, evidence)

    def test_wrong_prefix_or_anchor_split_fails(self):
        for field, value in (("prefix_token_ids", [3, 8]), ("split", "sealed_test")):
            evidence = copy.deepcopy(self.evidence)
            evidence["anchor"][field] = value
            with self.assertRaisesRegex(ValueError, "cache anchor"):
                module.validate_row_evidence(self.x, evidence)

    def test_changed_finite_operand_bytes_fail(self):
        self.x[7000] = np.float32(0.25)
        with self.assertRaisesRegex(ValueError, "operand bytes"):
            module.validate_row_evidence(self.x, self.evidence)

    def test_unpinned_archive_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "saved.zip"
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "package SHA"):
                module.prepare(path, "0" * 64, "absent", "absent", "absent", "absent")

    def test_unicode_content_identity_is_stable(self):
        a = {"messages": [{"role": "user", "content": "pi\u03c0\u2028message"}]}
        b = {"messages": [{"content": "pi\u03c0\u2028message", "role": "user"}]}
        self.assertEqual(module.message_hash(a), module.message_hash(b))
        b["messages"][0]["content"] += "changed"
        self.assertNotEqual(module.message_hash(a), module.message_hash(b))


if __name__ == "__main__":
    unittest.main()
