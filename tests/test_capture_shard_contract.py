"""CPU-only frozen shard and raw-logit coverage contracts."""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_binary_head_capture as runner  # noqa: E402
from audit_recurrent_binary_capture import (  # noqa: E402
    resolve_prompt_expectation,
    validate_shard_manifest,
)


class ShardContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prompts = self.root / "train_prompts.jsonl"
        self.rows = [
            {"id": f"shard-train-{index}", "messages": [{"role": "user", "content": "x"}]}
            for index in range(2)
        ]
        self.prompts.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.prompt_hash = runner.sha256(self.prompts)
        self.shard_path = self.root / "shard.json"
        self.shard = {
            "schema": "w1ax_capture_shard_v1",
            "prompts_path": self.prompts.name,
            "prompts_sha256": self.prompt_hash,
            "prompt_count": 2,
            "prompt_ids": [row["id"] for row in self.rows],
            "parent": {
                "split": "train_small", "manifest_sha256": "a" * 64,
                "prompts_sha256": "b" * 64, "index_sha256": "c" * 64, "count": 2000,
            },
            "target_vocab_size": runner.TARGET_VOCAB_SIZE,
            "bytes_per_raw_logit_row": runner.TARGET_VOCAB_SIZE * 4,
            "caps": {
                "max_prompts": 96,
                "max_verifier_logit_rows": 32,
                "max_raw_logit_bytes": 32 * runner.TARGET_VOCAB_SIZE * 4,
            },
        }
        self.shard_path.write_text(json.dumps(self.shard))
        for name in ("binary", "target", "draft"):
            (self.root / name).write_text(name)
        self.map_path = self.root / "map.npy"
        mapping = np.arange(32_000, dtype="<i8")
        np.save(self.map_path, mapping)
        self.args = argparse.Namespace(
            mode="recurrent-train", tokens=128, prompts=self.prompts,
            prompt_manifest=None, prompts_sha256=None,
            shard_manifest=self.shard_path,
            expected_prompt_sha256=self.prompt_hash, expected_prompt_count=2,
            binary=self.root / "binary", target=self.root / "target",
            output=self.root / "out", d2t=self.map_path,
            target_vocab_size=runner.TARGET_VOCAB_SIZE,
            target_logits_limit=32, target_features_limit=128, port=18092,
        )
        self.variants = {
            "d_d": {"body": "D", "head": "D", "draft": str(self.root / "draft")}
        }
        self.model_patches = (
            mock.patch.object(runner, "TARGET_F16_SHA256", runner.sha256(self.args.target)),
            mock.patch.object(runner, "DRAFT_D_SHA256", runner.sha256(self.root / "draft")),
            mock.patch.object(
                runner, "DRAFT_D_D2T_SHA256", hashlib.sha256(mapping.tobytes()).hexdigest()
            ),
        )

    def validate(self):
        with self.model_patches[0], self.model_patches[1], self.model_patches[2]:
            runner.validate_inputs(self.args, self.rows, self.variants)

    def test_custom_shard_accepts_frozen_order_and_rejects_mutations(self):
        self.validate()
        validate_shard_manifest(
            self.shard_path, self.prompts, self.prompt_hash, 2, self.shard["prompt_ids"]
        )
        for change, pattern in (
            ({"prompt_ids": list(reversed(self.shard["prompt_ids"]))}, "ordered prompt IDs"),
            ({"prompt_count": 3}, "ordered prompt IDs"),
            ({"caps": {**self.shard["caps"], "max_verifier_logit_rows": 31}}, "raw-logit cap"),
        ):
            with self.subTest(change=change):
                self.shard_path.write_text(json.dumps({**self.shard, **change}))
                with self.assertRaisesRegex(ValueError, pattern):
                    self.validate()
                self.shard_path.write_text(json.dumps(self.shard))
        self.args.expected_prompt_count = None
        with self.assertRaisesRegex(ValueError, "requires manifest"):
            self.validate()

    def test_expected_pair_is_explicit(self):
        self.assertEqual(resolve_prompt_expectation(self.prompt_hash, 2), (self.prompt_hash, 2))
        with self.assertRaisesRegex(ValueError, "both expected"):
            resolve_prompt_expectation(self.prompt_hash, None)

    def test_custom_shard_rejects_partial_raw_logits(self):
        cell = self.root / "cell"
        cell.mkdir()
        (cell / "heads.jsonl").write_text(
            json.dumps({"task_id": 1, "target_logits_row": 0}) + "\n"
            + json.dumps({"task_id": 1, "target_logits_row": None}) + "\n"
        )
        (cell / "heads.target_features.jsonl").write_text(
            json.dumps({"event": "decoded_row", "feature_row": 0, "task_id": 1}) + "\n"
            + json.dumps({"event": "disposition", "feature_row": 0, "task_id": 1}) + "\n"
        )
        (cell / "heads.target_features.f32").write_bytes(b"\0" * runner.FEATURE_WIDTH * 4)
        (cell / "heads.target_logits.f32").write_bytes(b"\0" * 4 * 4)
        (cell / "forced-rounds.jsonl").write_text(json.dumps({"task_id": 1}) + "\n")
        manifest = {
            "task_prompt_ids": {"1": "shard-train-0"},
            "requests": [{
                "task_id": "1", "capture_rows": [0, 2], "forced_round_rows": [0, 1],
                "target_feature_event_rows": [0, 2], "target_feature_rows": [0, 1],
                "target_logit_rows": [0, 1],
            }],
        }
        runner.audit_recurrent_files(cell, manifest, 4, 1, 1)
        with self.assertRaisesRegex(ValueError, "exceed frozen raw-logit cap"):
            runner.audit_recurrent_files(
                cell, manifest, 4, 1, 1, require_full_logits=True
            )


if __name__ == "__main__":
    unittest.main()
