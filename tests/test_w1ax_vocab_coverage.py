"""Small, synthetic tests for the post-run draft-vocabulary audit."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_w1ax_vocab_coverage.py"
SPEC = importlib.util.spec_from_file_location("analyze_w1ax_vocab_coverage", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MappingTests(unittest.TestCase):
    def test_absolute_mapping_is_not_offset_again(self):
        ids = audit.absolute_draft_ids(np.array([1, 3, 5], dtype=np.int64), 8, 3)
        membership = np.zeros(8, dtype=np.bool_)
        membership[ids] = True
        self.assertEqual(audit.token_coverage([1, 2, 3, 5], membership)["in_draft_vocab"], 3)

    def test_duplicate_out_of_range_and_dtype_are_rejected(self):
        for values in (
            np.array([1, 1], dtype=np.int64),
            np.array([-1, 2], dtype=np.int64),
            np.array([1, 8], dtype=np.int64),
            np.array([1, 2], dtype=np.int32),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.absolute_draft_ids(values, 8, 2)

    def test_token_coverage_counts_occurrences_and_distinct_ids(self):
        membership = np.zeros(7, dtype=np.bool_)
        membership[[1, 3]] = True
        result = audit.token_coverage([1, 1, 2, 3, 6], membership)
        self.assertEqual(result["emitted_ids"], 5)
        self.assertEqual(result["in_draft_vocab"], 3)
        self.assertEqual(result["outside_draft_vocab"], 2)
        self.assertEqual(result["distinct_in_draft_vocab"], 2)
        self.assertEqual(result["distinct_outside_draft_vocab"], 2)
        self.assertEqual(result["fraction_in_draft_vocab"], 0.6)
        for bad in ([7], [-1], [True], [1.0], None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit.token_coverage(bad, membership)


class RunTests(unittest.TestCase):
    def test_run_reports_target_and_variant_coverage_with_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            gguf = run / "draft.gguf"
            gguf.write_bytes(b"synthetic GGUF placeholder")
            (run / "prompts.jsonl").write_text(
                '{"id":"p1","category":"prose"}\n{"id":"p2","category":"code"}\n'
            )
            (run / "manifest.json").write_text('{"status":"complete"}\n')
            rows = [
                {
                    "variant": "target_only",
                    "repetition": 0,
                    "prompt_id": "p1",
                    "generated_token_ids": [1, 2, 1],
                },
                {
                    "variant": "target_only",
                    "repetition": 0,
                    "prompt_id": "p2",
                    "generated_token_ids": [4],
                },
                {
                    "variant": "draft_w1a1",
                    "repetition": 0,
                    "prompt_id": "p1",
                    "generated_token_ids": [1, 3],
                },
                {
                    "variant": "draft_w1a1",
                    "repetition": 0,
                    "prompt_id": "p2",
                    "generated_token_ids": [2, 4],
                },
            ]
            (run / "records.json").write_text(json.dumps(rows))
            with patch.object(
                audit, "read_draft_ids", return_value=np.array([1, 3], dtype=np.int64)
            ):
                result = audit.analyze(gguf, [run], target_vocab_size=5, expected_draft_size=2)
            self.assertEqual(result["target_only"]["overall"]["in_draft_vocab"], 2)
            self.assertEqual(result["target_only"]["overall"]["emitted_ids"], 4)
            self.assertEqual(result["target_only"]["by_category"]["code"]["outside_draft_vocab"], 1)
            prompt_key = f"{audit.sha256(run / 'prompts.jsonl')}:p1"
            self.assertEqual(result["target_only"]["by_prompt"][prompt_key]["in_draft_vocab"], 2)
            self.assertEqual(result["variants"]["draft_w1a1"]["overall"]["in_draft_vocab"], 2)
            self.assertEqual(
                result["variants"]["draft_w1a1"]["by_category"]["code"]["outside_draft_vocab"], 2
            )
            self.assertEqual(result["gguf"]["sha256"], audit.sha256(gguf))
            self.assertEqual(
                result["inputs"][0]["source_sha256"]["records.json"],
                audit.sha256(run / "records.json"),
            )
            self.assertEqual(
                result["inputs"][0]["source_sha256"]["prompts.jsonl"],
                audit.sha256(run / "prompts.jsonl"),
            )
            self.assertEqual(
                result["inputs"][0]["source_sha256"]["manifest.json"],
                audit.sha256(run / "manifest.json"),
            )
            self.assertIn("probability mass is unavailable", result["interpretation"])

    def test_duplicate_record_and_missing_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "prompts.jsonl").write_text('{"id":"p","category":"prose"}\n')
            row = {
                "variant": "target_only",
                "repetition": 0,
                "prompt_id": "p",
                "generated_token_ids": [1],
            }
            (run / "records.json").write_text(json.dumps([row, row]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.analyze_run(run, np.ones(3, dtype=np.bool_))
            (run / "records.json").write_text(json.dumps([{**row, "generated_token_ids": None}]))
            with self.assertRaisesRegex(ValueError, "generated_token_ids"):
                audit.analyze_run(run, np.ones(3, dtype=np.bool_))


if __name__ == "__main__":
    unittest.main()
