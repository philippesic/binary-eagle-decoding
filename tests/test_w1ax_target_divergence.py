"""Synthetic contracts for the historical target divergence replay."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_w1ax_target_divergence", ROOT / "scripts" / "check_w1ax_target_divergence.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostic)


def response(ids, logprobs=None, **verbose_fields):
    verbose = {"tokens": ids, **verbose_fields}
    choice = {"message": {"content": "synthetic"}, "finish_reason": "length"}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"__verbose": verbose, "choices": [choice]}


class W1AxTargetDivergenceTests(unittest.TestCase):
    def test_known_mismatch_is_context_not_an_asserted_expected_result(self):
        target_ids = [100] * 110
        eagle_ids = [100] * 110
        target_ids[109] = 12
        eagle_ids[109] = 29208
        target_logprobs = [{"token": str(value), "logprob": -0.2, "top_logprobs": []} for value in target_ids]
        eagle_logprobs = [{"token": str(value), "logprob": -0.3, "top_logprobs": []} for value in eagle_ids]
        result = diagnostic.comparison_record(
            response(target_ids, target_logprobs), response(eagle_ids, eagle_logprobs), "reasoning-02"
        )
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["first_mismatch"], {
            "zero_based_position": 109,
            "target_only_token_id": 12,
            "ordinary_eagle_token_id": 29208,
            "target_only_length": 110,
            "ordinary_eagle_length": 110,
        })
        self.assertTrue(result["historical_reference"]["matches_observed_first_mismatch"])
        self.assertEqual(
            result["api_completion_logprobs_at_first_mismatch"]["ordinary_eagle"]["entry"]["token"],
            "29208",
        )
        self.assertEqual(
            result["api_completion_logprobs_at_first_mismatch"]["ordinary_eagle"]["status"],
            "available",
        )
        self.assertEqual(
            result["verifier_row_logits"]["ordinary_eagle"]["status"], "unavailable"
        )
        self.assertTrue(
            result["api_completion_logprobs_at_first_mismatch"]["interpretation"].startswith("OpenAI-compatible")
        )

    def test_history_reference_does_not_force_different_observation(self):
        target = response([1, 2, 3])
        eagle = response([1, 2, 4])
        result = diagnostic.comparison_record(target, eagle, "reasoning-02")
        self.assertEqual(result["first_mismatch"]["zero_based_position"], 2)
        self.assertFalse(result["historical_reference"]["matches_observed_first_mismatch"])
        self.assertTrue(result["historical_reference"]["is_expected_diagnostic_context_only"])

    def test_api_logprobs_are_not_accepted_as_verifier_row_logits(self):
        logprobs = [{"token": "x", "logprob": -0.1, "top_logprobs": []}]
        result = diagnostic.verifier_row_logits(response([1], logprobs), 0)
        self.assertEqual(result["status"], "unavailable")
        explicit = diagnostic.verifier_row_logits({"verifier_row_logits": [[0.1, 0.9]]}, 0)
        self.assertEqual(explicit["status"], "available")
        self.assertEqual(explicit["position"], [0.1, 0.9])

    def test_incomplete_or_token_id_misaligned_logprobs_are_not_associated(self):
        ids = [4, 5, 6]
        incomplete = response(ids, [{"token": "4"}, {"token": "5"}])
        result = diagnostic.api_logprob_entry(incomplete, 1, ids)
        self.assertEqual(result["status"], "misaligned")
        self.assertEqual(result["logprob_content_count"], 2)
        self.assertEqual(result["generated_token_id_count"], 3)

        wrong_id = response(ids, [
            {"token": "4", "id": 4},
            {"token": "5", "token_id": 99},
            {"token": "6", "id": 6},
        ])
        result = diagnostic.api_logprob_entry(wrong_id, 1, ids)
        self.assertEqual(result["status"], "misaligned")
        self.assertEqual(result["generated_token_id"], 5)
        self.assertEqual(result["logprob_entry_token_id"], 99)

    def test_missing_ids_and_length_mismatch_are_preserved(self):
        missing = diagnostic.comparison_record({}, {}, "reasoning-02")
        self.assertIsNone(missing["exact_match"])
        self.assertIsNone(missing["first_mismatch"])
        shorter = diagnostic.comparison_record(response([1, 2]), response([1]), "other")
        self.assertEqual(shorter["first_mismatch"]["zero_based_position"], 1)
        self.assertIsNone(shorter["first_mismatch"]["ordinary_eagle_token_id"])

    def test_verifier_trace_environment_and_artifact_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary)
            environment = diagnostic.verifier_trace_environment(cell)
            self.assertEqual(environment["W1AX_VERIFY_TRACE_JSONL"], str(cell / "verifier-trace.jsonl"))
            self.assertEqual(environment["W1AX_VERIFY_TRACE_POSITIONS"], "109")
            trace_path = Path(environment["W1AX_VERIFY_TRACE_JSONL"])
            missing = diagnostic.inspect_verifier_trace(trace_path)
            self.assertEqual(missing["status"], "unavailable")

            trace_path.write_text(json.dumps({
                "schema": "w1ax_verify_logits_v1", "position": 109,
                "sampled": True, "emitted": True, "top5": [],
            }) + "\n")
            available = diagnostic.inspect_verifier_trace(trace_path)
            self.assertEqual(available["status"], "available")
            self.assertTrue(available["schema_confirmed"])
            self.assertEqual(available["event_lines"], 1)
            self.assertEqual(len(available["sha256"]), 64)

            trace_path.write_text('{"schema":"unexpected"}\n')
            wrong_schema = diagnostic.inspect_verifier_trace(trace_path)
            self.assertEqual(wrong_schema["status"], "available")
            self.assertFalse(wrong_schema["schema_confirmed"])


if __name__ == "__main__":
    unittest.main()
