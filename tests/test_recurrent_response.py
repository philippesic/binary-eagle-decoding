"""CPU-only joins of canonical rounds, terminal trace and raw response IDs."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_recurrent_continuity as continuity_fixture  # noqa: E402
from audit_recurrent_response import audit_response  # noqa: E402


class ResponseJoinTests(unittest.TestCase):
    def setUp(self):
        fixture = continuity_fixture.ContinuityTests(
            methodName="test_eos_is_not_inferred_as_full_emission"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.add_feature(7, [0, 1, 3, 5, 6, 7], "target_only", 4, True)
        fixture.save()
        self.trace_path = fixture.root / "round_trace.jsonl"
        self.request_path = fixture.root / "request.json"
        self.response_path = fixture.root / "response.json"
        self.trace = []
        for record in fixture.rounds:
            self.trace.append(
                {
                    "schema": "w1ax_eagle_round_v1",
                    "task_id": 9,
                    "round_index": record["round_index"],
                    "status": "complete",
                    "replay": False,
                    "n_proposed": len(record["draft_token_ids"]),
                    "n_accepted": record["accepted_drafts"],
                    "n_emitted": len(record["verifier_token_ids"]),
                    "proposed_token_ids": record["draft_token_ids"],
                    "emitted_token_ids": record["verifier_token_ids"],
                }
            )
        self.trace.append(
            {
                "schema": "w1ax_eagle_round_v1",
                "task_id": 9,
                "round_index": 2,
                "status": "no_proposal",
                "replay": False,
                "n_proposed": 0,
                "n_accepted": 0,
                "n_emitted": 1,
                "proposed_token_ids": [],
                "emitted_token_ids": [8],
            }
        )
        self.request = {"max_tokens": 5}
        self.response = {
            "__verbose": {
                "tokens": [3, 5, 6, 7, 8],
                "tokens_predicted": 5,
                "tokens_evaluated": 2,
                "truncated": False,
            },
            "usage": {"completion_tokens": 5, "prompt_tokens": 2},
            "choices": [{"finish_reason": "length"}],
        }
        self.save()

    def save(self):
        continuity_fixture.write_jsonl(self.trace_path, self.trace)
        self.request_path.write_text(json.dumps(self.request))
        self.response_path.write_text(json.dumps(self.response))

    def audit(self):
        f = self.fixture
        return audit_response(
            f.rounds_path,
            self.trace_path,
            f.features_path,
            f.mapping,
            self.request_path,
            self.response_path,
            "9",
        )

    def test_exact_response_including_no_proposal_terminal_trace(self):
        result = self.audit()
        self.assertEqual(result["status"], "captured_response_emissions_verified")
        self.assertEqual(result["round_emission_prefix_tokens"], 4)
        self.assertEqual(result["terminal_trace_tokens"], 1)
        self.assertEqual(result["accepted_drafts"], 1)
        self.assertEqual(result["terminal_sampler_parity"], "unverified")

    def test_rejects_changed_terminal_response_and_trace(self):
        self.response["__verbose"]["tokens"][-1] = 9
        self.save()
        with self.assertRaisesRegex(ValueError, "raw response differs"):
            self.audit()
        self.response["__verbose"]["tokens"][-1] = 8
        self.trace[-1]["emitted_token_ids"] = [9]
        self.save()
        with self.assertRaisesRegex(ValueError, "raw response differs"):
            self.audit()
        self.trace[-1]["emitted_token_ids"] = [8]
        self.trace[-1]["status"] = "complete"
        self.save()
        with self.assertRaisesRegex(ValueError, "terminal no-proposal"):
            self.audit()

    def test_rejects_missing_terminal_input_and_wrong_request_counts(self):
        self.fixture.events = [row for row in self.fixture.events if row["feature_row"] != 7]
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "terminal no-proposal"):
            self.audit()
        self.fixture.add_feature(7, [0, 1, 3, 5, 6, 7], "target_only", 4, True)
        self.fixture.save()
        self.response["usage"]["prompt_tokens"] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, "counts or prompt"):
            self.audit()

    def test_rejects_wrong_round_emission_and_output_cap(self):
        self.trace[1]["emitted_token_ids"] = [6, 8]
        self.save()
        with self.assertRaisesRegex(ValueError, "round trace disagrees"):
            self.audit()
        self.trace[1]["emitted_token_ids"] = [6, 7]
        self.request["max_tokens"] = 4
        self.save()
        with self.assertRaisesRegex(ValueError, "output length"):
            self.audit()

    def test_final_eos_clips_only_the_unemitted_canonical_suffix(self):
        self.fixture.rounds[1]["draft_token_ids"] = [6, 151645]
        self.fixture.rounds[1]["accepted_drafts"] = 2
        self.fixture.rounds[1]["verifier_token_ids"] = [6, 151645, 198]
        self.fixture.events = [row for row in self.fixture.events if row["feature_row"] != 7]
        for row in self.fixture.events:
            if row["feature_row"] == 6:
                if row["event"] == "decoded_row":
                    row["token_id"] = 151645
                    row["prefix_token_ids"][-1] = 151645
                else:
                    row["retained_input"] = True
                    row["reason"] = "accepted_prefix"
                    row["accepted_drafts"] = 2
            elif row["event"] == "disposition" and row["round_index"] == 1:
                row["accepted_drafts"] = 2
        self.fixture.save()
        self.trace[1]["proposed_token_ids"] = [6, 151645]
        self.trace[1]["n_accepted"] = 2
        self.trace[1]["emitted_token_ids"] = [6, 151645]
        self.trace[1]["n_emitted"] = 2
        self.trace.pop()  # no target-only terminal decode after EOS
        self.response["__verbose"]["tokens"] = [3, 5, 6, 151645]
        self.response["__verbose"]["tokens_predicted"] = 4
        self.response["usage"]["completion_tokens"] = 4
        self.response["choices"][0]["finish_reason"] = "stop"
        self.save()
        self.assertEqual(self.audit()["eos_clipped_final_rounds"], 1)
        self.trace[1]["emitted_token_ids"] = [6, 7]
        self.save()
        with self.assertRaisesRegex(ValueError, "round trace disagrees"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
