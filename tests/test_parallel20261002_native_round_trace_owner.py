import unittest

from research.parallel20261002.native_round_trace.reference.adapter import (
    adapt_session,
    pool_sessions,
)

STAGES = (
    "begin",
    "draft",
    "checkpoint",
    "target_decode_sync",
    "process",
    "check",
    "kv_repair",
    "accept_hook",
)


def event(index, proposed, accepted, emitted, status="complete", start=100, replay=False):
    # Exactly emit_round_trace field names; zero absent spans, details nested.
    spans = {s: [0, 0] for s in STAGES}
    spans.update(
        draft=[start + 1, start + 6],
        target_decode_sync=[start + 5, start + 12],
        process=[start + 12, start + 17],
    )
    row = dict(
        schema="w1ax_eagle_round_v1",
        clock="ggml_time_us_cpu_wall",
        task_id=7,
        parent_task_id=-1,
        slot_id=0,
        round_index=index,
        status=status,
        replay=replay,
        round_start_us=start,
        round_end_us=start + 20,
        round_us=20,
        n_proposed=len(proposed),
        n_accepted=accepted,
        n_emitted=len(emitted),
        proposed_token_ids=proposed,
        emitted_token_ids=emitted,
        spans_us=spans,
        residual_us=3,
    )
    row.update({s + "_us": b - a for s, (a, b) in spans.items()})
    return row


def metadata(output, **extra):
    return dict(
        session_id="synthetic7",
        sampling_mode="greedy",
        prompt_token_ids=[1, 2],
        generated_token_ids=output,
        generation_cap=len(output),
        eos_token_ids=[99],
        ancestry={
            "source": "9e2c7a9",
            "device": "synthetic CPU callbacks",
            "target_precision": "unexecuted",
        },
        **extra,
    )


class OwnerTests(unittest.TestCase):
    def test_roles_roots_and_exact_native_totals(self):
        rows = [event(0, [10, 11], 2, [10, 11, 12]), event(1, [13, 14], 0, [15], start=130)]
        result = adapt_session(
            rows, metadata([9, 10, 11, 12, 15]), dict(proposed=4, accepted=2, rounds=2)
        )
        self.assertEqual([r["root_generated_position"] for r in result["rounds"]], [1, 4])
        self.assertEqual([r["remaining_cap"] for r in result["rounds"]], [4, 1])
        self.assertEqual(result["counts"]["bonus"], 1)
        self.assertEqual(result["counts"]["correction"], 1)
        self.assertEqual(result["leading_untraced_emitted"], 1)
        self.assertIsNone(result["costs"]["full_session_us"])

    def test_terminal_acceptance_truncation(self):
        result = adapt_session([event(0, [10, 99, 12], 3, [10, 99])], metadata([10, 99]))
        self.assertEqual(result["counts"]["accepted_decisions"], 3)
        self.assertEqual(result["counts"]["accepted_emitted"], 2)
        self.assertEqual(result["counts"]["accepted_not_emitted"], 1)
        self.assertEqual(result["rounds"][0]["terminal_reasons"], ["eos", "cap_reached"])

    def test_overlap_not_clamped_residual(self):
        result = adapt_session([event(0, [10], 0, [11])], metadata([11]))
        self.assertEqual(result["rounds"][0]["stage_overlap_us"], 1)
        self.assertEqual(result["costs"]["other_inside_round_us"], 4)

    def test_replay_cost_not_acceptance(self):
        rows = [
            event(0, [10, 11], 1, [], status="checkpoint_replay"),
            event(1, [10, 11], 0, [10, 12], start=130, replay=True),
        ]
        result = adapt_session(rows, metadata([10, 12]), dict(proposed=4, accepted=0, rounds=1))
        self.assertEqual(result["counts"]["checkpoint_attempts"], 1)
        self.assertEqual(result["counts"]["accepted_decisions"], 0)
        self.assertEqual(result["counts"]["unresolved_role_emitted"], 2)
        self.assertEqual(result["costs"]["round_sum_us"], 40)

    def test_shared_clock_cost_union(self):
        left = adapt_session([event(0, [10], 0, [11])], metadata([11], clock_domain="process42"))
        right = adapt_session([event(0, [10], 0, [11])], metadata([11], clock_domain="process42"))
        right["session_id"] = "synthetic8"
        result = pool_sessions([left, right])
        self.assertEqual(result["round_union_us_by_clock"], {"process42": 20})
        self.assertEqual(result["counts"]["emitted"], 2)
        self.assertFalse(result["session_ranking_identified"])

    def test_boolean_counts_rejected(self):
        row = event(0, [10], 0, [11])
        row["n_emitted"] = True
        with self.assertRaisesRegex(ValueError, "count/ID"):
            adapt_session([row], metadata([11]))

    def test_callback_vectors_are_distinct_from_envelopes(self):
        row = event(0, [10], 0, [11])
        row["session_context_v1"] = dict(
            generated_before=0,
            remaining_before=1,
            native_stop_type="limit",
            target_batches=[
                dict(start_us=105, end_us=107, batch_tokens=2, slot_tokens=2, slot_output_rows=2),
                dict(start_us=110, end_us=112, batch_tokens=1, slot_tokens=1, slot_output_rows=1),
            ],
            process_batches=[
                dict(
                    start_us=112,
                    end_us=117,
                    batch_tokens=1,
                    feature_tokens=1,
                    draft_decode_tokens=2,
                )
            ],
        )
        result = adapt_session([row], metadata([11]))
        self.assertEqual(result["costs"]["stage_inclusive_union_us"]["target_decode_sync"], 7)
        self.assertEqual(result["costs"]["callback_union_us_by_stage"]["target_decode_sync"], 4)
        self.assertEqual(result["costs"]["callback_union_us_by_stage"]["process"], 5)

    def test_pool_rejects_unbound_or_duplicate_sessions(self):
        session = adapt_session([event(0, [10], 0, [11])], metadata([11]))
        with self.assertRaisesRegex(ValueError, "bound process"):
            pool_sessions([session])
        session["costs"]["clock_domain"] = "process42"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pool_sessions([session, session])

    def test_mismatched_native_counter_fails(self):
        with self.assertRaisesRegex(ValueError, "aggregate mismatch"):
            adapt_session([event(0, [10], 0, [11])], metadata([11]), {"proposed": 2})

    def test_unsupported_ratio_verifier_fails(self):
        m = metadata([11])
        m["sampling_mode"] = "probability_ratio"
        with self.assertRaisesRegex(ValueError, "unsupported verifier"):
            adapt_session([event(0, [10], 0, [11])], m)

    def test_legacy_unknown_state_preserved(self):
        result = adapt_session(
            [event(0, [], 0, [11], status="no_proposal")], {"sampling_mode": "sample_and_match"}
        )
        self.assertIn("committed_root_identity", result["unresolved_domains"])
        self.assertIn("remaining_cap", result["unresolved_domains"])
        self.assertEqual(result["counts"]["target_no_proposal"], 1)


if __name__ == "__main__":
    unittest.main()
