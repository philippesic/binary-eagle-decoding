"""Independent producer-shaped checks for native round accounting."""
from __future__ import annotations

import copy
import unittest

from research.parallel20261002.native_round_trace.reference.adapter import adapt_session


BASE = {
    "schema": "w1ax_eagle_round_v1",
    "clock": "ggml_time_us_cpu_wall",
    "task_id": 41,
    "parent_task_id": -1,
    "slot_id": 0,
    "round_index": 0,
    "status": "complete",
    "replay": False,
    "n_draft_max": 4,
    "n_draft_configured": 4,
    "draft_p_min": 0.0,
    "stopped_low_confidence": False,
    "stop_probability": -1.0,
    "discarded_below_min": False,
    "process_feature_copy_us": 0,
    "process_encoder_us": 0,
    "process_batch_build_us": 0,
    "process_draft_decode_us": 0,
    "process_batch_tokens": 0,
    "process_draft_decode_tokens": 0,
    "draft_seed_decode_us": 0,
    "draft_step_decode_us": 0,
    "draft_sampler_us": 0,
    "timing_scope": "CPU wall; begin_us is outside round_us; batched decode/process spans are shared; no CUDA events",
    "begin_us": 0,
    "round_start_us": 100,
    "round_end_us": 200,
    "round_us": 100,
    "draft_us": 0,
    "checkpoint_us": 0,
    "target_decode_sync_us": 0,
    "process_us": 0,
    "check_us": 0,
    "kv_repair_us": 0,
    "accept_hook_us": 0,
    "residual_us": 100,
    "n_proposed": 0,
    "n_accepted": 0,
    "n_emitted": 0,
    "proposed_token_ids": [],
    "emitted_token_ids": [],
    "spans_us": {
        "begin": [0, 0],
        "draft": [0, 0],
        "checkpoint": [0, 0],
        "target_decode_sync": [0, 0],
        "process": [0, 0],
        "check": [0, 0],
        "kv_repair": [0, 0],
        "accept_hook": [0, 0],
    },
}


def row(index, *, proposed=(), accepted=0, emitted=(), status="complete", replay=False):
    result = copy.deepcopy(BASE)
    result.update(
        round_index=index,
        round_start_us=100 + index * 100,
        round_end_us=200 + index * 100,
        status=status,
        replay=replay,
        proposed_token_ids=list(proposed),
        n_proposed=len(proposed),
        n_accepted=accepted,
        emitted_token_ids=list(emitted),
        n_emitted=len(emitted),
    )
    return result


def with_spans(result, **spans):
    """Set producer span shape and recompute the producer's literal counters."""
    named_total = 0
    for name, interval in spans.items():
        result["spans_us"][name] = list(interval)
        duration = interval[1] - interval[0]
        field = {
            "begin": "begin_us",
            "draft": "draft_us",
            "checkpoint": "checkpoint_us",
            "target_decode_sync": "target_decode_sync_us",
            "process": "process_us",
            "check": "check_us",
            "kv_repair": "kv_repair_us",
            "accept_hook": "accept_hook_us",
        }[name]
        result[field] = duration
        if name != "begin":
            named_total += duration
    result["residual_us"] = max(0, result["round_us"] - named_total)
    return result


def metadata(output, **overrides):
    values = {
        "session_id": "synthetic-session",
        "prompt_token_ids": [101, 102],
        "generated_token_ids": list(output),
        "generation_cap": 16,
        "eos_token_ids": [2],
        "sampling_mode": "greedy",
        "ancestry": {"fixture": "independently-authored native producer shape"},
    }
    values.update(overrides)
    return values


class NativeRoundTraceIndependentTests(unittest.TestCase):
    def test_full_accept_bonus_and_zero_accept_correction_are_distinct(self):
        records = [
            row(0, proposed=[11, 12], accepted=2, emitted=[11, 12, 13]),
            row(1, proposed=[21, 22], accepted=0, emitted=[23]),
        ]
        report = adapt_session(records, metadata([11, 12, 13, 23]), {"proposed": 4, "accepted": 2, "rounds": 2})
        self.assertEqual(report["counts"]["bonus"], 1)
        self.assertEqual(report["counts"]["correction"], 1)
        self.assertEqual(report["counts"]["accepted_emitted"], 2)
        self.assertEqual(report["counts"]["emitted"], 4)
        self.assertEqual(report["aggregate_reconciliation"]["accepted"]["difference"], 0)
        self.assertEqual(report["rounds"][1]["root_generated_position"], 3)
        self.assertNotEqual(report["rounds"][0]["root_prefix_sha256"], report["rounds"][1]["root_prefix_sha256"])

    def test_no_proposal_is_one_target_token(self):
        record = row(0, status="no_proposal", emitted=[77])
        report = adapt_session([record], metadata([77]))
        self.assertEqual(report["counts"]["target_no_proposal"], 1)
        self.assertEqual(report["counts"]["proposed_quality"], 0)
        self.assertEqual(report["counts"]["emitted"], 1)

    def test_checkpoint_attempt_and_replay_emission_roles_remain_separate(self):
        checkpoint = row(0, proposed=[31, 32], accepted=1, status="checkpoint_replay")
        replay = row(1, proposed=[31], accepted=0, emitted=[31], replay=True)
        report = adapt_session([checkpoint, replay], metadata([31]))
        self.assertEqual(report["counts"]["checkpoint_attempts"], 1)
        self.assertEqual(report["counts"]["proposed_all"], 3)
        self.assertEqual(report["counts"]["proposed_quality"], 1)
        self.assertEqual(report["counts"]["unresolved_role_emitted"], 1)
        self.assertIn("replay_emission_roles", report["unresolved_domains"])
        self.assertEqual(report["counts"]["accepted_emitted"], 0)

    def test_accepted_eos_truncation_and_cap_termination(self):
        record = row(0, proposed=[8, 2, 9], accepted=2, emitted=[8, 2])
        report = adapt_session([record], metadata([8, 2], generation_cap=2))
        self.assertEqual(report["counts"]["accepted_emitted"], 2)
        self.assertEqual(report["counts"]["accepted_not_emitted"], 0)
        self.assertEqual(report["counts"]["eos_emitted"], 1)
        self.assertEqual(report["counts"]["cap_terminated"], 1)
        self.assertEqual(report["rounds"][0]["terminal_reasons"], ["eos", "cap_reached"])

    def test_native_leading_seed_is_attributed_and_hashed_before_round_zero(self):
        record = row(0, proposed=[5], accepted=1, emitted=[5, 6])
        report = adapt_session([record], metadata([4, 5, 6]))
        self.assertEqual(report["leading_untraced_emitted"], 1)
        self.assertEqual(report["rounds"][0]["root_generated_position"], 1)
        self.assertEqual(report["rounds"][0]["remaining_cap"], 15)
        self.assertIn("leading_seed_cost", report["unresolved_domains"])

    def test_native_context_checks_response_position_and_remaining_cap(self):
        record = row(0, proposed=[5], accepted=1, emitted=[5, 6])
        record["session_context_v1"] = {
            "generated_before": 0,
            "remaining_before": 16,
            "native_stop_type": "none",
        }
        report = adapt_session([record], metadata([5, 6]))
        self.assertEqual(report["rounds"][0]["native_context"], record["session_context_v1"])

        bad = copy.deepcopy(record)
        bad["session_context_v1"]["remaining_before"] = 15
        with self.assertRaisesRegex(ValueError, "remaining cap mismatch"):
            adapt_session([bad], metadata([5, 6]))

        stopped = copy.deepcopy(record)
        stopped["session_context_v1"]["native_stop_type"] = "word"
        stopped_report = adapt_session([stopped], metadata([5, 6]))
        self.assertEqual(stopped_report["rounds"][0]["terminal_reasons"], ["native_word"])

    def test_timing_uses_union_and_keeps_begin_outside_round(self):
        record = with_spans(
            row(0, proposed=[1], accepted=1, emitted=[1, 2]),
            begin=(80, 90),
            draft=(110, 150),
            target_decode_sync=(130, 170),
            process=(160, 180),
        )
        report = adapt_session([record], metadata([1, 2]))
        round_report = report["rounds"][0]
        self.assertEqual(round_report["stage_union_us"], 70)
        self.assertEqual(round_report["stage_overlap_us"], 30)
        self.assertEqual(round_report["other_inside_round_us"], 30)
        self.assertEqual(report["costs"]["begin_union_us"], 10)
        self.assertEqual(report["costs"]["round_union_us"], 100)
        self.assertEqual(report["costs"]["gpu_us"], None)

    def test_missing_provenance_cap_and_native_aggregates_stay_unresolved(self):
        record = row(0, proposed=[9], accepted=1, emitted=[9, 10])
        report = adapt_session(
            [record],
            metadata([9, 10], prompt_token_ids=None, generation_cap=None, ancestry=None),
        )
        self.assertIsNone(report["rounds"][0]["root_prefix_sha256"])
        self.assertEqual(report["aggregate_reconciliation"], {})
        for name in (
            "committed_root_identity",
            "remaining_cap",
            "source_model_device_precision_ancestry",
            "native_aggregate_join",
        ):
            self.assertIn(name, report["unresolved_domains"])

    def test_rejects_bad_ids_mismatched_aggregates_and_cross_attempt_overlap(self):
        invalid = row(0, proposed=[3], accepted=1, emitted=[4, 5])
        with self.assertRaisesRegex(ValueError, "accepted prefix"):
            adapt_session([invalid], metadata([4, 5]))

        valid = row(0, proposed=[3], accepted=1, emitted=[3, 4])
        with self.assertRaisesRegex(ValueError, "native aggregate mismatch"):
            adapt_session([valid], metadata([3, 4]), {"accepted": 0})

        first = row(0, proposed=[3], accepted=1, emitted=[3, 4])
        second = row(1, proposed=[5], accepted=1, emitted=[5, 6])
        second.update(round_start_us=190, round_end_us=290, round_us=100)
        with self.assertRaisesRegex(ValueError, "overlapping attempts"):
            adapt_session([first, second], metadata([3, 4, 5, 6]))

    def test_rejects_inconsistent_duration_span_or_response_history(self):
        bad_duration = row(0, proposed=[1], accepted=1, emitted=[1, 2])
        bad_duration["draft_us"] = 1
        with self.assertRaisesRegex(ValueError, "duration/span mismatch"):
            adapt_session([bad_duration], metadata([1, 2]))

        valid = row(0, proposed=[1], accepted=1, emitted=[1, 2])
        with self.assertRaisesRegex(ValueError, "trace/response output mismatch"):
            adapt_session([valid], metadata([1, 99]))


if __name__ == "__main__":
    unittest.main()
