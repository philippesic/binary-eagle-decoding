"""CPU-only round partition, pairing and frozen-trajectory arithmetic gates."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_binary_rescue_rounds as analysis


def retime(row):
    for stage, (left, right) in row["spans_us"].items():
        row[f"{stage}_us"] = right - left
    row["residual_us"] = max(
        0, row["round_us"] - sum(row[f"{s}_us"] for s in analysis.TRACE_SPAN_STAGES)
    )


def round_row(**changes):
    spans = {stage: [0, 0] for stage in analysis.TRACE_SPAN_STAGES}
    spans.update(draft=[100, 140], target_decode_sync=[140, 180], check=[180, 190], begin=[90, 95])
    row = {
        "schema": "w1ax_eagle_round_v1",
        "clock": "ggml_time_us_cpu_wall",
        "status": "complete",
        "replay": False,
        "task_id": 3,
        "parent_task_id": -1,
        "round_index": 0,
        "round_start_us": 100,
        "round_end_us": 200,
        "round_us": 100,
        "spans_us": spans,
        "n_proposed": 2,
        "n_accepted": 1,
        "n_emitted": 2,
        "proposed_token_ids": [10, 11],
        "emitted_token_ids": [10, 20],
    }
    row.update(changes)
    retime(row)
    return row


def record(variant="D", repetition=0, ids=None):
    ids = [99, 10, 20] if ids is None else ids
    return {
        "variant": variant,
        "prompt_id": "p",
        "repetition": repetition,
        "warmup": False,
        "generated_token_ids": ids,
        "completion_tokens": len(ids),
        "directory": "/unused",
        "request_wall_s": 0.01 if variant == "q4_0" else 0.02,
        "server_predicted_ms": 5 if variant == "q4_0" else 10,
        "server_prompt_ms": 1,
        "speculative": {"accepted": 1, "proposed": 2, "rounds": 1},
        "request_digest": {
            "task_id": 3,
            "proposal": "1234567890abcdef",
            "output": analysis.output_digest(ids),
            "rounds": 1,
            "no_proposal": 1,
            "output_tokens": len(ids),
        },
    }


def manifest(mode, repetitions=1):
    return {
        "schema": "binary_rescue_benchmark_v1",
        "status": "complete",
        "mode": mode,
        "workload": "primary-development",
        "q4_variant": "q4_0",
        "prompt_sha256": "a" * 64,
        "policy": {"draft_length": 5, "p_min": 0, "tokens": 128, "context": 2048},
        "hashes": {
            "binary": "b" * 64,
            "target": "c" * 64,
            "drafts": {"D": "d" * 64, "q4_0": "e" * 64},
        },
        "records": [
            record(variant, rep) for variant in ("q4_0", "D") for rep in range(repetitions)
        ],
    }


def stage_call(start=105, index=0):
    boundaries = [
        ("seed_prepare", -1, 0, 5),
        ("seed_decode_call", 0, 5, 15),
        ("seed_sync_retrieve", 0, 15, 20),
        ("output_sampling", 0, 20, 30),
    ]
    spans = [
        {
            "stage": stage,
            "depth": depth,
            "start_us": start + a,
            "end_us": start + b,
            "duration_us": b - a,
        }
        for stage, depth, a, b in boundaries
    ]
    return {
        "schema": "eagle_draft_stage_v1",
        "clock": "ggml_time_us_cpu_wall",
        "scope": "draft_invocation_nested_within_round",
        "explicit_sync_before_sampling": True,
        "serialization_included": False,
        "call_index": index,
        "start_us": start,
        "end_us": start + 30,
        "total_us": 30,
        "status": "complete",
        "seed_rows": 1,
        "recurrent_rows": 0,
        "decode_calls": 1,
        "sampling_calls": 1,
        "retrieval_calls": 1,
        "sequences": [{"seq_id": 0, "pos0": 4, "seed_token_id": 99}],
        "spans": spans,
        "stage_totals_us": {s["stage"]: s["duration_us"] for s in spans},
        "partition_us": 30,
        "unassigned_us": 0,
    }


class DraftStageTests(unittest.TestCase):
    def test_nested_partition_preserves_outer_round_cost(self):
        rows = [round_row()]
        result = analysis.inspect_draft_stages(rows, [stage_call(), stage_call(5, 1)])
        self.assertEqual(result["matched_calls"], 1)
        self.assertEqual(result["nested_calls_total_us"], 30)
        self.assertEqual(result["outer_draft_remainder_us"], 10)
        self.assertEqual(result["stage_partition_us"]["output_sampling"], 10)
        self.assertEqual(result["counts"]["retrieval_calls"], 1)
        self.assertEqual(analysis.inspect_request(record(), rows)["timing_us"]["C_all_rounds"], 100)

    def test_call_bounds_internal_overlaps_and_count_errors_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses outer"):
            analysis.inspect_draft_stages([round_row()], [stage_call(90)])
        broken = stage_call()
        broken["spans"][1]["start_us"] -= 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            analysis.validate_stage_call(broken)
        broken = stage_call()
        broken["sampling_calls"] = 2
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            analysis.validate_stage_call(broken)

    def test_optional_stage_mapping_excludes_warmups_and_blocks_error_proxy(self):
        result = analysis.analyze(
            manifest("instrumented"),
            manifest("timed", 5),
            lambda _: [round_row()],
            load_stages=lambda _: [stage_call(), stage_call(5, 1)],
        )
        for item in result["draft_stage_files_coverage"]:
            self.assertEqual(item["file_calls"], 2)
            self.assertEqual(item["matched_measured_calls"], 1)
            self.assertEqual(item["unmatched_calls"], 1)
        d = result["variants"]["D"]
        self.assertFalse(d["draft_stages"]["additive_with_outer_round"])
        self.assertEqual(d["timing_us"]["C_all_rounds"], 100)
        self.assertEqual(d["paired_timed_candidate"]["speculative"]["complete_rounds"], 5)
        failed = stage_call()
        failed["status"] = "seed_decode_error"
        result = analysis.analyze(
            manifest("instrumented"),
            manifest("timed", 5),
            lambda _: [round_row()],
            load_stages=lambda _: [failed],
        )
        self.assertIsNone(result["variants"]["D"]["conditional_fixed_trajectory_proxy"])


class RoundTests(unittest.TestCase):
    def test_disjoint_partition_and_exclusive_removal(self):
        row = round_row()
        row["spans_us"].update(draft=[100, 170], target_decode_sync=[130, 180], process=[120, 160])
        retime(row)
        result = analysis.inspect_request(record(), [row])
        partition = result["disjoint_partition_us"]
        self.assertEqual(partition["draft"], 20)
        self.assertEqual(partition["target_verify_sync"], 10)
        self.assertEqual(partition["feature_process_cache_catchup"], 0)
        self.assertEqual(partition["shared_stage_overlap"], 50)
        self.assertEqual(partition["unattributed"], 10)
        self.assertEqual(sum(partition.values()), 100)
        self.assertEqual(result["timing_us"]["C_minus_D"], 80)
        self.assertEqual(result["untraced_leading_seed_tokens"], 1)
        self.assertEqual(result["begin_outside_round_us"], 5)

    def test_no_proposal_censoring_and_terminal_counts(self):
        row = round_row(
            n_proposed=0,
            proposed_token_ids=[],
            n_accepted=0,
            n_emitted=1,
            emitted_token_ids=[20],
            status="no_proposal",
        )
        result = analysis.inspect_request(record(ids=[99, 20]), [row])
        self.assertEqual(result["quality"]["no_proposal_rounds"], 1)
        self.assertIsNone(result["quality"]["depth"]["2"]["conditional_acceptance"])
        row = round_row(n_accepted=2, n_emitted=1, emitted_token_ids=[10])
        result = analysis.inspect_request(record(ids=[99, 10]), [row])
        self.assertEqual(result["counts"]["A_accepted_drafts_actually_emitted"], 1)
        self.assertEqual(result["counts"]["accepted_drafts_not_emitted_at_stop"], 1)
        self.assertEqual(result["counts"]["B_non_draft_emissions"], 0)

    def test_replay_cost_retained_quality_excluded(self):
        replay = round_row(status="checkpoint_replay", n_emitted=0, emitted_token_ids=[])
        row = round_row(round_index=1, round_start_us=200, round_end_us=300)
        row["spans_us"] = {
            stage: [left + 100, right + 100] if right else [0, 0]
            for stage, (left, right) in row["spans_us"].items()
        }
        retime(row)
        result = analysis.inspect_request(record(), [replay, row])
        self.assertEqual(result["timing_us"]["C_all_rounds"], 200)
        self.assertEqual(result["counts"]["checkpoint_replay_cost_rows"], 1)
        self.assertEqual(result["quality"]["rounds"], 1)
        self.assertEqual(result["quality"]["proposed"], 2)

    def test_invalid_overlap_out_of_round_misalignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "windows overlap"):
            analysis.inspect_request(record(), [round_row(), round_row(round_index=1)])
        row = round_row()
        row["spans_us"]["draft"] = [90, 140]
        retime(row)
        with self.assertRaises(ValueError):
            analysis.inspect_request(record(), [row])
        with self.assertRaisesRegex(ValueError, "emitted IDs differ"):
            analysis.inspect_request(record(ids=[99, 10, 88]), [round_row()])


class AnalysisTests(unittest.TestCase):
    def test_new_emission_proxy_uses_round_cost_not_server_time(self):
        result = analysis.analyze(
            manifest("instrumented"), manifest("timed", 5), lambda _: [round_row()]
        )
        d = result["variants"]["D"]
        self.assertTrue(d["timed_behavior_gate"]["passed"])
        self.assertEqual(d["emitted_per_quality_round"], 2)
        self.assertEqual(d["paired_timed_q4"]["server_decode_tok_s"], 600)
        proxy = d["conditional_fixed_trajectory_proxy"]
        self.assertEqual(proxy["fixed_emitted_tokens"], 2)
        self.assertEqual(proxy["remaining_complete_round_cpu_us"], 60)
        self.assertAlmostEqual(proxy["conditional_draft_removed_round_tok_s"], 2e6 / 60)
        self.assertEqual(d["paired_timed_candidate"]["server_decode_s"], 0.05)
        self.assertNotIn("maximum_fixed_trajectory_rate_tokens_per_s", json.dumps(result))

    def test_changed_proposals_with_same_outputs_withhold_proxy(self):
        timed = manifest("timed", 5)
        timed["records"][-1]["request_digest"]["proposal"] = "0000000000000000"
        result = analysis.analyze(manifest("instrumented"), timed, lambda _: [round_row()])
        d = result["variants"]["D"]
        self.assertFalse(d["timed_behavior_gate"]["passed"])
        self.assertIsNone(d["conditional_fixed_trajectory_proxy"])
        self.assertTrue(result["variants"]["q4_0"]["timed_behavior_gate"]["passed"])

    def test_identical_malformed_digests_do_not_release_proxy(self):
        traced, timed = manifest("instrumented"), manifest("timed", 5)
        for run in (traced, timed):
            for row in run["records"]:
                row["request_digest"]["proposal"] = "not-hex-digest!!"
        result = analysis.analyze(traced, timed, lambda _: [round_row()])
        self.assertIsNone(result["variants"]["D"]["conditional_fixed_trajectory_proxy"])

    def test_quality_gate_missing_digest_blocks_transfer(self):
        quality = manifest("quality")
        quality["records"][1].pop("request_digest")
        result = analysis.analyze(
            manifest("instrumented"), manifest("timed", 5), lambda _: [round_row()], quality
        )
        self.assertIsNone(result["variants"]["D"]["conditional_fixed_trajectory_proxy"])

    def test_hash_pairing_and_corrupt_aggregate_rejected(self):
        timed = manifest("timed", 5)
        timed["hashes"]["binary"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "binary hash"):
            analysis.analyze(manifest("instrumented"), timed, lambda _: [round_row()])
        timed = manifest("timed", 5)
        timed["records"].pop(0)
        with self.assertRaisesRegex(ValueError, "pairing set"):
            analysis.analyze(manifest("instrumented"), timed, lambda _: [round_row()])
        supplied = {
            "mode": "timed",
            "policy": timed["policy"],
            "variants": {"q4_0": {"requests": 123}},
        }
        with self.assertRaises(ValueError):
            analysis.analyze(
                manifest("instrumented"),
                manifest("timed", 5),
                lambda _: [round_row()],
                timed_analysis=supplied,
            )

    def test_cli_relocated_round_paths_preserves_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "instrumented.json").write_text(json.dumps(manifest("instrumented")))
            (root / "timed.json").write_text(json.dumps(manifest("timed", 5)))
            (root / "rounds.json").write_text(json.dumps([round_row()]))
            overrides = [
                {"variant": v, "prompt_id": "p", "repetition": 0, "path": "rounds.json"}
                for v in ("q4_0", "D")
            ]
            (root / "paths.json").write_text(json.dumps(overrides))
            command = [
                sys.executable,
                str(Path(analysis.__file__)),
                "--instrumented",
                str(root / "instrumented.json"),
                "--timed",
                str(root / "timed.json"),
                "--round-paths",
                str(root / "paths.json"),
                "--output",
                str(root / "result.json"),
            ]
            subprocess.run(command, check=True, capture_output=True)
            output = json.loads((root / "result.json").read_text())
            self.assertEqual(len(output["round_files"]), 2)
            self.assertEqual(
                output["round_files"][0]["sha256"], analysis.digest_file(root / "rounds.json")
            )


if __name__ == "__main__":
    unittest.main()
