"""Counterfactual arithmetic and fail-closed schema checks, with no GPU work."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_w1ax_break_even.py"
spec = importlib.util.spec_from_file_location("analyze_w1ax_break_even", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def trace():
    spans = {s: [0, 0] for s in analysis.TRACE_SPAN_STAGES}
    spans.update(draft=[100, 140], target_decode_sync=[140, 180], check=[180, 190], begin=[90, 95])
    row = {"schema": analysis.TRACE_SCHEMA, "clock": "ggml_time_us_cpu_wall", "status": "complete",
           "replay": False, "task_id": 3, "parent_task_id": -1, "round_index": 0,
           "round_start_us": 100, "round_end_us": 200, "round_us": 100,
           "spans_us": spans, "n_accepted": 1, "n_proposed": 2, "n_emitted": 2,
           "proposed_token_ids": [10, 11], "emitted_token_ids": [10, 20]}
    retime(row)
    return row


def retime(row):
    for stage, interval in row["spans_us"].items():
        row[f"{stage}_us"] = interval[1] - interval[0]
    row["residual_us"] = max(0, row["round_us"] - sum(row[f"{s}_us"] for s in analysis.TRACE_SPAN_STAGES))


class CounterfactualTests(unittest.TestCase):
    def test_counts_rates_and_raw_threshold_above_one(self):
        report = analysis.summarize_counterfactual([trace()], {"fp16": 100000})
        self.assertEqual(report["counts"]["B_non_draft_emissions"], 1)
        self.assertEqual(report["timing_us"]["C_minus_D"], 60)
        scenario = report["scenarios"]["exclusive_draft_removed"]
        self.assertAlmostEqual(scenario["frozen_emission_rate_tokens_per_s"], 2 / .000060)
        self.assertAlmostEqual(scenario["maximum_fixed_trajectory_rate_tokens_per_s"], 3 / .000060)
        threshold = scenario["acceptance_thresholds"]["fp16"]
        self.assertEqual(threshold["accepted_drafts_required_raw"], 5)
        self.assertEqual(threshold["accepted_per_proposed_required_raw"], 2.5)
        self.assertTrue(threshold["exceeds_fixed_proposal_count"])

    def test_overlapping_stages_remove_exclusive_union_only(self):
        row = trace()
        row["spans_us"].update(draft=[100, 170], process=[120, 160], target_decode_sync=[130, 180])
        retime(row)
        report = analysis.summarize_counterfactual([row], {"fp16": 1})
        self.assertEqual(report["timing_us"]["draft_overlap_other_union"], 50)
        self.assertEqual(report["timing_us"]["D_removable_exclusive_draft"], 20)
        self.assertEqual(report["timing_us"]["unclamped_residual_sum"], -70)
        self.assertEqual(report["timing_us"]["C_minus_D"], 80)
        self.assertLess(report["scenarios"]["observed_round_cost"]["acceptance_thresholds"]["fp16"]["accepted_per_proposed_required_raw"], 0)

    def test_stop_truncation_counts_only_emitted_accepted_prefix(self):
        row = trace()
        row.update(n_accepted=2, n_emitted=1, emitted_token_ids=[10])
        report = analysis.summarize_counterfactual([row], {"fp16": 1})
        self.assertEqual(report["counts"]["A_accepted_drafts_actually_emitted"], 1)
        self.assertEqual(report["counts"]["accepted_drafts_not_emitted_at_stop"], 1)
        self.assertEqual(report["counts"]["B_non_draft_emissions"], 0)

    def test_invalid_emission_prefix_or_count_rejected(self):
        for changes in ({"emitted_token_ids": [99, 20]}, {"n_accepted": 0}, {"n_emitted": 1}, {"n_accepted": True}):
            with self.subTest(changes=changes):
                row = trace()
                row.update(changes)
                with self.assertRaises(ValueError):
                    analysis.summarize_counterfactual([row], {"fp16": 1})

    def test_truncated_nonterminal_round_rejected(self):
        row = trace()
        row.update(n_accepted=2, n_emitted=1, emitted_token_ids=[10])
        with self.assertRaisesRegex(ValueError, "before request ended"):
            analysis.summarize_counterfactual([row, trace()], {"fp16": 1})

    def test_missing_outside_reversed_spans_and_scalar_mismatch_rejected(self):
        for mutation in ("missing", "outside", "reverse", "scalar", "round", "residual"):
            with self.subTest(mutation=mutation):
                row = trace()
                if mutation == "missing":
                    del row["spans_us"]["process"]
                elif mutation == "outside":
                    row["spans_us"]["draft"] = [90, 140]
                    retime(row)
                elif mutation == "reverse":
                    row["spans_us"]["draft"] = [140, 100]
                elif mutation == "scalar":
                    row["draft_us"] += 1
                elif mutation == "round":
                    row["round_us"] += 1
                else:
                    row["residual_us"] += 1
                with self.assertRaises(ValueError):
                    analysis.summarize_counterfactual([row], {"fp16": 1})

    def test_checkpoint_cost_included_but_quality_excluded(self):
        checkpoint = trace()
        checkpoint.update(status="checkpoint_replay", n_emitted=0, emitted_token_ids=[])
        report = analysis.summarize_counterfactual([checkpoint, trace()], {"fp16": 1})
        self.assertEqual(report["counts"]["checkpoint_replay_cost_rows"], 1)
        self.assertEqual(report["counts"]["N_quality_rounds"], 1)
        self.assertEqual(report["counts"]["P_proposed_ids_quality_rounds"], 2)
        self.assertEqual(report["timing_us"]["C_all_rounds"], 200)
        self.assertEqual(report["timing_us"]["D_removable_exclusive_draft"], 80)
        replay = trace()
        replay["replay"] = True
        with self.assertRaisesRegex(ValueError, "replay quality emission semantics unsupported"):
            analysis.summarize_counterfactual([replay], {"fp16": 1})

    def test_zero_remainder_rejected(self):
        row = trace()
        row["spans_us"] = {s: [0, 0] for s in (*analysis.TRACE_SPAN_STAGES, "begin")}
        row["spans_us"]["draft"] = [100, 200]
        retime(row)
        with self.assertRaisesRegex(ValueError, "nonpositive residual"):
            analysis.summarize_counterfactual([row], {"fp16": 1})

    def test_no_proposals_ratio_is_null(self):
        row = trace()
        row.update(status="no_proposal", n_proposed=0, proposed_token_ids=[], n_accepted=0, n_emitted=1, emitted_token_ids=[20])
        report = analysis.summarize_counterfactual([row], {"fp16": 1})
        self.assertIsNone(report["scenarios"]["observed_round_cost"]["acceptance_thresholds"]["fp16"]["accepted_per_proposed_required_raw"])


def write_runs(root):
    variants = ["ordinary_eagle", "draft_q4_0", "draft_w1a1"]
    manifest = {"prompt_ids": ["p0"], "request_options": {"max_tokens": 3}, "precision": {"target": "fp16"},
                "policy": {"warmup_requests": 0}, "files": {name: {"sha256": "a" * 64} for name in ("target", "ordinary_draft", "draft_q4_0", "draft_w1a1", "prompts")},
                "commands": {v: ["server", "--parallel", "1"] for v in variants}}
    records = [{"repetition": 0, "variant": v, "prompt_id": "p0", "request_id": v + "/p0", "server_request_index": 0,
                "generated_token_ids": [99, 10, 20], "completion_tokens": 3, "server_predicted_ms": ms}
               for v, ms in zip(variants, (10, 5, 20))]
    for role in ("trace", "primary"):
        path = root / role
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({**manifest, "round_trace_enabled": role == "trace", "llama_checkout_commit": role}))
        (path / "records.json").write_text(json.dumps(records))
        if role == "trace":
            for v in variants:
                parent = path / "rep-00" / v
                parent.mkdir(parents=True)
                (parent / "round-trace.jsonl").write_text(json.dumps(trace()) + "\n")
    return root / "trace", root / "primary"


class RunTests(unittest.TestCase):
    def test_cli_pairs_sources_and_preserves_different_runtime_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir, primary_dir = write_runs(Path(tmp))
            output = Path(tmp) / "report.json"
            subprocess.run([sys.executable, str(SCRIPT), str(trace_dir), str(primary_dir), "--output", str(output)], check=True, capture_output=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["pairing"]["paired_requests"], 3)
            self.assertEqual(len(report["trace_file_sha256"]), 3)
            measured = report["measured_primary"]["variants"]
            self.assertEqual(measured["ordinary_eagle"]["decode_tokens_per_s"], 300)
            self.assertEqual(measured["draft_w1a1"]["decode_rate_ratios"]["q4_0_eagle"], .25)
            self.assertEqual(report["trace_counterfactuals"]["draft_w1a1"]["untraced_leading_tokens"], 1)
            self.assertEqual(report["trace_counterfactuals"]["draft_w1a1"]["counts"]["E_emitted_ids"], 2)
            self.assertEqual([s["provenance"]["llama_checkout_commit"] for s in report["sources"]], ["trace", "primary"])

    def test_primary_pools_tokens_over_time_not_mean_rates(self):
        records = []
        for variant in analysis.ANCHORS.values():
            records.extend([{"variant": variant, "completion_tokens": 1, "generated_token_ids": [1], "server_predicted_ms": 1},
                            {"variant": variant, "completion_tokens": 3, "generated_token_ids": [1, 2, 3], "server_predicted_ms": 9}])
        rates = analysis.primary_rates(records)
        self.assertEqual(rates["ordinary_eagle"]["decode_tokens_per_s"], 400)

    def test_mismatched_pairing_inputs_and_instrumented_primary_rejected(self):
        for mutation in ("pairing", "hash", "instrumentation", "timing", "token_count"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                trace_dir, primary_dir = write_runs(Path(tmp))
                name = "records.json" if mutation in ("pairing", "timing", "token_count") else "manifest.json"
                path = primary_dir / name
                data = json.loads(path.read_text())
                if mutation == "pairing":
                    data.pop()
                elif mutation == "hash":
                    data["files"]["target"]["sha256"] = "b" * 64
                elif mutation == "instrumentation":
                    data["round_trace_enabled"] = True
                elif mutation == "token_count":
                    data[0]["completion_tokens"] += 1
                else:
                    data[0]["server_predicted_ms"] = 0
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    analysis.analyze_runs(trace_dir, primary_dir)

    def test_mapping_rejects_shifted_suffix_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir, primary_dir = write_runs(Path(tmp))
            path = trace_dir / "rep-00" / "draft_w1a1" / "round-trace.jsonl"
            row = trace()
            row["emitted_token_ids"] = [10, 88]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "do not match the full record"):
                analysis.analyze_runs(trace_dir, primary_dir)


if __name__ == "__main__":
    unittest.main()
