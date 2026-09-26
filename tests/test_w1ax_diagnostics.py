"""Offline W1Ax diagnostics checks using synthetic trace and replay rows."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_w1ax_diagnostics.py"
spec = importlib.util.spec_from_file_location("analyze_w1ax_diagnostics", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def trace(proposed, accepted, round_us=100, status="complete"):
    return {
        "schema": "w1ax_eagle_round_v1",
        "status": status,
        "n_proposed": len(proposed),
        "n_accepted": accepted,
        "n_emitted": accepted + 1,
        "proposed_token_ids": proposed,
        "round_us": round_us,
        "begin_us": 8,
        "draft_us": 20,
        "checkpoint_us": 4,
        "target_decode_sync_us": 30,
        "process_us": 15,
        "check_us": 7,
        "kv_repair_us": 2,
        "accept_hook_us": 3,
        "residual_us": 19,
    }


class W1AxDiagnosticsTests(unittest.TestCase):
    def test_quality_excludes_checkpoint_replay_and_infers_depth(self):
        rows = [
            trace([1, 2, 3], 2, 100),
            trace([4], 0, 80),
            trace([9, 8], 2, 90, "checkpoint_replay"),
        ]
        result = analysis.summarize_variant(rows)
        self.assertEqual(result["trace_rows"], 3)
        self.assertEqual(result["quality_rounds"], 2)
        self.assertEqual(result["excluded_checkpoint_replay_rows"], 1)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["proposed"], 4)
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["accepted_per_proposed"], 0.5)
        self.assertEqual(result["accepted_per_round"], 1)
        depth = result["conditional_acceptance_by_depth"]
        self.assertEqual(depth["1"]["conditional_acceptance"], 0.5)
        self.assertEqual(depth["2"]["conditional_acceptance"], 1.0)
        self.assertEqual(result["actual_proposal_lengths"]["median"], 2.0)

    def test_cpu_wall_stage_distribution_and_remainder(self):
        result = analysis.summarize_variant([trace([1], 1, 100), trace([2], 0, 200)])
        wall = result["cpu_wall_us"]
        self.assertEqual(wall["round"]["median"], 150)
        self.assertEqual(wall["stages"]["draft_us"]["median"], 20)
        self.assertEqual(wall["stages"]["residual_us"]["median"], 19)
        self.assertIn("No CUDA event data", wall["note"])

    def test_trace_span_accounting_flags_overlap_bounds_and_clamped_residual(self):
        row = trace([1, 2], 1, 100)
        row.update({
            "task_id": 19, "round_index": 2,
            "round_start_us": 100, "round_end_us": 200,
            "residual_us": 0,
            "spans_us": {
                "begin": [80, 95],
                "draft": [90, 150],
                "checkpoint": [120, 160],
                "process": [125, 140],
                "target_decode_sync": [170, 210],
                "accept_hook": [220, 225],
                "check": [0, 0], "kv_repair": [0, 0],
            },
        })
        result = analysis.summarize_variant([row])
        accounting = result["trace_span_accounting"]
        detail = accounting["per_round"][0]
        self.assertEqual(detail["attributed_union_us"], 90)
        self.assertEqual(detail["overlap_us"], 45)
        self.assertEqual(detail["outside_round_us"], 25)
        self.assertEqual(detail["unclamped_residual_us"], -35)
        self.assertEqual(detail["union_unattributed_us"], 10)
        self.assertEqual(detail["nested_stage_pairs"], [["draft", "process"], ["checkpoint", "process"]])
        self.assertIn("out_of_bounds_spans", detail["flags"])
        self.assertIn("runtime_residual_was_clamped", detail["flags"])
        self.assertEqual(detail["runtime_residual_us"], 0)
        self.assertEqual(accounting["distributions_us"]["begin_outside_round"]["median"], 15)
        self.assertIn("warmup task groups", result["quality_scope"])

    def test_identical_input_replay_pairs_per_sample_by_layer_shape(self):
        rows = [
            {"record_type": "operator_replay", "capture": "capture-a.bin", "sequence": 7, "K": 4096, "M": 8, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 16, "samples_us": [10, 12, 14]},
            {"record_type": "operator_replay", "capture": "capture-a.bin", "sequence": 7, "K": 4096, "M": 8, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 8, "samples_us": [5, 6, 7]},
            {"record_type": "operator_replay", "capture": "capture-a.bin", "sequence": 7, "K": 4096, "M": 8, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 4, "samples_us": [2, 3, 4]},
            {"record_type": "operator_replay", "capture": "capture-b.bin", "sequence": 7, "K": 4096, "M": 8, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 16, "samples_us": [20, 22]},
            {"record_type": "operator_replay", "capture": "capture-b.bin", "sequence": 7, "K": 4096, "M": 8, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 8, "samples_us": [10, 11]},
            {"record_type": "head_comparison", "capture": "capture-a.bin", "sequence": 7, "token": 0, "candidate_bits": 8, "top1_agree": True, "topk_set_overlap": {"1": 1, "5": 4, "10": 8}, "reference_top1_margin": 0.4, "candidate_top1_margin": 0.3},
            {"record_type": "head_comparison", "capture": "capture-b.bin", "sequence": 7, "token": 1, "candidate_bits": 8, "top1_agree": False, "topk_set_overlap": {"1": 0, "5": 3, "10": 7}, "reference_top1_margin": 0.2, "candidate_top1_margin": 0.1},
        ]
        result = analysis.summarize_replay(rows)
        shape = result["by_layer_shape"][0]
        self.assertEqual(result["rows"], 5)
        self.assertEqual(shape["captures"], 2)
        self.assertEqual(shape["precision"]["16"]["samples"], 5)
        self.assertEqual(shape["paired_ratios"]["8"]["paired_precision_ratio_vs_16"], 0.5)
        self.assertEqual(shape["paired_ratios"]["8"]["paired_samples"], 5)
        self.assertEqual(shape["paired_ratios"]["4"]["paired_samples"], 3)
        head = result["head_comparison"]["by_candidate_bits"]["8"]
        self.assertEqual(head["comparisons"], 2)
        self.assertEqual(head["top1_agreement_rate"], 0.5)
        self.assertEqual(head["topk_overlap_fraction"]["5"]["median"], 0.7)

    def test_anchor_operator_rows_pair_to_w1ax_by_capture(self):
        def row(record_type, capture, bits_or_format, samples):
            base = {
                "record_type": record_type, "capture": capture, "sequence": 2,
                "K": 64, "M": 4, "N": 2, "name": "q_proj", "samples_us": samples,
            }
            if record_type == "operator_replay":
                base.update(group="attention", replay_bits=bits_or_format)
            else:
                base.update(anchor_format=bits_or_format, group="attention")
            return base

        rows = [
            row("operator_replay", "a.bin", 16, [10, 12, 14]),
            row("operator_replay", "a.bin", 8, [5, 6, 7]),
            row("operator_replay", "a.bin", 4, [2, 3, 4]),
            row("anchor_operator_replay", "a.bin", "fp16", [20, 24, 28]),
            row("anchor_operator_replay", "a.bin", "q4_0", [4, 4, 4]),
            row("operator_replay", "b.bin", 16, [20, 22]),
            row("operator_replay", "b.bin", 8, [10, 11]),
            row("anchor_operator_replay", "b.bin", "fp16", [40, 44]),
            row("anchor_operator_replay", "b.bin", "q4_0", [8, 8]),
            {"record_type": "head_comparison", "capture": "a.bin", "sequence": 2,
             "candidate_bits": 4, "top1_agree": True, "topk_set_overlap": {"1": 1},
             "reference_top1_margin": 0.2, "candidate_top1_margin": 0.1},
        ]
        result = analysis.summarize_replay(rows)
        self.assertEqual(result["anchor_rows"], 4)
        anchor_shape = result["anchor_by_layer_shape"][0]
        self.assertEqual(anchor_shape["captures"], 2)
        self.assertEqual(anchor_shape["formats"]["fp16"]["median_us"], 28)
        comparisons = result["w1ax_vs_anchors"][0]["w1ax_vs_anchor"]
        self.assertEqual(comparisons["w1a8/fp16"]["paired_precision_ratio"], 0.25)
        self.assertEqual(comparisons["w1a8/fp16"]["paired_samples"], 5)
        self.assertEqual(comparisons["w1a8/q4_0"]["paired_precision_ratio"], 1.375)
        self.assertEqual(comparisons["w1a4/fp16"]["paired_samples"], 3)
        self.assertEqual(comparisons["w1a4/q4_0"]["paired_samples"], 3)
        self.assertEqual(result["head_comparison"]["comparisons"], 1)

    def test_run_reads_records_and_variant_round_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "records.json").write_text(json.dumps([{"variant": "draft_w1a8"}]))
            trace_path = run_dir / "rep-00" / "draft_w1a8" / "round-trace.jsonl"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(json.dumps(trace([1, 2], 1)) + "\n")
            result = analysis.analyze_run(run_dir)
            self.assertEqual(result["variants"]["draft_w1a8"]["accepted"], 1)
            self.assertIn("Unavailable", result["variants"]["draft_w1a8"]["request_mapping"])

    def test_replay_only_cli_needs_no_benchmark_run(self):
        replay_rows = [
            {"record_type": "operator_replay", "capture": "cap.bin", "sequence": 3, "K": 8, "M": 4, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 16, "samples_us": [8, 10]},
            {"record_type": "operator_replay", "capture": "cap.bin", "sequence": 3, "K": 8, "M": 4, "N": 2, "name": "q_proj", "group": "attention", "replay_bits": 8, "samples_us": [4, 5]},
            {"record_type": "head_comparison", "capture": "cap.bin", "sequence": 3, "token": 0, "candidate_bits": 8, "top1_agree": True, "topk_set_overlap": {"1": 1, "5": 4}, "reference_top1_margin": 0.3, "candidate_top1_margin": 0.2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "operator-records.jsonl"
            output_path = Path(tmp) / "summary.json"
            replay_path.write_text("".join(json.dumps(row) + "\n" for row in replay_rows))
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--replay-only", str(replay_path), "--output", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stdout, "")
            report = json.loads(output_path.read_text())
            self.assertEqual(report["operator_replay"]["rows"], 2)
            self.assertEqual(report["operator_replay"]["by_layer_shape"][0]["paired_ratios"]["8"]["paired_precision_ratio_vs_16"], 0.5)
            self.assertEqual(report["operator_replay"]["head_comparison"]["comparisons"], 1)

    def test_replay_requires_valid_samples(self):
        row = {"K": 1, "M": 1, "N": 1, "name": "x", "group": "g", "replay_bits": 16, "samples_us": []}
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            analysis.summarize_replay([row])

    def test_replay_rejects_ambiguous_duplicate_precision(self):
        row = {"record_type": "operator_replay", "capture": "same.bin", "sequence": 0, "K": 1, "M": 1, "N": 1, "name": "x", "group": "g", "replay_bits": 16, "samples_us": [1]}
        with self.assertRaisesRegex(ValueError, "duplicate precision row"):
            analysis.summarize_replay([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
