"""Checks for paired pooled-rate analysis without a GPU or server."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_native_benchmark.py"
spec = importlib.util.spec_from_file_location("analyze_native_benchmark", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class NativeBenchmarkAnalysisTests(unittest.TestCase):
    def records(self) -> list[dict]:
        rows = []
        for repetition in range(2):
            for prompt_id, tokens in (("prose-01", 2), ("code-01", 8)):
                for variant, wall in (
                    ("target_only", 2.0),
                    ("ordinary_eagle", 1.5),
                    ("packed_head_w1a1", 1.0),
                ):
                    rows.append(
                        {
                            "repetition": repetition,
                            "prompt_id": prompt_id,
                            "variant": variant,
                            "completion_tokens": tokens,
                            "request_wall_s": wall,
                            "server_predicted_ms": wall * 500,
                            "speculative": {
                                "proposed": None if variant == "target_only" else 4,
                                "accepted": None if variant == "target_only" else 1,
                                "rounds": None if variant == "target_only" else 2,
                            },
                        }
                    )
        return rows

    def records_with_mma(self) -> list[dict]:
        rows = self.records()
        for row in self.records():
            if row["variant"] == "packed_head_w1a1":
                mma = dict(row)
                mma["variant"] = "packed_head_w1a1_mma"
                mma["request_wall_s"] = 0.8
                mma["server_predicted_ms"] = 400.0
                rows.append(mma)
        return rows

    def test_pooled_speedup_and_paired_bootstrap(self) -> None:
        rows = self.records()
        self.assertEqual(analysis.pooled_speedup(rows, "target_only", "request"), 2.0)
        self.assertEqual(analysis.pooled_speedup(rows, "ordinary_eagle", "decode"), 1.5)
        first = analysis.paired_bootstrap(rows, 100, 42)
        self.assertEqual(first, analysis.paired_bootstrap(rows, 100, 42))
        self.assertEqual(first["target_only/request"]["median"], 2.0)
        self.assertEqual(first["ordinary_eagle/decode"]["p97_5"], 1.5)

    def test_incomplete_pairs_are_rejected(self) -> None:
        rows = self.records()[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete paired records"):
            analysis.paired_index(rows)

    def test_four_variant_pairs_and_mma_speedup(self) -> None:
        rows = self.records_with_mma()
        self.assertEqual(analysis.selected_variants(rows), analysis.ALL_VARIANTS)
        self.assertEqual(
            analysis.pooled_speedup(rows, "packed_head_w1a1", "request", analysis.MMA_VARIANT),
            1.25,
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "records.json").write_text(json.dumps(rows))
            (directory / "report.json").write_text(
                json.dumps({"status": "complete", "records": len(rows), "variants": list(analysis.ALL_VARIANTS)})
            )
            (directory / "prompts.jsonl").write_text(
                '{"id":"prose-01","category":"prose"}\n{"id":"code-01","category":"code"}\n'
            )
            result = analysis.analyze(directory, 100, 42)
            self.assertEqual(result["pooled_mma_speedup"]["packed_head_w1a1/request"], 1.25)
            self.assertEqual(
                result["paired_mma_bootstrap_95pct"]["packed_head_w1a1/request"]["median"],
                1.25,
            )
            self.assertIn(analysis.MMA_VARIANT, result["category_summary"]["code"])

        missing = rows[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete paired records"):
            analysis.paired_index(missing)
        with self.assertRaisesRegex(ValueError, "benchmark variants"):
            analysis.selected_variants(rows, list(analysis.VARIANTS))

    def test_missing_decode_metric_does_not_hide_request_interval(self) -> None:
        rows = self.records()
        rows[0]["server_predicted_ms"] = None
        intervals = analysis.paired_bootstrap(rows, 100, 42)
        self.assertIsNotNone(intervals["target_only/request"])
        self.assertIsNone(intervals["target_only/decode"])

    def test_run_artifacts_and_category_summary(self) -> None:
        rows = self.records()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "records.json").write_text(json.dumps(rows))
            (directory / "report.json").write_text(
                json.dumps({"status": "complete", "records": len(rows)})
            )
            (directory / "prompts.jsonl").write_text(
                '{"id":"prose-01","category":"prose"}\n{"id":"code-01","category":"code"}\n'
            )
            result = analysis.analyze(directory, 100, 42)
            self.assertEqual(result["prompts"], 2)
            self.assertEqual(result["repetitions"], 2)
            self.assertEqual(result["pooled_packed_speedup"]["target_only/request"], 2.0)
            self.assertEqual(
                result["category_summary"]["code"]["ordinary_eagle"]["accepted_per_round"], 0.5
            )


if __name__ == "__main__":
    unittest.main()
