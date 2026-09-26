"""CPU fixtures for strict context provenance, bins, pairs, and descriptive output checks."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_w1ax_context_matrix", ROOT / "scripts/analyze_w1ax_context_matrix.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def set_ids(row, ids):
    row["generated_token_ids"] = ids
    row["completion_tokens"] = len(ids)
    row["generated_token_ids_sha256"] = digest(json.dumps(ids, separators=(",", ":")))


class ContextMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = {cap: self.root / str(cap) for cap in (32, 128)}
        prompts_text = (ROOT / "data/w1ax-context/prompts.jsonl").read_text()
        self.prompts = [json.loads(line) for line in prompts_text.splitlines()]
        self.models = {
            k: digest("packed" if k in ANALYSIS.W1AX_BITS else k) for k in ANALYSIS.MODEL_KEYS
        }
        for cap, run in self.runs.items():
            run.mkdir()
            (run / "prompts.jsonl").write_text(prompts_text)
            config = (
                "\n".join(
                    ["[evaluation]"]
                    + [
                        f"{k} = {json.dumps(v)}"
                        for k, v in {
                            "max_output_tokens": cap,
                            "repetitions": 5,
                            "warmup_requests": 2,
                            "max_draft_tokens": 5,
                            "min_draft_probability": 0.0,
                            "prompt_set": "context_diagnostic",
                            "w1ax_matrix": True,
                            "w1ax_policy_diagnostic": False,
                        }.items()
                    ]
                )
                + "\n"
            )
            (run / "config.toml").write_text(config)
            manifest = {
                "variants": list(ANALYSIS.VARIANTS),
                "policy": ANALYSIS.POLICY,
                "config_sha256": digest(config),
                "request_options": {"max_tokens": cap, "seed": 42},
                "precision": {"target_weights": "f16", "target_kv": "f16"},
                "prompt_ids": [p["id"] for p in self.prompts],
                "files": {
                    **{k: {"sha256": h} for k, h in self.models.items()},
                    "prompt_file": {"sha256": ANALYSIS.FROZEN_PROMPT_SHA256},
                    "binary": {"sha256": digest("server")},
                },
                "variant_specs": {
                    k: {"draft_model_sha256": self.models[k]}
                    for k in ("draft_q8_0", "draft_q4_0", *ANALYSIS.W1AX_BITS)
                },
            }
            write_json(run / "manifest.json", manifest)
            write_json(
                run / "report.json",
                {
                    "status": "complete",
                    "records": 360,
                    "variants": list(ANALYSIS.VARIANTS),
                    "policy": ANALYSIS.POLICY,
                    "w1ax_dispatch_confirmed_by_variant": {v: True for v in ANALYSIS.W1AX_BITS},
                },
            )
            rows = []
            for rep in range(5):
                for prompt in self.prompts:
                    for variant in ANALYSIS.VARIANTS:
                        model = "ordinary_draft" if variant == "ordinary_eagle" else variant
                        length = {"short": 200, "medium": 500, "long": 1100}[prompt["context_bin"]]
                        row = {
                            "variant": variant,
                            "repetition": rep,
                            "prompt_id": prompt["id"],
                            "request_id": f"rep-{rep:02d}/{variant}/{prompt['id']}",
                            "policy_mode": "primary_matrix",
                            "max_draft_tokens": 5,
                            "min_draft_probability": 0.0,
                            "w1ax_activation_bits_selector": str(ANALYSIS.W1AX_BITS[variant])
                            if variant in ANALYSIS.W1AX_BITS
                            else None,
                            "draft_model_sha256": self.models.get(model),
                            "prompt_tokens": length,
                            "server_prompt_n": length,
                            "request_wall_s": 1.0 + rep,
                            "server_predicted_ms": 500 + rep,
                            "server_prompt_ms": 10 + rep,
                        }
                        set_ids(row, list(range(cap)))
                        rows.append(row)
            write_json(run / "records.json", rows)

    def change(self, cap, file, fn):
        path = self.runs[cap] / file
        value = json.loads(path.read_text())
        fn(value)
        write_json(path, value)

    def analyze(self, **kwargs):
        return ANALYSIS.analyze(self.runs[32], self.runs[128], **kwargs)

    def test_pooled_rates_and_cap_prefix_are_bounded(self):
        result = self.analyze()
        row = result["caps"]["32"]["by_bin"]["short"]["draft_w1a1"]
        self.assertEqual(row["requests"], 15)
        self.assertAlmostEqual(row["request_tokens_per_s"], 32 / 3)
        self.assertEqual(
            row["prefill_server_ms"], {"observations": 15, "median": 12, "p95": 14, "max": 14}
        )
        self.assertEqual(row["vs_anchors"]["fp16_eagle"]["decode"], 1)
        cross = result["cross_cap_outputs"]["draft_w1a1"]
        self.assertEqual(cross["mismatched_requests"], 0)
        self.assertEqual(cross["scope"], "common_prefix_only")
        self.assertEqual(cross["unequal_observed_length_requests"], 45)
        self.assertEqual(cross["length_scope_by_pair"][0]["compared_prefix_length"], 32)
        self.assertFalse(cross["strict_lossless_speedup_claim"])

    def test_native_bin_boundaries(self):
        for bin_name, (lower, upper) in ANALYSIS.BOUNDS.items():
            for boundary in (lower, upper):
                rows = json.loads((self.runs[32] / "records.json").read_text())
                ids = {p["id"] for p in self.prompts if p["context_bin"] == bin_name}
                for row in rows:
                    if row["prompt_id"] in ids:
                        row["prompt_tokens"] = boundary
                ANALYSIS.validate_records(rows, self.prompts, 32)
            rows[0]["prompt_tokens"] = 257  # first prompt is short
            with self.assertRaisesRegex(ValueError, "outside short bin"):
                ANALYSIS.validate_records(rows, self.prompts, 32)

    def test_missing_and_duplicate_pairs_are_fatal(self):
        self.change(32, "records.json", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "360 records"):
            self.analyze()
        self.setUp()
        self.change(32, "records.json", lambda rows: rows.__setitem__(1, rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze()

    def test_output_differences_are_nonfatal_and_located(self):
        def mutate(rows):
            row = next(r for r in rows if r["variant"] == "draft_w1a1")
            ids = row["generated_token_ids"][:]
            ids[7] = 999
            set_ids(row, ids)

        self.change(32, "records.json", mutate)
        result = self.analyze()
        comparison = result["caps"]["32"]["output_comparisons"]["draft_w1a1/vs/fp16_eagle"]
        self.assertEqual(comparison["mismatched_requests"], 1)
        self.assertEqual(
            comparison["mismatches"][0]["first_difference"],
            {"index": 7, "candidate_token_id": 999, "reference_token_id": 7},
        )
        self.assertIn("sha256", comparison["reference_source"])
        self.assertEqual(result["cross_cap_outputs"]["draft_w1a1"]["mismatched_requests"], 1)

    def test_within_cap_length_difference_is_mismatch(self):
        self.change(32, "records.json", lambda rows: set_ids(rows[0], list(range(31))))
        result = self.analyze()
        diff = result["caps"]["32"]["output_comparisons"]["ordinary_eagle/vs/target_only"]
        self.assertEqual(diff["mismatches"][0]["first_difference"]["index"], 31)
        self.assertIsNone(diff["mismatches"][0]["first_difference"]["reference_token_id"])
        self.assertEqual(result["cross_cap_outputs"]["target_only"]["mismatched_requests"], 0)

    def test_missing_raw_ids_is_not_a_mismatch(self):
        self.change(32, "records.json", lambda rows: rows[0].pop("generated_token_ids"))
        with self.assertRaisesRegex(ValueError, "missing or invalid raw token IDs"):
            self.analyze()

    def test_freeze_dispatch_policy_and_cap_are_strict(self):
        cases = [
            ("report.json", lambda r: r.__setitem__("status", "running"), "incomplete report"),
            (
                "report.json",
                lambda r: r["w1ax_dispatch_confirmed_by_variant"].__setitem__("draft_w1a1", False),
                "unconfirmed",
            ),
            (
                "manifest.json",
                lambda r: r["request_options"].__setitem__("max_tokens", 128),
                "wrong output cap",
            ),
            (
                "manifest.json",
                lambda r: r["policy"].__setitem__("warmup_requests", 1),
                "two warmups",
            ),
            (
                "manifest.json",
                lambda r: r["files"]["prompt_file"].__setitem__("sha256", "0" * 64),
                "frozen prompt SHA",
            ),
        ]
        for file, mutation, message in cases:
            with self.subTest(message=message):
                path = self.runs[32] / file
                original = path.read_text()
                self.change(32, file, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self.analyze()
                path.write_text(original)

    def test_cross_cap_model_hash_change_is_fatal(self):
        self.change(
            128, "manifest.json", lambda r: r["files"]["target"].__setitem__("sha256", "a" * 64)
        )
        with self.assertRaisesRegex(ValueError, "model hashes differ between caps"):
            self.analyze()

    def test_cross_cap_binary_hash_change_is_fatal(self):
        self.change(
            128, "manifest.json", lambda r: r["files"]["binary"].__setitem__("sha256", "a" * 64)
        )
        with self.assertRaisesRegex(ValueError, "server binary hashes differ between caps"):
            self.analyze()

    def test_cross_cap_commit_provenance_reports_missing_and_different(self):
        result = self.analyze()["cross_cap_build_provenance"]
        self.assertEqual(result["server_binary_sha256"]["status"], "match")
        self.assertEqual(result["project_commit"]["status"], "unavailable")
        self.assertEqual(result["llama_checkout_commit"]["status"], "unavailable")
        for cap in (32, 128):
            self.change(
                cap,
                "manifest.json",
                lambda r: r.update(project_commit="a" * 40, llama_checkout_commit="b" * 40),
            )
        self.change(128, "manifest.json", lambda r: r.__setitem__("project_commit", "c" * 40))
        result = self.analyze()["cross_cap_build_provenance"]
        self.assertEqual(result["project_commit"]["status"], "different")
        self.assertEqual(result["project_commit"]["by_cap"], {"32": "a" * 40, "128": "c" * 40})
        self.assertEqual(result["llama_checkout_commit"]["status"], "match")

    def test_hf_and_native_counts_keep_distinct_labels(self):
        audit = self.root / "hf.json"
        write_json(
            audit,
            {
                "prompts_sha256": ANALYSIS.FROZEN_PROMPT_SHA256,
                "rows": [{"id": p["id"], "tokenized_prompt_length": 190} for p in self.prompts],
            },
        )
        result = self.analyze(hf_audit=audit)
        row = result["caps"]["32"]["prompt_lengths"][0]
        self.assertEqual(row["hf_freeze_prompt_tokens"], 190)
        self.assertEqual(row["native_prompt_tokens_observed"], [200])
        self.assertEqual(row["native_minus_hf_observed"], [10])
        self.assertIsNotNone(result["hf_freeze_audit"]["sha256"])

    def test_bootstrap_option_reuses_paired_intervals(self):
        rows = json.loads((self.runs[32] / "records.json").read_text())
        selected = [r for r in rows if r["prompt_id"].endswith("-short")]
        result = ANALYSIS.summarize(selected, 100, 42)
        self.assertEqual(
            result["draft_w1a1"]["paired_bootstrap_vs_anchors"]["fp16_eagle"]["decode"],
            {"p2_5": 1, "median": 1, "p97_5": 1},
        )


if __name__ == "__main__":
    unittest.main()
