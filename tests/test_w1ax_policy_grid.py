"""Synthetic offline checks for the 12-cell W1Ax development policy grid."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_w1ax_policy_grid.py"
SPEC = importlib.util.spec_from_file_location("analyze_w1ax_policy_grid", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_w1ax_diagnostics", SCRIPT.with_name("prepare_w1ax_diagnostics.py")
)
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class PolicyGridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.suite_path = self.root / "suite.json"
        self.results = self.root / "results"
        self.results.mkdir()
        self.ids = [f"dev-{index:02d}" for index in range(24)]
        self.prompts_text = "".join(
            json.dumps({"id": value, "messages": []}) + "\n" for value in self.ids
        )
        self.prompt_hash = digest(self.prompts_text)
        self.models = tuple(ANALYSIS.MODEL_SOURCE_KEYS)
        self.model_hashes = {key: digest(key) for key in set(ANALYSIS.MODEL_SOURCE_KEYS.values())}
        self.model_hashes["server_binary"] = digest("server")
        inputs = {
            key: {"path": f"/frozen/{key}", "sha256": value}
            for key, value in self.model_hashes.items()
        }
        inputs["prompt:qat_development"] = {
            "path": "/frozen/development.jsonl",
            "sha256": self.prompt_hash,
        }
        inputs["primary_config"] = {"path": "/frozen/primary.toml", "sha256": digest("primary")}
        suite = {
            "schema": "w1ax_diagnostic_suite_v1",
            "input_files": inputs,
            "model_server_hashes": self.model_hashes,
            "primary_config_sha256": digest("primary"),
            "prompt_sets": {
                "qat_development": {
                    "path": "/frozen/development.jsonl",
                    "sha256": self.prompt_hash,
                    "count": 24,
                    "ids": self.ids,
                }
            },
            "configs": [],
        }
        progress = {
            "schema": "w1ax_diagnostic_progress_v1",
            "suite_path": str(self.suite_path),
            "configs": {},
        }
        for depth in ANALYSIS.DEPTHS:
            for floor in ANALYSIS.FLOORS:
                key = ANALYSIS.cell_key(depth, floor)
                run_id = f"synthetic-{key}"
                run = self.results / run_id
                run.mkdir()
                name = key + ".toml"
                config = f"depth={depth}\nfloor={floor}\n"
                (run / "config.toml").write_text(config)
                config_hash = digest(config)
                suite["configs"].append(
                    {
                        "name": name,
                        "sha256": config_hash,
                        "prompt_set": "qat_development",
                        "prompt_count": 24,
                        "prompt_file_sha256": self.prompt_hash,
                        "w1ax_policy_diagnostic": True,
                        "draft_depth": depth,
                        "min_draft_probability": floor,
                    }
                )
                progress["configs"][name] = {
                    "status": "succeeded",
                    "config_sha256": config_hash,
                    "attempts": [
                        {
                            "status": "succeeded",
                            "exit_code": 0,
                            "config_sha256": config_hash,
                            "run_id": run_id,
                        }
                    ],
                }
                policy = {
                    "mode": "policy_diagnostic",
                    "prompt_set": "qat_development",
                    "max_draft_tokens": depth,
                    "min_draft_probability": floor,
                    "warmup_requests": 2,
                    "repetitions": 5,
                }
                resolved_hashes = {
                    model: self.model_hashes[source_key]
                    for model, source_key in ANALYSIS.MODEL_SOURCE_KEYS.items()
                }
                files = {model: {"sha256": resolved_hashes[model]} for model in self.models}
                files["binary"] = {"sha256": self.model_hashes["server_binary"]}
                files["prompt_file"] = {"sha256": self.prompt_hash}
                manifest = {
                    "variants": list(ANALYSIS.VARIANTS),
                    "config_sha256": config_hash,
                    "prompt_ids": self.ids,
                    "files": files,
                    "policy": policy,
                    "variant_specs": {
                        variant: {"draft_model_sha256": resolved_hashes[variant]}
                        for variant in ("draft_q8_0", "draft_q4_0", *ANALYSIS.W1AX)
                    },
                }
                write_json(run / "manifest.json", manifest)
                (run / "prompts.jsonl").write_text(self.prompts_text)
                records = []
                for rep in range(5):
                    for variant in ANALYSIS.W1AX:
                        write_json(
                            run / f"rep-{rep:02d}" / variant / "dispatch-evidence.json",
                            {"cuda_w1ax_dispatch_confirmed": True},
                        )
                    for prompt_id in self.ids:
                        for variant in ANALYSIS.VARIANTS:
                            ids = [101, 102]
                            ids_hash = hashlib.sha256(
                                json.dumps(ids, separators=(",", ":")).encode()
                            ).hexdigest()
                            decode_ms = 100 if variant == "draft_w1a1" and depth == 2 else 200
                            records.append(
                                {
                                    "repetition": rep,
                                    "prompt_id": prompt_id,
                                    "variant": variant,
                                    "request_id": f"rep-{rep:02d}/{variant}/{prompt_id}",
                                    "policy_mode": "policy_diagnostic",
                                    "max_draft_tokens": depth,
                                    "min_draft_probability": floor,
                                    "generated_token_ids": ids,
                                    "generated_token_ids_sha256": ids_hash,
                                    "completion_tokens": 2,
                                    "request_wall_s": 0.4,
                                    "server_predicted_ms": decode_ms,
                                    "draft_model_sha256": (
                                        None
                                        if variant == "target_only"
                                        else resolved_hashes[
                                            "ordinary_draft"
                                            if variant == "ordinary_eagle"
                                            else variant
                                        ]
                                    ),
                                    "speculative": {"accepted": 1, "proposed": 2, "rounds": 1}
                                    if variant != "target_only"
                                    else {"accepted": None, "proposed": None, "rounds": None},
                                }
                            )
                write_json(run / "records.json", records)
                write_json(
                    run / "report.json",
                    {
                        "status": "complete",
                        "variants": list(ANALYSIS.VARIANTS),
                        "policy": policy,
                        "records": len(records),
                        "w1ax_dispatch_confirmed_by_variant": {
                            variant: True for variant in ANALYSIS.W1AX
                        },
                    },
                )
        write_json(self.suite_path, suite)
        write_json(self.root / "progress.json", progress)

    def tearDown(self):
        self.tmp.cleanup()

    def replace_raw_ids(self, cell, variant, token_ids):
        path = self.results / f"synthetic-{cell}" / "records.json"
        rows = json.loads(path.read_text())
        row = next(
            row
            for row in rows
            if row["variant"] == variant
            and row["repetition"] == 2
            and row["prompt_id"] == self.ids[3]
        )
        row["generated_token_ids"] = token_ids
        row["generated_token_ids_sha256"] = digest(json.dumps(token_ids, separators=(",", ":")))
        row["completion_tokens"] = len(token_ids)
        write_json(path, rows)
        return row, path

    def test_all_cells_pool_rates_and_select_development_policy(self):
        report = ANALYSIS.analyze(self.suite_path, self.results)
        self.assertEqual(len(report["cells"]), 12)
        self.assertEqual(
            report["best_development_policy_by_variant"]["draft_w1a1"]["cell"], "d2-pmin-0.0"
        )
        summary = report["cells"]["d2-pmin-0.0"]["variants"]["draft_w1a1"]
        self.assertEqual(summary["requests"], 120)
        self.assertAlmostEqual(summary["decode_tokens_per_s"], 20.0)
        self.assertAlmostEqual(summary["vs_anchors"]["fp16_eagle"]["decode"], 2.0)
        self.assertEqual(summary["mean_proposal_length"], 2)
        self.assertEqual(report["fixed_primary_policy"]["cell"], "d5-pmin-0.0")
        for comparison in summary["raw_output_vs"].values():
            self.assertEqual(comparison["paired_requests"], 120)
            self.assertEqual(comparison["matched_requests"], 120)
            self.assertEqual(comparison["mismatches"], [])
            self.assertTrue(comparison["all_observed_token_ids_match"])
            self.assertFalse(comparison["strict_lossless_speedup_claim"])
        selected = report["best_development_policy_by_variant"]["draft_w1a1"]
        self.assertEqual(selected["selection_scope"], "exploratory_development_only")
        self.assertEqual(len(selected["raw_output_comparisons"]), 3)
        for comparison in selected["raw_output_comparisons"].values():
            self.assertTrue(comparison["all_observed_token_ids_match"])
            self.assertEqual(comparison["mismatches"], [])

    def test_output_differences_preserve_cells_and_first_divergence_evidence(self):
        changed, changed_path = self.replace_raw_ids("d2-pmin-0.0", "draft_w1a1", [101, 999, 103])
        fixed, fixed_path = self.replace_raw_ids("d5-pmin-0.0", "draft_w1a1", [101, 999])
        report = ANALYSIS.analyze(self.suite_path, self.results)
        self.assertEqual(len(report["cells"]), 12)
        summary = report["cells"]["d2-pmin-0.0"]["variants"]["draft_w1a1"]
        for comparison in summary["raw_output_vs"].values():
            self.assertEqual(comparison["matched_requests"], 119)
            self.assertEqual(comparison["mismatched_requests"], 1)
            self.assertEqual(comparison["mismatched_prompt_count"], 1)
            mismatch = comparison["mismatches"][0]
            self.assertEqual((mismatch["repetition"], mismatch["prompt_id"]), (2, self.ids[3]))
            self.assertEqual(
                mismatch["candidate_token_ids_sha256"], changed["generated_token_ids_sha256"]
            )
            self.assertEqual(
                mismatch["first_difference"],
                {
                    "index": 1,
                    "candidate_token_id": 999,
                    "reference_token_id": 102,
                    "candidate_length": 3,
                    "reference_length": 2,
                },
            )
            self.assertEqual(comparison["candidate"]["records_path"], str(changed_path.resolve()))
            self.assertEqual(
                comparison["candidate"]["records_sha256"], ANALYSIS.sha256(changed_path)
            )
            self.assertFalse(comparison["strict_lossless_speedup_claim"])
            self.assertIn("No numerical or other cause", comparison["interpretation"])
        for ratio in summary["vs_anchors"].values():
            self.assertEqual(
                ratio["timing_classification"], "timing_observation_with_output_differences"
            )
        selected = report["best_development_policy_by_variant"]["draft_w1a1"]
        self.assertEqual(selected["cell"], "d2-pmin-0.0")
        comparisons = selected["raw_output_comparisons"]
        for label in ("selected_fp16_eagle", "selected_q4_0_eagle"):
            comparison = comparisons[label]
            self.assertEqual(comparison["reference"]["cell"], "d1-pmin-0.0")
            self.assertEqual(comparison["mismatches"][0]["first_difference"]["index"], 1)
            self.assertEqual(comparison["selection_scope"], "exploratory_development_only")
        fixed_comparison = comparisons["same_variant_fixed_d5_p0"]
        self.assertEqual(fixed_comparison["reference"]["records_path"], str(fixed_path.resolve()))
        mismatch = fixed_comparison["mismatches"][0]
        self.assertEqual(
            mismatch["reference_token_ids_sha256"], fixed["generated_token_ids_sha256"]
        )
        self.assertEqual(
            mismatch["first_difference"],
            {
                "index": 2,
                "candidate_token_id": 103,
                "reference_token_id": None,
                "candidate_length": 3,
                "reference_length": 2,
            },
        )

    def test_missing_grid_cell_rejected(self):
        suite = json.loads(self.suite_path.read_text())
        suite["configs"].pop()
        write_json(self.suite_path, suite)
        with self.assertRaisesRegex(ValueError, "12 policy configs"):
            ANALYSIS.analyze(self.suite_path, self.results)

    def test_changed_model_hash_rejected(self):
        run = self.results / "synthetic-d1-pmin-0.0"
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["files"]["target"]["sha256"] = digest("replacement")
        write_json(run / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "model hash changed: target"):
            ANALYSIS.analyze(self.suite_path, self.results)

    def test_missing_raw_pair_rejected(self):
        run = self.results / "synthetic-d1-pmin-0.0"
        records = json.loads((run / "records.json").read_text())
        records.pop()
        write_json(run / "records.json", records)
        with self.assertRaisesRegex(
            ValueError, "incomplete paired records|record count disagreement"
        ):
            ANALYSIS.analyze(self.suite_path, self.results)

    def test_changed_prompt_content_rejected(self):
        run = self.results / "synthetic-d1-pmin-0.0"
        (run / "prompts.jsonl").write_text(self.prompts_text + "\n")
        with self.assertRaisesRegex(ValueError, "development prompt IDs/content changed"):
            ANALYSIS.analyze(self.suite_path, self.results)

    def test_missing_dispatch_rejected(self):
        run = self.results / "synthetic-d1-pmin-0.0"
        evidence_path = run / "rep-03" / "draft_w1a4" / "dispatch-evidence.json"
        write_json(evidence_path, {"cuda_w1ax_dispatch_confirmed": False})
        with self.assertRaisesRegex(ValueError, "dispatch evidence missing"):
            ANALYSIS.analyze(self.suite_path, self.results)

    def test_source_key_mapping_matches_generator(self):
        for name in ("target", "ordinary", "q8", "q4", "w1ax", "server"):
            (self.root / name).write_text(name)
        config = self.root / "primary.toml"
        config.write_text(
            """schema_version = 1
[server]
binary = "server"
[models]
target = "target"
ordinary_draft = "ordinary"
[weight_only_variants.q8_0]
draft = "q8"
[weight_only_variants.q4_0]
draft = "q4"
[w1ax]
draft = "w1ax"
[evaluation]
w1ax_matrix = true
max_draft_tokens = 5
min_draft_probability = 0.0
max_output_tokens = 128
"""
        )
        dev = self.root / "development.jsonl"
        dev.write_text(self.prompts_text)
        context = self.root / "context.jsonl"
        context.write_text(
            "".join(json.dumps({"id": f"context-{i}", "messages": []}) + "\n" for i in range(9))
        )
        generated = PREPARE.prepare(config, dev, context, self.root / "generated", root=self.root)
        frozen = json.loads(generated.read_text())
        self.assertEqual(len(frozen["configs"]), 14)
        for source_key in set(ANALYSIS.MODEL_SOURCE_KEYS.values()):
            self.assertIn(source_key, frozen["input_files"])
            self.assertEqual(
                frozen["input_files"][source_key]["sha256"],
                frozen["model_server_hashes"][source_key],
            )
        self.assertNotIn("model:draft_w1a1", frozen["input_files"])


if __name__ == "__main__":
    unittest.main()
