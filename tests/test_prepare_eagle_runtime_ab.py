"""CPU-only checks for fair, immutable runtime A/B preparation and comparison."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_eagle_prune_quality as quality
import compare_eagle_prune_timing as timing
import prepare_eagle_runtime_ab as prepare
import run_binary_rescue_benchmark as runner


class RuntimePreparationTests(unittest.TestCase):
    def source(self):
        return {
            "schema_version": 1,
            "binary": "server",
            "target": "target.gguf",
            "prompts": "development.jsonl",
            "graph_env": {"GGML_EAGLE_PRUNE_UNUSED_HEAD": "0"},
            "variants": {
                "q4_0": {"draft": "q4.gguf"},
                "row_A4": {"draft": "row.gguf", "env": {"GGML_W1AX_ACT_BITS": "4"}},
                "fp16": {"draft": "fp16.gguf"},
                "target_only": {},
            },
        }

    def test_source_is_immutable_and_one_common_selector_differs(self):
        for name, selector in prepare.SELECTORS.items():
            source = self.source()
            source["variants"]["q4_0"]["env"] = {selector: "1"}
            original = copy.deepcopy(source)
            off, on = prepare.prepare(source, name)
            self.assertEqual(source, original)
            self.assertEqual(off["graph_env"][selector], "0")
            self.assertEqual(on["graph_env"][selector], "1")
            without_flag = copy.deepcopy(on)
            without_flag["graph_env"][selector] = "0"
            self.assertEqual(off, without_flag)
            for spec in off["variants"].values():
                self.assertNotIn(selector, spec.get("env", {}))
            runner.validate_config(off)
            runner.validate_config(on)

    def test_compact_cpu_sampling_is_common_to_both_conditions(self):
        off, on = prepare.prepare(self.source(), "compact_logits")
        for config in (off, on):
            self.assertFalse(config["draft_backend_sampling"])
            for spec in config["variants"].values():
                command = runner.command(config, spec, runner.PRIMARY)
                self.assertEqual(
                    "--no-spec-draft-backend-sampling" in command, bool(spec.get("draft"))
                )
        source = self.source()
        self.assertNotIn(
            "--no-spec-draft-backend-sampling",
            runner.command(source, source["variants"]["q4_0"], runner.PRIMARY),
        )
        source["draft_backend_sampling"] = "false"
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            runner.validate_config(source)

    def test_missing_q4_and_no_pack_variant_are_rejected(self):
        source = self.source()
        source["variants"].pop("q4_0")
        with self.assertRaisesRegex(ValueError, "Q4_0"):
            prepare.prepare(source, "kv_only")
        source = self.source()
        source["variants"]["row_A4"]["env"]["GGML_W1AX_ACT_BITS"] = "16"
        for selector in ("shared_pack", "warp_reduce"):
            with self.assertRaisesRegex(ValueError, "A1/A4/A8"):
                prepare.prepare(source, selector)

    def comparison_inputs(self, root, mode, selector):
        off, on = prepare.prepare(self.source(), "compact_logits")
        paths = []
        for name, config in (("off", off), ("on", on)):
            cell = root / name
            cell.mkdir()
            config["graph_env"][selector] = "0" if name == "off" else "1"
            (cell / "config.json").write_text(json.dumps(config))
            row = {
                "variant": "q4_0",
                "repetition": 0,
                "prompt_id": "synthetic",
                "warmup": False,
                "generated_token_ids": [7],
                "completion_tokens": 1,
                "finish_reason": "length",
                "directory": str(cell),
                "quality": {"rounds": 1, "proposed": 1, "accepted": 0, "emitted": 1},
            }
            (cell / "rounds.json").write_text(json.dumps([{"stop_probability": 0.25}]))
            manifest = {
                "schema": "binary_rescue_benchmark_v1",
                "mode": mode,
                "status": "complete",
                "records": [row],
                "workload": "synthetic",
                "policy": runner.PRIMARY,
                "q4_variant": "q4_0",
                "prompt_sha256": "a" * 64,
                "hashes": {},
            }
            path = cell / "manifest.json"
            path.write_text(json.dumps(manifest))
            paths.append(path)
        return paths

    def test_quality_comparator_tracks_selector_and_stop_probability(self):
        selector = prepare.SELECTORS["compact_logits"]
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.comparison_inputs(Path(directory), "quality", selector)
            report = quality.compare(off, on, selector)
            self.assertEqual(report["schema"], "eagle_runtime_quality_comparison_v1")
            self.assertEqual(report["mismatch_count"], 0)
            (on.parent / "rounds.json").write_text(json.dumps([{"stop_probability": 0.5}]))
            self.assertEqual(quality.compare(off, on, selector)["mismatch_count"], 1)
            config = json.loads((on.parent / "config.json").read_text())
            config["draft_backend_sampling"] = True
            (on.parent / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                quality.compare(off, on, selector)

    def test_timing_comparator_requires_only_selected_config_difference(self):
        selector = prepare.SELECTORS["compact_logits"]
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.comparison_inputs(Path(directory), "timed", selector)
            aggregate = {"client_request_tok_s": 1, "server_decode_tok_s": 2, "speculative": {}}
            paired = {
                "paired_requests": 1,
                "paired_prompts": 1,
                "same_raw_ids": True,
                "client_request_tok_s_ratio": 1,
                "client_request_tok_s_ratio_ci95": [1, 1],
                "server_decode_tok_s_ratio": 1,
                "server_decode_tok_s_ratio_ci95": [1, 1],
            }
            with (
                mock.patch.object(timing, "aggregate", return_value=aggregate),
                mock.patch.object(timing, "paired_comparison", return_value=paired),
            ):
                report = timing.compare(off, on, selector)
            self.assertEqual(report["schema"], "eagle_runtime_timing_comparison_v1")
            self.assertEqual(report["behavior_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
