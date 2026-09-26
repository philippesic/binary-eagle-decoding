"""W1Ax matrix contracts without a GPU."""

import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_script("benchmark_native_eagle.py")
analysis = load_script("analyze_native_benchmark.py")


class W1AxMatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((ROOT / "configs/native_benchmark_w1ax.toml").read_text())

    def test_chat_verbose_raw_generated_ids(self):
        response = {
            "id": "chatcmpl-example",
            "choices": [{"finish_reason": "length", "message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "__verbose": {"tokens": [101, 202, 303]},
        }
        self.assertEqual(bench.generated_token_ids(response), [101, 202, 303])
        record = bench.extract_record(response, 1.0, {"proposed": 3, "accepted": 2, "rounds": 1})
        self.assertEqual(record["generated_token_ids"], [101, 202, 303])
        self.assertEqual(record["server_response_id"], "chatcmpl-example")
        self.assertEqual(record["generated_token_ids_status"], "available")
        self.assertIsNone(bench.generated_token_ids({"__verbose": {"tokens": [True, 2]}}))

    def test_matrix_specs_and_distinct_dispatch(self):
        variants = bench.selected_variants(self.config["evaluation"])
        self.assertEqual(variants, analysis.W1AX_MATRIX_VARIANTS)
        self.assertEqual(len(bench.schedule(5, variants)), 5)
        self.assertTrue(all(set(order) == set(variants) for order in bench.schedule(5, variants)))
        specs = bench.w1ax_specs(self.config, variants)
        self.assertEqual({spec["draft"] for spec in specs.values()}, {self.config["w1ax"]["draft"]})
        for variant, spec in specs.items():
            log = "\n".join((
                spec["expected_loader_marker"],
                spec["expected_graph_marker"],
                spec["expected_cuda_marker"],
            ))
            self.assertTrue(bench.w1ax_dispatch_evidence(log, spec, specs)["cuda_w1ax_dispatch_confirmed"])
            self.assertFalse(bench.w1ax_dispatch_evidence(
                log.replace(spec["expected_loader_marker"], "partial loader"), spec, specs
            )["cuda_w1ax_dispatch_confirmed"])
            other = next(row for key, row in specs.items() if key != variant)
            self.assertFalse(bench.w1ax_dispatch_evidence(
                log + "\n" + other["expected_cuda_marker"], spec, specs
            )["cuda_w1ax_dispatch_confirmed"])
            if variant == "draft_w1a4":
                self.assertFalse(bench.w1ax_dispatch_evidence(
                    log + "\n" + bench.W1AX_A4_CONVENTIONAL_MARKER, spec, specs
                )["cuda_w1ax_dispatch_confirmed"])
        self.assertFalse(bench.w1ax_dispatch_evidence(
            specs["draft_w1a16"]["expected_graph_marker"] + "\n"
            + specs["draft_w1a16"]["expected_cuda_marker"],
            specs["draft_w1a1"], specs,
        )["cuda_w1ax_dispatch_confirmed"])

    def test_config_validation_and_commands(self):
        variants = bench.selected_variants(self.config["evaluation"])
        specs = bench.w1ax_specs(self.config, variants)
        with tempfile.TemporaryDirectory() as temporary:
            paths = {"binary": Path(temporary) / "server", "target": Path(temporary) / "target",
                     **{variant: Path(temporary) / "all-nine.gguf" for variant in specs},
                     "ordinary_draft": Path(temporary) / "f16.gguf",
                     "draft_q8_0": Path(temporary) / "q8.gguf",
                     "draft_q4_0": Path(temporary) / "q4.gguf"}
            commands = {variant: bench.command_for(self.config, paths, variant) for variant in variants}
            self.assertNotIn("-md", commands["target_only"])
            for variant in bench.W1AX_VARIANTS:
                self.assertEqual(commands[variant][commands[variant].index("-md") + 1],
                                 str(paths[variant]))
        bad = {**self.config, "w1ax": {**self.config["w1ax"], "expected_loader_marker": "head only"}}
        with self.assertRaisesRegex(ValueError, "all nine"):
            bench.w1ax_specs(bad, variants)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            bench.selected_variants({"w1ax_matrix": True, "group_matrix": True})
        with self.assertRaisesRegex(ValueError, "five"):
            bench.schedule(4, variants)

    def test_primary_policy_is_frozen_and_diagnostic_grid_is_distinct(self):
        primary = bench.w1ax_policy(self.config["evaluation"], True)
        self.assertEqual(primary["mode"], "primary_matrix")
        self.assertEqual(primary["max_draft_tokens"], 5)
        self.assertEqual(primary["min_draft_probability"], 0.0)
        for draft_length in (1, 2, 3, 5):
            for floor in (0.0, 0.1, 0.3):
                diagnostic = {
                    **self.config["evaluation"],
                    "w1ax_policy_diagnostic": True,
                    "prompt_set": "qat_development",
                    "max_draft_tokens": draft_length,
                    "min_draft_probability": floor,
                    "warmup_requests": 1,
                    "repetitions": 6,
                }
                policy = bench.w1ax_policy(diagnostic, True)
                self.assertEqual(policy["mode"], "policy_diagnostic")
                self.assertEqual((policy["max_draft_tokens"], policy["min_draft_probability"]),
                                 (draft_length, floor))
                self.assertEqual((policy["warmup_requests"], policy["repetitions"]), (1, 6))
                self.assertEqual(bench.selected_variants(diagnostic), analysis.W1AX_MATRIX_VARIANTS)
        with self.assertRaisesRegex(ValueError, "primary matrix"):
            bench.w1ax_policy({**self.config["evaluation"], "max_draft_tokens": 3}, True)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            bench.w1ax_policy({**diagnostic, "min_draft_probability": 0.2}, True)
        with self.assertRaisesRegex(ValueError, "prompt_set"):
            bench.w1ax_policy({**diagnostic, "prompt_set": "qat_final"}, True)
        with self.assertRaisesRegex(ValueError, "requires w1ax_matrix"):
            bench.selected_variants({"w1ax_policy_diagnostic": True})

    def test_dry_run_manifest_has_shared_gguf_and_mode_selectors(self):
        source = (ROOT / "configs/native_benchmark_w1ax.toml").read_text()
        replacements = {
            "build/llama-cuda/bin/llama-server": "server",
            "models/gguf/Qwen3-4B-f16.gguf": "target.gguf",
            "models/gguf/Qwen3-4B-eagle3-f16.gguf": "f16.gguf",
            "models/gguf/Qwen3-4B-eagle3-q8_0.gguf": "q8.gguf",
            "models/gguf/Qwen3-4B-eagle3-q4_0.gguf": "q4.gguf",
            "models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf": "all.gguf",
            "configs/acceptance_prompts.jsonl": "prompts.jsonl",
        }
        for old, new in replacements.items():
            source = source.replace(old, new)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("server", "target.gguf", "f16.gguf", "q8.gguf", "q4.gguf", "all.gguf"):
                (directory / name).write_bytes(b"fake")
            (directory / "prompts.jsonl").write_text(
                '{"id":"prose","messages":[{"role":"user","content":"Hi"}]}\n'
            )
            (directory / "config.toml").write_text(source)
            with (
                patch.object(bench, "ROOT", directory),
                patch.object(bench, "available_port", return_value=True),
                patch.object(bench, "gpu_snapshot", return_value={"available": False}),
                patch.object(bench, "environment_manifest", return_value={}),
                patch.object(bench, "git_output", side_effect=lambda *args: (
                    "160000 commit abc\tthird_party/llama.cpp\n"
                    if args == ("ls-tree", "HEAD", "third_party/llama.cpp") else "abc\n"
                )),
            ):
                output = bench.run(directory / "config.toml", "dry", dry_run=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["variants"], list(analysis.W1AX_MATRIX_VARIANTS))
            paths = {manifest["files"][variant]["path"] for variant in bench.W1AX_VARIANTS}
            self.assertEqual(paths, {str((directory / "all.gguf").resolve())})
            self.assertEqual(
                {variant: manifest["variant_environments"][variant]["GGML_W1AX_ACT_BITS"]
                 for variant in bench.W1AX_VARIANTS},
                {variant: str(bits) for variant, bits in bench.W1AX_BITS.items()},
            )
            self.assertEqual(
                manifest["variant_environments"]["draft_w1a4"]["GGML_W1AX_A4_KERNEL"],
                "bitserial",
            )
            self.assertNotIn("GGML_W1AX_ACT_BITS", manifest["variant_environments"]["ordinary_eagle"])
            diagnostic_config = source.replace("max_draft_tokens = 5", "max_draft_tokens = 2")
            diagnostic_config = diagnostic_config.replace(
                "min_draft_probability = 0.0", "min_draft_probability = 0.1"
            ).replace("warmup_requests = 2", "warmup_requests = 1").replace(
                "repetitions = 5", "repetitions = 6"
            ).replace(
                "round_trace = false", 'round_trace = false\nw1ax_policy_diagnostic = true\nprompt_set = "qat_development"'
            )
            (directory / "diagnostic.toml").write_text(diagnostic_config)
            (directory / "prompts.jsonl").write_text("".join(
                json.dumps({"id": f"dev-{index:02d}", "messages": [
                    {"role": "user", "content": "Hi"}
                ]}) + "\n"
                for index in range(24)
            ))
            with (
                patch.object(bench, "ROOT", directory),
                patch.object(bench, "available_port", return_value=True),
                patch.object(bench, "gpu_snapshot", return_value={"available": False}),
                patch.object(bench, "environment_manifest", return_value={}),
                patch.object(bench, "git_output", side_effect=lambda *args: (
                    "160000 commit abc\tthird_party/llama.cpp\n"
                    if args == ("ls-tree", "HEAD", "third_party/llama.cpp") else "abc\n"
                )),
            ):
                diagnostic_output = bench.run(directory / "diagnostic.toml", "diagnostic", dry_run=True)
            diagnostic_manifest = json.loads((diagnostic_output / "manifest.json").read_text())
            self.assertEqual(diagnostic_manifest["policy"]["mode"], "policy_diagnostic")
            self.assertEqual(diagnostic_manifest["policy"]["max_draft_tokens"], 2)
            self.assertEqual(diagnostic_manifest["policy"]["min_draft_probability"], 0.1)
            self.assertEqual(len(diagnostic_manifest["orders"]), 6)
            self.assertEqual(len(diagnostic_manifest["prompt_ids"]), 24)
            for variant in bench.W1AX_VARIANTS:
                command = diagnostic_manifest["commands"][variant]
                self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "2")
                self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0.1")

    def test_analysis_requires_dispatch_and_dual_anchors(self):
        rows = []
        for rep in range(5):
            for prompt_id in ("prose", "code"):
                for index, variant in enumerate(analysis.W1AX_MATRIX_VARIANTS):
                    rows.append({
                        "repetition": rep, "prompt_id": prompt_id, "variant": variant,
                        "completion_tokens": 4, "request_wall_s": 2.0 / (index + 1),
                        "generated_token_ids": [1, 2, 3, 4],
                        "server_predicted_ms": 1000.0 / (index + 1),
                        "w1ax_activation_bits_selector": str(analysis.W1AX_BITS[variant])
                        if variant in analysis.W1AX_BITS else None,
                        "speculative": {"proposed": 4, "accepted": 2, "rounds": 2},
                    })
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "records.json").write_text(json.dumps(rows))
            (directory / "prompts.jsonl").write_text(
                '{"id":"prose","category":"prose"}\n'
                '{"id":"code","category":"code"}\n'
            )
            report = {
                "status": "complete", "records": len(rows),
                "variants": list(analysis.W1AX_MATRIX_VARIANTS),
                "w1ax_dispatch_confirmed_by_variant": {
                    variant: True for variant in analysis.W1AX_VARIANTS
                },
            }
            (directory / "report.json").write_text(json.dumps(report))
            result = analysis.analyze(directory, 100, 42)
            dual = result["pooled_w1ax_speedup_vs_fp16_and_q4_0"]["draft_w1a1"]
            self.assertEqual(dual["ordinary_eagle/request"], 4.0)
            self.assertEqual(dual["draft_q4_0/request"], 2.0)
            self.assertIn("draft_q4_0/request",
                          result["paired_w1ax_bootstrap_95pct_vs_fp16_and_q4_0"]["draft_w1a4"])
            report["w1ax_dispatch_confirmed_by_variant"]["draft_w1a4"] = False
            (directory / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "dispatch is unconfirmed"):
                analysis.analyze(directory, 100, 42)


if __name__ == "__main__":
    unittest.main()
