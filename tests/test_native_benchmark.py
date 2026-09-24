"""Harness checks use a tiny local HTTP stand-in, never a GPU or model."""

import importlib.util
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_native_eagle.py"
SPEC = importlib.util.spec_from_file_location("benchmark_native_eagle", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

FAKE_SERVER = r"""#!PYTHON
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv
host = args[args.index("--host") + 1]
port = int(args[args.index("--port") + 1])
spec = args[args.index("--spec-type") + 1] != "none"
counts = {"proposed": 0, "accepted": 0, "rounds": 0}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def write(self, status, data, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(data.encode())

    def do_GET(self):
        if self.path == "/health":
            self.write(200, json.dumps({"status": "ok"}))
        elif self.path == "/props":
            self.write(200, json.dumps({"model_path": "fake.gguf"}))
        elif self.path == "/metrics":
            text = "\n".join([
                f"llamacpp:spec_decode_num_draft_tokens_total {counts['proposed']}",
                f"llamacpp:spec_decode_num_accepted_tokens_total {counts['accepted']}",
                f"llamacpp:spec_decode_num_drafts_total {counts['rounds']}",
            ]) + "\n"
            self.write(200, text, "text/plain")
        else:
            self.write(404, "{}")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.write(404, "{}")
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if os.environ.get("FAKE_FAIL") == "1":
            self.write(500, '{"error":"injected"}')
            return
        if spec:
            counts["proposed"] += 3
            counts["accepted"] += 2
            counts["rounds"] += 1
        self.write(200, json.dumps({
            "choices": [{"finish_reason": "length", "message": {"content": "test"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            "timings": {"prompt_n": 5, "prompt_ms": 20, "predicted_n": 4, "predicted_ms": 10},
            "seen": request["messages"],
        }))

ThreadingHTTPServer((host, port), Handler).serve_forever()
""".replace("#!PYTHON", f"#!{sys.executable}")


class NativeBenchmarkTests(unittest.TestCase):
    def test_schedule_and_missing_counters(self):
        orders = benchmark.schedule(6)
        self.assertEqual(len(orders), 6)
        self.assertTrue(all(set(order) == set(benchmark.VARIANTS) for order in orders))
        self.assertNotEqual(orders[0], orders[1])
        self.assertRaises(ValueError, benchmark.schedule, 4)
        self.assertEqual(
            benchmark.counter_delta(None, "llamacpp:spec_decode_num_drafts_total 5"),
            {"proposed": None, "accepted": None, "rounds": None},
        )
        evidence = benchmark.dispatch_evidence(
            "load: EAGLE3 using packed W1A1 draft head\nCUDA0 model buffer size = 2 MiB\n",
            "packed_head_w1a1",
        )
        self.assertTrue(evidence["packed_head_loader_log"])
        self.assertIsNone(evidence["cuda_w1a1_dispatch_confirmed"])
        confirmed = benchmark.dispatch_evidence(
            "EAGLE3 using packed W1A1 draft head\n"
            "CUDA packed W1A1 XOR/POPCOUNT dispatch (K=2560, rows=32000, tokens=1)\n",
            "packed_head_w1a1",
        )
        self.assertTrue(confirmed["cuda_w1a1_dispatch_confirmed"])
        marker_only = benchmark.dispatch_evidence(
            "CUDA packed W1A1 XOR/POPCOUNT dispatch", "packed_head_w1a1"
        )
        self.assertTrue(marker_only["cuda_w1a1_dispatch_confirmed"])

    def test_greedy_text_match_pairs(self):
        records = [
            {"repetition": 0, "prompt_id": "p", "variant": "target_only", "completion_sha256": "a"},
            {
                "repetition": 0,
                "prompt_id": "p",
                "variant": "ordinary_eagle",
                "completion_sha256": "a",
            },
            {
                "repetition": 0,
                "prompt_id": "p",
                "variant": "packed_head_w1a1",
                "completion_sha256": "b",
            },
        ]
        result = benchmark.completion_text_matches(records)
        self.assertEqual(result["ordinary_eagle"]["matched_text"], 1)
        self.assertEqual(
            result["packed_head_w1a1"]["mismatch_pairs"], [{"repetition": 0, "prompt_id": "p"}]
        )

    def test_aggregate_uses_ratio_of_sums(self):
        rows = []
        for variant in benchmark.VARIANTS:
            for tokens, wall, decode in ((2, 1.0, 10.0), (8, 2.0, 20.0)):
                rows.append(
                    {
                        "variant": variant,
                        "completion_tokens": tokens,
                        "request_wall_s": wall,
                        "server_predicted_ms": decode,
                        "speculative": {"accepted": None, "proposed": None, "rounds": None},
                    }
                )
        aggregate = benchmark.aggregate(rows)["ordinary_eagle"]
        self.assertAlmostEqual(aggregate["request_tokens_per_s"], 10 / 3)
        self.assertAlmostEqual(aggregate["decode_tokens_per_s"], 1000 / 3)
        self.assertIsNone(aggregate["acceptance_rate"])

    def test_relative_speedup_uses_pooled_rates(self):
        values = {
            "target_only": {"request_tokens_per_s": 10.0, "decode_tokens_per_s": 20.0},
            "ordinary_eagle": {"request_tokens_per_s": 8.0, "decode_tokens_per_s": 16.0},
            "packed_head_w1a1": {"request_tokens_per_s": 12.0, "decode_tokens_per_s": 24.0},
        }
        result = benchmark.relative_speedups(values)
        self.assertEqual(result["target_only"]["request_tokens_per_s"], 1.2)
        self.assertEqual(result["ordinary_eagle"]["decode_tokens_per_s"], 1.5)

    def test_speculative_timing_excludes_warmup(self):
        log = (
            "statistics draft-eagle3: dur(b,g,a) = 1.000, 4.000, 0.100 ms\n"
            "statistics draft-eagle3: dur(b,g,a) = 1.500, 7.250, 0.125 ms\n"
        )
        timing = benchmark.speculative_timing(log, "ordinary_eagle", 1, 1)
        self.assertEqual(timing["status"], "available")
        self.assertAlmostEqual(timing["measured_totals_ms"]["draft_ms"], 3.25)
        self.assertEqual(
            benchmark.speculative_timing(log, "target_only", 1, 1)["status"], "not_applicable"
        )
        self.assertEqual(
            benchmark.speculative_timing(log, "ordinary_eagle", 2, 1)["status"], "unavailable"
        )

    def test_summarize_draft_timing_uses_measured_rounds(self):
        timing = {
            "status": "available",
            "measured_totals_ms": {"begin_ms": 1.0, "draft_ms": 12.0, "accept_ms": 2.0},
        }
        entries = [
            {"variant": "ordinary_eagle", "timing": timing},
            {"variant": "packed_head_w1a1", "timing": timing},
        ]
        aggregated = {
            variant: {"speculative": {"rounds": 4}}
            for variant in ("ordinary_eagle", "packed_head_w1a1")
        }
        summary = benchmark.summarize_draft_timings(entries, aggregated)
        self.assertEqual(summary["ordinary_eagle"]["draft_ms_per_verification_round"], 3.0)

    def test_full_fake_run_preserves_artifacts_and_stops_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, port = self.prepare(root)
            with (
                patch.object(benchmark, "ROOT", root),
                patch.object(
                    benchmark,
                    "git_output",
                    return_value="160000 commit abc\tthird_party/llama.cpp\n",
                ),
            ):
                output = benchmark.run(config, "fake-run")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["records"], 15)
            self.assertEqual(len(report["repetitions"]), 5)
            self.assertIsNone(report["native_cuda_dispatch_confirmed"])
            self.assertEqual(
                report["aggregation"]["ordinary_eagle"]["speculative"],
                {
                    "accepted": 10,
                    "proposed": 15,
                    "rounds": 5,
                },
            )
            self.assertIsNone(report["aggregation"]["target_only"]["accepted_per_round"])
            self.assertTrue(
                (output / "rep-00/ordinary_eagle/measured/test-prompt/response.json").exists()
            )
            self.assertTrue((output / "rep-00/ordinary_eagle/server.log").exists())
            self.assertTrue((output / "rep-00/ordinary_eagle/dispatch-evidence.json").exists())
            self.assertTrue((output / "rep-00/ordinary_eagle/speculative-timing.json").exists())
            self.assertTrue(benchmark.available_port("127.0.0.1", port))

    def test_failed_request_stops_server_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, port = self.prepare(root, failure=True)
            with (
                patch.object(benchmark, "ROOT", root),
                patch.object(
                    benchmark,
                    "git_output",
                    return_value="160000 commit abc\tthird_party/llama.cpp\n",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                    benchmark.run(config, "failed-run")
            self.assertTrue((root / "results/failed-run/failure.json").exists())
            self.assertTrue(
                (
                    root / "results/failed-run/rep-00/target_only/warmup/request-00/response.json"
                ).exists()
            )
            self.assertTrue(benchmark.available_port("127.0.0.1", port))

    def prepare(self, root: Path, failure: bool = False) -> tuple[Path, int]:
        (root / "results").mkdir()
        fake = root / "fake-server"
        fake.write_text(FAKE_SERVER)
        fake.chmod(0o755)
        for name in ("target.gguf", "draft.gguf", "packed.gguf"):
            (root / name).write_bytes(b"fake")
        (root / "prompts.jsonl").write_text(
            json.dumps({"id": "test-prompt", "messages": [{"role": "user", "content": "Hi"}]})
            + "\n"
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = root / "config.toml"
        config.write_text(
            f'''schema_version = 1
[server]
binary = "fake-server"
host = "127.0.0.1"
port = {port}
startup_timeout_s = 5
request_timeout_s = 5
common_args = []
[models]
target = "target.gguf"
ordinary_draft = "draft.gguf"
packed_head_draft = "packed.gguf"
[evaluation]
prompt_file = "prompts.jsonl"
warmup_requests = 1
repetitions = 5
max_output_tokens = 4
temperature = 0.0
seed = 42
max_draft_tokens = 3
enable_thinking = false
[environment]
FAKE_FAIL = "{int(failure)}"
'''
        )
        return config, port


if __name__ == "__main__":
    unittest.main()
