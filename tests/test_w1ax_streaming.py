"""Synthetic streaming parser and frozen W1Ax configuration checks."""

import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_w1ax_streaming as streaming  # noqa: E402


class SSEParserTests(unittest.TestCase):
    def test_stream_request_measures_first_content_and_http_eof(self):
        class Response:
            status = 200
            headers = {"Content-Type": "text/event-stream; charset=utf-8"}

            def __init__(self):
                self.lines = iter((
                    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                    b"\n",
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                    b"\n",
                    b"data: [DONE]\n",
                    b"\n",
                ))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def readline(self):
                return next(self.lines, b"")

        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "raw.sse"
            times = iter((10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8))
            with patch.object(streaming.urllib.request, "urlopen", return_value=Response()), patch.object(
                streaming.time, "perf_counter", side_effect=lambda: next(times)
            ):
                result = streaming.stream_request("http://localhost/v1/chat/completions", {"stream": True}, 3, raw_path)
            self.assertAlmostEqual(result["ttft_s"], 0.4)
            self.assertAlmostEqual(result["request_wall_s"], 0.8)
            self.assertIn(b"data: [DONE]", raw_path.read_bytes())
            self.assertIn("no raw-ID parity", result["generated_token_ids_status"])

    def test_comments_multiline_crlf_and_content_boundary(self):
        parser = streaming.SSEParser()
        lines = (
            (b": keepalive\r\n", 0.1),
            (b"\r\n", 0.2),
            (b'data: {"id":"sample","choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}\r\n', 0.3),
            (b"\r\n", 0.4),
            (b'data: {"id":"sample","choices":[{"index":0,\r\n', 0.5),
            (b'data: "delta":{"content":"hello"}}]}\r\n', 0.6),
            (b"\r\n", 0.7),
            (b'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n', 0.8),
            (b"\n", 0.9),
            (b"data: [DONE]\n", 1.0),
            (b"\n", 1.1),
        )
        for line, time in lines:
            parser.feed_line(line, time)
        parser.finish(1.2)
        self.assertEqual(parser.first_content_time, 0.7)
        self.assertEqual("".join(parser.text_parts), "hello world")
        self.assertEqual(parser.finish_reason, "stop")
        self.assertEqual(parser.response_ids, {"sample"})
        self.assertEqual(parser.events, 3)

    def test_missing_done_or_content_and_bad_event_fail(self):
        no_done = streaming.SSEParser()
        no_done.feed_line(b'data: {"choices":[{"delta":{"content":"x"}}]}\n', 1)
        with self.assertRaisesRegex(ValueError, "without \[DONE\]"):
            no_done.finish(2)
        no_content = streaming.SSEParser()
        no_content.feed_line(b"data: [DONE]\n", 1)
        with self.assertRaisesRegex(ValueError, "no generated content"):
            no_content.finish(2)
        bad = streaming.SSEParser()
        bad.feed_line(b"data: {broken}\n", 1)
        with self.assertRaisesRegex(ValueError, "invalid SSE JSON"):
            bad.finish(2)

    def test_error_event_and_wrong_choice_index(self):
        parser = streaming.SSEParser()
        parser.feed_line(b'data: {"choices":[{"index":1,"delta":{"content":"ignored"}},{"index":0,"delta":{"content":[{"text":"ok"}]}}]}\n', 1)
        parser.feed_line(b"\n", 1.1)
        self.assertEqual(parser.first_content_time, 1.1)
        self.assertEqual(parser.text_parts, ["ok"])
        error = streaming.SSEParser()
        error.feed_line(b'data: {"error":{"message":"failed"}}\n', 1)
        with self.assertRaisesRegex(RuntimeError, "SSE server error"):
            error.finish(2)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((ROOT / "configs/native_benchmark_w1ax.toml").read_text())

    def test_latency_summary_counts_and_nearest_rank_p95(self):
        rows = [{"ttft_s": value, "request_wall_s": 2 * value} for value in (5, 1, 4, 2, 3)]
        summary = streaming.latency_summary(rows)
        self.assertEqual(summary, {
            "sample_count": 5,
            "ttft_median_s": 3,
            "ttft_p95_s": 5,
            "ttft_max_s": 5,
            "request_wall_median_s": 6,
            "request_wall_p95_s": 10,
            "request_wall_max_s": 10,
        })
        with self.assertRaisesRegex(ValueError, "empty latency sample"):
            streaming.latency_summary([])

    def test_frozen_matrix_selectors_and_prompts(self):
        variants, specs = streaming.validate_config(self.config)
        self.assertEqual(len(variants), 8)
        self.assertEqual({spec["draft"] for spec in specs.values()}, {self.config["w1ax"]["draft"]})
        self.assertEqual(streaming.PROMPT_IDS, (
            "context-prose-short", "context-code-medium", "context-reasoning-long"
        ))
        self.assertEqual(streaming.CAPS, (32, 128))
        self.assertEqual((streaming.WARMUPS, streaming.REPETITIONS), (2, 5))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            from generate_w1ax_context_prompts import build_rows
            path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in build_rows()))
            self.assertEqual([row["id"] for row in streaming.selected_prompts(path)], list(streaming.PROMPT_IDS))
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "frozen SHA256"):
                streaming.selected_prompts(path)

    def test_changed_policy_or_precision_rejected(self):
        bad = {**self.config, "evaluation": {**self.config["evaluation"], "max_draft_tokens": 3}}
        with self.assertRaisesRegex(ValueError, "primary matrix"):
            streaming.validate_config(bad)
        bad = {**self.config, "evaluation": {**self.config["evaluation"], "enable_thinking": True}}
        with self.assertRaisesRegex(ValueError, "greedy non-thinking"):
            streaming.validate_config(bad)
        bad = {**self.config, "precision": {**self.config["precision"], "target_kv": "q8_0"}}
        with self.assertRaisesRegex(ValueError, "F16 KV"):
            streaming.validate_config(bad)
        bad = {**self.config, "server": {**self.config["server"], "common_args": ["--parallel", "2"]}}
        with self.assertRaisesRegex(ValueError, "concurrency one"):
            streaming.validate_config(bad)


class GPUTelemetryTests(unittest.TestCase):
    raw = (
        "2026/09/25 12:00:00.000, GPU-one, RTX 2080 Ti, 11264, 8000, 62, 190.5, 1710, 7000\n"
        "2026/09/25 12:00:01.000, GPU-one, RTX 2080 Ti, 11264, 8100, 65, 200.0, 1695, 7000\n"
        "2026/09/25 12:00:02.000, GPU-one, RTX 2080 Ti, 11264, 8050, N/A, [Not Supported], 1700, 7000\n"
    )
    loaded = {
        "exit_code": 0,
        "command": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.draw"],
        "stdout": "RTX 2080 Ti, GPU-one, 999, 11264 MiB, 7900 MiB, 1710 MHz, 190 W\n",
    }

    def test_sampled_peak_loaded_memory_and_unavailable_values(self):
        result = streaming.summarize_gpu_telemetry(self.raw + "diagnostic text\n", self.loaded)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["rejected_raw_lines"], 1)
        device = result["devices"][0]
        self.assertEqual(device["loaded_memory_used_mib"], 7900)
        self.assertEqual(device["memory_used_sampled_peak_mib"], 8100)
        self.assertEqual(device["temperature_c"], {"available_samples": 2, "sampled_min": 62, "sampled_max": 65})
        self.assertEqual(device["power_w"]["sampled_max"], 200)
        self.assertEqual(device["sm_clock_mhz"]["sampled_min"], 1695)
        self.assertIn("not an allocator high-water", result["peak_semantics"])
        self.assertIsNone(streaming.numeric_gpu_value("N/A"))

    def test_poller_stops_on_success_and_exception_with_wsl_executable(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                child = Mock(pid=123, returncode=-15)
                child.poll.return_value = None

                def start_child(_command, **kwargs):
                    kwargs["stdout"].write(self.raw.encode())
                    return child

                with patch.object(streaming.base, "executable_path", return_value="/usr/lib/wsl/lib/nvidia-smi"), patch.object(
                    streaming.subprocess, "Popen", side_effect=start_child
                ) as popen, patch.object(streaming.base, "stop_server") as stop:
                    try:
                        with streaming.GPUSampler(directory) as sampler:
                            sampler.loaded_snapshot = self.loaded
                            if fail:
                                raise RuntimeError("synthetic request failure")
                    except RuntimeError:
                        self.assertTrue(fail)
                    stop.assert_called_once_with(child)
                    sampler.stop()  # idempotent cleanup must not signal a reused PID
                    stop.assert_called_once_with(child)
                    self.assertTrue(sampler.output.closed)
                    self.assertTrue(sampler.errors.closed)
                    self.assertEqual(popen.call_args.args[0][0], "/usr/lib/wsl/lib/nvidia-smi")
                    self.assertIn("--loop=1", popen.call_args.args[0])
                    self.assertTrue(popen.call_args.kwargs["start_new_session"])
                summary = json.loads((directory / "gpu-telemetry-summary.json").read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["sample_count"], 3)
                self.assertEqual((directory / "gpu-telemetry.csv").read_text(), self.raw)

    def test_missing_utility_and_early_exit_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(streaming.base, "executable_path", return_value=None), patch.object(
                streaming.subprocess, "Popen"
            ) as popen:
                with streaming.GPUSampler(directory) as sampler:
                    pass
                popen.assert_not_called()
            self.assertEqual(sampler.summary["status"], "unavailable")
            self.assertEqual(sampler.summary["sample_count"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            child = Mock(pid=123, returncode=1)
            child.poll.return_value = 1
            with patch.object(streaming.base, "executable_path", return_value="nvidia-smi"), patch.object(
                streaming.subprocess, "Popen", return_value=child
            ), patch.object(streaming.base, "stop_server") as stop:
                with streaming.GPUSampler(Path(temporary)) as sampler:
                    pass
                stop.assert_called_once_with(child)
            self.assertEqual(sampler.summary["status"], "poller_exited_early")
            self.assertEqual(sampler.summary["exit_code_before_stop"], 1)


if __name__ == "__main__":
    unittest.main()
