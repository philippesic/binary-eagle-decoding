"""Synthetic streaming parser and frozen W1Ax configuration checks."""

import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
