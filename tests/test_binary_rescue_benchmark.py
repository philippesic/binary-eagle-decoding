"""GPU-free streaming, policy, cleanup, graph accounting and statistics gates."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_binary_rescue_benchmark as analysis
import run_binary_rescue_benchmark as runner


def frame(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


def events():
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": "ab"}}]},
        {"choices": [{"index": 0, "delta": {"content": "c"}}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "__verbose": {"tokens": [10, 11, 12]},
            "usage": {"completion_tokens": 3},
            "timings": {"predicted_ms": 20, "prompt_ms": 5},
        },
    ]


def stats(**overrides):
    row = dict.fromkeys(
        (
            "calls",
            "launches",
            "captures",
            "recaptures",
            "direct_disabled",
            "direct_incompatible",
            "direct_warmup",
            "warmup_resets",
            "update_reinstantiations",
            "evictions",
            "w1ax_launches",
            "w1ax_captures",
        ),
        0,
    )
    row.update(
        calls=7,
        launches=5,
        captures=2,
        recaptures=1,
        direct_warmup=2,
        w1ax_launches=3,
        w1ax_captures=1,
    )
    row.update(overrides)
    return row


class StreamTests(unittest.TestCase):
    def test_final_verbose_ids_and_content_ttft(self):
        parser = runner.StreamParser()
        for at, event in enumerate(events(), 1):
            for line in frame(event).splitlines(keepends=True):
                parser.feed(line, at / 100)
        parser.feed(b"data: [DONE]\n", 0.06)
        parser.feed(b"\n", 0.06)
        row = parser.result(0.07)
        self.assertEqual(row["generated_token_ids"], [10, 11, 12])
        self.assertEqual(row["ttft_s"], 0.02)
        self.assertEqual(row["inter_token_s"], [])
        self.assertAlmostEqual(row["inter_chunk_s"][0], 0.01)
        self.assertEqual(row["server_predicted_ms"], 20)

    def test_single_id_chunks_and_validation(self):
        parser = runner.StreamParser()
        for at, ids in ((0.1, [10]), (0.2, [11]), (0.3, [12])):
            for line in frame({"tokens": ids}).splitlines(keepends=True):
                parser.feed(line, at)
        parser.feed(b"data: [DONE]\n", 0.4)
        parser.feed(b"\n", 0.4)
        self.assertEqual(len(parser.result(0.5)["inter_token_s"]), 2)
        parser.final_ids = [99]
        with self.assertRaisesRegex(ValueError, "disagree"):
            parser.result(0.5)
        parser.final_ids = [10, 11, 12]
        parser.usage = {"completion_tokens": 4}
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            parser.result(0.5)
        parser.done = False
        with self.assertRaisesRegex(ValueError, "without DONE"):
            parser.result(0.5)

    def test_utf8_tail_retained_in_final_ids_without_invented_itl(self):
        parser = runner.StreamParser()
        for event in (
            {"tokens": [10]},
            {"tokens": [11]},
            {"generated_token_ids": [10, 11, 12], "usage": {"completion_tokens": 3}},
        ):
            for line in frame(event).splitlines(keepends=True):
                parser.feed(line, 0.1)
        parser.feed(b"data: [DONE]\n", 0.2)
        parser.feed(b"\n", 0.2)
        row = parser.result(0.3)
        self.assertEqual(row["generated_token_ids"], [10, 11, 12])
        self.assertEqual(row["inter_token_s"], [])
        self.assertIn("unavailable", row["inter_token_status"])

    def test_real_http_fake_server(self):
        body = b"".join(frame(e) for e in events()) + b"data: [DONE]\n\n"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                row = runner.stream_request(
                    f"http://127.0.0.1:{server.server_port}", {"stream": True}, path
                )
                self.assertEqual(row["completion_tokens"], 3)
                self.assertGreater(row["request_wall_s"], row["ttft_s"])
                self.assertEqual((path / "response.sse").read_bytes(), body)
                self.assertEqual(len(json.loads((path / "events.json").read_text())), 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class ProtocolTests(unittest.TestCase):
    def config(self):
        return {
            "schema_version": 1,
            "variants": {"q4_0": {}, "D": {"env": {"GGML_W1AX_ACT_BITS": "16"}}},
        }

    def test_policy_and_environment(self):
        config = self.config()
        self.assertEqual(runner.validate_config(config), runner.PRIMARY)
        with mock.patch.dict(
            runner.os.environ,
            {
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "W1AX_ROUND_TRACE_JSONL": "bad",
                "EAGLE_STATE_TRACE_JSONL": "bad",
            },
        ):
            env = runner.server_env(config, config["variants"]["D"], "timed", Path("/tmp/run"))
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", env)
        self.assertNotIn("W1AX_ROUND_TRACE_JSONL", env)
        self.assertNotIn("EAGLE_STATE_TRACE_JSONL", env)
        self.assertEqual(env["GGML_CUDA_GRAPH_STATS"], "1")
        self.assertEqual(env["GGML_W1AX_ACT_BITS"], "16")
        config["variants"]["D"]["env"]["GGML_CUDA_DISABLE_GRAPHS"] = "0"
        with self.assertRaises(ValueError):
            runner.validate_config(config)
        config = self.config()
        config["repetitions"] = 4
        with self.assertRaises(ValueError):
            runner.validate_config(config)

    def test_orders_balance_full_cycle(self):
        names = ["a", "b", "c", "d"]
        schedule = runner.orders(names, 8)
        self.assertEqual(schedule[0], schedule[1][::-1])
        for position in range(4):
            self.assertEqual(sorted(row[position] for row in schedule), sorted(names * 2))

    def test_censored_conditional_acceptance(self):
        rows = [
            {
                "n_proposed": p,
                "n_accepted": a,
                "n_emitted": a + 1,
                "proposed_token_ids": list(range(p)),
            }
            for p, a in [(5, 0), (1, 1), (5, 1), (5, 2), (5, 3), (0, 0)]
        ]
        rows.append({"status": "checkpoint_replay"})
        quality = runner.round_quality(rows)
        self.assertEqual(quality["rounds"], 6)
        self.assertEqual(quality["accepted"], 7)
        self.assertEqual(quality["no_proposal_rounds"], 1)
        self.assertEqual(quality["depth"]["2"]["reached"], 3)
        self.assertEqual(quality["depth"]["2"]["conditional_acceptance"], 2 / 3)
        self.assertEqual(quality["depth"]["3"]["conditional_acceptance"], 1 / 2)

    def test_graph_counts_are_verified_not_assumed(self):
        self.assertFalse(runner.graph_stats("")["verified_launches"])
        result = runner.graph_stats("CUDA_GRAPH_STATS " + json.dumps(stats()))
        self.assertTrue(result["verified_launches"])
        self.assertTrue(result["custom_w1ax_launches_verified"])
        self.assertEqual(result["totals"]["recaptures"], 1)
        with self.assertRaisesRegex(ValueError, "sum to calls"):
            runner.graph_stats("CUDA_GRAPH_STATS " + json.dumps(stats(calls=8)))

    def test_digest_mapping_verifies_raw_outputs(self):
        ids = [10, 11, 12]
        digest = runner.output_digest(ids)
        log = (
            "slot print: eagle_request_digest task_id=3 proposal=0123456789abcdef "
            f"output={digest} rounds=2 no_proposal=1 output_tokens=3"
        )
        rows = [{"generated_token_ids": ids}]
        runner.attach_digests(log, rows)
        self.assertEqual(rows[0]["request_digest"]["task_id"], 3)
        self.assertEqual(rows[0]["request_digest"]["no_proposal"], 1)
        self.assertEqual(rows[0]["request_digest"]["completion_ordinal"], 0)
        with self.assertRaisesRegex(ValueError, "raw-output"):
            runner.attach_digests(log, [{"generated_token_ids": [10, 11, 99]}])
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            runner.attach_digests("", rows)

    def test_launched_fake_server_quality_end_to_end(self):
        import socket

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "dev.jsonl"
            prompt.write_text(
                json.dumps({"id": "p", "messages": [{"role": "user", "content": "hi"}]}) + "\n"
            )
            target = root / "target"
            target.write_bytes(b"target")
            binary = root / "fake-server"
            content = b"".join(frame(e) for e in events()) + b"data: [DONE]\n\n"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                + (
                    "import atexit, json, signal, sys\nfrom http.server import BaseHTTP"
                    "RequestHandler, HTTPServer\n"
                )
                + f"BODY = {content!r}\nDIGEST = {runner.output_digest([10, 11, 12])!r}\n"
                + f"GRAPH = {'CUDA_GRAPH_STATS ' + json.dumps(stats())!r}\n"
                + "atexit.register(lambda: print(GRAPH, flush=True))\n"
                + "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                + "class Handler(BaseHTTPRequestHandler):\n"
                + "    task = 0\n"
                + "    def do_GET(self):\n"
                + "        body = b'{\"status\": \"ok\"}' if self.path == '/health' else b''\n"
                + (
                    "        self.send_response(200)\n        self.end_headers()\n      "
                    "  self.wfile.write(body)\n"
                )
                + "    def do_POST(self):\n"
                + "        self.rfile.read(int(self.headers['Content-Length']))\n"
                + "        Handler.task += 1\n"
                + (
                    "        print(f'eagle_request_digest task_id={Handler.task} propo"
                    "sal=0123456789abcdef output={DIGEST} rounds=0 output_tokens=3', f"
                    "lush=True)\n"
                )
                + (
                    "        self.send_response(200)\n        self.send_header('Content"
                    "-Type','text/event-stream')\n"
                )
                + (
                    "        self.send_header('Content-Length',str(len(BODY)))\n       "
                    " self.end_headers()\n        self.wfile.write(BODY)\n"
                )
                + "    def log_message(self, *_): pass\n"
                + (
                    "HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port')+1]"
                    ")), Handler).serve_forever()\n"
                )
            )
            binary.chmod(0o755)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            config = {
                "schema_version": 1,
                "variants": {"q4_0": {}},
                "binary": str(binary),
                "target": str(target),
                "prompts": str(prompt),
                "port": port,
                "diagnostic": {
                    "label": "cpu-test",
                    "prompts": str(prompt),
                    "prompts_sha256": runner.base.sha256(prompt),
                },
            }
            sampler = mock.MagicMock()
            sampler.__enter__.return_value = sampler
            sampler.summary = {}
            with (
                mock.patch.object(runner, "TARGET_SHA256", runner.base.sha256(target)),
                mock.patch.object(runner.base, "environment_manifest", return_value={}),
                mock.patch.object(runner, "GPUSampler", return_value=sampler),
                mock.patch.object(runner.base, "gpu_snapshot", return_value={}),
            ):
                runner.run(config, "quality", root / "result", diagnostic=True)
            manifest = json.loads((root / "result" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["records"]), 3)
            self.assertEqual(manifest["blocks"][0]["graph_status"], "verified_launches")
            self.assertEqual(manifest["records"][-1]["request_digest"]["task_id"], 3)
            self.assertTrue(runner.base.available_port("127.0.0.1", port))

    def test_failed_request_stops_server_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "dev.jsonl"
            prompt.write_text(
                json.dumps({"id": "p", "messages": [{"role": "user", "content": "hi"}]}) + "\n"
            )
            target = root / "target"
            target.write_bytes(b"target")
            binary = root / "binary"
            binary.write_bytes(b"binary")
            config = self.config()
            config.update(
                binary=str(binary),
                target=str(target),
                prompts=str(prompt),
                diagnostic={
                    "label": "cpu-test",
                    "prompts": str(prompt),
                    "prompts_sha256": runner.base.sha256(prompt),
                },
            )
            proc = mock.Mock()
            sampler = mock.MagicMock()
            sampler.__enter__.return_value = sampler
            sampler.summary = {}
            with (
                mock.patch.object(runner, "TARGET_SHA256", runner.base.sha256(target)),
                mock.patch.object(runner.base, "environment_manifest", return_value={}),
                mock.patch.object(runner, "GPUSampler", return_value=sampler),
                mock.patch.object(runner.subprocess, "Popen", return_value=proc),
                mock.patch.object(runner.base, "wait_ready"),
                mock.patch.object(runner.base, "gpu_snapshot", return_value={}),
                mock.patch.object(runner.base, "metrics_text", return_value=None),
                mock.patch.object(runner.base, "stop_server", return_value={}) as stop,
                mock.patch.object(
                    runner, "stream_request", side_effect=ValueError("bad alignment")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "bad alignment"):
                    runner.run(config, "timed", root / "result", diagnostic=True)
            stop.assert_called_once_with(proc)
            manifest = json.loads((root / "result" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("server_stop", manifest["blocks"][0])


class AnalysisTests(unittest.TestCase):
    def rows(self, multiplier=1):
        return [
            {
                "prompt_id": p,
                "repetition": rep,
                "completion_tokens": 10,
                "request_wall_s": wall * multiplier,
                "server_predicted_ms": wall * 900 * multiplier,
                "server_prompt_ms": 10,
                "ttft_s": 0.02,
                "generated_token_ids": [1, 2],
                "speculative": {"accepted": 2, "proposed": 10, "rounds": 4},
            }
            for p, wall in [("slow", 2), ("fast", 1)]
            for rep in range(5)
        ]

    def test_paired_aggregate_prompt_cluster_interval(self):
        reference = self.rows()
        candidate = self.rows(0.5)
        result = analysis.paired_comparison(candidate, reference, samples=100)
        self.assertEqual(result["paired_requests"], 10)
        self.assertEqual(result["paired_prompts"], 2)
        self.assertEqual(result["client_request_tok_s_ratio"], 2)
        self.assertEqual(result["client_request_tok_s_ratio_ci95"], [2, 2])
        self.assertAlmostEqual(analysis.aggregate(reference)["client_request_tok_s"], 100 / 15)
        self.assertEqual(analysis.aggregate(reference)["speculative"]["accepted_per_round"], 0.5)

    def test_duplicate_pair_and_missing_timings(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            analysis.paired_comparison(rows + rows[:1], rows, samples=2)
        rows[0]["server_predicted_ms"] = None
        self.assertIsNone(analysis.aggregate(rows)["server_decode_tok_s"])
        self.assertEqual(analysis.divergence([1, 2], [1, 3])["position"], 1)
        self.assertEqual(analysis.distribution([1, 2, 3, 4, 5])["p95"], 5)

    def test_behavior_gate_does_not_invent_proposal_parity(self):
        row = self.rows()[0] | {"variant": "D", "directory": "/nonexistent"}
        result = analysis.behavior_compare([row], [row])
        self.assertTrue(result["outputs_and_counts_verified"])
        self.assertFalse(result["exact_proposals_verified"])
        self.assertIsNone(result["requests"][0]["proposal_ids_match"])


if __name__ == "__main__":
    unittest.main()
