"""Native request timing contract; synthetic HTTP measurements are not GPU evidence."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import a8_native_request_metrics as metrics  # noqa: E402
import run_binary_head_capture as capture  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402
from w1ax_capture_provider import TARGET_GGUF_SHA256  # noqa: E402


def response(tokens=2):
    return {
        "usage": {"completion_tokens": tokens, "prompt_tokens": 9},
        "timings": {"predicted_n": tokens, "predicted_ms": 100, "prompt_ms": 20},
        "generated_token_ids": list(range(tokens)),
        "choices": [{"finish_reason": "length", "message": {"content": "answer"}}],
    }


class RequestMetricsTests(unittest.TestCase):
    def test_response_wall_excludes_metrics_polling(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(metrics, "_metrics", side_effect=[None, None]),
            patch.object(
                metrics.benchmark, "request_json", return_value=(200, json.dumps(response()))
            ),
            patch.object(metrics.time, "perf_counter", side_effect=[10, 10.5]),
        ):
            row = metrics.measure_request(
                "http://test", {}, Path(temp) / "r", deadline=time.monotonic() + 20, stop_file=None
            )
            self.assertEqual(row["request_wall_s"], 0.5)
            self.assertEqual(row["request_tokens_per_s"], 4)
            self.assertEqual(row["decode_tokens_per_s"], 20)
            self.assertEqual(row["prefill_server_s"], 0.02)
            self.assertIsNone(row["time_to_first_token_s"])
            self.assertEqual(row["generated_token_ids"], [0, 1])

    def test_no_fabricated_count_when_response_ids_disagree(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(metrics, "_metrics", return_value=None),
            patch.object(
                metrics.benchmark,
                "request_json",
                return_value=(200, json.dumps({**response(), "generated_token_ids": [0]})),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "actual positive output count"):
                metrics.measure_request(
                    "http://test",
                    {},
                    Path(temp) / "r",
                    deadline=time.monotonic() + 20,
                    stop_file=None,
                )
            self.assertTrue((Path(temp) / "r/response.json").exists())

    def test_environment_has_no_capture_trace_or_inherited_precision_hooks(self):
        sources = {
            "native_runtime": {"ld_library_path": "/pinned"},
            "evaluation_env": metrics.EVALUATION_ENV,
        }
        with patch.dict(
            os.environ,
            {
                "EAGLE_CAPTURE_TARGET_FEATURES": "1",
                "GGML_W1AX_ACT_BITS": "1",
                "W1AX_ROUND_TRACE_JSONL": "x",
            },
        ):
            env = metrics.clean_environment(sources, "A8")
            self.assertEqual(env["GGML_W1AX_ACT_BITS"], "8")
            self.assertEqual(env["GGML_CUDA_DISABLE_GRAPHS"], "1")
            self.assertEqual(env["GGML_EAGLE_SHARED_PACK"], "1")
            self.assertEqual(env["GGML_EAGLE_PRUNE_UNUSED_HEAD"], "1")
            self.assertFalse(any(k.startswith(("EAGLE_", "W1AX_")) for k in env))
            self.assertNotIn("GGML_W1AX_ACT_BITS", metrics.clean_environment(sources, "Q4_0"))

    def test_aggregate_is_count_over_sum_time_not_mean_rate(self):
        rows = []
        for variant, times in (("Q4_0", [1, 3]), ("A8", [0.5, 1.5]), ("target_only", [2, 6])):
            for rep, wall in enumerate(times):
                row = metrics.benchmark.extract_record(
                    response(2 if rep == 0 else 6),
                    wall,
                    {"accepted": 1, "proposed": 2, "rounds": 1},
                )
                row.update(variant=variant, repetition=rep, prompt_id="x", prefill_server_s=0.02)
                rows.append(row)
        result = metrics._summary(rows, True)
        self.assertEqual(result["variants"]["Q4_0"]["request_tokens_per_s"], 2)
        self.assertEqual(result["speedup_vs_q4_0"]["request_tokens_per_s"], 2)
        self.assertEqual(result["variants"]["A8"]["completion_tokens"], 8)
        self.assertEqual(result["speedup_vs_no_speculation"]["request_tokens_per_s"], 4)
        self.assertEqual(result["generated_token_id_matches_q4_0"]["A8"]["matched_sequences"], 2)
        partial = metrics._summary(rows, False)
        self.assertIsNone(partial["variants"])
        self.assertIsNone(partial["speedup_vs_q4_0"])

    def test_deadline_and_stop_are_not_new_budgets(self):
        with self.assertRaises(TimeoutError):
            metrics.remaining(time.monotonic() - 1)
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp) / "STOP"
            stop.touch()
            with self.assertRaises(InterruptedError):
                metrics.remaining(time.monotonic() + 10, stop)
        self.assertLessEqual(metrics.remaining(time.monotonic() + 0.25), 0.25)

    def test_target_only_removes_every_draft_control_with_same_target_policy(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            binary=Path("binary"), target=Path("target"), port=18092, mode="recurrent-train"
        )
        drafts = {"A8": Path("a8"), "Q4_0": Path("q4")}
        command = metrics.request_command(args, "target_only", drafts)
        self.assertNotIn("-md", command)
        self.assertFalse(any("spec-draft" in argument for argument in command))
        self.assertEqual(command[command.index("--spec-type") + 1], "none")
        for key, expected in (
            ("--cache-type-k", "f16"),
            ("--cache-type-v", "f16"),
            ("--ctx-size", "2048"),
            ("--parallel", "1"),
        ):
            self.assertEqual(command[command.index(key) + 1], expected)
        for flag in ("--no-context-shift", "--no-cache-prompt", "--jinja", "--metrics"):
            self.assertIn(flag, command)
        self.assertEqual(
            metrics.request_command(args, "A8", drafts),
            capture.server_command(args, {"draft": "a8"}),
        )
        self.assertNotIn(
            "GGML_W1AX_ACT_BITS",
            metrics.clean_environment(
                {
                    "native_runtime": {"ld_library_path": "/pinned"},
                    "evaluation_env": metrics.EVALUATION_ENV,
                },
                "target_only",
            ),
        )

    def test_owned_cleanup_kills_only_owned_group(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
        )
        other = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
        )
        try:
            proof = metrics._stop_owned(process, time.monotonic() + 3)
            self.assertTrue(proof["stopped"])
            self.assertTrue(proof["process_group_gone"])
            self.assertIsNone(other.poll())
        finally:
            metrics._stop_owned(process, time.monotonic() + 3)
            metrics._stop_owned(other, time.monotonic() + 3)

    def test_frozen_command_and_request_policy(self):
        from types import SimpleNamespace

        command = capture.server_command(
            SimpleNamespace(
                binary=Path("binary"), target=Path("target"), port=18092, mode="recurrent-train"
            ),
            {"draft": "draft"},
        )
        for option in (
            "--cache-type-k",
            "--cache-type-v",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
        ):
            self.assertEqual(command[command.index(option) + 1], "f16")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "5")
        self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0")
        self.assertIn("--no-cache-prompt", command)
        self.assertIn("--no-spec-draft-backend-sampling", command)


class FullContractTests(unittest.TestCase):
    def fixture(self, folder):
        sources = {
            "binary": str(folder / "binary"),
            "target_gguf": str(folder / "target"),
            "q4_0_draft": str(folder / "q4"),
            "native_runtime": {"ld_library_path": str(folder)},
            "evaluation_native_commit": metrics.NATIVE_COMMIT,
            "evaluation_env": metrics.EVALUATION_ENV,
            "sha256": {"target_gguf": TARGET_GGUF_SHA256, "q4_0_draft": stages.Q4_0_GGUF_SHA256},
        }
        for name in ("binary", "target", "q4", "student"):
            (folder / name).write_text(name)
        prompts = folder / "fixed-development.jsonl"
        prompts.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"{('magicoder', 'dolly', 'gsm8k')[i % 3]}:line-{i:06d}-index-{i}",
                        "messages": [{"role": "user", "content": "x"}],
                    }
                )
                + "\n"
                for i in range(24)
            )
        )
        return sources, prompts, folder / "student"

    def run_synthetic(self, folder, *, fail=None):
        sources, prompts, draft = self.fixture(folder)
        process = Mock(pid=100001, returncode=-9)
        process.poll.return_value = None
        spawned = []

        def spawn(command, **kwargs):
            spawned.append((command, kwargs["env"]))
            kwargs["stdout"].write(
                (
                    metrics.benchmark.W1AX_LOADER_MARKER
                    + "\n"
                    + metrics.benchmark.W1AX_CUDA_MARKERS["8"]
                ).encode()
            )
            return process

        calls = []

        def measure(url, body, directory, **kwargs):
            calls.append((directory, kwargs["deadline"], body))
            if fail:
                raise fail
            row = metrics.benchmark.extract_record(
                response(), 0.1, {"accepted": 1, "proposed": 2, "rounds": 1}
            )
            row["prefill_server_s"] = 0.02
            return row

        deadline = time.monotonic() + 30
        with (
            patch.object(metrics, "DEVELOPMENT_PROMPTS_SHA256", metrics.benchmark.sha256(prompts)),
            patch.object(stages, "verify_sources"),
            patch.object(metrics.benchmark, "available_port", return_value=True),
            patch.object(metrics.subprocess, "Popen", side_effect=spawn),
            patch.object(metrics, "_wait_ready"),
            patch.object(capture, "verify_mapped_runtime", return_value={"verified": True}),
            patch.object(
                metrics, "_stop_owned", return_value={"stopped": True, "process_group_gone": True}
            ),
            patch.object(metrics, "measure_request", side_effect=measure),
        ):
            result = metrics.measure_a8_requests(
                sources, prompts, draft, folder / "timing", deadline=deadline
            )
        return result, spawned, calls, deadline

    def test_full_five_repeat_contract_alternates_and_preserves_all24_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            result, spawned, calls, deadline = self.run_synthetic(Path(temp))
            self.assertTrue(result["complete"])
            self.assertEqual(len(spawned), 15)
            self.assertEqual(len(result["records"]), 360)
            self.assertEqual(len(calls), 375)
            self.assertTrue(all(call[1] == deadline for call in calls))
            self.assertEqual(
                result["orders"][:2], [["Q4_0", "A8", "target_only"], ["A8", "target_only", "Q4_0"]]
            )
            self.assertEqual(result["variants"]["A8"]["requests"], 120)
            self.assertEqual(result["variants"]["target_only"]["requests"], 120)
            self.assertIsNone(result["drafts_sha256"]["target_only"])
            for repeat, order in enumerate(result["orders"]):
                self.assertEqual(order.index("Q4_0") < order.index("A8"), repeat % 2 == 0)
            self.assertTrue(
                all(cell["server_stop"]["process_group_gone"] for cell in result["servers"])
            )
            self.assertTrue((Path(temp) / "timing/manifest.json").exists())
            self.assertFalse(
                any(k.startswith(("EAGLE_", "W1AX_")) for _, env in spawned for k in env)
            )

    def test_timeout_retains_partial_artifacts_and_no_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            result, spawned, calls, _ = self.run_synthetic(
                Path(temp), fail=TimeoutError("deadline")
            )
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "deadline_exhausted")
            self.assertIsNone(result["speedup_vs_q4_0"])
            self.assertEqual(len(spawned), 1)
            self.assertEqual(len(calls), 1)
            self.assertTrue(result["servers"][0]["server_stop"]["process_group_gone"])
            self.assertTrue((Path(temp) / "timing/rep-00/Q4_0/server.log").exists())

    def test_fixed_prepared24_hash_and_original_dataset_ids_are_accepted(self):
        self.assertEqual(
            metrics.DEVELOPMENT_PROMPTS_SHA256,
            "131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081",
        )
        with tempfile.TemporaryDirectory() as temp:
            sources, prompts, _ = self.fixture(Path(temp))
            # Local synthetic bytes model the original dataset IDs, never change production pin.
            with patch.object(
                metrics, "DEVELOPMENT_PROMPTS_SHA256", metrics.benchmark.sha256(prompts)
            ):
                rows = metrics.development_prompt_rows(sources, prompts)
            self.assertEqual(len(rows), 24)
            self.assertEqual(
                {row["id"].split(":")[0] for row in rows}, {"magicoder", "dolly", "gsm8k"}
            )

    def test_changed_fixed_development_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            sources, prompts, _ = self.fixture(Path(temp))
            digest = metrics.benchmark.sha256(prompts)
            prompts.write_text(prompts.read_text() + "\n")
            with patch.object(metrics, "DEVELOPMENT_PROMPTS_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "content hash changed"):
                    metrics.development_prompt_rows(sources, prompts)

    def test_duplicates_and_final_ids_are_rejected_even_with_matching_hash(self):
        for invalid in ("duplicate", "final", "sealed"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                sources, prompts, _ = self.fixture(Path(temp))
                rows = [json.loads(line) for line in prompts.read_text().splitlines()]
                rows[-1]["id"] = (
                    rows[0]["id"] if invalid == "duplicate" else f"{invalid}:line-00001"
                )
                prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with patch.object(
                    metrics, "DEVELOPMENT_PROMPTS_SHA256", metrics.benchmark.sha256(prompts)
                ):
                    with self.assertRaisesRegex(ValueError, "unique unsealed"):
                        metrics.development_prompt_rows(sources, prompts)

    def test_final_path_rejected_before_reading_content(self):
        with patch.object(metrics.benchmark, "sha256") as digest:
            with self.assertRaisesRegex(ValueError, "sealed/final"):
                metrics.development_prompt_rows({}, Path("sealed-final.jsonl"))
            digest.assert_not_called()

    def test_historical_native_binary_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sources, prompts, draft = self.fixture(folder)
            sources["sha256"]["binary"] = metrics.HISTORICAL_CAPTURE_BINARY
            with (
                patch.object(stages, "verify_sources"),
                patch.object(metrics.subprocess, "Popen") as spawn,
            ):
                with self.assertRaisesRegex(ValueError, "current9e2"):
                    metrics.measure_a8_requests(
                        sources, prompts, draft, folder / "timing", deadline=time.monotonic() + 30
                    )
                spawn.assert_not_called()

    def test_short_contract_refused_before_model_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sources, prompts, draft = self.fixture(folder)
            for options in ({"repetitions": 4}, {"warmup_requests": 0}, {"tokens": 64}):
                with self.assertRaises(ValueError):
                    metrics.measure_a8_requests(
                        sources,
                        prompts,
                        draft,
                        folder / "timing",
                        deadline=time.monotonic() + 30,
                        **options,
                    )


if __name__ == "__main__":
    unittest.main()
