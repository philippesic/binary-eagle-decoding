"""GPU-free contracts for the bounded native W1Ax activation capture driver."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capture_w1ax_activations", ROOT / "scripts/capture_w1ax_activations.py"
)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class W1AxActivationCaptureTests(unittest.TestCase):
    def test_command_pins_policy_parallelism_and_fp16_kv(self):
        config = {
            "server": {
                "common_args": ["--ctx-size", "2048", "--parallel", "7", "--metrics"],
                "host": "127.0.0.1",
                "port": 19000,
            }
        }
        command = capture.server_command(config, Path("server"), Path("target"), Path("draft"))
        self.assertEqual(command.count("--parallel"), 1)
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "5")
        self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0.0")
        self.assertEqual(command[command.index("--spec-draft-type-k") + 1], "f16")
        self.assertEqual(command[command.index("--spec-draft-type-v") + 1], "f16")

    def test_request_is_bounded_and_requests_raw_ids(self):
        prompt = {"messages": [{"role": "user", "content": "frozen prompt"}]}
        body = capture.request_body(prompt, {"temperature": 0.0, "seed": 42})
        self.assertEqual(body["max_tokens"], 32)
        self.assertTrue(body["return_tokens"])
        self.assertTrue(body["verbose"])
        self.assertFalse(body["stream"])

    def test_token_id_parser_accepts_supported_shapes_and_rejects_booleans(self):
        self.assertEqual(capture.generated_token_ids({"__verbose": {"tokens": [1, 2]}}), [1, 2])
        self.assertEqual(
            capture.generated_token_ids({"tokens": [{"id": 3}, {"token_id": 4}]}), [3, 4]
        )
        self.assertIsNone(capture.generated_token_ids({"tokens": [True, 2]}))
        self.assertIsNone(capture.generated_token_ids({}))

    def test_prompt_loader_and_inventory_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prompt_path = directory / "prompts.jsonl"
            prompt_path.write_text(
                json.dumps(
                    {
                        "id": "historic-01",
                        "messages": [{"role": "user", "content": "frozen"}],
                    }
                )
                + "\n"
            )
            prompts = capture.load_prompts(prompt_path)
            self.assertEqual(prompts[0]["id"], "historic-01")
            artifact = directory / "op-000000000001.bin"
            artifact.write_bytes(b"preserve me")
            before = capture.file_inventory(directory)
            self.assertEqual(before[artifact.name]["bytes"], len(b"preserve me"))
            self.assertEqual(artifact.read_bytes(), b"preserve me")

    def test_capture_inventory_maps_sequences_and_labels_outside_round_ops(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture_dir = directory / "activations"
            capture_dir.mkdir()
            rows = [
                {
                    "schema_version": 1,
                    "sequence": 2,
                    "file": "op-000000000002.bin",
                    "weight_tensor": "ffn.w1a1_packed",
                    "k": 32,
                    "m": 8,
                    "n": 1,
                    "bits": 1,
                    "timestamp_us": 150,
                    "capture_start_us": 150,
                    "capture_end_us": 230,
                },
                {
                    "schema_version": 1,
                    "sequence": 1,
                    "file": "op-000000000001.bin",
                    "weight_tensor": "attn.w1a1_packed",
                    "k": 32,
                    "m": 8,
                    "n": 1,
                    "bits": 1,
                    "timestamp_us": 310,
                    "capture_start_us": 310,
                    "capture_end_us": 350,
                },
            ]
            for row in rows:
                (capture_dir / row["file"]).write_bytes(b"activation")
            (capture_dir / "captures.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            after = capture.capture_inventory(capture_dir)
            before = {"files": {}, "sequences": [], "sidecar_rows": []}
            delta = capture.request_capture_delta(
                before,
                after,
                [{"round_start_us": 200, "round_end_us": 300, "round_index": 0, "task_id": 7}],
            )
            self.assertEqual(delta["sequence_range"], [1, 2])
            self.assertEqual(delta["new_sequences"], [1, 2])
            self.assertEqual(delta["new_filenames"], ["op-000000000001.bin", "op-000000000002.bin"])
            self.assertEqual(
                [event["phase"] for event in delta["capture_events"]],
                ["prefill_or_outside_round", "round"],
            )
            self.assertEqual(delta["capture_events"][1]["round_task_ids"], [7])
            self.assertEqual(
                capture.benchmark_request_id("historic-01"), "rep-00/draft_w1a1/historic-01"
            )

    def test_capture_gate_rejects_missing_and_incomplete_sidecar_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture_dir = directory / "activations"
            capture_dir.mkdir()
            (capture_dir / "op-000000000000.bin").write_bytes(b"raw activation")
            trace_path = directory / "round-trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema": "w1ax_eagle_round_v1",
                        "round_start_us": 10,
                        "round_end_us": 20,
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "captures.jsonl is missing"):
                capture.validate_capture_run(capture_dir, trace_path, [])

            incomplete = {
                "schema_version": 1,
                "sequence": 0,
                "file": "op-000000000000.bin",
                "weight_tensor": "head.w1a1_packed",
                "k": 32,
                "m": 8,
                "n": 1,
                "bits": 1,
                "timestamp_us": 15,
                "capture_start_us": 15,
            }
            (capture_dir / "captures.jsonl").write_text(json.dumps(incomplete) + "\n")
            before = {"files": {}, "sequences": [], "sidecar_rows": []}
            after = capture.capture_inventory(capture_dir)
            delta = capture.request_capture_delta(
                before,
                after,
                [{"round_start_us": 10, "round_end_us": 20, "round_index": 0}],
            )
            with self.assertRaisesRegex(ValueError, "invalid capture_end_us"):
                capture.validate_request_capture_delta(delta)


if __name__ == "__main__":
    unittest.main()
