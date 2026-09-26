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


if __name__ == "__main__":
    unittest.main()
