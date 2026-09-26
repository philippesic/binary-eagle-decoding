"""Synthetic tests for W1Ax diagnostic config freezing and launch recovery."""

import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_w1ax_diagnostics.py")
launcher = load_script("run_w1ax_diagnostics.py")


class FakeProcess:
    next_codes = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.code = self.next_codes.pop(0) if self.next_codes else 0
        self.pid = 12345

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class W1AxDiagnosticJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/benchmark_native_eagle.py").write_text("# synthetic harness\n")
        for name in ("server", "target", "ordinary", "q8", "q4", "w1a16", "w1a8", "w1a4", "w1a1", "w1axdraft"):
            (self.root / name).write_bytes((name + " artifact\n").encode())
        self.config = self.root / "primary.toml"
        self.config.write_text(
            '''schema_version = 1

[server]
binary = "server"
host = "127.0.0.1"
port = 18080
common_args = ["--parallel", "1"]

[models]
target = "target"
ordinary_draft = "ordinary"
draft_q8_0 = "q8"
draft_q4_0 = "q4"
draft_w1a16 = "w1a16"
draft_w1a8 = "w1a8"
draft_w1a4 = "w1a4"
draft_w1a1 = "w1a1"

[w1ax]
draft = "w1axdraft"

[evaluation]
prompt_file = "historical.jsonl"
warmup_requests = 9
repetitions = 8
max_output_tokens = 128
max_draft_tokens = 5
min_draft_probability = 0.0
w1ax_matrix = true
'''
        )
        self.dev = self.root / "development.jsonl"
        self.dev.write_text("".join(json.dumps({"id": f"dev-{i:02d}", "messages": []}) + "\n" for i in range(24)))
        self.context = self.root / "context.jsonl"
        self.context.write_text("".join(json.dumps({"id": f"context-{i:02d}", "messages": []}) + "\n" for i in range(9)))
        self.output = self.root / "runs/diagnostic"
        self.output.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def make_suite(self):
        return prepare.prepare(self.config, self.dev, self.context, self.output, root=self.root)

    def test_preparation_freezes_12_policy_and_2_context_configs(self):
        suite_path = self.make_suite()
        suite = json.loads(suite_path.read_text())
        self.assertEqual(len(suite["configs"]), 14)
        self.assertEqual(
            [entry["name"] for entry in suite["configs"][:2]],
            ["context-cap-32.toml", "context-cap-128.toml"],
        )
        base = tomllib.loads(self.config.read_text())
        policy = [entry for entry in suite["configs"] if entry["prompt_set"] == "qat_development"]
        contexts = [entry for entry in suite["configs"] if entry["prompt_set"] == "context_diagnostic"]
        self.assertEqual(len(policy), 12)
        self.assertEqual(len(contexts), 2)
        matrix = set()
        for entry in policy:
            config = tomllib.loads(Path(entry["path"]).read_text())
            evaluation = config["evaluation"]
            self.assertEqual(config["server"], base["server"])
            self.assertEqual(config["models"], base["models"])
            self.assertEqual(evaluation["prompt_set"], "qat_development")
            self.assertTrue(evaluation["w1ax_policy_diagnostic"])
            self.assertEqual(evaluation["w1ax_matrix"], True)
            self.assertEqual(evaluation["warmup_requests"], 2)
            self.assertEqual(evaluation["repetitions"], 5)
            matrix.add((evaluation["max_draft_tokens"], evaluation["min_draft_probability"]))
        self.assertEqual(matrix, {(d, p) for d in (1, 2, 3, 5) for p in (0.0, 0.1, 0.3)})
        self.assertEqual(
            {tomllib.loads(Path(row["path"]).read_text())["evaluation"]["max_output_tokens"] for row in contexts},
            {32, 128},
        )
        for entry in contexts:
            evaluation = tomllib.loads(Path(entry["path"]).read_text())["evaluation"]
            self.assertTrue(evaluation["w1ax_matrix"])
            self.assertFalse(evaluation["w1ax_policy_diagnostic"])
            self.assertEqual(evaluation["max_draft_tokens"], 5)
            self.assertEqual(evaluation["min_draft_probability"], 0.0)
        self.assertEqual(suite["prompt_sets"]["qat_development"]["count"], 24)
        self.assertEqual(suite["prompt_sets"]["context_diagnostic"]["count"], 9)
        self.assertEqual(suite["model_server_hashes"]["model:w1ax:draft"], prepare.sha256(self.root / "w1axdraft"))
        self.assertEqual(suite["input_files"]["prompt:qat_development"]["sha256"], prepare.sha256(self.dev))
        self.assertEqual(suite["input_files"]["prompt:context_diagnostic"]["sha256"], prepare.sha256(self.context))

    def test_preparation_refuses_reserved_final_prompts(self):
        self.dev.write_text("".join(json.dumps({"id": f"final-{i}", "messages": []}) + "\n" for i in range(24)))
        with self.assertRaisesRegex(ValueError, "reserved/final"):
            self.make_suite()

    def test_launcher_stops_on_failure_then_resumes_without_repeating_successes(self):
        suite_path = self.make_suite()
        launcher.ROOT = self.root
        FakeProcess.next_codes = [0, 17]
        with mock.patch.object(launcher.subprocess, "Popen", FakeProcess):
            code = launcher.run_suite(suite_path, root=self.root, python="python3")
        self.assertEqual(code, 17)
        state_path = self.output / "progress.json"
        state = json.loads(state_path.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(state["configs"]), 2)
        self.assertEqual(state["configs"]["context-cap-32.toml"]["status"], "succeeded")
        self.assertEqual(state["configs"]["context-cap-128.toml"]["status"], "failed")
        FakeProcess.next_codes = [0] * 13
        with mock.patch.object(launcher.subprocess, "Popen", FakeProcess):
            code = launcher.run_suite(suite_path, root=self.root, python="python3")
        self.assertEqual(code, 0)
        resumed = json.loads(state_path.read_text())
        self.assertEqual([row["status"] for row in resumed["configs"].values()], ["succeeded"] * 14)
        self.assertEqual(len(resumed["configs"]["context-cap-32.toml"]["attempts"]), 1)
        attempts = resumed["configs"]["context-cap-128.toml"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]["run_id"], attempts[1]["run_id"])

    def test_launcher_rejects_changed_frozen_configs(self):
        suite_path = self.make_suite()
        entry = json.loads(suite_path.read_text())["configs"][0]
        Path(entry["path"]).write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "changed frozen config"):
            launcher.load_suite(suite_path)

    def test_launcher_rejects_changed_prompt_or_model_inputs(self):
        suite_path = self.make_suite()
        self.dev.write_text(self.dev.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "frozen input hash changed \(prompt:qat_development\)"):
            launcher.load_suite(suite_path)
        self.dev.write_text("".join(json.dumps({"id": f"dev-{i:02d}", "messages": []}) + "\n" for i in range(24)))
        (self.root / "w1axdraft").write_text("replaced model\n")
        with self.assertRaisesRegex(ValueError, "frozen input hash changed \(model:w1ax:draft\)"):
            launcher.load_suite(suite_path)


if __name__ == "__main__":
    unittest.main()
