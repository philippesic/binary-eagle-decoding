"""GPU-free contracts for the supervised binary scale screen runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_binary_scale_screen as screen  # noqa: E402

from scripts.analyze_binary_scale_screen import state_audit  # noqa: E402

REAL_READ_JSONL = screen.read_jsonl
REAL_POPEN = subprocess.Popen


class BinaryScaleQualityTests(unittest.TestCase):
    def test_round_metrics_filter_replays_and_keep_terminal_acceptance(self):
        rows = [
            {
                "status": "ok",
                "n_proposed": 3,
                "n_accepted": 2,
                "n_emitted": 1,
                "proposed_token_ids": [10, 11, 12],
            },
            # Replay rows may carry stale/incomplete counters and must not affect metrics.
            {"status": "checkpoint_replay", "n_proposed": 9, "n_accepted": 9},
            {
                "status": "ok",
                "n_proposed": 1,
                "n_accepted": 0,
                "n_emitted": 1,
                "proposed_token_ids": [20],
            },
        ]

        result = screen.quality(rows)

        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["emitted"], 2)
        self.assertEqual(result["accepted_emitted"], 1)
        self.assertEqual(result["accepted_not_emitted"], 1)
        self.assertEqual(result["proposed"], 4)
        self.assertEqual(result["accepted_per_round"], 1.0)
        self.assertEqual(result["depth"]["1"], {"survived": 1, "eligible": 2, "rate": 0.5})
        self.assertEqual(result["depth"]["2"], {"survived": 1, "eligible": 1, "rate": 1.0})
        self.assertEqual(result["depth"]["3"], {"survived": 0, "eligible": 1, "rate": 0.0})
        self.assertIsNone(result["depth"]["4"]["rate"])

    def test_quality_rejects_count_and_id_mismatches(self):
        invalid = [
            {"status": "ok", "n_proposed": 2, "n_accepted": 3, "proposed_token_ids": [1, 2]},
            {"status": "ok", "n_proposed": 2, "n_accepted": 1, "proposed_token_ids": [1]},
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                screen.quality([row])


class BinaryScaleStateAuditTests(unittest.TestCase):
    @staticmethod
    def _accept(accepted, proposed, position, digest):
        return {
            "schema": "eagle_state_v1",
            "event": "accept",
            "seq_id": 4,
            "accepted": accepted,
            "verify_rows": proposed + 1,
            "selected_row": accepted,
            "verify_pos_first": position - accepted,
            "position": position,
            "selected_hash": digest,
            "pending_hash": digest,
        }

    @staticmethod
    def _seed(position, digest, kv_max_after=None):
        return {
            "schema": "eagle_state_v1",
            "event": "seed",
            "seq_id": 4,
            "position": position,
            "token": 50,
            "kv_max_after": position - 1 if kv_max_after is None else kv_max_after,
            "pending_hash": digest,
        }

    def test_maps_zero_partial_full_to_actual_rounds_and_leaves_terminal_unpaired(self):
        rounds = [
            {"status": "ok", "n_proposed": 3, "n_accepted": 0},
            {"status": "ok", "n_proposed": 2, "n_accepted": 1},
            {"status": "ok", "n_proposed": 2, "n_accepted": 2},
            {"status": "checkpoint_replay", "n_proposed": 5, "n_accepted": 5},
        ]
        events = [
            self._accept(0, 3, 10, "a"),
            self._seed(10, "a"),
            self._accept(1, 2, 11, "b"),
            self._seed(11, "b"),
            self._accept(2, 2, 12, "c"),  # terminal decision has no following seed
        ]

        audit = state_audit(events, rounds)

        self.assertTrue(audit["passed"], audit["errors"])
        self.assertEqual(audit["counts"]["zero"], 1)
        self.assertEqual(audit["counts"]["partial"], 1)
        self.assertEqual(audit["counts"]["full"], 1)
        self.assertEqual(audit["counts"]["completed_pairs"], 2)
        self.assertEqual(audit["counts"]["paired_zero"], 1)
        self.assertEqual(audit["counts"]["paired_partial"], 1)
        self.assertEqual(audit["counts"]["paired_full"], 0)

    def test_empty_state_trace_is_not_a_pass(self):
        audit = state_audit([])
        self.assertFalse(audit["observed"])
        self.assertFalse(audit["passed"])

    def test_rejects_state_hash_mismatch_and_untruncated_cache(self):
        accept = self._accept(1, 2, 11, "selected")
        accept["pending_hash"] = "pending"
        events = [accept, self._seed(11, "different", kv_max_after=11)]

        audit = state_audit(events, [{"status": "ok", "n_proposed": 2, "n_accepted": 1}])

        self.assertFalse(audit["passed"])
        self.assertIn("selected/pending state differs", audit["errors"])
        self.assertIn("draft cache not truncated before seed", audit["errors"])
        self.assertIn("next seed differs from retained acceptance state", audit["errors"])


class BinaryScaleRunnerTests(unittest.TestCase):
    selected = sorted(screen.SELECTED)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pid_files: list[Path] = []
        self.trace_rows: list[dict] = []
        self.response_mode = "valid"
        self.missing_capture_name: str | None = None
        self.calls = 0
        self.capture_dir: Path | None = None
        self.state_rows: list[dict] = []

        # This executable stands in for llama-server. It owns a child so the test
        # proves stop_server tears down the complete process group on success/errors.
        self.binary = self.root / "fake-server"
        self.binary.write_text(
            "#!/usr/bin/env python3\n"
            "import os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "Path(os.environ['FAKE_PID_FILE']).write_text(str(child.pid))\n"
            "while True: time.sleep(1)\n"
        )
        self.binary.chmod(0o755)
        self.target = self.root / "target.gguf"
        self.draft = self.root / "draft.gguf"
        self.target.write_bytes(b"target")
        self.draft.write_bytes(b"draft")
        self.prompts = self.root / "prompts.jsonl"
        self.prompts.write_text(
            json.dumps({"id": "prompt-01", "messages": [{"role": "user", "content": "hi"}]}) + "\n"
        )
        self.variants = self.root / "variants.json"
        self.variants.write_text(json.dumps({"candidate": {"draft": str(self.draft)}}))

        def popen(cmd, **kwargs):
            env = kwargs["env"]
            self.capture_dir = (
                Path(env["GGML_W1AX_CAPTURE_DIR"]) if "GGML_W1AX_CAPTURE_DIR" in env else None
            )
            pid_file = self.root / f"child-{len(self.pid_files)}.pid"
            self.pid_files.append(pid_file)
            env["FAKE_PID_FILE"] = str(pid_file)
            return REAL_POPEN(cmd, **kwargs)

        def wait_ready(_url, _process, _timeout):
            deadline = time.monotonic() + 3
            while (
                self.pid_files and not self.pid_files[-1].exists() and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            if not self.pid_files[-1].exists():
                raise RuntimeError("fake server did not launch child")

        self.popen_patch = mock.patch.object(screen.subprocess, "Popen", side_effect=popen)
        self.ready_patch = mock.patch.object(screen, "wait_ready", side_effect=wait_ready)
        self.http_patch = mock.patch.object(screen, "http_json", side_effect=self._http_json)
        self.read_patch = mock.patch.object(screen, "read_jsonl", side_effect=self._read_jsonl)
        for patcher in (self.popen_patch, self.ready_patch, self.http_patch, self.read_patch):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup()

    def _read_jsonl(self, path):
        if path.name == "rounds.jsonl":
            return list(self.trace_rows)
        if path.name == "state.jsonl":
            return list(self.state_rows)
        return REAL_READ_JSONL(path)

    def _http_json(self, url, body=None, timeout=10):
        if body is None:
            return {"status": "ok"}
        self.calls += 1
        if self.response_mode == "missing_ids":
            return {"choices": [{"message": {"content": "no raw ids"}}]}

        self.trace_rows.append(
            {
                "status": "ok",
                "n_proposed": 2,
                "n_accepted": 2,
                "n_emitted": 1,
                "proposed_token_ids": [31, 32],
            }
        )
        self.state_rows.append(
            {"schema": "eagle_state_v1", "event": "seed", "position": 1, "token": 101}
        )
        capture_dir = self.capture_dir
        if capture_dir and capture_dir.is_dir():
            filename = f"op-{self.calls:012d}.bin"
            (capture_dir / filename).write_bytes(b"activation")
            names = [name for name in self.selected if name != self.missing_capture_name]
            sidecar = capture_dir / "captures.jsonl"
            with sidecar.open("a") as stream:
                for name in names:
                    stream.write(json.dumps({"file": filename, "weight_tensor": name}) + "\n")
        return {"generated_token_ids": [101, 102]}

    def _args(self, output, capture=False):
        return screen.argparse.Namespace(
            output=output,
            prompts=self.prompts,
            capture=capture,
            diagnostic=True,
            variants=self.variants,
            binary=self.binary,
            target=self.target,
            tokens=8,
            port=18090,
        )

    def _assert_descendant_stopped(self, index):
        pid_file = self.pid_files[index]
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pid_file.exists(), "fake server did not start its child")
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            # A defunct process cannot execute; some hosts reap orphaned children later.
            time.sleep(0.05)
        self.fail(f"fake server child {pid} survived process-group cleanup")

    def test_runner_records_raw_ids_quality_runtime_flags_and_shared_capture_manifest(self):
        # Ask the runner to capture via the same env contract used by the runtime.
        self.variants.write_text(
            json.dumps({"candidate": {"draft": str(self.draft), "env": {"TEST_CAPTURE": "1"}}})
        )
        output = self.root / "successful-run"

        screen.run(self._args(output, capture=True))

        manifest = json.loads((output / "candidate" / "manifest.json").read_text())
        self.assertEqual(manifest["requests"][0]["generated_token_ids"], [101, 102])
        self.assertEqual(manifest["requests"][0]["quality"]["accepted"], 2)
        self.assertEqual(manifest["requests"][0]["quality"]["accepted_not_emitted"], 1)
        self.assertEqual(manifest["requests"][0]["quality"]["rounds"], 1)
        self.assertEqual(manifest["server_stop"]["stopped"], True)
        command = manifest["command"]
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "5")
        self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0")
        self.assertIn("--cache-type-k", command)
        self.assertIn("--cache-type-v", command)
        self.assertEqual(manifest["env"]["GGML_EAGLE_DENSE_A16"], "1")
        self.assertEqual(
            manifest["env"]["EAGLE_STATE_TRACE_JSONL"],
            str((output / "candidate" / "state.jsonl").resolve()),
        )
        state_path = output / "candidate" / "000-prompt-01" / "state.json"
        self.assertTrue(state_path.is_file())
        self.assertEqual(json.loads(state_path.read_text()), self.state_rows)
        capture_manifest = json.loads((output / "candidate" / "capture-manifest.json").read_text())
        prompt_capture = capture_manifest["prompts"][0]
        self.assertEqual(prompt_capture["id"], "prompt-01")
        self.assertEqual(
            Path(prompt_capture["capture_dir"]), (output / "candidate" / "activations").resolve()
        )
        self.assertEqual(len(prompt_capture["files"]), 1)
        self.assertTrue(
            (Path(prompt_capture["capture_dir"]) / prompt_capture["files"][0]).is_file()
        )
        self._assert_descendant_stopped(0)

    def test_runner_rejects_missing_captured_layer_and_still_cleans_process_group(self):
        self.variants.write_text(
            json.dumps({"candidate": {"draft": str(self.draft), "env": {"TEST_CAPTURE": "1"}}})
        )
        self.missing_capture_name = self.selected[0]
        output = self.root / "missing-capture-run"

        with self.assertRaisesRegex(ValueError, "missing captured layer"):
            screen.run(self._args(output, capture=True))

        manifest = json.loads((output / "candidate" / "manifest.json").read_text())
        self.assertTrue(manifest["server_stop"]["stopped"])
        self._assert_descendant_stopped(0)

    def test_runner_rejects_missing_raw_token_ids_and_still_cleans_process_group(self):
        self.response_mode = "missing_ids"
        output = self.root / "missing-ids-run"

        with self.assertRaisesRegex(ValueError, "missing raw output token IDs"):
            screen.run(self._args(output))

        manifest = json.loads((output / "candidate" / "manifest.json").read_text())
        self.assertTrue(manifest["server_stop"]["stopped"])
        self._assert_descendant_stopped(0)


if __name__ == "__main__":
    unittest.main()
