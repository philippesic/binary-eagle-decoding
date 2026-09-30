"""Real CPU child-group cleanup under a signal; no native model/GPU work."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativeCancellationTests(unittest.TestCase):
    def exercise(self, *, during_spawn=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.write_bytes(b"CPU fixture")
            prompts = root / "prompts.jsonl"
            prompts.write_text(json.dumps({"id": "train-a", "messages": []}) + "\n")
            script = root / "parent.py"
            script.write_text("""import json, os, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from run_binary_head_capture import run_cell
from w1ax_continuous_stages import native_cancellation
root = Path(sys.argv[1])
args = SimpleNamespace(mode="diagnostic", output=root, prompts=root/"prompts.jsonl",
                       binary=root/"source", target=root/"source", port=1, tokens=1,
                       shard_manifest=None)
def waiting(*args):
    while True:
        time.sleep(.1)
real_popen = subprocess.Popen
def spawning(*args, **kwargs):
    child = real_popen(*args, **kwargs)
    if sys.argv[2] == "race":
        os.kill(os.getpid(), signal.SIGTERM)
    return child
process_patch = patch("run_binary_head_capture.subprocess.Popen", side_effect=spawning)
command_patch = patch("run_binary_head_capture.server_command",
    return_value=[sys.executable,"-c","import time; time.sleep(60)"])
wait_patch = patch("run_binary_head_capture.wait_ready", side_effect=waiting)
try:
    with native_cancellation() as guard, process_patch, command_patch, wait_patch:
        args.cancellation_guard = guard
        run_cell(args,"d_d",{"draft":str(root/"source")},[{"id":"train-a","messages":[]}])
except InterruptedError:
    (root/"finished").write_text("stopped")
""")
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
            }
            parent = subprocess.Popen(
                [sys.executable, str(script), str(root), "race" if during_spawn else "waiting"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.addCleanup(lambda: parent.kill() if parent.poll() is None else None)
            manifest = root / "d_d/manifest.json"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if manifest.exists():
                    try:
                        state = json.loads(manifest.read_text())
                    except json.JSONDecodeError:
                        continue
                    if "server_pid" in state:
                        break
                if parent.poll() is not None:
                    self.fail(parent.communicate()[1].decode())
                time.sleep(0.05)
            else:
                self.fail("CPU fixture server did not start")
            if not during_spawn:
                os.kill(parent.pid, signal.SIGTERM)
            stdout, stderr = parent.communicate(timeout=10)
            self.assertEqual(parent.returncode, 0, stderr.decode() + stdout.decode())
            state = json.loads(manifest.read_text())
            self.assertTrue(state["server_stop"]["stopped"])
            self.assertTrue(state["server_stop"]["process_group_gone"])
            self.assertEqual((root / "finished").read_text(), "stopped")
            with self.assertRaises(ProcessLookupError):
                os.killpg(state["server_pgid"], 0)

    def test_signal_unwinds_server_group_and_restores_handler(self):
        self.exercise()

    def test_signal_between_spawn_and_registration_is_deferred_then_cleans_up(self):
        self.exercise(during_spawn=True)


if __name__ == "__main__":
    unittest.main()
