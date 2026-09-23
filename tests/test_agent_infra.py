"""Check the two agent lifecycle boundaries that must survive long runs."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AgentInfrastructureTests(unittest.TestCase):
    def test_second_compact_requests_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            hook = ROOT / ".codex/hooks/compaction_handoff.py"
            event = {"cwd": str(repo), "session_id": "test-session", "source": "compact"}
            messages = []
            for _ in range(2):
                result = subprocess.run(
                    [sys.executable, str(hook)],
                    input=json.dumps(event),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                messages.append(
                    json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
                )
            self.assertIn("First compaction", messages[0])
            self.assertIn("fresh successor", messages[1])

    def test_remote_supervisor_interrupts_child_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            supervisor = root / "scripts/remote_job.py"
            shutil.copyfile(ROOT / "scripts/remote_job.py", supervisor)
            child = root / "child.py"
            child.write_text("import time\ntime.sleep(60)\n")
            runner = subprocess.Popen(
                [sys.executable, str(supervisor), "test-run", "--", sys.executable, str(child)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            record_path = root / "runs/test-run/state.json"
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if record_path.exists():
                        record = json.loads(record_path.read_text())
                        if record["status"] == "running":
                            break
                    time.sleep(0.05)
                else:
                    self.fail("supervisor did not start")
                runner.send_signal(signal.SIGINT)
                runner.communicate(timeout=15)
                self.assertEqual(runner.returncode, 130)
                record = json.loads(record_path.read_text())
                self.assertEqual(record["status"], "interrupted")
                self.assertIsNotNone(record["exit_code"])
                self.assertNotEqual(record["exit_code"], 0)
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(record["pid"])],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                self.assertTrue(not state or state.startswith("Z"), state)
            finally:
                if runner.poll() is None:
                    runner.kill()
                    runner.wait()
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if "pgid" in record:
                        try:
                            os.killpg(record["pgid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
