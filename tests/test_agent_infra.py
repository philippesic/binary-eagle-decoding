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

    def test_remote_supervisor_honors_configured_stop_grace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            supervisor = root / "scripts/remote_job.py"
            shutil.copyfile(ROOT / "scripts/remote_job.py", supervisor)
            marker = root / "graceful-stop.txt"
            child = root / "child.py"
            child.write_text(
                "import pathlib, signal, time\n"
                f"marker = pathlib.Path({str(marker)!r})\n"
                "def stop(*_):\n"
                "    marker.write_text('handled SIGTERM')\n"
                "    time.sleep(0.35)\n"
                "    raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "print('ready', flush=True)\n"
                "while True: time.sleep(1)\n"
            )
            runner = subprocess.Popen(
                [sys.executable, str(supervisor), "grace-run", "--stop-grace-seconds", "1",
                 "--", sys.executable, str(child)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            record_path = root / "runs/grace-run/state.json"
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
                runner.communicate(timeout=10)
                self.assertEqual(runner.returncode, 130)
                self.assertEqual(marker.read_text(), "handled SIGTERM")
                record = json.loads(record_path.read_text())
                self.assertEqual(record["stop_grace_seconds"], 1.0)
                self.assertEqual(record["exit_code"], 0)
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

    def test_remote_supervisor_bounds_rotated_stdout_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            supervisor = root / "scripts/remote_job.py"
            shutil.copyfile(ROOT / "scripts/remote_job.py", supervisor)
            child = root / "child.py"
            child.write_text(
                "for index in range(30):\n"
                "    print(f'{index:02d}:' + 'x' * 30, flush=True)\n"
            )
            result = subprocess.run(
                [sys.executable, str(supervisor), "log-run", "--max-log-bytes", "128",
                 "--log-backups", "2", "--", sys.executable, str(child)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = root / "runs/log-run"
            logs = [run_dir / "stdout.log", run_dir / "stdout.log.1", run_dir / "stdout.log.2"]
            existing = [path for path in logs if path.exists()]
            self.assertEqual(len(existing), 3)
            self.assertTrue(all(path.stat().st_size <= 128 for path in existing))
            retained = b"".join(path.read_bytes() for path in existing)
            self.assertIn(b"29:", retained)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["log_rotation"], {"max_bytes_per_file": 128, "backups": 2})


if __name__ == "__main__":
    unittest.main()
