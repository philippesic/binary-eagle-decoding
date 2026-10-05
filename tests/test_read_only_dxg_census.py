"""Fixed WSL read observer protocol; fixtures invoke no privileged command or GPU."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from w1a1_eagle import nine_model_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/read_only_dxg_census.py"
spec = importlib.util.spec_from_file_location("read_only_dxg_test", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CensusTests(unittest.TestCase):
    def proc_fixture(self, root):
        boot = root / "sys/kernel/random/boot_id"
        boot.parent.mkdir(parents=True)
        boot.write_text("fixture-boot")
        for pid, target in ((42, "/dev/dxg"), (17, "/dev/null")):
            directory = root / str(pid)
            (directory / "fd").mkdir(parents=True)
            os.symlink(target, directory / "fd/3")
            fields = ["S", "1", str(pid), *(["0"] * 16), "99"]
            (directory / "stat").write_text(f"{pid} (name with spaces) " + " ".join(fields))

    def record(self):
        return {
            "schema": "nine_model_read_only_dxg_census_v1",
            "complete": True,
            "read_only": True,
            "effective_uid": 0,
            "proc_root": "/proc",
            "observer_source_sha256": pipeline.sha256(HELPER),
            "boot_id": "fixture-boot",
            "pid_namespace": "pid:[123]",
            "holders": [{"pid": 42, "start_ticks": 99, "boot_id": "fixture-boot"}],
        }

    def bridge(self, record):
        original = Path.read_text

        def read(path, *args, **kwargs):
            return (
                "fixture-boot"
                if str(path) == "/proc/sys/kernel/random/boot_id"
                else original(path, *args, **kwargs)
            )

        with (
            patch.dict(
                os.environ,
                {"WSL_DISTRO_NAME": "Ubuntu", "WSL_INTEROP": "/run/WSL/fixture_interop"},
                clear=True,
            ),
            patch.object(Path, "read_text", read),
            patch.object(pipeline.os, "readlink", return_value="pid:[123]"),
            patch.object(
                pipeline.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(record))
            ) as run,
        ):
            result = pipeline._privileged_dxg_holders()
            return result, run.call_args

    def test_direct_and_helper_read_same_kernel_holder_identity_from_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.proc_fixture(root)
            expected = [{"pid": 42, "start_ticks": 99, "boot_id": "fixture-boot"}]
            self.assertEqual(pipeline.dxg_holders(root), expected)
            self.assertEqual(helper.census(root), {"boot_id": "fixture-boot", "holders": expected})

    def test_permission_fallback_is_fixed_isolated_read_command(self):
        record = self.record()
        result, call = self.bridge(record)
        self.assertEqual(result, record["holders"])
        self.assertEqual(
            call.args[0],
            [
                "/mnt/c/Windows/System32/wsl.exe",
                "-d",
                "Ubuntu",
                "-u",
                "root",
                "--",
                "/usr/bin/python3",
                "-B",
                "-I",
                str(HELPER),
                "--expected-sha256",
                record["observer_source_sha256"],
            ],
        )
        self.assertEqual(call.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(call.kwargs["timeout"], 15)
        self.assertEqual(set(call.kwargs["env"]), {"PATH", "LANG", "WSL_INTEROP"})
        self.assertEqual(call.kwargs["cwd"], "/")
        with (
            patch.object(pipeline, "_scan_dxg_holders", side_effect=PermissionError),
            patch.object(
                pipeline, "_privileged_dxg_holders", return_value=record["holders"]
            ) as bridge,
        ):
            self.assertEqual(pipeline.dxg_holders(), record["holders"])
            bridge.assert_called_once_with()

    def test_bad_protocol_hash_uid_namespace_or_identity_never_reports_empty(self):
        changes = (
            {"complete": False},
            {"read_only": False},
            {"effective_uid": 1000},
            {"observer_source_sha256": "0" * 64},
            {"boot_id": "other-boot"},
            {"pid_namespace": "pid:[999]"},
            {"holders": [{"pid": True, "start_ticks": 1, "boot_id": "fixture-boot"}]},
            {"holders": [{"pid": 42, "start_ticks": 1, "boot_id": "other-boot"}]},
        )
        for change in changes:
            record = copy.deepcopy(self.record())
            record.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "PENDING"):
                self.bridge(record)

    def test_custom_proc_denial_and_wrong_distro_never_escalate(self):
        with (
            patch.object(pipeline, "_scan_dxg_holders", side_effect=PermissionError),
            patch.object(pipeline, "_privileged_dxg_holders") as bridge,
        ):
            with self.assertRaisesRegex(ValueError, "PENDING"):
                pipeline.dxg_holders(Path("/fixture/proc"))
            bridge.assert_not_called()
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Other"}, clear=True),
            patch.object(pipeline.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "PENDING"):
                pipeline._privileged_dxg_holders()
            run.assert_not_called()

    def test_unavailable_bridge_and_helper_denial_remain_fail_closed(self):
        for error in (
            subprocess.TimeoutExpired("fixed-read-observer", 15),
            subprocess.CalledProcessError(1, "fixed-read-observer"),
            PermissionError("denied"),
        ):
            with (
                patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True),
                patch.object(pipeline.subprocess, "run", side_effect=error),
                self.assertRaisesRegex(ValueError, "PENDING"),
            ):
                pipeline._privileged_dxg_holders()
        original = Path.iterdir

        def denied(path):
            if path.name == "fd":
                raise PermissionError("fixture denied")
            return original(path)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.proc_fixture(root)
            with patch.object(Path, "iterdir", denied), self.assertRaises(PermissionError):
                helper.census(root)

    def test_new_global_holder_still_blocks_resource_return(self):
        policy = {
            "host_floor_bytes": 1,
            "gpu_floor_bytes": 1,
            "host_return_tolerance_bytes": 0,
            "gpu_return_tolerance_bytes": 0,
        }
        baseline = {
            "gpu_uuid": "fixture-gpu",
            "host_available_bytes": 100,
            "gpu_free_bytes": 100,
            "dxg_holders": [],
        }
        with self.assertRaisesRegex(ValueError, "new /dev/dxg"):
            pipeline.resource_gate(
                dict(baseline, dxg_holders=self.record()["holders"]), baseline, policy
            )

    def test_helper_cli_hash_or_arbitrary_proc_arguments_refuse_before_census(self):
        for extra in (
            ("--expected-sha256", "0" * 64),
            ("--expected-sha256", pipeline.sha256(HELPER), "--proc-root", "/fixture"),
        ):
            result = subprocess.run(
                [sys.executable, str(HELPER), *extra], capture_output=True, text=True, timeout=5
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
