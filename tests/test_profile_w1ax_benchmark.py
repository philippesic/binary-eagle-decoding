"""Profiling wrapper checks with fake Nsight libraries; no profiler or GPU needed."""

import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_w1ax_benchmark.py"
SPEC = importlib.util.spec_from_file_location("profile_w1ax_benchmark", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ProfileWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.nsys = self.root / "nsight-systems"
        self.nsys.mkdir()
        self.injection = self.nsys / "libToolsInjection64.so"
        self.preload = self.nsys / "libnsys.so"
        self.injection.write_bytes(b"fake injection")
        self.preload.write_bytes(b"fake preload")
        self.runner = self.root / "runner.py"
        self.runner.write_text("print('not executed by these tests')\n")
        self.config = self.root / "frozen.toml"
        self.source = b'''schema_version = 1
[evaluation]
max_draft_tokens = 5
min_draft_probability = 0.0
repetitions = 5
seed = 42
[environment]
CUDA_VISIBLE_DEVICES = "0"
[models]
target = "target-f16.gguf"
'''
        self.config.write_bytes(self.source)
        self.output = self.root / "profile.toml"
        self.environ = {
            "CUDA_INJECTION64_PATH": str(self.injection),
            "LD_PRELOAD": str(self.preload),
            "NSYS_PROFILING_SESSION_ID": "test-session",
        }

    def prepare(self, args=None, environ=None):
        return profile.prepare_profile(
            self.runner, self.config, self.output, self.nsys,
            ["--run-id", "profile-test"] if args is None else args,
            self.environ if environ is None else environ,
        )

    def test_exact_whitelist_preserves_numeric_config_and_records_provenance(self):
        observed = {key: "observed-value" for key in profile.NSIGHT_KEYS}
        observed.update(self.environ)
        observed.update({"NSYS_UNEXPECTED_SECRET": "do not pass", "OTHER_VARIABLE": "ignore"})
        command = self.prepare(environ=observed)
        generated = tomllib.loads(self.output.read_text())
        expected = tomllib.loads(self.source.decode())
        expected["environment"].update({key: observed[key] for key in profile.NSIGHT_KEYS})
        self.assertEqual(generated, expected)
        self.assertEqual(self.config.read_bytes(), self.source)
        sidecar = json.loads(Path(str(self.output) + ".provenance.json").read_text())
        self.assertEqual(set(sidecar["passed_keys"]), set(profile.NSIGHT_KEYS))
        self.assertNotIn("NSYS_UNEXPECTED_SECRET", sidecar["passed_environment"])
        for name, path in (("driver", SCRIPT), ("runner", self.runner),
                           ("source_config", self.config), ("profile_config", self.output)):
            self.assertEqual(sidecar[name]["sha256"], profile.sha256_bytes(path.read_bytes()))
        self.assertEqual(command, [sys.executable, str(self.runner), "--config", str(self.output),
                                   "--run-id", "profile-test"])
        self.assertEqual(sidecar["command"], command)

    def test_missing_live_profiler_essentials_fails_without_artifacts(self):
        for missing in profile.REQUIRED_KEYS:
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "live Nsight"):
                self.prepare(environ={key: value for key, value in self.environ.items() if key != missing})
            self.assertFalse(self.output.exists())

    def test_unrelated_preload_and_symlink_escape_are_rejected(self):
        unrelated = self.root / "unrelated.so"
        unrelated.write_bytes(b"unrelated")
        escaped = self.nsys / "escape.so"
        escaped.symlink_to(unrelated)
        for library in (unrelated, escaped):
            for key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH"):
                with self.subTest(library=library, key=key), self.assertRaisesRegex(ValueError, "within --nsys-root"):
                    self.prepare(environ={**self.environ, key: str(library)})
        self.assertFalse(self.output.exists())

    def test_observed_colon_separated_nsight_preload_list_is_preserved(self):
        names = ("libnvperf_target.so", "libnvperf_host.so", "libToolsInjectionProxy64.so",
                 "libLinuxKeyboardInterceptorProxy.so")
        for name in names:
            (self.nsys / name).write_bytes(b"fake Nsight library")
        preload = ":".join(str(self.nsys / name) for name in names)
        selected, libraries = profile.profiler_environment(
            {**self.environ, "LD_PRELOAD": preload}, self.nsys
        )
        self.assertEqual(selected["LD_PRELOAD"], preload)
        self.assertEqual(len(libraries), 5)
        unrelated = self.root / "outside.so"
        unrelated.write_bytes(b"outside")
        with self.assertRaisesRegex(ValueError, "within --nsys-root"):
            profile.profiler_environment(
                {**self.environ, "LD_PRELOAD": preload + ":" + str(unrelated)}, self.nsys
            )

    def test_conflicting_runner_config_arguments_are_rejected(self):
        for args in (["--config", "other.toml"], ["--config=other.toml"],
                     ["--conf", "other.toml"], ["--co=other.toml"]):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "override or abbreviate"):
                self.prepare(args=args)
        self.assertFalse(self.output.exists())

    def test_existing_output_or_sidecar_is_never_overwritten(self):
        self.prepare()
        generated = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.output.read_bytes(), generated)
        self.output.unlink()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.config.read_bytes(), self.source)

    def test_existing_environment_conflict_is_rejected(self):
        self.config.write_bytes(self.source.replace(
            b'[environment]\n', b'[environment]\nLD_PRELOAD = "/unrelated.so"\n'
        ))
        with self.assertRaisesRegex(ValueError, "config conflicts"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_environment_table_can_be_added_without_other_changes(self):
        source = b'[evaluation]\nseed = 42\nvalues = [1, 2, 3]\n'
        generated = profile.inject_environment(source, self.environ)
        self.assertEqual(tomllib.loads(generated.decode()), {
            "evaluation": {"seed": 42, "values": [1, 2, 3]}, "environment": self.environ,
        })

    def test_main_replaces_wrapper_with_python_runner(self):
        with patch.dict(profile.os.environ, self.environ, clear=True), patch.object(profile.os, "execv") as execv:
            profile.main([
                "--runner", str(self.runner), "--config", str(self.config),
                "--profile-config", str(self.output), "--nsys-root", str(self.nsys),
                "--", "--prompt-id", "reasoning-02",
            ])
        execv.assert_called_once_with(sys.executable, [
            sys.executable, str(self.runner), "--config", str(self.output),
            "--prompt-id", "reasoning-02",
        ])


if __name__ == "__main__":
    unittest.main()
