"""Bad source/capture evidence refuses before a native/GPU query."""

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from w1a1_eagle.nine_model_pipeline import sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "eagle_smoke_test", ROOT / "scripts/check_eagle_binary_native.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inputs = {}
        for key in ("binary", "model", "target", "export_audit", "train_prompts", "train_capture"):
            path = self.root / key
            path.write_text("fixture")
            self.inputs[key] = {"path": str(path), "sha256": sha256(path)}
        self.config = {
            "schema": "eagle_native_graph_smoke_config_v1",
            "inputs": self.inputs,
            "activation_bits": 8,
        }
        self.config_path = self.root / "config.json"

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def test_changed_model_refuses_before_gpu(self):
        self.write()
        Path(self.inputs["model"]["path"]).write_text("changed")
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                smoke.smoke(self.config_path, self.root / "receipt.json")
        self.assertFalse((self.root / "receipt.json").exists())

    def test_export_failed_cannot_be_reinterpreted_as_native_ready(self):
        path = Path(self.inputs["export_audit"]["path"])
        path.write_text(json.dumps({"serialization_audit_passed": False, "activation_bits": 8}))
        self.inputs["export_audit"]["sha256"] = sha256(path)
        self.write()
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "packed EAGLE export"):
                smoke.smoke(self.config_path, self.root / "receipt.json")

    def test_development_capture_cannot_substitute_train_prefix(self):
        audit = Path(self.inputs["export_audit"]["path"])
        audit.write_text(
            json.dumps(
                {
                    "serialization_audit_passed": True,
                    "activation_bits": 8,
                    "output": {"sha256": self.inputs["model"]["sha256"]},
                }
            )
        )
        capture = Path(self.inputs["train_capture"]["path"])
        capture.write_text(json.dumps({"split": "development"}))
        for key in ("export_audit", "train_capture"):
            self.inputs[key]["sha256"] = sha256(self.inputs[key]["path"])
        self.write()
        with patch.object(smoke, "LinuxResources", side_effect=AssertionError("GPU queried")):
            with self.assertRaisesRegex(ValueError, "TRAIN prefix"):
                smoke.smoke(self.config_path, self.root / "receipt.json")

    def test_launched_server_exposes_native_info_trace_at_current_required_level(self):
        header = (ROOT / "third_party/llama.cpp/common/log.h").read_text()
        callback = (ROOT / "third_party/llama.cpp/common/log.cpp").read_text()
        level = int(re.search(r"#define LOG_LEVEL_TRACE\s+(\d+)", header)[1])
        self.assertRegex(callback, r"case GGML_LOG_LEVEL_INFO:\s*return LOG_LEVEL_TRACE;")
        prompts = Path(self.inputs["train_prompts"]["path"])
        prompts.write_text(
            json.dumps(
                {
                    "id": "fixture",
                    "split": "train",
                    "messages": [{"role": "user", "content": "fixture"}],
                }
            )
        )
        audit = Path(self.inputs["export_audit"]["path"])
        audit.write_text(
            json.dumps(
                {
                    "serialization_audit_passed": True,
                    "activation_bits": 8,
                    "output": {"sha256": self.inputs["model"]["sha256"]},
                }
            )
        )
        capture = Path(self.inputs["train_capture"]["path"])
        capture.write_text(
            json.dumps(
                {
                    "split": "train",
                    "target_sha256": self.inputs["target"]["sha256"],
                    "prompts_sha256": sha256(prompts),
                }
            )
        )
        for key in ("train_prompts", "export_audit", "train_capture"):
            self.inputs[key]["sha256"] = sha256(self.inputs[key]["path"])
        policy = {
            "host_floor_bytes": 1,
            "gpu_floor_bytes": 1,
            "host_return_tolerance_bytes": 0,
            "gpu_return_tolerance_bytes": 0,
        }
        self.config.update(
            gpu_uuid="fixture-GPU",
            hardware="rtx5080",
            port=18990,
            context_tokens=2048,
            startup_wall_seconds=1,
            request_wall_seconds=1,
            resource_policy=policy,
        )
        self.write()
        observer = MagicMock()
        observer.snapshot.return_value = {
            "gpu_uuid": "fixture-GPU",
            "host_available_bytes": 100,
            "gpu_free_bytes": 100,
            "dxg_holders": [],
        }
        observer.require_released.return_value = {"fixture_release": True}
        commands = []

        def launch(argv, **kwargs):
            commands.append(argv)
            kwargs["stdout"].write(
                b"EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)\n"
                b"CUDA packed W1A8 INT8 dispatch\n"
            )
            kwargs["stdout"].flush()
            proc = MagicMock()
            proc.pid = 123
            return proc

        with (
            patch.object(smoke, "LinuxResources", return_value=observer),
            patch.object(smoke, "available_port", return_value=True),
            patch.object(smoke.subprocess, "Popen", side_effect=launch),
            patch.object(
                smoke,
                "process_identity",
                return_value={"pid": 123, "start_ticks": 1, "boot_id": "fixture"},
            ),
            patch.object(smoke, "wait_ready"),
            patch.object(smoke, "execute_request", return_value={"generated_token_ids": [1]}),
            patch.object(smoke, "stop_owned_server"),
            patch.object(
                smoke, "validate_cuda_dispatch", return_value={"selected_projection_count": 9}
            ) as typed,
        ):
            smoke.smoke(self.config_path, self.root / "receipt.json")
        self.assertEqual(len(commands), 1)
        argv = commands[0]
        self.assertEqual(int(argv[argv.index("--log-verbosity") + 1]), level)
        typed.assert_called_once()
        self.assertEqual(typed.call_args.kwargs, {"activation_bits": 8})


if __name__ == "__main__":
    unittest.main()
