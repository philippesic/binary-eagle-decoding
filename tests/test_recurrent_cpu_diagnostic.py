"""CPU-only native diagnostic refuses accelerator builds and GPU layers."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_recurrent_cpu_diagnostic as runner  # noqa: E402


class CpuDiagnosticPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        build = self.root / "cpu-build"
        (build / "bin").mkdir(parents=True)
        self.binary = build / "bin/llama-server"
        self.binary.write_bytes(b"cpu-fixture")
        self.cache = build / "CMakeCache.txt"
        self.cache.write_text(
            "".join(f"GGML_{backend}:BOOL=OFF\n" for backend in runner.CPU_BACKENDS)
        )
        self.target = self.root / "target.gguf"
        self.draft = self.root / "D.gguf"
        self.prompts = self.root / "train.jsonl"
        self.target.write_bytes(b"target")
        self.draft.write_bytes(b"draft")
        self.rows = [
            {
                "id": f"qat-revisit-train-prompt-{index:02d}",
                "messages": [{"role": "user", "content": "x"}],
            }
            for index in range(96)
        ]
        self.prompts.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def validate(self, device="cpu"):
        def fake_hash(path):
            return {
                self.target: runner.TARGET_F16_SHA256,
                self.draft: runner.DRAFT_D_SHA256,
                self.prompts: runner.TRAIN_PROMPTS_SHA256,
            }[path]

        with (
            mock.patch.object(runner, "sha256", side_effect=fake_hash),
            mock.patch.object(
                runner.subprocess, "check_output", return_value=runner.NATIVE_REVISION + "\n"
            ),
        ):
            return runner.validate_inputs(
                self.binary,
                self.cache,
                self.target,
                self.draft,
                self.prompts,
                self.rows[0]["id"],
                8,
                device,
            )

    def test_cpu_build_and_command_pin_both_gpu_layer_counts_to_zero(self):
        self.assertEqual(self.validate()["id"], self.rows[0]["id"])
        command = runner.native_command(self.binary, self.target, self.draft, 18557, "auto")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "0")
        self.assertIn("--no-context-shift", command)

    def test_metal_enabled_build_is_rejected_before_model_run(self):
        self.cache.write_text(
            self.cache.read_text().replace("GGML_METAL:BOOL=OFF", "GGML_METAL:BOOL=ON")
        )
        with self.assertRaisesRegex(ValueError, "GGML_METAL=OFF"):
            self.validate()

    def test_cuda_mode_requires_cuda_only_and_offloads_both_models(self):
        self.cache.write_text(
            self.cache.read_text().replace("GGML_CUDA:BOOL=OFF", "GGML_CUDA:BOOL=ON")
        )
        self.assertEqual(self.validate("cuda")["id"], self.rows[0]["id"])
        command = runner.native_command(self.binary, self.target, self.draft, 18557, "auto", "cuda")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")
        with self.assertRaisesRegex(ValueError, "CPU diagnostic requires GGML_CUDA=OFF"):
            self.validate()
        self.cache.write_text(
            self.cache.read_text().replace("GGML_METAL:BOOL=OFF", "GGML_METAL:BOOL=ON")
        )
        with self.assertRaisesRegex(ValueError, "CUDA diagnostic requires GGML_METAL=OFF"):
            self.validate("cuda")

    def test_cuda_mode_rejects_cpu_cache_hook_before_launch(self):
        args = argparse.Namespace(device="cuda", capture_cache=True, flash_attention="auto")
        with self.assertRaisesRegex(ValueError, "requires CPU mode"):
            runner.capture_one(args)


if __name__ == "__main__":
    unittest.main()
