"""Synthetic contracts for the W1Ax model parity driver; no GPU or models."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_w1ax_model_parity", ROOT / "scripts" / "check_w1ax_model_parity.py"
)
parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parity)


class W1AxModelParityTests(unittest.TestCase):
    def test_verbose_token_ids_are_strictly_integer_ids(self):
        self.assertEqual(parity.token_ids({"__verbose": {"tokens": [10, 20]}}), [10, 20])
        self.assertIsNone(parity.token_ids({"__verbose": {"tokens": [10, True]}}))
        self.assertIsNone(parity.token_ids({"choices": [{"tokens": [10, 20]}]}))

    def test_commands_keep_target_fixed_and_select_only_draft_placement(self):
        common = dict(
            binary=Path("server"), target=Path("target-f16.gguf"),
            draft=Path("all-nine-w1a1.gguf"), host="127.0.0.1", port=18080,
        )
        cpu = parity.build_server_command(**common, draft_ngl="0")
        cuda = parity.build_server_command(**common, draft_ngl="all")
        self.assertEqual(cpu[cpu.index("-m") + 1], cuda[cuda.index("-m") + 1])
        self.assertEqual(cpu[cpu.index("-md") + 1], cuda[cuda.index("-md") + 1])
        self.assertEqual(cpu[cpu.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(cpu[cpu.index("--spec-draft-ngl") + 1], "0")
        self.assertEqual(cuda[cuda.index("--spec-draft-ngl") + 1], "all")
        self.assertEqual(cpu[cpu.index("--spec-draft-n-max") + 1], "5")
        self.assertEqual(cpu[cpu.index("--spec-draft-p-min") + 1], "0")

    def test_mode_evidence_requires_loader_activation_and_placement_markers(self):
        for bits in parity.BITS:
            base = "\n".join((
                "llama_model_load: offloaded 36/37 layers to GPU",
                parity.LOADER_MARKER,
                f"EAGLE3 W1Ax activation bits: {bits}",
            ))
            command_cpu = parity.build_server_command(
                Path("server"), Path("target"), Path("draft"), "127.0.0.1", 10000, "0"
            )
            command_cuda = parity.build_server_command(
                Path("server"), Path("target"), Path("draft"), "127.0.0.1", 10000, "all"
            )
            cpu = parity.mode_evidence(bits, "cpu", command_cpu, base)
            cuda_log = base + "\n" + parity.CUDA_MARKERS[bits]
            cuda = parity.mode_evidence(bits, "cuda", command_cuda, cuda_log)
            self.assertTrue(cpu["placement_confirmed"], bits)
            self.assertTrue(cuda["placement_confirmed"], bits)
            self.assertFalse(parity.mode_evidence(
                bits, "cpu", command_cpu, cuda_log
            )["placement_confirmed"])
            self.assertFalse(parity.mode_evidence(
                bits, "cuda", command_cuda, base
            )["placement_confirmed"])
            wrong_bits = next(other for other in parity.BITS if other != bits)
            self.assertFalse(parity.mode_evidence(
                bits, "cuda", command_cuda,
                cuda_log + f"\nEAGLE3 W1Ax activation bits: {wrong_bits}",
            )["placement_confirmed"])
            self.assertFalse(parity.mode_evidence(
                bits, "cuda", command_cuda, cuda_log.replace(
                    "llama_model_load: offloaded 36/37 layers to GPU", "no GPU offload"
                ),
            )["placement_confirmed"])

    def test_prompt_loader_pins_requested_historical_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            prompts = Path(temporary) / "prompts.jsonl"
            prompts.write_text(
                '{"id":"old","messages":[{"role":"user","content":"old"}]}\n'
                '{"id":"prose-01","messages":[{"role":"user","content":"fixed"}]}\n'
            )
            self.assertEqual(parity.load_prompt(prompts, "prose-01")["messages"][0]["content"], "fixed")
            with self.assertRaisesRegex(ValueError, "not found"):
                parity.load_prompt(prompts, "missing")


if __name__ == "__main__":
    unittest.main()
