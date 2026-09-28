"""Synthetic CPU stage, capture-corruption and first-seed ancestry checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from check_recurrent_ffn_stage_parity import (  # noqa: E402
    BASELINE_MANIFEST_SHA256,
    CONTROL_FILES,
    STAGE_ORDER,
    STAGE_TAPS,
    compare_stages,
    read_first_seed,
    replay_stages,
    verify_ancestry,
    verify_baseline_identity,
    verify_capture,
)
from load_recurrent_binary_init import D_SHA256  # noqa: E402
from run_binary_head_capture import TARGET_F16_SHA256, TRAIN_PROMPTS_SHA256  # noqa: E402

from w1a1_eagle.recurrent_binary import pack_signs  # noqa: E402


class StageReplayTests(unittest.TestCase):
    def setUp(self):
        self.input = np.array([1.001, -0.751, 0.333, 1.117], np.float32)
        signs = np.ones((32, 4), np.float32)
        signs[1::2, 1] = -1
        down_signs = np.ones((4, 32), np.float32)
        down_signs[:, 1::2] = -1
        self.arrays = {
            "blk.0.ffn_gate": (pack_signs(signs), np.ones((32, 1), np.float32) * 0.2),
            "blk.0.ffn_up": (pack_signs(-signs), np.ones((32, 1), np.float32) * 0.3),
            "blk.0.ffn_down": (pack_signs(down_signs), np.ones((4, 1), np.float32) * 0.4),
        }

    def test_replays_all_five_f32_stages_and_finds_first_difference(self):
        for arithmetic in ("native_order", "group_matmul"):
            with self.subTest(arithmetic=arithmetic):
                replay = replay_stages(
                    self.input, self.arrays, arithmetic, expected_intermediate=32
                )
                self.assertEqual(tuple(replay), STAGE_ORDER)
                self.assertEqual(
                    [replay[name].shape for name in STAGE_ORDER], [(32,), (32,), (32,), (32,), (4,)]
                )
                self.assertIsNone(compare_stages(replay, replay)["first_divergent_stage"])
                for stage in STAGE_ORDER:
                    native = {name: values.copy() for name, values in replay.items()}
                    native[stage][0] += np.float32(1.0)
                    self.assertEqual(compare_stages(native, replay)["first_divergent_stage"], stage)

    def test_rejects_missing_weights_and_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "gate/up/down"):
            replay_stages(self.input, {}, "native_order", expected_intermediate=32)
        bad = self.input.copy()
        bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite F32"):
            replay_stages(bad, self.arrays, "native_order", expected_intermediate=32)


class CaptureSealTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.capture = Path(self.temp.name)
        self.ids = [1654, 525, 2661, 1447, 12, 3070, 47, 1510]
        for name in CONTROL_FILES:
            (self.capture / name).write_bytes(b"{}\n")
        (self.capture / "response.json").write_text(json.dumps({"__verbose": {"tokens": self.ids}}))
        from audit_recurrent_binary_capture import sha256

        self.manifest = {
            "schema": "recurrent_cpu_native_diagnostic_v1",
            "execution_device": "cpu",
            "training_eligible": False,
            "prompt_id": "qat-revisit-train-reasoning-rate-and-work-01",
            "generated_token_ids": self.ids,
            "source_sha256": {
                "target": TARGET_F16_SHA256,
                "draft": D_SHA256,
                "frozen_train_prompts": TRAIN_PROMPTS_SHA256,
            },
            "files": {
                name: {
                    "bytes": (self.capture / name).stat().st_size,
                    "sha256": sha256(self.capture / name),
                }
                for name in CONTROL_FILES
            },
        }
        (self.capture / "manifest.json").write_text(json.dumps(self.manifest))

    def test_sealed_cpu_capture_and_corrupt_payload(self):
        manifest, seal = verify_capture(self.capture, require_stages=True)
        self.assertEqual(manifest["generated_token_ids"], self.ids)
        self.assertEqual(seal["raw_ids"], self.ids)
        (self.capture / "heads.f32").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "size or SHA256"):
            verify_capture(self.capture, require_stages=True)

    def test_rejects_unsealed_extra_file_and_changed_model(self):
        (self.capture / "extra.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "file set"):
            verify_capture(self.capture, require_stages=False)
        (self.capture / "extra.bin").unlink()
        self.manifest["source_sha256"]["target"] = "0" * 64
        (self.capture / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "not pinned"):
            verify_capture(self.capture, require_stages=False)


class AncestryTests(unittest.TestCase):
    def setUp(self):
        self.seed = {
            "head_state": np.array([1, 2], np.float32),
            "input": np.array([3, 4], np.float32),
            "output": np.array([5, 6], np.float32),
        }
        self.seal = {
            "raw_ids": list(range(8)),
            "files": {
                "prompt.json": {"sha256": "p"},
                "request.json": {"sha256": "r"},
            },
        }

    def test_exact_first_seed_ancestry(self):
        result = verify_ancestry(self.seed, self.seed, self.seal, self.seal)
        self.assertEqual(result["first_eight_raw_ids"], list(range(8)))
        altered = {**self.seed, "input": np.array([3, 4.001], np.float32)}
        with self.assertRaisesRegex(ValueError, "first-seed input"):
            verify_ancestry(self.seed, altered, self.seal, self.seal)
        changed = {**self.seal, "raw_ids": list(range(7)) + [99]}
        with self.assertRaisesRegex(ValueError, "eight raw output IDs"):
            verify_ancestry(self.seed, self.seed, self.seal, changed)
        changed = {
            **self.seal,
            "files": {**self.seal["files"], "request.json": {"sha256": "different"}},
        }
        with self.assertRaisesRegex(ValueError, "request.json bytes"):
            verify_ancestry(self.seed, self.seed, self.seal, changed)

    def test_baseline_must_be_the_archived_manifest(self):
        verify_baseline_identity({"manifest": BASELINE_MANIFEST_SHA256})
        with self.assertRaisesRegex(ValueError, "archived frozen reasoning"):
            verify_baseline_identity({"manifest": "0" * 64})


class GraphTapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.capture = Path(self.temp.name)
        self.head = np.zeros(2560, np.float32)
        self.head[0] = 1
        (self.capture / "heads.jsonl").write_text(
            json.dumps(
                {
                    "schema": "eagle_head_state_v1",
                    "round_index": 0,
                    "depth": 0,
                    "state_dim": 2560,
                    "state_boundary": "native_output_norm_f32_before_head_operand_conversion",
                    "state_row": 0,
                }
            )
            + "\n"
        )
        self.head.tofile(self.capture / "heads.f32")
        vectors = [self.head, np.ones(2560, np.float32) * 2]
        self.rows = [self.row("result_norm", 0, 2560), self.row("post_attn_norm-0", 2560, 2560)]
        offset = 5120
        for stage, name in STAGE_TAPS.items():
            width = 2560 if stage == "down" else 9728
            self.rows.append(self.row(name, offset, width))
            vectors.append(np.ones(width, np.float32) * (len(vectors) + 1))
            offset += width
        self.values = np.concatenate(vectors)

    @staticmethod
    def row(name: str, offset: int, width: int) -> dict:
        return {
            "group_kind": "decoder",
            "group_execution": 2,
            "tensor_name": name,
            "n_tokens": 1,
            "token_width": width,
            "dtype": "f32",
            "ne": [width, 1],
            "token_axis": 1,
            "f32_offset": offset,
        }

    def read(self):
        fake_model = type(
            "FakeModel",
            (),
            {
                "tensors": {
                    "output_norm.weight": type("Tensor", (), {"data": np.ones(2560, np.float32)})()
                }
            },
        )()
        with (
            patch(
                "check_recurrent_ffn_stage_parity._read_graph",
                return_value=(self.rows, self.values, {}),
            ),
            patch("check_recurrent_ffn_stage_parity.Model", return_value=fake_model),
        ):
            return read_first_seed(self.capture, Path("unused.gguf"), require_stages=True)

    def test_joined_graph_requires_all_five_correct_stage_taps(self):
        capture = self.read()
        self.assertEqual(capture["join"]["native_group_execution"], 2)
        self.assertEqual(tuple(capture["stages"]), STAGE_ORDER)
        self.assertEqual(capture["stages"]["gate"].shape, (9728,))
        self.rows[2]["token_width"] = 9727
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.read()

    def test_missing_stage_and_wrong_join_are_rejected(self):
        missing = self.rows.pop(2)
        with self.assertRaisesRegex(ValueError, "lacks all four"):
            self.read()
        self.rows.insert(2, missing)
        for row in self.rows:
            row["group_execution"] = 3
        with self.assertRaisesRegex(ValueError, "execution 2, column 0"):
            self.read()


if __name__ == "__main__":
    unittest.main()
