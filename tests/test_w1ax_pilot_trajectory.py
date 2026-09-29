from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_w1ax_pilot_trajectory", ROOT / "scripts/check_w1ax_pilot_trajectory.py"
)
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)


class TrajectoryGateTests(unittest.TestCase):
    def test_checkpoint_loader_maps_weight_names_to_module_paths(self):
        class Linear:
            def __init__(self):
                self.latent_sign = torch.nn.Parameter(torch.zeros((2, 4)))
                self.initial_scale = torch.tensor([0.25, 0.5])
                self.scale_offset = torch.nn.Parameter(torch.zeros(2))

        linears = {
            name.removesuffix(".weight"): Linear() for name in trajectory.CHECKPOINT_NAMES.values()
        }
        base_hash = "a" * 64
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "joint.npz"
            manifest = root / "joint.json"
            arrays = {}
            for name in trajectory.CHECKPOINT_NAMES.values():
                arrays[name + ".latent"] = np.full((2, 4), 0.75, dtype=np.float32)
                arrays[name + ".scale"] = np.array([0.4, 0.8], dtype=np.float32)
            np.savez(checkpoint, **arrays)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "scale_layout": "row",
                        "activation_bits": 16,
                        "objective": "hard_ce",
                        "base_gguf_sha256": base_hash,
                        "checkpoint_sha256": trajectory.sha256(checkpoint),
                    }
                )
            )

            trajectory._load_row_checkpoint(checkpoint, manifest, linears, base_hash)

        for module in linears.values():
            torch.testing.assert_close(module.latent_sign, torch.full((2, 4), 0.75))
            torch.testing.assert_close(module.scale_offset, torch.tensor([0.15, 0.3]))

    def test_relative_rms_uses_native_reference_floor(self):
        self.assertAlmostEqual(
            trajectory.relative_rms(np.array([2.0, 4.0]), np.array([1.0, 2.0])), 1.0
        )
        self.assertEqual(trajectory.relative_rms(np.array([1e-9]), np.array([0.0])), 0.1)

    def test_top_two_margin_is_raw_logit_difference(self):
        self.assertAlmostEqual(
            trajectory.top_two_margin(torch.tensor([3.1, 3.0, -2.0])), 0.1, places=6
        )
        with self.assertRaises(ValueError):
            trajectory.top_two_margin(torch.tensor([float("nan"), 0.0]))

    def test_cache_contract_requires_finite_f16_exact_contiguous_rows(self):
        cache = SimpleNamespace(
            key=torch.zeros((8, 2, 128), dtype=torch.float32),
            value=torch.ones((8, 2, 128), dtype=torch.float32),
        )
        result = trajectory.check_cache_contract(cache, 2, torch.device("cpu"))
        self.assertEqual(result["key"]["shape"], [8, 2, 128])
        cache.key[0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "F16"):
            trajectory.check_cache_contract(cache, 2, torch.device("cpu"))

    def test_join_uses_full_prefix_and_first_two_common_roots(self):
        prefixes = [(10, 11), (10, 11, 12), (10, 11, 13)]
        anchors = {}
        round_data = {}
        for i, prefix in enumerate(prefixes):
            key = ("dolly:one", i)
            anchors[key] = object()
            round_data[key] = SimpleNamespace(
                prefix_token_ids=prefix,
                anchor=SimpleNamespace(prefix_token_ids=prefix[:-1], seed_token_id=prefix[-1]),
            )

        class Capture:
            pass

        capture = Capture()
        capture.anchors = anchors
        capture.round_inputs = lambda prompt, index: round_data[(prompt, index)]
        heads = [
            {
                "task_id": 1,
                "depth": 0,
                "parent_position": 3,
                "round_index": 2,
                "prefix_token_ids": [10, 11, 13],
            },
            {
                "task_id": 1,
                "depth": 0,
                "parent_position": 1,
                "round_index": 0,
                "prefix_token_ids": [10, 11],
            },
            {
                "task_id": 1,
                "depth": 1,
                "parent_position": 2,
                "round_index": 1,
                "prefix_token_ids": [10, 11, 12],
            },
            # This row differs at the final token and must not join by a prefix subset.
            {
                "task_id": 1,
                "depth": 0,
                "parent_position": 2,
                "round_index": 1,
                "prefix_token_ids": [10, 11, 99],
            },
        ]
        joined, _, _, _ = trajectory.join_roots(heads, {1: "dolly:one"}, capture)
        joined = joined["dolly:one"]
        self.assertEqual(
            [row["prefix_token_ids"] for row, _, _ in joined], [[10, 11], [10, 11, 13]]
        )

    def test_packed_a16_head_replay_matches_explicit_sign_linear(self):
        signs = np.array([[1, 0, 1, 1, 0], [0, 0, 1, 0, 1]], dtype=np.uint8)
        padded = np.pad(signs, ((0, 0), (0, 27)))
        packed = np.packbits(padded, axis=1, bitorder="little").view("<u4")
        scales = np.array([0.25, 1.5], dtype=np.float32)
        state = torch.tensor([0.1011, -0.2123, 0.3333, -0.4444, 0.5555])
        got = trajectory.packed_head_replay(packed, scales, 5, state, torch.device("cpu"))
        expected_weights = torch.tensor(signs, dtype=torch.float32) * 2 - 1
        expected = torch.nn.functional.linear(
            state.half().float(), expected_weights
        ) * torch.from_numpy(scales)
        torch.testing.assert_close(got, expected)

    def test_domain_contract_requires_one_prompt_per_frozen_domain(self):
        domains = trajectory._validate_three_domains(
            {1: "dolly:line-005896", 2: "gsm8k:train-000315", 3: "mbpp:task-496"}
        )
        self.assertEqual(set(domains), {"prose", "reasoning", "code"})
        with self.assertRaises(ValueError):
            trajectory._validate_three_domains(
                {1: "dolly:one", 2: "gsm8k:one", 3: "mbpp:one", 4: "dolly:two"}
            )


if __name__ == "__main__":
    unittest.main()
