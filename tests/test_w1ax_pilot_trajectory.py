from __future__ import annotations

import importlib.util
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
