"""Declared optimization profiles validate without model or GPU discovery."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.prepare_qat_optimization_recipe import prepare
from w1a1_eagle.continuous_qat import ContinuousConfig

ROOT = Path(__file__).resolve().parents[1]


class ProfilesTests(unittest.TestCase):
    def test_every_declared_profile_is_valid_preparation_only(self):
        base = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())
        profiles = json.loads((ROOT / "configs/qat_optimization_profiles.json").read_text())
        original = json.dumps(base, sort_keys=True)
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
            for name in profiles["profiles"]:
                result = prepare(base, profiles, name, "a" * 40)
                config = ContinuousConfig(**result["training"])
                self.assertIsNone(config.optimization_readiness)
                self.assertEqual(result["provider"], base["provider"])
                self.assertEqual(result["hardware"], base["hardware"])
                self.assertEqual(result["native"]["expected_commit"], "a" * 40)
        self.assertEqual(original, json.dumps(base, sort_keys=True))
        with self.assertRaises(ValueError):
            prepare(base, profiles, "unknown", "a" * 40)
        with self.assertRaises(ValueError):
            prepare(base, profiles, "speed", "short")


if __name__ == "__main__":
    unittest.main()
