"""CPU crash accounting checks with fixed clocks; no hardware access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w1a1_eagle.continuous_budget import TrainingBudget
from w1a1_eagle.continuous_qat import atomic_json


class BudgetTests(unittest.TestCase):
    def test_hard_crash_charges_after_checkpoint_and_downtime_without_double_charge(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("w1a1_eagle.continuous_budget.boot_identity", return_value="boot"),
            patch("w1a1_eagle.continuous_budget.process_birth", return_value=None),
            patch("w1a1_eagle.continuous_budget.time.monotonic") as clock,
        ):
            path = Path(folder) / "budget.json"
            clock.return_value = 100
            first = TrainingBudget(path, "source", 7200, atomic_json)
            first.begin(20)
            self.assertIsNotNone(json.loads(path.read_text())["active_attempt"])
            clock.return_value = 135
            resumed = TrainingBudget(path, "source", 7200, atomic_json)
            self.assertEqual(resumed.load(30), 55)  # 20 + entire unresolved 35s
            self.assertEqual(resumed.load(30), 55)  # already settled exactly once
            resumed.begin(30)
            clock.return_value = 145
            self.assertEqual(resumed.finish(), 65)
            self.assertIsNone(json.loads(path.read_text())["active_attempt"])

    def test_changed_boot_consumes_remaining_budget(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("w1a1_eagle.continuous_budget.boot_identity", return_value="old") as boot,
            patch("w1a1_eagle.continuous_budget.time.monotonic", return_value=10),
        ):
            path = Path(folder) / "budget.json"
            first = TrainingBudget(path, "source", 7200, atomic_json)
            first.begin(7000)
            boot.return_value = "new"
            self.assertEqual(TrainingBudget(path, "source", 7200, atomic_json).load(7001), 7200)

    def test_live_owner_and_budget_extension_rejected(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("w1a1_eagle.continuous_budget.boot_identity", return_value="boot"),
            patch("w1a1_eagle.continuous_budget.process_birth", return_value="birth"),
        ):
            path = Path(folder) / "budget.json"
            first = TrainingBudget(path, "source", 7200, atomic_json)
            first.begin(0)
            with self.assertRaisesRegex(ValueError, "still alive"):
                TrainingBudget(path, "source", 7200, atomic_json).load(0)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                TrainingBudget(path, "source", 8000, atomic_json).load(0)

    def test_normal_finish_excludes_later_standalone_evaluation(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("w1a1_eagle.continuous_budget.time.monotonic") as clock,
        ):
            path = Path(folder) / "budget.json"
            clock.return_value = 10
            first = TrainingBudget(path, "source", 7200, atomic_json)
            first.begin(0)
            clock.return_value = 20
            self.assertEqual(first.finish(), 10)
            clock.return_value = 1200
            self.assertEqual(TrainingBudget(path, "source", 7200, atomic_json).load(0), 10)
