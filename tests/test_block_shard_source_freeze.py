"""Integration regression against an actual frozen 10000-prompt CPU schedule.

Set DSPARK_CAPTURE_SCHEDULE to the existing full-pool-schedule.json. The artifact
is deliberately external to Git, like the authenticated original prompt indexes.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w1a1_eagle.block_shard_lifecycle import (  # noqa: E402
    FrozenShardPlan,
    atomic_json,
    deployed_source_pins,
    freeze_capture_schedule,
    open_provider,
    pin,
    validate_deployed_source,
)


class FullScheduleFreezeTests(unittest.TestCase):
    def test_actual_schedule_factory_pins_and_opens_deployed_provider(self):
        location = os.environ.get("DSPARK_CAPTURE_SCHEDULE")
        if not location:
            self.skipTest("set DSPARK_CAPTURE_SCHEDULE to authenticated 10000-prompt schedule")
        schedule = Path(location).resolve()
        value = freeze_capture_schedule(pin(schedule), plan_root=schedule.parent)
        self.assertEqual(len(value["chains"]), 10000)
        plan = FrozenShardPlan(value)
        validate_deployed_source(plan)
        for name, digest in deployed_source_pins().items():
            self.assertEqual(value["source"][name], digest)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "logical-plan.json"
            atomic_json(path, value)
            provider = open_provider(
                {
                    "provider": "rotating_block_v1",
                    **pin(path),
                    "cache_root": str(root / "train-cache"),
                }
            )
            try:
                self.assertEqual(provider.sha256, plan.sha256)
                self.assertEqual(
                    provider.plan.role_counts,
                    {"train": 9856, "calibration_fit": 96, "calibration_validation": 48},
                )
                self.assertEqual(len(provider.order("train", 8101)), 9856)
            finally:
                provider.close()


if __name__ == "__main__":
    unittest.main()
