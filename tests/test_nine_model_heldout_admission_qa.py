"""Independent source-join QA using synthetic opaque metadata only."""

import json
import tempfile
import unittest
from pathlib import Path

import test_nine_model_heldout_admission as heldout


class HeldoutSourceJoinQATests(unittest.TestCase):
    def test_eagle_capture_positions_and_index_anchor_are_verified(self):
        for change in ("position", "index_hash"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                value, source, contexts, exports, validators = heldout.HeldoutTests().fixture(root)
                context = contexts["eagle_a8"]
                config_path = Path(context["spec"]["eagle_config"]["path"])
                config = json.loads(config_path.read_text())
                capture = config["stages"]["captures"][0]
                if change == "position":
                    capture["source_positions"] = [1]
                else:
                    capture["source_index_sha256"] = "f" * 64
                updated = heldout.write(root, "eagle_a8/continuous", config)
                context["spec"]["eagle_config"] = updated
                with self.assertRaises(ValueError):
                    heldout.HeldoutTests().validate(value, source, validators)

    def test_block_trainer_membership_must_join_original_train_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value, source, contexts, exports, validators = heldout.HeldoutTests().fixture(root)
            context = contexts["dspark_a8"]
            data_path = Path(context["spec"]["data"]["path"])
            data = json.loads(data_path.read_text())
            data["chains"][0]["prompt_id"] = "final-id"
            updated = heldout.write(root, "dspark_a8/data", data)
            context["spec"]["data"] = updated
            context["source_bindings"]["data_manifest_sha256"] = updated["sha256"]
            selection = source["final_selection"]
            selection["origins"] = heldout.c.final_origins(value, contexts, exports)
            heldout.HeldoutTests().receipts(root, value, source, contexts, exports)
            with self.assertRaisesRegex(
                ValueError, "block actual TRAIN ID/content/index join differs"
            ):
                heldout.HeldoutTests().validate(value, source, validators)


if __name__ == "__main__":
    unittest.main()
