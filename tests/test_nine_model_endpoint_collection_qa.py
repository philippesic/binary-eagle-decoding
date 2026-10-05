"""Focused source-control binding regression test; no model or runtime access."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_nine_model_endpoint_collection as fixtures

from w1a1_eagle import nine_model_endpoint_collection as collection


class OriginalControlSourceTests(unittest.TestCase):
    def test_block_controls_require_a_source_bundle_binding(self):
        fixture = fixtures.CollectionTests()
        with tempfile.TemporaryDirectory() as tmp:
            value, source, _, _, validators = fixture.fixture(Path(tmp))
            del source["controls"]["dspark"]
            with patch(
                "prepare_nine_model_lane_endpoint.ORIGINAL_EAGLE_Q4_SHA256",
                value["controls"]["eagle"]["model"]["sha256"],
            ):
                with self.assertRaisesRegex(ValueError, "original control source PENDING: dspark"):
                    collection.validate_collection(
                        value,
                        validators=validators,
                        source_loader=lambda locator, files: source,
                    )


if __name__ == "__main__":
    unittest.main()
