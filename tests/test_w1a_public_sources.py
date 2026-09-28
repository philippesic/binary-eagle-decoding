"""Offline provenance check for the public-source adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_w1a_public_sources import build_catalog  # noqa: E402
from prepare_w1a_data import sha256  # noqa: E402


class PublicSourcesTest(unittest.TestCase):
    def test_pinned_offline_source_and_transform_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "raw/dolly.jsonl"
            raw.parent.mkdir()
            raw.write_text(
                json.dumps(
                    {
                        "category": "summarization",
                        "instruction": "Summarize the note.",
                        "context": "A short local fixture note.",
                        "response": "Unused reference answer.",
                    }
                )
                + "\n"
            )
            spec = {
                "dolly": {
                    "uri": "fixture://dolly",
                    "revision": "test-revision",
                    "license": "fixture-only",
                    "upstream_sha256": sha256(raw),
                    "domain": "prose",
                }
            }
            with patch.dict("fetch_w1a_public_sources.SOURCES", spec, clear=True):
                result = build_catalog(directory, offline=True)
            catalog = json.loads(result["catalog"].read_text())
            source = catalog["sources"][0]
            self.assertEqual(source["upstream_sha256"], sha256(raw))
            self.assertEqual(source["sha256"], sha256(directory / source["path"]))
            self.assertEqual(len(source["transform_sha256"]), 64)
            transformed = json.loads((directory / source["path"]).read_text())
            self.assertEqual(transformed["messages"][0]["role"], "user")
            self.assertNotIn("Unused reference answer.", transformed["messages"][0]["content"])
            with patch.dict("fetch_w1a_public_sources.SOURCES", spec, clear=True):
                with self.assertRaisesRegex(ValueError, "exists"):
                    build_catalog(directory, offline=True)
            raw.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "wrong SHA256"):
                # A new directory is not required to verify a cached raw hash.
                from fetch_w1a_public_sources import download

                download("dolly", directory / "raw", offline=True)


if __name__ == "__main__":
    unittest.main()
