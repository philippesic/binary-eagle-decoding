"""CPU fixtures for deterministic packet packaging and sealed exclusion."""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pack_continuous_w1ax_inputs import pack, safe_name  # noqa: E402
from prepare_w1a_data import canonical, sha256  # noqa: E402


class PacketTests(unittest.TestCase):
    def fixture(self, base):
        corpus = base / "freeze-fixture"
        corpus.mkdir()
        lock = base / "lock.json"
        lock.write_text('{"schema":"fixture"}\n')
        files = {}
        for name in ["train", "dev", "sealed_test", "train_reserve"]:
            index = corpus / f"{name}-00000.index.jsonl"
            index.write_bytes(canonical({"id": name, "content_sha256": "a" * 64}) + b"\n")
            prompt = corpus / f"{name}-00000.jsonl"
            prompt.write_bytes(
                canonical({"id": name, "messages": [{"role": "user", "content": name}]}) + b"\n"
            )
            files[name] = {
                "prompts_count": 1,
                "shards": [
                    {
                        "index": index.name,
                        "index_sha256": sha256(index),
                        "prompts": prompt.name,
                        "prompts_sha256": sha256(prompt),
                        "prompts_count": 1,
                    }
                ],
            }
        excluded = base / "legacy.index.jsonl"
        excluded.write_bytes(canonical({"id": "old", "content_sha256": "b" * 64}) + b"\n")
        manifest = corpus / "manifest.json"
        manifest.write_bytes(
            canonical(
                {
                    "schema": "continuous_w1ax_prompt_manifest_v1",
                    "source_lock_sha256": sha256(lock),
                    "files": files,
                    "opaque_exclusions": [
                        {"file": excluded.name, "sha256": sha256(excluded), "rows": 1}
                    ],
                }
            )
            + b"\n"
        )
        return manifest, lock, excluded

    def test_deterministic_and_never_read_sealed_payload(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            manifest, lock, excluded = self.fixture(base)
            real_open = Path.open

            def guarded(path, mode="r", *args, **kwargs):
                if (
                    path.name.startswith("sealed_test-")
                    and ".index." not in path.name
                    and "r" in mode
                ):
                    raise AssertionError("sealed payload opened")
                return real_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", guarded):
                a = pack(manifest, lock, base / "one.tar", [excluded])
                b = pack(manifest, lock, base / "two.tar", [excluded])
                reserve = pack(manifest, lock, base / "reserve.tar", [excluded], True)
            self.assertEqual(a, b)
            self.assertEqual((base / "one.tar").read_bytes(), (base / "two.tar").read_bytes())
            self.assertEqual(a["prompt_counts"], {"train": 1, "dev": 1})
            self.assertEqual(reserve["prompt_counts"]["train_reserve"], 1)
            with tarfile.open(base / "one.tar") as archive:
                names = archive.getnames()
                self.assertIn("data/continuous-w1ax/freeze-fixture/manifest.json", names)
                self.assertNotIn(
                    "data/continuous-w1ax/freeze-fixture/sealed_test-00000.jsonl", names
                )
                self.assertNotIn(
                    "data/continuous-w1ax/freeze-fixture/train_reserve-00000.jsonl", names
                )
                self.assertIn(
                    "data/continuous-w1ax/freeze-fixture/sealed_test-00000.index.jsonl", names
                )
                for member in archive.getmembers():
                    self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                    self.assertFalse(member.name.startswith("/"))
                    self.assertNotIn("..", Path(member.name).parts)
            with self.assertRaisesRegex(ValueError, "already exists"):
                pack(manifest, lock, base / "one.tar", [excluded])

    def test_wrong_exclusions_and_tampered_train_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            manifest, lock, excluded = self.fixture(base)
            with self.assertRaisesRegex(ValueError, "match original manifest"):
                pack(manifest, lock, base / "missing.tar", [])
            (manifest.parent / "train-00000.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                pack(manifest, lock, base / "tampered.tar", [excluded])
            self.assertFalse((base / "tampered.tar").exists())

    def test_traversal_and_sealed_mislabel_rejected(self):
        for value in ["../sealed.jsonl", "/sealed.jsonl", "dir/file.jsonl", "..", "dir\\file"]:
            with self.assertRaises(ValueError):
                safe_name(value)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            manifest, lock, excluded = self.fixture(base)
            content = json.loads(manifest.read_text())
            content["files"]["train"]["shards"][0]["prompts"] = "sealed_test-00000.jsonl"
            manifest.write_bytes(canonical(content) + b"\n")
            with self.assertRaisesRegex(ValueError, "allowed split"):
                pack(manifest, lock, base / "mislabel.tar", [excluded])


if __name__ == "__main__":
    unittest.main()
