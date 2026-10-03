"""Shared provider weight hashes; tiny CPU files, no model/capture execution."""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import w1ax_continuous_stages as stages  # noqa: E402


class ProviderCommonHashTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        stages._VERIFIED_RECORDS.clear()
        self.addCleanup(stages._VERIFIED_RECORDS.clear)
        self.sources = {}
        for name in (*stages.PROVIDER_SHARED_WEIGHTS, "absolute_d2t", "binary"):
            path = self.root / name
            path.write_bytes((name + ": immutable synthetic CPU data").encode())
            self.sources[name] = str(path)
        snapshot = self.root / "snapshot.json"
        snapshot.write_text(json.dumps({"models": {
            name: {"directory": str(self.root / (name + "-model"))} for name in ("target", "draft")
        }}))
        self.sources["model_snapshot_manifest"] = str(snapshot)
        self.original_sha = stages.sha256
        self.sources["sha256"] = {name: self.original_sha(Path(path)) for name, path in self.sources.items()}
        self.common = {name: self.sources["sha256"][name] for name in stages.PROVIDER_SHARED_WEIGHTS}
        self.ready = self.root / "readiness.json"
        self.ready.write_text('{"CPU_fixture": true}')
        self.captures = []
        for ordinal in range(3):
            folder = self.root / f"capture-{ordinal}"
            folder.mkdir()
            (folder / "prompts.jsonl").write_text(json.dumps({"id": f"train-{ordinal}"}) + "\n")
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"split": "train", "prompt_count": 1,
                                           "files": {"prompts": {"path": "prompts.jsonl"}}}))
            self.captures.append(manifest)

    def manifest(self, ordinal, *, cached=True, suffix="", common=None):
        output = self.root / f"provider-{ordinal}{suffix}.json"
        kwargs = {"common_source_sha256": self.common if common is None else common} if cached else {}
        spec = stages.provider_manifest(self.sources, self.captures[ordinal], self.ready, output, **kwargs)
        return spec, output

    def spy(self, counter):
        def digest(path):
            counter[Path(path)] += 1
            return self.original_sha(path)
        return digest

    def test_shared_weights_hash_once_across_shards_owned_inputs_still_hash(self):
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            for ordinal in range(3):
                self.manifest(ordinal)
        for name in stages.PROVIDER_SHARED_WEIGHTS:
            self.assertEqual(counts[Path(self.sources[name])], 1)
        for capture in self.captures:
            self.assertEqual(counts[capture], 2)  # Preserve capture_id and owned manifest hashing.
            self.assertEqual(counts[capture.parent / "prompts.jsonl"], 1)
        self.assertEqual(counts[Path(self.sources["absolute_d2t"])], 3)

    def test_verify_sources_cache_is_reused_by_provider_publication(self):
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            stages.verify_sources(self.sources)
            for ordinal in range(3):
                self.manifest(ordinal)
        for name in stages.PROVIDER_SHARED_WEIGHTS:
            self.assertEqual(counts[Path(self.sources[name])], 1)

    def test_cached_and_legacy_manifest_bytes_are_identical(self):
        for ordinal in range(3):
            cached, cached_path = self.manifest(ordinal)
            legacy, legacy_path = self.manifest(ordinal, cached=False, suffix="-legacy")
            self.assertEqual(cached, legacy)
            self.assertEqual(cached_path.read_bytes(), legacy_path.read_bytes())

    def test_default_legacy_path_still_hashes_each_shared_file_each_time(self):
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            for ordinal in range(3):
                self.manifest(ordinal, cached=False)
        for name in stages.PROVIDER_SHARED_WEIGHTS:
            self.assertEqual(counts[Path(self.sources[name])], 3)

    def test_same_length_mutation_with_restored_mtime_rehashes_and_refuses(self):
        path = Path(self.sources["target_gguf"])
        self.manifest(0)
        initial = path.stat()
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        self.assertEqual(path.stat().st_size, initial.st_size)
        self.assertEqual(path.stat().st_mtime_ns, initial.st_mtime_ns)
        self.assertNotEqual(path.stat().st_ctime_ns, initial.st_ctime_ns)
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.manifest(1)
        self.assertEqual(counts[path], 1)
        self.assertFalse((self.root / "provider-1.json").exists())

    def test_size_change_rehashes_and_refuses(self):
        path = Path(self.sources["target_gguf"])
        self.manifest(0)
        path.write_bytes(path.read_bytes() + b"x")
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.manifest(1)
        self.assertEqual(counts[path], 1)

    def test_same_bytes_replacement_inode_rehashes_before_acceptance(self):
        path = Path(self.sources["target_gguf"])
        self.manifest(0)
        initial = path.stat()
        replacement = self.root / "replacement.gguf"
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        os.replace(replacement, path)
        self.assertNotEqual(path.stat().st_ino, initial.st_ino)
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            self.manifest(1)
        self.assertEqual(counts[path], 1)

    def test_symlink_replacement_is_rejected_even_if_target_bytes_match(self):
        path = Path(self.sources["target_gguf"])
        self.manifest(0)
        target = self.root / "link-target.gguf"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "absolute regular file"):
            self.manifest(1)
        self.assertFalse((self.root / "provider-1.json").exists())

    def test_mutation_during_initial_hash_does_not_cache_or_publish(self):
        path = Path(self.sources["target_gguf"])
        original = path.read_bytes()
        def changing_hash(value):
            digest = self.original_sha(value)
            if Path(value) == path:
                path.write_bytes(original + b"x")
            return digest
        with patch.object(stages, "sha256", side_effect=changing_hash):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                self.manifest(0)
        self.assertFalse(any(key[0] == str(path) for key in stages._VERIFIED_RECORDS))
        self.assertFalse((self.root / "provider-0.json").exists())
        path.write_bytes(original)
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            self.manifest(1)
        self.assertEqual(counts[path], 1)

    def test_identity_change_with_same_content_during_hash_is_rejected(self):
        path = Path(self.sources["target_gguf"])
        def replacing_hash(value):
            digest = self.original_sha(value)
            if Path(value) == path:
                other = self.root / "replacement.gguf"
                other.write_bytes(path.read_bytes())
                os.replace(other, path)
            return digest
        with patch.object(stages, "sha256", side_effect=replacing_hash):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                self.manifest(0)
        self.assertFalse(any(key[0] == str(path) for key in stages._VERIFIED_RECORDS))

    def test_new_process_cache_must_authenticate_bytes_again(self):
        self.manifest(0)
        stages._VERIFIED_RECORDS.clear()
        counts = Counter()
        with patch.object(stages, "sha256", side_effect=self.spy(counts)):
            self.manifest(1)
        for name in stages.PROVIDER_SHARED_WEIGHTS:
            self.assertEqual(counts[Path(self.sources[name])], 1)

    def test_context_requires_exact_inventory_and_actual_pinned_digests(self):
        for wrong in ({}, {"target_gguf": self.common["target_gguf"]}, {**self.common, "prompts": "0"*64}):
            with self.subTest(context=wrong), self.assertRaisesRegex(ValueError, "hash inventory"):
                self.manifest(0, common=wrong)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.manifest(0, common={**self.common, "target_gguf": "0"*64})
        with self.assertRaisesRegex(ValueError, "lowercase SHA256"):
            self.manifest(0, common={**self.common, "target_gguf": "bad"})


if __name__ == "__main__":
    unittest.main()
