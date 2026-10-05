"""Frozen artifact identity excludes read access but rejects real mutation."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w1a1_eagle import nine_model_pipeline as pipeline  # noqa: E402


def changed_stat(value, **updates):
    fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
    return SimpleNamespace(**(fields | updates))


class FrozenFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "frozen.bin"
        self.path.write_bytes(b"frozen-content")
        self.record = {"path": str(self.path), "sha256": pipeline.sha256(self.path)}
        self.files = pipeline.Files()

    def test_hash_read_atime_change_is_accepted_and_identity_is_cached(self):
        original_stat, original_hash = Path.stat, pipeline.sha256
        accessed = False

        def observe(path, *args, **kwargs):
            stat = original_stat(path, *args, **kwargs)
            if path == self.path:
                return changed_stat(stat, st_atime=1 + int(accessed), st_atime_ns=1 + int(accessed))
            return stat

        def read_hash(path):
            nonlocal accessed
            result = original_hash(path)
            accessed = True
            return result

        with patch.object(Path, "stat", observe), patch.object(pipeline, "sha256", read_hash):
            self.assertEqual(self.files.check(self.record), self.path)
        self.assertEqual(len(self.files.cache), 1)
        self.assertEqual(len(self.files.identities), 1)

    def test_cached_reads_ignore_atime_without_rehashing(self):
        self.files.check(self.record)
        original_stat = Path.stat

        def observe(path, *args, **kwargs):
            stat = original_stat(path, *args, **kwargs)
            return changed_stat(stat, st_atime_ns=1) if path == self.path else stat

        with patch.object(Path, "stat", observe), patch.object(pipeline, "sha256") as hash_file:
            self.path.read_bytes()
            self.assertEqual(self.files.check(self.record), self.path)
            hash_file.assert_not_called()

    def test_real_stale_access_time_read_passes_on_current_filesystem(self):
        stat = self.path.stat()
        os.utime(self.path, ns=(1, stat.st_mtime_ns))
        self.assertEqual(self.files.check(self.record), self.path)
        self.path.read_bytes()
        self.assertEqual(self.files.check(self.record), self.path)

    def test_cached_byte_change_size_change_replacement_and_chmod_are_refused(self):
        original = self.path.read_bytes()
        for change in ("bytes", "size", "replacement", "mode", "mtime", "links"):
            with self.subTest(change=change):
                os.chmod(self.path, 0o600)
                self.path.write_bytes(original)
                files = pipeline.Files()
                files.check(self.record)
                if change == "bytes":
                    self.path.write_bytes(b"alteredcontent")
                elif change == "size":
                    self.path.write_bytes(original + b"extra")
                elif change == "replacement":
                    replacement = self.path.with_name("replacement.bin")
                    replacement.write_bytes(original)
                    replacement.replace(self.path)
                elif change == "mode":
                    os.chmod(self.path, 0o400)
                elif change == "mtime":
                    stat = self.path.stat()
                    os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                else:
                    link = self.path.with_name("linked.bin")
                    os.link(self.path, link)
                with self.assertRaisesRegex(ValueError, "artifact identity changed"):
                    files.check(self.record)
                if change == "links":
                    link.unlink()

    def test_cached_identity_checks_device_inode_permissions_owners_links_and_ns_times(self):
        self.files.check(self.record)
        original_stat = Path.stat
        actual = original_stat(self.path)
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_mtime_ns",
            "st_ctime_ns",
        ):

            def observe(path, *args, field=name, **kwargs):
                stat = original_stat(path, *args, **kwargs)
                return (
                    changed_stat(stat, **{field: getattr(actual, field) + 1})
                    if path == self.path
                    else stat
                )

            with (
                self.subTest(field=name),
                patch.object(Path, "stat", observe),
                self.assertRaisesRegex(ValueError, "artifact identity changed"),
            ):
                self.files.check(self.record)

    def test_hash_refuses_mid_read_mutation_and_never_caches_it(self):
        original_hash = pipeline.sha256

        def mutate(path):
            verified = original_hash(path)
            path.write_bytes(b"mutatedcontent")
            return verified

        with (
            patch.object(pipeline, "sha256", mutate),
            self.assertRaisesRegex(ValueError, "artifact changed during hash"),
        ):
            self.files.check(self.record)
        self.assertFalse(self.files.cache)
        self.assertFalse(self.files.identities)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.files.check(self.record)
        self.assertFalse(self.files.cache)

    def test_metadata_mutation_during_hash_is_refused_with_unchanged_content(self):
        original_hash = pipeline.sha256

        def mutate(path):
            verified = original_hash(path)
            os.chmod(path, 0o400)
            return verified

        with (
            patch.object(pipeline, "sha256", mutate),
            self.assertRaisesRegex(ValueError, "artifact changed during hash"),
        ):
            self.files.check(self.record)
        self.assertFalse(self.files.cache)
        self.assertFalse(self.files.identities)

    def test_cached_observation_race_is_refused(self):
        self.files.check(self.record)
        actual = self.files.fingerprint(self.path.stat())
        changed = (*actual[:-1], actual[-1] + 1)
        with (
            patch.object(self.files, "fingerprint", side_effect=[actual, changed]),
            self.assertRaisesRegex(ValueError, "artifact changed during cache check"),
        ):
            self.files.check(self.record)

    def test_fresh_checker_still_hashes_bytes_and_refuses_wrong_sha(self):
        self.path.write_bytes(b"alteredcontent")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.files.check(self.record)
        self.assertFalse(self.files.cache)
        self.assertFalse(self.files.identities)


class ObserverFileGuardTests(unittest.TestCase):
    def observe(self, mutation):
        helper = Path(pipeline.__file__).resolve().parents[2] / "scripts/read_only_dxg_census.py"
        expected = pipeline.sha256(helper)
        original_stat, original_text, original_link = Path.stat, Path.read_text, os.readlink
        accessed = False

        def stat(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == helper:
                changes = {"st_atime_ns": 2 if accessed else 1}
                if accessed and mutation:
                    changes[mutation] = getattr(result, mutation) + 1
                return changed_stat(result, **changes)
            return result

        def text(path, *args, **kwargs):
            if str(path) == "/proc/sys/kernel/random/boot_id":
                return "fixture-boot"
            return original_text(path, *args, **kwargs)

        def link(path, *args, **kwargs):
            if str(path) == "/proc/self/ns/pid":
                return "pid:[fixture]"
            return original_link(path, *args, **kwargs)

        def census(*args, **kwargs):
            nonlocal accessed
            accessed = True
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "schema": "nine_model_read_only_dxg_census_v1",
                        "complete": True,
                        "read_only": True,
                        "effective_uid": 0,
                        "proc_root": "/proc",
                        "observer_source_sha256": expected,
                        "boot_id": "fixture-boot",
                        "pid_namespace": "pid:[fixture]",
                        "holders": [],
                    }
                )
            )

        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}),
            patch.object(Path, "stat", stat),
            patch.object(Path, "read_text", text),
            patch.object(os, "readlink", link),
            patch.object(pipeline.subprocess, "run", census),
        ):
            return pipeline._privileged_dxg_holders()

    def test_read_only_observer_access_time_change_is_accepted(self):
        self.assertEqual(self.observe(None), [])

    def test_read_only_observer_content_identity_changes_still_refuse(self):
        for field in ("st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "release PENDING"):
                self.observe(field)


if __name__ == "__main__":
    unittest.main()
