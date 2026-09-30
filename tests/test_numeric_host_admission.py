"""CPU-only evidence and unchanged numeric model-load admission."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_continuous_w1ax_readiness as checker


class NumericHostAdmissionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_snapshot_records_available_rss_and_anonymous_rss(self):
        (self.root / "self").mkdir()
        (self.root / "meminfo").write_text("MemAvailable: 123 kB\nMemTotal: 456 kB\n")
        (self.root / "self/status").write_text("VmRSS: 34 kB\nRssAnon: 12 kB\n")
        result = checker._host_memory_snapshot(self.root)
        self.assertEqual(result["status"], "recorded")
        self.assertEqual(result["host_available_bytes"], 123 * 1024)
        self.assertEqual(result["process_rss_bytes"], 34 * 1024)
        self.assertEqual(result["process_rss_anon_bytes"], 12 * 1024)

    def test_unavailable_snapshot_is_explicit(self):
        self.assertEqual(checker._host_memory_snapshot(self.root)["status"], "unavailable")

    def test_optional_glibc_trim_records_return_without_claiming_admission(self):
        for return_code in (0, 1):
            with self.subTest(return_code=return_code):
                libc = Mock()
                libc.gnu_get_libc_version.return_value = b"2.39"
                libc.malloc_trim.return_value = return_code
                with patch.object(checker.sys, "platform", "linux"), patch.object(
                    checker.ctypes, "CDLL", return_value=libc
                ):
                    result = checker._trim_host_allocator()
                self.assertEqual(result, {
                    "status": "called", "glibc_version": "2.39", "return_code": return_code
                })
                libc.malloc_trim.assert_called_once_with(0)
                self.assertEqual(libc.malloc_trim.argtypes, [checker.ctypes.c_size_t])
                self.assertEqual(libc.malloc_trim.restype, checker.ctypes.c_int)

    def test_trim_unavailable_is_optional(self):
        with patch.object(checker.sys, "platform", "darwin"), patch.object(
            checker.ctypes, "CDLL"
        ) as load:
            self.assertEqual(checker._trim_host_allocator()["status"], "unavailable")
            load.assert_not_called()
        for error in (OSError("library missing"), AttributeError("malloc_trim missing")):
            with self.subTest(error=str(error)), patch.object(checker.sys, "platform", "linux"), patch.object(
                checker.ctypes, "CDLL", side_effect=error
            ):
                self.assertEqual(checker._trim_host_allocator()["status"], "unavailable")
        with patch.object(checker.sys, "platform", "linux"), patch.object(
            checker.ctypes, "CDLL", return_value=object()
        ):
            self.assertEqual(checker._trim_host_allocator()["status"], "unavailable")

    def test_trim_call_failure_is_recorded(self):
        libc = Mock()
        libc.gnu_get_libc_version.return_value = b"2.39"
        libc.malloc_trim.side_effect = OSError("trim failed")
        with patch.object(checker.sys, "platform", "linux"), patch.object(
            checker.ctypes, "CDLL", return_value=libc
        ):
            self.assertEqual(checker._trim_host_allocator(), {
                "status": "failed", "glibc_version": "2.39", "reason": "trim failed"
            })

    def test_diagnostic_precedes_original_admission_even_on_failure(self):
        before = {"status": "recorded", "host_available_bytes": 13 * 1024**3}
        after = {"status": "recorded", "host_available_bytes": 14 * 1024**3}
        for trim in (
            {"status": "called", "return_code": 1},
            {"status": "unavailable"},
            {"status": "failed", "reason": "trim failed"},
        ):
            for fail in (False, True):
                events = []
                def snapshot():
                    events.append("snapshot")
                    return before if len(events) == 1 else after
                def admission(stage, amount):
                    events.append("admission")
                    diagnostic = json.loads((self.root / filename).read_text())
                    self.assertEqual(diagnostic["before"], before)
                    self.assertEqual(diagnostic["after"], after)
                    self.assertEqual(diagnostic["malloc_trim"], trim)
                    self.assertEqual(diagnostic["gc_collected_objects"], 7)
                    self.assertEqual(stage, expected_stage)
                    self.assertEqual(diagnostic["admission_stage"], expected_stage)
                    self.assertEqual(amount, 12 * 1024**3)
                    if fail:
                        raise RuntimeError("unchanged admission failed")
                    return {"accepted": True}
                with self.subTest(trim=trim, fail=fail), patch.object(
                    checker, "_host_memory_snapshot", side_effect=snapshot
                ), patch.object(checker.gc, "collect", side_effect=lambda: events.append("gc") or 7), patch.object(
                    checker, "_trim_host_allocator", side_effect=lambda: events.append("trim") or trim
                ), patch.object(checker, "host_admission", side_effect=admission):
                    for expected_stage, filename in (
                        ("bounded checkpoint-zero CPU initialization", "host-memory-checkpoint-zero-admission.json"),
                        ("bounded numeric gate CPU target/draft load", "host-memory-numeric-admission.json"),
                    ):
                        events.clear()
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "unchanged admission failed"):
                                checker._reclaim_and_admit_host(self.root, expected_stage, filename)
                        else:
                            self.assertEqual(
                                checker._reclaim_and_admit_host(self.root, expected_stage, filename),
                                {"accepted": True},
                            )
                        self.assertEqual(events, ["snapshot", "gc", "trim", "snapshot", "admission"])


if __name__ == "__main__":
    unittest.main()
