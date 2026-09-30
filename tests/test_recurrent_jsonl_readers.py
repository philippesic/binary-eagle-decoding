"""JSONL record delimiters preserve valid Unicode prompt content."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_recurrent_binary_capture import read_jsonl as audit_read_jsonl
from prepare_recurrent_native_rows import read_jsonl as native_read_jsonl


class RecurrentJsonlReaderTests(unittest.TestCase):
    readers = (audit_read_jsonl, native_read_jsonl)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "prompts.jsonl"

    def assert_readers(self, expected):
        for reader in self.readers:
            with self.subTest(reader=reader.__module__):
                self.assertEqual(reader(self.path), expected)

    def test_unicode_separators_are_string_content(self):
        for separator in ("\u2028", "\u2029", "\u0085"):
            with self.subTest(separator=repr(separator)):
                records = [
                    {"id": "first", "prompt": f"before{separator}middle{separator}after"},
                    {"id": "second", "prompt": "ordinary"},
                ]
                self.path.write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
                    encoding="utf-8",
                )
                self.assert_readers(records)

    def test_lf_crlf_and_final_record_without_newline(self):
        records = [{"id": "first"}, {"id": "second"}]
        for delimiter in (b"\n", b"\r\n"):
            for trailing_newline in (False, True):
                with self.subTest(delimiter=delimiter, trailing=trailing_newline):
                    data = delimiter.join(json.dumps(row).encode() for row in records)
                    self.path.write_bytes(data + (delimiter if trailing_newline else b""))
                    self.assert_readers(records)

    def test_blank_lines_preserve_existing_skip_behavior(self):
        self.path.write_text('\n \t\n{"id":"first"}\n\n', encoding="utf-8")
        self.assert_readers([{"id": "first"}])

    def test_empty_or_non_object_rows_are_rejected(self):
        for content in ("", "\n \t\n", "[]\n", "null\n", '1\n', '{"id":"first"}\n[]\n'):
            self.path.write_text(content, encoding="utf-8")
            for reader in self.readers:
                with self.subTest(content=content, reader=reader.__module__):
                    with self.assertRaises(ValueError):
                        reader(self.path)

    def test_malformed_json_rows_are_rejected(self):
        for content in ('{"id":"first"}\n{"id":\n', '{"prompt":"literal\nnewline"}\n'):
            self.path.write_text(content, encoding="utf-8")
            for reader in self.readers:
                with self.subTest(content=content, reader=reader.__module__):
                    with self.assertRaises(json.JSONDecodeError):
                        reader(self.path)


if __name__ == "__main__":
    unittest.main()
