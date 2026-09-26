"""GPU-free contracts for deterministic W1Ax capture selection."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_w1ax_captures", ROOT / "scripts/select_w1ax_captures.py"
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def write_capture(path: Path, sequence: int, name: str, n: int, *, k: int = 2, m: int = 3):
    encoded_name = name.encode("utf-8")
    header = selector.HEADER.pack(
        selector.MAGIC,
        sequence,
        k,
        m,
        n,
        1,
        encoded_name.ljust(128, b"\0"),
    )
    path.write_bytes(header + bytes(k * n * 4))


class W1AxCaptureSelectionTests(unittest.TestCase):
    def make_capture_set(self, directory: Path):
        sequence = 0
        for name in sorted(selector.REQUIRED_TENSORS):
            for n, count in ((1, 5), (2, 3), (37, 4), (38, 2)):
                for _ in range(count):
                    write_capture(directory / f"op-{sequence:012d}.bin", sequence, name, n)
                    sequence += 1

    def test_header_parser_validates_dimensions_and_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary)
            capture = capture_dir / "op-000000000007.bin"
            write_capture(capture, 7, "fc.w1a1_packed", 2)
            parsed = selector.parse_capture(capture)
            self.assertEqual((parsed["sequence"], parsed["K"], parsed["M"], parsed["N"]), (7, 2, 3, 2))
            self.assertEqual(parsed["name"], "fc.w1a1_packed")
            capture.write_bytes(capture.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "payload size"):
                selector.parse_capture(capture)

    def test_selection_is_deterministic_stratified_and_covers_required_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary)
            self.make_capture_set(capture_dir)
            captures = selector.scan_captures(capture_dir)
            first = selector.build_manifest(captures, capture_dir)
            second = selector.build_manifest(selector.scan_captures(capture_dir), capture_dir)
            self.assertEqual(first, second)
            self.assertEqual(first["capture_count"], 14 * len(selector.REQUIRED_TENSORS))
            self.assertEqual(set(first["observed_tensor_names"]), selector.REQUIRED_TENSORS)
            by_shape = {}
            for row in first["selected"]:
                by_shape.setdefault((row["name"], row["N"]), []).append(row)
            for name in selector.REQUIRED_TENSORS:
                self.assertEqual(len(by_shape[(name, 1)]), 3)
                self.assertEqual(len(by_shape[(name, 2)]), 3)
                self.assertEqual(len(by_shape[(name, 37)]), 3)
                self.assertEqual(len(by_shape[(name, 38)]), 2)
                self.assertEqual(
                    [row["sequence"] for row in by_shape[(name, 1)]],
                    sorted(row["sequence"] for row in by_shape[(name, 1)]),
                )
            histogram = first["invocation_histogram"]
            self.assertEqual(sum(row["count"] for row in histogram), first["capture_count"])

    def test_materialization_symlinks_selected_files_and_preserves_raw_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir = root / "raw"
            capture_dir.mkdir()
            self.make_capture_set(capture_dir)
            originals = {path.name: path.read_bytes() for path in capture_dir.iterdir()}
            captures = selector.scan_captures(capture_dir)
            manifest = selector.build_manifest(captures, capture_dir)
            output_dir = root / "runs" / "selection"
            manifest_path = selector.materialize(captures, manifest, output_dir)
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(list(output_dir.glob("op-*.bin"))), manifest["selected_count"])
            self.assertTrue(all(path.is_symlink() for path in output_dir.glob("op-*.bin")))
            self.assertEqual(originals, {path.name: path.read_bytes() for path in capture_dir.iterdir()})
            with self.assertRaisesRegex(ValueError, "new or empty"):
                selector.materialize(captures, manifest, output_dir)

    def test_missing_tensor_and_output_inside_raw_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir = root / "raw"
            capture_dir.mkdir()
            write_capture(capture_dir / "op-000000000000.bin", 0, "fc.w1a1_packed", 1)
            with self.assertRaisesRegex(ValueError, "missing tensor names"):
                selector.scan_captures(capture_dir)

            complete = root / "complete"
            complete.mkdir()
            self.make_capture_set(complete)
            captures = selector.scan_captures(complete)
            manifest = selector.build_manifest(captures, complete)
            with self.assertRaisesRegex(ValueError, "inside the raw capture directory"):
                selector.materialize(captures, manifest, complete / "selected")


if __name__ == "__main__":
    unittest.main()
