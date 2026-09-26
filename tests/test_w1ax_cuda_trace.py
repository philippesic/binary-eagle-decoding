"""Synthetic SQLite tests; these do not validate CUDA hardware performance."""

import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_w1ax_cuda_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_w1ax_cuda_trace", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class CudaTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "trace with spaces.sqlite"

    def create(self, name_columns=("shortName", "demangledName")):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
                       "start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
                       "gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
                       "blockX INTEGER, blockY INTEGER, blockZ INTEGER"
                       + "".join(f", {column} INTEGER" for column in name_columns) + ")")

    def add(self, start, end, symbol, *, stream=7, device=0, grid=(2, 1, 1),
            short=None, name_columns=("shortName", "demangledName")):
        with sqlite3.connect(self.path) as db:
            name_ids = []
            for column in name_columns:
                value = short if column == "shortName" and short else symbol
                name_ids.append(db.execute("INSERT INTO StringIds(value) VALUES (?)", (value,)).lastrowid)
            values = [start, end, device, stream, *grid, 32, 4, 1, *name_ids]
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                       + ",".join("?" for _ in values) + ")", values)

    def test_statistics_overlap_and_shapes(self):
        self.create()
        self.add(0, 10, "w1ax_quantize")
        self.add(10, 30, "void w1ax_integer_dot(float*)", short="w1ax_integer_dot", grid=(8, 2, 1))
        self.add(15, 25, "unknown_kernel", stream=8)
        self.add(40, 80, "void w1ax_integer_dot(float*)", short="w1ax_integer_dot", grid=(8, 4, 1))
        before = self.path.read_bytes()
        report = ANALYSIS.analyze(self.path, 4)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(report["overall"], {
            "count": 4, "sum_ns": 80, "union_ns": 70, "overlap_excess_ns": 10,
            "span_ns": 80, "no_kernel_ns_within_span": 10,
        })
        dot = next(row for row in report["kernels"] if row["category"] == "w1_dot_rescale_fused")
        self.assertEqual((dot["count"], dot["sum_ns"], dot["median_ns"], dot["p95_ns"]), (2, 60, 30, 39))
        self.assertEqual([r["grid"] for r in dot["launch_configurations"]], [[8, 2, 1], [8, 4, 1]])
        self.assertEqual(report["act_bits_annotation"], 4)
        self.assertTrue(any(row["symbol"] == "unknown_kernel" and row["category"] == "unclassified"
                            for row in report["kernels"]))

    def test_categories_use_function_boundaries(self):
        expected = {
            "void w1a1_pack_activations(float*)": "w1_packing_quantization",
            "w1a1_xor_popc": "w1_dot_rescale_fused",
            "w1a16_signadd": "w1a16_signadd_rescale_fused",
            "w1ax_validate_integer_dots": "w1ax_validation",
            "void quantize_q8_1<4>(float*)": "anchor_packing_quantization",
            "quantize_mmq_q8_1<128>": "anchor_packing_quantization",
            "mul_mat_q_stream_k_fixup<2>": "anchor_quantized_mulmat",
            "mul_mat_vec_q<2>": "anchor_quantized_mulmat",
            "mul_mat_vec_f<half, 1>": "anchor_float_mulmat",
            "w1ax_integer_dot_extra": "unclassified",
            "other_w1ax_quantize": "unclassified",
            "cublas_unknown": "unclassified",
        }
        for symbol, category in expected.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(ANALYSIS.classify(symbol), category)

    def test_multiple_devices_and_triple_overlap(self):
        self.create()
        self.add(0, 20, "one")
        self.add(0, 20, "two", stream=8)
        self.add(0, 20, "three", device=1)
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["overall"]["union_ns"], 20)
        self.assertEqual(report["overall"]["overlap_excess_ns"], 40)
        self.assertEqual([d["overlap_excess_ns"] for d in report["devices"]], [20, 0])

    def test_empty_valid_export(self):
        self.create()
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["overall"]["count"], 0)
        self.assertEqual(report["overall"]["union_ns"], 0)
        self.assertEqual(report["kernels"], [])

    def test_short_name_only_and_unresolved_id(self):
        self.create(("shortName",))
        self.add(0, 4, "w1a16_signadd", name_columns=("shortName",))
        self.add(5, 9, "lost", name_columns=("shortName",))
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM StringIds WHERE value='lost'")
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["overall"]["count"], 2)
        unresolved = next(k for k in report["kernels"] if k["symbol"].startswith("unresolved:"))
        self.assertEqual(unresolved["category"], "unclassified")

    def test_demangled_name_only(self):
        self.create(("demangledName",))
        self.add(0, 4, "void w1ax_quantize(float*)", name_columns=("demangledName",))
        self.assertEqual(ANALYSIS.analyze(self.path)["kernels"][0]["category"], "w1_packing_quantization")

    def test_missing_file_is_not_created(self):
        with self.assertRaises(sqlite3.OperationalError):
            ANALYSIS.analyze(self.path)
        self.assertFalse(self.path.exists())

    def test_missing_tables_and_columns(self):
        with sqlite3.connect(self.path):
            pass
        with self.assertRaisesRegex(ValueError, "missing required SQLite table"):
            ANALYSIS.analyze(self.path)
        self.create()
        with sqlite3.connect(self.path) as db:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME COLUMN streamId TO other")
        with self.assertRaisesRegex(ValueError, "missing kernel column.*streamId"):
            ANALYSIS.analyze(self.path)

    def test_invalid_duration_fails(self):
        self.create()
        self.add(10, 9, "bad")
        with self.assertRaisesRegex(ValueError, "end precedes start"):
            ANALYSIS.analyze(self.path)

    def test_cli_output_and_input_protection(self):
        self.create()
        self.add(0, 10, "w1a16_signadd")
        output = Path(self.tmp.name) / "report.json"
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.path), "--act-bits", "16",
                              "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(output.read_text())["act_bits_annotation"], 16)
        before = self.path.read_bytes()
        bad = subprocess.run([sys.executable, str(SCRIPT), str(self.path), "--output", str(self.path)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("must not overwrite", bad.stderr)
        self.assertEqual(self.path.read_bytes(), before)
        alias = self.path.with_name("hardlink.sqlite")
        alias.hardlink_to(self.path)
        bad = subprocess.run([sys.executable, str(SCRIPT), str(self.path), "--output", str(alias)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
