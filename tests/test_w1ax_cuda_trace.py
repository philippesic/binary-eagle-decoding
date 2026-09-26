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

    def create(self, name_columns=("shortName", "demangledName"), identity_columns=()):
        self.identity_columns = identity_columns
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
                       "start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
                       "gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
                       "blockX INTEGER, blockY INTEGER, blockZ INTEGER"
                       + "".join(f", {column} INTEGER" for column in (*name_columns, *identity_columns)) + ")")

    def add(self, start, end, symbol, *, stream=7, device=0, grid=(2, 1, 1),
            short=None, name_columns=("shortName", "demangledName"), identity=None):
        with sqlite3.connect(self.path) as db:
            name_ids = []
            for column in name_columns:
                value = short if column == "shortName" and short else symbol
                name_ids.append(db.execute("INSERT INTO StringIds(value) VALUES (?)", (value,)).lastrowid)
            values = [start, end, device, stream, *grid, 32, 4, 1, *name_ids,
                      *((identity or {}).get(column) for column in self.identity_columns)]
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

    def test_pairs_measure_sum_before_aggregation_and_keep_gaps(self):
        self.create()
        for start, pack_duration, dot_duration, gap in ((0, 10, 100, 5), (200, 100, 10, 10), (400, 20, 20, 0)):
            self.add(start, start + pack_duration, "void w1ax_quantize(float*)", grid=(2, 1, 1))
            dot_start = start + pack_duration + gap
            self.add(dot_start, dot_start + dot_duration, "void w1ax_integer_dot(float*)", grid=(8000, 2, 1))
        # Another stream's activity does not break same-stream adjacency.
        self.add(11, 14, "unrelated", stream=8)
        pairs = ANALYSIS.analyze(self.path, 4)["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 3)
        self.assertEqual(pairs["unpaired_kernel_count"], 0)
        timing = pairs["timing"]
        self.assertEqual(timing["pack"]["median_ns"], 20)
        self.assertEqual(timing["dot"]["median_ns"], 20)
        self.assertEqual(timing["kernel_sum"]["median_ns"], 110)
        self.assertEqual(timing["elapsed"]["median_ns"], 115)
        self.assertEqual(timing["gap"]["median_ns"], 5)
        self.assertEqual(timing["elapsed"]["sum_ns"],
                         timing["kernel_sum"]["sum_ns"] + timing["gap"]["sum_ns"])
        group = pairs["launch_configurations"][0]
        self.assertEqual(group["pair_count"], 3)
        self.assertEqual(group["pack_grid"], [2, 1, 1])
        self.assertEqual(group["dot_grid"], [8000, 2, 1])
        self.assertEqual(group["timing"], timing)

    def test_pairs_reject_grid_family_overlap_and_intervening_kernel(self):
        self.create()
        # Token grid mismatch.
        self.add(0, 10, "w1ax_quantize", grid=(2, 1, 1))
        self.add(10, 20, "w1ax_integer_dot", grid=(8, 3, 1))
        # Wrong symbol family even though the grid matches.
        self.add(30, 40, "w1ax_quantize", grid=(2, 1, 1))
        self.add(40, 50, "w1a1_xor_popc", grid=(8, 2, 1))
        # Overlap is not added as a sequential operator pair.
        self.add(60, 80, "w1ax_quantize", grid=(2, 1, 1))
        self.add(70, 90, "w1ax_integer_dot", grid=(8, 2, 1))
        # Even an unclassified intervening kernel breaks adjacency.
        self.add(100, 110, "w1ax_quantize", grid=(2, 1, 1))
        self.add(110, 115, "unknown")
        self.add(115, 125, "w1ax_integer_dot", grid=(8, 2, 1))
        pairs = ANALYSIS.analyze(self.path)["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 0)
        self.assertEqual(pairs["unpaired_pack_count"], 4)
        self.assertEqual(pairs["unpaired_dot_count"], 4)
        self.assertEqual(pairs["unpaired_kernel_count"], 8)
        self.assertIsNone(pairs["timing"]["kernel_sum"]["median_ns"])

    def test_pairs_require_same_device_stream_and_do_not_reuse_launches(self):
        self.create()
        self.add(0, 10, "w1a1_pack_activations", device=0, stream=7)
        self.add(10, 20, "w1a1_xor_popc", grid=(8, 2, 1), device=1, stream=7)
        self.add(30, 40, "w1a1_pack_activations", device=0, stream=8)
        self.add(40, 50, "w1a1_xor_popc", grid=(8, 2, 1), device=0, stream=9)
        # A single pack can match only the immediate first dot.
        self.add(60, 70, "w1a1_pack_activations", device=2, stream=7)
        self.add(70, 90, "w1a1_xor_popc", grid=(8, 2, 1), device=2, stream=7)
        self.add(90, 110, "w1a1_xor_popc", grid=(8, 2, 1), device=2, stream=7)
        self.add(120, 150, "w1a16_signadd", device=2, stream=7)
        pairs = ANALYSIS.analyze(self.path)["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 1)
        self.assertEqual(pairs["unpaired_pack_count"], 2)
        self.assertEqual(pairs["unpaired_dot_count"], 3)
        self.assertEqual(pairs["launch_configurations"][0]["pack_symbol"], "w1a1_pack_activations")
        self.assertEqual(pairs["timing"]["kernel_sum"]["median_ns"], 30)

    def test_pair_groups_separate_launch_shapes(self):
        self.create()
        self.add(0, 10, "w1ax_quantize", grid=(2, 1, 1))
        self.add(10, 30, "w1ax_integer_dot", grid=(8, 2, 1))
        self.add(40, 50, "w1ax_quantize", grid=(3, 1, 1))
        self.add(50, 80, "w1ax_integer_dot", grid=(16, 3, 1))
        pairs = ANALYSIS.analyze(self.path)["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 2)
        self.assertEqual([g["dot_grid"] for g in pairs["launch_configurations"]], [[8, 2, 1], [16, 3, 1]])
        self.assertEqual([g["timing"]["kernel_sum"]["median_ns"] for g in pairs["launch_configurations"]], [30, 40])

    def test_pairs_do_not_cross_process_or_context(self):
        self.create(identity_columns=("globalPid", "contextId", "greenContextId"))
        for offset, first, second in (
            (0, {"globalPid": 100, "contextId": 1}, {"globalPid": 101, "contextId": 1}),
            (40, {"globalPid": 200, "contextId": 1}, {"globalPid": 200, "contextId": 2}),
        ):
            self.add(offset, offset + 10, "w1ax_quantize", identity=first)
            self.add(offset + 10, offset + 30, "w1ax_integer_dot", grid=(8, 2, 1), identity=second)
        report = ANALYSIS.analyze(self.path)
        pairs = report["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 0)
        self.assertEqual(pairs["unpaired_kernel_count"], 4)
        self.assertFalse(report["identity_attribution"]["limited"])

    def test_matching_context_pairs_preserve_identity_in_launch_summaries(self):
        self.create(identity_columns=("globalPid", "contextId", "greenContextId"))
        identities = [
            {"globalPid": 100, "contextId": 1, "greenContextId": None},
            {"globalPid": 100, "contextId": 2, "greenContextId": None},
            {"globalPid": 101, "contextId": 1, "greenContextId": 0},
        ]
        for index, identity in enumerate(identities):
            start = 40 * index
            self.add(start, start + 10, "w1ax_quantize", identity=identity)
            self.add(start + 10, start + 30, "w1ax_integer_dot", grid=(8, 2, 1), identity=identity)
        report = ANALYSIS.analyze(self.path)
        pairs = report["packing_inclusive_pairs"]
        self.assertEqual(pairs["pair_count"], 3)
        groups = pairs["launch_configurations"]
        self.assertEqual([(g["global_pid"], g["context_id"], g["green_context_id"]) for g in groups],
                         [(100, 1, None), (100, 2, None), (101, 1, 0)])
        self.assertEqual([g["pair_count"] for g in groups], [1, 1, 1])
        for kernel in report["kernels"]:
            self.assertEqual(len(kernel["launch_configurations"]), 3)
            self.assertEqual([g["context_id"] for g in kernel["launch_configurations"]], [1, 2, 1])

    def test_present_null_process_context_ids_are_not_paired(self):
        self.create(identity_columns=("globalPid", "contextId"))
        identity = {"globalPid": 100, "contextId": None}
        self.add(0, 10, "w1ax_quantize", identity=identity)
        self.add(10, 30, "w1ax_integer_dot", grid=(8, 2, 1), identity=identity)
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["packing_inclusive_pairs"]["pair_count"], 0)
        self.assertEqual(report["identity_attribution"]["null_process_context_row_count"], 2)
        self.assertTrue(report["identity_attribution"]["limited"])

    def test_legacy_identity_limit_and_partial_context_partition(self):
        self.create(identity_columns=("contextId",))
        self.add(0, 10, "w1ax_quantize", identity={"contextId": 1})
        self.add(10, 30, "w1ax_integer_dot", grid=(8, 2, 1), identity={"contextId": 2})
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["packing_inclusive_pairs"]["pair_count"], 0)
        self.assertEqual(report["identity_attribution"]["missing_process_context_columns"], ["globalPid"])
        self.assertTrue(report["identity_attribution"]["limited"])

    def test_empty_valid_export(self):
        self.create()
        report = ANALYSIS.analyze(self.path)
        self.assertEqual(report["overall"]["count"], 0)
        self.assertEqual(report["overall"]["union_ns"], 0)
        self.assertEqual(report["kernels"], [])
        self.assertTrue(report["identity_attribution"]["limited"])
        self.assertEqual(report["identity_attribution"]["missing_process_context_columns"], ["globalPid", "contextId"])

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
