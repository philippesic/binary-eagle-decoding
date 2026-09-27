"""Synthetic trace gates only; no GPU performance claims."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_binary_rescue_kernels as analysis  # noqa: E402


class BinaryRescueKernelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profile trace.sqlite"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
                       "start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,"
                       "gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,"
                       "demangledName INTEGER,globalPid INTEGER,contextId INTEGER,graphNodeId INTEGER)")

    def add_kernel(self, name, start, end, grid=(1, 1, 1), block=(128, 1, 1), pid=1000, context=1, node=None):
        with sqlite3.connect(self.path) as db:
            name_id = db.execute("INSERT INTO StringIds(value) VALUES(?)", (name,)).lastrowid
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (start, end, 0, 7, *grid, *block, name_id, pid, context, node))

    def test_group128_classification_and_frozen_head_inference(self):
        self.add_kernel("void w1a16_group128_signadd(float*)", 0, 10, (250, 1, 1), node=3)
        self.add_kernel("w1a16_group128_signadd", 10, 13, (20, 1, 1))
        self.add_kernel("w1a16_group128_signadd_validator", 13, 14, (250, 1, 1))
        self.add_kernel("w1a16_signadd", 14, 20, (250, 3, 1))
        report = analysis.analyze(self.path, "D")
        categories = {row["category"]: row for row in report["categories"]}
        self.assertEqual(categories["binary_a16_group128_fused"]["count"], 2)
        self.assertEqual(categories["unclassified"]["count"], 1)
        self.assertEqual(report["binary_head_shape_inference"]["count"], 2)
        self.assertEqual(report["binary_head_shape_inference"]["status"], "inferred")
        self.assertEqual(report["binary_head_shape_inference"]["sum_ns"], 16)
        self.assertEqual(report["binary_a16_fused"]["count"], 3)
        self.assertEqual(report["variant_annotation"], "D")

    def test_standard_packing_costs_and_overlap_are_separate(self):
        self.add_kernel("quantize_q8_1<32>", 0, 10)
        self.add_kernel("mul_mat_vec_q<8>", 5, 25)
        self.add_kernel("quantize_mmq_q8_1", 30, 32)
        self.add_kernel("mul_mat_q<4>", 32, 42)
        self.add_kernel("mul_mat_q_stream_k_fixup", 42, 43)
        self.add_kernel("mul_mat_vec_f<half>", 45, 50)
        report = analysis.analyze(self.path)
        standard = report["packing_inclusive_standard_quantized"]
        self.assertEqual(standard["count"], 5)
        self.assertEqual(standard["sum_ns"], 43)
        self.assertEqual(standard["union_ns"], 38)
        self.assertEqual(standard["overlap_excess_ns"], 5)
        self.assertEqual(report["overall_kernels"]["sum_ns"], 48)
        self.assertEqual(report["binary_head_shape_inference"]["status"], "not_observed")

    def test_graph_envelopes_api_calls_and_node_counters(self):
        self.add_kernel("w1a16_group128_signadd", 10, 20, node=5)
        self.add_kernel("other", 30, 35, node=0)
        self.add_kernel("other", 40, 45)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,graphId INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(0,100,0,7,3)")
            db.execute("CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER)")
            db.executemany("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?)", [(5,), (6,)])
            for table in ("RUNTIME", "DRIVER"):
                db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{table}(start INTEGER,end INTEGER,nameId INTEGER,returnValue INTEGER)")
                name = "cudaStreamBeginCapture_v10000" if table == "RUNTIME" else "cuGraphLaunch"
                name_id = db.execute("INSERT INTO StringIds(value) VALUES(?)", (name,)).lastrowid
                db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_{table} VALUES(?,?,?,?)", [(0, 5, name_id, 0), (10, 15, name_id, None)])
        report = analysis.analyze(self.path)
        self.assertEqual(report["overall_kernels"]["union_ns"], 20)
        self.assertEqual(report["gpu_activities"]["combined"]["union_ns"], 100)
        self.assertEqual(report["gpu_activities"]["combined"]["sum_ns"], 120)
        graph = report["graph_evidence"]
        self.assertEqual(graph["kernel_rows_with_nonzero_graph_node_id"], 1)
        self.assertEqual(graph["kernel_rows_with_zero_graph_node_id"], 1)
        self.assertEqual(graph["kernel_rows_with_missing_graph_node_id"], 1)
        for table in graph["api_tables"]:
            self.assertEqual(table["calls"][0]["count"], 2)
            self.assertEqual(table["calls"][0]["success_count"], 1)
            self.assertEqual(table["calls"][0]["unknown_return_count"], 1)
        self.assertEqual({row["table"]: row["row_count"] for row in graph["graph_tables"]}["CUDA_GRAPH_NODE_EVENTS"], 2)

    def test_no_projection_guess_for_bad_launch_or_process_identity(self):
        self.add_kernel("w1a16_group128_signadd", 0, 10, (250, 1, 1), block=(64, 1, 1), pid=10)
        self.add_kernel("w1a16_group128_signadd", 0, 10, (250, 1, 2), pid=20)
        report = analysis.analyze(self.path)
        self.assertEqual(report["binary_head_shape_inference"]["count"], 0)
        self.assertEqual(len(report["identity_partitions"]), 2)
        self.assertTrue(all(not table["present"] for table in report["graph_evidence"]["api_tables"]))
        self.assertEqual(report["graph_evidence"]["graph_tables"], [])

    def test_graph_only_export_does_not_infer_kernel_costs(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
            db.execute("DROP TABLE StringIds")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(0,100,0,7)")
        report = analysis.analyze(self.path)
        self.assertFalse(report["kernel_availability"]["table_present"])
        self.assertEqual(report["binary_head_shape_inference"]["status"], "unavailable")
        self.assertEqual(report["overall_kernels"]["count"], 0)
        self.assertEqual(report["gpu_activities"]["combined"]["union_ns"], 100)

    def test_exact_global_pid_filters_all_gpu_activities_and_api_scope(self):
        raw_pid = 281474976715656
        self.add_kernel("w1a16_signadd", 10, 20, pid=raw_pid, node=5)
        self.add_kernel("w1a16_signadd", 200, 230, pid=5000, node=6)
        with sqlite3.connect(self.path) as db:
            for kind in ("GRAPH_TRACE", "MEMCPY", "MEMSET"):
                db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{kind}(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,globalPid INTEGER)")
                db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_{kind} VALUES(?,?,?,?,?)", [(0,100,0,7,raw_pid), (200,300,0,7,5000)])
            name_id = db.execute("INSERT INTO StringIds(value) VALUES('cudaGraphLaunch')").lastrowid
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(0,2,?,?)", (name_id,raw_pid))
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,nameId INTEGER,globalPid INTEGER)")
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,?,?)", [(0,2,name_id,raw_pid), (200,202,name_id,5000)])
            db.execute("CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER)")
            db.executemany("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?)", [(5,), (6,)])
        report = analysis.analyze(self.path, "D", global_pid=raw_pid)
        self.assertEqual(report["selection"]["global_pid"], raw_pid)
        self.assertEqual(report["overall_kernels"]["count"], 1)
        self.assertEqual(report["overall_kernels"]["union_ns"], 10)
        self.assertEqual(report["gpu_activities"]["combined"]["union_ns"], 100)
        self.assertTrue(all(row["selected_row_count"] == 1 for row in report["selection"]["activities"]))
        self.assertEqual(report["graph_evidence"]["kernel_rows_with_nonzero_graph_node_id"], 1)
        runtime, driver = report["graph_evidence"]["api_tables"]
        self.assertEqual(runtime["scope"], "omitted_without_globalPid")
        self.assertEqual(runtime["calls"], [])
        self.assertEqual(driver["scope"], "exact_global_pid")
        self.assertEqual(driver["calls"][0]["count"], 1)
        tables = {row["table"]: row for row in report["graph_evidence"]["graph_tables"]}
        self.assertEqual(tables["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"]["row_count"], 1)
        self.assertEqual(tables["CUDA_GRAPH_NODE_EVENTS"]["scope"], "unfiltered_full_export")
        self.assertEqual(tables["CUDA_GRAPH_NODE_EVENTS"]["row_count"], 2)
        output = Path(self.temp.name) / "filtered.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_binary_rescue_kernels.py"),
                        str(self.path), "--global-pid", str(raw_pid), "--output", str(output)], check=True)
        self.assertEqual(json.loads(output.read_text())["selection"]["global_pid"], raw_pid)

    def test_global_pid_filter_rejects_unavailable_columns(self):
        self.add_kernel("w1a16_signadd", 0, 1)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER)")
        with self.assertRaisesRegex(ValueError, "unavailable.*MEMCPY"):
            analysis.analyze(self.path, global_pid=1000)
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY")
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN globalPid")
        with self.assertRaisesRegex(ValueError, "unavailable.*KERNEL"):
            analysis.analyze(self.path, global_pid=1000)
        self.assertEqual(analysis.analyze(self.path)["overall_kernels"]["count"], 1)

    def test_global_pid_no_match_is_empty_not_decoded(self):
        self.add_kernel("w1a16_signadd", 0, 1, pid=281474976715656)
        report = analysis.analyze(self.path, global_pid=5000)
        self.assertEqual(report["overall_kernels"]["count"], 0)
        self.assertEqual(report["selection"]["activities"][0]["export_row_count"], 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            analysis.analyze(self.path, global_pid=True)

    def test_analysis_readonly_and_cli_rejects_input_overwrite(self):
        self.add_kernel("w1a16_signadd", 0, 1)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = analysis.analyze(self.path)
        self.assertEqual(report["input"]["sha256"], before)
        command = [sys.executable, str(ROOT / "scripts" / "analyze_binary_rescue_kernels.py"), str(self.path)]
        result = subprocess.run(command + ["--output", str(self.path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not overwrite", result.stderr)
        output = Path(self.temp.name) / "report.json"
        subprocess.run(command + ["--output", str(output), "--variant", "D"], check=True)
        self.assertTrue(output.exists())
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
