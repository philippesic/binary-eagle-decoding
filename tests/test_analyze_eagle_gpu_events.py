"""Synthetic trace accounting tests; these values are not GPU measurements."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_eagle_gpu_events as analysis


def event(identifier, kind, start, stop, parent=0, **fields):
    return {
        "schema": "cuda_eagle_event_v1",
        "id": identifier,
        "frame": 1,
        "parent": parent,
        "device": 0,
        "context": "0x1",
        "kind": kind,
        "stage": "draft",
        "op": "",
        "tensor": "",
        "gpu_begin_ms": start,
        "gpu_end_ms": stop,
        "cuda_ms": None if start is None else stop - start,
        "captured_inventory_only": start is None,
        "host_begin_us": 100 + identifier,
        "host_end_us": 110 + identifier,
        "host_clock": "CLOCK_MONOTONIC",
        **fields,
    }


class EventAccountingTests(unittest.TestCase):
    def test_nested_pack_dot_are_not_added_to_projection_parent(self):
        rows = [
            event(1, "graph", 0, 10),
            event(2, "node", 1, 6, 1, op="W1A1_MUL_MAT"),
            event(3, "activation_pack_scales", 1, 2, 2),
            event(4, "dot_output", 2, 5, 2),
        ]
        frame = analysis.frame_accounting(rows)
        self.assertEqual(frame["gpu_span_ms"], 10)
        self.assertEqual(frame["components_ms"]["activation_pack_scales"], 1)
        self.assertEqual(frame["components_ms"]["dot_output"], 3)
        self.assertEqual(frame["components_ms"]["projection_combined_unsplit"], 1)
        self.assertEqual(frame["unassigned_ms"], 5)
        self.assertEqual(sum(frame["components_ms"].values()), 10)

    def test_overlapping_siblings_have_one_overlap_bucket(self):
        rows = [
            event(1, "graph", 0, 10),
            event(2, "dot_output", 1, 6, 1),
            event(3, "node", 4, 8, 1, op="RMS_NORM"),
        ]
        frame = analysis.frame_accounting(rows)
        self.assertEqual(frame["components_ms"]["overlapping_spans"], 2)
        self.assertEqual(sum(frame["components_ms"].values()), 10)
        self.assertEqual(frame["unassigned_ms"], 3)

    def test_capture_inventory_stays_unassigned(self):
        rows = [
            event(1, "graph", 0, 10, graph_enabled=True, graph_capture=True),
            event(2, "node", None, None, 1, op="RMS_NORM"),
        ]
        frame = analysis.frame_accounting(rows)
        self.assertEqual(frame["unassigned_ms"], 10)
        self.assertEqual(frame["inventory_only_records"], 1)
        self.assertNotIn("norms", frame["components_ms"])

    def test_cycle_and_missing_root_are_explicit(self):
        row = event(2, "dot_output", 1, 2, 2)
        self.assertIsNone(analysis.frame_accounting([row])["gpu_span_ms"])
        missing = analysis.frame_accounting(
            [event(1, "graph", 0, 10), event(2, "dot_output", 1, 2, 99)]
        )
        self.assertEqual(missing["missing_parent_ids"], [99])
        with self.assertRaisesRegex(ValueError, "cycle"):
            analysis.frame_accounting([event(1, "graph", 0, 10), row])

    def test_request_host_union_preserves_unassigned_without_gpu_subtraction(self):
        rows = [
            event(
                1, "graph", 0, 10, source="/run/r0/server.log", host_begin_us=100, host_end_us=200
            ),
            event(
                2,
                "dot_output",
                1,
                4,
                1,
                source="/run/r0/server.log",
                host_begin_us=120,
                host_end_us=180,
            ),
        ]
        request = {
            "prompt_id": "synthetic",
            "directory": "/run/r0/prompt",
            "client_request_begin_monotonic_s": 0.00005,
            "client_request_end_monotonic_s": 0.00025,
            "client_clock_implementation": "clock_gettime(CLOCK_MONOTONIC)",
        }
        row = analysis.request_accounting(rows, {"records": [request]})[0]
        self.assertAlmostEqual(row["instrumented_host_union_us"], 100)
        self.assertAlmostEqual(row["host_unassigned_us"], 100)
        request.pop("client_clock_implementation")
        self.assertIsNone(
            analysis.request_accounting(rows, {"records": [request]})[0]["host_unassigned_us"]
        )

    def test_graph_inventory_counts_references_and_null_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            node = {
                "schema": "cuda_eagle_graph_node_v1",
                "frame": 1,
                "device": 0,
                "context": "0x1",
                "cuda_ms": None,
            }
            summary = {
                "schema": "cuda_eagle_graph_inventory_v1",
                "frame": 1,
                "device": 0,
                "context": "0x1",
                "emitted_nodes": 1,
                "total_nodes": 2,
            }
            path.write_text(
                "CUDA_EAGLE_GRAPH_NODE "
                + json.dumps(node)
                + "\n"
                + "CUDA_EAGLE_GRAPH_INVENTORY "
                + json.dumps(summary)
            )
            report = analysis.analyze([path])
            self.assertTrue(report["trace_truncated"])
            self.assertEqual(report["orphan_inventory_records"], 2)
            node["cuda_ms"] = 1
            path.write_text("CUDA_EAGLE_GRAPH_NODE " + json.dumps(node))
            with self.assertRaisesRegex(ValueError, "null timing"):
                analysis.parse(path)

    def test_parser_marks_caps_and_rejects_partial_gpu_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            row = event(1, "graph", 0, 10)
            text = "CUDA_EAGLE_EVENT " + json.dumps(row) + "\n"
            text += (
                'CUDA_EAGLE_EVENT {"schema":"cuda_eagle_event_v1","kind":"truncation","limit":1}\n'
            )
            path.write_text(text)
            result = analysis.analyze([path])
            self.assertTrue(result["trace_truncated"])
            self.assertEqual(result["event_records"], 1)
            row["cuda_ms"] = None
            path.write_text("CUDA_EAGLE_EVENT " + json.dumps(row))
            with self.assertRaisesRegex(ValueError, "partial"):
                analysis.parse(path)


if __name__ == "__main__":
    unittest.main()
