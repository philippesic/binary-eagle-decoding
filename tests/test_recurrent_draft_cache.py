"""The stored-cache gate rejects altered bytes, slots, and causal masks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_recurrent_draft_cache import audit  # noqa: E402


class StoredDraftCacheAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        graph = []
        values = []
        graph_specs = (
            ("inp_embd", 1, False),
            ("embd_norm-0", 1, False),
            ("Kcur_rope-0", 1024, False),
            ("Vcur-0", 1024, False),
            ("eagle3_prenorm-0", 1, False),
            ("result_norm", 1, True),
        )
        for name, width, group_end in graph_specs:
            graph.append(
                {
                    "schema": "eagle_draft_graph_v1",
                    "event": "tensor",
                    "group_kind": "decoder",
                    "group_execution": 0,
                    "group_end": group_end,
                    "tensor_name": name,
                    "ne": [width, 1],
                    "token_axis": 1,
                    "token_width": width,
                    "n_tokens": 1,
                    "f32_count": width,
                    "f32_offset": len(values),
                    "f32_bytes": width * 4,
                }
            )
            values.extend([0.0] * width)
        graph.append(
            {
                "schema": "eagle_draft_graph_v1",
                "event": "capture_end",
                "scope": "cache",
                "result_output_markers": 1,
                "status": "complete",
                "reason": "",
                "tensor_rows": len(graph_specs),
                "execution_count": len(graph_specs),
                "decoder_groups": 1,
                "bytes_written": len(values) * 4,
            }
        )
        self.write_jsonl("heads.draft_graph.jsonl", graph)
        np.asarray(values, dtype="<f4").tofile(self.root / "heads.draft_graph.f32")
        self.events = [
            {
                "schema": "eagle_draft_cache_v1",
                "event": "execution",
                "execution": 0,
                "n_tokens": 1,
                "n_kv": 1,
                "cache_buffer_type": "CPU",
                "cache_buffer_is_host": True,
                "mask_buffer_type": "CPU",
                "mask_buffer_is_host": True,
                "mask_dtype": "f16",
                "mask_offset": 0,
                "mask_bytes": 2,
            },
            {
                "schema": "eagle_draft_cache_v1",
                "event": "row",
                "execution": 0,
                "column": 0,
                "position": 0,
                "token_id": 7,
                "slot": 0,
                "row_offset": 0,
                "key_bytes": 2048,
                "value_bytes": 2048,
            },
            {
                "schema": "eagle_draft_cache_v1",
                "event": "capture_end",
                "executions": 1,
                "rows": 1,
                "row_bytes": 4096,
                "mask_bytes": 2,
                "max_rows": 8192,
                "max_bytes": 512 * 1024 * 1024,
            },
        ]
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        (self.root / "heads.draft_cache.f16").write_bytes(bytes(4096))
        np.asarray([0], dtype="<f2").tofile(self.root / "heads.draft_cache.mask")

    def write_jsonl(self, name, events):
        (self.root / name).write_text("".join(json.dumps(event) + "\n" for event in events))

    def test_matching_stored_rows_and_mask(self):
        result = audit(self.root)
        self.assertEqual(result["key_equal_elements"], 1024)
        self.assertEqual(result["value_equal_elements"], 1024)
        self.assertEqual(result["exact_prefix_mask_rows"], 1)
        self.assertEqual(result["execution_device"], "cpu")

    def test_cuda_buffer_metadata_is_reported(self):
        self.events[0]["cache_buffer_type"] = "CUDA0"
        self.events[0]["cache_buffer_is_host"] = False
        self.events[0]["mask_buffer_type"] = "CUDA0"
        self.events[0]["mask_buffer_is_host"] = False
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        result = audit(self.root)
        self.assertEqual(result["execution_device"], "cuda")
        self.assertEqual(result["mask_device"], "cuda")
        self.assertEqual(result["cache_buffer_types"], ["CUDA0"])

    def test_unbounded_or_unsupported_capture_metadata_is_rejected(self):
        self.events[-1]["max_rows"] = 65537
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        with self.assertRaisesRegex(ValueError, "capture limits"):
            audit(self.root)
        self.events[-1]["max_rows"] = 8192
        self.events[0]["cache_buffer_type"] = "Vulkan0"
        self.events[0]["cache_buffer_is_host"] = False
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        with self.assertRaisesRegex(ValueError, "unsupported non-host backend"):
            audit(self.root)

    def test_cuda_host_buffer_is_not_reported_as_cuda_execution(self):
        self.events[0]["mask_buffer_type"] = "CUDA_Host"
        self.events[0]["mask_buffer_is_host"] = True
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        result = audit(self.root)
        self.assertEqual(result["mask_device"], "cuda_host")

    def test_cache_scope_requires_projection_and_boundary_records(self):
        graph_path = self.root / "heads.draft_graph.jsonl"
        graph = [json.loads(line) for line in graph_path.read_text().splitlines()]
        next(row for row in graph if row.get("tensor_name") == "Kcur_rope-0")["tensor_name"] = "other"
        graph_path.write_text("".join(json.dumps(row) + "\n" for row in graph))
        with self.assertRaisesRegex(ValueError, "cache projection inputs or group boundaries"):
            audit(self.root)

    def test_invalid_graph_scope_is_rejected(self):
        graph_path = self.root / "heads.draft_graph.jsonl"
        graph = [json.loads(line) for line in graph_path.read_text().splitlines()]
        graph[-1]["scope"] = "projections"
        graph_path.write_text("".join(json.dumps(row) + "\n" for row in graph))
        with self.assertRaisesRegex(ValueError, "scope metadata"):
            audit(self.root)

    def test_changed_stored_key_is_rejected(self):
        path = self.root / "heads.draft_cache.f16"
        data = bytearray(path.read_bytes())
        data[0] = 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "stored draft cache bytes differ"):
            audit(self.root)

    def test_wrong_slot_or_mask_is_rejected(self):
        self.events[1]["slot"] = 1
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        with self.assertRaisesRegex(ValueError, "slot or offset"):
            audit(self.root)
        self.events[1]["slot"] = 0
        self.write_jsonl("heads.draft_cache.jsonl", self.events)
        np.asarray([-np.inf], dtype="<f2").tofile(self.root / "heads.draft_cache.mask")
        with self.assertRaisesRegex(ValueError, "mask is not the captured exact prefix"):
            audit(self.root)


if __name__ == "__main__":
    unittest.main()
