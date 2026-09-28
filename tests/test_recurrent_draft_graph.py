"""CPU-only graph-tap payload and head-state join checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_recurrent_draft_graph import DECODER_TAPS, QK_HEADS, compare  # noqa: E402


class DraftGraphJoinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.head = np.zeros(2560, dtype="<f4")
        self.head[:2] = [3, 4]
        self.head.tofile(self.root / "heads.f32")
        (self.root / "heads.jsonl").write_text(
            json.dumps(
                {
                    "schema": "eagle_head_state_v1",
                    "state_row": 0,
                    "round_index": 0,
                    "depth": 0,
                    "state_dim": 2560,
                }
            )
            + "\n"
        )
        self.index = []
        values = []
        adapter = {}
        cursor = 0
        for ordinal, name in enumerate(DECODER_TAPS):
            native = (
                self.head.copy()
                if name == "result_norm"
                else np.ones(QK_HEADS[name] * 128, dtype="<f4")
                if name in QK_HEADS
                else np.array([1, 2], dtype="<f4")
            )
            python = native.copy()
            if name == "result_norm":
                python[0] += 0.25
            adapter[name] = python
            rope = name in {"Qcur_rope-0", "Kcur_rope-0"}
            head_count = QK_HEADS.get(name)
            self.index.append(
                {
                    "schema": "eagle_draft_graph_v1",
                    "event": "tensor",
                    "tensor_name": name,
                    "dtype": "f32",
                    "ne": [128, head_count, 1] if rope else [len(native), 1],
                    "nb": [4, 128 * 4, len(native) * 4] if rope else [4, len(native) * 4],
                    "execution_ordinal": ordinal,
                    "group_kind": "decoder",
                    "group_order": ordinal,
                    "group_execution": 0,
                    "group_begin": name == "inp_embd",
                    "group_end": name == "result_norm",
                    "n_tokens": 1,
                    "token_axis": 2 if rope else 1,
                    "token_width": len(native),
                    "f32_offset": cursor,
                    "f32_count": len(native),
                    "f32_bytes": len(native) * 4,
                }
            )
            cursor += len(native)
            values.append(native)
        self.values = np.concatenate(values)
        np.savez(self.root / "adapter.npz", **adapter)
        self.save()

    def save(self):
        footer = {
            "schema": "eagle_draft_graph_v1",
            "event": "capture_end",
            "status": "complete",
            "decoder_groups": len({row["group_execution"] for row in self.index}),
            "encoder_groups": 0,
            "execution_count": len(self.index),
            "tensor_rows": len(self.index),
            "bytes_written": int(self.values.nbytes),
            "reason": "",
        }
        (self.root / "heads.draft_graph.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in (*self.index, footer))
        )
        self.values.tofile(self.root / "heads.draft_graph.f32")

    def test_joins_exact_head_row_then_compares_same_execution(self):
        result = compare(self.root, self.root / "adapter.npz")
        self.assertEqual(result["native_group_execution"], 0)
        self.assertEqual(result["native_token_column"], 0)
        self.assertEqual(result["differences"]["inp_embd"]["max_abs"], 0)
        self.assertEqual(result["differences"]["result_norm"]["max_abs"], 0.25)

    def test_rejects_gap_in_raw_payload_or_wrong_head_state(self):
        self.index[1]["f32_offset"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "offsets"):
            compare(self.root, self.root / "adapter.npz")
        self.index[1]["f32_offset"] -= 1
        self.save()
        changed = self.head.copy()
        changed[0] = 9
        changed.tofile(self.root / "heads.f32")
        with self.assertRaisesRegex(ValueError, "prenorm cannot be joined uniquely"):
            compare(
                self.root,
                self.root / "adapter.npz",
                native_output_norm=np.ones(2560, np.float32),
            )

    def test_skips_context_group_without_result_norm(self):
        complete = [dict(row) for row in self.index]
        complete_values = self.values.copy()
        context = [dict(row) for row in complete[:-1]]
        context[-1]["group_end"] = True
        context_count = sum(row["f32_count"] for row in context)
        for ordinal, row in enumerate(complete, start=len(context)):
            row["group_execution"] = 1
            row["group_order"] = 1
            row["execution_ordinal"] = ordinal
            row["f32_offset"] += context_count
        self.index = context + complete
        self.values = np.concatenate((complete_values[:context_count], complete_values))
        self.save()
        result = compare(self.root, self.root / "adapter.npz")
        self.assertEqual(result["native_group_execution"], 1)
        self.assertEqual(result["native_decoder_groups"], 2)

    def test_rejects_incomplete_capture_footer(self):
        path = self.root / "heads.draft_graph.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["status"] = "incomplete"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "complete footer"):
            compare(self.root, self.root / "adapter.npz")

    def test_unique_prenorm_reconstruction_joins_without_result_norm(self):
        prenorm = np.zeros(2560, dtype="<f4")
        prenorm[:2] = [3, 4]
        normalized = prenorm.astype(np.float64) / np.sqrt(
            np.mean(prenorm.astype(np.float64) ** 2) + 1e-6
        )
        normalized.astype("<f4").tofile(self.root / "heads.f32")
        self.values = np.concatenate((self.values[: -2560 - 2], prenorm))
        self.index = self.index[:-1]
        last = self.index[-1]
        last.update(
            ne=[2560, 1],
            nb=[4, 2560 * 4],
            token_width=2560,
            f32_count=2560,
            f32_bytes=2560 * 4,
            group_end=True,
        )
        with np.load(self.root / "adapter.npz") as file:
            adapter = {name: file[name] for name in file.files}
        adapter["eagle3_prenorm-0"] = prenorm
        np.savez(self.root / "adapter.npz", **adapter)
        self.save()
        result = compare(
            self.root, self.root / "adapter.npz", native_output_norm=np.ones(2560, np.float32)
        )
        self.assertEqual(result["join_method"], "f32_output_norm_reconstruction")
        self.assertLess(result["join_error"]["rms"], 1e-4)
        self.assertNotIn("result_norm", result["differences"])

    def test_mapped_logits_may_contain_negative_infinity(self):
        self.index[-1]["group_end"] = False
        logits = np.array([1, -np.inf, 2, -np.inf], dtype="<f4")
        output = dict(self.index[-1])
        output.update(
            tensor_name="result_output",
            ne=[4, 1],
            nb=[4, 16],
            execution_ordinal=len(self.index),
            group_begin=False,
            group_end=True,
            token_width=4,
            f32_offset=len(self.values),
            f32_count=4,
            f32_bytes=16,
        )
        self.index.append(output)
        self.values = np.concatenate((self.values, logits))
        self.save()
        result = compare(self.root, self.root / "adapter.npz")
        self.assertEqual(result["join_method"], "exact_result_norm")
        self.values[-1] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "NaN"):
            compare(self.root, self.root / "adapter.npz")


if __name__ == "__main__":
    unittest.main()
