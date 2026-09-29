"""CPU contracts for the frozen W1Ax calibration readiness assembler."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_w1ax_calibration_readiness import (  # noqa: E402
    EXPECTED_DOMAINS,
    _bridge_roots,
    _exact_response_pairs,
    assemble,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CalibrationReadinessTests(unittest.TestCase):
    def test_missing_provider_artifact_emits_no_readiness_or_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "readiness.json"
            overlay = root / "provider-overlay.json"
            args = SimpleNamespace(
                provider_manifest=root / "missing-provider.json",
                report=report,
                provider_overlay=overlay,
            )
            with self.assertRaises(FileNotFoundError):
                assemble(args)
            self.assertFalse(report.exists())
            self.assertFalse(overlay.exists())

    def test_benchmark_response_ids_must_match_each_frozen_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = directory / "manifest.json"
            records = []
            for alias in ("pilot-prose", "pilot-reasoning", "pilot-code"):
                for variant in ("q4_0", "row_a16_checkpoint_zero"):
                    records.append(
                        {
                            "prompt_id": alias,
                            "variant": variant,
                            "warmup": False,
                            "generated_token_ids": [1, 3, 5],
                        }
                    )
            manifest_path.write_text(json.dumps({"records": records}))
            self.assertEqual(
                len(_exact_response_pairs(manifest_path, json.loads(manifest_path.read_text()))), 3
            )
            records[-1]["generated_token_ids"] = [1, 3, 6]
            manifest_path.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "response IDs differ"):
                _exact_response_pairs(manifest_path, json.loads(manifest_path.read_text()))

    def test_selected_roots_join_head_seed_torch_cache_and_graph_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary)
            heads, states, cache_rows = [], [], []
            numeric, gradients, task_map = {}, {}, {}
            graph_rows, graph_values = [], []
            offset = 0
            for execution, (domain, prompt) in enumerate(EXPECTED_DOMAINS.items()):
                task_id = 11 + execution
                task_map[task_id] = prompt
                parent = 2 + execution
                prefix = list(range(parent + 2))
                token = prefix[-1]
                state = np.asarray([execution + 0.25, -execution - 0.5], dtype=np.float32)
                heads.append(
                    {
                        "task_id": task_id,
                        "round_index": 0,
                        "depth": 0,
                        "parent_position": parent,
                        "input_position": parent + 1,
                        "label_position": parent + 2,
                        "input_token_id": token,
                        "prefix_token_ids": prefix,
                        "state_row": execution,
                        "state_dim": 2,
                        "state_dtype": "float32_native_endian",
                        "state_boundary": "native_output_norm_f32_before_head_operand_conversion",
                        "finite": True,
                        "alignment_valid": True,
                    }
                )
                states.append(
                    {
                        "event": "seed",
                        "seq_id": task_id,
                        "position": parent,
                        "token": token,
                        "kv_max_before": parent - 1,
                    }
                )
                cache_rows.append(
                    {
                        "event": "row",
                        "execution": execution,
                        "column": 0,
                        "position": parent,
                        "token_id": token,
                    }
                )
                graph_rows.append(
                    {
                        "schema": "eagle_draft_graph_v1",
                        "event": "tensor",
                        "tensor_name": "result_norm",
                        "group_kind": "decoder",
                        "group_execution": execution,
                        "n_tokens": 1,
                        "token_width": 2,
                        "ne": [2, 1],
                        "token_axis": 1,
                        "f32_offset": offset,
                        "f32_count": 2,
                        "f32_bytes": 8,
                    }
                )
                graph_values.extend(state.tolist())
                offset += 2
                numeric[domain] = [
                    {
                        "round_index": 0,
                        "parent_position": parent,
                        "prefix_token_ids": prefix,
                        "preceding_student_round_outcome": "unknown",
                    }
                ]
                gradients[domain] = [
                    {
                        "root_index": 0,
                        "round_index": 0,
                        "context_decoder_positions": list(range(parent)),
                        "proposal_decoder_positions": [parent],
                        "final_cache_length": parent + 1,
                    }
                ]

            heads_path = capture_dir / "heads.jsonl"
            write_jsonl(heads_path, heads)
            (capture_dir / "heads.f32").write_bytes(
                np.asarray(
                    [
                        value
                        for index in range(len(heads))
                        for value in (index + 0.25, -index - 0.5)
                    ],
                    dtype="<f4",
                ).tobytes()
            )
            write_jsonl(capture_dir / "state.jsonl", states)
            write_jsonl(capture_dir / "heads.draft_cache.jsonl", cache_rows)
            graph_path = capture_dir / "heads.draft_graph.jsonl"
            footer = {
                "schema": "eagle_draft_graph_v1",
                "event": "capture_end",
                "status": "complete",
                "reason": "",
                "tensor_rows": len(graph_rows),
                "execution_count": len(graph_rows),
                "bytes_written": len(graph_values) * 4,
                "decoder_groups": len(graph_rows),
                "encoder_groups": 0,
            }
            write_jsonl(graph_path, [*graph_rows, footer])
            graph_values_path = capture_dir / "heads.draft_graph.f32"
            graph_values_path.write_bytes(np.asarray(graph_values, dtype="<f4").tobytes())

            bridge = _bridge_roots(
                numeric,
                gradients,
                task_map,
                capture_dir,
                states,
                cache_rows,
                graph_path,
                graph_values_path,
            )
            self.assertEqual(len(bridge), 3)
            self.assertTrue(all(row["native_result_norm_state_exact"] for row in bridge))

            states[0]["kv_max_before"] = 0
            with self.assertRaisesRegex(ValueError, "logical cache length"):
                _bridge_roots(
                    numeric,
                    gradients,
                    task_map,
                    capture_dir,
                    states,
                    cache_rows,
                    graph_path,
                    graph_values_path,
                )


if __name__ == "__main__":
    unittest.main()
