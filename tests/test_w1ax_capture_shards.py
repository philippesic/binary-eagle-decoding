"""CPU-only split/row-cap and observed native-cell storage gates."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_w1a_data import canonical, sha256  # noqa: E402
from prepare_w1ax_capture_shards import emit_teacher_rows, observe_cell, plan_shards  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def fixture(base: Path, count: int = 6, same_topic: bool = False) -> Path:
    prompts = base / "train_small.jsonl"
    index = base / "train_small.index.jsonl"
    with prompts.open("wb") as prompt_file, index.open("wb") as index_file:
        for i in range(count):
            messages = [{"role": "user", "content": f"Synthetic train task {i} unique details."}]
            row = {"id": f"fixture:{i}", "domain": "prose", "messages": messages}
            prompt_file.write(canonical(row) + b"\n")
            meta = {"id": row["id"], "group": f"group:{i}",
                    "topic": "topic:same" if same_topic else None,
                    "word_count": 10,
                    "content_sha256": hashlib.sha256(canonical(messages)).hexdigest()}
            index_file.write(canonical(meta) + b"\n")
    manifest = base / "manifest.json"
    write_json(manifest, {"schema": "w1a_data_manifest_v1", "files": {
        "train_small": {"prompts": prompts.name, "prompts_sha256": sha256(prompts),
                        "index": index.name, "index_sha256": sha256(index),
                        "prompts_count": count},
        "final": {"prompts": "must-not-open-final.jsonl", "prompts_sha256": "0" * 64},
    }})
    return manifest


class ShardPlanTest(unittest.TestCase):
    def test_hash_pinned_coverage_deterministic_and_no_final_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = fixture(base)
            first = plan_shards(manifest, base / "first", target_vocab=5,
                                max_prompts=3, max_rows=6, max_raw_bytes=120,
                                base_rows_per_prompt=2, rows_per_word=0)
            second = plan_shards(manifest, base / "second", target_vocab=5,
                                 max_prompts=3, max_rows=6, max_raw_bytes=120,
                                 base_rows_per_prompt=2, rows_per_word=0)
            self.assertEqual(sha256(base / "first/plan.json"), sha256(base / "second/plan.json"))
            self.assertEqual(first, second)
            self.assertEqual(first["shard_count"], 2)
            self.assertEqual(sum(s["prompt_count"] for s in first["shards"]), 6)
            self.assertTrue(all(s["estimated_raw_logit_bytes"] <= 120
                                for s in first["shards"]))
            self.assertFalse(first["raw_retirement_allowed"])

    def test_topic_family_over_cap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = fixture(base, count=3, same_topic=True)
            with self.assertRaisesRegex(ValueError, "family exceeds"):
                plan_shards(manifest, base / "out", target_vocab=5,
                            max_prompts=2, max_rows=10, max_raw_bytes=200,
                            base_rows_per_prompt=2, rows_per_word=0)
            self.assertFalse((base / "out").exists())

    def test_observed_cell_row_cap_and_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = fixture(base, count=2)
            plan_shards(manifest, base / "shards", target_vocab=5,
                        max_prompts=2, max_rows=4, max_raw_bytes=80,
                        base_rows_per_prompt=2, rows_per_word=0)
            shard_manifest = base / "shards/shard-0000/manifest.json"
            shard = json.loads(shard_manifest.read_text())
            cell_dir = base / "cell"
            cell_dir.mkdir()
            raw = cell_dir / "heads.target_logits.f32"
            raw.write_bytes(b"\0" * 80)
            cell = {"schema": "binary_head_capture_cell_v1", "complete": True,
                    "prompts_sha256": shard["prompts_sha256"],
                    "task_prompt_ids": {"0": "fixture:0", "1": "fixture:1"},
                    "requests": [{"id": "fixture:0", "task_id": "0",
                                  "target_logit_rows": [0, 2]},
                                 {"id": "fixture:1", "task_id": "1",
                                  "target_logit_rows": [2, 4]}],
                    "files": {raw.name: {"bytes": 80, "sha256": sha256(raw)}}}
            cell_manifest = cell_dir / "manifest.json"
            write_json(cell_manifest, cell)
            report = observe_cell(shard_manifest, cell_manifest, base / "observed.json")
            self.assertEqual(report["observed_verifier_logit_rows"], 4)
            self.assertFalse(report["raw_retirement_allowed"])
            raw.write_bytes(b"\0" * 100)
            cell["files"][raw.name] = {"bytes": 100, "sha256": sha256(raw)}
            cell["requests"][1]["target_logit_rows"] = [2, 5]
            write_json(cell_manifest, cell)
            with self.assertRaisesRegex(ValueError, "exceed shard hard cap"):
                observe_cell(shard_manifest, cell_manifest, base / "too-large.json")
            self.assertFalse((base / "too-large.json").exists())

    def test_teacher_row_adapter_keeps_logit_prefix_label_join(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = fixture(base, count=2)
            plan_shards(manifest, base / "shards", target_vocab=5,
                        max_prompts=2, max_rows=4, max_raw_bytes=80,
                        base_rows_per_prompt=2, rows_per_word=0)
            shard = base / "shards/shard-0000/manifest.json"
            shard_spec = json.loads(shard.read_text())
            bundle = base / "bundle"
            bundle.mkdir()
            raw = bundle / "target_logits.f32"
            raw.write_bytes(b"\0" * 80)
            rows = bundle / "rows.jsonl"
            with rows.open("wb") as stream:
                for i in range(4):
                    row = {"prompt_id": shard_spec["prompt_ids"][i // 2],
                           "split": "train", "valid": True, "alignment_valid": True,
                           "forced": False, "prefix_token_ids": [1, i + 1],
                           "verifier_token_id": i,
                           "label_source": (
                               "cloned_native_verifier_sampler_at_actual_proposal_prefix"
                           ),
                           "target_logits_row": i,
                           "target_logits_source": (
                               "raw_target_verifier_at_exact_proposal_prefix"
                           )}
                    stream.write(canonical(row) + b"\n")
            bundle_manifest = bundle / "manifest.json"
            write_json(bundle_manifest, {
                "schema": "recurrent_binary_capture_v1", "split": "train",
                "prompts_sha256": shard_spec["prompts_sha256"], "target_vocab_size": 5,
                "training_eligible": False,
                "rows": {"path": rows.name, "sha256": sha256(rows)},
                "target_logits": {"path": raw.name, "sha256": sha256(raw)},
            })
            write_json(bundle / "audit.json", {
                "schema": "recurrent_binary_capture_audit_v1",
                "capture_manifest_sha256": sha256(bundle_manifest),
                "training_prompts_sha256": shard_spec["prompts_sha256"],
                "source_sha256": {"rows": sha256(rows), "target_logits": sha256(raw)},
                "raw_target_logit_rows": 4,
            })
            output = base / "teacher_rows.jsonl"
            result = emit_teacher_rows(shard, bundle_manifest, output)
            self.assertEqual(result["rows"], 4)
            self.assertFalse(result["training_eligible"])
            mapped = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["logits_row"] for row in mapped], list(range(4)))
            self.assertEqual(mapped[3]["prefix_token_ids"], [1, 4])
            self.assertEqual(mapped[3]["next_target_id"], 3)
            raw.write_bytes(b"\0" * 60)
            with self.assertRaisesRegex(ValueError, "file/hash mismatch"):
                emit_teacher_rows(shard, bundle_manifest, base / "invalid_rows.jsonl")


if __name__ == "__main__":
    unittest.main()
