"""CPU-only fixtures for larger-data preparation and compact teacher rows."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compact_w1a_teacher import compact, iter_verified_shards  # noqa: E402
from prepare_w1a_data import deduplicate, prepare, sha256  # noqa: E402

HASH = "a" * 64


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class DataPreparationTest(unittest.TestCase):
    def test_near_duplicate_candidate_verified(self) -> None:
        words = [f"unique{i}" for i in range(100)]
        changed = words.copy()
        changed[40] = "altered"
        kept, dropped = deduplicate(
            [{"text": " ".join(words)}, {"text": " ".join(changed)}], 0.88
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped["near"], 1)

    def test_disjoint_grouped_nested_tiers_and_reproducible_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources = []
            for domain in ("prose", "code", "reasoning"):
                path = base / f"{domain}.jsonl"
                entries = []
                for i in range(18):
                    words = f"{domain} topic {i} " + " ".join(f"item{i}word{j}" for j in range(16))
                    entries.append({"id": f"{domain}-{i}",
                                    "messages": [{"role": "user", "content": words}],
                                    "group_id": f"g{i // 2}", "topic_id": f"t{i // 2}"})
                entries.append(dict(entries[0], id="duplicate"))
                path.write_text("".join(json.dumps(row) + "\n" for row in entries))
                sources.append({"id": domain, "uri": f"fixture://{domain}",
                                "path": path.name, "sha256": sha256(path),
                                "revision": "fixture-r1", "license": "fixture-only",
                                "domain": domain})
            catalog = base / "catalog.json"
            catalog.write_text(json.dumps({"schema": "w1a_source_catalog_v1", "sources": sources}))
            first = prepare(catalog, base / "first", 3, 9, 3, 3, 17, 10, 10000, 0.88)
            second = prepare(catalog, base / "second", 3, 9, 3, 3, 17, 10, 10000, 0.88)
            self.assertEqual(
                sha256(base / "first/manifest.json"), sha256(base / "second/manifest.json")
            )
            self.assertEqual(first["near_dedup"]["dropped"]["exact"], 3)
            self.assertEqual(first["files"], second["files"])
            splits = {name: lines(base / "first" / f"{name}.index.jsonl")
                      for name in ("train_small", "train_large", "dev", "final")}
            groups = {name: {row["group"] for row in rows} for name, rows in splits.items()}
            self.assertTrue(groups["train_small"] <= groups["train_large"])
            for left, right in (("train_large", "dev"), ("train_large", "final"),
                                ("dev", "final")):
                self.assertFalse(groups[left] & groups[right])
            self.assertGreaterEqual(first["files"]["train_large"]["prompts_count"], 9)
            self.assertEqual(first["token_counts"],
                             "pending_pinned_target_tokenizer; word counts are not token counts")

    def test_source_hash_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text('{"id":"a","messages":[{"role":"user","content":"hello"}]}\n')
            catalog = base / "catalog.json"
            catalog.write_text(json.dumps({"schema": "w1a_source_catalog_v1",
                                           "sources": [{"id": "s", "uri": "fixture://s",
                                                        "path": source.name, "sha256": HASH,
                                                        "revision": "r1", "license": "fixture",
                                                        "domain": "prose"}]}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare(catalog, base / "out", 1, 1, 1, 1, 1, 0, 100, 0.88)


class TeacherPreparationTest(unittest.TestCase):
    def test_full_target_mass_and_exact_prefix_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            logits = base / "logits.npy"
            d2t = base / "d2t.npy"
            rows = base / "rows.jsonl"
            np.save(logits, np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.float32))
            np.save(d2t, np.array([0, 2, 4], dtype=np.int32))
            rows.write_text("".join(json.dumps({"id": f"r{i}", "prompt_id": f"p{i}",
                                                  "capture_id": "native-a", "logits_row": i,
                                                  "prefix_token_ids": [1, i + 2],
                                                  "next_target_id": 4 - i}) + "\n"
                                    for i in range(2)))
            manifest = compact(logits, "npy", rows, d2t, base / "teacher", target_vocab=5,
                               topk=2, shard_rows=1, target_gguf_sha256=HASH,
                               prompts_sha256=HASH, capture_manifest_sha256=HASH)
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(len(manifest["shards"]), 2)
            with np.load(base / "teacher" / manifest["shards"][0]["path"]) as shard:
                self.assertAlmostEqual(float(shard["draft_topk_probs"].sum()
                                             + shard["draft_tail_mass"][0]
                                             + shard["outside_draft_mass"][0]), 1, places=6)
                self.assertGreater(float(shard["outside_draft_mass"][0]), 0)
                self.assertEqual(int(shard["next_draft_id"][0]), 2)
            index = lines(base / "teacher/rows.jsonl")
            expected = hashlib.sha256(b"[1,2]").hexdigest()
            self.assertEqual(index[0]["prefix_sha256"], expected)
            self.assertEqual(manifest["index_sha256"], sha256(base / "teacher/rows.jsonl"))
            verified = list(iter_verified_shards(
                base / "teacher", expected_prompts_sha256=HASH,
                expected_target_gguf_sha256=HASH, expected_d2t_sha256=sha256(d2t),
                expected_prefixes={"r0": [1, 2], "r1": [1, 3]},
            ))
            self.assertEqual(sum(len(rows) for rows, _ in verified), 2)
            with self.assertRaisesRegex(ValueError, "prefix does not match"):
                list(iter_verified_shards(
                    base / "teacher", expected_prompts_sha256=HASH,
                    expected_target_gguf_sha256=HASH, expected_d2t_sha256=sha256(d2t),
                    expected_prefixes={"r0": [1, 9], "r1": [1, 3]},
                ))


if __name__ == "__main__":
    unittest.main()
