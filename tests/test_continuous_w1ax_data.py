"""CPU fixtures: source pinning, split isolation, sealing and chat tokens."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_continuous_w1ax_data import (  # noqa: E402
    CpuTokenizer,
    DuplicateFilter,
    fetch,
    opaque_exclusions,
    prepare,
    require_untracked_destination,
    staged_output,
    transform,
)
from prepare_w1a_data import canonical, sha256  # noqa: E402


class ContinuousCorpusTests(unittest.TestCase):
    def test_atomic_publication_and_cleanup_on_write_failure(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "frozen"
            with self.assertRaisesRegex(OSError, "disk full"):
                with staged_output(output) as temporary:
                    (temporary / "partial.jsonl").write_text("partial")
                    raise OSError("disk full")
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(d).glob(".corpus-*")))
            with staged_output(output) as temporary:
                (temporary / "manifest.json").write_text("ready")
                self.assertFalse(output.exists())
            self.assertTrue((output / "manifest.json").exists())

    def test_refuse_git_tracked_destination(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "ignored by Git"):
            require_untracked_destination(root / "experiments" / "raw-dataset")
        require_untracked_destination(root / "data" / "new-ignored-corpus")

    def test_near_and_exact_duplicates(self):
        words = [f"specific{i}" for i in range(100)]
        duplicate = DuplicateFilter()
        self.assertTrue(duplicate.add(" ".join(words)))
        self.assertFalse(duplicate.add(" ".join(words)))
        words[50] = "altered"
        self.assertFalse(duplicate.add(" ".join(words)))
        self.assertEqual(dict(duplicate.dropped), {"exact_normalized": 1, "near_jaccard": 1})

    def test_opaque_index_rejects_sealed_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sealed.index.jsonl"
            p.write_text(json.dumps({"id": "x", "messages": ["forbidden"]}) + "\n")
            with self.assertRaisesRegex(ValueError, "forbidden prompt text"):
                opaque_exclusions([p])
            with self.assertRaisesRegex(ValueError, "opaque"):
                opaque_exclusions([Path(d) / "sealed.jsonl"])

    def test_source_integrity_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dolly.jsonl"
            p.write_text("tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch({"id": "dolly", "sha256": "0" * 64}, Path(d), True)
            with self.assertRaisesRegex(ValueError, "missing pinned"):
                fetch({"id": "gsm8k", "sha256": "0" * 64}, Path(d), True)

    def test_code_reused_upstream_index_has_unique_ids_but_shared_seed_group(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "code.jsonl"
            p.write_text(
                "".join(
                    json.dumps(
                        {"problem": f"task {i}", "index": 1, "seed": "same", "lang": "python"}
                    )
                    + "\n"
                    for i in range(2)
                )
            )
            rows = list(transform({"id": "magicoder", "domain": "code"}, p, 0))
            self.assertNotEqual(rows[0]["id"], rows[1]["id"])
            self.assertEqual(rows[0]["topic"], rows[1]["topic"])
            self.assertEqual(len(list(transform({"id": "magicoder", "domain": "code"}, p, 1))), 1)

    def test_tokenizer_matches_frozen_chat_template_without_model_load(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            tokenizer = Tokenizer(
                models.WordLevel(
                    {"<unk>": 0, "user": 1, "hello": 2, "assistant": 3, "no_think": 4},
                    unk_token="<unk>",
                )
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(p / "tokenizer.json"))
            (p / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "user {{ messages[0].content }} "
                        "{% if add_generation_prompt %}assistant{% endif %}"
                        "{% if not enable_thinking %} no_think{% endif %}"
                    }
                )
            )
            spec = {
                "files": {n: sha256(p / n) for n in ("tokenizer.json", "tokenizer_config.json")}
            }
            counter = CpuTokenizer(p, spec)
            self.assertEqual(counter.count([{"role": "user", "content": "hello"}]), 4)
            (p / "tokenizer_config.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "frozen tokenizer hash"):
                CpuTokenizer(p, spec)

    def test_deterministic_grouped_split_legacy_exclusion_and_no_sealed_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            sources = []
            for sid, domain in (("dolly", "prose"), ("gsm8k", "reasoning"), ("magicoder", "code")):
                raw = []
                for i in range(24):
                    text = " ".join(f"{sid}{i}w{j}" for j in range(20))
                    if sid == "dolly":
                        raw.append({"instruction": text, "context": "", "category": "open_qa"})
                    elif sid == "gsm8k":
                        raw.append({"question": text})
                    else:
                        raw.append(
                            {"problem": text, "index": i, "seed": f"seed{i // 2}", "lang": "python"}
                        )
                p = base / f"{sid}.jsonl"
                p.write_bytes(b"".join(canonical(r) + b"\n" for r in raw))
                sources.append({"id": sid, "domain": domain, "sha256": sha256(p)})
            lock = base / "lock.json"
            lock.write_text(
                json.dumps(
                    {"schema": "continuous_w1ax_sources_v1", "tokenizer": {}, "sources": sources}
                )
            )
            index = base / "legacy.index.jsonl"
            index.write_text(
                json.dumps({"id": "gsm8k:train-000001", "content_sha256": "a" * 64}) + "\n"
            )
            real_open = Path.open

            def guarded_open(path, mode="r", *a, **kw):
                if (
                    path.name.startswith("sealed_test")
                    and "r" in mode
                    and ".index." not in path.name
                ):
                    raise AssertionError("sealed text was reopened")
                return real_open(path, mode, *a, **kw)

            with (
                patch("prepare_continuous_w1ax_data.CpuTokenizer") as tokenizer,
                patch.object(Path, "open", guarded_open),
            ):
                tokenizer.return_value.count.return_value = 44
                first = prepare(lock, base, base, base / "one", [index], 12, 6, 6, shard_rows=4)
                second = prepare(lock, base, base, base / "two", [index], 12, 6, 6, shard_rows=4)
            self.assertEqual(first, second)
            self.assertEqual(first["filter"]["rejected"]["opaque_legacy_exclusion"], 1)
            identities, topics, exact = {}, {}, {}
            for name, split in first["files"].items():
                entries = []
                for shard in split["shards"]:
                    p = base / "one" / shard["index"]
                    self.assertEqual(sha256(p), shard["index_sha256"])
                    entries.extend(json.loads(line) for line in p.read_text().splitlines())
                identities[name] = {r["id"] for r in entries}
                topics[name] = {r["topic"] for r in entries} - {None}
                exact[name] = {r["content_sha256"] for r in entries}
            for i, name in enumerate(identities):
                for other in list(identities)[i + 1 :]:
                    self.assertFalse(identities[name] & identities[other])
                    self.assertFalse(topics[name] & topics[other])
                    self.assertFalse(exact[name] & exact[other])
            self.assertNotIn("gsm8k:train-000001", set.union(*identities.values()))
            self.assertGreaterEqual(first["files"]["train"]["prompts_count"], 12)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(lock, base, base, base / "one", [], 12, 6, 6)


if __name__ == "__main__":
    unittest.main()
