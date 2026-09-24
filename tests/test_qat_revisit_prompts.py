import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_qat_revisit_prompts.py"
SPEC = importlib.util.spec_from_file_location("qat_revisit_prompts", SCRIPT)
qat = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(qat)


class TestQATRevisitPrompts(unittest.TestCase):
    def test_manifest_is_balanced_disjoint_and_family_separated(self):
        splits = qat.build_splits()
        self.assertEqual(
            {key: len(rows) for key, rows in splits.items()},
            {"train": 96, "development": 24, "final": 24},
        )
        for split, rows in splits.items():
            expected = 32 if split == "train" else 8
            self.assertEqual(
                {
                    category: sum(row["category"] == category for row in rows)
                    for category in qat.CORPUS
                },
                {"prose": expected, "code": expected, "reasoning": expected},
            )

        acceptance = qat.read_jsonl(qat.ROOT / "configs/acceptance_prompts.jsonl")
        audits = qat.audit_splits(splits, qat.ROOT / "configs/acceptance_prompts.jsonl")
        self.assertEqual(audits[0]["prompts"], len(acceptance))
        self.assertEqual(len(acceptance), 12)
        topic_owner = {}
        template_owner = {}
        for split, rows in splits.items():
            for row in rows:
                self.assertEqual(row["content_sha256"], qat.content_hash(row["messages"]))
                self.assertEqual(topic_owner.setdefault(row["topic_family"], split), split)
                self.assertEqual(template_owner.setdefault(row["template_family"], split), split)

    def test_generation_is_byte_deterministic(self):
        acceptance = qat.ROOT / "configs/acceptance_prompts.jsonl"
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            qat.generate(first, acceptance, pilot_dir=None)
            qat.generate(second, acceptance, pilot_dir=None)
            for name in ("train.jsonl", "development.jsonl", "final.jsonl", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_pilot_manifest_overlap_is_rejected(self):
        splits = qat.build_splits()
        first = splits["train"][0]
        with tempfile.TemporaryDirectory() as temporary:
            pilot_dir = Path(temporary) / "pilot"
            pilot_dir.mkdir()
            (pilot_dir / "train.jsonl").write_text(
                json.dumps({"id": "pilot-only-id", "messages": first["messages"]}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "content overlap with first-pilot"):
                qat.audit_splits(splits, qat.ROOT / "configs/acceptance_prompts.jsonl", pilot_dir)

    def test_original_acceptance_prompt_overlap_is_rejected(self):
        splits = qat.build_splits()
        with tempfile.TemporaryDirectory() as temporary:
            acceptance = Path(temporary) / "acceptance.jsonl"
            acceptance.write_text(
                json.dumps({"id": "historical-only-id", "messages": splits["final"][0]["messages"]})
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "content overlap with acceptance"):
                qat.audit_splits(splits, acceptance)


if __name__ == "__main__":
    unittest.main()
