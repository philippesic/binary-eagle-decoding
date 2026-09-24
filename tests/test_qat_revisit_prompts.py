import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_qat_revisit_prompts.py"
SPEC = importlib.util.spec_from_file_location("qat_revisit_prompts", SCRIPT)
qat = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(qat)


def test_manifest_is_balanced_disjoint_and_family_separated(tmp_path):
    splits = qat.build_splits()
    assert {key: len(rows) for key, rows in splits.items()} == {
        "train": 96,
        "development": 24,
        "final": 24,
    }
    for split, rows in splits.items():
        assert {
            category: sum(row["category"] == category for row in rows) for category in qat.CORPUS
        } == {
            "prose": 32 if split == "train" else 8,
            "code": 32 if split == "train" else 8,
            "reasoning": 32 if split == "train" else 8,
        }

    acceptance = qat.read_jsonl(qat.ROOT / "configs/acceptance_prompts.jsonl")
    audits = qat.audit_splits(splits, qat.ROOT / "configs/acceptance_prompts.jsonl")
    assert audits[0]["prompts"] == len(acceptance) == 12
    topic_owner = {}
    template_owner = {}
    for split, rows in splits.items():
        for row in rows:
            assert row["content_sha256"] == qat.content_hash(row["messages"])
            assert topic_owner.setdefault(row["topic_family"], split) == split
            assert template_owner.setdefault(row["template_family"], split) == split


def test_generation_is_byte_deterministic(tmp_path):
    acceptance = qat.ROOT / "configs/acceptance_prompts.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    qat.generate(first, acceptance, pilot_dir=None)
    qat.generate(second, acceptance, pilot_dir=None)
    for name in ("train.jsonl", "development.jsonl", "final.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_pilot_manifest_overlap_is_rejected(tmp_path):
    splits = qat.build_splits()
    first = splits["train"][0]
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    (pilot_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "id": "pilot-only-id",
                "messages": first["messages"],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="content overlap with first-pilot"):
        qat.audit_splits(splits, qat.ROOT / "configs/acceptance_prompts.jsonl", pilot_dir)


def test_original_acceptance_prompt_overlap_is_rejected(tmp_path):
    splits = qat.build_splits()
    acceptance = tmp_path / "acceptance.jsonl"
    acceptance.write_text(
        json.dumps(
            {
                "id": "historical-only-id",
                "messages": splits["final"][0]["messages"],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="content overlap with acceptance"):
        qat.audit_splits(splits, acceptance)
