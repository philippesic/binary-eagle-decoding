"""CPU checks for frozen prompts and post-wrapper head input capture."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.capture_qat_head_inputs import (  # noqa: E402
    HeadCapture,
    HeadInputReservoir,
    save_split,
    trajectory_for_index,
    validate_prompt_manifest,
)
from scripts.generate_qat_head_prompts import (  # noqa: E402
    content_hash,
    generate,
    read_jsonl,
    validate_disjoint,
)
from w1a1_eagle import install_w1a1  # noqa: E402


class MockDrafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)

    def topK_genrate(self, inputs):
        return [self.lm_head(inputs + index) for index in range(2)]


class PromptGeneratorTests(unittest.TestCase):
    def test_balanced_nonoverlap_and_deterministic_files(self):
        heldout = ROOT / "configs/acceptance_prompts.jsonl"
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            manifest_a = generate(Path(left), heldout)
            manifest_b = generate(Path(right), heldout)
            self.assertEqual(manifest_a, manifest_b)
            parsed, splits = validate_prompt_manifest(Path(left), heldout)
            self.assertEqual(parsed, manifest_a)
            self.assertEqual((len(splits["train"]), len(splits["validation"])), (96, 24))
            for split, expected in (("train", 32), ("validation", 8)):
                for category in ("prose", "code", "reasoning"):
                    self.assertEqual(
                        sum(row["category"] == category for row in splits[split]), expected
                    )
                    self.assertEqual(
                        sum(
                            row["category"] == category
                            and trajectory_for_index(index) == "ordinary"
                            for index, row in enumerate(splits[split])
                        ),
                        expected // 2,
                    )
                name = f"{split}.jsonl"
                self.assertEqual(
                    (Path(left) / name).read_bytes(), (Path(right) / name).read_bytes()
                )

            heldout_rows = read_jsonl(heldout)
            validate_disjoint(splits["train"] + splits["validation"], heldout_rows)
            duplicate = {
                "id": heldout_rows[0]["id"],
                "messages": [{"role": "user", "content": "An otherwise unique test prompt."}],
            }
            duplicate["content_sha256"] = content_hash(duplicate["messages"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_disjoint(splits["train"] + [duplicate], heldout_rows)
            duplicate["id"] = "unique-id"
            duplicate["messages"] = heldout_rows[0]["messages"]
            duplicate["content_sha256"] = content_hash(duplicate["messages"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_disjoint(splits["train"] + [duplicate], heldout_rows)

    def test_tampered_prompt_file_is_rejected(self):
        heldout = ROOT / "configs/acceptance_prompts.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            generate(directory, heldout)
            path = directory / "train.jsonl"
            rows = read_jsonl(path)
            rows[0]["messages"][0]["content"] += " tampered"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "file hash changed"):
                validate_prompt_manifest(directory, heldout)


class CaptureTests(unittest.TestCase):
    def test_post_wrapper_hook_and_depth_metadata(self):
        drafter = MockDrafter()
        target = nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
        original_head = drafter.lm_head
        wrapper = install_w1a1(drafter, ["lm_head"], enabled=False, target=target)
        self.assertIsNot(drafter.lm_head, original_head)
        vectors = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        try:
            with HeadCapture(drafter, per_prompt_cap=20, seed=7, expected_width=4) as hook:
                hook.start_prompt("ordinary", "ordinary", 0)
                ordinary_outputs = drafter.topK_genrate(vectors)
                ordinary_rows, ordinary_seen, ordinary_calls = hook.finish_prompt()
                wrapper.set_enabled(True)
                hook.start_prompt("binary", "head_w1a1", 1)
                binary_outputs = drafter.topK_genrate(vectors)
                binary_rows, binary_seen, binary_calls = hook.finish_prompt()
            self.assertEqual((ordinary_seen, binary_seen), (6, 6))
            self.assertEqual((ordinary_calls, binary_calls), (1, 1))
            for row_set, trajectory in ((ordinary_rows, "ordinary"), (binary_rows, "head_w1a1")):
                self.assertEqual(
                    [item[1]["tree_depth_index"] for item in row_set], [0, 0, 0, 1, 1, 1]
                )
                self.assertEqual([item[1]["row_in_call"] for item in row_set], [0, 1, 2, 0, 1, 2])
                self.assertTrue(all(item[1]["trajectory"] == trajectory for item in row_set))
                self.assertTrue(
                    torch.equal(torch.stack([item[0] for item in row_set[:3]]), vectors)
                )
            self.assertFalse(torch.equal(ordinary_outputs[0], binary_outputs[0]))
        finally:
            wrapper.uninstall()
        self.assertIs(drafter.lm_head, original_head)

    def test_reservoir_is_capped_and_repeatable(self):
        values = torch.arange(80, dtype=torch.bfloat16).reshape(20, 4)
        outputs = []
        for _ in range(2):
            reservoir = HeadInputReservoir(cap=5, seed=12)
            reservoir.add(values[:10], {"prompt_id": "p", "trajectory": "ordinary"})
            reservoir.add(values[10:], {"prompt_id": "p", "trajectory": "ordinary"})
            self.assertEqual(reservoir.seen, 20)
            self.assertEqual(len(reservoir.rows), 5)
            outputs.append(reservoir.rows)
        self.assertEqual([meta for _, meta in outputs[0]], [meta for _, meta in outputs[1]])
        self.assertTrue(
            torch.equal(
                torch.stack([row for row, _ in outputs[0]]),
                torch.stack([row for row, _ in outputs[1]]),
            )
        )

    def test_saved_artifact_preserves_bf16_and_global_cap(self):
        records = [
            (torch.full((4,), index, dtype=torch.bfloat16), {"index": index}) for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.pt"
            spec = save_split(path, records, global_cap=6, seed=3)
            artifact = torch.load(path, weights_only=False)
            self.assertEqual(spec["rows"], 6)
            self.assertEqual(artifact["inputs"].shape, (6, 4))
            self.assertEqual(artifact["inputs"].dtype, torch.bfloat16)
            self.assertEqual(len(artifact["rows"]), 6)


if __name__ == "__main__":
    unittest.main()
