"""CPU contract tests for the bounded cached-input drafter-head trainer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_qat_head import (  # noqa: E402
    _batch_metrics,
    load_capture,
    make_parser,
    sha256_file,
    train,
)


class TinyCapture:
    def __init__(self, directory: Path):
        self.directory = directory
        self.heldout = directory / "heldout.jsonl"
        self.output = directory / "output"
        self.width = 5
        self.vocab = 7
        self._write()

    @staticmethod
    def _hash(messages: list[dict]) -> str:
        import hashlib

        text = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def _prompt(self, prompt_id: str) -> dict:
        messages = [{"role": "user", "content": f"A distinct question: {prompt_id}"}]
        return {
            "id": prompt_id,
            "category": "prose",
            "messages": messages,
            "content_sha256": self._hash(messages),
        }

    @staticmethod
    def _json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value) + "\n")

    def _write(self) -> None:
        torch.manual_seed(3)
        self._json(self.heldout, self._prompt("heldout"))
        split_specs = {}
        capture_specs = {}
        prompt_stats = []
        for split, count in (("train", 4), ("validation", 3)):
            ids = [f"{split}-{index}" for index in range(2)]
            prompts = [self._prompt(prompt_id) for prompt_id in ids]
            prompt_path = self.directory / f"{split}-prompts.jsonl"
            prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
            split_specs[split] = {
                "path": f"{split}.jsonl",
                "sha256": sha256_file(prompt_path),
                "prompts": len(prompts),
            }
            inputs = torch.randn(count, self.width, dtype=torch.bfloat16)
            rows = []
            for index in range(count):
                row = {
                    "prompt_id": ids[index % len(ids)],
                    "trajectory": "ordinary" if index % 2 == 0 else "head_w1a1",
                    "draft_call_index": index,
                    "head_call_index": 0,
                    "tree_depth_index": 0,
                    "row_in_call": 0,
                }
                rows.append(row)
            torch.save({"inputs": inputs, "rows": rows}, self.directory / f"{split}.pt")
            capture_specs[split] = {
                "path": f"{split}.pt",
                "sha256": sha256_file(self.directory / f"{split}.pt"),
                "rows": count,
                "shape": [count, self.width],
                "dtype": "torch.bfloat16",
            }
            prompt_stats += [
                {
                    "split": split,
                    "prompt_id": prompt_id,
                    "trajectory": "ordinary" if index == 0 else "head_w1a1",
                }
                for index, prompt_id in enumerate(ids)
            ]
        self._json(
            self.directory / "prompt-manifest.json",
            {
                "schema_version": 1,
                "heldout_sha256": sha256_file(self.heldout),
                "splits": split_specs,
            },
        )
        weight = torch.randn(self.vocab, self.width, dtype=torch.bfloat16)
        torch.save({"weight": weight}, self.directory / "teacher-head.pt")
        self._json(
            self.directory / "capture-manifest.json",
            {
                "schema_version": 1,
                "prompt_manifest_sha256": sha256_file(self.directory / "prompt-manifest.json"),
                "heldout_prompt_sha256": sha256_file(self.heldout),
                "teacher_head": {
                    "path": "teacher-head.pt",
                    "sha256": sha256_file(self.directory / "teacher-head.pt"),
                    "shape": [self.vocab, self.width],
                    "dtype": "torch.bfloat16",
                },
                "captures": capture_specs,
                "prompts": prompt_stats,
            },
        )

    def args(self, *extra: str):
        return make_parser().parse_args(
            [
                "--capture-dir",
                str(self.directory),
                "--output-dir",
                str(self.output),
                "--heldout-prompts",
                str(self.heldout),
                "--device",
                "cpu",
                "--hidden-size",
                str(self.width),
                "--vocab-size",
                str(self.vocab),
                "--batch-size",
                "2",
                "--validation-batch-size",
                "2",
                "--warmup-steps",
                "1",
                "--max-steps",
                "3",
                "--validation-interval",
                "1",
                *extra,
            ]
        )


class HeadTrainerTests(unittest.TestCase):
    def test_dry_run_verifies_capture_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = TinyCapture(Path(temporary))
            result = train(capture.args("--dry-run"))
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["rows"], {"train": 4, "validation": 3})
            self.assertGreaterEqual(result["step0"]["kl"], 0)
            self.assertFalse(capture.output.exists())

    def test_bounded_training_exports_best_weight_and_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = TinyCapture(Path(temporary))
            result = train(capture.args())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps_done"], 3)
            self.assertEqual(result["stop_reason"], "early_stop")
            self.assertIn(result["best_step"], range(4))
            for name in ("step0.pt", "best.pt", "last.pt", "best-head.pt"):
                self.assertEqual(
                    sha256_file(capture.output / name), result["checkpoint_sha256"][name]
                )
            best = torch.load(capture.output / "best.pt", weights_only=True)
            first = torch.load(capture.output / "step0.pt", weights_only=True)
            last = torch.load(capture.output / "last.pt", weights_only=True)
            exported = torch.load(capture.output / "best-head.pt", weights_only=True)
            self.assertEqual(best["latent_weight"].dtype, torch.float32)
            self.assertFalse(torch.equal(first["latent_weight"], last["latent_weight"]))
            self.assertTrue(last["optimizer"]["state"])
            self.assertTrue(
                torch.equal(exported["weight"], best["latent_weight"].to(torch.bfloat16))
            )
            self.assertIn("optimizer", best)
            self.assertIn("torch_rng_state", best)
            self.assertIn("python_rng_state", best)
            self.assertIn("sampler_order", best)
            self.assertIn("sampler_cursor", best)
            self.assertEqual(
                json.loads((capture.output / "training-summary.json").read_text())["best_step"],
                result["best_step"],
            )

    def test_rejects_changed_artifact_hash_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = TinyCapture(Path(temporary))
            with (capture.directory / "train.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                train(capture.args("--dry-run"))
            self.assertFalse(capture.output.exists())

    def test_rejects_train_validation_prompt_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = TinyCapture(Path(temporary))
            path = capture.directory / "validation.pt"
            payload = torch.load(path, weights_only=True)
            payload["rows"][0]["prompt_id"] = "train-0"
            torch.save(payload, path)
            manifest_path = capture.directory / "capture-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["captures"]["validation"]["sha256"] = sha256_file(path)
            capture._json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_capture(capture.directory, capture.width, capture.vocab, capture.heldout)

    def test_rejects_heldout_content_with_different_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = TinyCapture(Path(temporary))
            path = capture.directory / "train-prompts.jsonl"
            prompts = [json.loads(line) for line in path.read_text().splitlines()]
            heldout = json.loads(capture.heldout.read_text())
            prompts[0]["messages"] = heldout["messages"]
            prompts[0]["content_sha256"] = heldout["content_sha256"]
            path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
            prompt_manifest_path = capture.directory / "prompt-manifest.json"
            prompt_manifest = json.loads(prompt_manifest_path.read_text())
            prompt_manifest["splits"]["train"]["sha256"] = sha256_file(path)
            capture._json(prompt_manifest_path, prompt_manifest)
            capture_manifest_path = capture.directory / "capture-manifest.json"
            capture_manifest = json.loads(capture_manifest_path.read_text())
            capture_manifest["prompt_manifest_sha256"] = sha256_file(prompt_manifest_path)
            capture._json(capture_manifest_path, capture_manifest)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_capture(capture.directory, capture.width, capture.vocab, capture.heldout)

    def test_nonfinite_logits_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _batch_metrics(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
