"""Synthetic native metadata checks; no model/GPU execution."""

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_recurrent_capture_v2 import CaptureV2Tests  # noqa: E402
from w1ax_continuous_stages import (  # noqa: E402
    audit_native_labels,
    build_native_labels,
    load_native_labels,
    sha256,
)


class NativeLabelStagesTests(unittest.TestCase):
    def setUp(self):
        fixture = CaptureV2Tests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.f = fixture
        values = fixture.cell / "heads.target_features.f32"
        shutil.copyfile(fixture.values, values)
        heads = fixture.cell / "heads.jsonl"
        rows = [json.loads(line) for line in heads.read_text().splitlines()]
        for row in rows:
            for key in ("target_logits_row", "target_logits_dim", "target_logits_source"):
                row[key] = None
        heads.write_text("".join(json.dumps(r) + "\n" for r in rows))
        request_path = fixture.cell / "request-000/request.json"
        request = json.loads(request_path.read_text())
        request.update(temperature=0, seed=42, cache_prompt=False)
        request_path.write_text(json.dumps(request))
        cell = json.loads(fixture.cell_manifest.read_text())
        cell["requests"][0]["request_sha256"] = sha256(request_path)
        cell.update(env={"GGML_W1AX_ACT_BITS": "16"}, binary_sha256="c" * 64)
        for path in (heads, values):
            cell["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        fixture.cell_manifest.write_text(json.dumps(cell))
        self.output = fixture.root / "fresh-labels"

    def build(self, split="train"):
        return build_native_labels(
            self.f.capture,
            self.f.native.prompts,
            self.f.native.absolute_map,
            self.output,
            split=split,
            target_vocab_size=8,
        )

    def test_native_label_only_exact_prefix_and_unsupported_mask(self):
        old_map = (self.f.capture / "task_prompt_ids.json").read_bytes()
        result = self.build()
        capture = load_native_labels(
            self.output / "manifest.json",
            expected_prompt_sha256=self.f.prompt_hash,
            expected_prompt_count=1,
        )
        self.assertFalse(result["training_eligible"])
        self.assertEqual(capture.report["counts"]["supported"], 1)
        self.assertEqual(capture.report["counts"]["unsupported"], 1)
        self.assertIsNone(capture.round_inputs("train-a", 0).rows[1]["target_logits_row"])
        self.assertEqual(old_map, (self.f.capture / "task_prompt_ids.json").read_bytes())
        self.assertFalse(any("logits" in p.name for p in self.output.iterdir()))

    def test_development_provenance_remains_independent(self):
        result = self.build("development")
        self.assertEqual(result["split"], "development")
        self.assertFalse(result["training_eligible"])
        report = audit_native_labels(
            self.output / "manifest.json",
            expected_prompt_sha256=self.f.prompt_hash,
            expected_prompt_count=1,
        )
        self.assertEqual(report["split"], "development")

    def test_corrupted_features_are_rejected(self):
        self.build()
        path = self.output / "features.npy"
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_native_labels(
                self.output / "manifest.json",
                expected_prompt_sha256=self.f.prompt_hash,
                expected_prompt_count=1,
            )

    def test_unclaimed_logits_cannot_hide_in_bundle(self):
        self.build()
        (self.output / "anything.f32").write_bytes(b"0")
        with self.assertRaisesRegex(ValueError, "unclaimed"):
            audit_native_labels(
                self.output / "manifest.json",
                expected_prompt_sha256=self.f.prompt_hash,
                expected_prompt_count=1,
            )

    def test_changed_label_rehash_does_not_evade_rederivation(self):
        self.build()
        path = self.output / "rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["verifier_token_id"] = 5
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = self.output / "manifest.json"
        m = json.loads(manifest.read_text())
        m["files"]["rows"]["sha256"] = sha256(path)
        manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "derivation"):
            audit_native_labels(
                manifest, expected_prompt_sha256=self.f.prompt_hash, expected_prompt_count=1
            )

    def test_import_does_not_query_accelerator(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")):
            self.build()


if __name__ == "__main__":
    unittest.main()
