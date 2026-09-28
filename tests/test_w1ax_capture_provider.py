"""Lazy project provider tests with tiny files and injected model doubles."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_w1ax_train_manifest as prep  # noqa: E402
import w1ax_capture_provider as provider_module  # noqa: E402
from compact_w1a_teacher import compact  # noqa: E402
from test_recurrent_provider import TinyStep, dense_drafter, provider_round  # noqa: E402

from w1a1_eagle.recurrent_provider import train_from_provider  # noqa: E402
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract  # noqa: E402


class FakeCapture:
    def __init__(self):
        original = provider_round()
        rows = []
        for i, row in enumerate(original.rows):
            rows.append(dict(row, target_logits_row=i))
        self.round = SimpleNamespace(
            anchor=original.anchor,
            rows=tuple(rows),
            prefix_token_ids=original.prefix_token_ids,
            raw_target_features=original.raw_target_features,
            feature_positions=original.feature_positions,
        )
        self.anchors = {("p", 0): original.anchor}
        self.rows = {("p", 0): tuple(rows)}

    def round_inputs(self, prompt_id, round_index):
        if (prompt_id, round_index) != ("p", 0):
            raise KeyError((prompt_id, round_index))
        return self.round


class ProviderFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture_dir = self.root / "capture"
        self.capture_dir.mkdir()
        self.prompts = self.capture_dir / "train_prompts.jsonl"
        self.prompts.write_text(
            json.dumps({"id": "p", "messages": [{"role": "user", "content": "x"}]}) + "\n"
        )
        self.offsets = self.capture_dir / "offsets.npy"
        np.save(self.offsets, np.array([0, 1, 1], dtype=np.int64))
        self.absolute = self.root / "d2t.npy"
        np.save(self.absolute, np.array([0, 2, 3], dtype=np.int64))
        self.target = self.root / "target.gguf"
        self.candidate = self.root / "candidate.gguf"
        self.base = self.root / "draft-f16.gguf"
        for path, data in (
            (self.target, b"target"),
            (self.candidate, b"candidate"),
            (self.base, b"draft"),
        ):
            path.write_bytes(data)
        self.target_hash = provider_module.sha256(self.target)
        self.candidate_hash = provider_module.sha256(self.candidate)
        self.base_hash = provider_module.sha256(self.base)
        self.cell = self.capture_dir / "source_cell_manifest.json"
        self.cell.write_text(
            json.dumps({"target_sha256": self.target_hash, "draft_sha256": self.candidate_hash})
        )
        self.capture_manifest = self.capture_dir / "manifest.json"
        self._write_capture(False)
        self.model_manifest = self.root / "models.json"
        models = {}
        for role, repo, revision in (
            ("target", provider_module.TARGET_REPO, provider_module.TARGET_REVISION),
            ("draft", provider_module.DRAFT_REPO, provider_module.DRAFT_REVISION),
        ):
            directory = self.root / role
            directory.mkdir()
            file = directory / "tiny.safetensors"
            file.write_bytes(role.encode())
            models[role] = {
                "repo": repo,
                "revision": revision,
                "directory": str(directory),
                "files": [
                    {
                        "path": file.name,
                        "bytes": file.stat().st_size,
                        "sha256": provider_module.sha256(file),
                    }
                ],
            }
        self.model_manifest.write_text(json.dumps({"models": models}))
        self.capture_loader_calls = 0
        self.model_loader_calls = 0

    def _write_capture(self, eligible):
        value = {
            "schema": "recurrent_binary_capture_v1",
            "split": "train",
            "training_eligible": eligible,
            "readiness": "full_body_qat_eligible" if eligible else "preparation_only",
            "unverified_gates": [] if eligible else ["native_trajectory"],
            "pinned_source_artifact_hashes_verified": True,
            "prompts_sha256": provider_module.sha256(self.prompts),
            "target_vocab_size": 4,
            "draft_vocab_size": 3,
            "max_depth": 2,
            "offsets": {"path": self.offsets.name, "sha256": provider_module.sha256(self.offsets)},
            "source_report_sha256": {"cell_manifest": provider_module.sha256(self.cell)},
        }
        self.capture_manifest.write_text(json.dumps(value))

    def _capture_loader(self, *args, **kwargs):
        self.capture_loader_calls += 1
        self.assertEqual(kwargs["expected_prompt_count"], 1)
        return FakeCapture()

    def _model_loader(self, paths):
        self.model_loader_calls += 1
        return dense_drafter(), nn.Linear(4, 4)

    def _prepare(self, output, *, teacher_dir=None):
        return prep.prepare(
            self.capture_manifest,
            self.prompts,
            self.absolute,
            self.target,
            self.candidate,
            self.base,
            self.base_hash,
            self.model_manifest,
            "cap",
            output,
            teacher_dir=teacher_dir,
        )

    def _make_teacher(self):
        logits = self.root / "logits.npy"
        np.save(logits, np.array([[0.0, 0.1, 1.0, -0.2], [0.2, -0.1, 1.0, 0.0]], dtype=np.float32))
        rows = self.root / "teacher-rows.jsonl"
        records = [
            {
                "id": "r0",
                "prompt_id": "p",
                "capture_id": "cap",
                "logits_row": 0,
                "prefix_token_ids": [0, 1],
                "next_target_id": 2,
            },
            {
                "id": "r1",
                "prompt_id": "p",
                "capture_id": "cap",
                "logits_row": 1,
                "prefix_token_ids": [0, 1, 1],
                "next_target_id": 2,
            },
        ]
        rows.write_text("".join(json.dumps(row) + "\n" for row in records))
        output = self.root / "teacher"
        compact(
            logits,
            "npy",
            rows,
            self.absolute,
            output,
            target_vocab=4,
            topk=2,
            shard_rows=1,
            target_gguf_sha256=self.target_hash,
            prompts_sha256=provider_module.sha256(self.prompts),
            capture_manifest_sha256=provider_module.sha256(self.capture_manifest),
        )
        return output

    def test_ineligible_bundle_fails_before_capture_or_model_loader(self):
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            manifest = self.root / "train.json"
            result = self._prepare(manifest)
            self.assertFalse(result["training_eligible"])
            with self.assertRaisesRegex(ValueError, "not eligible"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4)),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            tampered = json.loads(manifest.read_text())
            tampered["training_eligible"] = True
            manifest.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "remains training-ineligible"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4)),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            self.assertEqual(self.capture_loader_calls, 0)
            self.assertEqual(self.model_loader_calls, 0)

    def test_eligible_compact_provider_binds_shards_then_loads_doubles(self):
        self._write_capture(True)
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            teacher = self._make_teacher()
            manifest = self.root / "train.json"
            self._prepare(manifest, teacher_dir=teacher)
            config = JointQATConfig(W1AxContract(4), objective="compact_probability")
            provider = provider_module.NativeCaptureProvider(
                config,
                manifest,
                capture_loader=self._capture_loader,
                model_loader=self._model_loader,
                adapter_factory=TinyStep,
            )
            self.assertEqual(self.model_loader_calls, 0)
            self.assertEqual(len(list(provider.rounds())), 1)
            linears, metrics = train_from_provider(provider, config, max_rounds=1)
            self.assertEqual(len(linears), 9)
            self.assertGreater(metrics[0]["loss"], 0)
            self.assertEqual(self.model_loader_calls, 1)
            self.assertEqual(metrics[0]["gradient_tensors"], 18)

    def test_teacher_from_other_capture_fails_before_model_loader(self):
        self._write_capture(True)
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            teacher = self._make_teacher()
            teacher_manifest = teacher / "manifest.json"
            content = json.loads(teacher_manifest.read_text())
            content["capture_manifest_sha256"] = "0" * 64
            teacher_manifest.write_text(json.dumps(content))
            manifest = self.root / "wrong-teacher.json"
            self._prepare(manifest, teacher_dir=teacher)
            with self.assertRaisesRegex(ValueError, "different native capture"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4), objective="compact_probability"),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            self.assertEqual(self.model_loader_calls, 0)


if __name__ == "__main__":
    unittest.main()
