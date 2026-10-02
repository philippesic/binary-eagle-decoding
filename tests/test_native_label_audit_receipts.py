"""Durable per-shard audit reuse; tiny synthetic CPU captures only."""

import copy
import json
import os
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_w1ax_capture_provider as provider_fixture  # noqa: E402
import test_w1ax_continuous_stages as fixture_module  # noqa: E402
import w1ax_capture_provider as provider  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402

from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract  # noqa: E402


class NativeLabelReceiptTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.NativeLabelStagesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.f = fixture
        self.manifest = fixture.output / "manifest.json"
        self.receipt = fixture.f.root / "audit-receipts/shard.json"
        self.kwargs = {
            "expected_prompt_sha256": fixture.f.prompt_hash,
            "expected_prompt_count": 1,
        }
        stages._VERIFIED_RECORDS.clear()
        stages._OBSERVED_RECORDS.clear()

    def build(self, *, receipt=False):
        stages.build_native_labels(
            self.f.f.capture,
            self.f.f.native.prompts,
            self.f.f.native.absolute_map,
            self.f.output,
            split="train",
            target_vocab_size=8,
            audit_receipt_path=self.receipt if receipt else None,
        )
        self.manifest_sha = stages.sha256(self.manifest)

    def audit(self):
        return stages.audit_native_labels_with_receipt(
            self.manifest,
            expected_manifest_sha256=self.manifest_sha,
            receipt_path=self.receipt,
            **self.kwargs,
        )

    def load(self):
        return stages.load_native_labels(
            self.manifest,
            expected_manifest_sha256=self.manifest_sha,
            audit_receipt=stages.file_record(self.receipt),
            **self.kwargs,
        )

    def test_old_sidecar_requires_full_audit_once_then_loads_are_equivalent(self):
        self.build()
        original = stages.load_native_labels(self.manifest, **self.kwargs)
        with patch.object(stages, "audit_native_labels", wraps=stages.audit_native_labels) as audit:
            report = self.audit()
            first, second = self.load(), self.load()
            self.assertEqual(audit.call_count, 1)
        self.assertEqual(report, original.report)
        self.assertEqual(first.report, second.report)
        self.assertEqual(first.anchors, original.anchors)
        self.assertEqual(first.rows, original.rows)
        for capture in (first, second):
            actual = capture.round_inputs("train-a", 0)
            expected = original.round_inputs("train-a", 0)
            self.assertEqual(actual.rows, expected.rows)
            self.assertEqual(actual.anchor, expected.anchor)
            self.assertEqual(
                actual.raw_target_features.numpy().tobytes(),
                expected.raw_target_features.numpy().tobytes(),
            )
        self.assertFalse(report["training_eligible"])

    def test_fresh_builder_publishes_original_full_audit_immediately(self):
        with patch.object(stages, "audit_native_labels", wraps=stages.audit_native_labels) as audit:
            self.build(receipt=True)
            self.audit()
            self.load()
            self.assertEqual(audit.call_count, 1)
        receipt = json.loads(self.receipt.read_text())
        self.assertTrue(receipt["full_semantic_audit"])
        self.assertEqual(receipt["binding"]["capture_manifest"]["sha256"], self.manifest_sha)
        self.assertIn(
            "scripts/prepare_recurrent_native_features.py",
            receipt["binding"]["audit_source"]["files"],
        )

    def test_interrupted_audit_and_failed_publication_leave_no_receipt(self):
        self.build()
        with patch.object(stages, "audit_native_labels", side_effect=InterruptedError("stop")):
            with self.assertRaises(InterruptedError):
                self.audit()
        self.assertFalse(self.receipt.exists())
        with patch.object(stages.os, "replace", side_effect=InterruptedError("stop")):
            with self.assertRaises(InterruptedError):
                self.audit()
        self.assertFalse(self.receipt.exists())
        self.assertEqual(list(self.receipt.parent.iterdir()), [])
        self.audit()
        self.assertTrue(self.receipt.is_file())

    def test_changed_manifest_even_rebound_digest_is_rejected(self):
        self.build(receipt=True)
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.audit()
        self.manifest_sha = stages.sha256(self.manifest)
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.audit()

    def test_changed_payload_request_and_prompt_reject_in_process_cache(self):
        self.build(receipt=True)
        self.audit()
        m = json.loads(self.manifest.read_text())
        paths = [
            m["files"]["features"]["path"],
            *(m["requests"][0][k]["path"] for k in ("request", "response", "prompt")),
            m["files"]["prompts"]["path"],
        ]
        for basename in paths:
            path = self.manifest.parent / basename
            original = path.read_bytes()
            with self.subTest(path=basename):
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.audit()
            path.write_bytes(original)
            self.audit()

    def test_changed_hardlink_with_restored_mtime_rejects(self):
        self.build(receipt=True)
        self.audit()
        manifest = json.loads(self.manifest.read_text())
        payload = self.manifest.parent / manifest["files"]["native_feature_values"]["path"]
        alias = self.receipt.parent / "alias.f32"
        os.link(payload, alias)
        stat = payload.stat()
        value = alias.read_bytes()
        alias.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
        os.utime(alias, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(payload.stat().st_mtime_ns, stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.audit()

    def test_inventory_checked_on_each_access(self):
        self.build(receipt=True)
        self.audit()
        (self.manifest.parent / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.audit()

    def test_source_change_and_prompt_binding_change_reject_without_semantic_audit(self):
        self.build(receipt=True)
        source = copy.deepcopy(stages.native_label_audit_source())
        source["files"]["scripts/prepare_recurrent_native_rows.py"] = "0" * 64
        with (
            patch.object(stages, "native_label_audit_source", return_value=source),
            patch.object(stages, "audit_native_labels", side_effect=AssertionError("re-audit")),
        ):
            with self.assertRaisesRegex(ValueError, "binding differs"):
                self.audit()
        self.kwargs["expected_prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt/split/policy"):
            self.audit()

    def test_source_bytes_are_rechecked_after_identity_change(self):
        source_root = self.f.f.root / "source"
        for name in stages.native_label_audit_source()["files"]:
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stages.ROOT / name, destination)
        with patch.object(stages, "ROOT", source_root):
            self.build(receipt=True)
            source_file = source_root / "scripts/audit_recurrent_response.py"
            stat = source_file.stat()
            value = source_file.read_bytes()
            source_file.write_bytes(value[:-1] + b" ")
            os.utime(source_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "binding differs"):
                self.audit()

    def test_new_process_hashes_actual_payloads_once_then_reuses_verified_identity(self):
        self.build(receipt=True)
        stages._VERIFIED_RECORDS.clear()
        stages._OBSERVED_RECORDS.clear()
        with patch.object(stages, "sha256", wraps=stages.sha256) as digest:
            self.audit()
            self.audit()
            counts = Counter(Path(call.args[0]).resolve() for call in digest.call_args_list)
        manifest = json.loads(self.manifest.read_text())
        for record in manifest["files"].values():
            self.assertEqual(counts[(self.manifest.parent / record["path"]).resolve()], 1)
        for request in manifest["requests"]:
            for kind in ("request", "response", "prompt"):
                self.assertEqual(
                    counts[(self.manifest.parent / request[kind]["path"]).resolve()],
                    1,
                )

    def test_receipt_hash_binding_and_external_path_required(self):
        self.build(receipt=True)
        binding = stages.file_record(self.receipt)
        self.receipt.write_bytes(self.receipt.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            stages.load_native_labels(
                self.manifest,
                audit_receipt=binding,
                expected_manifest_sha256=self.manifest_sha,
                **self.kwargs,
            )
        with self.assertRaisesRegex(ValueError, "external"):
            stages.audit_native_labels_with_receipt(
                self.manifest,
                receipt_path=self.manifest.parent / "receipt.json",
                expected_manifest_sha256=self.manifest_sha,
                **self.kwargs,
            )

    def test_report_only_receipt_corruption_rejects_before_stage_reuse(self):
        self.build(receipt=True)
        receipt = json.loads(self.receipt.read_text())
        receipt["report"]["counts"]["supported"] += 1
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "report binding"):
            self.audit()

    def test_legacy_loads_still_full_audit_every_time(self):
        self.build()
        with patch.object(stages, "audit_native_labels", wraps=stages.audit_native_labels) as audit:
            stages.load_native_labels(self.manifest, **self.kwargs)
            stages.load_native_labels(self.manifest, **self.kwargs)
            self.assertEqual(audit.call_count, 2)
        self.assertFalse(self.receipt.exists())

    def test_repeated_provider_construction_uses_shared_receipt_without_model_loading(self):
        fixture = provider_fixture.ProviderFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cell_path = self.f.f.cell_manifest
        cell = json.loads(cell_path.read_text())
        cell.update(target_sha256=fixture.target_hash, draft_sha256=fixture.candidate_hash)
        cell_path.write_text(json.dumps(cell))
        self.build()
        ready = {
            "teacher_capture_manifest_sha256": [self.manifest_sha],
            "native_binary_sha256": "c" * 64,
            "native_runtime": {"CPU_fixture": True},
        }
        ready_path = fixture.root / "readiness.json"
        ready_path.write_text(json.dumps(ready))
        sources = {
            "target_gguf": fixture.target,
            "candidate_d_gguf": fixture.candidate,
            "base_draft_gguf": fixture.base,
            "absolute_d2t": self.f.f.native.absolute_map,
            "model_snapshot_manifest": fixture.model_manifest,
        }
        spec_path = fixture.root / "provider.json"
        with (
            patch.object(stages, "audit_native_labels", wraps=stages.audit_native_labels) as audit,
            patch.object(provider, "TARGET_GGUF_SHA256", fixture.target_hash),
            patch.object(provider, "CANDIDATE_D_SHA256", fixture.candidate_hash),
            patch.object(stages, "validate_readiness", return_value=ready),
            patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")),
        ):
            self.audit()
            stages.provider_manifest(
                sources,
                self.manifest,
                ready_path,
                spec_path,
                native_label_audit_receipt=stages.file_record(self.receipt),
            )
            first = provider.NativeCaptureProvider(JointQATConfig(W1AxContract(8)), spec_path)
            second = provider.NativeCaptureProvider(JointQATConfig(W1AxContract(8)), spec_path)
            self.assertEqual(audit.call_count, 1)
        self.assertEqual(first.capture.report, second.capture.report)
        self.assertEqual(first.capture_round_count, second.capture_round_count)
        self.assertIsNone(first.candidate_d)
        self.assertFalse(json.loads(self.manifest.read_text())["training_eligible"])

    def test_stage_resume_carries_explicit_receipts_into_provider_manifests(self):
        """Exercise the CPU wiring with precompleted shards and mocked native gates."""
        run = self.f.f.root / "run"
        captures = []
        for ordinal, (split, count) in enumerate((("train", 1), ("development", 24))):
            folder = run / f"stages/capture-{ordinal:05d}/labels"
            folder.mkdir(parents=True)
            prompts = folder / "prompts.jsonl"
            prompts.write_text(
                "".join(json.dumps({"id": f"{split}-{i}"}) + "\n" for i in range(count))
            )
            stages.write_json(
                folder / "manifest.json",
                {
                    "split": split,
                    "prompt_count": count,
                    "files": {"prompts": {"path": prompts.name}},
                },
            )
            captures.append(
                {
                    "prompts": str(prompts),
                    "prompts_sha256": stages.sha256(prompts),
                    "prompt_count": count,
                    "split": split,
                    "storage_forecast": {"upper_bound_bytes": 0},
                }
            )
        report = self.f.f.root / "gate.json"
        report.write_text("{}")
        config = {
            "schema": stages.STAGES_SCHEMA,
            "captures": captures,
            "sources": {
                "q4_0_draft": str(report),
                "native_runtime": {"CPU_fixture": True},
                "sha256": {
                    "target_gguf": provider.TARGET_GGUF_SHA256,
                    "candidate_d_gguf": provider.CANDIDATE_D_SHA256,
                    **{
                        name: "0" * 64
                        for name in (
                            "base_draft_gguf",
                            "absolute_d2t",
                            "model_snapshot_manifest",
                            "binary",
                        )
                    },
                },
            },
            "development_prompts": captures[1]["prompts"],
            "development_prompts_sha256": captures[1]["prompts_sha256"],
            "gate_prompts": captures[0]["prompts"],
            "gate_prompts_sha256": captures[0]["prompts_sha256"],
            "max_capture_storage_bytes": 1,
            "min_free_disk_bytes": 0,
            "native_label_audit_receipts_dir": str(run / "external-receipts"),
        }

        def receipt_audit(manifest, *, receipt_path, **kwargs):
            stages.write_json(receipt_path, {"manifest": str(manifest), **kwargs})
            return {}

        def make_provider(sources, manifest, readiness, output, **kwargs):
            spec = {"paths": {"prompts": str(manifest.parent / "prompts.jsonl")}, **kwargs}
            stages.write_json(output, spec)
            return spec

        with (
            patch.object(stages, "verify_sources"),
            patch.object(stages, "audit_q4_file"),
            patch.object(stages, "require_unsealed_prompts"),
            patch.object(stages, "host_admission"),
            patch.object(stages, "stage_progress"),
            patch.object(stages, "validate_readiness"),
            patch("check_continuous_w1ax_readiness.run_gate", return_value=report),
            patch.object(stages, "native_capture", side_effect=AssertionError("recapture")),
            patch.object(stages, "build_native_labels", side_effect=AssertionError("rebuild")),
            patch.object(stages, "audit_native_labels", side_effect=AssertionError("legacy audit")),
            patch.object(
                stages, "audit_native_labels_with_receipt", side_effect=receipt_audit
            ) as audits,
            patch.object(stages, "provider_manifest", side_effect=make_provider),
        ):
            result = stages._run_stages(config, run)
            self.assertTrue(result.is_file())
            self.assertEqual(audits.call_count, 2)
        for ordinal in range(2):
            spec = json.loads((run / f"stages/provider-{ordinal:05d}.json").read_text())
            binding = spec["native_label_audit_receipt"]
            self.assertEqual(
                binding,
                stages.file_record(
                    run / f"external-receipts/capture-{ordinal:05d}.json",
                ),
            )


if __name__ == "__main__":
    unittest.main()
