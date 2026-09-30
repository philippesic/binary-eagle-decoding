"""Current-student capture ancestry with mocked eligibility; CPU only."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_w1ax_capture_provider as fixture_module  # noqa: E402
import w1ax_capture_provider as provider  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402

from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH  # noqa: E402
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract  # noqa: E402


class RefreshedProviderTests(unittest.TestCase):
    def setUp(self):
        f = fixture_module.ProviderFixture()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.root = f.root
        self.spec_path = self.root / "refreshed-provider.json"
        with (
            patch.object(fixture_module.prep, "TARGET_GGUF_SHA256", f.target_hash),
            patch.object(fixture_module.prep, "CANDIDATE_D_SHA256", f.candidate_hash),
        ):
            self.spec = f._prepare(self.spec_path)
        self.spec.update(schema=provider.V2_SCHEMA, training_eligible=True)
        binary = self.root / "native-binary"
        binary.write_bytes(b"CPU binary provenance fixture")
        exported = self.root / "student.gguf"
        exported.write_bytes(b"CPU current-student export fixture")
        checkpoint = self.root / "joint.npz"
        checkpoint.write_bytes(b"CPU checkpoint provenance fixture")
        self.common = {k: self.spec["sha256"][k] for k in stages_common()}
        cell = json.loads(f.cell.read_text())
        cell["draft_sha256"] = stages.sha256(exported)
        f.cell.write_text(json.dumps(cell))
        capture = json.loads(f.capture_manifest.read_text())
        capture.update(
            schema=stages.LABEL_SCHEMA,
            activation_bits=8,
            binary_sha256=stages.sha256(binary),
            files={"source_cell": {"path": f.cell.name, "sha256": stages.sha256(f.cell)}},
        )
        f.capture_manifest.write_text(json.dumps(capture))
        self.spec["sha256"]["capture_manifest"] = stages.sha256(f.capture_manifest)
        manifest_path = self.root / "joint.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "scale_layout": "row",
                    "objective": "hard_ce",
                    "activation_bits": 8,
                    "activation_rule": (
                        "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
                    ),
                    "weight_rule": "hard_sign_zero_positive_clipped_identity_ste",
                    "qk_row_order": "original_checkpoint",
                    "base_gguf_sha256": self.common["base_draft_gguf"],
                    "checkpoint_sha256": stages.sha256(checkpoint),
                    "projections": {name: {} for name in CANDIDATE_D_BASE_TO_PATH},
                }
            )
        )
        audit_path = self.root / "export-audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "serialization_audit_passed": True,
                    "scale_layout": "row",
                    "activation_bits": 8,
                    "base_gguf": {"sha256": self.common["base_draft_gguf"]},
                    "checkpoint": stages.file_record(checkpoint),
                    "checkpoint_manifest": stages.file_record(manifest_path),
                    "output": stages.file_record(exported),
                    "projections": {name: {} for name in CANDIDATE_D_BASE_TO_PATH},
                }
            )
        )
        receipt_path = self.root / "refresh.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": "w1ax_exact_prefix_refresh_v2",
                    "training_eligible": False,
                    "changed_prefix_labels_reused": False,
                    "activation_bits": 8,
                    "export": stages.file_record(exported),
                    "checkpoint": stages.file_record(checkpoint),
                    "checkpoint_manifest": stages.file_record(manifest_path),
                    "capture_manifest": stages.file_record(f.capture_manifest),
                    "prompts": stages.file_record(f.prompts),
                    "native_binary": stages.file_record(binary),
                    "common_source_sha256": self.common,
                }
            )
        )
        self.binding = {
            "export": stages.file_record(exported),
            "checkpoint": stages.file_record(checkpoint),
            "checkpoint_manifest": stages.file_record(manifest_path),
            "export_audit": stages.file_record(audit_path),
            "refresh_receipt": stages.file_record(receipt_path),
        }
        self.ready = {
            "teacher_capture_manifest_sha256": [stages.sha256(f.capture_manifest)],
            "native_binary_sha256": stages.sha256(binary),
            "native_runtime": {"CPU_fixture": True},
        }

    def validate(self, binding):
        return provider.validate_captured_drafter(
            binding,
            capture_manifest_sha256=self.spec["sha256"]["capture_manifest"],
            captured_draft_sha256=self.binding["export"]["sha256"],
            prompts_sha256=self.spec["sha256"]["prompts"],
            common_hashes=self.common,
            activation_bits=8,
            native_binary_sha256=self.ready["native_binary_sha256"],
        )

    def test_matching_receipt_validates_and_keeps_ineligible_capture_unchanged(self):
        original = self.f.capture_manifest.read_bytes()
        self.assertFalse(self.validate(self.binding)["training_eligible"])
        self.assertEqual(original, self.f.capture_manifest.read_bytes())

    def test_missing_or_wrong_actor_binding_fails(self):
        with self.assertRaisesRegex(ValueError, "requires exact"):
            self.validate(None)
        changed = copy.deepcopy(self.binding)
        changed["export"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate(changed)

    def test_refreshed_capture_loads_with_new_mocked_eligibility_for_both_precisions(self):
        self.spec["captured_drafter"] = self.binding
        self.spec_path.write_text(json.dumps(self.spec))
        with (
            patch.object(provider, "TARGET_GGUF_SHA256", self.f.target_hash),
            patch.object(provider, "CANDIDATE_D_SHA256", self.f.candidate_hash),
            patch.object(stages, "validate_readiness", return_value=self.ready),
            patch.object(stages, "load_native_labels", return_value=fixture_module.FakeCapture()),
            patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")),
        ):
            for bits in (8, 1):
                result = provider.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(bits)), self.spec_path
                )
                self.assertTrue(result.full_body_qat_eligible)
                self.assertEqual(
                    result.source_metadata["captured_draft_sha256"],
                    self.binding["export"]["sha256"],
                )
                self.assertEqual(result.hashes["candidate_d_gguf"], self.f.candidate_hash)
        self.assertFalse(json.loads(self.f.capture_manifest.read_text())["training_eligible"])

    def test_refreshed_capture_missing_binding_rejected_before_loader(self):
        self.spec_path.write_text(json.dumps(self.spec))
        with (
            patch.object(provider, "TARGET_GGUF_SHA256", self.f.target_hash),
            patch.object(provider, "CANDIDATE_D_SHA256", self.f.candidate_hash),
            patch.object(stages, "validate_readiness", return_value=self.ready),
            patch.object(stages, "load_native_labels", side_effect=AssertionError("loader")),
        ):
            with self.assertRaisesRegex(ValueError, "requires exact"):
                provider.NativeCaptureProvider(JointQATConfig(W1AxContract(8)), self.spec_path)


def stages_common():
    return (
        "target_gguf",
        "candidate_d_gguf",
        "base_draft_gguf",
        "absolute_d2t",
        "model_snapshot_manifest",
    )


if __name__ == "__main__":
    unittest.main()
