"""Synthetic hashed evidence validates contracts; never executes a CUDA gate."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_continuous_w1ax_readiness as checker  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.common = {
            key: "a" * 64
            for key in (
                "target_gguf",
                "candidate_d_gguf",
                "base_draft_gguf",
                "absolute_d2t",
                "model_snapshot_manifest",
            )
        }

    def evidence(self, directory, name, value):
        path = directory / name
        if isinstance(value, dict):
            path.write_text(json.dumps(value))
        else:
            path.write_bytes(value)
        return stages.file_record(path)

    def report(self, bits):
        folder = self.root / f"A{bits}"
        folder.mkdir()
        checkpoint = self.evidence(folder, "checkpoint.npz", b"synthetic CPU evidence only")
        exported = self.evidence(folder, "student.gguf", b"synthetic CPU export only")
        evidence = {"checkpoint": checkpoint, "export": exported}
        evidence["checkpoint_manifest"] = self.evidence(
            folder,
            "checkpoint.json",
            {
                "activation_bits": bits,
                "objective": "hard_ce",
                "scale_layout": "row",
                "base_gguf_sha256": self.common["base_draft_gguf"],
                "checkpoint_sha256": checkpoint["sha256"],
                "activation_rule": (
                    "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
                ),
                "weight_rule": "hard_sign_zero_positive_clipped_identity_ste",
            },
        )
        evidence["export_audit"] = self.evidence(
            folder,
            "export.json",
            {"activation_bits": bits, "serialization_audit_passed": True, "output": exported},
        )
        cell = {
            "env": {"GGML_W1AX_ACT_BITS": str(bits)},
            "target_sha256": "a" * 64,
            "draft_sha256": exported["sha256"],
            "binary_sha256": "b" * 64,
        }
        evidence["native_cell"] = self.evidence(folder, "cell.json", cell)
        evidence["native_capture_manifest"] = self.evidence(
            folder,
            "capture.json",
            {
                "activation_bits": bits,
                "target_sha256": "a" * 64,
                "draft_sha256": exported["sha256"],
            },
        )
        source_hashes = {}
        for field, name in (
            ("cache_index", "heads.draft_cache.jsonl"),
            ("cache_rows", "heads.draft_cache.f16"),
            ("cache_masks", "heads.draft_cache.mask"),
            ("graph_index", "heads.draft_graph.jsonl"),
            ("graph_values", "heads.draft_graph.f32"),
        ):
            source_hashes[field] = self.evidence(folder, name, b"synthetic cache metadata")[
                "sha256"
            ]
        evidence["native_cache_audit"] = self.evidence(
            folder,
            "cache.json",
            {
                "schema": "recurrent_stored_draft_cache_audit_v2",
                "status": "stored_f16_rows_and_exact_prefix_masks_compared",
                "execution_device": "cuda",
                "key_elements": 1,
                "key_equal_elements": 1,
                "value_elements": 1,
                "value_equal_elements": 1,
                "source_sha256": source_hashes,
            },
        )
        return {
            "schema": checker.SCHEMA,
            "activation_bits": bits,
            "execution_device": "cuda:0",
            "scale_layout": "row",
            "objective": "hard_ce",
            "optimizer_steps": 0,
            "common_source_sha256": self.common,
            "native_binary_sha256": "b" * 64,
            "checks": {k: True for k in checker.CHECKS},
            "limits": {"relative_rms": 0.10, "decision_margin": 0.02},
            "domains": ["code", "prose", "reasoning"],
            "evidence": evidence,
            "roots": [
                {
                    "domain": d,
                    "state_relative_rms": 0.01,
                    "logit_relative_rms": 0.01,
                    "decision_margin": 0.2,
                    "top_choice_matches": True,
                    "loss": 1.0,
                    "finite_gradient_tensors": 18,
                }
                for d in ("code", "prose", "reasoning")
                for _ in range(2)
            ],
        }

    def test_independent_hashed_a8_a1_readiness_cpu_only(self):
        a8, a1 = self.report(8), self.report(1)
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")):
            checker.validate_gate_report(a8, 8, self.common)
            checker.validate_gate_report(a1, 1, self.common)
            paths = {
                "8": self.evidence(self.root, "a8.json", a8),
                "1": self.evidence(self.root, "a1.json", a1),
            }
            ready = self.evidence(
                self.root,
                "ready.json",
                {
                    "schema": stages.READINESS_SCHEMA,
                    "training_eligible": True,
                    "objective": "hard_ce",
                    "scale_layout": "row",
                    "unresolved_gates": [],
                    "common_source_sha256": self.common,
                    "precisions": paths,
                },
            )
            result = stages.validate_readiness(ready, activation_bits=8, common_hashes=self.common)
            self.assertTrue(result["training_eligible"])

    def test_a16_report_cannot_qualify_a8(self):
        report = self.report(8)
        report["activation_bits"] = 16
        with self.assertRaises(ValueError):
            checker.validate_gate_report(report, 8, self.common)

    def test_nan_and_string_pass_flags_fail(self):
        report = self.report(8)
        nan = copy.deepcopy(report)
        nan["roots"][0]["state_relative_rms"] = float("nan")
        with self.assertRaises(ValueError):
            checker.validate_gate_report(nan, 8, self.common)
        report["checks"][checker.CHECKS[0]] = "pass"
        with self.assertRaises(ValueError):
            checker.validate_gate_report(report, 8, self.common)

    def test_missing_domain_and_high_margin_decision_change_fail(self):
        report = self.report(8)
        missing = copy.deepcopy(report)
        missing["roots"][0]["domain"] = "prose"
        with self.assertRaises(ValueError):
            checker.validate_gate_report(missing, 8, self.common)
        report["roots"][0]["top_choice_matches"] = False
        with self.assertRaises(ValueError):
            checker.validate_gate_report(report, 8, self.common)

    def test_cached_evidence_mutation_is_detected(self):
        report = self.report(8)
        checker.validate_gate_report(report, 8, self.common)
        path = Path(report["evidence"]["export"]["path"])
        path.write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            checker.validate_gate_report(report, 8, self.common)

    def test_native_evidence_cannot_silently_use_a16(self):
        report = self.report(8)
        evidence = report["evidence"]["native_cell"]
        data = json.loads(Path(evidence["path"]).read_text())
        data["env"]["GGML_W1AX_ACT_BITS"] = "16"
        report["evidence"]["native_cell"] = self.evidence(self.root / "A8", "cell.json", data)
        with self.assertRaisesRegex(ValueError, "contracts"):
            checker.validate_gate_report(report, 8, self.common)


if __name__ == "__main__":
    unittest.main()
