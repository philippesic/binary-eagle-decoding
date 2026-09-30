"""CPU-only publication preserves failed gate metrics without admitting them."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_continuous_w1ax_readiness as checker


class GateFailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.report_path = self.root / "gate.json"
        self.common = {"base_draft_gguf": "a" * 64}
        self.report = {
            "schema": checker.SCHEMA,
            "activation_bits": 1,
            "execution_device": "cuda:0",
            "scale_layout": "row",
            "objective": "hard_ce",
            "optimizer_steps": 0,
            "common_source_sha256": self.common,
            "limits": {"relative_rms": 0.10, "decision_margin": 0.02},
            "checks": {name: True for name in checker.CHECKS},
            "roots": [{"domain": "prose", "logit_relative_rms": 0.25} for _ in range(6)],
        }

    def publish_failure(self):
        with self.assertRaises(ValueError):
            checker._publish_gate_report(self.report, 1, self.common, self.report_path)
        self.assertFalse(self.report_path.exists())
        return sorted(self.root.glob("gate-candidate-*.json")), sorted(
            self.root.glob("gate-failure-*.json")
        )

    def test_failed_candidate_metrics_and_hash_bound_receipt_are_retained(self):
        name = "exact_prefix_states_and_logits"
        self.report["checks"][name] = False
        candidates, receipts = self.publish_failure()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(receipts), 1)
        candidate = json.loads(candidates[0].read_text())
        receipt = json.loads(receipts[0].read_text())
        for field in ("roots", "checks", "optimizer_steps"):
            self.assertEqual(candidate[field], self.report[field])
        for diagnostic in (candidate, receipt):
            self.assertIs(diagnostic["training_eligible"], False)
            self.assertIs(diagnostic["readiness_evidence"], False)
            self.assertEqual(diagnostic["optimizer_steps"], 0)
        self.assertEqual(receipt["candidate"], checker.file_record(candidates[0]))
        self.assertEqual(receipt["failed_checks"], [name])
        self.assertIn(name, receipt["validation_error"])
        self.assertEqual(receipt["validation_error_type"], "ValueError")

    def test_aggregate_error_identifies_metadata_and_check_inventory(self):
        self.report["activation_bits"] = 8
        self.report["execution_device"] = "cpu"
        self.report["optimizer_steps"] = 1
        self.report["roots"] = []
        self.report["checks"].pop("all_nine_finite_gradients")
        self.report["checks"]["unexpected"] = 1
        with self.assertRaises(ValueError) as raised:
            checker.validate_gate_report(self.report, 1, self.common)
        for detail in (
            "activation_bits", "execution_device", "optimizer_steps", "six entries",
            "all_nine_finite_gradients", "unexpected",
        ):
            self.assertIn(detail, str(raised.exception))

    def test_non_boolean_pass_flag_still_fails(self):
        self.report["checks"]["all_nine_finite_gradients"] = "true"
        self.publish_failure()

    def test_later_validation_failure_also_keeps_candidate(self):
        with patch.object(checker, "validate_gate_report", side_effect=ValueError("root limit")):
            candidates, receipts = self.publish_failure()
        self.assertEqual(json.loads(candidates[0].read_text())["roots"], self.report["roots"])
        self.assertEqual(json.loads(receipts[0].read_text())["validation_error"], "root limit")

    def test_success_validates_after_candidate_then_publishes_exact_report(self):
        original = copy.deepcopy(self.report)
        def validate(report, bits, common):
            self.assertFalse(self.report_path.exists())
            candidate = json.loads(next(self.root.glob("gate-candidate-*.json")).read_text())
            self.assertEqual(candidate["roots"], report["roots"])
            self.assertEqual((bits, common), (1, self.common))
        with patch.object(checker, "validate_gate_report", side_effect=validate) as validator:
            checker._publish_gate_report(self.report, 1, self.common, self.report_path)
        validator.assert_called_once_with(self.report, 1, self.common)
        self.assertEqual(json.loads(self.report_path.read_text()), original)
        self.assertEqual(self.report, original)
        self.assertEqual(list(self.root.glob("gate-failure-*.json")), [])

    def test_rerun_preserves_previous_failure_and_candidate(self):
        self.report["checks"]["exact_prefix_states_and_logits"] = False
        with patch.object(checker.time, "time_ns", side_effect=[100, 101]):
            candidates, receipts = self.publish_failure()
            saved = {path.name: path.read_bytes() for path in candidates + receipts}
            self.report["checks"]["exact_prefix_states_and_logits"] = True
            with patch.object(checker, "validate_gate_report"):
                checker._publish_gate_report(self.report, 1, self.common, self.report_path)
        self.assertEqual(len(list(self.root.glob("gate-candidate-*.json"))), 2)
        self.assertEqual(len(list(self.root.glob("gate-failure-*.json"))), 1)
        for name, content in saved.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertTrue(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()
