"""Frozen CPU sample checks for refresh ancestry, caps and curve decisions."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from w1a1_eagle.trajectory_refresh import audit_plan, build_plan, digest, file_record

ROOT = Path(__file__).resolve().parents[1]


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.policy = {
            "schema": "w1ax_refresh_policy_v1",
            "split": "train",
            "train_prompts_sha256": "a" * 64,
            "development_sample_sha256": "b" * 64,
            "train_sample": [
                {"prompt_id": domain, "domain": domain, "split": "train"}
                for domain in ("prose", "code", "reasoning")
            ],
            "caps": {
                "max_rounds": 2,
                "max_student_rows": 10,
                "max_new_label_rows": 3,
                "max_new_feature_rows": 3,
                "max_prefix_tokens": 8,
            },
            "learning_curve": {
                "min_completed_steps": 200,
                "min_relative_ce_improvement": 0.05,
                "max_acceptance_regression": 0.02,
                "min_changed_prefix_fraction": 0.2,
            },
        }
        checkpoint = self.write("checkpoint.bin", b"student-checkpoint")
        exported = self.write("student.gguf", b"student-export")
        self.identity = {
            "checkpoint_sha256": checkpoint["sha256"],
            "export_sha256": exported["sha256"],
        }
        self.contract = {
            "target_gguf_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "absolute_d2t_sha256": "e" * 64,
            "native_revision": "0" * 40,
            "execution_policy_sha256": "f" * 64,
            "target_vocab_size": 100,
        }
        self.student_rows, self.teacher_rows = [], []
        for domain in ("prose", "code", "reasoning"):
            self.student_rows.append(
                {
                    "id": domain + "-row",
                    "prompt_id": domain,
                    "split": "train",
                    "prefix_token_ids": [1, 2, 4],
                    "feature_prefix_token_ids": [1, 2],
                    **self.identity,
                }
            )
            for tokens in ([1], [1, 2]):
                self.teacher_rows.append(
                    {
                        "kind": "feature",
                        "prompt_id": domain,
                        "split": "train",
                        "prefix_token_ids": tokens,
                        "artifact": "native",
                        "row": len(tokens) - 1,
                        "capture_id": "fixture",
                    }
                )
            self.teacher_rows.append(
                {
                    "kind": "label",
                    "prompt_id": domain,
                    "split": "train",
                    "prefix_token_ids": [1, 2, 3],
                    "artifact": "native",
                    "row": 2,
                    "capture_id": "fixture",
                    "next_target_id": 7,
                }
            )
        self.teacher = {
            "schema": "w1ax_refresh_teacher_index_v1",
            "split": "train",
            "native_teacher_contract": self.contract,
            "train_prompts_sha256": "a" * 64,
            "artifacts": {"native": self.write("native.bin", b"native-capture")},
        }
        self.curve = {
            "schema": "w1ax_refresh_learning_curve_v1",
            "split": "development",
            "development_sample_sha256": "b" * 64,
            "checkpoint_sha256": checkpoint["sha256"],
            "previous": {
                "checkpoint_sha256": "0" * 64,
                "completed_steps": 100,
                "hard_ce": 2.0,
                "native_acceptance": 0.3,
            },
            "current": {
                "checkpoint_sha256": checkpoint["sha256"],
                "completed_steps": 200,
                "hard_ce": 1.8,
                "native_acceptance": 0.31,
            },
        }
        self.spec = {
            "schema": "w1ax_refresh_request_v1",
            "refresh_round": 0,
            "student": self.identity,
            "native_teacher_contract": self.contract,
            "inputs": {"checkpoint": checkpoint, "student_export": exported},
        }
        prompt_path = self.directory / "train.jsonl"
        prompt_path.write_text(
            "".join(json.dumps({"id": domain}) + "\n" for domain in ("prose", "code", "reasoning"))
        )
        self.spec["inputs"]["train_prompts"] = file_record(prompt_path)
        self.policy["train_prompts_sha256"] = file_record(prompt_path)["sha256"]
        self.teacher["train_prompts_sha256"] = self.policy["train_prompts_sha256"]
        self.sync()

    def write(self, name, value):
        path = self.directory / name
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return file_record(path)

    def sync(self):
        index_path = self.directory / "teacher-rows.jsonl"
        index_path.write_text("".join(json.dumps(row) + "\n" for row in self.teacher_rows))
        self.teacher["index"] = file_record(index_path)
        student_path = self.directory / "student-rows.jsonl"
        student_path.write_text("".join(json.dumps(row) + "\n" for row in self.student_rows))
        self.spec["inputs"].update(
            {
                "policy": self.write("policy.json", self.policy),
                "teacher_manifest": self.write("teacher.json", self.teacher),
                "student_rows": file_record(student_path),
                "learning_curve": self.write("curve.json", self.curve),
            }
        )

    def test_changed_labels_need_capture_features_reuse_exactly(self):
        plan = build_plan(self.spec)
        self.assertEqual(plan["counts"]["new_label_rows"], 3)
        self.assertEqual(plan["counts"]["new_feature_rows"], 0)
        self.assertEqual(len(plan["reused"]), 6)
        self.assertTrue(plan["capture_queue_ready"])
        self.assertFalse(plan["training_eligible"])
        self.assertEqual(plan["frozen_sample_sha256"], digest(self.policy["train_sample"]))

    def test_changed_root_requires_full_feature_chain(self):
        for row in self.student_rows:
            row["feature_prefix_token_ids"] = [1, 2, 4]
        self.sync()
        plan = build_plan(self.spec)
        self.assertEqual(plan["counts"]["new_feature_rows"], 3)
        self.assertTrue(plan["capture_queue_ready"])

    def test_cap_overflow_never_silently_truncates(self):
        self.policy["caps"]["max_new_label_rows"] = 2
        self.sync()
        plan = build_plan(self.spec)
        self.assertEqual(len(plan["capture_requests"]), 3)
        self.assertFalse(plan["within_refresh_cap"])
        self.assertFalse(plan["capture_queue_ready"])

    def test_exact_teacher_labels_complete_ancestry_but_do_not_approve_training(self):
        for row in self.teacher_rows:
            if row["kind"] == "label":
                row["prefix_token_ids"] = [1, 2, 4]
        self.sync()
        plan = build_plan(self.spec)
        self.assertEqual(plan["capture_requests"], [])
        self.assertFalse(plan["capture_queue_ready"])
        self.assertFalse(plan["training_eligible"])

    def test_execution_policy_mismatch_rejected(self):
        self.teacher["native_teacher_contract"] = copy.deepcopy(self.contract)
        self.teacher["native_teacher_contract"]["execution_policy_sha256"] = "1" * 64
        self.sync()
        with self.assertRaisesRegex(ValueError, "execution contract"):
            build_plan(self.spec)

    def test_final_row_or_policy_rejected(self):
        self.student_rows[0]["split"] = "final"
        self.sync()
        with self.assertRaisesRegex(ValueError, "frozen train"):
            build_plan(self.spec)
        self.policy["split"] = "final"
        self.sync()
        with self.assertRaisesRegex(ValueError, "sealed"):
            build_plan(self.spec)

    def test_sample_must_belong_to_actual_train_freeze(self):
        self.policy["train_sample"][0]["prompt_id"] = "sealed-other"
        self.sync()
        with self.assertRaisesRegex(ValueError, "actual training freeze"):
            build_plan(self.spec)

    def test_development_identity_nonfinite_and_step_gate(self):
        self.curve["development_sample_sha256"] = "0" * 64
        self.sync()
        with self.assertRaisesRegex(ValueError, "frozen development"):
            build_plan(self.spec)
        self.curve["development_sample_sha256"] = "b" * 64
        self.curve["current"]["hard_ce"] = float("nan")
        self.sync()
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "stop")
        self.curve["current"]["hard_ce"] = 1.8
        self.curve["current"]["completed_steps"] = 150
        self.sync()
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "hold")

    def test_checkpoint_and_feature_root_mismatch_rejected(self):
        self.student_rows[0]["checkpoint_sha256"] = "0" * 64
        self.sync()
        with self.assertRaisesRegex(ValueError, "checkpoint ancestry"):
            build_plan(self.spec)
        self.student_rows[0].update(self.identity)
        self.student_rows[0]["feature_prefix_token_ids"] = [1, 3]
        self.sync()
        with self.assertRaisesRegex(ValueError, "ancestor"):
            build_plan(self.spec)

    def test_ambiguous_teacher_rejected(self):
        self.teacher_rows.append(copy.deepcopy(self.teacher_rows[0]))
        self.sync()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            build_plan(self.spec)

    def test_gate_stop_hold_and_pending(self):
        self.curve["current"]["native_acceptance"] = 0.1
        self.sync()
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "stop")
        self.curve["current"]["native_acceptance"] = 0.3
        self.curve["current"]["hard_ce"] = 1.99
        self.sync()
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "hold")
        del self.spec["inputs"]["learning_curve"]
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "pending")
        self.spec["refresh_round"] = 2
        self.assertEqual(build_plan(self.spec)["learning_gate"]["status"], "stop")

    def test_reaudits_every_artifact_and_plan(self):
        plan = build_plan(self.spec)
        path = self.directory / "plan.json"
        path.write_text(json.dumps(plan))
        self.assertEqual(audit_plan(path)["status"], "pass")
        plan["capture_requests"][0]["prefix_token_ids"] = [1, 2, 3]
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "recomputed"):
            audit_plan(path)
        (self.directory / "native.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_plan(self.spec)

    def test_v1_index_preserves_audit_and_coalesces_only_identical_labels(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from plan_w1ax_trajectory_refresh import index_v1

        cell = self.write("source_cell_manifest.json", {"target_sha256": "c" * 64})
        native_rows = self.directory / "v1-rows.jsonl"
        native_rows.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in [
                    {"valid": False},
                    {
                        "valid": True,
                        "prompt_id": "code",
                        "prefix_token_ids": [1, 2],
                        "verifier_token_id": 7,
                    },
                    {
                        "valid": True,
                        "prompt_id": "code",
                        "prefix_token_ids": [1, 2],
                        "verifier_token_id": 7,
                    },
                ]
            )
        )
        capture = self.write(
            "v1-manifest.json",
            {
                "target_vocab_size": 100,
                "source_report_sha256": {"cell_manifest": cell["sha256"]},
                "rows": {"path": native_rows.name},
            },
        )
        args = SimpleNamespace(
            output=self.directory / "normalized",
            capture_manifest=Path(capture["path"]),
            prompts=Path(self.spec["inputs"]["train_prompts"]["path"]),
            prompts_sha256=self.policy["train_prompts_sha256"],
            prompt_count=3,
            native_contract=Path(self.write("native-contract.json", self.contract)["path"]),
        )
        auditor = SimpleNamespace(
            load_audited_capture=lambda *a, **k: SimpleNamespace(
                feature_lookup={("code", (1,)): 0},
                report={"status": "fixture-audited"},
            )
        )
        with patch.dict(sys.modules, {"audit_recurrent_binary_capture": auditor}):
            result = index_v1(args)
        indexed = [
            json.loads(line) for line in Path(result["index"]["path"]).read_text().splitlines()
        ]
        self.assertEqual(len(indexed), 2)
        self.assertEqual(indexed[1]["row"], 1)
        self.assertEqual(indexed[1]["equivalent_source_rows"], [1, 2])
        self.assertIn("audit", result["artifacts"])
        self.assertIn("source_cell", result["artifacts"])
        self.assertFalse(result["training_eligible"])

    def test_cli_prepares_and_audits_immutable_artifacts(self):
        request = self.directory / "request.json"
        request.write_text(json.dumps(self.spec))
        plan = self.directory / "plan.json"
        command = [sys.executable, str(ROOT / "scripts/plan_w1ax_trajectory_refresh.py")]
        subprocess.run(
            command + ["plan", "--request", str(request), "--output", str(plan)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            command
            + ["audit", "--plan", str(plan), "--output", str(self.directory / "audit.json")],
            check=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "pass")
        duplicate = subprocess.run(
            command + ["plan", "--request", str(request), "--output", str(plan)],
            capture_output=True,
        )
        self.assertNotEqual(duplicate.returncode, 0)


if __name__ == "__main__":
    unittest.main()
